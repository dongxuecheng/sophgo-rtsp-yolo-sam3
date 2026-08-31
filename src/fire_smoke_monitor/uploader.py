from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
import logging
import threading
import time
from typing import Callable, Optional, Sequence, Set

import httpx

from .config import PipelineConfig
from .domain import UploadFrame
from .health import HealthRegistry


logger = logging.getLogger(__name__)


class UploadError(RuntimeError):
    """A safe error message without URLs, credentials or remote response text."""


class PipelineClient:
    def __init__(self, config: PipelineConfig, http: httpx.AsyncClient, api_key: str):
        self._config = config
        self._http = http
        self._headers = {"Accept-Encoding": "identity"}
        if api_key:
            self._headers["X-API-Key"] = api_key

    async def upload(self, frame: UploadFrame) -> None:
        fields = {
            "machine_id": self._config.machine_id,
            "stream_id": frame.camera_id,
            "captured_at": frame.captured_at.isoformat(),
        }
        if frame.stream_name:
            fields["stream_name"] = frame.stream_name
        async with self._http.stream(
            "POST", self._config.endpoint,
            data=fields,
            files={"image": ("frame.jpg", frame.jpeg, "image/jpeg")},
            headers=self._headers,
        ) as response:
            if response.status_code != 202:
                raise UploadError("HTTP {}".format(response.status_code))
            # The API response is tiny. Never retain an unbounded response body.
            body = bytearray()
            async for chunk in response.aiter_raw():
                if len(body) + len(chunk) > 4096:
                    raise UploadError("202 响应过大")
                body.extend(chunk)
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeError):
                raise UploadError("202 响应不是有效 JSON") from None
            if not isinstance(payload, dict) or payload.get("accepted") is not True:
                raise UploadError("202 响应缺少 accepted=true")


class FrameUploader:
    """Bounded, latest-per-camera queue on an independent HTTP event loop.

    At most one request per camera is in flight. Replacing a queued frame keeps
    its FIFO position so a busy camera cannot starve another camera. Failed or
    timed-out requests are never retried: the server has no idempotency key.
    """

    def __init__(
        self, config: PipelineConfig, camera_ids: Sequence[str],
        clock: Callable[[], float] = time.monotonic,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        health: Optional[HealthRegistry] = None,
    ):
        self._config = config
        self._camera_ids = frozenset(camera_ids)
        self._clock = clock
        self._transport = transport
        self._health = health
        self._api_key = config.api_key()
        self._lock = threading.Lock()
        self._pending: OrderedDict[str, UploadFrame] = OrderedDict()
        self._active: Set[str] = set()
        self._closed = False
        self._ready = threading.Event()
        self._loop = None
        self._wake = None
        self._stop = None
        self._startup_error = None
        self._thread = threading.Thread(target=self._run, name="pipeline-upload", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            self.close()
            raise RuntimeError("上传线程启动超时")
        if self._startup_error is not None:
            self.close()
            raise RuntimeError("上传线程启动失败: {}".format(self._startup_error))

    def submit(self, frame: UploadFrame) -> str:
        if frame.camera_id not in self._camera_ids:
            raise ValueError("未配置的摄像头")
        if not frame.jpeg or len(frame.jpeg) > self._config.max_image_bytes:
            raise ValueError("JPEG 为空或超过上传限制")
        with self._lock:
            if self._closed:
                return "stopped"
            replaced = frame.camera_id in self._pending
            self._pending[frame.camera_id] = frame
        self._notify(self._wake)
        return "replaced" if replaced else "queued"

    def _notify(self, event) -> None:
        loop = self._loop
        if loop is not None and event is not None:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # close() may race with a final enqueue during shutdown.
                pass

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as exc:
            self._startup_error = type(exc).__name__
            logger.error("上传线程异常退出 error=%s", self._startup_error)
            with self._lock:
                expected_stop = self._closed
            if self._health is not None and not expected_stop:
                self._health.fatal("uploader_thread_{}".format(type(exc).__name__))
        finally:
            with self._lock:
                self._closed = True
                self._pending.clear()
            self._ready.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        async with httpx.AsyncClient(
            timeout=self._config.timeout_sec, trust_env=False, follow_redirects=False,
            limits=httpx.Limits(max_connections=self._config.workers,
                               max_keepalive_connections=self._config.workers),
            transport=self._transport,
        ) as http:
            client = PipelineClient(self._config, http, self._api_key)
            workers = [asyncio.create_task(self._worker(client))
                       for _ in range(self._config.workers)]
            heartbeat = asyncio.create_task(self._heartbeat())
            for task in workers + [heartbeat]:
                task.add_done_callback(self._task_done)
            self._ready.set()
            with self._lock:
                if self._closed:
                    self._stop.set()
            try:
                await self._stop.wait()
            finally:
                for worker in workers:
                    worker.cancel()
                heartbeat.cancel()
                await asyncio.gather(*workers, heartbeat, return_exceptions=True)

    def _task_done(self, task) -> None:
        if task.cancelled():
            return
        try:
            error = task.exception()
        except (asyncio.CancelledError, RuntimeError):
            return
        with self._lock:
            expected_stop = self._closed
        if expected_stop:
            return
        error_type = type(error).__name__ if error is not None else "UnexpectedStop"
        self._startup_error = error_type
        logger.error("上传内部任务异常退出 error=%s", error_type)
        if self._health is not None:
            self._health.fatal("uploader_task_{}".format(error_type))
        if self._stop is not None:
            self._stop.set()

    async def _heartbeat(self) -> None:
        while True:
            if self._health is not None:
                self._health.beat("uploader")
            await asyncio.sleep(2)

    async def _take(self) -> Optional[UploadFrame]:
        while True:
            with self._lock:
                if self._closed:
                    return None
                camera = next((key for key in self._pending if key not in self._active), None)
                if camera is not None:
                    self._active.add(camera)
                    return self._pending.pop(camera)
                self._wake.clear()
            await self._wake.wait()

    async def _worker(self, client: PipelineClient) -> None:
        while True:
            frame = await self._take()
            if frame is None:
                return
            started = self._clock()
            try:
                age = started - frame.created_monotonic
                if age > self._config.max_pending_age_sec:
                    logger.warning("丢弃过期待传帧 camera=%s age=%.1fs", frame.camera_id, age)
                    continue
                # Total deadline includes connection, upload AND response reading.
                await asyncio.wait_for(client.upload(frame), timeout=self._config.timeout_sec)
                if self._health is not None:
                    self._health.recover("remote")
                logger.info("远端已接收 camera=%s status=202 bytes=%d queue_wait=%.3fs elapsed=%.3fs",
                            frame.camera_id, len(frame.jpeg), age, self._clock() - started)
            except asyncio.TimeoutError:
                if self._health is not None:
                    self._health.degrade("remote", "timeout")
                logger.warning("上传超时，不补发 camera=%s timeout=%.1fs",
                               frame.camera_id, self._config.timeout_sec)
            except UploadError as exc:
                if self._health is not None:
                    self._health.degrade("remote", "rejected")
                logger.warning("上传未获确认，不补发 camera=%s reason=%s", frame.camera_id, exc)
            except Exception as exc:
                if self._health is not None:
                    self._health.degrade("remote", type(exc).__name__)
                # httpx exception strings can contain endpoints; only log the type.
                logger.warning("上传失败，不补发 camera=%s error=%s", frame.camera_id, type(exc).__name__)
            finally:
                with self._lock:
                    self._active.discard(frame.camera_id)
                self._wake.set()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            dropped = len(self._pending)
            self._pending.clear()
        self._notify(self._stop)
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            logger.warning("上传线程未在关闭期限内退出")
        if dropped:
            logger.info("退出时丢弃待传帧 count=%d", dropped)

    def is_healthy(self) -> bool:
        with self._lock:
            closed = self._closed
        return not closed and self._thread.is_alive() and self._startup_error is None
