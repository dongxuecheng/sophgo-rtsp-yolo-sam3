from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Sequence

from .config import CameraConfig


logger = logging.getLogger(__name__)


class FrameUnavailableError(RuntimeError):
    """No decoded frame is currently available for a camera."""


class _CameraStream:
    """Own one decoder lifecycle so a broken RTSP URL is fully isolated."""

    def __init__(
        self,
        sail_module: Any,
        camera: CameraConfig,
        dev_id: int,
        queue_size: int,
        unavailable_timeout_sec: float,
        reconnect_cooldown_sec: float,
    ) -> None:
        self._sail = sail_module
        self._camera = camera
        self._dev_id = dev_id
        self._queue_size = queue_size
        self._unavailable_timeout_sec = unavailable_timeout_sec
        self._reconnect_cooldown_sec = reconnect_cooldown_sec

        self._state_lock = threading.Lock()
        self._decoder: Optional[Any] = None
        self._channel: Optional[int] = None
        self._unavailable_since: Optional[float] = time.monotonic()
        self._last_warning_at = float("-inf")
        self._ever_received_frame = False

        self._stop_event = threading.Event()
        self._restart_event = threading.Event()
        self._thread = threading.Thread(
            target=self._supervise,
            name="rtsp-supervisor-{}".format(camera.camera_id),
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def read(self) -> Any:
        now = time.monotonic()
        warning_detail: Optional[str] = None
        recovered = False
        first_frame = False
        valid = False
        failure_cause: Optional[Exception] = None
        failure = FrameUnavailableError(
            "RTSP 通道正在连接: {}".format(self._camera.camera_id)
        )
        image = self._sail.BMImage()

        with self._state_lock:
            decoder = self._decoder
            channel = self._channel
            if decoder is None or channel is None:
                warning_detail = self._record_unavailable_locked(
                    now, "channel is reconnecting"
                )
            else:
                try:
                    # read_mode=0 is SAIL's documented non-blocking path.
                    result = int(decoder.read(channel, image, 0))
                    valid = result == 0 and image.width() > 0 and image.height() > 0
                except Exception as exc:
                    result = -1
                    failure_cause = exc
                    failure = FrameUnavailableError(
                        "RTSP 通道读取异常: {}".format(self._camera.camera_id)
                    )
                else:
                    failure = FrameUnavailableError(
                        "RTSP 通道当前无可用帧: {}".format(
                            self._camera.camera_id
                        )
                    )

                if valid:
                    first_frame = not self._ever_received_frame
                    recovered = (
                        self._ever_received_frame
                        and self._unavailable_since is not None
                    )
                    self._ever_received_frame = True
                    self._unavailable_since = None
                    self._last_warning_at = float("-inf")
                    # Cancel a requested rebuild if SAIL recovered before the
                    # supervisor detached this decoder.
                    self._restart_event.clear()
                else:
                    warning_detail = self._record_unavailable_locked(
                        now, "read result={}".format(result)
                    )
                    if (
                        self._unavailable_since is not None
                        and now - self._unavailable_since
                        >= self._unavailable_timeout_sec
                    ):
                        self._restart_event.set()

        if warning_detail is not None:
            logger.warning(
                "RTSP 当前无可用帧，跳过本轮 camera=%s detail=%s",
                self._camera.camera_id,
                warning_detail,
            )
        if first_frame:
            logger.info("RTSP 首帧已就绪 camera=%s", self._camera.camera_id)
        elif recovered:
            logger.info("RTSP 帧已自动恢复 camera=%s", self._camera.camera_id)
        if not valid:
            if failure_cause is not None:
                raise failure from failure_cause
            raise failure
        return image

    def _record_unavailable_locked(self, now: float, detail: str) -> Optional[str]:
        if self._unavailable_since is None:
            self._unavailable_since = now
        if now - self._last_warning_at < self._reconnect_cooldown_sec:
            return None
        self._last_warning_at = now
        return detail

    def _supervise(self) -> None:
        while not self._stop_event.is_set():
            decoder = None
            channel = -1
            try:
                decoder = self._sail.MultiDecoder(self._queue_size, self._dev_id, 1)
                decoder.set_read_timeout(int(max(1, self._unavailable_timeout_sec)))
                channel = int(decoder.add_channel(self._camera.url, 0))
            except Exception:
                logger.exception(
                    "创建 RTSP 解码通道失败，稍后重试 camera=%s",
                    self._camera.camera_id,
                )
            if decoder is None or channel < 0:
                logger.warning(
                    "RTSP 解码通道尚未建立，稍后重试 camera=%s retry_sec=%.1f",
                    self._camera.camera_id,
                    self._reconnect_cooldown_sec,
                )
                self._stop_event.wait(self._reconnect_cooldown_sec)
                continue

            published = False
            with self._state_lock:
                if not self._stop_event.is_set():
                    self._decoder = decoder
                    self._channel = channel
                    # Give a new channel a full grace period to produce its
                    # first frame before forcing another rebuild.
                    self._unavailable_since = time.monotonic()
                    self._restart_event.clear()
                    published = True

            if published:
                logger.info(
                    "RTSP 解码通道已建立 camera=%s channel=%d",
                    self._camera.camera_id,
                    channel,
                )
                self._wait_until_restart(decoder)

            with self._state_lock:
                if self._decoder is decoder:
                    self._decoder = None
                    self._channel = None

            self._destroy_decoder(decoder, channel)
            if not self._stop_event.is_set():
                logger.warning(
                    "RTSP 解码通道将在后台重建 camera=%s retry_sec=%.1f",
                    self._camera.camera_id,
                    self._reconnect_cooldown_sec,
                )
                self._stop_event.wait(self._reconnect_cooldown_sec)

    def _wait_until_restart(self, decoder: Any) -> None:
        while not self._stop_event.is_set():
            if not self._restart_event.wait(0.5):
                continue
            self._restart_event.clear()
            with self._state_lock:
                if self._decoder is not decoder:
                    return
                # A frame may have recovered after requesting a rebuild but
                # before the supervisor acquired the state lock.
                if self._unavailable_since is None:
                    continue
                # Detach atomically with the health check. A scheduler read
                # can no longer race in, recover a frame, and then have that
                # healthy decoder destroyed by this supervisor.
                self._decoder = None
                self._channel = None
            return

    def _destroy_decoder(self, decoder: Any, channel: int) -> None:
        if channel < 0:
            return
        try:
            decoder.del_channel(channel)
        except Exception:
            logger.exception(
                "关闭 RTSP 解码通道失败 camera=%s channel=%d",
                self._camera.camera_id,
                channel,
            )

    def request_stop(self) -> None:
        self._stop_event.set()
        self._restart_event.set()

    def join(self, timeout: float) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()


class MultiRtspSource:
    """Independent per-camera decoders with non-blocking newest-frame reads."""

    _SHUTDOWN_TIMEOUT_SEC = 25.0

    def __init__(
        self,
        sail_module: Any,
        cameras: Sequence[CameraConfig],
        dev_id: int,
        queue_size: int,
        read_timeout_sec: int,
        reconnect_cooldown_sec: float,
    ) -> None:
        self._streams: Dict[str, _CameraStream] = {}
        try:
            for camera in cameras:
                stream = _CameraStream(
                    sail_module,
                    camera,
                    dev_id,
                    queue_size,
                    float(read_timeout_sec),
                    reconnect_cooldown_sec,
                )
                self._streams[camera.camera_id] = stream
            for stream in self._streams.values():
                stream.start()
        except Exception:
            self.close()
            raise

    def read(self, camera_id: str) -> Any:
        return self._streams[camera_id].read()

    def close(self) -> None:
        streams = list(getattr(self, "_streams", {}).items())
        for _, stream in streams:
            stream.request_stop()
        deadline = time.monotonic() + self._SHUTDOWN_TIMEOUT_SEC
        for camera_id, stream in streams:
            remaining = max(0.0, deadline - time.monotonic())
            if not stream.join(remaining):
                logger.warning("RTSP 后台线程未在期限内退出 camera=%s", camera_id)
        if hasattr(self, "_streams"):
            self._streams.clear()
