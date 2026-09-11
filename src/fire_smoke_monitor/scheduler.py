from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any, Callable, Optional

from .config import AppConfig, CameraConfig
from .domain import UploadFrame
from .health import HealthRegistry
from .streams import FrameUnavailableError


logger = logging.getLogger(__name__)


@dataclass
class _Job:
    camera: CameraConfig
    interval_sec: float
    next_due: float


class InferenceScheduler:
    """Single hardware thread; HTTP is always handled by the background uploader."""

    def __init__(
        self, config: AppConfig, source: Any, encoder: Any, uploader: Any,
        detector: Optional[Any] = None,
        clock: Callable[[], float] = time.monotonic,
        health: Optional[HealthRegistry] = None,
    ) -> None:
        if (config.mode == "yolo_trigger") != (detector is not None):
            raise ValueError("YOLO 检测器必须与 mode 一致")
        self._mode = config.mode
        self._source = source
        self._encoder = encoder
        self._uploader = uploader
        self._detector = detector
        self._clock = clock
        self._health = health
        self._cameras = config.cameras
        self._cursor = 0
        self._periodic_interval = config.periodic.upload_interval_sec
        now = clock()
        self._jobs = [] if config.mode == "periodic" else [
            _Job(camera, config.interval_for(camera), now) for camera in config.cameras
        ]
        if self._health is not None:
            self._health.beat("scheduler")
            self._health.require_heartbeat("scheduler", config.health.scheduler_stall_sec)

    def run(self, stop_event: threading.Event) -> None:
        logger.info("取帧调度已启动 mode=%s cameras=%d local_save=disabled",
                    self._mode, len(self._cameras))
        if self._mode == "periodic":
            logger.info("统一轮询已启用 upload_interval=%.3fs latest_only=true pending_fifo=disabled",
                        self._periodic_interval)
        while not stop_event.is_set() and not self._fatal():
            self._beat()
            if self.run_iteration(stop_event):
                continue
            delay = (0.1 if self._mode == "periodic" else
                     min(job.next_due for job in self._jobs) - self._clock())
            stop_event.wait(max(0.01, min(0.5, delay)))

    def run_iteration(self, stop_event: Optional[threading.Event] = None) -> int:
        now = self._clock()
        self._beat()
        if self._mode == "periodic":
            return self._run_periodic(stop_event)
        due = sorted((job for job in self._jobs if job.next_due <= now),
                     key=lambda job: job.next_due)
        count = 0
        for job in due:
            if (stop_event is not None and stop_event.is_set()) or self._fatal():
                break
            self._run_job(job)
            self._beat()
            count += 1
        return count

    def _run_job(self, job: _Job) -> None:
        try:
            self._process_camera(job.camera, self._uploader.submit)
        finally:
            # YOLO retains per-camera fixed delays, with no replay/catch-up.
            job.next_due = self._clock() + job.interval_sec

    def _run_periodic(self, stop_event: Optional[threading.Event]) -> int:
        if (stop_event is not None and stop_event.is_set()) or self._fatal():
            return 0
        camera = self._cameras[self._cursor]
        state = self._uploader.try_reserve(camera.camera_id)
        if state == "stopped":
            raise RuntimeError("periodic 上传器已停止")
        if state != "reserved":
            # Busy or globally rate-limited: no image is sampled or encoded.
            return 0
        try:
            if (stop_event is not None and stop_event.is_set()) or self._fatal():
                return 0
            # An offline camera consumes this turn, not the whole scheduler.
            self._cursor = (self._cursor + 1) % len(self._cameras)
            self._process_camera(camera, self._uploader.submit_reserved)
            return 1
        finally:
            # No-op after handoff; releases the slot on read/encode failure.
            self._uploader.cancel_reservation(camera.camera_id)
            self._beat()

    def _process_camera(self, camera: CameraConfig, submit: Callable[[UploadFrame], str]) -> None:
        started = self._clock()
        try:
            image = self._source.read(camera.camera_id)
            if self._health is not None:
                self._health.recover("camera_{}".format(camera.camera_id))
            # Host sampling time, not a camera PTS or an HTTP-send timestamp.
            captured_at = datetime.now(timezone.utc)
            frame_time = self._clock()
            detections = []
            if self._detector is not None:
                detections = self._detector.detect(image)
                if not detections:
                    logger.debug("YOLO 未命中 camera=%s", camera.camera_id)
                    if self._health is not None:
                        self._health.processing_success()
                    return
            jpeg = self._encoder.encode(image)
            state = submit(UploadFrame(
                camera.camera_id, camera.name, captured_at, frame_time, jpeg,
            ))
            if state == "stopped":
                logger.error("上传器已停止，当前帧未提交 camera=%s", camera.camera_id)
                if self._health is not None:
                    self._health.fatal("uploader_stopped")
            else:
                log = logger.warning if state == "replaced" else logger.debug
                log("图片待上传 mode=%s camera=%s targets=%d queue=%s bytes=%d elapsed=%.3fs",
                            self._mode, camera.camera_id, len(detections),
                            state, len(jpeg), self._clock() - started)
                if self._health is not None:
                    self._health.processing_success()
        except FrameUnavailableError:
            logger.debug("跳过无帧摄像头 camera=%s", camera.camera_id)
            if self._health is not None:
                self._health.degrade(
                    "camera_{}".format(camera.camera_id), "frame_unavailable"
                )
        except Exception as exc:
            logger.exception("当前帧处理失败 camera=%s mode=%s", camera.camera_id, self._mode)
            if self._health is not None:
                count = self._health.processing_failure(type(exc).__name__)
                logger.warning(
                    "连续内部处理失败 count=%d limit=%d",
                    count, self._health.processing_failure_limit,
                )
    def _beat(self) -> None:
        if self._health is not None:
            self._health.beat("scheduler")

    def _fatal(self) -> bool:
        return self._health is not None and self._health.fatal_event.is_set()
