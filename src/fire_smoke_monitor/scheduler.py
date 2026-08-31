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
        now = clock()
        self._jobs = [_Job(camera, config.interval_for(camera), now) for camera in config.cameras]
        if self._health is not None:
            self._health.beat("scheduler")
            self._health.require_heartbeat("scheduler", config.health.scheduler_stall_sec)

    def run(self, stop_event: threading.Event) -> None:
        logger.info("取帧调度已启动 mode=%s cameras=%d local_save=disabled",
                    self._mode, len(self._jobs))
        while not stop_event.is_set() and not self._fatal():
            self._beat()
            if self.run_iteration(stop_event):
                continue
            delay = min(job.next_due for job in self._jobs) - self._clock()
            stop_event.wait(max(0.01, min(0.5, delay)))

    def run_iteration(self, stop_event: Optional[threading.Event] = None) -> int:
        now = self._clock()
        self._beat()
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
        camera = job.camera
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
            state = self._uploader.submit(UploadFrame(
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
        finally:
            # Fixed delay: no replay/catch-up even if hardware work was slow.
            job.next_due = self._clock() + job.interval_sec

    def _beat(self) -> None:
        if self._health is not None:
            self._health.beat("scheduler")

    def _fatal(self) -> bool:
        return self._health is not None and self._health.fatal_event.is_set()
