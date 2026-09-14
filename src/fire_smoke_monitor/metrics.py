"""Bounded per-camera YOLO coverage and delivery counters (no images retained)."""
from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import threading
import time
from typing import Callable, Optional, Sequence

from .config import HealthConfig
from .health import HealthRegistry


logger = logging.getLogger(__name__)


@dataclass
class CameraStats:
    inferences: int = 0
    hits: int = 0
    no_frame: int = 0
    errors: int = 0
    submitted: int = 0
    replaced: int = 0
    expired: int = 0
    accepted: int = 0
    failed: int = 0
    last_inference_at: Optional[float] = None
    last_capture_at: Optional[float] = None
    last_upload_at: Optional[float] = None
    last_hit_capture: Optional[float] = None
    unconfirmed_since: Optional[float] = None
    inference_sec: float = 0.0
    processing_sec: float = 0.0
    inference_gap_sec: float = 0.0


class CameraMetrics:
    def __init__(self, camera_ids: Sequence[str], config: HealthConfig,
                 health: Optional[HealthRegistry] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._rows = {camera: CameraStats() for camera in camera_ids}
        self._config = config
        self._health = health
        self._clock = clock
        self._started = clock()
        self._last_report = self._started
        self._lock = threading.Lock()

    def event(self, camera: str, name: str) -> None:
        if name not in {"no_frame", "errors", "submitted", "replaced", "expired", "failed"}:
            raise ValueError("unknown camera counter")
        with self._lock:
            row = self._rows[camera]
            setattr(row, name, getattr(row, name) + 1)

    def inferred(self, camera: str, hit: bool, duration: float, captured: float) -> None:
        with self._lock:
            row = self._rows[camera]
            row.inferences += 1
            row.last_inference_at = self._clock()
            row.inference_sec = duration
            if row.last_capture_at is not None:
                row.inference_gap_sec = captured - row.last_capture_at
            row.last_capture_at = captured
            if hit:
                row.hits += 1
                row.last_hit_capture = captured
                if row.unconfirmed_since is None:
                    row.unconfirmed_since = row.last_inference_at

    def processed(self, camera: str, duration: float) -> None:
        with self._lock:
            self._rows[camera].processing_sec = duration

    def accepted(self, camera: str, captured: float) -> None:
        with self._lock:
            row = self._rows[camera]
            row.accepted += 1
            row.last_upload_at = self._clock()
            # An older request completing cannot acknowledge a newer hit.
            row.unconfirmed_since = (
                row.last_inference_at
                if row.last_hit_capture is not None and row.last_hit_capture > captured
                else None
            )

    def snapshot(self, camera: str) -> CameraStats:
        with self._lock:
            return replace(self._rows[camera])

    def inspect(self) -> bool:
        now = self._clock()
        with self._lock:
            rows = [(camera, replace(row)) for camera, row in self._rows.items()]
            report = now - self._last_report >= self._config.status_log_interval_sec
            if report:
                self._last_report = now
        for camera, row in rows:
            inference_age = now - (row.last_inference_at if row.last_inference_at is not None else self._started)
            hit_age = 0.0 if row.unconfirmed_since is None else now - row.unconfirmed_since
            inference_stale = inference_age >= self._config.inference_stale_sec
            delivery_stale = row.unconfirmed_since is not None and hit_age >= self._config.upload_stale_sec
            if self._health is not None:
                for name, stale in (("inference_" + camera, inference_stale),
                                    ("delivery_" + camera, delivery_stale)):
                    if stale:
                        self._health.degrade(name, "coverage_stale")
                    else:
                        self._health.recover(name)
            if report:
                log = logger.warning if inference_stale or delivery_stale else logger.info
                log(
                    "摄像头覆盖 camera=%s inferred=%d hits=%d no_frame=%d errors=%d "
                    "infer_ms=%.1f process_ms=%.1f infer_gap=%.2fs last_infer_age=%.1fs "
                    "submitted=%d accepted=%d replaced=%d expired=%d failed=%d "
                    "last_upload_age=%.1fs unconfirmed_hit_age=%.1fs",
                    camera, row.inferences, row.hits, row.no_frame, row.errors,
                    row.inference_sec * 1000, row.processing_sec * 1000,
                    row.inference_gap_sec, inference_age, row.submitted, row.accepted,
                    row.replaced, row.expired, row.failed,
                    -1.0 if row.last_upload_at is None else now - row.last_upload_at, hit_age,
                )
        # Lack of coverage is degraded, not a reason to restart every camera.
        return True
