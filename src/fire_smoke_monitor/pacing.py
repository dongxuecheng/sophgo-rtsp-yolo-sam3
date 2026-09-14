"""Machine-wide admission pacing; never queues images or replays missed ticks."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .config import PeriodicConfig, UploadBackoffConfig


logger = logging.getLogger(__name__)


class UploadPacer:
    def __init__(self, base_interval: float, backoff: UploadBackoffConfig, mode: str,
                 clock: Callable[[], float] = time.monotonic):
        self._base_interval = base_interval
        self._backoff = backoff
        self._mode = mode
        self._clock = clock
        self._lock = threading.Lock()
        self._interval = base_interval
        self._next_allowed = float("-inf")
        self._blocked_until = float("-inf")
        self._last_started = float("-inf")
        self._failures = 0
        self._successes = 0

    def try_acquire(self) -> bool:
        with self._lock:
            now = self._clock()
            if now < self._next_allowed:
                return False
            self._next_allowed = now + self._interval
            return True

    def start_upload(self) -> bool:
        """Recheck overload after encoding, then space future captures from now."""
        with self._lock:
            now = self._clock()
            if now < max(self._blocked_until, self._last_started + self._interval):
                return False
            self._last_started = now
            self._next_allowed = max(self._next_allowed, now + self._interval)
            return True

    def delay(self) -> float:
        with self._lock:
            return max(0.0, self._next_allowed - self._clock())

    def report_result(self, status_code: Optional[int]) -> None:
        message = None
        with self._lock:
            if status_code in (429, 503):
                self._successes = 0
                self._failures += 1
                if self._failures % self._backoff.after_failures == 0:
                    self._interval = min(
                        self._interval * 2 if self._interval else self._backoff.initial_interval_sec,
                        self._backoff.max_interval_sec,
                    )
                    self._blocked_until = self._clock() + self._interval
                    self._next_allowed = max(self._next_allowed, self._blocked_until)
                    message = (
                        "%s 全机降频 status=%s consecutive=%d upload_interval=%.3fs",
                        self._mode, status_code, self._failures, self._interval,
                    )
            else:
                self._failures = 0
                self._successes = self._successes + 1 if status_code == 202 else 0
                if self._successes >= self._backoff.recover_after_successes:
                    self._successes = 0
                    if self._interval > self._base_interval:
                        if self._base_interval == 0 and self._interval <= self._backoff.initial_interval_sec:
                            self._interval = 0.0
                        else:
                            self._interval = max(self._base_interval, self._interval / 2)
                        # Never pull an existing deadline forward after recovery.
                        message = ("%s 上传频率逐步恢复 upload_interval=%.3fs", self._mode, self._interval)
        if message is not None:
            logger.warning(*message)


class PeriodicPacer(UploadPacer):
    def __init__(self, config: PeriodicConfig, clock: Callable[[], float] = time.monotonic):
        super().__init__(config.upload_interval_sec, UploadBackoffConfig(
            config.backoff_after_failures, config.upload_interval_sec,
            config.backoff_max_interval_sec, config.recover_after_successes,
        ), "periodic", clock)


class YoloUploadPacer(UploadPacer):
    def __init__(self, config: UploadBackoffConfig, clock: Callable[[], float] = time.monotonic):
        # No artificial delay in normal operation. Only HTTP is throttled.
        super().__init__(0.0, config, "yolo_trigger", clock)
