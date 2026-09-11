"""Machine-wide admission pacing; never queues images or replays missed ticks."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .config import PeriodicConfig


logger = logging.getLogger(__name__)


class PeriodicPacer:
    def __init__(self, config: PeriodicConfig, clock: Callable[[], float] = time.monotonic):
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._interval = config.upload_interval_sec
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

    def report_result(self, status_code: Optional[int]) -> None:
        message = None
        with self._lock:
            if status_code in (429, 503):
                self._successes = 0
                self._failures += 1
                if self._failures % self._config.backoff_after_failures == 0:
                    self._interval = min(
                        self._interval * 2, self._config.backoff_max_interval_sec
                    )
                    self._blocked_until = self._clock() + self._interval
                    self._next_allowed = max(self._next_allowed, self._blocked_until)
                    message = (
                        "periodic 全机降频 status=%s consecutive=%d upload_interval=%.3fs",
                        status_code, self._failures, self._interval,
                    )
            else:
                self._failures = 0
                self._successes = self._successes + 1 if status_code == 202 else 0
                if self._successes >= self._config.recover_after_successes:
                    self._successes = 0
                    if self._interval > self._config.upload_interval_sec:
                        self._interval = max(
                            self._config.upload_interval_sec, self._interval / 2
                        )
                        # Never pull an existing deadline forward after recovery.
                        message = ("periodic 上传频率逐步恢复 upload_interval=%.3fs", self._interval)
        if message is not None:
            logger.warning(*message)
