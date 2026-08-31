"""Bounded file logs, disk protection and native stdout/stderr capture."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import errno
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from typing import Callable, Optional, Sequence, TextIO, Tuple

from .health import HealthRegistry


_LOG_NAME = re.compile(
    r"monitor-(\d{4}-\d{2}-\d{2})(?:\.(\d{3,6}))?\.log\Z"
)
_RTSP_URL = re.compile(r"rtsps?://[^\s\"'<>]+", re.IGNORECASE)
_MAINTENANCE_INTERVAL_SEC = 60
_MIB = 1024 * 1024
_GIB = 1024 * _MIB


class _ConsoleHandler(logging.StreamHandler):
    def handleError(self, record: logging.LogRecord) -> None:
        pass


class RedactingFormatter(logging.Formatter):
    def __init__(self, secrets: Sequence[str] = ()):
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._secrets = tuple(value for value in secrets if value)

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        text = _RTSP_URL.sub("rtsp://<redacted>", text)
        for secret in self._secrets:
            text = text.replace(secret, "<redacted>")
        return text


class DailyFileHandler(logging.Handler):
    """Date/size rotation with age, total-size and filesystem-free guards."""

    def __init__(
        self,
        directory: Path,
        retention_days: int = 30,
        clock: Callable[[], float] = time.time,
        error_stream: TextIO = sys.stderr,
        max_file_bytes: int = 100 * _MIB,
        max_total_bytes: int = _GIB,
        min_free_bytes: int = 0,
        disk_usage: Callable[[Path], object] = shutil.disk_usage,
        health: Optional[HealthRegistry] = None,
    ):
        super().__init__()
        if not 1 <= retention_days <= 30:
            raise ValueError("日志保留天数必须为 1-30")
        if max_file_bytes < 1 or max_total_bytes < max_file_bytes:
            raise ValueError("日志容量限制无效")
        if min_free_bytes < 0:
            raise ValueError("磁盘最小剩余空间不能为负数")
        self.directory = directory.resolve()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._days = retention_days
        self._clock = clock
        self._error_stream = error_stream
        self._max_file_bytes = max_file_bytes
        self._max_total_bytes = max_total_bytes
        self._min_free_bytes = min_free_bytes
        self._disk_usage = disk_usage
        self._health = health
        self._recovery_margin = (
            min(512 * _MIB, max(64 * _MIB, min_free_bytes // 10))
            if min_free_bytes else 0
        )
        self._day: Optional[date] = None
        self._active_path: Optional[Path] = None
        self._stream = None
        self._current_size = 0
        self._managed_total_bytes = 0
        self._last_error = float("-inf")
        self._disk_low = False
        self._stopped = False
        self._rotate_day()
        self._maintain_locked()

    @property
    def disk_low(self) -> bool:
        return self._disk_low

    def _active_name(self, day: date) -> Path:
        return self.directory / "monitor-{}.log".format(day.isoformat())

    def _open_active(self) -> None:
        path = self._active_path
        if path is None:
            raise OSError("日志活动文件未初始化")
        if path.is_symlink():
            raise OSError("拒绝写入符号链接日志")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(path), flags, 0o600)
        self._current_size = os.fstat(fd).st_size
        self._stream = os.fdopen(fd, "ab", buffering=0)

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _rotate_day(self) -> None:
        today = datetime.fromtimestamp(self._clock()).date()
        if self._day == today and self._stream is not None:
            return
        self._close_stream()
        self._day = today
        self._active_path = self._active_name(today)
        if (
            self._active_path.exists()
            and not self._active_path.is_symlink()
            and self._active_path.stat().st_size >= self._max_file_bytes
        ):
            self._roll_active_file()
            return
        self._open_active()

    def _next_segment_path(self) -> Path:
        if self._day is None:
            raise OSError("日志日期未初始化")
        highest = 0
        prefix = "monitor-{}".format(self._day.isoformat())
        for _, segment, path, _ in self._managed_files():
            if path != self._active_path and path.name.startswith(prefix):
                highest = max(highest, segment)
        return self.directory / "monitor-{}.{:03d}.log".format(
            self._day.isoformat(), highest + 1
        )

    def _roll_active_file(self) -> None:
        path = self._active_path
        if path is None:
            return
        self._close_stream()
        if path.exists():
            if path.is_symlink():
                raise OSError("拒绝轮转符号链接日志")
            target = self._next_segment_path()
            while target.exists():
                if target.is_symlink():
                    raise OSError("拒绝覆盖符号链接日志")
                match = _LOG_NAME.fullmatch(target.name)
                next_index = int(match.group(2)) + 1 if match and match.group(2) else 1
                target = self.directory / "monitor-{}.{:03d}.log".format(
                    self._day.isoformat(), next_index
                )
            os.replace(str(path), str(target))
        self._open_active()

    def _managed_files(self) -> Sequence[Tuple[date, int, Path, int]]:
        result = []
        for path in self.directory.iterdir():
            match = _LOG_NAME.fullmatch(path.name)
            if not match or path.is_symlink() or not path.is_file():
                continue
            try:
                day = date.fromisoformat(match.group(1))
                size = path.stat().st_size
            except (ValueError, OSError):
                continue
            segment = int(match.group(2)) if match.group(2) else 1_000_000_000
            result.append((day, segment, path, size))
        return result

    def _delete(self, path: Path) -> bool:
        if path == self._active_path or path.is_symlink():
            return False
        try:
            path.unlink()
            return True
        except OSError as exc:
            self._report_error(exc)
            return False

    def _free_bytes(self) -> int:
        return int(self._disk_usage(self.directory).free)

    def _cleanup_storage(self) -> None:
        if self._day is None:
            return
        cutoff = self._day - timedelta(days=self._days - 1)
        for day, _, path, _ in self._managed_files():
            if day < cutoff:
                self._delete(path)

        files = sorted(self._managed_files(), key=lambda item: (item[0], item[1]))
        total = sum(item[3] for item in files)
        try:
            free = self._free_bytes()
        except OSError as exc:
            self._report_error(exc)
            return
        for _, _, path, size in files:
            if total <= self._max_total_bytes and (
                not self._min_free_bytes or free >= self._min_free_bytes
            ):
                break
            if self._delete(path):
                total -= size
                try:
                    free = self._free_bytes()
                except OSError as exc:
                    self._report_error(exc)
                    break
        self._managed_total_bytes = total
        self._update_disk_state(free)

    def _update_disk_state(self, free: int) -> None:
        if not self._min_free_bytes:
            low = False
        elif self._disk_low:
            low = free < self._min_free_bytes + self._recovery_margin
        else:
            low = free < self._min_free_bytes
        if low == self._disk_low:
            return
        self._disk_low = low
        if low:
            if self._health is not None:
                self._health.degrade("disk", "low_free_space")
            self._write_emergency(
                "日志进入低磁盘紧急模式：暂停 DEBUG/INFO 文件日志 free_bytes={}\n".format(free)
            )
        else:
            if self._health is not None:
                self._health.recover("disk")
            self._write_emergency(
                "日志退出低磁盘紧急模式 free_bytes={}\n".format(free)
            )

    def _write_emergency(self, message: str) -> None:
        try:
            self._error_stream.write(message)
            self._error_stream.flush()
        except (OSError, ValueError):
            pass

    def _report_error(self, error: Exception) -> None:
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            if not self._disk_low:
                self._disk_low = True
                if self._health is not None:
                    self._health.degrade("disk", "enospc")
        now = time.monotonic()
        if now - self._last_error >= 60:
            self._last_error = now
            self._write_emergency(
                "日志写入/清理失败（检查空间和权限）: {}\n".format(type(error).__name__)
            )

    def _maintain_locked(self) -> None:
        self._rotate_day()
        if self._current_size >= self._max_file_bytes and self._current_size > 0:
            self._roll_active_file()
        self._cleanup_storage()

    def maintain(self) -> None:
        with self.lock:
            if not self._stopped:
                try:
                    self._maintain_locked()
                except (OSError, ValueError) as exc:
                    self._report_error(exc)

    def emit(self, record: logging.LogRecord) -> None:
        if self._stopped or (self._disk_low and record.levelno < logging.ERROR):
            return
        try:
            self._rotate_day()
            encoded = (self.format(record) + "\n").encode("utf-8", errors="backslashreplace")
            if self._current_size and self._current_size + len(encoded) > self._max_file_bytes:
                self._roll_active_file()
                self._cleanup_storage()
            if self._disk_low and record.levelno < logging.ERROR:
                return
            self._stream.write(encoded)
            self._current_size += len(encoded)
            self._managed_total_bytes += len(encoded)
            if self._managed_total_bytes > self._max_total_bytes:
                self._cleanup_storage()
        except (OSError, ValueError) as exc:
            self._report_error(exc)

    def close(self) -> None:
        with self.lock:
            self._stopped = True
            try:
                self._close_stream()
            except OSError as exc:
                self._report_error(exc)
        super().close()


class NativeOutputCapture:
    """Capture SAIL/FFmpeg fd output and suppress identical high-rate lines."""

    def __init__(self, repeat_window_sec: float = 60.0):
        self._saved = {}
        self._thread = None
        self._reader = None
        self._repeat_window = repeat_window_sec

    def start(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        read_fd, write_fd = os.pipe()
        self._reader = read_fd
        try:
            for fd in (1, 2):
                self._saved[fd] = os.dup(fd)
                os.dup2(write_fd, fd)
            self._thread = threading.Thread(target=self._read, name="native-logs", daemon=True)
            self._thread.start()
        except Exception:
            for fd, saved in self._saved.items():
                os.dup2(saved, fd)
                os.close(saved)
            self._saved.clear()
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)

    @staticmethod
    def _level(text: str) -> int:
        lowered = text.lower()
        if "[error]" in lowered:
            return logging.ERROR
        if "[warning]" in lowered or "[warn]" in lowered:
            return logging.WARNING
        return logging.INFO

    def _read(self) -> None:
        logger = logging.getLogger("sophon.native")
        last_text = ""
        last_level = logging.INFO
        repeated = 0
        window_started = time.monotonic()

        def flush_repeated() -> None:
            nonlocal repeated
            if repeated:
                logger.log(last_level, "suppressed repeated native log count=%d", repeated)
                repeated = 0

        try:
            with os.fdopen(self._reader, "rb") as stream:
                while True:
                    line = stream.readline(16384)
                    if not line:
                        return
                    text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not text:
                        continue
                    now = time.monotonic()
                    if text == last_text and now - window_started < self._repeat_window:
                        repeated += 1
                        continue
                    if text == last_text:
                        flush_repeated()
                        window_started = now
                        logger.log(last_level, "%s", text)
                        continue
                    flush_repeated()
                    last_text = text
                    last_level = self._level(text)
                    window_started = now
                    logger.log(last_level, "%s", text)
        finally:
            flush_repeated()

    def is_healthy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self) -> None:
        if not self._saved:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, saved in self._saved.items():
            os.dup2(saved, fd)
            os.close(saved)
        self._saved.clear()
        if self._thread is not None:
            self._thread.join(timeout=3)


class LogManager:
    def __init__(
        self,
        directory: Path,
        retention_days: int,
        level: str,
        secrets: Sequence[str] = (),
        console: bool = True,
        capture_native: bool = True,
        max_file_bytes: int = 100 * _MIB,
        max_total_bytes: int = _GIB,
        min_free_bytes: int = 0,
        maintenance_interval_sec: Optional[float] = None,
        health: Optional[HealthRegistry] = None,
    ):
        self._root = logging.getLogger()
        self._old_handlers = self._root.handlers[:]
        self._old_level = self._root.level
        self._console = None
        self._file = None
        self._capture = NativeOutputCapture() if capture_native else None
        self._stop = threading.Event()
        self._maintenance = None
        self._closed = False
        self._interval = (
            _MAINTENANCE_INTERVAL_SEC
            if maintenance_interval_sec is None else maintenance_interval_sec
        )
        self._health = health
        self._noisy_levels = {
            name: logging.getLogger(name).level for name in ("httpx", "httpcore")
        }
        try:
            self._console = os.fdopen(
                os.dup(2), "w", encoding="utf-8", errors="backslashreplace", buffering=1
            )
            self._file = DailyFileHandler(
                directory,
                retention_days,
                error_stream=self._console,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
                min_free_bytes=min_free_bytes,
                health=health,
            )
            formatter = RedactingFormatter(secrets)
            self._file.setFormatter(formatter)
            handlers = [self._file]
            if console:
                handler = _ConsoleHandler(self._console)
                handler.setFormatter(formatter)
                handlers.append(handler)
            self._root.handlers = handlers
            self._root.setLevel(level)
            for name in self._noisy_levels:
                logging.getLogger(name).setLevel(logging.WARNING)
            if self._capture is not None:
                self._capture.start()
            self._maintenance = threading.Thread(
                target=self._maintain, name="log-retention", daemon=True
            )
            self._maintenance.start()
            if self._health is not None:
                self._health.register_probe("logging", self.is_healthy)
        except Exception:
            self.close()
            raise

    def _maintain(self) -> None:
        while not self._stop.wait(self._interval):
            self._file.maintain()

    def is_healthy(self) -> bool:
        maintenance_ok = self._maintenance is not None and self._maintenance.is_alive()
        capture_ok = self._capture is None or self._capture.is_healthy()
        return not self._closed and maintenance_ok and capture_ok

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._maintenance is not None:
            self._maintenance.join(timeout=2)
        if self._capture is not None:
            self._capture.close()
        self._root.handlers = self._old_handlers
        self._root.setLevel(self._old_level)
        for name, level in self._noisy_levels.items():
            logging.getLogger(name).setLevel(level)
        if self._file is not None:
            self._file.close()
        if self._console is not None:
            self._console.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
