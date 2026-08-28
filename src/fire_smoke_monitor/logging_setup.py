"""Daily file logs, age-based retention and native stdout/stderr capture."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Callable, Sequence, TextIO


_LOG_NAME = re.compile(r"monitor-(\d{4}-\d{2}-\d{2})\.log\Z")
_RTSP_URL = re.compile(r"rtsps?://[^\s\"'<>]+", re.IGNORECASE)
_MAINTENANCE_INTERVAL_SEC = 60


class _ConsoleHandler(logging.StreamHandler):
    def handleError(self, record: logging.LogRecord) -> None:
        # A broken console must not produce a traceback back into captured fd 2.
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
    """One append-only file per local date; keep today and the previous N-1 days.

    Unlike a backup-count-only handler, this also expires logs after long periods
    of downtime. Only regular monitor-YYYY-MM-DD.log files are managed.
    """

    def __init__(self, directory: Path, retention_days: int = 30,
                 clock: Callable[[], float] = time.time, error_stream: TextIO = sys.stderr):
        super().__init__()
        if not 1 <= retention_days <= 30:
            raise ValueError("日志保留天数必须为 1-30")
        self.directory = directory.resolve()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._days = retention_days
        self._clock = clock
        self._error_stream = error_stream
        self._day = None
        self._stream = None
        self._last_error = float("-inf")
        self._stopped = False
        self._rotate()

    def _rotate(self) -> None:
        today = datetime.fromtimestamp(self._clock()).date()
        if self._day == today:
            return
        filename = self.directory / "monitor-{}.log".format(today.isoformat())
        if filename.is_symlink():
            raise OSError("拒绝写入符号链接日志")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(filename), flags, 0o600)
        stream = os.fdopen(fd, "a", encoding="utf-8", errors="backslashreplace")
        if self._stream is not None:
            self._stream.close()
        self._stream, self._day = stream, today
        self._prune(today)

    def _prune(self, today: date) -> None:
        cutoff = today - timedelta(days=self._days - 1)
        for path in self.directory.iterdir():
            match = _LOG_NAME.fullmatch(path.name)
            if not match or path.is_symlink() or not path.is_file():
                continue
            try:
                day = date.fromisoformat(match.group(1))
            except ValueError:
                continue
            if day < cutoff:
                try:
                    path.unlink()
                except OSError as exc:
                    self._report_error(exc)

    def _report_error(self, error: Exception) -> None:
        # Do not call logging/handleError here: stderr itself may be captured.
        now = time.monotonic()
        if now - self._last_error >= 60:
            self._last_error = now
            try:
                self._error_stream.write("日志写入/清理失败（检查空间和权限）: {}\n".format(type(error).__name__))
                self._error_stream.flush()
            except (OSError, ValueError):
                pass

    def maintain(self) -> None:
        with self.lock:
            if not self._stopped:
                try:
                    self._rotate()
                    self._prune(self._day)
                except (OSError, ValueError) as exc:
                    self._report_error(exc)

    def emit(self, record: logging.LogRecord) -> None:
        if self._stopped:
            return
        try:
            self._rotate()
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except (OSError, ValueError) as exc:
            self._report_error(exc)

    def close(self) -> None:
        with self.lock:
            self._stopped = True
            if self._stream is not None:
                try:
                    self._stream.close()
                except OSError as exc:
                    self._report_error(exc)
                self._stream = None
        super().close()


class NativeOutputCapture:
    """Capture fd 1/2, including SAIL/FFmpeg writes that bypass Python logging.

    The console handler MUST point at a saved fd, not the redirected sys.stderr.
    Use stdbuf in start.sh to keep C stdio from buffering diagnostics in the pipe.
    """

    def __init__(self):
        self._saved = {}
        self._thread = None
        self._reader = None

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

    def _read(self) -> None:
        logger = logging.getLogger("sophon.native")
        # readline(size) also bounds unterminated/progress-output lines.
        with os.fdopen(self._reader, "rb") as stream:
            while True:
                line = stream.readline(16384)
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not text:
                    continue
                lowered = text.lower()
                level = logging.ERROR if "[error]" in lowered else (
                    logging.WARNING if "[warning]" in lowered or "[warn]" in lowered else logging.INFO
                )
                logger.log(level, "%s", text)

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
    def __init__(self, directory: Path, retention_days: int, level: str,
                 secrets: Sequence[str] = (), console: bool = True,
                 capture_native: bool = True):
        self._root = logging.getLogger()
        self._old_handlers = self._root.handlers[:]
        self._old_level = self._root.level
        self._console = None
        self._file = None
        self._capture = NativeOutputCapture() if capture_native else None
        self._stop = threading.Event()
        self._maintenance = None
        self._closed = False
        self._noisy_levels = {name: logging.getLogger(name).level for name in ("httpx", "httpcore")}
        try:
            self._console = os.fdopen(os.dup(2), "w", encoding="utf-8", errors="backslashreplace", buffering=1)
            self._file = DailyFileHandler(directory, retention_days, error_stream=self._console)
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
            self._maintenance = threading.Thread(target=self._maintain, name="log-retention", daemon=True)
            self._maintenance.start()
        except Exception:
            self.close()
            raise

    def _maintain(self) -> None:
        while not self._stop.wait(_MAINTENANCE_INTERVAL_SEC):
            self._file.maintain()

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
