from datetime import datetime, timedelta
import io
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from fire_smoke_monitor import logging_setup
from fire_smoke_monitor.health import HealthRegistry
from fire_smoke_monitor.logging_setup import DailyFileHandler, LogManager, RedactingFormatter


def record(message="hello", level=logging.INFO):
    return logging.LogRecord("test", level, __file__, 1, message, (), None)


def test_retention_is_calendar_age_not_backup_count_and_preserves_unmanaged_files(tmp_path):
    today = datetime(2026, 8, 28, 12)
    for offset in range(45):
        day = (today - timedelta(days=offset)).date()
        (tmp_path / "monitor-{}.log".format(day)).write_text("old\n", encoding="utf-8")
    legacy = tmp_path / "periodic-20260701.log"
    legacy.write_text("manual log", encoding="utf-8")
    invalid = tmp_path / "monitor-2026-02-31.log"
    invalid.touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    nested_log = nested / "monitor-2026-01-01.log"
    nested_log.touch()

    handler = DailyFileHandler(tmp_path, clock=lambda: today.timestamp())
    handler.handle(record())
    handler.close()

    assert not (tmp_path / "monitor-2026-07-29.log").exists()
    assert (tmp_path / "monitor-2026-07-30.log").exists()  # day 30, including today
    assert (tmp_path / "monitor-2026-08-28.log").read_text().endswith("hello\n")
    assert len(list(tmp_path.glob("monitor-2026-0[78]-*.log"))) == 30
    assert legacy.exists() and invalid.exists() and nested_log.exists()


def test_restart_after_long_downtime_expires_even_a_single_old_log(tmp_path):
    old = tmp_path / "monitor-2025-01-01.log"
    old.touch()
    handler = DailyFileHandler(tmp_path, clock=lambda: datetime(2026, 8, 28).timestamp())
    handler.close()
    assert not old.exists()


def test_rotate_and_prune_without_any_new_record(tmp_path):
    now = [datetime(2026, 8, 28, 23, 59)]
    handler = DailyFileHandler(tmp_path, 1, clock=lambda: now[0].timestamp())
    handler.handle(record("yesterday"))
    now[0] += timedelta(minutes=2)
    handler.maintain()
    assert not (tmp_path / "monitor-2026-08-28.log").exists()
    handler.handle(record("today"))
    handler.close()
    assert (tmp_path / "monitor-2026-08-29.log").read_text() == "today\n"


def test_restart_same_day_appends_instead_of_truncating(tmp_path):
    clock = lambda: datetime(2026, 8, 28).timestamp()
    for text in ("first", "second"):
        handler = DailyFileHandler(tmp_path, clock=clock)
        handler.handle(record(text))
        handler.close()
    assert (tmp_path / "monitor-2026-08-28.log").read_text() == "first\nsecond\n"


def test_log_file_is_private_on_linux(tmp_path):
    handler = DailyFileHandler(tmp_path)
    handler.close()
    if os.name == "posix":
        assert next(tmp_path.glob("monitor-*.log")).stat().st_mode & 0o777 == 0o600


def test_symlink_is_never_pruned_or_used_as_log(tmp_path):
    outside = tmp_path / "important.txt"
    outside.write_text("keep")
    link = tmp_path / "monitor-2025-01-01.log"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink permission unavailable")
    handler = DailyFileHandler(tmp_path)
    handler.close()
    assert link.is_symlink() and outside.read_text() == "keep"
    current = datetime.now().date()
    today = tmp_path / "monitor-{}.log".format(current)
    today.unlink()
    today.symlink_to(outside)
    with pytest.raises(OSError):
        DailyFileHandler(tmp_path)
    assert outside.read_text() == "keep"


def test_disk_write_failure_is_rate_limited_without_recursive_logging(tmp_path):
    errors = io.StringIO()
    handler = DailyFileHandler(tmp_path, error_stream=errors)
    handler._stream.close()

    class FullDisk:
        def write(self, text):
            raise OSError("disk full")

        def close(self):
            pass

    handler._stream = FullDisk()
    for _ in range(10):
        handler.handle(record())
    handler.close()
    assert errors.getvalue().count("OSError") == 1


def test_size_rotation_and_total_capacity_are_bounded(tmp_path):
    handler = DailyFileHandler(
        tmp_path, max_file_bytes=64, max_total_bytes=128
    )
    for index in range(10):
        handler.handle(record("{:02d}-{}".format(index, "x" * 35)))
    handler.close()

    files = list(tmp_path.glob("monitor-*.log"))
    assert len(files) >= 2
    assert any(len(path.name.split(".")) == 3 for path in files)
    assert sum(path.stat().st_size for path in files) <= 128


def test_low_disk_mode_prunes_old_logs_and_only_keeps_errors(tmp_path):
    old = tmp_path / "monitor-2026-01-01.log"
    old.write_bytes(b"x" * 100)
    free = [50 * 1024 * 1024]
    health = HealthRegistry()

    class Usage:
        @property
        def free(self):
            return free[0]

    handler = DailyFileHandler(
        tmp_path,
        clock=lambda: datetime(2026, 8, 28).timestamp(),
        max_file_bytes=1024,
        max_total_bytes=4096,
        min_free_bytes=100 * 1024 * 1024,
        disk_usage=lambda path: Usage(),
        health=health,
    )
    assert not old.exists()
    assert handler.disk_low
    assert "disk" in health.snapshot().degraded
    handler.handle(record("dropped-info"))
    handler.handle(record("kept-error", logging.ERROR))

    free[0] = 200 * 1024 * 1024
    handler.maintain()
    assert not handler.disk_low
    assert "disk" not in health.snapshot().degraded
    handler.handle(record("kept-info"))
    handler.close()
    text = "".join(path.read_text() for path in tmp_path.glob("monitor-*.log"))
    assert "dropped-info" not in text
    assert "kept-error" in text and "kept-info" in text


def test_redacts_rtsp_and_api_key_in_messages_and_tracebacks():
    formatter = RedactingFormatter(["api-secret"])
    text = formatter.format(record("open rtsp://admin:password@10.0.0.1/live api-secret"))
    assert "password" not in text and "10.0.0.1" not in text and "api-secret" not in text
    try:
        raise RuntimeError("rtsps://admin:other@host/live api-secret")
    except RuntimeError:
        error = logging.LogRecord("test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
        text = formatter.format(error)
    assert "Traceback" in text
    assert "admin:other" not in text and "api-secret" not in text


def test_manager_restores_handlers_and_silences_duplicate_httpx_logs(tmp_path):
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    old_httpx = logging.getLogger("httpx").level
    with LogManager(tmp_path, 30, "INFO", capture_native=False, console=False):
        logging.getLogger("httpx").info("duplicate-http-request")
        logging.getLogger("test").info("application-record")
    assert root.handlers == old_handlers and root.level == old_level
    assert logging.getLogger("httpx").level == old_httpx
    text = next(tmp_path.glob("monitor-*.log")).read_text(encoding="utf-8")
    assert "application-record" in text and "duplicate-http-request" not in text


def test_idle_maintenance_thread_runs_and_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_setup, "_MAINTENANCE_INTERVAL_SEC", 0.01)
    with LogManager(tmp_path, 1, "INFO", capture_native=False, console=False) as manager:
        now = datetime.now() + timedelta(days=1)
        manager._file._clock = lambda: now.timestamp()
        filename = tmp_path / "monitor-{}.log".format(now.date())
        deadline = time.monotonic() + 2
        while not filename.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert filename.exists()
    assert not manager._maintenance.is_alive()


def test_native_fd_capture_and_restoration_in_isolated_process(tmp_path):
    # Never redirect pytest's own process-wide stdout/stderr.
    code = '''
import logging, os, sys
from pathlib import Path
from fire_smoke_monitor.logging_setup import LogManager
with LogManager(Path(sys.argv[1]), 30, "INFO", secrets=("api-secret",)):
    logging.info("python-business-log")
    print("native-python-print", flush=True)
    os.write(1, b"native-stdout rtsp://user:password@host/live\\n")
    os.write(2, b"[warning] native-stderr api-secret\\n")
print("restored-stdout", flush=True)
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run([sys.executable, "-u", "-c", code, str(tmp_path)],
                               capture_output=True, timeout=10, env=env)
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert completed.stdout.strip() == b"restored-stdout"
    text = next(tmp_path.glob("monitor-*.log")).read_text(encoding="utf-8")
    for marker in ("python-business-log", "native-python-print", "native-stdout", "native-stderr"):
        assert marker in text
    assert "password" not in text and "api-secret" not in text
    assert b"password" not in completed.stderr and b"api-secret" not in completed.stderr
    assert "WARNING sophon.native" in text
