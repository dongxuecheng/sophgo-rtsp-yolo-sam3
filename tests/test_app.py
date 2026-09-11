from pathlib import Path
import json
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from fire_smoke_monitor import app, cli, models
from fire_smoke_monitor.config import parse_config


def periodic_config():
    return parse_config({
        "mode": "periodic",
        "pipeline": {"endpoint": "http://example.invalid/v1/frames", "machine_id": "box-1"},
        "cameras": [{"id": "cam1", "url": "rtsp://example.invalid/live"}],
    }, Path("."))


def install_fakes(monkeypatch, events):
    package = ModuleType("sophon")
    sail = ModuleType("sophon.sail")
    sail.Handle = lambda dev: SimpleNamespace(get_target=lambda: "BM1684X")
    package.sail = sail
    monkeypatch.setitem(sys.modules, "sophon", package)
    monkeypatch.setitem(sys.modules, "sophon.sail", sail)

    def no_yolo(*args):
        raise AssertionError("periodic must not construct YOLO")

    class FakeUploader:
        def __init__(self, *args, **kwargs):
            assert kwargs["periodic"].upload_interval_sec == 2.5
            events.append("uploader-init")

        def is_healthy(self):
            return True

        def close(self):
            events.append("uploader-close")

    class FakeSource:
        def __init__(self, *args, **kwargs):
            assert args[3] == 1  # latest-only decoder queue reaches hardware
            events.append("source-init")

        def is_healthy(self):
            return True

        def close(self):
            events.append("source-close")

    monkeypatch.setattr(models, "Yolo26Detector", no_yolo)
    monkeypatch.setattr(app, "JpegEncoder", lambda *args: object())
    monkeypatch.setattr(app, "FrameUploader", FakeUploader)
    monkeypatch.setattr(app, "MultiRtspSource", FakeSource)


def test_periodic_application_never_loads_yolo_and_closes_once(monkeypatch):
    events = []
    install_fakes(monkeypatch, events)
    application = app.build_application(periodic_config())
    assert application._detector is None
    application.close()
    application.close()
    assert events == ["uploader-init", "source-init", "uploader-close", "source-close"]


def test_startup_failure_closes_started_uploader(monkeypatch):
    events = []
    install_fakes(monkeypatch, events)

    def fail_source(*args, **kwargs):
        raise RuntimeError("source init error")

    monkeypatch.setattr(app, "MultiRtspSource", fail_source)
    with pytest.raises(RuntimeError, match="source init error"):
        app.build_application(periodic_config())
    assert events == ["uploader-init", "uploader-close"]


def test_check_periodic_config_without_hardware_models_or_network(monkeypatch):
    def forbidden(*args):
        raise AssertionError("must not initialize application")

    monkeypatch.setattr(cli, "build_application", forbidden)
    path = Path(__file__).resolve().parents[1] / "config/config.periodic.example.yaml"
    assert cli.main(["--config", str(path), "--check-config"]) == 0


def test_yaml_syntax_error_does_not_log_config_contents(tmp_path, caplog):
    path = tmp_path / "invalid.yaml"
    path.write_text("cameras: [rtsp://secret-password\n", encoding="utf-8")
    assert cli.main(["--config", str(path), "--check-config"]) == 2
    assert "secret-password" not in caplog.text


@pytest.mark.parametrize("failure", ["none", "init", "run", "close"])
def test_cli_captures_lifecycle_and_native_logs_without_hardware(tmp_path, failure):
    # Isolate process-wide fd capture and signal handlers from the pytest runner.
    config_path = tmp_path / "config.json"
    log_dir = tmp_path / "logs"
    config_path.write_text(json.dumps({
        "mode": "periodic",
        "runtime": {"log_dir": str(log_dir)},
        "pipeline": {"endpoint": "http://example.invalid/v1/frames", "machine_id": "box-1"},
        "cameras": [{"id": "cam1", "url": "rtsp://user:password@example.invalid/live"}],
    }), encoding="utf-8")
    code = '''
import os, signal, sys
from fire_smoke_monitor import cli
failure = sys.argv[2]
old_handler = signal.getsignal(signal.SIGTERM)
class FakeApplication:
    def run(self, stop_event):
        os.write(1, b"native-run rtsp://user:password@example.invalid/live\\n")
        if failure == "run":
            raise RuntimeError("run failure")
        signal.raise_signal(signal.SIGTERM)
        assert stop_event.is_set()
    def close(self):
        os.write(2, b"native-close\\n")
        if failure == "close":
            raise RuntimeError("close failure")
def build(config, health=None):
    if failure == "init":
        raise RuntimeError("init failure")
    return FakeApplication()
cli.build_application = build
status = cli.main(["--config", sys.argv[1], "--no-console"])
assert signal.getsignal(signal.SIGTERM) == old_handler
sys.exit(status)
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-u", "-c", code, str(config_path), failure],
        capture_output=True, timeout=10, env=env,
    )
    assert completed.returncode == (0 if failure == "none" else 1), completed.stderr
    assert completed.stdout == completed.stderr == b""
    text = next(log_dir.glob("monitor-*.log")).read_text(encoding="utf-8")
    assert "日志已启用" in text and "服务已停止" in text
    if failure != "init":
        assert "native-run" in text and "native-close" in text
    if failure != "none":
        assert "Traceback" in text and failure + " failure" in text
    assert "password" not in text
