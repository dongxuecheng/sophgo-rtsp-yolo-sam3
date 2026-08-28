from pathlib import Path
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
        def __init__(self, *args):
            events.append("uploader-init")

        def close(self):
            events.append("uploader-close")

    class FakeSource:
        def __init__(self, *args):
            events.append("source-init")

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

    def fail_source(*args):
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
