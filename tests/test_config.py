from pathlib import Path

import pytest

from fire_smoke_monitor.config import (
    ConfigError,
    MAX_CAMERAS,
    load_config,
    parse_config,
    validate_runtime_paths,
)


def valid_data():
    return {
        "mode": "yolo_trigger",
        "runtime": {"dev_id": 0},
        "pipeline": {"endpoint": "http://example.invalid:18080/v1/frames", "machine_id": "box-1"},
        "cameras": [
            {"id": "north_gate", "url": "rtsp://example.invalid/stream"}
        ],
        "models": {
            "yolo26": {
                "enabled": True,
                "bmodel": "../models/yolo.bmodel",
                "class_names": "fire_smoke.names",
                "target_classes": ["fire", "smoke"],
                "interval_sec": 3,
            },
        },
    }


def test_parse_valid_yolo_only_and_resolve_relative_paths(tmp_path: Path):
    config = parse_config(valid_data(), tmp_path / "config")

    assert config.mode == "yolo_trigger"
    assert config.yolo26.interval_sec == 3
    assert config.yolo26.class_names == (tmp_path / "config/fire_smoke.names").resolve()


def test_reject_yolo_disabled(tmp_path: Path):
    data = valid_data()
    data["models"]["yolo26"]["enabled"] = False

    with pytest.raises(ConfigError, match="必须为 true"):
        parse_config(data, tmp_path)


def test_accept_maximum_camera_count(tmp_path: Path):
    data = valid_data()
    data["cameras"] = [
        {"id": "cam_{}".format(index), "url": "rtsp://example.invalid/{}".format(index)}
        for index in range(MAX_CAMERAS)
    ]

    config = parse_config(data, tmp_path)

    assert len(config.cameras) == MAX_CAMERAS


def test_reject_more_than_maximum_cameras(tmp_path: Path):
    data = valid_data()
    data["cameras"] = [
        {"id": "cam_{}".format(index), "url": "rtsp://example.invalid/{}".format(index)}
        for index in range(MAX_CAMERAS + 1)
    ]

    with pytest.raises(ConfigError, match="1 到 {}".format(MAX_CAMERAS)):
        parse_config(data, tmp_path)


def test_reject_non_boolean_model_switch(tmp_path: Path):
    data = valid_data()
    data["models"]["yolo26"]["enabled"] = "false"

    with pytest.raises(ConfigError, match="必须为 true"):
        parse_config(data, tmp_path)


def test_reject_camera_path_traversal(tmp_path: Path):
    data = valid_data()
    data["cameras"][0]["id"] = ".."

    with pytest.raises(ConfigError, match="camera id"):
        parse_config(data, tmp_path)


def test_example_yaml_is_parseable():
    project_root = Path(__file__).resolve().parents[1]

    config = load_config(project_root / "config/config.example.yaml")

    assert config.cameras[0].camera_id == "camera_01"
    assert config.yolo26 is not None


def test_runtime_path_check_only_requires_model_and_class_names(tmp_path: Path):
    data = valid_data()
    config = parse_config(data, tmp_path / "config")
    config.yolo26.bmodel.parent.mkdir(parents=True)
    config.yolo26.bmodel.touch()
    config.yolo26.class_names.parent.mkdir(parents=True, exist_ok=True)
    config.yolo26.class_names.write_text("fire\nsmoke\n", encoding="utf-8")

    validate_runtime_paths(config)


def test_periodic_needs_no_models_and_ignores_disabled_model_paths(tmp_path):
    data = valid_data()
    data["mode"] = "periodic"
    data.pop("models")
    data["periodic"] = {"interval_sec": 12}
    config = parse_config(data, tmp_path)
    assert config.yolo26 is None
    assert config.interval_for(config.cameras[0]) == 12
    validate_runtime_paths(config)
    data["models"] = {"yolo26": {"enabled": True, "bmodel": "/missing"}}
    assert parse_config(data, tmp_path).yolo26 is None


@pytest.mark.parametrize("mode", ["periodic", "yolo_trigger"])
def test_camera_interval_overrides_mode_default(tmp_path, mode):
    data = valid_data()
    data["mode"] = mode
    data["cameras"][0]["interval_sec"] = 7.5
    config = parse_config(data, tmp_path)
    assert config.interval_for(config.cameras[0]) == 7.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, True])
def test_reject_bad_intervals(tmp_path, value):
    data = valid_data()
    data["cameras"][0]["interval_sec"] = value
    with pytest.raises(ConfigError):
        parse_config(data, tmp_path)


@pytest.mark.parametrize("endpoint", [
    "https://github.com/dongxuecheng/ascend-sam3-llm-pipeline",
    "file:///v1/frames", "http://host/v1/frames/", "http://host:bad/v1/frames",
    "http://user:secret@host/v1/frames", "http://host/v1/frames?key=secret",
])
def test_reject_invalid_api_endpoint(tmp_path, endpoint):
    data = valid_data()
    data["pipeline"]["endpoint"] = endpoint
    with pytest.raises(ConfigError, match="endpoint"):
        parse_config(data, tmp_path)


@pytest.mark.parametrize("key,value", [("workers", 0), ("workers", 9), ("workers", 1.5),
                                      ("timeout_sec", 0), ("max_image_bytes", -1),
                                      ("max_pending_age_sec", float("inf")),
                                      ("machine_id", "box.1")])
def test_reject_invalid_pipeline_options(tmp_path, key, value):
    data = valid_data()
    data["pipeline"][key] = value
    with pytest.raises(ConfigError):
        parse_config(data, tmp_path)


def test_api_key_is_resolved_from_environment_without_exposing_it(tmp_path, monkeypatch):
    data = valid_data()
    data["mode"] = "periodic"
    data["pipeline"]["api_key_env"] = "TEST_PIPELINE_KEY"
    config = parse_config(data, tmp_path)
    monkeypatch.delenv("TEST_PIPELINE_KEY", raising=False)
    with pytest.raises(ConfigError, match="TEST_PIPELINE_KEY"):
        validate_runtime_paths(config)
    monkeypatch.setenv("TEST_PIPELINE_KEY", "not-a-real-key")
    validate_runtime_paths(config)
    assert config.pipeline.api_key() == "not-a-real-key"
    assert "not-a-real-key" not in repr(config)
    assert "rtsp://" not in repr(config)


def test_legacy_config_requires_explicit_migration(tmp_path):
    data = valid_data()
    data.pop("mode")
    with pytest.raises(ConfigError, match="mode"):
        parse_config(data, tmp_path)


def test_legacy_save_fields_are_ignored_with_warning(tmp_path, caplog):
    data = valid_data()
    data["runtime"]["output_root"] = "../old-output"
    data["models"]["yolo26"]["save_cooldown_sec"] = 100
    config = parse_config(data, tmp_path)
    assert not hasattr(config.runtime, "output_root")
    assert "已停用" in caplog.text


def test_log_defaults_and_retention_limit(tmp_path):
    data = valid_data()
    config = parse_config(data, tmp_path / "config")
    assert config.runtime.log_dir == (tmp_path / "logs").resolve()
    assert config.runtime.log_retention_days == 30
    for days in (0, 31, -1, 1.5, True):
        data["runtime"]["log_retention_days"] = days
        with pytest.raises(ConfigError, match="log_retention_days"):
            parse_config(data, tmp_path)


def test_key_file_is_literal_environment_takes_precedence(tmp_path, monkeypatch):
    data = valid_data()
    data["mode"] = "periodic"
    data["pipeline"]["api_key_env"] = "TEST_KEY"
    monkeypatch.delenv("TEST_KEY", raising=False)
    path = tmp_path / "pipeline.env"
    path.write_text('# comment\nTEST_KEY="literal-$HOME-$(hostname)"\n', encoding="utf-8")
    config = parse_config(data, tmp_path)
    assert config.pipeline.api_key() == "literal-$HOME-$(hostname)"
    assert "TEST_KEY" not in __import__("os").environ  # no environment mutation
    monkeypatch.setenv("TEST_KEY", "environment-value")
    assert config.pipeline.api_key() == "environment-value"


@pytest.mark.parametrize("contents", ['TEST_KEY="unclosed\n', 'TEST_KEY=one\nTEST_KEY=two\n'])
def test_invalid_key_file_is_rejected_without_leaking_secret(tmp_path, monkeypatch, contents):
    data = valid_data()
    data["pipeline"]["api_key_env"] = "TEST_KEY"
    monkeypatch.delenv("TEST_KEY", raising=False)
    (tmp_path / "pipeline.env").write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigError, match="pipeline.env"):
        parse_config(data, tmp_path).pipeline.api_key()
