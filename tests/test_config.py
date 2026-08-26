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
        "runtime": {
            "dev_id": 0,
            "output_root": "../output",
        },
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

    assert config.yolo26.enabled is True
    assert config.yolo26.interval_sec == 3
    assert config.runtime.output_root == (tmp_path / "output").resolve()
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

    with pytest.raises(ConfigError, match="true 或 false"):
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
    assert config.yolo26.enabled is True


def test_runtime_path_check_only_requires_model_and_class_names(tmp_path: Path):
    data = valid_data()
    config = parse_config(data, tmp_path / "config")
    config.yolo26.bmodel.parent.mkdir(parents=True)
    config.yolo26.bmodel.touch()
    config.yolo26.class_names.parent.mkdir(parents=True, exist_ok=True)
    config.yolo26.class_names.write_text("fire\nsmoke\n", encoding="utf-8")

    validate_runtime_paths(config)
