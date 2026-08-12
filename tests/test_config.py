from pathlib import Path

import pytest

from fire_smoke_monitor.config import (
    ConfigError,
    load_config,
    parse_config,
    validate_runtime_paths,
)


def valid_data():
    return {
        "runtime": {
            "dev_id": 0,
            "sophon_demo_root": "../vendor/sophon-demo",
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
            "sam3": {
                "enabled": False,
                "model_dir": "../models/BM1684X_504",
                "postprocess_weights": "../models/post_process_weights.npz",
                "bpe_path": "../models/bpe_simple_vocab_16e6.txt.gz",
                "prompts": [],
            },
        },
    }


def test_parse_valid_yolo_only_and_resolve_relative_paths(tmp_path: Path):
    config = parse_config(valid_data(), tmp_path / "config")

    assert config.yolo26.enabled is True
    assert config.sam3.enabled is False
    assert config.yolo26.interval_sec == 3
    assert config.runtime.output_root == (tmp_path / "output").resolve()
    assert config.yolo26.class_names == (tmp_path / "config/fire_smoke.names").resolve()


def test_both_models_can_be_enabled(tmp_path: Path):
    data = valid_data()
    data["models"]["sam3"]["enabled"] = True
    data["models"]["sam3"]["prompts"] = [
        {"label": "fire", "text": "fire"},
        {"label": "smoke", "text": "smoke"},
    ]

    config = parse_config(data, tmp_path)

    assert config.yolo26.enabled and config.sam3.enabled
    assert [prompt.label for prompt in config.sam3.prompts] == ["fire", "smoke"]


def test_reject_both_models_disabled(tmp_path: Path):
    data = valid_data()
    data["models"]["yolo26"]["enabled"] = False

    with pytest.raises(ConfigError, match="至少启用一个"):
        parse_config(data, tmp_path)


def test_reject_more_than_five_cameras(tmp_path: Path):
    data = valid_data()
    data["cameras"] = [
        {"id": "cam_{}".format(index), "url": "rtsp://example.invalid/{}".format(index)}
        for index in range(6)
    ]

    with pytest.raises(ConfigError, match="1 到 5"):
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


def test_runtime_path_check_does_not_import_hardware(tmp_path: Path):
    data = valid_data()
    config = parse_config(data, tmp_path / "config")
    official_source = (
        config.runtime.sophon_demo_root
        / "sample/YOLO26/python/yolo26_bmcv.py"
    )
    official_source.parent.mkdir(parents=True)
    official_source.touch()
    config.yolo26.bmodel.parent.mkdir(parents=True)
    config.yolo26.bmodel.touch()
    config.yolo26.class_names.parent.mkdir(parents=True, exist_ok=True)
    config.yolo26.class_names.write_text("fire\nsmoke\n", encoding="utf-8")

    validate_runtime_paths(config)
