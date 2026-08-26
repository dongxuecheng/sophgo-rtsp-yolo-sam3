from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, List, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    pass


_CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_CAMERAS = 16


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    url: str


@dataclass(frozen=True)
class RuntimeConfig:
    dev_id: int
    sophon_demo_root: Path
    output_root: Path
    decoder_queue_size: int
    read_timeout_sec: int
    reconnect_cooldown_sec: float
    min_free_disk_gb: float
    log_level: str


@dataclass(frozen=True)
class Yolo26Config:
    enabled: bool
    bmodel: Path
    class_names: Path
    target_classes: Sequence[str]
    conf_threshold: float
    interval_sec: float
    save_cooldown_sec: float


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig
    cameras: Sequence[CameraConfig]
    yolo26: Yolo26Config


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError("{} 必须是对象".format(name))
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ConfigError("{} 必须是列表".format(name))
    return value


def _required(mapping: Mapping[str, Any], key: str, prefix: str) -> Any:
    if key not in mapping:
        raise ConfigError("缺少配置项 {}.{}".format(prefix, key))
    return mapping[key]


def _path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError("{} 必须是 true 或 false".format(name))
    return value


def _probability(value: Any, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ConfigError("{} 必须在 0 到 1 之间".format(name))
    return number


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if number <= 0:
        raise ConfigError("{} 必须大于 0".format(name))
    return number


def _non_negative(value: Any, name: str) -> float:
    number = float(value)
    if number < 0:
        raise ConfigError("{} 不能小于 0".format(name))
    return number


def parse_config(data: Mapping[str, Any], base_dir: Path) -> AppConfig:
    root = _mapping(data, "根配置")
    runtime_raw = _mapping(_required(root, "runtime", "root"), "runtime")
    models_raw = _mapping(_required(root, "models", "root"), "models")
    yolo_raw = _mapping(_required(models_raw, "yolo26", "models"), "models.yolo26")

    runtime = RuntimeConfig(
        dev_id=int(runtime_raw.get("dev_id", 0)),
        sophon_demo_root=_path(
            _required(runtime_raw, "sophon_demo_root", "runtime"), base_dir
        ),
        output_root=_path(runtime_raw.get("output_root", "../output"), base_dir),
        decoder_queue_size=int(runtime_raw.get("decoder_queue_size", 2)),
        read_timeout_sec=int(runtime_raw.get("read_timeout_sec", 2)),
        reconnect_cooldown_sec=_positive(
            runtime_raw.get("reconnect_cooldown_sec", 10),
            "runtime.reconnect_cooldown_sec",
        ),
        min_free_disk_gb=_non_negative(
            runtime_raw.get("min_free_disk_gb", 2), "runtime.min_free_disk_gb"
        ),
        log_level=str(runtime_raw.get("log_level", "INFO")).upper(),
    )
    if runtime.dev_id < 0:
        raise ConfigError("runtime.dev_id 不能小于 0")
    if not 1 <= runtime.decoder_queue_size <= 8:
        raise ConfigError("runtime.decoder_queue_size 必须在 1 到 8 之间")
    if runtime.read_timeout_sec <= 0:
        raise ConfigError("runtime.read_timeout_sec 必须大于 0")
    if runtime.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("runtime.log_level 仅支持 DEBUG/INFO/WARNING/ERROR")

    camera_items = _sequence(_required(root, "cameras", "root"), "cameras")
    if not 1 <= len(camera_items) <= MAX_CAMERAS:
        raise ConfigError(
            "cameras 数量必须在 1 到 {} 路之间".format(MAX_CAMERAS)
        )
    cameras: List[CameraConfig] = []
    seen_ids = set()
    for index, item in enumerate(camera_items):
        raw = _mapping(item, "cameras[{}]".format(index))
        camera_id = str(_required(raw, "id", "cameras[{}]".format(index))).strip()
        url = str(_required(raw, "url", "cameras[{}]".format(index))).strip()
        if not _CAMERA_ID.fullmatch(camera_id) or camera_id in {".", ".."}:
            raise ConfigError(
                "camera id 必须为 1-64 位字母、数字、下划线、点或横线，且以字母/数字开头"
            )
        if camera_id in seen_ids:
            raise ConfigError("camera id 重复: {}".format(camera_id))
        if not url.lower().startswith(("rtsp://", "rtsps://")):
            raise ConfigError("摄像头 {} 的 url 必须是 RTSP 地址".format(camera_id))
        seen_ids.add(camera_id)
        cameras.append(CameraConfig(camera_id=camera_id, url=url))

    yolo_enabled = _boolean(yolo_raw.get("enabled", False), "models.yolo26.enabled")
    if not yolo_enabled:
        raise ConfigError("本项目仅保留 YOLO26，models.yolo26.enabled 必须为 true")
    target_classes = [str(v).strip() for v in yolo_raw.get("target_classes", [])]
    if not target_classes:
        raise ConfigError("YOLO26 target_classes 不能为空")
    yolo = Yolo26Config(
        enabled=yolo_enabled,
        bmodel=_path(yolo_raw.get("bmodel", "missing.bmodel"), base_dir),
        class_names=_path(yolo_raw.get("class_names", "fire_smoke.names"), base_dir),
        target_classes=tuple(target_classes),
        conf_threshold=_probability(
            yolo_raw.get("conf_threshold", 0.25), "models.yolo26.conf_threshold"
        ),
        interval_sec=_positive(
            yolo_raw.get("interval_sec", 3), "models.yolo26.interval_sec"
        ),
        save_cooldown_sec=_non_negative(
            yolo_raw.get("save_cooldown_sec", 30),
            "models.yolo26.save_cooldown_sec",
        ),
    )

    return AppConfig(runtime=runtime, cameras=tuple(cameras), yolo26=yolo)


def load_config(path: Path) -> AppConfig:
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError("配置文件不存在: {}".format(config_path))
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        raise ConfigError("配置文件为空")
    return parse_config(_mapping(data, "根配置"), config_path.parent)


def validate_runtime_paths(config: AppConfig) -> None:
    demo = config.runtime.sophon_demo_root
    official_source = demo / "sample" / "YOLO26" / "python" / "yolo26_bmcv.py"
    if not official_source.is_file():
        raise ConfigError("sophon-demo 官方文件不存在: {}".format(official_source))
    if not config.yolo26.bmodel.is_file():
        raise ConfigError("YOLO26 BModel 不存在: {}".format(config.yolo26.bmodel))
    if not config.yolo26.class_names.is_file():
        raise ConfigError("YOLO26 类别文件不存在: {}".format(config.yolo26.class_names))
    with config.yolo26.class_names.open("r", encoding="utf-8") as stream:
        class_names = [line.strip() for line in stream if line.strip()]
    if not class_names or len(class_names) != len(set(class_names)):
        raise ConfigError("YOLO26 类别文件为空或包含重复名称")
    missing_targets = [
        name for name in config.yolo26.target_classes if name not in class_names
    ]
    if missing_targets:
        raise ConfigError(
            "YOLO26 类别文件缺少目标: {}".format(", ".join(missing_targets))
        )
