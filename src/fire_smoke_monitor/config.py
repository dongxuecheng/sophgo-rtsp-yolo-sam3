from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

import yaml


class ConfigError(ValueError):
    pass


logger = logging.getLogger(__name__)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_CAMERAS = 16


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    url: str = field(repr=False)
    name: str = ""
    interval_sec: Optional[float] = None


@dataclass(frozen=True)
class RuntimeConfig:
    dev_id: int
    decoder_queue_size: int
    read_timeout_sec: int
    reconnect_cooldown_sec: float
    log_level: str


@dataclass(frozen=True)
class Yolo26Config:
    bmodel: Path
    class_names: Path
    target_classes: Sequence[str]
    conf_threshold: float
    interval_sec: float


@dataclass(frozen=True)
class PipelineConfig:
    endpoint: str
    machine_id: str
    api_key_env: str
    workers: int
    timeout_sec: float
    max_pending_age_sec: float
    max_image_bytes: int
    max_image_pixels: int

    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        value = os.environ.get(self.api_key_env, "")
        if not value:
            raise ConfigError("请设置上传密钥环境变量 {}".format(self.api_key_env))
        if not all(32 < ord(char) < 127 for char in value):
            raise ConfigError("上传密钥必须是无空白的可打印 ASCII 字符")
        return value


@dataclass(frozen=True)
class AppConfig:
    mode: str
    runtime: RuntimeConfig
    cameras: Sequence[CameraConfig]
    pipeline: PipelineConfig
    yolo26: Optional[Yolo26Config]
    periodic_interval_sec: float

    def interval_for(self, camera: CameraConfig) -> float:
        if camera.interval_sec is not None:
            return camera.interval_sec
        if self.yolo26 is not None:
            return self.yolo26.interval_sec
        return self.periodic_interval_sec


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
    return (base_dir / path if not path.is_absolute() else path).resolve()


def _number(value: Any, name: str) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError()
        number = float(value)
        if not math.isfinite(number):
            raise ValueError()
    except (TypeError, ValueError, OverflowError):
        raise ConfigError("{} 必须是有限数值".format(name)) from None
    return number


def _positive(value: Any, name: str) -> float:
    number = _number(value, name)
    if number <= 0:
        raise ConfigError("{} 必须大于 0".format(name))
    return number


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    number = _number(value, name)
    if not number.is_integer() or not minimum <= number <= maximum:
        raise ConfigError("{} 必须是 {} 到 {} 之间的整数".format(name, minimum, maximum))
    return int(number)


def _identifier(value: Any, name: str) -> str:
    result = str(value).strip()
    if not _IDENTIFIER.fullmatch(result):
        raise ConfigError("{} 必须为 1-64 位字母、数字、下划线或横线（接口不接受点）".format(name))
    return result


def _pipeline_config(raw: Mapping[str, Any]) -> PipelineConfig:
    endpoint = str(_required(raw, "endpoint", "pipeline")).strip()
    try:
        url = urlsplit(endpoint)
        valid = (
            url.scheme in {"http", "https"} and url.hostname and
            not url.username and not url.password and not url.query and not url.fragment and
            url.path.endswith("/v1/frames") and
            not any(char.isspace() or ord(char) < 32 for char in endpoint) and
            (url.port is None or 1 <= url.port <= 65535)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ConfigError("pipeline.endpoint 必须是完整 HTTP(S) /v1/frames 地址，不带账号、查询参数或片段")
    api_key_env = str(raw.get("api_key_env", "")).strip()
    if api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        raise ConfigError("pipeline.api_key_env 必须是环境变量名称，不能直接填写密钥")
    return PipelineConfig(
        endpoint=endpoint,
        machine_id=_identifier(_required(raw, "machine_id", "pipeline"), "pipeline.machine_id"),
        api_key_env=api_key_env,
        workers=_integer(raw.get("workers", 2), "pipeline.workers", 1, 8),
        timeout_sec=_positive(raw.get("timeout_sec", 10), "pipeline.timeout_sec"),
        max_pending_age_sec=_positive(
            raw.get("max_pending_age_sec", 30), "pipeline.max_pending_age_sec"
        ),
        max_image_bytes=_integer(
            raw.get("max_image_bytes", 8388608), "pipeline.max_image_bytes", 1, 64 * 1024 * 1024
        ),
        max_image_pixels=_integer(
            raw.get("max_image_pixels", 16000000), "pipeline.max_image_pixels", 1, 64000000
        ),
    )


def parse_config(data: Mapping[str, Any], base_dir: Path) -> AppConfig:
    root = _mapping(data, "根配置")
    mode = str(_required(root, "mode", "root"))
    if mode not in {"yolo_trigger", "periodic"}:
        raise ConfigError("mode 仅支持 yolo_trigger 或 periodic")
    runtime_raw = _mapping(root.get("runtime", {}), "runtime")
    runtime = RuntimeConfig(
        dev_id=_integer(runtime_raw.get("dev_id", 0), "runtime.dev_id", 0, 255),
        decoder_queue_size=_integer(
            runtime_raw.get("decoder_queue_size", 2), "runtime.decoder_queue_size", 1, 8
        ),
        read_timeout_sec=_integer(
            runtime_raw.get("read_timeout_sec", 2), "runtime.read_timeout_sec", 1, 3600
        ),
        reconnect_cooldown_sec=_positive(
            runtime_raw.get("reconnect_cooldown_sec", 10), "runtime.reconnect_cooldown_sec"
        ),
        log_level=str(runtime_raw.get("log_level", "INFO")).upper(),
    )
    if runtime.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("runtime.log_level 仅支持 DEBUG/INFO/WARNING/ERROR")
    pipeline = _pipeline_config(_mapping(_required(root, "pipeline", "root"), "pipeline"))

    camera_items = _sequence(_required(root, "cameras", "root"), "cameras")
    if not 1 <= len(camera_items) <= MAX_CAMERAS:
        raise ConfigError("cameras 数量必须在 1 到 {} 路之间".format(MAX_CAMERAS))
    cameras: List[CameraConfig] = []
    seen_ids = set()
    for index, item in enumerate(camera_items):
        prefix = "cameras[{}]".format(index)
        raw = _mapping(item, prefix)
        camera_id = _identifier(_required(raw, "id", prefix), "camera id")
        url = str(_required(raw, "url", prefix)).strip()
        if camera_id in seen_ids:
            raise ConfigError("camera id 重复: {}".format(camera_id))
        if not url.lower().startswith(("rtsp://", "rtsps://")):
            raise ConfigError("摄像头 {} 的 url 必须是 RTSP 地址".format(camera_id))
        name = str(raw.get("name", ""))
        if len(name) > 256:
            raise ConfigError("{}.name 最长为 256 字符".format(prefix))
        interval = raw.get("interval_sec")
        if interval is not None:
            interval = _positive(interval, "{}.interval_sec".format(prefix))
        seen_ids.add(camera_id)
        cameras.append(CameraConfig(camera_id, url, name, interval))

    yolo = None
    if mode == "yolo_trigger":
        models = _mapping(_required(root, "models", "root"), "models")
        raw = _mapping(_required(models, "yolo26", "models"), "models.yolo26")
        if "enabled" in raw and raw["enabled"] is not True:
            raise ConfigError("yolo_trigger 模式 models.yolo26.enabled 必须为 true，或删除 enabled 由 mode 控制")
        targets = tuple(str(v).strip() for v in _sequence(
            _required(raw, "target_classes", "models.yolo26"), "models.yolo26.target_classes"
        ))
        if not targets or any(not v for v in targets) or len(set(targets)) != len(targets):
            raise ConfigError("YOLO26 target_classes 不能为空或包含重复/空类别")
        threshold = _number(raw.get("conf_threshold", 0.25), "models.yolo26.conf_threshold")
        if not 0 <= threshold <= 1:
            raise ConfigError("models.yolo26.conf_threshold 必须在 0 到 1 之间")
        yolo = Yolo26Config(
            bmodel=_path(_required(raw, "bmodel", "models.yolo26"), base_dir),
            class_names=_path(_required(raw, "class_names", "models.yolo26"), base_dir),
            target_classes=targets,
            conf_threshold=threshold,
            interval_sec=_positive(raw.get("interval_sec", 3), "models.yolo26.interval_sec"),
        )
        if "save_cooldown_sec" in raw:
            logger.warning("save_cooldown_sec 已停用：每次采样命中都提交上传，不保存本地图片")
    if "output_root" in runtime_raw or "min_free_disk_gb" in runtime_raw:
        logger.warning("output_root/min_free_disk_gb 已停用：两种模式均不保存本地图片")
    periodic = _mapping(root.get("periodic", {}), "periodic")
    periodic_interval = _positive(periodic.get("interval_sec", 10), "periodic.interval_sec")
    return AppConfig(mode, runtime, tuple(cameras), pipeline, yolo, periodic_interval)


def load_config(path: Path) -> AppConfig:
    config_path = path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return parse_config(data, config_path.parent)


def validate_runtime_paths(config: AppConfig) -> None:
    config.pipeline.api_key()
    if config.yolo26 is None:
        return
    if not config.yolo26.bmodel.is_file():
        raise ConfigError("YOLO26 BModel 不存在: {}".format(config.yolo26.bmodel))
    if not config.yolo26.class_names.is_file():
        raise ConfigError("YOLO26 类别文件不存在: {}".format(config.yolo26.class_names))
    with config.yolo26.class_names.open("r", encoding="utf-8") as stream:
        class_names = [line.strip() for line in stream if line.strip()]
    if not class_names or len(class_names) != len(set(class_names)):
        raise ConfigError("YOLO26 类别文件为空或包含重复名称")
    missing = [name for name in config.yolo26.target_classes if name not in class_names]
    if missing:
        raise ConfigError("YOLO26 类别文件缺少目标: {}".format(", ".join(missing)))
