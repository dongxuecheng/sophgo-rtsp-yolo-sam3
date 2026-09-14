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
# Software configuration limit; actual throughput depends on the stream profiles.
MAX_CAMERAS = 64


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
    log_dir: Path
    log_retention_days: int
    log_max_file_bytes: int
    log_max_total_bytes: int
    min_free_disk_bytes: int
    disk_check_interval_sec: int


@dataclass(frozen=True)
class HealthConfig:
    max_consecutive_processing_failures: int
    scheduler_stall_sec: float
    status_log_interval_sec: float
    inference_stale_sec: float = 60.0
    upload_stale_sec: float = 300.0


@dataclass(frozen=True)
class UploadBackoffConfig:
    after_failures: int = 2
    initial_interval_sec: float = 2.5
    max_interval_sec: float = 60.0
    recover_after_successes: int = 3


@dataclass(frozen=True)
class PeriodicConfig:
    upload_interval_sec: float = 2.5
    backoff_after_failures: int = 2
    backoff_max_interval_sec: float = 60.0
    recover_after_successes: int = 3


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
    env_file: Optional[Path] = None
    backoff: UploadBackoffConfig = field(default_factory=UploadBackoffConfig)

    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        value = os.environ.get(self.api_key_env, "")
        if not value and self.env_file is not None and self.env_file.is_file():
            value = _read_env_value(self.env_file, self.api_key_env)
        if not value:
            raise ConfigError("请设置 {} 环境变量或配置目录中的 pipeline.env".format(self.api_key_env))
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
    periodic: PeriodicConfig
    health: HealthConfig

    def interval_for(self, camera: CameraConfig) -> float:
        if self.yolo26 is None:
            raise ValueError("periodic 使用全机轮询间隔，不使用每路定时器")
        if camera.interval_sec is not None:
            return camera.interval_sec
        return self.yolo26.interval_sec


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


def _read_env_value(path: Path, name: str) -> str:
    """Read one literal KEY=value; never execute shell code or expand variables."""
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key.strip() != name:
            continue
        if found is not None:
            raise ConfigError("pipeline.env 中密钥变量重复")
        value = value.strip()
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ConfigError("pipeline.env 中密钥引号未闭合")
            value = value[1:-1]
        found = value
    return found or ""


def _pipeline_config(raw: Mapping[str, Any], base_dir: Path) -> PipelineConfig:
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
    backoff_raw = _mapping(raw.get("backoff", {}), "pipeline.backoff")
    backoff = UploadBackoffConfig(
        after_failures=_integer(backoff_raw.get("after_failures", 2),
                                "pipeline.backoff.after_failures", 1, 100),
        initial_interval_sec=_positive(backoff_raw.get("initial_interval_sec", 2.5),
                                       "pipeline.backoff.initial_interval_sec"),
        max_interval_sec=_positive(backoff_raw.get("max_interval_sec", 60),
                                   "pipeline.backoff.max_interval_sec"),
        recover_after_successes=_integer(backoff_raw.get("recover_after_successes", 3),
                                         "pipeline.backoff.recover_after_successes", 1, 100),
    )
    if backoff.max_interval_sec < backoff.initial_interval_sec:
        raise ConfigError("pipeline.backoff.max_interval_sec 不能小于 initial_interval_sec")
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
        env_file=base_dir / "pipeline.env",
        backoff=backoff,
    )


def parse_config(data: Mapping[str, Any], base_dir: Path) -> AppConfig:
    root = _mapping(data, "根配置")
    mode = str(_required(root, "mode", "root"))
    if mode not in {"yolo_trigger", "periodic"}:
        raise ConfigError("mode 仅支持 yolo_trigger 或 periodic")
    runtime_raw = _mapping(root.get("runtime", {}), "runtime")
    log_max_file_mb = _integer(
        runtime_raw.get("log_max_file_mb", 100), "runtime.log_max_file_mb", 1, 1024
    )
    log_max_total_mb = _integer(
        runtime_raw.get("log_max_total_mb", 1024), "runtime.log_max_total_mb", 1, 16384
    )
    if log_max_total_mb < log_max_file_mb:
        raise ConfigError("runtime.log_max_total_mb 不能小于 log_max_file_mb")
    min_free_disk_gb = _positive(
        runtime_raw.get("min_free_disk_gb", 5), "runtime.min_free_disk_gb"
    )
    decoder_queue_size = _integer(
        runtime_raw.get("decoder_queue_size", 1),
        "runtime.decoder_queue_size", 1, 8,
    )
    if decoder_queue_size != 1:
        logger.warning("只缓存每路最新一帧，decoder_queue_size 已设为 1 mode=%s", mode)
        decoder_queue_size = 1
    runtime = RuntimeConfig(
        dev_id=_integer(runtime_raw.get("dev_id", 0), "runtime.dev_id", 0, 255),
        decoder_queue_size=decoder_queue_size,
        read_timeout_sec=_integer(
            runtime_raw.get("read_timeout_sec", 2), "runtime.read_timeout_sec", 1, 3600
        ),
        reconnect_cooldown_sec=_positive(
            runtime_raw.get("reconnect_cooldown_sec", 10), "runtime.reconnect_cooldown_sec"
        ),
        log_level=str(runtime_raw.get("log_level", "INFO")).upper(),
        log_dir=_path(runtime_raw.get("log_dir", "../logs"), base_dir),
        log_retention_days=_integer(
            runtime_raw.get("log_retention_days", 30), "runtime.log_retention_days", 1, 30
        ),
        log_max_file_bytes=log_max_file_mb * 1024 * 1024,
        log_max_total_bytes=log_max_total_mb * 1024 * 1024,
        min_free_disk_bytes=int(min_free_disk_gb * 1024 * 1024 * 1024),
        disk_check_interval_sec=_integer(
            runtime_raw.get("disk_check_interval_sec", 60),
            "runtime.disk_check_interval_sec", 10, 3600,
        ),
    )
    if runtime.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ConfigError("runtime.log_level 仅支持 DEBUG/INFO/WARNING/ERROR")
    if runtime.log_dir.parent == runtime.log_dir:
        raise ConfigError("runtime.log_dir 不能是文件系统根目录")
    pipeline = _pipeline_config(_mapping(_required(root, "pipeline", "root"), "pipeline"), base_dir)
    if mode == "periodic" and pipeline.workers > 2:
        raise ConfigError("periodic 模式 pipeline.workers 只能为 1 或 2")
    health_raw = _mapping(root.get("health", {}), "health")
    health = HealthConfig(
        max_consecutive_processing_failures=_integer(
            health_raw.get("max_consecutive_processing_failures", 10),
            "health.max_consecutive_processing_failures", 1, 1000,
        ),
        scheduler_stall_sec=_positive(
            health_raw.get("scheduler_stall_sec", 30), "health.scheduler_stall_sec"
        ),
        status_log_interval_sec=_positive(
            health_raw.get("status_log_interval_sec", 300),
            "health.status_log_interval_sec",
        ),
        inference_stale_sec=_positive(health_raw.get("inference_stale_sec", 60),
                                      "health.inference_stale_sec"),
        upload_stale_sec=_positive(health_raw.get("upload_stale_sec", 300),
                                   "health.upload_stale_sec"),
    )

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
            if mode == "periodic":
                raise ConfigError("periodic 不再支持每路 interval_sec，请删除并设置 periodic.upload_interval_sec")
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
    if "output_root" in runtime_raw:
        logger.warning("output_root 已停用：两种模式均不保存本地图片")
    periodic = PeriodicConfig()
    if mode == "periodic":
        raw = _mapping(root.get("periodic", {}), "periodic")
        if "interval_sec" in raw:
            raise ConfigError("periodic.interval_sec 已停用，请删除并设置全机 periodic.upload_interval_sec（例如 2.5）")
        periodic = PeriodicConfig(
            upload_interval_sec=_positive(
                raw.get("upload_interval_sec", 2.5), "periodic.upload_interval_sec"
            ),
            backoff_after_failures=_integer(
                raw.get("backoff_after_failures", 2), "periodic.backoff_after_failures", 1, 100,
            ),
            backoff_max_interval_sec=_positive(
                raw.get("backoff_max_interval_sec", 60), "periodic.backoff_max_interval_sec",
            ),
            recover_after_successes=_integer(
                raw.get("recover_after_successes", 3), "periodic.recover_after_successes", 1, 100,
            ),
        )
        if periodic.backoff_max_interval_sec < periodic.upload_interval_sec:
            raise ConfigError("periodic.backoff_max_interval_sec 不能小于 upload_interval_sec")
    return AppConfig(mode, runtime, tuple(cameras), pipeline, yolo, periodic, health)


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
