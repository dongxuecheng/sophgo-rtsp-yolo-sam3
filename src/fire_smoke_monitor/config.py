from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml


class ConfigError(ValueError):
    pass


_CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


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
class SamPrompt:
    label: str
    text: str


@dataclass(frozen=True)
class Sam3Config:
    enabled: bool
    model_dir: Path
    postprocess_weights: Path
    bpe_path: Path
    prompts: Sequence[SamPrompt]
    precision: str
    resolution: int
    score_threshold: float
    interval_sec: float
    save_cooldown_sec: float


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig
    cameras: Sequence[CameraConfig]
    yolo26: Yolo26Config
    sam3: Sam3Config


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
    sam_raw = _mapping(_required(models_raw, "sam3", "models"), "models.sam3")

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
    if not 1 <= len(camera_items) <= 5:
        raise ConfigError("cameras 数量必须在 1 到 5 路之间")
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
    target_classes = [str(v).strip() for v in yolo_raw.get("target_classes", [])]
    if yolo_enabled and not target_classes:
        raise ConfigError("启用 YOLO26 时 target_classes 不能为空")
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

    sam_enabled = _boolean(sam_raw.get("enabled", False), "models.sam3.enabled")
    prompt_items = sam_raw.get("prompts", [])
    prompts: List[SamPrompt] = []
    if sam_enabled:
        for index, item in enumerate(_sequence(prompt_items, "models.sam3.prompts")):
            raw = _mapping(item, "models.sam3.prompts[{}]".format(index))
            label = str(_required(raw, "label", "models.sam3.prompts")).strip()
            prompt_text = str(_required(raw, "text", "models.sam3.prompts")).strip()
            if not label or not prompt_text:
                raise ConfigError("SAM3 prompt 的 label/text 不能为空")
            prompts.append(SamPrompt(label=label, text=prompt_text))
        if not prompts:
            raise ConfigError("启用 SAM3 时 prompts 不能为空")

    sam = Sam3Config(
        enabled=sam_enabled,
        model_dir=_path(sam_raw.get("model_dir", "missing"), base_dir),
        postprocess_weights=_path(
            sam_raw.get("postprocess_weights", "post_process_weights.npz"), base_dir
        ),
        bpe_path=_path(sam_raw.get("bpe_path", "bpe_simple_vocab_16e6.txt.gz"), base_dir),
        prompts=tuple(prompts),
        precision=str(sam_raw.get("precision", "f16")).lower(),
        resolution=int(sam_raw.get("resolution", 504)),
        score_threshold=_probability(
            sam_raw.get("score_threshold", 0.3), "models.sam3.score_threshold"
        ),
        interval_sec=_positive(
            sam_raw.get("interval_sec", 20), "models.sam3.interval_sec"
        ),
        save_cooldown_sec=_non_negative(
            sam_raw.get("save_cooldown_sec", 60), "models.sam3.save_cooldown_sec"
        ),
    )
    if sam.enabled and sam.precision != "f16":
        raise ConfigError("BM1684X SoC 的 SAM3 当前仅建议/支持 precision=f16")
    if sam.enabled and sam.resolution != 504:
        raise ConfigError("16GB BM1684X SoC 的完整 SAM3 检测链路必须使用 resolution=504")
    if sam.enabled and sam.postprocess_weights.name != "post_process_weights.npz":
        raise ConfigError("SAM3 后处理权重文件名必须为 post_process_weights.npz")
    if not yolo.enabled and not sam.enabled:
        raise ConfigError("YOLO26 和 SAM3 至少启用一个")

    return AppConfig(runtime=runtime, cameras=tuple(cameras), yolo26=yolo, sam3=sam)


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
    required_demo_files = []
    if config.yolo26.enabled:
        required_demo_files.append(demo / "sample" / "YOLO26" / "python" / "yolo26_bmcv.py")
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
    if config.sam3.enabled:
        required_demo_files.append(demo / "sample" / "SAM3" / "python" / "sam3_infer.py")
        if not config.sam3.model_dir.is_dir():
            raise ConfigError("SAM3 BModel 目录不存在: {}".format(config.sam3.model_dir))
        expected_bmodels = [
            *("vit/sam3_vit_part{}_f16_1b.bmodel".format(index) for index in range(5)),
            "neck/sam3_neck_f16_1b.bmodel",
            "grounding/sam3_grounding_encoder_f16_1b.bmodel",
            "grounding/sam3_grounding_decoder_f16_1b.bmodel",
            "grounding/sam3_text_encoder_f16_1b.bmodel",
        ]
        missing_bmodels = [
            relative
            for relative in expected_bmodels
            if not (config.sam3.model_dir / relative).is_file()
        ]
        if missing_bmodels:
            raise ConfigError(
                "SAM3 BModel 不完整，缺少: {}".format(", ".join(missing_bmodels))
            )
        if not config.sam3.postprocess_weights.is_file():
            raise ConfigError(
                "SAM3 后处理权重不存在: {}".format(config.sam3.postprocess_weights)
            )
        if not config.sam3.bpe_path.is_file():
            raise ConfigError("SAM3 BPE 文件不存在: {}".format(config.sam3.bpe_path))
    for file_path in required_demo_files:
        if not file_path.is_file():
            raise ConfigError("sophon-demo 官方文件不存在: {}".format(file_path))
