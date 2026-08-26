from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Sequence

import numpy as np

from .config import Yolo26Config
from .domain import Detection
from .official_loader import load_official_module


logger = logging.getLogger(__name__)


class Yolo26Detector:
    """Adapter for sophon-demo's official BMCV YOLO26 implementation."""

    name = "yolo26"

    def __init__(self, config: Yolo26Config, demo_root: Path, dev_id: int) -> None:
        source = demo_root / "sample" / "YOLO26" / "python" / "yolo26_bmcv.py"
        module = load_official_module("fire_smoke_vendor_yolo26", source)
        args = SimpleNamespace(
            bmodel=str(config.bmodel),
            dev_id=dev_id,
            conf_thresh=config.conf_threshold,
        )
        self._model = module.YOLO26(args)
        self._class_names = self._read_class_names(config.class_names)
        name_to_id = {name: index for index, name in enumerate(self._class_names)}
        missing = [name for name in config.target_classes if name not in name_to_id]
        if missing:
            raise ValueError("YOLO26 类别文件中找不到目标: {}".format(", ".join(missing)))
        self._target_ids = {name_to_id[name] for name in config.target_classes}
        logger.info(
            "YOLO26 已加载 bmodel=%s targets=%s",
            config.bmodel.name,
            ",".join(config.target_classes),
        )

    @staticmethod
    def _read_class_names(path: Path) -> Sequence[str]:
        with path.open("r", encoding="utf-8") as stream:
            names = [line.strip() for line in stream if line.strip()]
        if not names:
            raise ValueError("YOLO26 类别文件为空: {}".format(path))
        if len(names) != len(set(names)):
            raise ValueError("YOLO26 类别文件包含重复名称: {}".format(path))
        return tuple(names)

    def detect(self, image: Any) -> List[Detection]:
        raw_results = self._model([image])
        if not raw_results:
            return []
        height, width = int(image.height()), int(image.width())
        detections: List[Detection] = []
        for row in raw_results[0]:
            if len(row) < 6 or not np.all(np.isfinite(row[:6])):
                continue
            class_id = int(row[5])
            if class_id not in self._target_ids or not 0 <= class_id < len(self._class_names):
                continue
            x1 = max(0, min(width - 1, int(round(float(row[0])))))
            y1 = max(0, min(height - 1, int(round(float(row[1])))))
            x2 = max(0, min(width - 1, int(round(float(row[2])))))
            y2 = max(0, min(height - 1, int(round(float(row[3])))))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    label=self._class_names[class_id],
                    score=float(row[4]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections

    def close(self) -> None:
        self._model = None
