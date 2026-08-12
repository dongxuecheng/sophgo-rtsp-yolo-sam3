from __future__ import annotations

import logging
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .domain import Detection


logger = logging.getLogger(__name__)
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


class DetectionWriter:
    """Draw and JPEG-encode positive frames with BMCV."""

    def __init__(
        self, sail_module: Any, dev_id: int, output_root: Path, min_free_disk_gb: float
    ) -> None:
        self._root = output_root
        self._root.mkdir(parents=True, exist_ok=True)
        self._min_free_bytes = int(min_free_disk_gb * 1024 * 1024 * 1024)
        self._last_saved: Dict[Tuple[str, str, Tuple[str, ...]], float] = {}
        self._disk_warning_active = False
        self._bmcv = sail_module.Bmcv(sail_module.Handle(dev_id))

    def save(
        self,
        model_name: str,
        camera_id: str,
        image: Any,
        detections: Sequence[Detection],
        cooldown_sec: float,
    ) -> Any:
        if not detections:
            return None
        label_signature = tuple(sorted({item.label for item in detections}))
        key = (model_name, camera_id, label_signature)
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_saved.get(key, float("-inf")) < cooldown_sec:
            return None
        if not self._has_disk_space():
            return None

        now = datetime.now()
        day_dir = self._root / model_name / camera_id / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        labels = "-".join(sorted({self._safe(item.label) for item in detections}))
        filename = "{}_{}_{}.jpg".format(
            now.strftime("%Y%m%d_%H%M%S_%f")[:-3], labels, uuid.uuid4().hex[:6]
        )
        destination = day_dir / filename
        temporary = day_dir / ("." + destination.stem + ".part.jpg")

        drawable = self._bmcv.convert_format(image)
        thickness = max(2, min(int(image.width()), int(image.height())) // 400)
        for detection in detections:
            box_width = detection.x2 - detection.x1
            box_height = detection.y2 - detection.y1
            if box_width <= thickness * 2 or box_height <= thickness * 2:
                continue
            self._bmcv.rectangle(
                drawable,
                detection.x1,
                detection.y1,
                box_width,
                box_height,
                self._color(detection.label),
                thickness,
            )
        try:
            self._bmcv.imwrite(str(temporary), drawable)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise IOError("BMCV 未生成有效 JPEG")
            os.replace(str(temporary), str(destination))
        finally:
            if temporary.exists():
                temporary.unlink()
        self._last_saved[key] = now_monotonic
        return destination

    def _has_disk_space(self) -> bool:
        free = shutil.disk_usage(str(self._root)).free
        if free < self._min_free_bytes:
            if not self._disk_warning_active:
                logger.error(
                    "磁盘剩余空间低于保护阈值，暂停保存图片 free_gb=%.2f",
                    free / 1024 / 1024 / 1024,
                )
                self._disk_warning_active = True
            return False
        self._disk_warning_active = False
        return True

    @staticmethod
    def _safe(value: str) -> str:
        return _SAFE_COMPONENT.sub("_", value).strip("._") or "target"

    @staticmethod
    def _color(label: str) -> Sequence[int]:
        normalized = label.lower()
        if "fire" in normalized or "flame" in normalized:
            return [0, 0, 255]
        if "smoke" in normalized:
            return [0, 255, 255]
        return [0, 255, 0]
