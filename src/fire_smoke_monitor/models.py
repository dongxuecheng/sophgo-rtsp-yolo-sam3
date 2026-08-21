from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import Sam3Config, SamPrompt, Yolo26Config
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


class _BoxOnlyMaskDecoder:
    """Drop-in official SAM3 mask decoder replacement for box-only mode."""

    def _ensure_weights(self) -> None:
        return None

    def predict_masks(self, *args: Any, **kwargs: Any) -> Dict[str, np.ndarray]:
        return {}


class Sam3Detector:
    """Box-only adapter for sophon-demo's nine-BModel SAM3 pipeline."""

    name = "sam3"

    def __init__(self, config: Sam3Config, demo_root: Path, dev_id: int) -> None:
        source = demo_root / "sample" / "SAM3" / "python" / "sam3_infer.py"
        module = load_official_module("fire_smoke_vendor_sam3", source)

        # The official grounding class resolves NumpyMaskDecoder at construction
        # time. Replacing it here avoids loading mask-head weights and avoids the
        # expensive CPU NumPy mask path entirely.
        module.NumpyMaskDecoder = _BoxOnlyMaskDecoder
        checkpoint_hint = config.postprocess_weights.parent / "sam3.pt"
        self._pipeline = module.SAM3Pipeline(
            str(config.model_dir),
            precision=config.precision,
            dev_id=dev_id,
            resolution=config.resolution,
            bpe_path=str(config.bpe_path),
            ckpt_path=str(checkpoint_hint),
            mode="bmodel",
        )
        self._prompts: Tuple[SamPrompt, ...] = tuple(config.prompts)
        self._threshold = config.score_threshold
        self._nms_threshold = config.nms_threshold
        self._max_detections_per_prompt = config.max_detections_per_prompt
        prompt_texts = [prompt.text for prompt in self._prompts]

        # Text prompts are static for this service, so encode them once on TPU.
        self._text_features, self._text_mask, self._inputs_embeds = (
            self._pipeline.text_encoder(prompt_texts)
        )
        logger.info(
            "SAM3 已加载 resolution=%d precision=%s prompts=%s "
            "score_threshold=%.3f nms_threshold=%.3f max_per_prompt=%d "
            "mask=disabled",
            config.resolution,
            config.precision,
            ",".join(prompt.label for prompt in self._prompts),
            self._threshold,
            self._nms_threshold,
            self._max_detections_per_prompt,
        )

    def detect(self, image: Any) -> List[Detection]:
        # SAIL performs the device-side format conversion and returns BGR packed
        # uint8. SAM3's official preprocessor then consumes the NumPy view/copy.
        bgr = np.ascontiguousarray(image.asmat())
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise RuntimeError("BMImage.asmat() 未返回 HWC BGR 图像")
        height, width = bgr.shape[:2]
        self._pipeline.set_image(bgr)

        detections: List[Detection] = []
        for index, prompt in enumerate(self._prompts):
            text_features = self._text_features[:, index : index + 1, :]
            text_mask = self._text_mask[index : index + 1, :]
            inputs_embeds = None
            if self._inputs_embeds is not None:
                inputs_embeds = self._inputs_embeds[:, index : index + 1, :]
            result = self._pipeline.grounding(
                self._pipeline.fpn_feats,
                self._pipeline.fpn_pos,
                text_features,
                text_mask,
                inputs_embeds,
                [prompt.text],
                None,
                None,
            )
            if result is None:
                raise RuntimeError("SAM3 TPU grounding 未返回结果；项目禁止 CPU 回退")
            detections.extend(
                self._detections_from_result(
                    result, prompt.label, width=width, height=height
                )
            )
        return detections

    def _detections_from_result(
        self, result: Dict[str, np.ndarray], label: str, width: int, height: int
    ) -> List[Detection]:
        score_output = np.asarray(result["pred_scores"])
        box_output = np.asarray(result["pred_boxes"])
        if (
            score_output.ndim != 3
            or score_output.shape[0] < 1
            or score_output.shape[2] < 1
        ):
            raise RuntimeError(
                "SAM3 pred_scores 形状无效: {}".format(score_output.shape)
            )
        if box_output.ndim != 3 or box_output.shape[0] < 1 or box_output.shape[2] < 4:
            raise RuntimeError("SAM3 pred_boxes 形状无效: {}".format(box_output.shape))

        scores = score_output[0, :, 0]
        boxes = box_output[0, :, :4]
        if scores.shape[0] != boxes.shape[0]:
            raise RuntimeError(
                "SAM3 分数与框数量不一致: scores={} boxes={}".format(
                    scores.shape[0], boxes.shape[0]
                )
            )

        candidate_indices = np.flatnonzero(
            np.isfinite(scores) & (scores >= self._threshold)
        )
        candidates: List[Detection] = []
        for index in candidate_indices:
            score = float(scores[index])
            cx, cy, box_w, box_h = [float(value) for value in boxes[index]]
            if not np.all(np.isfinite([cx, cy, box_w, box_h])):
                continue
            x1 = max(0, min(width - 1, int(round((cx - box_w / 2.0) * width))))
            y1 = max(0, min(height - 1, int(round((cy - box_h / 2.0) * height))))
            x2 = max(0, min(width - 1, int(round((cx + box_w / 2.0) * width))))
            y2 = max(0, min(height - 1, int(round((cy + box_h / 2.0) * height))))
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append(
                Detection(
                    label=label,
                    score=score,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        kept = self._nms(candidates)
        logger.debug(
            "SAM3 框后处理 label=%s raw=%d above_threshold=%d valid=%d kept=%d",
            label,
            scores.size,
            candidate_indices.size,
            len(candidates),
            len(kept),
        )
        return kept

    def _nms(self, detections: Sequence[Detection]) -> List[Detection]:
        ordered = sorted(detections, key=lambda item: item.score, reverse=True)
        kept: List[Detection] = []
        for candidate in ordered:
            if any(
                self._box_iou(candidate, selected) > self._nms_threshold
                for selected in kept
            ):
                continue
            kept.append(candidate)
            if len(kept) >= self._max_detections_per_prompt:
                break
        return kept

    @staticmethod
    def _box_iou(left: Detection, right: Detection) -> float:
        intersection_width = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
        intersection_height = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
        intersection = intersection_width * intersection_height
        if intersection == 0:
            return 0.0
        left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
        right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
        union = left_area + right_area - intersection
        return intersection / union if union > 0 else 0.0

    def close(self) -> None:
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None:
            pipeline.close()
            self._pipeline = None
