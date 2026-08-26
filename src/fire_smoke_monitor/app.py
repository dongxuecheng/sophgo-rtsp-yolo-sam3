from __future__ import annotations

import logging
import threading
from typing import Any, Dict

from .config import AppConfig
from .domain import ModelRuntime
from .models import Yolo26Detector
from .scheduler import InferenceScheduler
from .streams import MultiRtspSource
from .writer import DetectionWriter


logger = logging.getLogger(__name__)


class MonitorApplication:
    def __init__(self, config: AppConfig) -> None:
        try:
            import sophon.sail as sail
        except ImportError as exc:
            raise RuntimeError(
                "无法导入 sophon.sail；请在 BM1684X SoC 上安装匹配 SDK 的 SAIL Python 包"
            ) from exc

        self._source = None
        self._detectors: Dict[str, Any] = {}
        self._scheduler = None
        self._closed = False
        try:
            handle = sail.Handle(config.runtime.dev_id)
            target = str(handle.get_target())
            if "1684X" not in target.upper():
                raise RuntimeError("当前设备不是 BM1684X: {}".format(target))
            logger.info("硬件设备已确认 dev_id=%d target=%s", config.runtime.dev_id, target)

            detector = Yolo26Detector(
                config.yolo26,
                config.runtime.sophon_demo_root,
                config.runtime.dev_id,
            )
            self._detectors["yolo26"] = detector

            self._source = MultiRtspSource(
                sail,
                config.cameras,
                config.runtime.dev_id,
                config.runtime.decoder_queue_size,
                config.runtime.read_timeout_sec,
                config.runtime.reconnect_cooldown_sec,
            )
            writer = DetectionWriter(
                sail,
                config.runtime.dev_id,
                config.runtime.output_root,
                config.runtime.min_free_disk_gb,
            )
            runtimes: Dict[str, ModelRuntime] = {
                "yolo26": ModelRuntime(
                    detector=self._detectors["yolo26"],
                    interval_sec=config.yolo26.interval_sec,
                    save_cooldown_sec=config.yolo26.save_cooldown_sec,
                )
            }
            self._scheduler = InferenceScheduler(
                [camera.camera_id for camera in config.cameras],
                runtimes,
                self._source,
                writer,
            )
        except Exception:
            self.close()
            raise

    def run(self, stop_event: threading.Event) -> None:
        if self._scheduler is None:
            raise RuntimeError("应用未成功初始化")
        self._scheduler.run(stop_event)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._source is not None:
            self._source.close()
            self._source = None
        for name, detector in tuple(self._detectors.items()):
            try:
                detector.close()
            except Exception:
                logger.exception("关闭模型失败 model=%s", name)
        self._detectors.clear()


def build_application(config: AppConfig) -> MonitorApplication:
    return MonitorApplication(config)
