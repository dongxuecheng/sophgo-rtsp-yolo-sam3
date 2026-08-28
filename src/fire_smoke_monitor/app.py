from __future__ import annotations

import logging
import threading

from .config import AppConfig
from .encoding import JpegEncoder
from .scheduler import InferenceScheduler
from .streams import MultiRtspSource
from .uploader import FrameUploader


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
        self._detector = None
        self._uploader = None
        self._scheduler = None
        self._closed = False
        try:
            handle = sail.Handle(config.runtime.dev_id)
            target = str(handle.get_target())
            if "1684X" not in target.upper():
                raise RuntimeError("当前设备不是 BM1684X: {}".format(target))
            logger.info("硬件设备已确认 dev_id=%d target=%s", config.runtime.dev_id, target)

            encoder = JpegEncoder(
                sail, config.runtime.dev_id,
                config.pipeline.max_image_bytes, config.pipeline.max_image_pixels,
            )
            if config.yolo26 is not None:
                from .models import Yolo26Detector

                self._detector = Yolo26Detector(config.yolo26, config.runtime.dev_id)
            else:
                logger.info("定时直传模式：不加载 YOLO，不进行本地目标检测")

            self._uploader = FrameUploader(
                config.pipeline, [camera.camera_id for camera in config.cameras]
            )

            self._source = MultiRtspSource(
                sail,
                config.cameras,
                config.runtime.dev_id,
                config.runtime.decoder_queue_size,
                config.runtime.read_timeout_sec,
                config.runtime.reconnect_cooldown_sec,
            )
            self._scheduler = InferenceScheduler(
                config, self._source, encoder, self._uploader, self._detector,
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
        if self._uploader is not None:
            self._uploader.close()
            self._uploader = None
        if self._source is not None:
            self._source.close()
            self._source = None
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                logger.exception("关闭 YOLO 模型失败")
            self._detector = None
        self._scheduler = None


def build_application(config: AppConfig) -> MonitorApplication:
    return MonitorApplication(config)
