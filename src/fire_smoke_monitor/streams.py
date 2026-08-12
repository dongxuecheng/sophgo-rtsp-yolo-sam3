from __future__ import annotations

import logging
import time
from typing import Any, Dict, Sequence

from .config import CameraConfig


logger = logging.getLogger(__name__)


class MultiRtspSource:
    """Thin wrapper around SAIL MultiDecoder with newest-frame semantics."""

    def __init__(
        self,
        sail_module: Any,
        cameras: Sequence[CameraConfig],
        dev_id: int,
        queue_size: int,
        read_timeout_sec: int,
        reconnect_cooldown_sec: float,
    ) -> None:
        self._decoder = sail_module.MultiDecoder(queue_size, dev_id, 1)
        self._decoder.set_read_timeout(read_timeout_sec)
        self._channels: Dict[str, int] = {}
        self._last_reconnect: Dict[str, float] = {}
        self._reconnect_cooldown_sec = reconnect_cooldown_sec

        try:
            for camera in cameras:
                channel = int(self._decoder.add_channel(camera.url, 0))
                if channel < 0:
                    raise RuntimeError("打开 RTSP 通道失败: {}".format(camera.camera_id))
                self._channels[camera.camera_id] = channel
                logger.info("RTSP 通道已启动 camera=%s channel=%d", camera.camera_id, channel)
        except Exception:
            self.close()
            raise

    def read(self, camera_id: str) -> Any:
        channel = self._channels[camera_id]
        try:
            image = self._decoder.read(channel)
            if image is None or image.width() <= 0 or image.height() <= 0:
                raise RuntimeError("解码器未返回有效图像")
            return image
        except Exception:
            self._try_reconnect(camera_id, channel)
            raise

    def _try_reconnect(self, camera_id: str, channel: int) -> None:
        now = time.monotonic()
        if now - self._last_reconnect.get(camera_id, 0.0) < self._reconnect_cooldown_sec:
            return
        self._last_reconnect[camera_id] = now
        reconnect = getattr(self._decoder, "reconnect", None)
        if reconnect is None:
            return
        try:
            result = reconnect(channel)
            logger.warning("请求重连 RTSP camera=%s result=%s", camera_id, result)
        except Exception:
            logger.exception("RTSP 重连调用失败 camera=%s", camera_id)

    def close(self) -> None:
        channels = list(getattr(self, "_channels", {}).items())
        for camera_id, channel in channels:
            try:
                self._decoder.del_channel(channel)
            except Exception:
                logger.exception("关闭 RTSP 通道失败 camera=%s", camera_id)
        if hasattr(self, "_channels"):
            self._channels.clear()

