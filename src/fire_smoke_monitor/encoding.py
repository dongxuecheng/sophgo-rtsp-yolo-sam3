from typing import Any


class JpegEncoder:
    """Encode an unannotated frame in RAM using the Sophon SDK multimedia path."""

    def __init__(self, sail_module: Any, dev_id: int, max_bytes: int, max_pixels: int):
        self._sail = sail_module
        self._handle = sail_module.Handle(dev_id)
        self._bmcv = sail_module.Bmcv(self._handle)
        self._max_bytes = max_bytes
        self._max_pixels = max_pixels
        if not hasattr(self._bmcv, "imencode"):
            raise RuntimeError("SAIL 缺少 Bmcv.imencode，请安装包含 Sophon OpenCV 的完整 SAIL")

    def encode(self, image: Any) -> bytes:
        width, height = int(image.width()), int(image.height())
        if width <= 0 or height <= 0 or width * height > self._max_pixels:
            raise ValueError("图片尺寸无效或超过 pipeline.max_image_pixels；请降低摄像头分辨率")
        # imencode expects BGR_PACKED, not convert_format's default BGR_PLANAR.
        # Never draw boxes or copy raw pixels into a Python/CPU OpenCV pipeline.
        packed = self._sail.BMImage(
            self._handle, height, width,
            self._sail.Format.FORMAT_BGR_PACKED,
            self._sail.ImgDtype.DATA_TYPE_EXT_1N_BYTE,
        )
        if self._bmcv.convert_format(image, packed) != 0:
            raise RuntimeError("BMCV BGR_PACKED 转换失败")
        data = self._bmcv.imencode(".jpg", packed).tobytes()
        if not data or len(data) > self._max_bytes:
            raise ValueError("JPEG 为空或超过 pipeline.max_image_bytes；请降低摄像头分辨率")
        return data
