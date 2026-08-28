from types import SimpleNamespace

import numpy as np
import pytest

from fire_smoke_monitor.encoding import JpegEncoder


class Image:
    def __init__(self, *args):
        self.args = args

    def width(self):
        return 640

    def height(self):
        return 480


def encoder_fixture(max_bytes=1024, max_pixels=16000000):
    events = []

    class Bmcv:
        def __init__(self, handle):
            pass

        def convert_format(self, image, packed):
            events.append((image, packed.args))
            return 0

        def imencode(self, ext, image):
            assert ext == ".jpg"
            return np.frombuffer(b"jpeg-bytes", dtype=np.uint8)

    sail = SimpleNamespace(
        Handle=lambda dev_id: "handle", Bmcv=Bmcv, BMImage=Image,
        Format=SimpleNamespace(FORMAT_BGR_PACKED="packed"),
        ImgDtype=SimpleNamespace(DATA_TYPE_EXT_1N_BYTE="u8"),
    )
    return JpegEncoder(sail, 0, max_bytes, max_pixels), events


def test_sdk_in_memory_encoding_converts_to_packed_without_drawing_or_files():
    encoder, events = encoder_fixture()
    image = Image()
    assert encoder.encode(image) == b"jpeg-bytes"
    assert events == [(image, ("handle", 480, 640, "packed", "u8"))]


@pytest.mark.parametrize("max_bytes,max_pixels", [(1, 16000000), (1024, 100)])
def test_reject_image_over_server_limits(max_bytes, max_pixels):
    encoder, _ = encoder_fixture(max_bytes, max_pixels)
    with pytest.raises(ValueError):
        encoder.encode(Image())
