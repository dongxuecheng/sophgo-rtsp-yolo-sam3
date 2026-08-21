import threading
import time

import pytest

from fire_smoke_monitor.config import CameraConfig
from fire_smoke_monitor.streams import FrameUnavailableError, MultiRtspSource


class FakeBMImage:
    def __init__(self):
        self.valid = False

    def width(self):
        return 640 if self.valid else 0

    def height(self):
        return 640 if self.valid else 0


class FakeMultiDecoder:
    def __init__(self, sail, queue_size, dev_id, discard_mode):
        self.sail = sail
        self.init_args = (queue_size, dev_id, discard_mode)
        self.url = None
        self.read_calls = []
        self.deleted = []

    def set_read_timeout(self, timeout):
        self.timeout = timeout

    def add_channel(self, url, frame_skip_num):
        del frame_skip_num
        self.url = url
        with self.sail.lock:
            self.sail.add_attempts.append(url)
            online = self.sail.online.get(url, True)
        return 0 if online else -1

    def read(self, *args):
        if len(args) == 1:
            raise AssertionError("blocking MultiDecoder.read overload was used")
        channel, image, read_mode = args
        assert channel == 0
        self.read_calls.append((channel, read_mode))
        with self.sail.lock:
            online = self.sail.online.get(self.url, True)
        image.valid = online
        return 0 if online else 1

    def reconnect(self, channel):
        raise AssertionError(
            "scheduler path must not call blocking reconnect({})".format(channel)
        )

    def del_channel(self, channel):
        self.deleted.append(channel)
        return 0


class FakeSail:
    BMImage = FakeBMImage

    def __init__(self):
        self.lock = threading.Lock()
        self.online = {}
        self.add_attempts = []
        self.decoders = []

    def MultiDecoder(self, queue_size, dev_id, discard_mode):
        decoder = FakeMultiDecoder(self, queue_size, dev_id, discard_mode)
        with self.lock:
            self.decoders.append(decoder)
        return decoder

    def set_online(self, url, online):
        with self.lock:
            self.online[url] = online

    def attempts_for(self, url):
        with self.lock:
            return self.add_attempts.count(url)


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def can_read(source, camera_id):
    try:
        return source.read(camera_id).width() == 640
    except FrameUnavailableError:
        return False


def make_source(sail, cameras, unavailable_timeout=5, reconnect_cooldown=0.01):
    return MultiRtspSource(
        sail,
        cameras,
        dev_id=0,
        queue_size=2,
        read_timeout_sec=unavailable_timeout,
        reconnect_cooldown_sec=reconnect_cooldown,
    )


def test_failed_channel_read_is_non_blocking_and_does_not_affect_healthy_channel():
    bad_url = "rtsp://example.invalid/bad"
    good_url = "rtsp://example.invalid/good"
    sail = FakeSail()
    source = make_source(
        sail,
        [
            CameraConfig(camera_id="bad", url=bad_url),
            CameraConfig(camera_id="good", url=good_url),
        ],
    )
    wait_for(lambda: can_read(source, "bad") and can_read(source, "good"))

    sail.set_online(bad_url, False)
    with pytest.raises(FrameUnavailableError, match="bad"):
        source.read("bad")

    assert source.read("good").width() == 640
    with sail.lock:
        read_modes = [
            mode
            for decoder in sail.decoders
            for _, mode in decoder.read_calls
        ]
    assert read_modes and set(read_modes) == {0}
    source.close()


def test_camera_offline_at_startup_connects_automatically_when_it_comes_online():
    bad_url = "rtsp://example.invalid/later"
    good_url = "rtsp://example.invalid/always"
    sail = FakeSail()
    sail.set_online(bad_url, False)
    source = make_source(
        sail,
        [
            CameraConfig(camera_id="later", url=bad_url),
            CameraConfig(camera_id="always", url=good_url),
        ],
    )

    wait_for(lambda: can_read(source, "always"))
    with pytest.raises(FrameUnavailableError, match="later"):
        source.read("later")

    sail.set_online(bad_url, True)
    wait_for(lambda: can_read(source, "later"))

    assert sail.attempts_for(bad_url) >= 2
    assert source.read("always").width() == 640
    source.close()


def test_runtime_disconnect_rebuilds_only_that_camera_and_recovers():
    flaky_url = "rtsp://example.invalid/flaky"
    stable_url = "rtsp://example.invalid/stable"
    sail = FakeSail()
    source = make_source(
        sail,
        [
            CameraConfig(camera_id="flaky", url=flaky_url),
            CameraConfig(camera_id="stable", url=stable_url),
        ],
        unavailable_timeout=0.02,
    )
    wait_for(lambda: can_read(source, "flaky") and can_read(source, "stable"))

    sail.set_online(flaky_url, False)
    with pytest.raises(FrameUnavailableError):
        source.read("flaky")
    stream = source._streams["flaky"]
    wait_for(
        lambda: time.monotonic() - stream._unavailable_since
        >= stream._unavailable_timeout_sec,
    )
    with pytest.raises(FrameUnavailableError):
        source.read("flaky")

    wait_for(lambda: sail.attempts_for(flaky_url) >= 2)
    assert source.read("stable").width() == 640

    sail.set_online(flaky_url, True)
    wait_for(lambda: can_read(source, "flaky"))

    assert source.read("stable").width() == 640
    source.close()


def test_sixteen_independent_camera_streams_can_start_and_read():
    sail = FakeSail()
    cameras = [
        CameraConfig(
            camera_id="camera_{:02d}".format(index),
            url="rtsp://example.invalid/camera_{:02d}".format(index),
        )
        for index in range(16)
    ]
    source = make_source(sail, cameras)

    wait_for(
        lambda: all(can_read(source, camera.camera_id) for camera in cameras),
        timeout=2.0,
    )

    assert len(sail.add_attempts) == 16
    source.close()
