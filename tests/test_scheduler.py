from pathlib import Path
import threading

from fire_smoke_monitor.config import parse_config
from fire_smoke_monitor.domain import Detection
from fire_smoke_monitor.health import HealthRegistry
from fire_smoke_monitor.scheduler import InferenceScheduler
from fire_smoke_monitor.streams import FrameUnavailableError


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeSource:
    def __init__(self, events):
        self.events = events
        self.offline = set()

    def read(self, camera_id):
        self.events.append("read:" + camera_id)
        if camera_id in self.offline:
            raise FrameUnavailableError()
        return "frame:" + camera_id


class FakeDetector:
    def __init__(self, events):
        self.events = events
        self.hits = [Detection("fire", 0.9, 1, 2, 10, 20)] * 3

    def detect(self, image):
        self.events.append("detect:" + image)
        return self.hits


class FakeEncoder:
    def __init__(self, events):
        self.events = events

    def encode(self, image):
        self.events.append("encode:" + image)
        return image.encode()


class FakeUploader:
    def __init__(self, events):
        self.events = events
        self.frames = []

    def submit(self, frame):
        self.events.append("submit:" + frame.camera_id)
        self.frames.append(frame)
        return "queued"


def make_scheduler(mode="yolo_trigger", interval=3, health=None):
    config = parse_config({
        "mode": mode,
        "pipeline": {"endpoint": "http://example.invalid/v1/frames", "machine_id": "box-1"},
        "cameras": [{"id": cam, "name": "门口", "url": "rtsp://example.invalid/" + cam}
                    for cam in ("cam1", "cam2")],
        "periodic": {"interval_sec": interval},
        "models": {"yolo26": {"bmodel": "missing", "class_names": "missing",
                              "target_classes": ["fire"], "interval_sec": interval}},
    }, Path("."))
    events, clock = [], FakeClock()
    detector = FakeDetector(events) if mode == "yolo_trigger" else None
    source, encoder, uploader = FakeSource(events), FakeEncoder(events), FakeUploader(events)
    scheduler = InferenceScheduler(
        config, source, encoder, uploader, detector, clock, health=health
    )
    return scheduler, source, encoder, uploader, detector, clock, events


def test_yolo_multiple_hits_upload_one_original_frame_per_camera():
    scheduler, _, _, uploader, _, _, events = make_scheduler()
    assert scheduler.run_iteration() == 2
    assert events == [
        "read:cam1", "detect:frame:cam1", "encode:frame:cam1", "submit:cam1",
        "read:cam2", "detect:frame:cam2", "encode:frame:cam2", "submit:cam2",
    ]
    assert len(uploader.frames) == 2
    assert uploader.frames[0].jpeg == b"frame:cam1"
    assert uploader.frames[0].stream_name == "门口"
    assert uploader.frames[0].captured_at.utcoffset() is not None
    assert scheduler.run_iteration() == 0


def test_yolo_no_hits_neither_encodes_nor_uploads():
    scheduler, _, _, uploader, detector, _, events = make_scheduler()
    detector.hits = []
    scheduler.run_iteration()
    assert not uploader.frames
    assert not any(event.startswith("encode") for event in events)


def test_periodic_skips_detector_and_uses_interval():
    scheduler, _, _, uploader, detector, clock, events = make_scheduler("periodic", 10)
    assert detector is None
    assert scheduler.run_iteration() == 2
    assert len(uploader.frames) == 2
    assert not any(event.startswith("detect") for event in events)
    clock.now += 9
    assert scheduler.run_iteration() == 0
    clock.now += 1
    assert scheduler.run_iteration() == 2


def test_fixed_delay_prevents_catch_up_burst():
    scheduler, _, _, _, _, clock, _ = make_scheduler()
    assert scheduler.run_iteration() == 2
    clock.now += 100
    assert scheduler.run_iteration() == 2
    assert scheduler.run_iteration() == 0
    clock.now += 3
    assert scheduler.run_iteration() == 2


def test_offline_stream_does_not_block_other_camera_and_resumes():
    scheduler, source, _, uploader, _, clock, _ = make_scheduler("periodic")
    source.offline.add("cam1")
    scheduler.run_iteration()
    assert [frame.camera_id for frame in uploader.frames] == ["cam2"]
    source.offline.clear()
    clock.now += 3
    scheduler.run_iteration()
    assert [frame.camera_id for frame in uploader.frames] == ["cam2", "cam1", "cam2"]


def test_one_encoding_failure_does_not_block_next_camera():
    scheduler, _, encoder, uploader, _, _, _ = make_scheduler()
    encode = encoder.encode

    def fail_one(image):
        if image == "frame:cam1":
            raise ValueError("bad frame")
        return encode(image)

    encoder.encode = fail_one
    scheduler.run_iteration()
    assert [frame.camera_id for frame in uploader.frames] == ["cam2"]


def test_stop_interrupts_iteration_before_next_camera():
    scheduler, _, _, uploader, _, _, _ = make_scheduler("periodic")
    stop = threading.Event()
    submit = uploader.submit

    def submit_and_stop(frame):
        stop.set()
        return submit(frame)

    uploader.submit = submit_and_stop
    assert scheduler.run_iteration(stop) == 1
    assert len(uploader.frames) == 1


def test_consecutive_internal_failures_become_fatal_but_offline_does_not():
    health = HealthRegistry(max_processing_failures=2)
    scheduler, source, encoder, _, _, clock, _ = make_scheduler(
        "periodic", health=health
    )
    source.offline.add("cam1")
    scheduler.run_iteration()
    assert not health.fatal_event.is_set()
    assert health.snapshot().processing_failures == 0

    source.offline.clear()
    encoder.encode = lambda image: (_ for _ in ()).throw(ValueError("bad frame"))
    clock.now += 3
    scheduler.run_iteration()
    assert health.fatal_event.is_set()
    assert health.snapshot().processing_failures == 2
