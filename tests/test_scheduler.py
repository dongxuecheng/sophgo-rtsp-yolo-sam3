from pathlib import Path
import threading

from fire_smoke_monitor.config import parse_config
from fire_smoke_monitor.domain import Detection
from fire_smoke_monitor.health import HealthRegistry
from fire_smoke_monitor.pacing import PeriodicPacer
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
    def __init__(self, events, config, clock):
        self.events = events
        self.frames = []
        self.reserved = set()
        self.busy = False
        self.pacer = PeriodicPacer(config.periodic, clock)

    def try_reserve(self, camera_id):
        if self.busy:
            return "busy"
        if not self.pacer.try_acquire():
            return "paused"
        self.reserved.add(camera_id)
        return "reserved"

    def cancel_reservation(self, camera_id):
        self.reserved.discard(camera_id)

    def submit_reserved(self, frame):
        assert frame.camera_id in self.reserved
        self.cancel_reservation(frame.camera_id)
        assert self.pacer.start_upload()
        self.submit(frame)
        return "dispatched"

    def submit(self, frame):
        self.events.append("submit:" + frame.camera_id)
        self.frames.append(frame)
        return "queued"


def make_scheduler(mode="yolo_trigger", interval=3, health=None, cameras=("cam1", "cam2")):
    config = parse_config({
        "mode": mode,
        "pipeline": {"endpoint": "http://example.invalid/v1/frames", "machine_id": "box-1"},
        "cameras": [{"id": cam, "name": "门口", "url": "rtsp://example.invalid/" + cam}
                    for cam in cameras],
        "periodic": {"upload_interval_sec": interval},
        "models": {"yolo26": {"bmodel": "missing", "class_names": "missing",
                              "target_classes": ["fire"], "interval_sec": interval}},
    }, Path("."))
    events, clock = [], FakeClock()
    detector = FakeDetector(events) if mode == "yolo_trigger" else None
    source, encoder = FakeSource(events), FakeEncoder(events)
    uploader = FakeUploader(events, config, clock)
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
    assert scheduler.run_iteration() == 1
    assert len(uploader.frames) == 1
    assert not any(event.startswith("detect") for event in events)
    clock.now += 9
    assert scheduler.run_iteration() == 0
    clock.now += 1
    assert scheduler.run_iteration() == 1
    assert [frame.camera_id for frame in uploader.frames] == ["cam1", "cam2"]
    assert not scheduler._jobs


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
    assert not uploader.frames
    assert not uploader.reserved
    clock.now += 3
    scheduler.run_iteration()
    assert [frame.camera_id for frame in uploader.frames] == ["cam2"]
    source.offline.clear()
    clock.now += 3
    scheduler.run_iteration()
    assert [frame.camera_id for frame in uploader.frames] == ["cam2", "cam1"]


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
    scheduler, _, _, uploader, _, _, _ = make_scheduler()
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
    assert not health.fatal_event.is_set()
    assert not scheduler._uploader.reserved
    clock.now += 3
    scheduler.run_iteration()
    assert health.fatal_event.is_set()
    assert health.snapshot().processing_failures == 2


def test_periodic_33_cameras_are_round_robin_not_a_batch():
    cameras = tuple("cam{}".format(i) for i in range(33))
    scheduler, _, _, uploader, _, clock, _ = make_scheduler(
        "periodic", interval=2.5, cameras=cameras,
    )
    for index in range(34):
        assert scheduler.run_iteration() == 1
        assert uploader.frames[-1].camera_id == cameras[index % 33]
        assert len(uploader.frames) == index + 1
        assert scheduler.run_iteration() == 0
        clock.now += 2.5
    assert uploader.frames[33].created_monotonic - uploader.frames[0].created_monotonic == 82.5


def test_periodic_busy_does_not_capture_and_recovery_uses_fresh_frame():
    scheduler, source, _, uploader, _, clock, events = make_scheduler("periodic", 2.5)
    uploader.busy = True
    for _ in range(10):
        assert scheduler.run_iteration() == 0
        clock.now += 10
    assert not events  # no reads, no JPEG encoding, no pending frames
    uploader.busy = False
    source.read = lambda camera: "fresh-at-{}".format(clock.now)
    assert scheduler.run_iteration() == 1
    assert uploader.frames[0].camera_id == "cam1"
    assert uploader.frames[0].jpeg == b"fresh-at-200.0"
    assert scheduler.run_iteration() == 0  # missed ticks never get replayed


def test_periodic_slow_encoding_does_not_compress_upload_spacing():
    scheduler, _, encoder, uploader, _, clock, _ = make_scheduler("periodic", 2.5)
    original_encode = encoder.encode

    def slow_encode(image):
        clock.now += 5
        return original_encode(image)

    encoder.encode = slow_encode
    assert scheduler.run_iteration() == 1
    assert scheduler.run_iteration() == 0
    assert len(uploader.frames) == 1
    clock.now += 2.5
    assert scheduler.run_iteration() == 1


def test_periodic_stopped_or_fatal_never_captures():
    health = HealthRegistry()
    scheduler, _, _, _, _, _, events = make_scheduler("periodic", health=health)
    stop = threading.Event()
    stop.set()
    assert scheduler.run_iteration(stop) == 0
    stop.clear()
    health.fatal("test")
    assert scheduler.run_iteration(stop) == 0
    assert not events


def test_periodic_encoding_failure_releases_slot_and_moves_to_next_camera():
    scheduler, _, encoder, uploader, _, clock, _ = make_scheduler("periodic", 2.5)
    original_encode = encoder.encode
    encoder.encode = lambda image: (_ for _ in ()).throw(ValueError("test"))
    assert scheduler.run_iteration() == 1
    assert not uploader.frames and not uploader.reserved
    encoder.encode = original_encode
    clock.now += 2.5
    assert scheduler.run_iteration() == 1
    assert uploader.frames[0].camera_id == "cam2"
