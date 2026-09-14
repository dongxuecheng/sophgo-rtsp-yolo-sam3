from pathlib import Path
import threading

from fire_smoke_monitor.config import parse_config
from fire_smoke_monitor.domain import Detection
from fire_smoke_monitor.health import HealthRegistry
from fire_smoke_monitor.metrics import CameraMetrics
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
        config, source, encoder, uploader, detector, clock, health=health,
        metrics=CameraMetrics(cameras, config.health, health, clock) if detector else None,
    )
    return scheduler, source, encoder, uploader, detector, clock, events


def test_yolo_multiple_hits_upload_one_original_frame_per_camera():
    scheduler, _, _, uploader, _, clock, events = make_scheduler()
    assert scheduler.run_iteration() == 1
    assert scheduler.run_iteration() == 0
    clock.now += 1.5
    assert scheduler.run_iteration() == 1
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


def test_yolo_missed_periods_are_not_replayed():
    scheduler, _, _, _, _, clock, _ = make_scheduler()
    assert scheduler.run_iteration() == 1
    clock.now += 100
    assert scheduler.run_iteration() == 1
    assert scheduler.run_iteration() == 1
    assert scheduler.run_iteration() == 0
    clock.now += 3
    assert scheduler.run_iteration() == 1
    assert scheduler.run_iteration() == 1
    assert scheduler.run_iteration() == 0


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
    scheduler, _, encoder, uploader, _, clock, _ = make_scheduler()
    encode = encoder.encode

    def fail_one(image):
        if image == "frame:cam1":
            raise ValueError("bad frame")
        return encode(image)

    encoder.encode = fail_one
    scheduler.run_iteration()
    clock.now += 1.5
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


def test_yolo_33_slow_cameras_all_inferred_even_when_http_is_busy():
    cameras = tuple("cam{}".format(i) for i in range(33))
    scheduler, _, encoder, uploader, detector, clock, events = make_scheduler(
        interval=3, cameras=cameras,
    )
    uploader.busy = True  # YOLO must never reserve/wait for HTTP before detecting.
    detect = detector.detect
    encode = encoder.encode

    def slow_detect(image):
        clock.now += 0.2
        return detect(image)

    def slow_encode(image):
        clock.now += 0.05
        return encode(image)

    detector.detect, encoder.encode = slow_detect, slow_encode
    for _ in range(99):
        assert scheduler.run_iteration() == 1
    reads = [event[len("read:"):] for event in events if event.startswith("read:")]
    assert reads == list(cameras) * 3
    assert len(uploader.frames) == 99
    for camera in cameras:
        row = scheduler._metrics.snapshot(camera)
        assert row.inferences == row.hits == 3
        assert abs(row.inference_sec - 0.2) < 1e-6
        assert abs(row.processing_sec - 0.25) < 1e-6
        assert abs(row.inference_gap_sec - 8.25) < 1e-6


def test_yolo_interval_includes_actual_inference_and_encoding_time():
    scheduler, _, encoder, _, detector, clock, _ = make_scheduler(cameras=("cam1",), interval=3)
    detect, encode = detector.detect, encoder.encode

    def infer(image):
        clock.now += 0.7
        return detect(image)

    def jpeg(image):
        clock.now += 0.3
        return encode(image)

    detector.detect, encoder.encode = infer, jpeg
    scheduler.run_iteration()
    assert scheduler._jobs[0].next_due == 103.0  # not 104: processing is included
    clock.now = 102.99
    assert scheduler.run_iteration() == 0
    clock.now = 103.0
    assert scheduler.run_iteration() == 1


def test_yolo_fast_repeatedly_due_camera_cannot_starve_other_due_cameras():
    scheduler, _, _, _, detector, clock, events = make_scheduler(
        cameras=("fast", "slow", "normal"), interval=0.01,
    )
    scheduler._jobs[0].interval_sec = 0.001
    scheduler._jobs[1].interval_sec = 0.1
    scheduler._jobs[2].interval_sec = 0.1
    original = detector.detect

    def detect(image):
        clock.now += 0.4 if "slow" in image else 0.05
        return original(image)

    detector.detect = detect
    for _ in range(12):
        assert scheduler.run_iteration() == 1
    reads = [event for event in events if event.startswith("read:")]
    assert reads == ["read:fast", "read:slow", "read:normal"] * 4


def test_yolo_offline_camera_does_not_stop_inference_and_can_recover():
    scheduler, source, _, _, _, clock, _ = make_scheduler(interval=1)
    source.offline.add("cam1")
    scheduler.run_iteration()
    clock.now += 0.5
    scheduler.run_iteration()
    assert scheduler._metrics.snapshot("cam1").no_frame == 1
    assert scheduler._metrics.snapshot("cam2").inferences == 1
    source.offline.clear()
    clock.now += 0.5
    scheduler.run_iteration()
    assert scheduler._metrics.snapshot("cam1").inferences == 1


def test_yolo_nonhit_never_submits_a_replacement_for_pending_hit():
    scheduler, _, _, uploader, detector, clock, _ = make_scheduler(cameras=("cam1",))
    scheduler.run_iteration()
    saved_hit = uploader.frames[0]
    detector.hits = []
    clock.now += 3
    scheduler.run_iteration()
    assert uploader.frames == [saved_hit]
    assert scheduler._metrics.snapshot("cam1").inferences == 2
    assert scheduler._metrics.snapshot("cam1").hits == 1
