from fire_smoke_monitor.domain import Detection, ModelRuntime
from fire_smoke_monitor.scheduler import InferenceScheduler


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeSource:
    def __init__(self, events):
        self.events = events

    def read(self, camera_id):
        self.events.append("read:{}".format(camera_id))
        return "frame:{}".format(camera_id)


class FakeDetector:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def detect(self, image):
        self.events.append("detect:{}:{}".format(self.name, image))
        return [Detection("fire", 0.9, 1, 2, 10, 20)]


class FakeWriter:
    def __init__(self, events):
        self.events = events

    def save(self, model, camera, image, detections, cooldown):
        del image, detections, cooldown
        self.events.append("save:{}:{}".format(model, camera))
        return "/tmp/result.jpg"


def test_yolo_jobs_run_before_only_one_sam_job_per_iteration():
    events = []
    clock = FakeClock()
    models = {
        # Intentionally insert SAM first: scheduler priority must not rely on dict order.
        "sam3": ModelRuntime(FakeDetector("sam3", events), 20, 60),
        "yolo26": ModelRuntime(FakeDetector("yolo26", events), 3, 30),
    }
    scheduler = InferenceScheduler(
        ["cam1", "cam2"], models, FakeSource(events), FakeWriter(events), clock
    )

    assert scheduler.run_iteration() == 3

    detects = [event for event in events if event.startswith("detect:")]
    assert detects == [
        "detect:yolo26:frame:cam1",
        "detect:yolo26:frame:cam2",
        "detect:sam3:frame:cam1",
    ]

    events.clear()
    assert scheduler.run_iteration() == 1
    assert "detect:sam3:frame:cam2" in events


def test_fixed_delay_prevents_catch_up_burst():
    events = []
    clock = FakeClock()
    models = {"yolo26": ModelRuntime(FakeDetector("yolo26", events), 3, 0)}
    scheduler = InferenceScheduler(
        ["cam1"], models, FakeSource(events), FakeWriter(events), clock
    )

    assert scheduler.run_iteration() == 1
    assert scheduler.run_iteration() == 0
    clock.now += 3
    assert scheduler.run_iteration() == 1

