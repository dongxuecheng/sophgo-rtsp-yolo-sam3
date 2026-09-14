from dataclasses import replace

from fire_smoke_monitor.config import HealthConfig
from fire_smoke_monitor.health import HealthRegistry
from fire_smoke_monitor.metrics import CameraMetrics


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


def make_metrics(**overrides):
    clock = Clock()
    health = HealthRegistry(clock=clock)
    config = replace(HealthConfig(10, 30, 1, 10, 20), **overrides)
    metrics = CameraMetrics(["cam1", "cam2"], config, health, clock)
    return metrics, health, clock


def test_each_camera_is_checked_for_inference_coverage_not_just_the_scheduler():
    metrics, health, clock = make_metrics()
    clock.now += 11
    metrics.inferred("cam1", False, 0.02, clock())
    metrics.inspect()
    assert health.snapshot().degraded == ("inference_cam2",)
    assert not health.fatal_event.is_set()
    metrics.inferred("cam2", False, 0.03, clock())
    metrics.inspect()
    assert not health.snapshot().degraded


def test_unconfirmed_hit_warns_until_accepted_but_no_hit_never_warns():
    metrics, health, clock = make_metrics(inference_stale_sec=100)
    metrics.inferred("cam1", True, 0.02, clock())
    clock.now += 10
    metrics.inferred("cam1", True, 0.02, clock())
    clock.now += 11
    metrics.inspect()
    assert health.snapshot().degraded == ("delivery_cam1",)
    metrics.accepted("cam1", clock())
    metrics.inspect()
    assert not health.snapshot().degraded
    assert metrics.snapshot("cam1").accepted == 1


def test_old_request_success_does_not_acknowledge_newer_hit():
    metrics, _, clock = make_metrics()
    metrics.inferred("cam1", True, 0.02, clock())
    older = clock()
    clock.now += 2
    metrics.inferred("cam1", True, 0.02, clock())
    metrics.accepted("cam1", older)
    assert metrics.snapshot("cam1").unconfirmed_since is not None
    metrics.accepted("cam1", clock())
    assert metrics.snapshot("cam1").unconfirmed_since is None


def test_counters_timing_and_limited_status_reports(caplog):
    caplog.set_level("INFO")
    metrics, _, clock = make_metrics()
    metrics.inferred("cam1", False, 0.02, clock())
    clock.now += 3
    metrics.inferred("cam1", True, 0.05, clock())
    metrics.processed("cam1", 0.08)
    for name in ("submitted", "replaced", "expired", "failed", "no_frame", "errors"):
        metrics.event("cam1", name)
    row = metrics.snapshot("cam1")
    assert row.inferences == 2 and row.hits == 1
    assert row.inference_gap_sec == 3
    assert row.processing_sec == 0.08
    row.hits = 900
    assert metrics.snapshot("cam1").hits == 1
    metrics.inspect()
    metrics.inspect()
    assert caplog.text.count("摄像头覆盖 camera=cam1") == 1
    assert "inferred=2" in caplog.text and "infer_ms=50.0" in caplog.text
    assert "expired=1" in caplog.text and "last_upload_age=-1.0s" in caplog.text
