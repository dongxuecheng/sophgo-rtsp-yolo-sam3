import pytest

from fire_smoke_monitor.config import PeriodicConfig, UploadBackoffConfig
from fire_smoke_monitor.pacing import PeriodicPacer, YoloUploadPacer


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


def test_global_interval_and_no_catch_up():
    clock = Clock()
    pacer = PeriodicPacer(PeriodicConfig(), clock)
    assert pacer.try_acquire()
    assert pacer.start_upload()
    assert not pacer.try_acquire()
    clock.now += 2.49
    assert not pacer.try_acquire()
    clock.now += 0.01
    assert pacer.try_acquire()
    clock.now += 1000
    assert pacer.try_acquire()
    assert not pacer.try_acquire()


def test_consecutive_mixed_overload_responses_back_off_for_entire_machine(caplog):
    clock = Clock()
    pacer = PeriodicPacer(PeriodicConfig(), clock)
    for expected in (5, 10, 20, 40, 60, 60):
        pacer.report_result(429)
        pacer.report_result(503)
        assert pacer._interval == expected
        assert not pacer.try_acquire()
        assert not pacer.start_upload()  # reservation before failure is dropped
        clock.now += expected
        assert pacer.try_acquire()
    assert "全机降频" in caplog.text


@pytest.mark.parametrize("interruption", [202, 400, None])
def test_only_consecutive_overload_responses_trigger_backoff(interruption):
    pacer = PeriodicPacer(PeriodicConfig(), Clock())
    pacer.report_result(429)
    pacer.report_result(interruption)
    pacer.report_result(503)
    assert pacer._interval == 2.5


def test_recovery_is_gradual_and_does_not_advance_deadline():
    clock = Clock()
    pacer = PeriodicPacer(PeriodicConfig(), clock)
    for _ in range(4):
        pacer.report_result(503)
    assert pacer._interval == 10
    previous_deadline = pacer._next_allowed
    pacer.report_result(202)
    pacer.report_result(202)
    assert pacer._interval == 10
    pacer.report_result(202)
    assert pacer._interval == 5
    assert pacer._next_allowed == previous_deadline
    assert not pacer.try_acquire()
    for _ in range(6):
        pacer.report_result(202)
    assert pacer._interval == 2.5
    clock.now = previous_deadline
    assert pacer.try_acquire()
    assert not pacer.try_acquire()


def test_slow_handoff_never_dispatches_two_requests_together():
    clock = Clock()
    pacer = PeriodicPacer(PeriodicConfig(), clock)
    assert pacer.try_acquire()
    clock.now += 3
    assert pacer.try_acquire()
    # Both handoffs arrive after a delayed HTTP loop: discard the second one.
    assert pacer.start_upload()
    assert not pacer.start_upload()
    assert not pacer.try_acquire()
    clock.now += 2.5
    assert pacer.try_acquire()


def test_other_failures_break_success_recovery_streak():
    pacer = PeriodicPacer(PeriodicConfig(), Clock())
    pacer.report_result(503)
    pacer.report_result(503)
    pacer.report_result(202)
    pacer.report_result(202)
    pacer.report_result(None)
    pacer.report_result(202)
    assert pacer._interval == 5


def test_yolo_normal_upload_has_no_artificial_wait_and_overload_is_bounded():
    clock = Clock()
    pacer = YoloUploadPacer(UploadBackoffConfig(), clock)
    for _ in range(33):
        assert pacer.delay() == 0
        assert pacer.start_upload()
    for interval in (2.5, 5, 10, 20, 40, 60, 60):
        pacer.report_result(429)
        pacer.report_result(503)
        assert pacer._interval == interval
        assert pacer.delay() == interval
        assert not pacer.start_upload()
        clock.now += interval
        assert pacer.start_upload()
        assert not pacer.start_upload()


def test_yolo_success_recovery_returns_to_immediate_without_catchup():
    clock = Clock()
    pacer = YoloUploadPacer(UploadBackoffConfig(), clock)
    for _ in range(4):
        pacer.report_result(503)
    assert pacer._interval == 5
    deadline = pacer._next_allowed
    for _ in range(3):
        pacer.report_result(202)
    assert pacer._interval == 2.5
    for _ in range(3):
        pacer.report_result(202)
    assert pacer._interval == 0
    assert pacer._next_allowed == deadline
    assert not pacer.start_upload()
    clock.now = deadline
    assert pacer.start_upload()
    assert pacer.start_upload()
