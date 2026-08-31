import os
import threading
import time

from fire_smoke_monitor.health import (
    HealthRegistry,
    HealthSupervisor,
    SystemdNotifier,
)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_registry_detects_stalled_heartbeat_and_dead_probe():
    clock = FakeClock()
    registry = HealthRegistry(clock=clock)
    registry.require_heartbeat("scheduler", 5)
    registry.register_probe("uploader", lambda: True)
    assert not registry.inspect().fatal_reason
    clock.now += 6
    assert registry.inspect().fatal_reason == "scheduler_stalled"

    other = HealthRegistry(clock=clock)
    other.register_probe("logging", lambda: False)
    assert other.inspect().fatal_reason == "logging_stopped"


def test_processing_success_resets_consecutive_failure_counter():
    registry = HealthRegistry(max_processing_failures=2)
    assert registry.processing_failure("ValueError") == 1
    registry.processing_success()
    assert registry.snapshot().processing_failures == 0
    assert registry.processing_failure("ValueError") == 1
    assert not registry.fatal_event.is_set()
    assert registry.processing_failure("ValueError") == 2
    assert registry.fatal_event.is_set()


def test_systemd_notifier_sends_to_configured_unix_socket():
    calls = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect(self, address):
            calls.append(("connect", address))

        def sendall(self, payload):
            calls.append(("send", payload))

    notifier = SystemdNotifier(
        {"NOTIFY_SOCKET": "@test", "WATCHDOG_USEC": "60000000"},
        socket_factory=lambda *args: Socket(),
    )
    assert notifier.watchdog_sec == 60
    assert notifier.notify("READY=1")
    assert calls == [("connect", "\0test"), ("send", b"READY=1")]

    wrong_pid = SystemdNotifier({
        "NOTIFY_SOCKET": "@test", "WATCHDOG_PID": str(os.getpid() + 1),
    })
    assert not wrong_pid.enabled


def test_supervisor_pings_watchdog_then_stops_on_fatal():
    class Notifier:
        watchdog_sec = 0.03

        def __init__(self):
            self.states = []

        def notify(self, state):
            self.states.append(state)
            return True

    registry = HealthRegistry()
    stop = threading.Event()
    notifier = Notifier()
    supervisor = HealthSupervisor(
        registry, stop, notifier=notifier,
        check_interval_sec=0.005, status_interval_sec=1,
    )
    supervisor.start()
    deadline = time.monotonic() + 1
    while not any(state == "WATCHDOG=1" for state in notifier.states):
        assert time.monotonic() < deadline
        time.sleep(0.005)
    registry.fatal("test_failure")
    assert stop.wait(1)
    supervisor.close()
    assert notifier.states[0].startswith("READY=1")
    assert any("fatal: test_failure" in state for state in notifier.states)
    assert notifier.states[-1].startswith("STOPPING=1")
