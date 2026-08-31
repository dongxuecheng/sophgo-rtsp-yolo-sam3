"""Process health supervision and dependency-free systemd notifications."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import socket
import threading
import time
from typing import Callable, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


class HealthFailure(RuntimeError):
    """The process is alive, but a critical internal component is not."""


@dataclass(frozen=True)
class HealthSnapshot:
    fatal_reason: str
    degraded: Tuple[str, ...]
    processing_failures: int

    @property
    def status_text(self) -> str:
        if self.fatal_reason:
            return "fatal: {}".format(self.fatal_reason)
        if self.degraded:
            return "degraded: {}".format(",".join(self.degraded))
        return "healthy"


class HealthRegistry:
    """Thread-safe state shared by components and the watchdog supervisor."""

    def __init__(self, max_processing_failures: int = 10,
                 clock: Callable[[], float] = time.monotonic):
        if max_processing_failures < 1:
            raise ValueError("连续处理失败阈值必须大于 0")
        self._clock = clock
        self._limit = max_processing_failures
        self._lock = threading.Lock()
        self._beats: Dict[str, float] = {}
        self._required_beats: Dict[str, float] = {}
        self._probes: Dict[str, Callable[[], bool]] = {}
        self._degraded: Dict[str, str] = {}
        self._fatal_reason = ""
        self._processing_failures = 0
        self.fatal_event = threading.Event()

    def beat(self, component: str) -> None:
        with self._lock:
            self._beats[component] = self._clock()

    @property
    def processing_failure_limit(self) -> int:
        return self._limit

    def require_heartbeat(self, component: str, max_age_sec: float) -> None:
        with self._lock:
            self._beats.setdefault(component, self._clock())
            self._required_beats[component] = max_age_sec

    def register_probe(self, component: str, probe: Callable[[], bool]) -> None:
        with self._lock:
            self._probes[component] = probe

    def degrade(self, component: str, reason: str) -> None:
        del reason  # Status must remain short and must not expose remote/native text.
        with self._lock:
            self._degraded[component] = "degraded"

    def recover(self, component: str) -> None:
        with self._lock:
            self._degraded.pop(component, None)

    def processing_success(self) -> None:
        with self._lock:
            self._processing_failures = 0
            self._degraded.pop("processing", None)

    def processing_failure(self, error_type: str) -> int:
        del error_type
        with self._lock:
            self._processing_failures += 1
            count = self._processing_failures
            self._degraded["processing"] = "degraded"
        if count >= self._limit:
            self.fatal("processing_failed_{}_times".format(count))
        return count

    def fatal(self, reason: str) -> None:
        safe_reason = "".join(
            char if char.isalnum() or char in "_-:." else "_" for char in reason
        )[:160] or "unknown"
        with self._lock:
            if not self._fatal_reason:
                self._fatal_reason = safe_reason
        self.fatal_event.set()

    def inspect(self) -> HealthSnapshot:
        now = self._clock()
        with self._lock:
            required = tuple(self._required_beats.items())
            beats = dict(self._beats)
            probes = tuple(self._probes.items())
        for component, max_age in required:
            if now - beats.get(component, float("-inf")) > max_age:
                self.fatal("{}_stalled".format(component))
        for component, probe in probes:
            try:
                alive = bool(probe())
            except Exception as exc:
                self.fatal("{}_probe_{}".format(component, type(exc).__name__))
                continue
            if not alive:
                self.fatal("{}_stopped".format(component))
        with self._lock:
            return HealthSnapshot(
                self._fatal_reason,
                tuple(sorted(self._degraded)),
                self._processing_failures,
            )

    def snapshot(self) -> HealthSnapshot:
        with self._lock:
            return HealthSnapshot(
                self._fatal_reason,
                tuple(sorted(self._degraded)),
                self._processing_failures,
            )


class SystemdNotifier:
    """Small sd_notify-compatible client using only the Python standard library."""

    def __init__(self, environment: Optional[Dict[str, str]] = None,
                 socket_factory: Callable[..., socket.socket] = socket.socket):
        env = os.environ if environment is None else environment
        self._address = env.get("NOTIFY_SOCKET", "")
        self._socket_factory = socket_factory
        self._watchdog_sec = 0.0
        watchdog_pid = env.get("WATCHDOG_PID", "")
        if watchdog_pid and watchdog_pid != str(os.getpid()):
            self._address = ""
            return
        try:
            usec = int(env.get("WATCHDOG_USEC", "0"))
        except ValueError:
            usec = 0
        if usec > 0:
            self._watchdog_sec = usec / 1_000_000.0

    @property
    def enabled(self) -> bool:
        return bool(self._address)

    @property
    def watchdog_sec(self) -> float:
        return self._watchdog_sec

    def notify(self, state: str) -> bool:
        if not self._address:
            return False
        address = self._address
        if address.startswith("@"):
            address = "\0" + address[1:]
        try:
            with self._socket_factory(
                getattr(socket, "AF_UNIX", 1), socket.SOCK_DGRAM
            ) as client:
                client.connect(address)
                client.sendall(state.encode("utf-8"))
            return True
        except OSError:
            return False


class HealthSupervisor:
    """Send watchdog pings only while every critical internal component is healthy."""

    def __init__(self, registry: HealthRegistry, stop_event: threading.Event,
                 notifier: Optional[SystemdNotifier] = None,
                 check_interval_sec: float = 1.0,
                 status_interval_sec: float = 300.0,
                 clock: Callable[[], float] = time.monotonic):
        self._registry = registry
        self._stop_event = stop_event
        self._notifier = notifier or SystemdNotifier()
        self._check_interval = check_interval_sec
        self._status_interval = status_interval_sec
        self._clock = clock
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="health-supervisor", daemon=True
        )
        self._last_status = ""
        self._last_status_at = float("-inf")

    def start(self) -> None:
        snapshot = self._registry.inspect()
        if snapshot.fatal_reason:
            self._stop_event.set()
            self._notifier.notify("STATUS={}".format(snapshot.status_text))
            return
        self._thread.start()
        self._notifier.notify("READY=1\nSTATUS={}".format(snapshot.status_text))

    def _run(self) -> None:
        watchdog = self._notifier.watchdog_sec
        ping_interval = watchdog / 3.0 if watchdog > 0 else self._check_interval
        next_ping = self._clock()
        while not self._stop.wait(self._check_interval):
            snapshot = self._registry.inspect()
            now = self._clock()
            if snapshot.fatal_reason:
                logger.error("组件健康检查失败 status=%s", snapshot.status_text)
                self._notifier.notify("STATUS={}".format(snapshot.status_text))
                self._stop_event.set()
                return
            if (
                snapshot.status_text != self._last_status
                or now - self._last_status_at >= self._status_interval
            ):
                self._last_status = snapshot.status_text
                self._last_status_at = now
                log = logger.warning if snapshot.degraded else logger.info
                log("组件健康状态 status=%s", snapshot.status_text)
                self._notifier.notify("STATUS={}".format(snapshot.status_text))
            if watchdog > 0 and now >= next_ping:
                self._notifier.notify("WATCHDOG=1")
                next_ping = now + ping_interval

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        self._notifier.notify("STOPPING=1\nSTATUS=stopping")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()
