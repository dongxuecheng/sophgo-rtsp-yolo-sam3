import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import httpx
import pytest

from fire_smoke_monitor.config import PipelineConfig
from fire_smoke_monitor.domain import UploadFrame
from fire_smoke_monitor.health import HealthRegistry
from fire_smoke_monitor.uploader import (
    FrameUploader,
    PipelineClient,
    UploadError,
    _await_upload,
)


def config(**overrides):
    return replace(PipelineConfig(
        "http://example.invalid:18080/v1/frames", "se7-01", "", 2, 1, 30,
        8388608, 16000000,
    ), **overrides)


def frame(camera="cam1", jpeg=b"original-jpeg", created=None):
    return UploadFrame(camera, "仓库东门", datetime.now(timezone.utc),
                       time.monotonic() if created is None else created, jpeg)


def reply(status=202, body=b'{"accepted":true}'):
    return httpx.Response(status, stream=httpx.ByteStream(body))


def multipart(content_type, body):
    message = BytesParser(policy=policy.default).parsebytes(
        ("Content-Type: " + content_type + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body
    )
    return {part.get_param("name", header="content-disposition"): part
            for part in message.iter_parts()}


def fields_from_request(request):
    return multipart(request.headers["content-type"], request.content)


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition timed out")


async def wait_event(event):
    while not event.is_set():
        await asyncio.sleep(0.005)


def test_upload_timeout_race_always_retrieves_child_exception():
    """Python 3.8 wait_for may hide an exception raised during cancellation."""
    contexts = []

    async def cancellation_race():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise httpx.WriteTimeout("")

    async def run():
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda current_loop, context: contexts.append(context))
        try:
            with pytest.raises((asyncio.TimeoutError, httpx.WriteTimeout)):
                await _await_upload(cancellation_race(), 0.01)
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous)

    asyncio.run(run())
    assert not contexts


def test_multipart_contract_key_original_image_and_aware_timestamp():
    sent = frame()

    async def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/v1/frames"
        assert request.headers["X-API-Key"] == "test-key"
        fields = fields_from_request(request)
        assert set(fields) == {"image", "machine_id", "stream_id", "stream_name", "captured_at"}
        assert fields["image"].get_payload(decode=True) == sent.jpeg
        assert fields["image"].get_content_type() == "image/jpeg"
        assert fields["machine_id"].get_payload(decode=True) == b"se7-01"
        assert fields["stream_id"].get_payload(decode=True) == b"cam1"
        assert fields["stream_name"].get_payload(decode=True).decode() == "仓库东门"
        assert fields["captured_at"].get_payload(decode=True).decode() == sent.captured_at.isoformat()
        return reply()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await PipelineClient(config(), http, "test-key").upload(sent)

    asyncio.run(run())


@pytest.mark.parametrize("status,body", [
    (200, b'{"accepted":true}'), (202, b'{"accepted":false}'), (202, b'{}'),
    (202, b'not json'), (202, b'x' * 4097), (202, b'[]'),
    (401, b'secret'), (413, b'too large'), (429, b'full'), (503, b'busy'),
    (307, b'redirect'),
])
def test_only_valid_202_is_success_and_never_retries(status, body):
    calls = []

    async def handler(request):
        calls.append(request)
        return reply(status, body)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            with pytest.raises(UploadError):
                await PipelineClient(config(), http, "").upload(frame())

    asyncio.run(run())
    assert len(calls) == 1
    assert "X-API-Key" not in calls[0].headers


def test_slow_camera_request_does_not_block_other_camera_or_scheduler_and_coalesces():
    started, release, other_done = threading.Event(), threading.Event(), threading.Event()
    calls = []

    async def handler(request):
        fields = fields_from_request(request)
        camera = fields["stream_id"].get_payload(decode=True).decode()
        jpeg = fields["image"].get_payload(decode=True)
        calls.append((camera, jpeg))
        if camera == "cam1" and jpeg == b"first":
            started.set()
            await wait_event(release)
        if camera == "cam2":
            other_done.set()
        return reply()

    uploader = FrameUploader(config(timeout_sec=5), ["cam1", "cam2"],
                             transport=httpx.MockTransport(handler))
    try:
        uploader.submit(frame(jpeg=b"first"))
        assert started.wait(2)
        assert uploader.submit(frame(jpeg=b"old")) == "queued"
        assert uploader.submit(frame(jpeg=b"newest")) == "replaced"
        uploader.submit(frame("cam2"))
        assert other_done.wait(2)  # cam1 is still blocked, but cam2 is sent.
        assert not release.is_set()
        assert ("cam1", b"newest") not in calls  # only one in flight per camera
        release.set()
        wait_for(lambda: ("cam1", b"newest") in calls)
        assert ("cam1", b"old") not in calls
    finally:
        release.set()
        uploader.close()
    assert not uploader._thread.is_alive()


def test_pending_queue_is_bounded_to_camera_count():
    started, release = threading.Event(), threading.Event()

    async def handler(request):
        started.set()
        await wait_event(release)
        return reply()

    cameras = ["cam{}".format(i) for i in range(16)]
    uploader = FrameUploader(config(workers=1, timeout_sec=5), cameras,
                             transport=httpx.MockTransport(handler))
    try:
        uploader.submit(frame("cam0"))
        assert started.wait(2)
        for _ in range(20):
            for camera in cameras:
                uploader.submit(frame(camera))
        with uploader._lock:
            assert len(uploader._pending) == 16
            assert len(uploader._active) == 1
    finally:
        uploader.close()
    assert not uploader._thread.is_alive()
    assert not uploader._pending


def test_total_timeout_releases_worker_and_next_cycle_recovers(caplog):
    calls, recovered = [], threading.Event()

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            await asyncio.sleep(2)
        recovered.set()
        return reply()

    uploader = FrameUploader(config(workers=1, timeout_sec=0.05), ["cam1", "cam2"],
                             transport=httpx.MockTransport(handler))
    try:
        uploader.submit(frame())
        wait_for(lambda: len(calls) == 1)
        uploader.submit(frame("cam2"))
        assert recovered.wait(2)
        assert len(calls) == 2
        assert "上传超时" in caplog.text
    finally:
        uploader.close()


def test_httpx_phase_timeout_is_logged_once_and_worker_continues(caplog):
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.WriteTimeout("")
        return reply()

    uploader = FrameUploader(
        config(workers=1), ["cam1"], transport=httpx.MockTransport(handler)
    )
    try:
        uploader.submit(frame())
        wait_for(lambda: "scope=httpx error=WriteTimeout" in caplog.text)
        uploader.submit(frame(jpeg=b"fresh"))
        wait_for(lambda: len(calls) == 2)
        assert "Task exception was never retrieved" not in caplog.text
    finally:
        uploader.close()


def test_429_is_dropped_and_a_new_frame_can_succeed(caplog):
    calls = []

    async def handler(request):
        calls.append(request)
        return reply(429) if len(calls) == 1 else reply()

    uploader = FrameUploader(config(), ["cam1"], transport=httpx.MockTransport(handler))
    try:
        uploader.submit(frame())
        wait_for(lambda: "HTTP 429" in caplog.text)
        uploader.submit(frame(jpeg=b"fresh"))
        wait_for(lambda: len(calls) == 2)
    finally:
        uploader.close()
    assert len(calls) == 2


def test_remote_failure_is_degraded_not_fatal_and_success_recovers():
    calls = []
    health = HealthRegistry()

    async def handler(request):
        calls.append(request)
        return reply(503) if len(calls) == 1 else reply()

    uploader = FrameUploader(
        config(), ["cam1"], transport=httpx.MockTransport(handler), health=health
    )
    try:
        uploader.submit(frame())
        wait_for(lambda: "remote" in health.snapshot().degraded)
        assert not health.fatal_event.is_set()
        uploader.submit(frame(jpeg=b"fresh"))
        wait_for(lambda: "remote" not in health.snapshot().degraded)
        assert not health.fatal_event.is_set()
    finally:
        uploader.close()


def test_expired_frames_are_not_sent_and_unknown_camera_rejected(caplog):
    calls = []

    async def handler(request):
        calls.append(request)
        return reply()

    uploader = FrameUploader(config(max_pending_age_sec=1), ["cam1"],
                             transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError):
            uploader.submit(frame("not-configured"))
        uploader.submit(frame(created=time.monotonic() - 5))
        wait_for(lambda: "过期" in caplog.text)
        assert not calls
    finally:
        uploader.close()
    assert uploader.submit(frame()) == "stopped"


def test_real_http_upload_to_local_server(monkeypatch):
    received, done = [], threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, self.headers, body))
            payload = json.dumps({"accepted": True}).encode()
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            done.set()

        def log_message(self, *args):
            pass

    # Proxies must not silently forward camera frames or keys elsewhere.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    uploader = FrameUploader(config(endpoint="http://127.0.0.1:{}/v1/frames".format(
        server.server_port)), ["cam1"])
    try:
        uploader.submit(frame())
        assert done.wait(2)
        path, headers, body = received[0]
        assert path == "/v1/frames"
        assert multipart(headers["Content-Type"], body)["image"].get_payload(decode=True) == b"original-jpeg"
    finally:
        uploader.close()
        server.shutdown()
        server.server_close()
        thread.join(2)
