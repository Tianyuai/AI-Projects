from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from paper_search.domain.models import StructuredSearchResponse


def reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def child_environment() -> dict[str, str]:
    environment = {
        "PYTHONPATH": str(Path("src").resolve()),
        "PYTHONUTF8": "1",
    }
    for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
        if value := os.environ.get(name):
            environment[name] = value
    return environment


class MockServerProcess:
    def __init__(self, port: int) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "paper_search.api.mock_server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=Path.cwd(),
            env=child_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.stderr = ""

    def __enter__(self) -> MockServerProcess:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.terminate_and_reap()

    def exit_category(self) -> str:
        return "process-exited" if self.process.poll() is not None else "running"

    def terminate_and_reap(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        _, stderr = self.process.communicate(timeout=3)
        self.stderr = "\n".join((stderr or "").splitlines()[-20:])


@contextmanager
def mock_server_process(port: int) -> Iterator[MockServerProcess]:
    server = MockServerProcess(port)
    try:
        yield server
    finally:
        server.terminate_and_reap()


def wait_until_live(base_url: str, server: MockServerProcess) -> None:
    deadline = time.monotonic() + 5
    with httpx.Client(timeout=httpx.Timeout(0.2)) as client:
        while time.monotonic() < deadline:
            if server.process.poll() is not None:
                pytest.fail(f"mock server exited before readiness: {server.exit_category()}")
            try:
                if client.get(f"{base_url}/health/live").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
    pytest.fail("mock server startup timed out")


def test_mock_server_process_serves_ready_and_synthetic_search() -> None:
    port = reserve_loopback_port()
    base_url = f"http://127.0.0.1:{port}"

    with mock_server_process(port) as server:
        wait_until_live(base_url, server)
        with httpx.Client(timeout=httpx.Timeout(1.0)) as client:
            ready = client.get(f"{base_url}/health/ready")
            response = client.post(
                f"{base_url}/v1/search",
                json={
                    "query_id": "synthetic-process-q1",
                    "query": "Synthetic process smoke query",
                    "budget_profile": "low",
                    "include_trace": False,
                },
            )

    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert response.status_code == 200
    assert StructuredSearchResponse.model_validate(response.json()).query_id == (
        "synthetic-process-q1"
    )
    assert "mock server blocks non-loopback network access" not in server.stderr


def test_mock_server_process_survives_invalid_request_and_unknown_route() -> None:
    port = reserve_loopback_port()
    base_url = f"http://127.0.0.1:{port}"

    with mock_server_process(port) as server:
        wait_until_live(base_url, server)
        with httpx.Client(timeout=httpx.Timeout(1.0)) as client:
            invalid = client.post(
                f"{base_url}/v1/search",
                content=b'{"query_id":"synthetic-process-q2","extra":true}',
                headers={"content-type": "application/json"},
            )
            missing = client.get(f"{base_url}/missing")
            live_after_failures = client.get(f"{base_url}/health/live")

    assert invalid.status_code == 422
    assert missing.status_code == 404
    assert live_after_failures.status_code == 200


def test_mock_server_process_reports_occupied_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = int(listener.getsockname()[1])
        with mock_server_process(port) as server:
            deadline = time.monotonic() + 5
            while server.process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            category = server.exit_category()

    assert category == "process-exited"
    assert server.process.returncode not in {None, 0}


def test_mock_server_process_reaps_delayed_request_child() -> None:
    port = reserve_loopback_port()
    delayed_script = """
import http.server
import time

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(1)

    def log_message(self, format, *args):
        pass

http.server.HTTPServer(("127.0.0.1", __PORT__), Handler).serve_forever()
""".replace("__PORT__", str(port))
    child = subprocess.Popen(
        [sys.executable, "-c", delayed_script],
        env=child_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        with pytest.raises(httpx.ReadTimeout):
            with httpx.Client(timeout=httpx.Timeout(0.01)) as client:
                client.get(f"http://127.0.0.1:{port}/slow")
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=3)

    assert child.poll() is not None
