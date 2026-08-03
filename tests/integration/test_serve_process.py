from __future__ import annotations

import os
import runpy
import signal
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


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _environment() -> dict[str, str]:
    result = {"PYTHONPATH": str(Path("src").resolve()), "PYTHONUTF8": "1"}
    for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
        if value := os.environ.get(name):
            result[name] = value
    return result


class _ServeProcess:
    def __init__(self, fixture: dict[str, Path], port: int) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "paper_search.cli",
                "serve",
                "--lock",
                str(fixture["replay_lock"]),
                "--mode",
                "replay",
                "--snapshot-manifest",
                str(fixture["manifest"]),
                "--capture-output-root",
                str(fixture["root"] / "captures"),
                "--port",
                str(port),
            ],
            cwd=fixture["root"],
            env=_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        self.stderr = ""

    def reap(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        _, stderr = self.process.communicate(timeout=5)
        self.stderr = stderr or ""


@contextmanager
def _serve_process(fixture: dict[str, Path], port: int) -> Iterator[_ServeProcess]:
    process = _ServeProcess(fixture, port)
    try:
        yield process
    finally:
        process.reap()


def _wait_ready(base_url: str, process: _ServeProcess) -> None:
    deadline = time.monotonic() + 5
    with httpx.Client(timeout=0.25) as client:
        while time.monotonic() < deadline:
            if process.process.poll() is not None:
                pytest.fail(f"serve exited before readiness: {process.stderr}")
            try:
                if client.get(f"{base_url}/health/live").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
    pytest.fail("serve startup timed out")


def _sealed_replay_fixture(tmp_path: Path) -> dict[str, Path]:
    helpers = runpy.run_path("tests/integration/test_smoke_cli.py")
    return helpers["_smoke_fixture"](tmp_path)  # type: ignore[no-any-return,operator]


def test_serve_process_binds_cached_replay_and_canonical_search(tmp_path: Path) -> None:
    fixture = _sealed_replay_fixture(tmp_path)
    port = _reserve_port()
    base_url = f"http://127.0.0.1:{port}"

    with _serve_process(fixture, port) as server:
        _wait_ready(base_url, server)
        with httpx.Client(timeout=1.0) as client:
            live = client.get(f"{base_url}/health/live")
            ready = client.get(f"{base_url}/health/ready")
            response = client.post(
                f"{base_url}/v1/search",
                json={
                    "query_id": "serve-replay-q1",
                    "query": "resource-aware scholarly paper search",
                    "mode": "replay",
                    "include_trace": False,
                },
            )

    assert live.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["execution_mode"] == "replay"
    assert ready.json()["snapshot_set_id"]
    canonical = StructuredSearchResponse.model_validate(response.json())
    assert canonical.query_id == "serve-replay-q1"
    assert canonical.execution_mode == "replay"
    assert canonical.snapshot_set_id == ready.json()["snapshot_set_id"]
    assert server.process.returncode == 130


def test_serve_process_reports_occupied_port_without_traceback(tmp_path: Path) -> None:
    fixture = _sealed_replay_fixture(tmp_path)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = int(listener.getsockname()[1])
        with _serve_process(fixture, port) as server:
            deadline = time.monotonic() + 5
            while server.process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)

    assert server.process.returncode not in {None, 0}
    assert "serve failed: startup error" in server.stderr
    assert "Traceback" not in server.stderr


def test_serve_subprocess_import_has_no_server_side_effects() -> None:
    process = subprocess.run(
        [sys.executable, "-c", "import paper_search.cli; print('ok')"],
        cwd=Path.cwd(),
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert process.returncode == 0
    assert process.stdout == "ok\n"
    assert process.stderr == ""
