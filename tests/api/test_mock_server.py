from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from paper_search.api import mock_server
from paper_search.api.mock_server import create_mock_app, mock_readiness


def _child_environment() -> dict[str, str]:
    environment = {
        "PYTHONPATH": str(Path("src").resolve()),
        "PYTHONUTF8": "1",
    }
    for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
        if value := os.environ.get(name):
            environment[name] = value
    return environment


async def _request(application: object, method: str, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path)


def test_mock_readiness_is_fixed_and_complete() -> None:
    assert mock_readiness() == {
        "openalex": True,
        "semantic_scholar": True,
    }


def test_create_mock_app_reports_both_fixed_providers_ready() -> None:
    response = asyncio.run(_request(create_mock_app(), "GET", "/health/ready"))

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "providers": {
            "openalex": "ready",
            "semantic_scholar": "ready",
        },
    }


def test_main_starts_only_loopback_uvicorn_in_fresh_process() -> None:
    script = """
from paper_search.api import mock_server
seen = {}
def run(application, **kwargs):
    seen["application"] = application
    seen.update(kwargs)
mock_server.uvicorn.run = run
assert mock_server.main(["--host", "127.0.0.1", "--port", "43123"]) == 0
assert seen["host"] == "127.0.0.1"
assert seen["port"] == 43123
assert seen["access_log"] is False
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=_child_environment(),
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_main_reports_fixed_startup_failure_category_in_fresh_process() -> None:
    script = """
from paper_search.api import mock_server
def run(application, **kwargs):
    raise OSError("private bind detail")
mock_server.uvicorn.run = run
assert mock_server.main(["--host", "127.0.0.1", "--port", "43123"]) == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=_child_environment(),
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr.strip() == "mock server startup failed"


@pytest.mark.parametrize(
    "argv",
    [
        ["--host", "0.0.0.0"],
        ["--host", "localhost"],
        ["--port", "0"],
        ["--port", "65536"],
        ["--out"],
        ["--api-key", "forbidden"],
    ],
)
def test_main_rejects_non_mock_server_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        mock_server.main(argv)


def test_mock_server_module_exposes_help_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "paper_search.api.mock_server", "--help"],
        check=False,
        capture_output=True,
        env=_child_environment(),
        text=True,
    )

    assert result.returncode == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout


def test_loopback_guard_rejects_external_connection_in_fresh_process() -> None:
    script = """
from paper_search.api.mock_server import _install_loopback_only_guard
import socket
_install_loopback_only_guard()
socket.create_connection(("203.0.113.1", 443), timeout=0.01)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=_child_environment(),
        text=True,
    )

    assert result.returncode != 0
    assert "mock server blocks non-loopback network access" in result.stderr


@pytest.mark.parametrize(
    ("event", "arguments"),
    [
        ("socket.bind", "(sock, ('0.0.0.0', 0))"),
        ("socket.sendto", "(sock, b'x', ('0.0.0.0', 9))"),
        ("socket.gethostbyname", "('0.0.0.0',)"),
        ("socket.gethostbyname_ex", "('0.0.0.0',)"),
        ("socket.getaddrinfo", "('localhost', 80)"),
        ("socket.getaddrinfo", "(None, 80)"),
    ],
)
def test_loopback_guard_rejects_non_loopback_audit_events(
    event: str,
    arguments: str,
) -> None:
    script = f"""
import socket
import sys
from paper_search.api.mock_server import _install_loopback_only_guard
_install_loopback_only_guard()
sock = socket.socket()
sys.audit({event!r}, *{arguments})
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=_child_environment(),
        text=True,
    )

    assert result.returncode != 0
    assert "mock server blocks non-loopback network access" in result.stderr


def test_loopback_guard_allows_literal_loopback_getnameinfo() -> None:
    script = """
import socket
import sys
from paper_search.api.mock_server import _install_loopback_only_guard
_install_loopback_only_guard()
sys.audit("socket.getnameinfo", ("127.0.0.1", 80), 0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=_child_environment(),
        text=True,
    )

    assert result.returncode == 0, result.stderr
