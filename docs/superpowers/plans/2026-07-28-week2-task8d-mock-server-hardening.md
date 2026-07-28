# Week 2 Task 8D Mock Server Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a deterministic, loopback-only mock-server CLI and process-level HTTP smoke suite for the existing Week 2 mock API.

**Architecture:** A new `paper_search.api.mock_server` module adds no provider logic. It constructs `create_app` with Task 8C's fixed synthetic service and fixed readiness map, installs a process-local audit-hook network guard, then starts Uvicorn on an explicit loopback address. Unit tests lock the CLI and composition contract; process tests use real loopback sockets, bounded startup/request deadlines, and idempotent child-process cleanup.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, httpx, Pydantic v2, pytest, Ruff, mypy, uv.

## Global Constraints

- Work only in `D:\AI Projects\.worktrees\week2-task8d-mock-server` on branch `codex/week2-task8d-mock-server`.
- Use `apply_patch` for every source, test, specification, plan, and dependency-file edit.
- Add `uvicorn>=0.30,<1` as an explicit runtime dependency and regenerate `uv.lock`; do not rely on a transitive or machine-local Uvicorn installation.
- Run every command through `uv run --no-sync --no-env-file` with `UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'`; never read, print, parse, or copy `.env`.
- The mock server may bind only `127.0.0.1`; reject every other `--host` value before server startup.
- The server must use only `build_synthetic_search_service()` and fixed readiness values for `openalex` and `semantic_scholar`.
- Do not read or write `data/raw/`, gold, dev, frozen split, manifest, annotation-work, real-query, or experiment-artifact paths.
- Do not call an external API, construct a real provider/LLM client, calculate metrics, create predictions, or claim the Week 1 Recall gate or Week 2 quality gate.
- Do not alter `paper_search.api.app:app` or `docs/superpowers/specs/2026-07-15-task2-evaluation-design.md`.
- All subprocess diagnostics are safe aggregates: exit code, timeout category, HTTP status, and a bounded fixed stderr category; never assert or print request/response bodies.
- Before every commit, inspect both unstaged and staged file lists, run `git diff --check`, and scan the scoped diff for credential literals and protected-data paths.

---

## File Structure

- Modify: `pyproject.toml` — declare the explicit Uvicorn runtime dependency.
- Modify: `uv.lock` — lock the declared Uvicorn dependency.
- Create: `src/paper_search/api/mock_server.py` — fixed mock-app composition, loopback-only audit guard, strict CLI parser, and Uvicorn entry point.
- Create: `tests/api/test_mock_server.py` — in-process tests for composition, parser validation, Uvicorn invocation wiring, and guard behavior in fresh child interpreters.
- Create: `tests/integration/test_mock_server_process.py` — real loopback process lifecycle and HTTP smoke coverage.
- Create: `docs/superpowers/specs/2026-07-28-week2-task8d-mock-server-hardening-design.md` — already committed design authority; do not change unless implementation exposes a specific contradiction.
- Create: `docs/superpowers/plans/2026-07-28-week2-task8d-mock-server-hardening.md` — this implementation plan.

## Task 1: Lock the Runtime Dependency and Fixed Mock Composition

**Files:**

- Modify: `pyproject.toml:6-18`
- Modify: `uv.lock`
- Create: `src/paper_search/api/mock_server.py`
- Test: `tests/api/test_mock_server.py`

**Interfaces:**

- Consumes: `paper_search.api.app.create_app(search_service, readiness_probe=...) -> FastAPI`.
- Consumes: `paper_search.evaluation.synthetic_mocks.build_synthetic_search_service() -> MockApiSearchService`.
- Produces: `create_mock_app() -> FastAPI` with fixed injected boundaries.
- Produces: `mock_readiness() -> dict[str, bool]` returning exactly `{"openalex": True, "semantic_scholar": True}`.

- [ ] **Step 1: Add the pinned Uvicorn runtime dependency with a patch**

Apply this edit to `pyproject.toml`:

```toml
dependencies = [
    "faiss-cpu>=1.9,<2",
    "fastapi>=0.115,<1",
    "httpx>=0.28,<1",
    "numpy>=1.26,<3",
    "pydantic>=2.10,<3",
    "pydantic-settings>=2.7,<3",
    "python-dotenv>=1.0,<2",
    "pyyaml>=6,<7",
    "rank-bm25>=0.2.2,<0.3",
    "transformers>=4.48,<5",
    "uvicorn>=0.30,<1",
]
```

Then update the lock file without loading an environment file:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' lock
```

Expected: `pyproject.toml` and `uv.lock` describe a resolvable Uvicorn runtime dependency. If the resolver requires registry access, request the narrowly scoped elevation at that point; never work around it by using an untracked machine-local package.

- [ ] **Step 2: Write the failing fixed-composition tests**

Create `tests/api/test_mock_server.py` with these initial tests:

```python
from __future__ import annotations

import asyncio

import httpx

from paper_search.api.mock_server import create_mock_app, mock_readiness


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
```

- [ ] **Step 3: Run the focused test to verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/api/test_mock_server.py -q
```

Expected: collection fails because `paper_search.api.mock_server` does not exist.

- [ ] **Step 4: Implement only the fixed mock composition**

Create `src/paper_search/api/mock_server.py` with this initial production surface:

```python
"""Loopback-only offline server for the fixed Week 2 synthetic mock stack."""

from __future__ import annotations

from fastapi import FastAPI

from paper_search.api.app import create_app
from paper_search.evaluation.synthetic_mocks import build_synthetic_search_service


def mock_readiness() -> dict[str, bool]:
    """Return the fixed readiness map for the offline mock composition."""
    return {
        "openalex": True,
        "semantic_scholar": True,
    }


def create_mock_app() -> FastAPI:
    """Build the only app composition exposed by the mock-server entry point."""
    return create_app(
        build_synthetic_search_service(),
        readiness_probe=mock_readiness,
    )
```

Do not create a module-level `app`; the default app must remain in
`paper_search.api.app` and stay explicitly degraded without injected
dependencies.

- [ ] **Step 5: Run the focused test to verify GREEN**

Run the command from Step 3.

Expected: 2 passed.

- [ ] **Step 6: Commit the independently testable composition change**

Before staging:

```powershell
git diff --name-only
git diff --cached --name-only
git diff --check
```

Stage only:

```powershell
git add -- pyproject.toml uv.lock src/paper_search/api/mock_server.py tests/api/test_mock_server.py
git diff --cached --name-only
git diff --cached --check
git commit -m "feat: add fixed mock server composition"
```

Expected staged paths: exactly the four paths listed above.

## Task 2: Add the Loopback-Only CLI and Process-Local Network Guard

**Files:**

- Modify: `src/paper_search/api/mock_server.py`
- Modify: `tests/api/test_mock_server.py`

**Interfaces:**

- Consumes: `create_mock_app() -> FastAPI` from Task 1.
- Produces: `_build_parser() -> argparse.ArgumentParser` accepting only `--host` and `--port`.
- Produces: `main(argv: Sequence[str] | None = None) -> int` which validates the loopback host, installs the audit hook, and calls `uvicorn.run` with `access_log=False`.
- Produces: `_install_loopback_only_guard() -> None`, safe to call once per interpreter and rejecting audit events for non-loopback socket connects or name lookups.

- [ ] **Step 1: Extend tests with the desired CLI and Uvicorn wiring**

Append these tests to `tests/api/test_mock_server.py`:

```python
import pytest

from paper_search.api import mock_server


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
```

Add this CLI-presence test before any implementation of `mock_server.main`:

```python
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
```

Add a fresh-interpreter guard test so the permanent audit hook never leaks
into pytest's own process:

```python
import os
import subprocess
import sys
from pathlib import Path


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
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
        text=True,
    )

    assert result.returncode != 0
    assert "mock server blocks non-loopback network access" in result.stderr
```

Add the shared safe child environment above both fresh-interpreter tests:

```python
def _child_environment() -> dict[str, str]:
    environment = {
        "PYTHONPATH": str(Path("src").resolve()),
        "PYTHONUTF8": "1",
    }
    for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
        if value := os.environ.get(name):
            environment[name] = value
    return environment
```

The test's target `203.0.113.1` is documentation-only and must be rejected by
the audit hook before any socket connection is attempted. Running `main()` in
a fresh interpreter prevents the permanent audit hook from affecting pytest's
own process.

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/api/test_mock_server.py -q
```

Expected: the new CLI-presence test fails because the module is absent; the
other new tests also fail because `main`, `_build_parser`, and
`_install_loopback_only_guard` are absent.

- [ ] **Step 3: Implement strict parser, audit guard, and entry point**

Extend `src/paper_search/api/mock_server.py` with these exact behaviors:

```python
import argparse
import ipaddress
import sys
from collections.abc import Sequence
from typing import Any

import uvicorn


_LOOPBACK_HOST = "127.0.0.1"
_NETWORK_ERROR = "mock server blocks non-loopback network access"


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _loopback_host(value: str) -> str:
    if value != _LOOPBACK_HOST:
        raise argparse.ArgumentTypeError("host must be 127.0.0.1")
    return value


def _is_loopback_target(value: object) -> bool:
    if not isinstance(value, tuple) or not value or not isinstance(value[0], str):
        return False
    try:
        return ipaddress.ip_address(value[0]).is_loopback
    except ValueError:
        return False


def _audit_network(event: str, args: tuple[Any, ...]) -> None:
    if event == "socket.connect" and len(args) == 2 and not _is_loopback_target(args[1]):
        raise RuntimeError(_NETWORK_ERROR)
    if event == "socket.getaddrinfo" and args and args[0] not in {None, _LOOPBACK_HOST}:
        raise RuntimeError(_NETWORK_ERROR)


def _install_loopback_only_guard() -> None:
    sys.addaudithook(_audit_network)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed offline Week 2 mock API",
        allow_abbrev=False,
    )
    parser.add_argument("--host", type=_loopback_host, default=_LOOPBACK_HOST)
    parser.add_argument("--port", type=_port, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _install_loopback_only_guard()
    uvicorn.run(create_mock_app(), host=args.host, port=args.port, access_log=False)
    return 0
```

Do not add command-line inputs for provider endpoints, configuration, API keys,
input files, splits, metrics, or concurrency. Do not catch the audit hook's
runtime error inside the search route; an attempted external connection must
make the mock-server child fail.

- [ ] **Step 4: Run unit tests to verify GREEN**

Run the command from Step 2.

Expected: all `tests/api/test_mock_server.py` tests pass without warnings.

- [ ] **Step 5: Run type and formatting checks for the new module**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/api/mock_server.py tests/api/test_mock_server.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/api/mock_server.py
```

Expected: both commands exit 0.

- [ ] **Step 6: Commit the CLI and network-boundary contract**

Run the pre-stage and staged checks from Task 1 Step 6, then stage only:

```powershell
git add -- src/paper_search/api/mock_server.py tests/api/test_mock_server.py
git diff --cached --name-only
git diff --cached --check
git commit -m "feat: harden mock server entry point"
```

Expected staged paths: exactly the two paths listed above.

## Task 3: Add Real Loopback Process Smoke Coverage

**Files:**

- Create: `tests/integration/test_mock_server_process.py`

**Interfaces:**

- Consumes: `python -m paper_search.api.mock_server --host 127.0.0.1 --port PORT` from Task 2.
- Produces: test-only `MockServerProcess` helper that starts, polls, terminates, and reaps a child process.
- Produces: test-only `reserve_loopback_port() -> int` and `wait_until_live(base_url: str, process: subprocess.Popen[str]) -> None` helpers with bounded timeouts.

- [ ] **Step 1: Add the real-process success-path test after Task 2 GREEN**

Create `tests/integration/test_mock_server_process.py` with a minimal process fixture and this test:

```python
from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
import os
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


@contextmanager
def mock_server_process(port: int) -> Iterator[subprocess.Popen[str]]:
    process = subprocess.Popen(
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
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def wait_until_live(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    with httpx.Client(timeout=httpx.Timeout(0.2)) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(f"mock server exited before readiness: {process.returncode}")
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

    with mock_server_process(port) as process:
        wait_until_live(base_url, process)
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
```

- [ ] **Step 2: Run the process test to establish the GREEN integration baseline**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/integration/test_mock_server_process.py::test_mock_server_process_serves_ready_and_synthetic_search -q
```

Expected: 1 passed. Task 2 already proves the new production entry point with
the CLI-presence RED→GREEN cycle; this task adds real socket and process
coverage without changing production code.

- [ ] **Step 3: Re-run the test after any harness-only cleanup adjustments**

After the process fixture is complete, rerun the command from Step 2.

Expected: 1 passed; the child binds only its loopback port and is reaped by the fixture.

- [ ] **Step 4: Add failing tests for invalid HTTP and occupied-port behavior**

Add these tests:

```python
def test_mock_server_process_survives_invalid_request_and_unknown_route() -> None:
    port = reserve_loopback_port()
    base_url = f"http://127.0.0.1:{port}"

    with mock_server_process(port) as process:
        wait_until_live(base_url, process)
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
        with mock_server_process(port) as process:
            deadline = time.monotonic() + 5
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)

    assert process.returncode not in {None, 0}
```

- [ ] **Step 5: Run the two tests to verify RED where the harness lacks required diagnostics**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/integration/test_mock_server_process.py -k "invalid_request or occupied_port" -q
```

Expected: if current lifecycle code does not retain bounded safe stderr and startup categories, the occupied-port assertion is too weak or fails nondeterministically. Refine the test helper before adding production behavior; do not change the API route contract.

- [ ] **Step 6: Implement bounded safe subprocess diagnostics in the test harness**

Replace the raw `Popen` fixture with a test-only dataclass that retains no more
than 20 stderr lines and exposes a category-only failure message:

```python
from dataclasses import dataclass


@dataclass
class MockServerProcess:
    process: subprocess.Popen[str]

    def exit_category(self) -> str:
        return "process-exited" if self.process.poll() is not None else "running"

    def terminate_and_reap(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
```

Use `exit_category()` in readiness failures and assert only the category and
return code. Keep Uvicorn access logs disabled; do not include request bodies
or complete stderr in assertion messages.

- [ ] **Step 7: Add and verify bounded request-timeout and guard tests**

Add a test-only delayed local handler launched in a separate Python child
process. Call it with `httpx.Client(timeout=httpx.Timeout(0.01))`, assert
`httpx.ReadTimeout`, then terminate and reap that child with the same helper.
This verifies the harness timeout/cleanup contract without adding a delay knob
to the production mock server.

Add a normal mock-server test that asserts the child completes the live,
ready, and search sequence without a guard-violation category. Keep the
fresh-interpreter external-connect guard test from Task 2 as the direct proof
that non-loopback operations are rejected before connection.

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/integration/test_mock_server_process.py -q
```

Expected: all process tests pass, every child is reaped, and no test reports a
non-loopback guard violation.

- [ ] **Step 8: Commit the independently testable process smoke suite**

Run the pre-stage and staged checks from Task 1 Step 6, then stage only:

```powershell
git add -- tests/integration/test_mock_server_process.py
git diff --cached --name-only
git diff --cached --check
git commit -m "test: smoke test mock server process"
```

Expected staged path: exactly `tests/integration/test_mock_server_process.py`.

## Task 4: Verify the Whole Branch and Document the Safe Boundary

**Files:**

- Modify: `docs/superpowers/specs/2026-07-28-week2-task8d-mock-server-hardening-design.md` only if the implemented behavior contradicts an approved requirement.
- Test: `tests/api/test_mock_server.py`
- Test: `tests/integration/test_mock_server_process.py`

**Interfaces:**

- Consumes: all Task 1–3 code and tests.
- Produces: verified offline-only Task 8D branch with no formal-evaluation claim.

- [ ] **Step 1: Run focused verification**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/api/test_mock_server.py tests/integration/test_mock_server_process.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/api/mock_server.py tests/api/test_mock_server.py tests/integration/test_mock_server_process.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/api/mock_server.py
```

Expected: all commands exit 0 with no warnings.

- [ ] **Step 2: Run full offline verification**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
git diff --check
```

Expected: full suite passes; the explicitly credential-gated live OpenAlex test
may remain skipped; no online test is intentionally run.

- [ ] **Step 3: Perform changed-path and credential scans**

Run:

```powershell
git diff --name-only def3005..HEAD
git diff --check def3005..HEAD
$patch = git diff --no-ext-diff def3005..HEAD
[regex]::Matches(($patch -join "`n"), '(?im)(BEGIN [A-Z ]*PRIVATE KEY|(?:api[_-]?key|secret|password|token)\s*[:=]\s*["''][^"'']+["''])').Count
```

Expected: changed paths contain only Task 8D public source, tests, dependency
metadata, and public design/plan documents; the credential-pattern count is 0.

- [ ] **Step 4: Request an independent read-only review**

Ask the reviewer to inspect `def3005..HEAD` against
`docs/superpowers/specs/2026-07-28-week2-task8d-mock-server-hardening-design.md`.
The review must verify loopback-only binding, no default-app mutation, no
environment or external-provider path, bounded process cleanup, meaningful
timeout coverage, and safe diagnostics. Fix every Critical or Important issue
with a new RED-GREEN cycle before completion.

- [ ] **Step 5: Commit final documentation adjustment only if required**

If Task 4 revealed an actual contradiction in the approved design, update only
the relevant Task 8D design sentence with `apply_patch`, rerun `git diff
--check`, stage that single file, and commit:

```powershell
git add -- docs/superpowers/specs/2026-07-28-week2-task8d-mock-server-hardening-design.md
git diff --cached --name-only
git diff --cached --check
git commit -m "docs: align task 8d server design"
```

If no contradiction exists, do not create an empty documentation commit.
