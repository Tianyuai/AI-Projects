from __future__ import annotations

import asyncio
import json
import runpy
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

import paper_search.application.composition as composition_module
from paper_search.api.app import create_app
from paper_search.application.artifacts import CaptureSession
from paper_search.application.composition import CompositionRoot
from paper_search.application.contracts import SearchRequest
from paper_search.cli import main
from paper_search.storage.dependency_snapshot import DependencySnapshotManifestV2
from tests.integration.test_serve_process import (
    _serve_process,
    _reserve_port,
    _sealed_replay_fixture,
    _wait_ready,
)


QUERY = "resource-aware scholarly paper search"


def _business_projection(payload: dict[str, Any]) -> bytes:
    fields = (
        "selected_paper_ids",
        "fused_papers",
        "high_relevance",
        "partial_relevance",
        "citation_edges",
        "is_partial",
        "planner_status",
        "planner_fallback",
        "warnings",
        "stop_reason",
        "config_hash",
        "snapshot_set_id",
    )
    return json.dumps(
        {name: payload.get(name) for name in fields},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _install_outbound_tripwire(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "offline-tripwire"
    root.mkdir()
    marker = root / "loaded.txt"
    (root / "sitecustomize.py").write_text(
        """import os
import socket
from pathlib import Path

_original_create_connection = socket.create_connection
_original_getaddrinfo = socket.getaddrinfo
_loopback = {"127.0.0.1", "::1", "localhost"}

def _guarded_create_connection(address, *args, **kwargs):
    host = str(address[0])
    if host not in _loopback:
        raise RuntimeError("outbound network denied")
    return _original_create_connection(address, *args, **kwargs)

def _guarded_getaddrinfo(host, *args, **kwargs):
    if host is not None and str(host) not in _loopback:
        raise RuntimeError("outbound name resolution denied")
    return _original_getaddrinfo(host, *args, **kwargs)

socket.create_connection = _guarded_create_connection
socket.getaddrinfo = _guarded_getaddrinfo
Path(os.environ["PAPER_SEARCH_TRIPWIRE_MARKER"]).write_text("loaded", encoding="utf-8")
""",
        encoding="utf-8",
    )
    return root, marker


def test_replay_process_is_offline_and_ui_matches_canonical_api(tmp_path: Path) -> None:
    fixture = _sealed_replay_fixture(tmp_path)
    tripwire_root, marker = _install_outbound_tripwire(tmp_path)
    port = _reserve_port()
    base_url = f"http://127.0.0.1:{port}"

    with _serve_process(
        fixture,
        port,
        extra_pythonpath=(tripwire_root,),
        environment={"PAPER_SEARCH_TRIPWIRE_MARKER": str(marker)},
    ) as server:
        _wait_ready(base_url, server)
        with httpx.Client(timeout=2.0) as client:
            page = client.get(f"{base_url}/")
            script = client.get(f"{base_url}/static/app.js")
            direct = client.post(
                f"{base_url}/v1/search",
                json={
                    "query_id": "direct-replay",
                    "query": QUERY,
                    "mode": "replay",
                    "include_trace": False,
                },
            )
            ui = client.post(
                f"{base_url}/v1/search",
                json={
                    "query_id": "ui-replay",
                    "query": QUERY,
                    "budget_profile": "balanced",
                    "mode": "replay",
                    "include_trace": True,
                },
            )
            repeated = client.post(
                f"{base_url}/v1/search",
                json={
                    "query_id": "repeat-replay",
                    "query": QUERY,
                    "mode": "replay",
                    "include_trace": False,
                },
            )

    assert marker.read_text(encoding="utf-8") == "loaded"
    assert page.status_code == script.status_code == 200
    assert 'id="provenance"' in page.text
    for label in ("Selected paper IDs", "Snapshot set", "Config hash", "Run ID"):
        assert label in script.text
    assert direct.status_code == ui.status_code == repeated.status_code == 200
    assert direct.json()["selected_paper_ids"] == ui.json()["selected_paper_ids"]
    assert _business_projection(direct.json()) == _business_projection(ui.json())
    assert _business_projection(direct.json()) == _business_projection(repeated.json())


def _smoke_helpers() -> dict[str, Any]:
    return runpy.run_path("tests/integration/test_smoke_cli.py")


def _prepare_live_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Path], object, type[httpx.AsyncClient], Path]:
    helpers = _smoke_helpers()
    fixture = helpers["_smoke_fixture"](tmp_path)
    fake_client = helpers["_fake_live_client_factory"](fixture)
    real_client = httpx.AsyncClient
    capture_root = fixture["root"] / "server-captures"
    monkeypatch.chdir(fixture["root"])
    monkeypatch.setenv("LLM_API_KEY", "fake-live-key")
    monkeypatch.setattr(composition_module.httpx, "AsyncClient", fake_client)
    assert main(
        [
            "smoke",
            "--lock",
            str(fixture["candidate_lock"]),
            "--output-root",
            str(capture_root),
            "--mode",
            "live",
            "--allow-network",
        ]
    ) == 0
    source = next(capture_root.glob("smoke-*"))
    server = CompositionRoot.compose_server(
        replay_lock_path=source / "replay.lock.yaml",
        snapshot_manifest_path=source / "snapshot-manifest.json",
        artifact_root=fixture["root"],
        capture_output_root=capture_root,
        live_authorized=True,
        environ={"LLM_API_KEY": "fake-live-key"},
    )
    return fixture, server, real_client, capture_root


async def _post(
    server: object,
    client_type: type[httpx.AsyncClient],
    payload: dict[str, object],
) -> httpx.Response:
    app = create_app(server.service_router)  # type: ignore[attr-defined]
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with client_type(transport=transport, base_url="http://testserver") as client:
        return await client.post("/v1/search", json=payload)


def test_live_authorization_matrix_and_replay_request_isolation(tmp_path: Path) -> None:
    fixture = _sealed_replay_fixture(tmp_path)
    no_live_lock = fixture["root"] / "replay-no-live.lock.yaml"
    raw = yaml.safe_load(fixture["replay_lock"].read_bytes())
    raw["runtime_allow_live"] = False
    no_live_lock.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    async def scenario() -> None:
        for lock_path in (fixture["replay_lock"], no_live_lock):
            server = CompositionRoot.compose_server(
                replay_lock_path=lock_path,
                snapshot_manifest_path=fixture["manifest"],
                artifact_root=fixture["root"],
                capture_output_root=fixture["root"] / "captures",
                live_authorized=False,
                environ={},
            )
            try:
                response = await _post(
                    server,
                    httpx.AsyncClient,
                    {"query_id": "unauthorized", "query": QUERY, "mode": "live"},
                )
                replay = await _post(
                    server,
                    httpx.AsyncClient,
                    {"query_id": "implicit-replay", "query": QUERY},
                )
            finally:
                await server.aclose()
            assert response.status_code == 403
            assert response.json()["code"] == "live_not_authorized"
            assert replay.status_code == 200
            assert replay.json()["execution_mode"] == "replay"

    asyncio.run(scenario())


def test_authorized_fake_live_publishes_one_capture_before_http_200(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = tmp_path_factory.mktemp("live")
    _, server, real_client, capture_root = _prepare_live_server(tmp_path, monkeypatch)
    before = {path.name for path in capture_root.iterdir()}

    async def scenario() -> tuple[httpx.Response, httpx.Response, set[str]]:
        try:
            replay = await _post(
                server,
                real_client,
                {"query_id": "replay-on-live-http", "query": QUERY, "mode": "replay"},
            )
            after_replay = {path.name for path in capture_root.iterdir()}
            live = await _post(
                server,
                real_client,
                {"query_id": "live-e2e", "query": QUERY, "mode": "live"},
            )
            return replay, live, after_replay
        finally:
            await server.aclose()  # type: ignore[attr-defined]

    replay, response, after_replay = asyncio.run(scenario())
    assert replay.status_code == 200
    assert replay.json()["execution_mode"] == "replay"
    assert after_replay == before
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    published = capture_root / run_id
    created = {path.name for path in capture_root.iterdir()} - before
    assert created == {run_id}
    assert json.loads((published / "run.json").read_bytes())["status"] == "complete"
    manifest = DependencySnapshotManifestV2.model_validate_json(
        (published / "snapshot-manifest.json").read_bytes()
    )
    assert response.json()["snapshot_set_id"] == manifest.snapshot_set_id
    assert not list(capture_root.glob(".*.incomplete"))


def test_publication_failure_never_returns_200_or_complete_capture(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = tmp_path_factory.mktemp("live-fail")
    _, server, real_client, capture_root = _prepare_live_server(tmp_path, monkeypatch)
    before = {path.name for path in capture_root.iterdir()}

    def fail_publish(self: CaptureSession) -> Path:
        raise ValueError("synthetic publication failure")

    monkeypatch.setattr(CaptureSession, "publish", fail_publish)

    async def scenario() -> httpx.Response:
        try:
            return await _post(
                server,
                real_client,
                {"query_id": "live-fail", "query": QUERY, "mode": "live"},
            )
        finally:
            await server.aclose()  # type: ignore[attr-defined]

    response = asyncio.run(scenario())
    created = [path for path in capture_root.iterdir() if path.name not in before]
    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert len(created) == 1
    assert created[0].name.endswith(".failed")
    assert json.loads((created[0] / "run.json").read_bytes())["status"] == "failed"
    assert not list(capture_root.glob(".*.incomplete"))


def test_cancelled_live_request_never_publishes_complete_capture(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = tmp_path_factory.mktemp("live-cancel")
    _, server, _, capture_root = _prepare_live_server(tmp_path, monkeypatch)
    before = {path.name for path in capture_root.iterdir()}
    started = asyncio.Event()

    async def blocked_execute(
        self: object,
        request: SearchRequest,
        *,
        run_id: str | None = None,
    ) -> object:
        del self, request, run_id
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(
        "paper_search.application.service.SearchApplicationService.execute",
        blocked_execute,
    )

    async def scenario() -> None:
        task = asyncio.create_task(
            server.service_router.execute(  # type: ignore[attr-defined]
                SearchRequest(query_id="live-cancel", query=QUERY, mode="live")
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await server.aclose()  # type: ignore[attr-defined]

    asyncio.run(scenario())
    created = [path for path in capture_root.iterdir() if path.name not in before]
    assert len(created) == 1
    assert created[0].name.endswith(".failed")
    assert json.loads((created[0] / "run.json").read_bytes())["status"] == "failed"
    assert not list(capture_root.glob(".*.incomplete"))
