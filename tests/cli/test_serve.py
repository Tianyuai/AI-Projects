from __future__ import annotations

import asyncio
import importlib
import os
import runpy
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

import paper_search.cli as cli
import paper_search.application.composition as composition_module
from paper_search.application.composition import CompositionRoot
from paper_search.application.contracts import SearchRequest
from paper_search.application.contracts import SearchExecutionResult, SearchFailure, SearchSuccess
from paper_search.application.locks import CandidateLock, ReplayLock
from paper_search.cli import main
from paper_search.storage.dependency_snapshot import DependencySnapshotManifestV2


def test_serve_parser_requires_bound_replay_inputs() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["serve"])

    assert error.value.code == 2


def test_serve_parser_is_replay_only_and_loopback_by_default() -> None:
    args = cli.build_parser().parse_args(
        [
            "serve",
            "--lock",
            "replay.lock.yaml",
            "--mode",
            "replay",
            "--snapshot-manifest",
            "snapshot-manifest.json",
            "--capture-output-root",
            "runs",
        ]
    )

    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.mode == "replay"

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "serve",
                "--lock",
                "replay.lock.yaml",
                "--mode",
                "live",
                "--snapshot-manifest",
                "snapshot-manifest.json",
                "--capture-output-root",
                "runs",
            ]
        )


def test_serve_import_has_no_runtime_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "paper_search.cli", raising=False)
    monkeypatch.delitem(sys.modules, "uvicorn", raising=False)

    imported = importlib.import_module("paper_search.cli")

    assert "uvicorn" not in sys.modules
    assert imported.build_parser().prog == "paper-search"


def test_serve_startup_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "_run_serve",
        lambda args: (_ for _ in ()).throw(RuntimeError("private startup secret")),
    )

    assert (
        cli.main(
            [
                "serve",
                "--lock",
                "replay.lock.yaml",
                "--mode",
                "replay",
                "--snapshot-manifest",
                "snapshot-manifest.json",
                "--capture-output-root",
                "runs",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "serve failed: startup error\n"


def test_serve_interrupt_returns_130(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(_: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run_serve", interrupt)

    assert (
        cli.main(
            [
                "serve",
                "--lock",
                "replay.lock.yaml",
                "--mode",
                "replay",
                "--snapshot-manifest",
                "snapshot-manifest.json",
                "--capture-output-root",
                "runs",
            ]
        )
        == 130
    )


def test_compose_server_uses_replay_bundle_and_canonical_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    replay = _fake_replay_bundle()

    def compose(**kwargs: object) -> object:
        calls.append(kwargs)
        return replay

    (tmp_path / "replay.lock.yaml").write_bytes(b"replay")
    monkeypatch.setattr(
        composition_module,
        "load_verified_input_lock_bytes",
        lambda *_args, **_kwargs: SimpleNamespace(lock=_replay_lock(runtime_allow_live=True)),
    )
    monkeypatch.setattr(CompositionRoot, "compose", classmethod(lambda cls, **kwargs: compose(**kwargs)))

    bundle = CompositionRoot.compose_server(
        replay_lock_path=tmp_path / "replay.lock.yaml",
        snapshot_manifest_path=tmp_path / "snapshot-manifest.json",
        artifact_root=tmp_path,
        capture_output_root=tmp_path / "captures",
        live_authorized=False,
        environ={},
    )

    assert calls[0]["mode"] == "replay"
    assert bundle.live_factory is None
    assert bundle.service_router._replay_service is bundle.replay.service  # noqa: SLF001


def test_server_uses_one_verified_replay_lock_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "replay.lock.yaml"
    original = b"verified replay lock"
    replay_path.write_bytes(original)
    replay = _fake_replay_bundle()
    composed: list[dict[str, object]] = []

    def load_bytes(payload: bytes, **_: object) -> object:
        assert payload == original
        replay_path.write_bytes(b"replacement replay lock")
        return SimpleNamespace(lock=_replay_lock(runtime_allow_live=True))

    monkeypatch.setattr(composition_module, "load_verified_input_lock_bytes", load_bytes)
    monkeypatch.setattr(
        composition_module,
        "load_verified_input_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("reread lock")),
    )
    monkeypatch.setattr(
        CompositionRoot,
        "compose",
        classmethod(lambda cls, **kwargs: composed.append(kwargs) or replay),
    )

    CompositionRoot.compose_server(
        replay_lock_path=replay_path,
        snapshot_manifest_path=tmp_path / "snapshot-manifest.json",
        artifact_root=tmp_path,
        capture_output_root=tmp_path / "captures",
        live_authorized=False,
        environ={},
    )

    assert composed[0]["lock_bytes"] == original


def test_server_bundle_cleanup_is_idempotent_and_closes_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    class Replay:
        service = object()
        readiness_probe = staticmethod(lambda: object())
        artifact_factory = object()

        async def aclose(self) -> None:
            closed.append("replay")

    monkeypatch.setattr(
        CompositionRoot,
        "compose",
        classmethod(lambda cls, **kwargs: Replay()),
    )
    (tmp_path / "replay.lock.yaml").write_bytes(b"replay")
    monkeypatch.setattr(
        composition_module,
        "load_verified_input_lock_bytes",
        lambda *_args, **_kwargs: SimpleNamespace(lock=_replay_lock(runtime_allow_live=True)),
    )
    bundle = CompositionRoot.compose_server(
        replay_lock_path=tmp_path / "replay.lock.yaml",
        snapshot_manifest_path=tmp_path / "snapshot-manifest.json",
        artifact_root=tmp_path,
        capture_output_root=tmp_path / "captures",
        live_authorized=False,
        environ={},
    )

    asyncio.run(bundle.aclose())
    asyncio.run(bundle.aclose())

    assert closed == ["replay"]


def test_server_live_factory_returns_an_isolated_bundle_for_each_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _fake_replay_bundle()
    live_bundles = [object(), object()]
    replay_lock = _replay_lock(runtime_allow_live=True)
    source_lock = tmp_path / "captures" / replay_lock.source_capture_run_id / "config.lock.yaml"
    source_lock.parent.mkdir(parents=True)
    source_lock.write_bytes(Path("tests/fixtures/application/candidate.lock.yaml").read_bytes())
    (source_lock.parent / "replay.lock.yaml").write_bytes(b"operator-replay-lock")
    (tmp_path / "replay.lock.yaml").write_bytes(b"operator-replay-lock")

    def compose(**kwargs: object) -> object:
        if kwargs["mode"] == "replay":
            return replay
        return live_bundles.pop(0)

    load_calls = 0

    def load_bytes(*_args: object, **_kwargs: object) -> object:
        nonlocal load_calls
        load_calls += 1
        source = _candidate_lock().model_copy(
            update={
                field: getattr(replay_lock, field)
                for field in (
                    "schema_version", "source_git_sha", "runtime_allow_live", "frozen_data",
                    "baseline", "budget_config", "pricing_policy", "quality_gates",
                    "capture_policy", "project_ledger",
                )
            }
        )
        return SimpleNamespace(
            lock=(replay_lock if load_calls == 1 else source)
        )

    monkeypatch.setattr(composition_module, "load_verified_input_lock_bytes", load_bytes)
    monkeypatch.setattr(
        CompositionRoot,
        "compose",
        classmethod(lambda cls, **kwargs: compose(**kwargs)),
    )

    bundle = CompositionRoot.compose_server(
        replay_lock_path=tmp_path / "replay.lock.yaml",
        snapshot_manifest_path=tmp_path / "snapshot-manifest.json",
        artifact_root=tmp_path,
        capture_output_root=tmp_path / "captures",
        live_authorized=True,
        environ={"LLM_API_KEY": "not-output"},
    )

    assert bundle.live_factory is not None
    assert bundle.live_factory() is not bundle.live_factory()
    assert bundle.capture_artifact_factory is not bundle.replay.artifact_factory


def test_server_rejects_allow_live_when_verified_replay_lock_forbids_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_lock = _replay_lock(runtime_allow_live=False)
    compose_calls: list[object] = []
    (tmp_path / "replay.lock.yaml").write_bytes(b"replay")
    monkeypatch.setattr(
        composition_module,
        "load_verified_input_lock_bytes",
        lambda *_args, **_kwargs: SimpleNamespace(lock=replay_lock),
    )
    monkeypatch.setattr(
        CompositionRoot,
        "compose",
        classmethod(lambda cls, **kwargs: compose_calls.append(kwargs)),
    )

    with pytest.raises(ValueError, match="allow live"):
        CompositionRoot.compose_server(
            replay_lock_path=tmp_path / "replay.lock.yaml",
            snapshot_manifest_path=tmp_path / "snapshot-manifest.json",
            artifact_root=tmp_path,
            capture_output_root=tmp_path / "captures",
            live_authorized=True,
            environ={},
        )

    assert compose_calls == []


def test_server_rejects_source_capture_run_id_that_escapes_capture_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_lock = _replay_lock(runtime_allow_live=True).model_copy(
        update={"source_capture_run_id": "../outside"}
    )
    (tmp_path / "replay.lock.yaml").write_bytes(b"replay")
    monkeypatch.setattr(
        composition_module,
        "load_verified_input_lock_bytes",
        lambda *_args, **_kwargs: SimpleNamespace(lock=replay_lock),
    )

    with pytest.raises(ValueError, match="source capture run id"):
        CompositionRoot.compose_server(
            replay_lock_path=tmp_path / "replay.lock.yaml",
            snapshot_manifest_path=tmp_path / "snapshot-manifest.json",
            artifact_root=tmp_path,
            capture_output_root=tmp_path / "captures",
            live_authorized=True,
            environ={},
        )


def test_server_rejects_live_source_without_matching_replay_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_lock = _replay_lock(runtime_allow_live=True)
    source_root = tmp_path / "captures" / replay_lock.source_capture_run_id
    source_root.mkdir(parents=True)
    (tmp_path / "replay.lock.yaml").write_bytes(b"operator-replay-lock")
    (source_root / "replay.lock.yaml").write_bytes(b"other-replay-lock")
    (source_root / "config.lock.yaml").write_bytes(
        Path("tests/fixtures/application/candidate.lock.yaml").read_bytes()
    )
    monkeypatch.setattr(
        composition_module,
        "load_verified_input_lock_bytes",
        lambda *_args, **_kwargs: SimpleNamespace(lock=replay_lock),
    )

    with pytest.raises(ValueError, match="source capture lineage"):
        CompositionRoot.compose_server(
            replay_lock_path=tmp_path / "replay.lock.yaml",
            snapshot_manifest_path=tmp_path / "snapshot-manifest.json",
            artifact_root=tmp_path,
            capture_output_root=tmp_path / "captures",
            live_authorized=True,
            environ={},
        )


def test_server_rejects_source_config_with_mismatched_common_lock_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_lock = _replay_lock(runtime_allow_live=True)
    source_root = tmp_path / "captures" / replay_lock.source_capture_run_id
    source_root.mkdir(parents=True)
    (tmp_path / "replay.lock.yaml").write_bytes(b"same-replay-lock")
    (source_root / "replay.lock.yaml").write_bytes(b"same-replay-lock")
    (source_root / "config.lock.yaml").write_bytes(b"source-live-lock")
    source_lock = _candidate_lock().model_copy(update={"source_git_sha": "different"})
    calls = 0

    def load_bytes(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return SimpleNamespace(lock=replay_lock if calls == 1 else source_lock)

    monkeypatch.setattr(composition_module, "load_verified_input_lock_bytes", load_bytes)

    with pytest.raises(ValueError, match="does not match replay lineage"):
        CompositionRoot.compose_server(
            replay_lock_path=tmp_path / "replay.lock.yaml",
            snapshot_manifest_path=tmp_path / "snapshot-manifest.json",
            artifact_root=tmp_path,
            capture_output_root=tmp_path / "captures",
            live_authorized=True,
            environ={},
        )


@pytest.mark.parametrize("filename", ["config.lock.yaml", "replay.lock.yaml"])
def test_server_rejects_source_lock_symlink_escape(
    filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_lock = _replay_lock(runtime_allow_live=True)
    source_root = tmp_path / "captures" / replay_lock.source_capture_run_id
    source_root.mkdir(parents=True)
    outside = tmp_path / f"outside-{filename}"
    outside.write_bytes(b"outside")
    (tmp_path / "replay.lock.yaml").write_bytes(b"operator-replay")
    safe = source_root / ("replay.lock.yaml" if filename == "config.lock.yaml" else "config.lock.yaml")
    safe.write_bytes(b"operator-replay" if safe.name == "replay.lock.yaml" else b"source")
    try:
        os.symlink(outside, source_root / filename)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlinks unavailable: {error}")
    monkeypatch.setattr(
        composition_module,
        "load_verified_input_lock_bytes",
        lambda *_args, **_kwargs: SimpleNamespace(lock=replay_lock),
    )

    with pytest.raises(ValueError, match="source live capture lock is unavailable"):
        CompositionRoot.compose_server(
            replay_lock_path=tmp_path / "replay.lock.yaml",
            snapshot_manifest_path=tmp_path / "snapshot-manifest.json",
            artifact_root=tmp_path,
            capture_output_root=tmp_path / "captures",
            live_authorized=True,
            environ={},
        )


def test_server_rejects_source_lock_replaced_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "config.lock.yaml"
    replacement = tmp_path / "replacement.lock.yaml"
    source.write_bytes(b"original")
    replacement.write_bytes(b"replacement")
    real_open = os.open

    def replace_before_open(path: object, flags: int, *args: object) -> int:
        if Path(path) == source:
            os.replace(replacement, source)
        return real_open(path, flags, *args)

    monkeypatch.setattr(composition_module.os, "open", replace_before_open)

    with pytest.raises(ValueError, match="source live capture lock is unavailable"):
        composition_module._read_confined_source_capture_file(  # noqa: SLF001
            source_root,
            "config.lock.yaml",
        )


def test_server_router_real_live_failure_records_failed_capture_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, server, real_client = _live_server_fixture(tmp_path, monkeypatch)

    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b'{"error":"rejected"}', request=request)

    monkeypatch.setattr(
        composition_module.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(reject), **kwargs),
    )
    monkeypatch.setattr(composition_module, "uuid4", lambda: "failure")
    request = SearchRequest(
        query_id="failure-live",
        query="resource-aware scholarly paper search",
        mode="live",
    )

    try:
        result = asyncio.run(server.service_router.execute(request))
    finally:
        asyncio.run(server.aclose())

    assert isinstance(result.outcome, SearchFailure)
    assert result.outcome.error.code == "dependency_failure"
    assert (fixture["capture_root"] / "serve-failure.failed").is_dir()
    assert not (fixture["capture_root"] / "serve-failure").exists()
    assert not list(fixture["capture_root"].glob(".*.incomplete"))
    assert not server.capture_artifact_factory.has_capture_session(run_id="serve-failure")
    assert server.capture_artifact_factory._clients == []  # noqa: SLF001


def test_server_router_real_live_cancellation_records_failed_capture_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, server, real_client = _live_server_fixture(tmp_path, monkeypatch)
    dispatched = asyncio.Event()

    async def block(request: httpx.Request) -> httpx.Response:
        dispatched.set()
        await asyncio.Event().wait()
        raise AssertionError(f"unexpected completed request: {request.url}")

    monkeypatch.setattr(
        composition_module.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(block), **kwargs),
    )
    monkeypatch.setattr(composition_module, "uuid4", lambda: "cancelled")
    request = SearchRequest(
        query_id="cancel-live",
        query="resource-aware scholarly paper search",
        mode="live",
    )

    async def cancel_after_dispatch() -> None:
        task = asyncio.create_task(server.service_router.execute(request))
        await asyncio.wait_for(dispatched.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(cancel_after_dispatch())
    finally:
        asyncio.run(server.aclose())

    assert (fixture["capture_root"] / "serve-cancelled.failed").is_dir()
    assert not (fixture["capture_root"] / "serve-cancelled").exists()
    assert not list(fixture["capture_root"].glob(".*.incomplete"))
    assert not server.capture_artifact_factory.has_capture_session(run_id="serve-cancelled")
    assert server.capture_artifact_factory._clients == []  # noqa: SLF001


def test_serve_sigterm_handler_is_registered_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_int, old_term = object(), object()
    handlers: dict[int, object] = {}
    registrations: list[tuple[int, object]] = []
    received: dict[str, object] = {}

    class Bundle:
        service_router = object()

        async def aclose(self) -> None:
            return None

    class Server:
        started = True
        should_exit = False

        def __init__(self, _: object) -> None:
            received["server"] = self

        def run(self) -> None:
            handler = handlers[signal.SIGTERM]
            assert callable(handler)
            handler(signal.SIGTERM, None)

    def get_signal(signum: int) -> object:
        return old_int if signum == signal.SIGINT else old_term

    def set_signal(signum: int, handler: object) -> None:
        registrations.append((signum, handler))
        handlers[signum] = handler

    monkeypatch.setattr(
        CompositionRoot,
        "compose_server",
        classmethod(lambda cls, **kwargs: Bundle()),
    )
    api_app = importlib.import_module("paper_search.api.app")
    monkeypatch.setattr(api_app, "create_app", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli.signal, "getsignal", get_signal)
    monkeypatch.setattr(cli.signal, "signal", set_signal)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(Config=lambda app, **kwargs: received.update(kwargs) or app, Server=Server),
    )

    assert cli._run_serve(  # noqa: SLF001
        SimpleNamespace(
            lock="replay.lock.yaml",
            snapshot_manifest="snapshot-manifest.json",
            capture_output_root="captures",
            allow_live=False,
            host="0.0.0.0",
            port=8000,
        )
    ) == 130

    assert received["host"] == "0.0.0.0"
    assert received["server"].should_exit is True
    assert (signal.SIGINT, old_int) in registrations
    assert (signal.SIGTERM, old_term) in registrations


def test_live_request_records_seals_and_publishes_before_success(
) -> None:
    events: list[str] = []
    request = SearchRequest(query_id="q1", query="offline", mode="live")
    service = composition_module._RequestLiveCaptureService(  # noqa: SLF001
        bundle=_live_bundle(events, _live_success(events)),
        input_lock_bytes=b"candidate-lock",
        release_bundle=lambda _: None,
        run_id_factory=lambda: "live-1",
    )

    result = asyncio.run(service.execute_and_publish(request))

    assert events == ["execute", "record", "seal", "publish", "close"]
    assert isinstance(result.outcome, SearchSuccess)
    assert result.outcome.response.snapshot_set_id == "sealed-snapshot"
    assert result.outcome.response.snapshot_captured_at == datetime(2026, 8, 3, tzinfo=UTC)


@pytest.mark.parametrize("cancelled", [False, True])
def test_live_request_failure_or_cancellation_closes_without_publication(
    cancelled: bool,
) -> None:
    events: list[str] = []
    request = SearchRequest(query_id="q1", query="offline", mode="live")

    async def cancelled_execution(_: SearchRequest, *, run_id: str) -> SearchExecutionResult:
        events.append("execute")
        raise asyncio.CancelledError

    execution = cancelled_execution if cancelled else _live_failure(events)
    service = composition_module._RequestLiveCaptureService(  # noqa: SLF001
        bundle=_live_bundle(events, execution),
        input_lock_bytes=b"candidate-lock",
        release_bundle=lambda _: None,
        run_id_factory=lambda: "live-1",
    )

    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(service.execute_and_publish(request))
    else:
        asyncio.run(service.execute_and_publish(request))

    assert events == (
        ["execute", "fail", "close"]
        if cancelled
        else ["execute", "record", "fail", "close"]
    )


def test_server_router_publishes_each_fake_live_capture_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers = runpy.run_path("tests/integration/test_smoke_cli.py")
    fixture = helpers["_smoke_fixture"](tmp_path)
    fake_client = helpers["_fake_live_client_factory"](fixture)
    monkeypatch.chdir(fixture["root"])
    monkeypatch.setenv("LLM_API_KEY", "fake-live-key")
    monkeypatch.setattr(composition_module.httpx, "AsyncClient", fake_client)
    source_root = fixture["root"] / "server-captures"
    assert main([
        "smoke", "--lock", str(fixture["candidate_lock"]), "--output-root", str(source_root),
        "--mode", "live", "--allow-network",
    ]) == 0
    source = next(source_root.glob("smoke-*"))
    server = CompositionRoot.compose_server(
        replay_lock_path=source / "replay.lock.yaml",
        snapshot_manifest_path=source / "snapshot-manifest.json",
        artifact_root=fixture["root"],
        capture_output_root=source_root,
        live_authorized=True,
        environ={"LLM_API_KEY": "fake-live-key"},
    )

    async def run() -> list[SearchExecutionResult]:
        return await asyncio.gather(
            server.service_router.execute(SearchRequest(query_id="live-a", query="resource-aware scholarly paper search", mode="live")),
            server.service_router.execute(SearchRequest(query_id="live-b", query="resource-aware scholarly paper search", mode="live")),
        )

    results = asyncio.run(run())
    asyncio.run(server.aclose())
    run_ids: set[str] = set()
    for result in results:
        assert isinstance(result.outcome, SearchSuccess)
        response = result.outcome.response
        run_ids.add(response.run_id)
        published = source_root / response.run_id
        manifest = DependencySnapshotManifestV2.model_validate_json(
            (published / "snapshot-manifest.json").read_bytes()
        )
        replay = ReplayLock.model_validate(
            yaml.safe_load((published / "replay.lock.yaml").read_bytes())
        )
        assert published.is_dir()
        assert response.snapshot_set_id == manifest.snapshot_set_id == replay.snapshot_set_id
        assert response.snapshot_captured_at == manifest.sealed_at
        assert not server.capture_artifact_factory.has_capture_session(run_id=response.run_id)
    assert len(run_ids) == 2
    assert not list(source_root.glob(".*.incomplete"))
    assert server.capture_artifact_factory._clients == []  # noqa: SLF001



def _replay_lock(*, runtime_allow_live: bool) -> ReplayLock:
    raw = yaml.safe_load(Path("tests/fixtures/application/replay.lock.yaml").read_bytes())
    raw["runtime_allow_live"] = runtime_allow_live
    return ReplayLock.model_validate(raw)


def _candidate_lock() -> CandidateLock:
    return CandidateLock.model_validate(
        yaml.safe_load(Path("tests/fixtures/application/candidate.lock.yaml").read_bytes())
    )


def _live_server_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Path], object, type[httpx.AsyncClient]]:
    helpers = runpy.run_path("tests/integration/test_smoke_cli.py")
    fixture = helpers["_smoke_fixture"](tmp_path)
    real_client = httpx.AsyncClient
    fake_client = helpers["_fake_live_client_factory"](fixture)
    capture_root = fixture["root"] / "server-captures"
    monkeypatch.chdir(fixture["root"])
    monkeypatch.setenv("LLM_API_KEY", "fake-live-key")
    monkeypatch.setattr(composition_module.httpx, "AsyncClient", fake_client)
    assert main([
        "smoke", "--lock", str(fixture["candidate_lock"]), "--output-root", str(capture_root),
        "--mode", "live", "--allow-network",
    ]) == 0
    source = next(capture_root.glob("smoke-*"))
    server = CompositionRoot.compose_server(
        replay_lock_path=source / "replay.lock.yaml",
        snapshot_manifest_path=source / "snapshot-manifest.json",
        artifact_root=fixture["root"],
        capture_output_root=capture_root,
        live_authorized=True,
        environ={"LLM_API_KEY": "fake-live-key"},
    )
    fixture["capture_root"] = capture_root
    return fixture, server, real_client


def _fake_replay_bundle() -> object:
    return type(
        "ReplayBundle",
        (),
        {
            "service": object(),
            "readiness_probe": staticmethod(lambda: object()),
            "artifact_factory": object(),
            "aclose": staticmethod(lambda: None),
        },
    )()


def _live_success(events: list[str]) -> object:
    async def execute(_: SearchRequest, *, run_id: str) -> SearchExecutionResult:
        events.append("execute")
        return SearchExecutionResult.model_construct(
            outcome=SearchSuccess.model_construct(
                response=SimpleNamespace(
                    model_copy=lambda *, update: SimpleNamespace(**update),
                )
            ),
        )

    return execute


def _live_failure(events: list[str]) -> object:
    async def execute(_: SearchRequest, *, run_id: str) -> SearchExecutionResult:
        events.append("execute")
        return SearchExecutionResult.model_construct(
            outcome=SearchFailure.model_construct(
                error=SimpleNamespace(code="dependency_failure"),
            ),
        )

    return execute


def _live_bundle(events: list[str], execution: object) -> object:
    class Session:
        work_dir = Path.cwd()

        def record_execution(self, _: object) -> None:
            events.append("record")

        def seal(self) -> tuple[object, object]:
            events.append("seal")
            return (
                SimpleNamespace(
                    snapshot_set_id="sealed-snapshot",
                    sealed_at=datetime(2026, 8, 3, tzinfo=UTC),
                ),
                object(),
            )

        def publish(self) -> None:
            events.append("publish")

        def fail(self, _: object) -> None:
            events.append("fail")

    class Factory:
        def start_capture(self, **_: object) -> Session:
            return Session()

    class Service:
        execute = staticmethod(execution)

    async def close() -> None:
        events.append("close")

    return SimpleNamespace(
        artifact_factory=Factory(),
        service=Service(),
        aclose=close,
    )
