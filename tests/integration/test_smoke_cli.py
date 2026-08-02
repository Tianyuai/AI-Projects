from __future__ import annotations

import asyncio
import hashlib
import json
import socket
from datetime import UTC, datetime
from pathlib import Path

import pytest
import httpx
import yaml

import paper_search.application.composition as composition_module
from paper_search.application.artifacts import ArtifactFactory, CaptureSession
from paper_search.application.contracts import (
    SearchExecutionResult,
    SearchRequest,
    SearchSuccess,
)
from paper_search.application.locks import CandidateLock, ReplayLock, lock_sha256
from paper_search.cli import build_parser, main
from paper_search.domain.models import StructuredSearchResponse, UsageActual
from paper_search.retrieval.openalex import OPENALEX_SELECT_FIELDS
from paper_search.retrieval.semantic_scholar import _FIELDS
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencyRequestIdentity,
)


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _candidate_lock_bytes(*, runtime_allow_live: bool = True) -> bytes:
    raw = yaml.safe_load(Path("tests/fixtures/application/candidate.lock.yaml").read_bytes())
    raw["runtime_allow_live"] = runtime_allow_live
    return yaml.safe_dump(raw, sort_keys=False).encode("utf-8")


def _write_artifact(root: Path, relative: str, payload: bytes) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256(payload)


def _llm_response(data: dict[str, object]) -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"content": json.dumps(data)}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _smoke_fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "artifact-root"
    root.mkdir()
    payloads = {
        "data/manifest.json": b"{}\n",
        "data/identifier-map.json": b"{}\n",
        "configs/prompts/query_analyze.yaml": b"prompt: smoke fixture\n",
        "configs/budget_balanced.yaml": Path("configs/budget_balanced.yaml").read_bytes(),
        "configs/pricing_v1.yaml": b"""schema_version: pricing-policy-v1
currency: CNY
effective_at: '2026-07-01T00:00:00Z'
source_identity: p2d-fake-live-policy
rounding_quantum_cny: '0.000001'
rates:
  - {dependency: llm, model_or_adapter: qwen3.7-plus, unit: input_token, price_cny_per_unit: '0.000002'}
  - {dependency: llm, model_or_adapter: qwen3.7-plus, unit: output_token, price_cny_per_unit: '0.000003'}
  - {dependency: llm, model_or_adapter: qwen3.7-plus, unit: request, price_cny_per_unit: '0.000100'}
  - {dependency: openalex, model_or_adapter: openalex-works-v1, unit: request, price_cny_per_unit: '0.000050'}
  - {dependency: semantic_scholar, model_or_adapter: semantic-graph-v1, unit: request, price_cny_per_unit: '0.000060'}
""",
        "configs/quality_gates_v1.yaml": b"{}\n",
    }
    hashes = {
        relative: _write_artifact(root, relative, payload)
        for relative, payload in payloads.items()
    }

    store = DependencyCaptureStore(root / "snapshots" / "smoke")
    query = "resource-aware scholarly paper search"
    subqueries = [
        ("sq-1", "smoke openalex one", "openalex"),
        ("sq-2", "smoke openalex two", "openalex"),
        ("sq-3", "smoke semantic", "semantic_scholar"),
    ]
    analysis = {
        "query_spec": {
            "original_query": query,
            "research_goal": "verify the offline smoke path",
        },
        "search_plan": {
            "subqueries": [
                {
                    "query_id": query_id,
                    "text": text,
                    "query_type": "exact",
                    "target_constraints": [],
                    "priority": index,
                    "provider_hint": provider,
                }
                for index, (query_id, text, provider) in enumerate(subqueries, start=1)
            ],
            "inherited_hard_filters": {},
            "rationale": "deterministic smoke fixture",
        },
    }
    llm_identity = DependencyRequestIdentity.from_canonical_request(
        dependency="llm",
        operation="generate_json",
        method="POST",
        endpoint="/chat/completions",
        model_or_adapter="qwen3.7-plus",
        canonical_request={
            "prompt_name": "query_analyze",
            "payload": {"query": query},
            "prompt_version": "query-analyze-v1",
        },
    )
    store.stage_success(
        llm_identity,
        response_bytes=_llm_response(analysis),
        safe_headers={"content-type": "application/json"},
        captured_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    openalex_bytes = json.dumps(
        {
            "meta": {"count": 1, "per_page": 50, "next_cursor": None},
            "results": [
                {
                    "id": "https://openalex.org/W-SMOKE",
                    "title": "Offline Smoke Paper",
                    "display_name": "Offline Smoke Paper",
                    "authorships": [],
                    "publication_year": 2025,
                    "cited_by_count": 1,
                    "is_retracted": False,
                }
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    for _, text, _ in subqueries:
        identity = DependencyRequestIdentity.from_canonical_request(
            dependency="openalex",
            operation="search",
            method="GET",
            endpoint="/works",
            model_or_adapter="openalex-works-v1",
            canonical_request={
                "query": text,
                "filters": {},
                "limit": 50,
                "cursor": "*",
                "per_page": 50,
                "select": OPENALEX_SELECT_FIELDS,
            },
        )
        store.stage_success(
            identity,
            response_bytes=openalex_bytes,
            safe_headers={"content-type": "application/json"},
            captured_at=datetime(2026, 8, 2, tzinfo=UTC),
        )
    semantic_identity = DependencyRequestIdentity.from_canonical_request(
        dependency="semantic_scholar",
        operation="search",
        method="GET",
        endpoint="/paper/search",
        model_or_adapter="semantic-graph-v1",
        canonical_request={
            "query": "smoke semantic",
            "filters": {},
            "limit": 50,
            "fields": _FIELDS,
        },
    )
    store.stage_success(
        semantic_identity,
        response_bytes=Path("tests/fixtures/semantic_scholar/search.json").read_bytes(),
        safe_headers={"content-type": "application/json"},
        captured_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    manifest = store.seal()
    llm_fixture = root / "fake-live-llm.json"
    openalex_fixture = root / "fake-live-openalex.json"
    semantic_fixture = root / "fake-live-semantic.json"
    llm_fixture.write_bytes(_llm_response(analysis))
    openalex_fixture.write_bytes(openalex_bytes)
    semantic_fixture.write_bytes(
        Path("tests/fixtures/semantic_scholar/search.json").read_bytes()
    )

    def write_lock(kind: str, *, runtime_allow_live: bool = True) -> Path:
        raw = yaml.safe_load(Path(f"tests/fixtures/application/{kind}.lock.yaml").read_bytes())
        raw["runtime_allow_live"] = runtime_allow_live
        for section, key, relative in (
            ("frozen_data", "manifest", "data/manifest.json"),
            ("frozen_data", "identifier_map", "data/identifier-map.json"),
        ):
            raw[section][key]["sha256"] = hashes[relative]
        raw["baseline"]["planner"]["prompt_config"]["sha256"] = hashes[
            "configs/prompts/query_analyze.yaml"
        ]
        for key, relative in (
            ("budget_config", "configs/budget_balanced.yaml"),
            ("pricing_policy", "configs/pricing_v1.yaml"),
            ("quality_gates", "configs/quality_gates_v1.yaml"),
        ):
            raw[key]["path"] = relative
            raw[key]["sha256"] = hashes[relative]
        if kind == "replay":
            raw["snapshot_set_id"] = manifest.snapshot_set_id
            raw["snapshot_manifest_sha256"] = store.manifest_sha256
        path = root / f"{kind}-{runtime_allow_live}.lock.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return path

    return {
        "root": root,
        "manifest": store.manifest_path,
        "replay_lock": write_lock("replay"),
        "candidate_lock": write_lock("candidate"),
        "candidate_no_live_lock": write_lock("candidate", runtime_allow_live=False),
        "llm_fixture": llm_fixture,
        "openalex_fixture": openalex_fixture,
        "semantic_fixture": semantic_fixture,
    }


def _fake_live_client_factory(fixture: dict[str, Path]) -> object:
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            payload = fixture["llm_fixture"].read_bytes()
        elif request.url.host == "api.openalex.org":
            payload = fixture["openalex_fixture"].read_bytes()
        elif request.url.host == "api.semanticscholar.org":
            payload = fixture["semantic_fixture"].read_bytes()
        else:
            raise AssertionError(f"unexpected fake-live URL: {request.url}")
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "application/json"},
            request=request,
        )

    def client_factory(**kwargs: object) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    return client_factory


def _success_execution(*, run_id: str = "smoke-live-1") -> SearchExecutionResult:
    input_lock = CandidateLock.model_validate(yaml.safe_load(_candidate_lock_bytes()))
    return SearchExecutionResult.model_construct(
        outcome=SearchSuccess.model_construct(
            response=StructuredSearchResponse.model_construct(
                run_id=run_id,
                query_id="smoke-query-1",
                usage=UsageActual(),
                stop_reason="completed",
                config_hash=lock_sha256(input_lock),
            )
        ),
        diagnostics=[],
        business_result_sha256=_sha256(b"business-result-v1\n"),
    )


def _stage_response(session: object) -> bytes:
    response_bytes = b'{"choices":[]}\n'
    identity = DependencyRequestIdentity.from_canonical_request(
        dependency="llm",
        operation="generate_json",
        method="POST",
        endpoint="/chat/completions",
        model_or_adapter="qwen3.7-plus",
        canonical_request={
            "prompt_name": "query_analyze",
            "payload": {"query": "smoke"},
            "prompt_version": "query-analyze-v1",
        },
    )
    session.snapshot_store.stage_success(  # type: ignore[attr-defined]
        identity,
        response_bytes=response_bytes,
        safe_headers={"content-type": "application/json"},
        captured_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    return response_bytes


def test_parser_exposes_stable_smoke_contract_and_defaults_to_replay(tmp_path: Path) -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["smoke", "--lock", "replay.lock.yaml", "--output-root", str(tmp_path)]
    )

    assert args.command == "smoke"
    assert args.mode == "replay"
    assert args.snapshot_manifest is None
    assert args.allow_network is False


@pytest.mark.parametrize(
    "run_id",
    ["CON", "con.txt", "smoke.", "smoke-", "a" * 65],
)
def test_capture_rejects_windows_hostile_or_oversized_run_ids(
    tmp_path: Path,
    run_id: str,
) -> None:
    with pytest.raises(ValueError, match="run_id is invalid"):
        ArtifactFactory(output_root=tmp_path).start_capture(
            run_id=run_id,
            input_lock_bytes=_candidate_lock_bytes(),
        )


def test_capture_session_seals_manifest_replay_lock_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    lock_bytes = _candidate_lock_bytes(runtime_allow_live=True)
    session = ArtifactFactory(output_root=tmp_path).start_capture(
        run_id="smoke-live-1",
        input_lock_bytes=lock_bytes,
    )
    response_bytes = _stage_response(session)
    session.record_execution(_success_execution())

    manifest, replay_lock = session.seal()
    published = session.publish()

    assert published == tmp_path / "smoke-live-1"
    assert not session.work_dir.exists()
    assert (published / "config.lock.yaml").read_bytes() == lock_bytes
    response_paths = list(published.glob("responses/llm/*.bin"))
    assert response_paths
    assert response_bytes in [path.read_bytes() for path in response_paths]
    assert ReplayLock.model_validate(yaml.safe_load((published / "replay.lock.yaml").read_bytes())) == replay_lock
    assert replay_lock.runtime_allow_live is True
    assert replay_lock.snapshot_set_id == manifest.snapshot_set_id
    assert replay_lock.snapshot_manifest_sha256 == _sha256(
        (published / "snapshot-manifest.json").read_bytes()
    )
    run = json.loads((published / "run.json").read_bytes())
    assert run["status"] == "complete"
    assert run["business_result_sha256"] == _success_execution().business_result_sha256


def test_capture_store_claims_are_bound_to_exact_run_ids_under_concurrency(
    tmp_path: Path,
) -> None:
    factory = ArtifactFactory(output_root=tmp_path)
    first = factory.start_capture(
        run_id="smoke-concurrent-a",
        input_lock_bytes=_candidate_lock_bytes(),
    )
    second = factory.start_capture(
        run_id="smoke-concurrent-b",
        input_lock_bytes=_candidate_lock_bytes(),
    )

    async def claim(run_id: str) -> DependencyCaptureStore:
        await asyncio.sleep(0)
        return factory.start_dependency_capture(run_id=run_id)

    async def claim_both() -> tuple[DependencyCaptureStore, DependencyCaptureStore]:
        return tuple(
            await asyncio.gather(
                claim("smoke-concurrent-a"),
                claim("smoke-concurrent-b"),
            )
        )  # type: ignore[return-value]

    first_store, second_store = asyncio.run(claim_both())

    assert first_store is first.snapshot_store
    assert second_store is second.snapshot_store


def test_failed_capture_is_diagnostic_and_never_emits_replay_lock(tmp_path: Path) -> None:
    session = ArtifactFactory(output_root=tmp_path).start_capture(
        run_id="smoke-failed-1",
        input_lock_bytes=_candidate_lock_bytes(),
    )
    _stage_response(session)

    failed = session.fail("internal_error")

    assert failed == tmp_path / "smoke-failed-1.failed"
    assert not (failed / "replay.lock.yaml").exists()
    run = json.loads((failed / "run.json").read_bytes())
    assert run == {
        "business_result_sha256": None,
        "error_code": "internal_error",
        "run_id": "smoke-failed-1",
        "status": "failed",
    }


def test_capture_refuses_to_publish_tampered_snapshot_bytes(tmp_path: Path) -> None:
    session = ArtifactFactory(output_root=tmp_path).start_capture(
        run_id="smoke-live-corrupt",
        input_lock_bytes=_candidate_lock_bytes(runtime_allow_live=True),
    )
    _stage_response(session)
    session.record_execution(_success_execution(run_id="smoke-live-corrupt"))
    manifest, _ = session.seal()
    response_path = session.work_dir / manifest.entries[0].response_path
    response_path.write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="snapshot response hash mismatch"):
        session.publish()

    assert session.work_dir.is_dir()
    assert not (tmp_path / "smoke-live-corrupt").exists()


@pytest.mark.parametrize(
    ("artifact_name", "tampered"),
    [
        ("execution.json", b"{}\n"),
        ("usage.json", b'{"cost_cny":"999"}\n'),
        ("run.json", b'{"business_result_sha256":null}\n'),
    ],
)
def test_capture_refuses_tampered_canonical_execution_evidence(
    tmp_path: Path,
    artifact_name: str,
    tampered: bytes,
) -> None:
    run_id = "smoke-evidence-corrupt"
    session = ArtifactFactory(output_root=tmp_path).start_capture(
        run_id=run_id,
        input_lock_bytes=_candidate_lock_bytes(),
    )
    _stage_response(session)
    session.record_execution(_success_execution(run_id=run_id))
    session.seal()
    (session.work_dir / artifact_name).write_bytes(tampered)

    with pytest.raises(ValueError, match="captured smoke evidence changed"):
        session.publish()

    assert session.work_dir.is_dir()
    assert not (tmp_path / run_id).exists()


@pytest.mark.parametrize(
    ("binding_path", "replacement"),
    [
        (("source_capture_run_id",), "different-run"),
        (("source_git_sha",), "b" * 40),
        (("runtime_allow_live",), False),
        (("frozen_data", "partition_sha256"), "sha256:" + "1" * 64),
        (
            ("baseline", "planner", "prompt_config", "sha256"),
            "sha256:" + "2" * 64,
        ),
        (("budget_config", "sha256"), "sha256:" + "3" * 64),
        (("pricing_policy", "sha256"), "sha256:" + "4" * 64),
        (("quality_gates", "sha256"), "sha256:" + "5" * 64),
        (("capture_policy", "capture_policy_sha256"), "sha256:" + "6" * 64),
    ],
)
def test_capture_refuses_tampered_replay_lock_input_bindings(
    tmp_path: Path,
    binding_path: tuple[str, ...],
    replacement: object,
) -> None:
    run_id = "smoke-replay-lock-corrupt"
    session = ArtifactFactory(output_root=tmp_path).start_capture(
        run_id=run_id,
        input_lock_bytes=_candidate_lock_bytes(),
    )
    _stage_response(session)
    session.record_execution(_success_execution(run_id=run_id))
    session.seal()
    replay_path = session.work_dir / "replay.lock.yaml"
    replay = yaml.safe_load(replay_path.read_bytes())
    cursor = replay
    for component in binding_path[:-1]:
        cursor = cursor[component]
    cursor[binding_path[-1]] = replacement
    replay_path.write_text(yaml.safe_dump(replay, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="sealed replay lock changed"):
        session.publish()

    assert session.work_dir.is_dir()


def test_failed_capture_removes_replay_lock_emitted_before_failure(tmp_path: Path) -> None:
    run_id = "smoke-failed-after-seal"
    session = ArtifactFactory(output_root=tmp_path).start_capture(
        run_id=run_id,
        input_lock_bytes=_candidate_lock_bytes(),
    )
    _stage_response(session)
    session.record_execution(_success_execution(run_id=run_id))
    session.seal()

    failed = session.fail("internal_error")

    assert not (failed / "replay.lock.yaml").exists()


def test_capture_refuses_coordinated_manifest_and_replay_lock_replacement(
    tmp_path: Path,
) -> None:
    run_id = "smoke-coordinated-replacement"
    session = ArtifactFactory(output_root=tmp_path).start_capture(
        run_id=run_id,
        input_lock_bytes=_candidate_lock_bytes(),
    )
    _stage_response(session)
    session.record_execution(_success_execution(run_id=run_id))
    session.seal()

    replacement_store = DependencyCaptureStore(tmp_path / "replacement-snapshot")
    replacement_identity = DependencyRequestIdentity.from_canonical_request(
        dependency="llm",
        operation="generate_json",
        method="POST",
        endpoint="/chat/completions",
        model_or_adapter="qwen3.7-plus",
        canonical_request={
            "prompt_name": "query_analyze",
            "payload": {"query": "replacement"},
            "prompt_version": "query-analyze-v1",
        },
    )
    replacement_store.stage_success(
        replacement_identity,
        response_bytes=b'{"replacement":true}\n',
        safe_headers={"content-type": "application/json"},
        captured_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    replacement_manifest = replacement_store.seal()
    for entry in replacement_manifest.entries:
        source = replacement_store.root / entry.response_path
        destination = session.work_dir / entry.response_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (session.work_dir / "snapshot-manifest.json").write_bytes(
        replacement_store.manifest_path.read_bytes()
    )
    replay_path = session.work_dir / "replay.lock.yaml"
    replay = yaml.safe_load(replay_path.read_bytes())
    replay["snapshot_set_id"] = replacement_manifest.snapshot_set_id
    replay["snapshot_manifest_sha256"] = replacement_store.manifest_sha256
    replay_path.write_text(yaml.safe_dump(replay, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="sealed replay lock changed"):
        session.publish()


def test_replay_requires_manifest_with_fixed_safe_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "smoke",
            "--lock",
            str(tmp_path / "missing.lock.yaml"),
            "--output-root",
            str(tmp_path / "output"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "smoke failed: snapshot_unavailable\n"


def test_replay_runs_twice_with_zero_network_and_identical_business_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _smoke_fixture(tmp_path)
    attempts: list[str] = []

    def tripwire(*_args: object, **_kwargs: object) -> None:
        attempts.append("network")
        raise AssertionError("replay attempted network access")

    monkeypatch.chdir(fixture["root"])
    monkeypatch.setattr(socket, "create_connection", tripwire)
    monkeypatch.setattr(socket, "getaddrinfo", tripwire)
    output_root = tmp_path / "runs"
    command = [
        "smoke",
        "--lock",
        str(fixture["replay_lock"]),
        "--output-root",
        str(output_root),
        "--snapshot-manifest",
        str(fixture["manifest"]),
    ]

    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(command) == 0
    second = json.loads(capsys.readouterr().out)

    assert attempts == []
    assert first["business_result_sha256"] == second["business_result_sha256"]
    first_bytes = (Path(first["path"]) / "execution.json").read_bytes()
    second_bytes = (Path(second["path"]) / "execution.json").read_bytes()
    assert first_bytes == second_bytes


@pytest.mark.parametrize("allow_network", [False, True])
def test_live_requires_both_cli_and_lock_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    allow_network: bool,
) -> None:
    fixture = _smoke_fixture(tmp_path)
    monkeypatch.chdir(fixture["root"])
    monkeypatch.setenv("LLM_API_KEY", "must-not-appear-in-output")
    lock = (
        fixture["candidate_lock"]
        if not allow_network
        else fixture["candidate_no_live_lock"]
    )
    command = [
        "smoke",
        "--lock",
        str(lock),
        "--output-root",
        str(tmp_path / "runs"),
        "--mode",
        "live",
    ]
    if allow_network:
        command.append("--allow-network")

    assert main(command) == 2

    captured = capsys.readouterr()
    assert "must-not-appear-in-output" not in captured.out + captured.err
    assert captured.out == ""
    assert captured.err.startswith("smoke failed: ")
    assert not (tmp_path / "runs").exists()


def test_fake_live_capture_is_priced_sealed_and_immediately_replayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _smoke_fixture(tmp_path)
    client_factory = _fake_live_client_factory(fixture)
    events: list[str] = []
    real_record_execution = CaptureSession.record_execution
    real_seal = DependencyCaptureStore.seal

    def record_execution(
        self: CaptureSession,
        result: SearchExecutionResult,
    ) -> None:
        events.append("record")
        real_record_execution(self, result)

    def seal(self: DependencyCaptureStore) -> object:
        events.append("seal")
        return real_seal(self)

    monkeypatch.chdir(fixture["root"])
    monkeypatch.setenv("LLM_API_KEY", "fake-live-secret-must-not-leak")
    monkeypatch.setattr(composition_module.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(CaptureSession, "record_execution", record_execution)
    monkeypatch.setattr(DependencyCaptureStore, "seal", seal)
    live_output = fixture["root"] / "live-runs"

    assert (
        main(
            [
                "smoke",
                "--lock",
                str(fixture["candidate_lock"]),
                "--output-root",
                str(live_output),
                "--mode",
                "live",
                "--allow-network",
            ]
        )
        == 0
    )
    live_summary = json.loads(capsys.readouterr().out)
    live_path = Path(live_summary["path"])

    assert events[:2] == ["record", "seal"]
    assert (live_path / "snapshot-manifest.json").is_file()
    assert (live_path / "replay.lock.yaml").is_file()
    usage = json.loads((live_path / "usage.json").read_bytes())
    assert 0 < float(usage["cost_cny"]) <= 0.30
    replay_lock = ReplayLock.model_validate(
        yaml.safe_load((live_path / "replay.lock.yaml").read_bytes())
    )
    assert replay_lock.snapshot_manifest_sha256 == _sha256(
        (live_path / "snapshot-manifest.json").read_bytes()
    )

    assert (
        main(
            [
                "smoke",
                "--lock",
                str(live_path / "replay.lock.yaml"),
                "--output-root",
                str(fixture["root"] / "replay-runs"),
                "--snapshot-manifest",
                str(live_path / "snapshot-manifest.json"),
            ]
        )
        == 0
    )
    replay_summary = json.loads(capsys.readouterr().out)

    assert live_summary["business_result_sha256"] == replay_summary[
        "business_result_sha256"
    ]
    assert "fake-live-secret-must-not-leak" not in (
        json.dumps(live_summary) + json.dumps(replay_summary)
    )


def test_live_smoke_binds_execution_to_the_single_archived_lock_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _smoke_fixture(tmp_path)
    client_factory = _fake_live_client_factory(fixture)
    candidate_path = fixture["candidate_lock"]
    archived_lock_bytes = candidate_path.read_bytes()
    replacement = yaml.safe_load(archived_lock_bytes)
    replacement["source_git_sha"] = "b" * 40
    replacement_lock_bytes = yaml.safe_dump(replacement, sort_keys=False).encode("utf-8")
    original_compose = composition_module.CompositionRoot.compose

    def replace_before_compose(**kwargs: object) -> object:
        candidate_path.write_bytes(replacement_lock_bytes)
        return original_compose(**kwargs)  # type: ignore[arg-type]

    monkeypatch.chdir(fixture["root"])
    monkeypatch.setenv("LLM_API_KEY", "fake-live-secret-must-not-leak")
    monkeypatch.setattr(composition_module.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(
        composition_module.CompositionRoot,
        "compose",
        replace_before_compose,
    )
    output_root = fixture["root"] / "r"

    assert (
        main(
            [
                "smoke",
                "--lock",
                str(candidate_path),
                "--output-root",
                str(output_root),
                "--mode",
                "live",
                "--allow-network",
            ]
        )
        == 0
    )
    run_path = Path(json.loads(capsys.readouterr().out)["path"])
    execution = json.loads((run_path / "execution.json").read_bytes())
    archived_lock = CandidateLock.model_validate(yaml.safe_load(archived_lock_bytes))

    assert (run_path / "config.lock.yaml").read_bytes() == archived_lock_bytes
    assert execution["config_hash"] == lock_sha256(archived_lock)


def test_overlapping_live_executions_keep_capture_artifacts_bound_to_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _smoke_fixture(tmp_path)
    client_factory = _fake_live_client_factory(fixture)
    lock_bytes = fixture["candidate_lock"].read_bytes()
    monkeypatch.chdir(fixture["root"])
    monkeypatch.setattr(composition_module.httpx, "AsyncClient", client_factory)
    bundle = composition_module.CompositionRoot.compose(
        lock_path=fixture["candidate_lock"],
        lock_bytes=lock_bytes,
        mode="live",
        artifact_root=fixture["root"],
        output_root=fixture["root"] / "overlap",
        network_authorized=True,
        environ={"LLM_API_KEY": "fake-live-secret-must-not-leak"},
    )
    run_ids = ("smoke-overlap-a", "smoke-overlap-b")
    sessions = {
        run_id: bundle.artifact_factory.start_capture(
            run_id=run_id,
            input_lock_bytes=lock_bytes,
        )
        for run_id in run_ids
    }

    async def execute(run_id: str) -> tuple[Path, SearchExecutionResult]:
        execution = await bundle.service.execute(
            SearchRequest(
                query_id="smoke-query-1",
                query="resource-aware scholarly paper search",
                mode="live",
            ),
            run_id=run_id,
        )
        session = sessions[run_id]
        session.record_execution(execution)
        session.seal()
        return session.publish(), execution

    async def execute_both() -> list[tuple[Path, SearchExecutionResult]]:
        try:
            return list(await asyncio.gather(*(execute(run_id) for run_id in run_ids)))
        finally:
            await bundle.aclose()

    completed = asyncio.run(execute_both())

    assert all(
        not bundle.artifact_factory.has_capture_session(run_id=run_id)
        for run_id in run_ids
    )
    for expected_run_id, (run_path, execution) in zip(run_ids, completed, strict=True):
        assert isinstance(execution.outcome, SearchSuccess)
        assert execution.outcome.response.run_id == expected_run_id
        replay_lock = ReplayLock.model_validate(
            yaml.safe_load((run_path / "replay.lock.yaml").read_bytes())
        )
        assert replay_lock.source_capture_run_id == expected_run_id
        assert (run_path / "responses" / "llm").is_dir()
