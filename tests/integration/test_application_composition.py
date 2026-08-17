from __future__ import annotations

import asyncio
import hashlib
import socket
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from pydantic import ValidationError

import paper_search.application.service as service_module
import paper_search.application.composition as composition_module
from paper_search.application.composition import ApplicationBundle, CompositionRoot
from paper_search.application.contracts import SearchRequest
from paper_search.application.contracts import SearchSuccess
from paper_search.application.modes import ModeBinding
from paper_search.control.budget import HardBudgetController
from paper_search.config import load_runtime_config
from paper_search.domain.models import (
    QueryAnalysisResult,
    QuerySpec,
    SearchBudget,
    SearchPlan,
    SubQuery,
    UsageActual,
)
from paper_search.llm.snapshot_adapters import ReplayLLMAnalyzer
from paper_search.retrieval.snapshot_adapters import ReplaySearchProvider
from paper_search.pipeline.orchestrator import OrchestratorResult
from paper_search.storage.dependency_snapshot import DependencyCaptureStore


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _write_artifact(root: Path, relative: str, payload: bytes) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256(payload)


def _pricing_policy() -> bytes:
    return b"""schema_version: pricing-policy-v1
currency: CNY
effective_at: '2026-07-01T00:00:00Z'
source_identity: composition-test-policy
rounding_quantum_cny: '0.000001'
rates:
  - {dependency: llm, model_or_adapter: deepseek-v4-flash, unit: input_token, price_cny_per_unit: '0.000002'}
  - {dependency: llm, model_or_adapter: deepseek-v4-flash, unit: output_token, price_cny_per_unit: '0.000003'}
  - {dependency: llm, model_or_adapter: deepseek-v4-flash, unit: request, price_cny_per_unit: '0.000100'}
  - {dependency: openalex, model_or_adapter: openalex-works-v1, unit: request, price_cny_per_unit: '0.000050'}
  - {dependency: semantic_scholar, model_or_adapter: semantic-graph-v1, unit: request, price_cny_per_unit: '0.000060'}
"""


def _query_analyze_prompt_artifact() -> bytes:
    return b"""name: query_analyze
version: query-analyze-v1
temperature: 0
response_model: QueryAnalysisResult
instructions:
  - Preserve the original query and every explicit hard constraint.
  - Return one QuerySpec and one SearchPlan as a JSON object.
  - Generate three to five targeted subqueries.
  - Do not infer facts that are not stated by the user.
"""


@pytest.fixture
def composition_fixture(tmp_path: Path) -> Iterator[dict[str, Path]]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    payloads = {
        "data/manifest.json": b"{}",
        "data/identifier-map.json": b"{}",
        "configs/prompts/query_analyze.yaml": _query_analyze_prompt_artifact(),
        "configs/budget_balanced.yaml": (
            Path("configs/budget_balanced.yaml").read_bytes()
        ),
        "configs/budget_low.yaml": Path("configs/budget_low.yaml").read_bytes(),
        "configs/pricing_v1.yaml": _pricing_policy(),
        "configs/quality_gates_v1.yaml": b"{}",
    }
    hashes = {
        relative: _write_artifact(artifact_root, relative, payload)
        for relative, payload in payloads.items()
    }
    snapshot_root = artifact_root / "snapshots" / "fixture"
    store = DependencyCaptureStore(snapshot_root)
    manifest = store.seal()
    manifest_path = store.manifest_path

    def write_lock(
        kind: str,
        *,
        runtime_allow_live: bool = True,
        budget_profile: str = "balanced",
    ) -> Path:
        fixture_path = Path(f"tests/fixtures/application/{kind}.lock.yaml")
        raw = yaml.safe_load(fixture_path.read_bytes())
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
            ("budget_config", f"configs/budget_{budget_profile}.yaml"),
            ("pricing_policy", "configs/pricing_v1.yaml"),
            ("quality_gates", "configs/quality_gates_v1.yaml"),
        ):
            raw[key]["path"] = relative
            raw[key]["sha256"] = hashes[relative]
        if kind == "replay":
            raw["snapshot_set_id"] = manifest.snapshot_set_id
            raw["snapshot_manifest_sha256"] = store.manifest_sha256
        lock_path = tmp_path / f"{kind}-{runtime_allow_live}-{budget_profile}.lock.yaml"
        lock_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return lock_path

    yield {
        "artifact_root": artifact_root,
        "output_root": tmp_path / "output",
        "manifest_path": manifest_path,
        "replay_lock": write_lock("replay"),
        "replay_no_live_lock": write_lock("replay", runtime_allow_live=False),
        "candidate_lock": write_lock("candidate"),
        "candidate_low_lock": write_lock("candidate", budget_profile="low"),
    }


def _compose_replay(fixture: dict[str, Path]) -> ApplicationBundle:
    return CompositionRoot.compose(
        lock_path=fixture["replay_lock"],
        mode="replay",
        artifact_root=fixture["artifact_root"],
        output_root=fixture["output_root"],
        snapshot_manifest_path=fixture["manifest_path"],
        environ={},
    )


def test_prompt_system_message_keeps_query_analyze_output_stable() -> None:
    assert composition_module._prompt_system_message(  # noqa: SLF001
        Path("configs/prompts/query_analyze.yaml").read_bytes()
    ) == "\n".join(
        [
            "Respond with a JSON object.",
            "The JSON object must match the QueryAnalysisResult contract.",
            "- Preserve the original query and every explicit hard constraint.",
            "- Return one QuerySpec and one SearchPlan as a JSON object.",
            "- Generate three to five targeted subqueries.",
            "- Do not infer facts that are not stated by the user.",
        ]
    )


def test_analyzer_bridge_forwards_repair_payload() -> None:
    adapter = AsyncMock()
    bridge = composition_module._AnalyzerBridge(adapter=adapter)  # noqa: SLF001
    reservation = object()

    asyncio.run(
        bridge.repair(
            "graph retrieval",
            "{}",
            reservation,  # type: ignore[arg-type]
        )
    )

    adapter.generate_json.assert_awaited_once_with(
        prompt_name="query_analyze",
        payload={
            "query": "graph retrieval",
            "invalid_analysis": "{}",
        },
        reservation=reservation,
    )


def test_replay_composition_binds_one_snapshot_and_pure_readiness(
    composition_fixture: dict[str, Path],
) -> None:
    bundle = _compose_replay(composition_fixture)

    assert bundle.experiment_id == "main-baseline"
    assert bundle.prompt_version == "query-analyze-v1"
    assert bundle.config_hash.startswith("sha256:")
    assert bundle.mode_binding == ModeBinding(
        mode="replay",
        network_authorized=False,
        snapshot_set_id=bundle.mode_binding.snapshot_set_id,
        snapshot_manifest_sha256=bundle.mode_binding.snapshot_manifest_sha256,
    )
    first = bundle.readiness_probe()
    second = bundle.readiness_probe()
    assert first == second
    assert first.status == "ready"
    assert first.execution_mode == "replay"
    assert first.snapshot_set_id == bundle.mode_binding.snapshot_set_id
    assert [status.state for status in first.dependencies] == [
        "replayed",
        "replayed",
        "replayed",
    ]
    assert all(status.cache_hit for status in first.dependencies)
    assert first.last_authorized_probe_at is None


@pytest.mark.parametrize("manifest_case", ["missing", "mismatched", "outside"])
def test_replay_rejects_invalid_operator_manifest(
    composition_fixture: dict[str, Path],
    tmp_path: Path,
    manifest_case: str,
) -> None:
    manifest_path = composition_fixture["manifest_path"]
    if manifest_case == "missing":
        selected: Path | None = None
    elif manifest_case == "mismatched":
        manifest_path.write_bytes(b"{}")
        selected = manifest_path
    else:
        selected = tmp_path / "outside-snapshot-manifest.json"
        selected.write_bytes(manifest_path.read_bytes())

    with pytest.raises(ValueError, match="manifest"):
        CompositionRoot.compose(
            lock_path=composition_fixture["replay_lock"],
            mode="replay",
            artifact_root=composition_fixture["artifact_root"],
            output_root=composition_fixture["output_root"],
            snapshot_manifest_path=selected,
            environ={},
        )


def test_replay_composition_and_readiness_never_touch_network(
    composition_fixture: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def tripwire(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("replay attempted network access")

    monkeypatch.setattr(socket, "socket", tripwire)
    monkeypatch.setattr(socket, "getaddrinfo", tripwire)

    bundle = _compose_replay(composition_fixture)
    assert bundle.readiness_probe().status == "ready"


@pytest.mark.parametrize(
    ("lock_key", "mode", "network_authorized", "environ", "message"),
    [
        ("candidate_lock", "replay", False, {}, "replay lock"),
        ("replay_no_live_lock", "live", True, {"LLM_API_KEY": "secret"}, "allow live"),
        ("candidate_lock", "live", False, {"LLM_API_KEY": "secret"}, "network"),
        ("candidate_lock", "live", True, {}, "LLM_API_KEY"),
    ],
)
def test_mode_composition_rejects_each_missing_authorization_key(
    composition_fixture: dict[str, Path],
    lock_key: str,
    mode: str,
    network_authorized: bool,
    environ: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CompositionRoot.compose(
            lock_path=composition_fixture[lock_key],
            mode=mode,  # type: ignore[arg-type]
            artifact_root=composition_fixture["artifact_root"],
            output_root=composition_fixture["output_root"],
            network_authorized=network_authorized,
            environ=environ,
        )


def test_live_secret_is_resolved_only_into_private_clients(
    composition_fixture: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "composition-live-secret"
    client_calls: list[dict[str, object]] = []

    class RecordingLLMClient:
        prompt_version = "query-analyze-v1"

    def recording_llm_client(**kwargs: object) -> RecordingLLMClient:
        client_calls.append(kwargs)
        return RecordingLLMClient()

    monkeypatch.setattr(
        composition_module,
        "OpenAICompatibleLLMClient",
        recording_llm_client,
    )
    bundle = CompositionRoot.compose(
        lock_path=composition_fixture["candidate_lock"],
        mode="live",
        artifact_root=composition_fixture["artifact_root"],
        output_root=composition_fixture["output_root"],
        network_authorized=True,
        environ={"LLM_API_KEY": secret},
    )

    assert bundle.mode_binding.network_authorized is True
    assert secret not in repr(bundle)
    assert secret not in repr(bundle.__dict__)
    assert not hasattr(bundle, "environ")
    assert not hasattr(bundle, "llm_api_key")
    assert client_calls == []
    readiness = bundle.readiness_probe()
    assert readiness.status == "degraded"
    assert readiness.last_authorized_probe_at is None


def test_live_readiness_uses_authorized_probe_evidence(
    composition_fixture: dict[str, Path],
) -> None:
    from datetime import UTC, datetime

    from paper_search.application.readiness import (
        AuthorizedCapability,
        AuthorizedReadinessEvidence,
        write_authorized_readiness,
    )

    now = datetime.now(UTC)
    evidence = AuthorizedReadinessEvidence(
        schema_version="gate0-readiness-v1",
        generated_at=now,
        capabilities=[
            AuthorizedCapability(name=name, state="ready", observed_at=now)
            for name in ("llm", "openalex", "semantic_scholar")
        ],
    )
    root = composition_fixture["artifact_root"]
    write_authorized_readiness(
        root / "data" / "annotation_work" / "provider_readiness.live.json",
        evidence,
    )
    bundle = CompositionRoot.compose(
        lock_path=composition_fixture["candidate_lock"],
        mode="live",
        artifact_root=root,
        output_root=composition_fixture["output_root"],
        network_authorized=True,
        environ={"LLM_API_KEY": "secret"},
    )

    response = bundle.readiness_probe()
    assert response.status == "ready"
    assert response.last_authorized_probe_at == now


def test_service_constructs_a_fresh_controller_for_each_request(
    composition_fixture: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controllers: list[HardBudgetController] = []
    real_controller = service_module.HardBudgetController

    def recording_controller(*args: object, **kwargs: object) -> HardBudgetController:
        controller = real_controller(*args, **kwargs)  # type: ignore[arg-type]
        controllers.append(controller)
        return controller

    monkeypatch.setattr(service_module, "HardBudgetController", recording_controller)
    bundle = _compose_replay(composition_fixture)
    request = SearchRequest(query_id="q1", query="offline fixture", mode="replay")

    first = asyncio.run(bundle.service.execute(request))
    second = asyncio.run(
        bundle.service.execute(request.model_copy(update={"query_id": "q2"}))
    )

    assert len(controllers) == 2
    assert controllers[0] is not controllers[1]
    assert first.outcome.kind == "failure"
    assert first.outcome.error.code == "snapshot_unavailable"
    assert second.outcome.kind == "failure"
    assert second.outcome.error.code == "snapshot_unavailable"


def test_replay_orchestrator_uses_only_replay_adapters_and_baseline_modules(
    composition_fixture: dict[str, Path],
) -> None:
    bundle = _compose_replay(composition_fixture)
    budget = SearchBudget.model_validate(
        yaml.safe_load(
            (composition_fixture["artifact_root"] / "configs/budget_balanced.yaml").read_bytes()
        )
    )
    orchestrator = bundle.service._orchestrator_factory(  # noqa: SLF001
        HardBudgetController(budget, formal_live=False),
        "routing-inspection-run",
    )

    analyzer_owner = orchestrator._analyzer.adapter  # type: ignore[attr-defined]  # noqa: SLF001
    assert isinstance(analyzer_owner, ReplayLLMAnalyzer)
    assert all(
        isinstance(provider, ReplaySearchProvider)
        for provider in orchestrator._providers.values()  # noqa: SLF001
    )
    assert orchestrator._routing_limits == (3, 6, 2)  # noqa: SLF001


def test_explicit_main_baseline_uses_lock_identity(
    composition_fixture: dict[str, Path],
) -> None:
    runtime_config = load_runtime_config(
        Path("configs/base.yaml"),
        env_file=None,
    ).model_copy(update={"experiment": "main-baseline"})

    bundle = CompositionRoot.compose(
        lock_path=composition_fixture["replay_lock"],
        mode="replay",
        artifact_root=composition_fixture["artifact_root"],
        output_root=composition_fixture["output_root"],
        snapshot_manifest_path=composition_fixture["manifest_path"],
        environ={},
        runtime_config=runtime_config,
    )

    assert bundle.config_hash == _compose_replay(composition_fixture).config_hash
    assert bundle.experiment_config is None


def test_low_budget_lock_accepts_low_profile_request(
    composition_fixture: dict[str, Path],
) -> None:
    bundle = CompositionRoot.compose(
        lock_path=composition_fixture["candidate_low_lock"],
        mode="live",
        artifact_root=composition_fixture["artifact_root"],
        output_root=composition_fixture["output_root"],
        network_authorized=True,
        environ={"LLM_API_KEY": "fixture-secret"},
    )
    request = SearchRequest(
        query_id="low-1",
        query="offline fixture",
        mode="live",
        budget_profile="low",
    )

    assert "low" in bundle.service._budgets  # noqa: SLF001
    assert "balanced" not in bundle.service._budgets  # noqa: SLF001
    assert request.budget_profile == "low"


def test_provider_secrets_are_confined_to_redacted_private_factory(
    composition_fixture: dict[str, Path],
) -> None:
    secrets = {
        "LLM_API_KEY": "llm-closure-secret",
        "OPENALEX_API_KEY": "openalex-closure-secret",
        "SEMANTIC_SCHOLAR_API_KEY": "s2-closure-secret",
    }
    bundle = CompositionRoot.compose(
        lock_path=composition_fixture["candidate_lock"],
        mode="live",
        artifact_root=composition_fixture["artifact_root"],
        output_root=composition_fixture["output_root"],
        network_authorized=True,
        environ=secrets,
    )
    factory = bundle.service._orchestrator_factory  # noqa: SLF001
    closure = getattr(factory, "__closure__", ()) or ()
    exposed = repr([cell.cell_contents for cell in closure])

    assert all(secret not in repr(factory) for secret in secrets.values())
    assert all(secret not in exposed for secret in secrets.values())


def test_openalex_api_keys_collects_numbered_environment_variables() -> None:
    keys = composition_module._openalex_api_keys(
        {
            "OPENALEX_API_KEY": "key-one",
            "OPENALEX_API_KEY_2": "key-one",
            "OPENALEX_API_KEY_3": "key-three",
            "LLM_API_KEY": "ignored",
        }
    )

    assert [key.get_secret_value() for key in keys] == ["key-one", "key-three"]


def test_openalex_api_keys_stops_at_first_missing_number() -> None:
    keys = composition_module._openalex_api_keys(
        {
            "OPENALEX_API_KEY": "key-one",
            "OPENALEX_API_KEY_3": "key-three",
        }
    )

    assert [key.get_secret_value() for key in keys] == ["key-one"]


def _empty_live_result() -> OrchestratorResult:
    return OrchestratorResult(
        query_analysis=QueryAnalysisResult(
            query_spec=QuerySpec(
                original_query="offline fixture",
                research_goal="offline fixture",
            ),
            search_plan=SearchPlan(
                subqueries=[
                    SubQuery(
                        query_id="sq-1",
                        text="offline fixture",
                        query_type="exact",
                        target_constraints=[],
                        priority=1,
                        provider_hint="either",
                    )
                ],
                inherited_hard_filters={},
                rationale="fixture",
            ),
        ),
        fused_papers=[],
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        provider_results={},
        diagnostics=[],
        planner_status="primary",
        trace=[],
        usage=UsageActual(),
        stop_reason="completed",
        is_partial=False,
        warnings=[],
        config_hash="sha256:" + "4" * 64,
        prompt_version="query-analyze-v1",
    )


def test_live_requests_use_priced_estimates_and_isolated_sealed_resources(
    composition_fixture: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[object] = []
    constructed: list[dict[str, object]] = []

    class RecordingClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.closed = False
            clients.append(self)

        async def aclose(self) -> None:
            self.closed = True

    class RecordingOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)

        async def run(
            self,
            query: str,
            *,
            max_provider_results: int,
        ) -> OrchestratorResult:
            assert query == "offline fixture"
            assert max_provider_results == 50
            return _empty_live_result()

    monkeypatch.setattr(composition_module.httpx, "AsyncClient", RecordingClient)
    monkeypatch.setattr(
        composition_module,
        "MockSearchOrchestrator",
        RecordingOrchestrator,
    )
    bundle = CompositionRoot.compose(
        lock_path=composition_fixture["candidate_lock"],
        mode="live",
        artifact_root=composition_fixture["artifact_root"],
        output_root=composition_fixture["output_root"],
        network_authorized=True,
        environ={"LLM_API_KEY": "fixture-secret"},
    )

    first = asyncio.run(
        bundle.service.execute(
            SearchRequest(query_id="live-1", query="offline fixture", mode="live")
        )
    )
    second = asyncio.run(
        bundle.service.execute(
            SearchRequest(query_id="live-2", query="offline fixture", mode="live")
        )
    )
    asyncio.run(bundle.aclose())

    assert isinstance(first.outcome, SearchSuccess)
    assert isinstance(second.outcome, SearchSuccess)
    assert first.outcome.response.snapshot_set_id.startswith("sha256:")
    assert second.outcome.response.snapshot_set_id.startswith("sha256:")
    manifests = list(
        (composition_fixture["output_root"] / "captures").glob(
            "*/dependency-snapshot/snapshot-manifest.json"
        )
    )
    assert len(manifests) == 2
    assert len(clients) == 2
    assert all(getattr(client, "closed") for client in clients)
    assert len(constructed) == 2
    for kwargs in constructed:
        estimates = kwargs["provider_estimates"]
        assert isinstance(estimates, dict)
        assert estimates["openalex"].cost_cny is not None
        assert estimates["semantic_scholar"].cost_cny is not None
        assert estimates["openalex"].cost_cny != estimates["semantic_scholar"].cost_cny
        assert kwargs["routing_limits"] == (3, 6, 2)


def test_snapshot_binding_is_immutable_after_composition(
    composition_fixture: dict[str, Path],
) -> None:
    bundle = _compose_replay(composition_fixture)
    original = bundle.mode_binding.snapshot_set_id

    with pytest.raises(ValidationError):
        bundle.mode_binding.snapshot_set_id = "replacement"  # type: ignore[misc]

    assert bundle.mode_binding.snapshot_set_id == original
