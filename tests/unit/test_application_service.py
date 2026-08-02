from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from paper_search.application.contracts import (
    DependencyDiagnostic,
    SearchFailure,
    SearchRequest,
    SearchSuccess,
    SnapshotRef,
)
from paper_search.application.service import (
    SearchApplicationError,
    SearchApplicationService,
)
from paper_search.domain.models import (
    ErrorDetail,
    Paper,
    ProviderResult,
    QueryAnalysisResult,
    QuerySpec,
    SearchBudget,
    SearchPlan,
    SubQuery,
    UsageActual,
)
from paper_search.pipeline.orchestrator import OrchestratorResult
from paper_search.ranking.fusion import FusedPaper


NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _budget() -> SearchBudget:
    return SearchBudget(
        max_search_api_calls=12,
        target_search_api_calls=8,
        max_llm_calls=5,
        target_llm_calls=3,
        max_elapsed_seconds=90,
        soft_deadline_seconds=80,
        max_total_tokens=24_000,
        max_cost_cny=0.30,
    )


def _analysis() -> QueryAnalysisResult:
    spec = QuerySpec(
        original_query="graph retrieval",
        research_goal="find graph retrieval papers",
    )
    return QueryAnalysisResult(
        query_spec=spec,
        search_plan=SearchPlan(
            subqueries=[
                SubQuery(
                    query_id="sq-1",
                    text="graph retrieval",
                    query_type="exact",
                    target_constraints=[],
                    priority=1,
                    provider_hint="either",
                )
            ],
            inherited_hard_filters={},
            rationale="fixture",
        ),
    )


def _snapshot_ref(dependency: str, index: int) -> SnapshotRef:
    digest = f"{index:x}" * 64
    return SnapshotRef(
        entry_id=f"entry-{index}",
        dependency=dependency,
        cache_key=f"sha256:{digest[:64]}",
        response_sha256=f"sha256:{digest[:64]}",
        captured_at=NOW,
        snapshot_path=f"responses/{dependency}/{index}.bin",
    )


def _diagnostic(
    dependency: str,
    *,
    errors: list[ErrorDetail] | None = None,
    refs: list[SnapshotRef] | None = None,
) -> DependencyDiagnostic:
    return DependencyDiagnostic(
        dependency=dependency,
        endpoint="/fixture",
        model_id="fixture-v1",
        usage=UsageActual(search_api_calls=1),
        latency_ms=1,
        cache_hit=True,
        snapshot_refs=refs or [],
        errors=errors or [],
    )


def _provider_result(dependency: str, paper: Paper) -> ProviderResult[list[Paper]]:
    return ProviderResult[list[Paper]](
        data=[paper],
        usage=UsageActual(search_api_calls=1),
        provenance={
            "provider": dependency,
            "endpoint": "/fixture",
            "model_id": "fixture-v1",
            "requested_at": NOW.isoformat(),
            "response_hash": "sha256:" + "a" * 64,
        },
        cache_hit=True,
        latency_ms=1,
        errors=[],
    )


def _result(
    *,
    diagnostics: list[DependencyDiagnostic] | None = None,
    planner_status: str = "primary",
    warnings: list[str] | None = None,
    stop_reason: str = "completed",
    papers: list[Paper] | None = None,
) -> OrchestratorResult:
    selected = papers if papers is not None else [
        Paper(
            canonical_id="openalex:W1",
            title="Graph Retrieval",
            openalex_id="W1",
            sources=["openalex"],
        )
    ]
    fused = [
        FusedPaper(
            paper=paper,
            score=1.0 / (60 + index),
            source_ranks={paper.sources[0]: index},
        )
        for index, paper in enumerate(selected, start=1)
    ]
    provider_results = {
        paper.sources[0]: _provider_result(paper.sources[0], paper)
        for paper in selected
    }
    return OrchestratorResult(
        query_analysis=_analysis(),
        fused_papers=fused,
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        provider_results=provider_results,
        diagnostics=diagnostics
        or [
            _diagnostic("llm"),
            _diagnostic("openalex"),
            _diagnostic("semantic_scholar"),
        ],
        planner_status=planner_status,
        trace=[{"step": "fuse", "count": len(fused)}],
        usage=UsageActual(search_api_calls=1, llm_calls=1, cost_cny=Decimal("0.10")),
        stop_reason=stop_reason,
        is_partial=bool(warnings) or planner_status == "rules_fallback",
        warnings=warnings or [],
        config_hash="sha256:" + "c" * 64,
        prompt_version="query-analyze-v1",
    )


class StubOrchestrator:
    def __init__(self, result: OrchestratorResult) -> None:
        self.result = result

    async def run(self, query: str, *, max_provider_results: int) -> OrchestratorResult:
        assert query == "graph retrieval"
        assert max_provider_results == 50
        return self.result


def _service(
    result: OrchestratorResult,
    *,
    controllers: list[object] | None = None,
) -> SearchApplicationService:
    observed = controllers if controllers is not None else []

    def factory(controller: object) -> StubOrchestrator:
        observed.append(controller)
        return StubOrchestrator(result)

    run_ids = iter(["run-1", "run-2", "run-3"])
    return SearchApplicationService(
        orchestrator_factory=factory,
        budgets={"low": _budget(), "balanced": _budget()},
        mode="replay",
        snapshot_set_id="snapshot-set-1",
        snapshot_captured_at=NOW,
        git_sha="abc1234",
        max_provider_results=50,
        run_id_factory=lambda: next(run_ids),
    )


def _request(*, include_trace: bool = True) -> SearchRequest:
    return SearchRequest(
        query_id="query-1",
        query="graph retrieval",
        include_trace=include_trace,
    )


def test_service_creates_fresh_request_scope_and_success_outcome() -> None:
    controllers: list[object] = []
    service = _service(_result(), controllers=controllers)

    first = asyncio.run(service.execute(_request()))
    second = asyncio.run(service.execute(_request()))

    assert isinstance(first.outcome, SearchSuccess)
    assert isinstance(second.outcome, SearchSuccess)
    assert first.outcome.response.run_id == "run-1"
    assert second.outcome.response.run_id == "run-2"
    assert len(controllers) == 2
    assert controllers[0] is not controllers[1]


def test_service_one_provider_degradation_is_partial_success() -> None:
    error = ErrorDetail(
        code="server_error",
        message="provider failed",
        retryable=True,
        provider="openalex",
    )
    result = _result(
        diagnostics=[
            _diagnostic("llm"),
            _diagnostic("openalex", errors=[error]),
            _diagnostic("semantic_scholar"),
        ],
        warnings=["openalex: provider returned errors"],
    )

    execution = asyncio.run(_service(result).execute(_request()))

    assert isinstance(execution.outcome, SearchSuccess)
    assert execution.outcome.response.is_partial is True


def test_service_both_provider_failures_are_hard_failure() -> None:
    provider_errors = [
        ErrorDetail(
            code="server_error",
            message="provider failed",
            retryable=True,
            provider=provider,
        )
        for provider in ("openalex", "semantic_scholar")
    ]
    result = _result(
        papers=[],
        diagnostics=[
            _diagnostic("llm"),
            _diagnostic("openalex", errors=[provider_errors[0]]),
            _diagnostic("semantic_scholar", errors=[provider_errors[1]]),
        ],
        warnings=[
            "openalex: provider returned errors",
            "semantic_scholar: provider returned errors",
        ],
    )

    execution = asyncio.run(_service(result).execute(_request()))

    assert isinstance(execution.outcome, SearchFailure)
    assert execution.outcome.error.code == "dependency_failure"
    assert execution.business_result_sha256 is not None


def test_service_rules_fallback_is_partial_success_with_fixed_warning() -> None:
    result = _result(planner_status="rules_fallback")

    execution = asyncio.run(_service(result).execute(_request()))

    assert isinstance(execution.outcome, SearchSuccess)
    assert execution.outcome.response.planner_fallback is True
    assert execution.outcome.response.is_partial is True
    assert "planner_rules_fallback" in execution.outcome.response.warnings


def test_service_suppresses_trace_without_changing_business_hash() -> None:
    service = _service(_result())

    traced = asyncio.run(service.execute(_request(include_trace=True)))
    hidden = asyncio.run(service.execute(_request(include_trace=False)))

    assert isinstance(traced.outcome, SearchSuccess)
    assert isinstance(hidden.outcome, SearchSuccess)
    assert traced.outcome.response.search_trace == [{"step": "fuse", "count": 1}]
    assert hidden.outcome.response.search_trace == []
    assert traced.business_result_sha256 == hidden.business_result_sha256


def test_service_preserves_all_snapshot_refs_and_suppresses_unsafe_warning() -> None:
    refs = [_snapshot_ref("openalex", 1), _snapshot_ref("openalex", 2)]
    result = _result(
        diagnostics=[
            _diagnostic("llm"),
            _diagnostic("openalex", refs=refs),
            _diagnostic("semantic_scholar"),
        ],
        warnings=["openalex: provider returned errors", "secret=/private/query"],
    )

    execution = asyncio.run(_service(result).execute(_request()))

    assert execution.diagnostics[1].snapshot_refs == refs
    assert isinstance(execution.outcome, SearchSuccess)
    public = execution.outcome.response.model_dump_json()
    assert "secret" not in public
    assert "/private/query" not in public


def test_service_call_raises_typed_application_error_for_failure() -> None:
    error = ErrorDetail(
        code="snapshot_unavailable",
        message="snapshot unavailable",
        retryable=False,
        provider="llm",
    )
    result = _result(
        papers=[],
        diagnostics=[
            _diagnostic("llm", errors=[error]),
            _diagnostic("openalex"),
            _diagnostic("semantic_scholar"),
        ],
        stop_reason="snapshot_unavailable",
    )
    service = _service(result)

    with pytest.raises(SearchApplicationError) as raised:
        asyncio.run(service(_request()))

    assert raised.value.error.code == "snapshot_unavailable"
    assert raised.value.usage == result.usage
    assert raised.value.diagnostics == result.diagnostics


def test_application_package_exports_canonical_service_boundary() -> None:
    import paper_search.application as application

    assert application.SearchApplicationService is SearchApplicationService
    assert application.SearchApplicationError is SearchApplicationError


def test_execute_maps_factory_failure_to_safe_typed_outcome() -> None:
    def broken_factory(controller: object) -> StubOrchestrator:
        del controller
        raise RuntimeError("private-path=/secret/query")

    service = SearchApplicationService(
        orchestrator_factory=broken_factory,
        budgets={"low": _budget(), "balanced": _budget()},
        mode="replay",
        snapshot_set_id="snapshot-set-1",
        snapshot_captured_at=NOW,
        git_sha="abc1234",
        max_provider_results=50,
        run_id_factory=lambda: "run-failed",
    )

    execution = asyncio.run(service.execute(_request()))

    assert isinstance(execution.outcome, SearchFailure)
    assert execution.outcome.error.code == "internal_error"
    assert execution.outcome.error.detail == "The search could not be completed"
    assert "/secret/query" not in execution.model_dump_json()
