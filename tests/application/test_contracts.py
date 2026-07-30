from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from paper_search.application import (
    DependencyDiagnostic,
    DependencyStatus,
    MoneyCny,
    ReadyHealthResponse,
    SafeRelativePath,
    SearchExecutionResult,
    SearchFailure,
    SearchSuccess,
    Sha256,
    SnapshotRef,
    StructuredSearchResponse,
)
from paper_search.domain.models import (
    QueryAnalysisResult,
    QuerySpec,
    SearchPlan,
    SubQuery,
    UsageActual,
)


class PrimitivePayload(BaseModel):
    digest: Sha256
    path: SafeRelativePath
    amount: MoneyCny


def _analysis() -> QueryAnalysisResult:
    return QueryAnalysisResult(
        query_spec=QuerySpec(
            original_query="graph retrieval",
            research_goal="find graph retrieval papers",
        ),
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
            rationale="synthetic",
        ),
    )


def _dependency_statuses() -> list[DependencyStatus]:
    return [
        DependencyStatus(dependency="llm", state="replayed", cache_hit=True, error_codes=[]),
        DependencyStatus(
            dependency="openalex", state="replayed", cache_hit=True, error_codes=[]
        ),
        DependencyStatus(
            dependency="semantic_scholar", state="replayed", cache_hit=True, error_codes=[]
        ),
    ]


def _response(**updates: object) -> StructuredSearchResponse:
    values: dict[str, object] = {
        "run_id": "run-1",
        "query_id": "query-1",
        "execution_mode": "replay",
        "snapshot_set_id": "snapshot-v1",
        "snapshot_captured_at": datetime(2026, 7, 30, tzinfo=UTC),
        "query_analysis": _analysis(),
        "selected_paper_ids": ["openalex:W1"],
        "high_relevance": [],
        "partial_relevance": [],
        "citation_edges": [],
        "search_trace": [{"step": "fuse"}],
        "usage": UsageActual(cost_cny=Decimal("0.123456")),
        "stop_reason": "completed",
        "is_partial": False,
        "planner_fallback": False,
        "planner_status": "primary",
        "dependency_status": _dependency_statuses(),
        "warnings": [],
        "prompt_version": "query-analyze-v1",
        "config_hash": "sha256:" + "a" * 64,
        "git_sha": "abc1234",
    }
    values.update(updates)
    return StructuredSearchResponse.model_validate(values)


def test_primitives_validate_hash_normalize_paths_and_serialize_decimal() -> None:
    payload = PrimitivePayload(
        digest="sha256:" + "a" * 64,
        path="snapshots\\2026-07-30\\entry.json",
        amount=Decimal("0.123456"),
    )

    assert payload.path == "snapshots/2026-07-30/entry.json"
    assert payload.model_dump(mode="json")["amount"] == "0.123456"
    for invalid_hash in ("sha256:" + "A" * 64, "sha256:abc", "md5:" + "a" * 64):
        with pytest.raises(ValidationError):
            PrimitivePayload(digest=invalid_hash, path="snapshots/entry.json", amount=Decimal("0"))
    for unsafe_path in ("../entry.json", "/entry.json", "C:\\entry.json", "snapshots/../entry.json"):
        with pytest.raises(ValidationError):
            PrimitivePayload(
                digest="sha256:" + "a" * 64,
                path=unsafe_path,
                amount=Decimal("0"),
            )


def test_outcomes_are_discriminated_and_diagnostics_are_strict() -> None:
    success = SearchExecutionResult(
        outcome=SearchSuccess(response=_response()),
        diagnostics=[
            DependencyDiagnostic(
                dependency="llm",
                endpoint="/responses",
                model_id="gpt-test",
                usage=UsageActual(),
                latency_ms=1,
                cache_hit=True,
                snapshot_refs=[
                    SnapshotRef(
                        entry_id="entry-1",
                        dependency="llm",
                        cache_key="sha256:" + "b" * 64,
                        response_sha256="sha256:" + "c" * 64,
                        captured_at=datetime(2026, 7, 30, tzinfo=UTC),
                        snapshot_path="snapshots/entry-1.json",
                    )
                ],
                errors=[],
            )
        ],
        business_result_sha256="sha256:" + "d" * 64,
    )
    failure = SearchFailure(
        query_id="query-1",
        run_id="run-1",
        error={
            "code": "dependency_failure",
            "detail": "provider unavailable",
            "retryable": True,
            "run_id": "run-1",
        },
        usage=UsageActual(),
        stop_reason="dependency_failure",
    )

    assert success.outcome.kind == "success"
    assert failure.kind == "failure"
    with pytest.raises(ValidationError):
        SearchExecutionResult.model_validate({"outcome": {"kind": "unknown"}, "diagnostics": []})
    with pytest.raises(ValidationError):
        DependencyDiagnostic.model_validate(
            {
                "dependency": "llm",
                "endpoint": "/responses",
                "model_id": None,
                "usage": {},
                "latency_ms": 0,
                "cache_hit": True,
                "snapshot_refs": [],
                "errors": [],
                "extra": True,
            }
        )


def test_dependency_order_and_planner_invariants_are_enforced() -> None:
    response = _response()
    assert [status.dependency for status in response.dependency_status] == [
        "llm",
        "openalex",
        "semantic_scholar",
    ]
    fallback = _response(
        planner_status="rules_fallback",
        planner_fallback=True,
        is_partial=True,
        warnings=["planner_rules_fallback"],
    )
    assert fallback.planner_fallback is True
    with pytest.raises(ValidationError):
        _response(dependency_status=list(reversed(_dependency_statuses())))
    with pytest.raises(ValidationError):
        _response(planner_status="primary", planner_fallback=True)
    with pytest.raises(ValidationError):
        _response(planner_status="rules_fallback", planner_fallback=False, is_partial=True)
    with pytest.raises(ValidationError):
        _response(planner_status="rules_fallback", planner_fallback=True, is_partial=False)
    with pytest.raises(ValidationError):
        _response(planner_status="rules_fallback", planner_fallback=True, is_partial=True)


def test_ready_health_uses_canonical_dependency_order() -> None:
    response = ReadyHealthResponse(
        status="ready",
        execution_mode="replay",
        snapshot_set_id="snapshot-v1",
        dependencies=_dependency_statuses(),
        last_authorized_probe_at=None,
    )

    assert response.status == "ready"
    with pytest.raises(ValidationError):
        ReadyHealthResponse(
            status="ready",
            execution_mode="replay",
            snapshot_set_id="snapshot-v1",
            dependencies=list(reversed(_dependency_statuses())),
            last_authorized_probe_at=None,
        )
