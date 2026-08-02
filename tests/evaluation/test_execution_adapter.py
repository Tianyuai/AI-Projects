from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from paper_search.application.contracts import (
    DependencyDiagnostic,
    SearchErrorResponse,
    SearchExecutionResult,
    SearchFailure,
    SearchSuccess,
    SnapshotRef,
)
from paper_search.domain.models import (
    DependencyStatus,
    ErrorDetail,
    QueryAnalysisResult,
    QuerySpec,
    SearchPlan,
    StructuredSearchResponse,
    SubQuery,
    UsageActual,
)
from paper_search.evaluation.business_results import (
    business_result_from_response,
    business_result_sha256,
)
from paper_search.evaluation.execution_adapter import (
    EvaluationExecutionRecord,
    EvaluationFailureRecord,
    adapt_execution,
)


NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _diagnostic(
    *,
    request_id: str = "safe-request",
    message: str = "safe provider failure",
) -> DependencyDiagnostic:
    return DependencyDiagnostic(
        dependency="openalex",
        endpoint="/works",
        model_id=None,
        usage=UsageActual(search_api_calls=1),
        latency_ms=12,
        cache_hit=True,
        snapshot_refs=[
            SnapshotRef(
                entry_id="entry-1",
                dependency="openalex",
                cache_key="sha256:" + "a" * 64,
                response_sha256="sha256:" + "b" * 64,
                captured_at=NOW,
                snapshot_path="responses/openalex/one.json",
            )
        ],
        errors=[
            ErrorDetail(
                code="server_error",
                message=message,
                retryable=True,
                provider="openalex",
                request_id=request_id,
            )
        ],
    )


def _response(*, partial: bool = False, selected: list[str] | None = None) -> StructuredSearchResponse:
    query_spec = QuerySpec(original_query="graph retrieval", research_goal="find papers")
    return StructuredSearchResponse(
        run_id="run-1",
        query_id="q1",
        execution_mode="replay",
        snapshot_set_id="snapshot-v1",
        snapshot_captured_at=NOW,
        query_analysis=QueryAnalysisResult(
            query_spec=query_spec,
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
        ),
        selected_paper_ids=["openalex:W1"] if selected is None else selected,
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        search_trace=[{"request_id": "transport-only"}],
        usage=UsageActual(search_api_calls=1),
        stop_reason="soft_stop" if partial else "completed",
        is_partial=partial,
        planner_fallback=False,
        planner_status="primary",
        dependency_status=[
            DependencyStatus(dependency="llm", state="replayed", cache_hit=True, error_codes=[]),
            DependencyStatus(dependency="openalex", state="replayed", cache_hit=True, error_codes=[]),
            DependencyStatus(dependency="semantic_scholar", state="replayed", cache_hit=True, error_codes=[]),
        ],
        warnings=["openalex: provider returned errors"] if partial else [],
        prompt_version="prompt-v1",
        config_hash="sha256:" + "c" * 64,
        git_sha="abc1234",
    )


def _success(*, partial: bool = False, selected: list[str] | None = None) -> SearchExecutionResult:
    response = _response(partial=partial, selected=selected)
    return SearchExecutionResult(
        outcome=SearchSuccess(response=response),
        diagnostics=[_diagnostic()],
        business_result_sha256=business_result_sha256(
            business_result_from_response(response)
        ),
    )


@pytest.mark.parametrize(
    ("partial", "selected"),
    [(False, ["openalex:W1"]), (True, ["openalex:W1"]), (False, [])],
)
def test_adapt_execution_success_partial_and_empty_retrieval(
    partial: bool,
    selected: list[str],
) -> None:
    adapted = adapt_execution(expected_query_id="q1", result=_success(partial=partial, selected=selected))

    assert adapted.prediction.query_id == "q1"
    assert adapted.prediction.selected_paper_ids == selected
    assert adapted.execution.outcome_kind == "success"
    assert adapted.execution.is_partial is partial
    assert "safe provider failure" not in adapted.execution.model_dump_json()
    assert "safe-request" not in adapted.execution.model_dump_json()
    assert adapted.failure is None
    assert adapted.business_result.selected_paper_ids == selected


def test_adapt_execution_hard_failure_emits_one_safe_failure_record() -> None:
    diagnostic = _diagnostic()
    result = SearchExecutionResult(
        outcome=SearchFailure(
            query_id="q1",
            run_id="run-failure",
            error=SearchErrorResponse(
                code="dependency_failure",
                detail="A dependency failed",
                retryable=True,
                run_id="run-failure",
            ),
            usage=UsageActual(search_api_calls=1),
            stop_reason="dependency_failure",
        ),
        diagnostics=[diagnostic],
        business_result_sha256=None,
    )

    adapted = adapt_execution(expected_query_id="q1", result=result)

    assert adapted.prediction.selected_paper_ids == []
    assert adapted.execution.outcome_kind == "failure"
    assert adapted.business_result.query_analysis is None
    assert adapted.business_result.hard_failure_code == "dependency_failure"
    assert adapted.failure is not None
    assert adapted.failure.dependency_error_codes == ["server_error"]
    assert adapted.failure.diagnostics[0].snapshot_refs == diagnostic.snapshot_refs
    assert adapted.failure.diagnostics[0].usage == diagnostic.usage
    assert adapted.failure.diagnostics[0].latency_ms == diagnostic.latency_ms
    persisted = adapted.failure.model_dump_json()
    assert "safe provider failure" not in persisted
    assert "safe-request" not in persisted


def test_adapt_execution_rejects_mismatched_query_id() -> None:
    with pytest.raises(ValueError, match="query_id mismatch"):
        adapt_execution(expected_query_id="different", result=_success())


def test_failure_diagnostic_hash_uses_sanitized_canonical_diagnostics() -> None:
    base = SearchExecutionResult(
        outcome=SearchFailure(
            query_id="q1",
            run_id="run-failure",
            error=SearchErrorResponse(code="dependency_failure", detail="safe", retryable=True, run_id="run-failure"),
            usage=UsageActual(),
            stop_reason="dependency_failure",
        ),
        diagnostics=[_diagnostic(request_id="one")],
        business_result_sha256=None,
    )
    changed = base.model_copy(
        update={
            "diagnostics": [
                _diagnostic(request_id="two", message="private=/provider/detail")
            ]
        }
    )

    first = adapt_execution(expected_query_id="q1", result=base).failure
    second = adapt_execution(expected_query_id="q1", result=changed).failure

    assert first is not None and second is not None
    assert first.diagnostics_sha256 == second.diagnostics_sha256
    assert "private=/provider/detail" not in second.model_dump_json()
    assert '"request_id":null' in second.model_dump_json()

    latency_changed = base.model_copy(
        update={
            "diagnostics": [
                base.diagnostics[0].model_copy(update={"latency_ms": 13})
            ]
        }
    )
    third = adapt_execution(expected_query_id="q1", result=latency_changed).failure
    assert third is not None
    assert third.diagnostics_sha256 != first.diagnostics_sha256


def test_execution_models_reject_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvaluationExecutionRecord.model_validate(
            {
                "schema_version": "evaluation-execution-v1",
                "query_id": "q1",
                "run_id": "run-1",
                "outcome_kind": "success",
                "business_result_sha256": "sha256:" + "a" * 64,
                "usage": {},
                "diagnostics": [],
                "is_partial": False,
                "planner_status": "primary",
                "planner_fallback": False,
                "stop_reason": "completed",
                "unexpected": True,
            }
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvaluationFailureRecord.model_validate(
            {
                "schema_version": "evaluation-failure-v1",
                "query_id": "q1",
                "run_id": "run-1",
                "error_code": "dependency_failure",
                "retryable": True,
                "stop_reason": "dependency_failure",
                "usage": {},
                "dependency_error_codes": [],
                "diagnostics": [],
                "diagnostics_sha256": "sha256:" + "a" * 64,
                "unexpected": True,
            }
        )
