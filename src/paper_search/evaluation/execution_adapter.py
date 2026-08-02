"""Adapt application outcomes into strict, auditable evaluation records."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, cast

from pydantic import Field, model_validator

from paper_search.application.contracts import (
    DependencyDiagnostic,
    SearchErrorCode,
    SearchExecutionResult,
    SearchFailure,
)
from paper_search.domain.models import (
    DependencyErrorCode,
    DomainModel,
    ErrorDetail,
    NonEmptyStr,
    PlannerStatus,
    Sha256,
    UsageActual,
)
from paper_search.evaluation.business_results import (
    BusinessResultRecord,
    business_result_from_response,
    business_result_sha256,
    hard_failure_business_result,
)
from paper_search.evaluation.official_adapter import InternalPredictionRecord
from paper_search.evaluation.predictions import prediction_from_response


_DEPENDENCY_ERROR_CODES = frozenset(
    {
        "timeout",
        "network_error",
        "rate_limited",
        "server_error",
        "authentication_error",
        "invalid_request",
        "invalid_response",
        "invalid_record",
        "missing_record",
        "empty_response",
        "invalid_json",
        "budget_exhausted",
        "provider_error",
    }
)
_SAFE_DIAGNOSTIC_ERROR_CODES = _DEPENDENCY_ERROR_CODES | {
    "integrity_failure",
    "snapshot_unavailable",
}


class EvaluationFailureRecord(DomainModel):
    schema_version: Literal["evaluation-failure-v1"] = "evaluation-failure-v1"
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    error_code: SearchErrorCode
    retryable: bool
    stop_reason: NonEmptyStr
    usage: UsageActual
    dependency_error_codes: list[DependencyErrorCode]
    diagnostics: list[DependencyDiagnostic]
    diagnostics_sha256: Sha256


class EvaluationExecutionRecord(DomainModel):
    schema_version: Literal["evaluation-execution-v1"] = "evaluation-execution-v1"
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    outcome_kind: Literal["success", "failure"]
    business_result_sha256: Sha256
    usage: UsageActual
    diagnostics: list[DependencyDiagnostic]
    retrieved_paper_ids: list[NonEmptyStr] = Field(default_factory=list)
    post_filter_paper_ids: list[NonEmptyStr] = Field(default_factory=list)
    is_partial: bool
    planner_status: PlannerStatus | None
    planner_fallback: bool
    stop_reason: NonEmptyStr

    @model_validator(mode="after")
    def validate_filter_evidence(self) -> EvaluationExecutionRecord:
        if not set(self.post_filter_paper_ids) <= set(self.retrieved_paper_ids):
            raise ValueError("post-filter IDs must be a subset of retrieved IDs")
        return self


class AdaptedExecution(DomainModel):
    prediction: InternalPredictionRecord
    execution: EvaluationExecutionRecord
    business_result: BusinessResultRecord
    failure: EvaluationFailureRecord | None


def _diagnostics_sha256(diagnostics: list[DependencyDiagnostic]) -> Sha256:
    encoded = (
        json.dumps(
            [item.model_dump(mode="json") for item in diagnostics],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _sanitized_diagnostics(
    diagnostics: list[DependencyDiagnostic],
) -> list[DependencyDiagnostic]:
    """Retain audit facts while removing provider-controlled text and request IDs."""
    sanitized: list[DependencyDiagnostic] = []
    for diagnostic in diagnostics:
        errors = [
            ErrorDetail(
                code=(
                    error.code
                    if error.code in _SAFE_DIAGNOSTIC_ERROR_CODES
                    else "provider_error"
                ),
                message="Dependency execution reported an error",
                retryable=error.retryable,
                provider=diagnostic.dependency,
                request_id=None,
            )
            for error in diagnostic.errors
        ]
        sanitized.append(
            diagnostic.model_copy(
                update={
                    "endpoint": "dependency",
                    "errors": errors,
                }
            )
        )
    return sanitized


def _validated_business_hash(
    result: SearchExecutionResult,
    record: BusinessResultRecord,
) -> Sha256:
    calculated = business_result_sha256(record)
    if (
        result.business_result_sha256 is not None
        and result.business_result_sha256 != calculated
    ):
        raise ValueError("business_result_sha256 does not match canonical business result")
    return calculated


def adapt_execution(
    *,
    expected_query_id: str,
    result: SearchExecutionResult,
) -> AdaptedExecution:
    """Convert exactly one application outcome without weakening prediction records."""
    outcome = result.outcome
    diagnostics = _sanitized_diagnostics(result.diagnostics)
    if isinstance(outcome, SearchFailure):
        if outcome.query_id != expected_query_id:
            raise ValueError(
                f"query_id mismatch: expected {expected_query_id!r}, got {outcome.query_id!r}"
            )
        business = hard_failure_business_result(
            query_id=outcome.query_id,
            error_code=outcome.error.code,
            stop_reason=outcome.stop_reason,
        )
        digest = _validated_business_hash(result, business)
        dependency_codes = [
            cast(DependencyErrorCode, error.code)
            for diagnostic in diagnostics
            for error in diagnostic.errors
            if error.code in _DEPENDENCY_ERROR_CODES
        ]
        failure = EvaluationFailureRecord(
            query_id=outcome.query_id,
            run_id=outcome.run_id,
            error_code=outcome.error.code,
            retryable=outcome.error.retryable,
            stop_reason=outcome.stop_reason,
            usage=outcome.usage,
            dependency_error_codes=dependency_codes,
            diagnostics=diagnostics,
            diagnostics_sha256=_diagnostics_sha256(diagnostics),
        )
        return AdaptedExecution(
            prediction=InternalPredictionRecord(
                query_id=outcome.query_id,
                selected_paper_ids=[],
            ),
            execution=EvaluationExecutionRecord(
                query_id=outcome.query_id,
                run_id=outcome.run_id,
                outcome_kind="failure",
                business_result_sha256=digest,
                usage=outcome.usage,
                diagnostics=diagnostics,
                retrieved_paper_ids=result.retrieved_paper_ids,
                post_filter_paper_ids=result.post_filter_paper_ids,
                is_partial=False,
                planner_status=None,
                planner_fallback=False,
                stop_reason=outcome.stop_reason,
            ),
            business_result=business,
            failure=failure,
        )

    response = outcome.response
    if response.query_id != expected_query_id:
        raise ValueError(
            f"query_id mismatch: expected {expected_query_id!r}, got {response.query_id!r}"
        )
    business = business_result_from_response(response)
    digest = _validated_business_hash(result, business)
    return AdaptedExecution(
        prediction=prediction_from_response(response),
        execution=EvaluationExecutionRecord(
            query_id=response.query_id,
            run_id=response.run_id,
            outcome_kind="success",
            business_result_sha256=digest,
            usage=response.usage,
            diagnostics=diagnostics,
            retrieved_paper_ids=result.retrieved_paper_ids,
            post_filter_paper_ids=result.post_filter_paper_ids,
            is_partial=response.is_partial,
            planner_status=response.planner_status,
            planner_fallback=response.planner_fallback,
            stop_reason=response.stop_reason,
        ),
        business_result=business,
        failure=None,
    )
