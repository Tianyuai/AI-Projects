"""Application-layer request, outcome, diagnostic, and readiness contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from paper_search.domain.models import (
    DependencyName,
    DependencyStatus,
    DomainModel,
    ErrorDetail,
    NonEmptyStr,
    NonNegativeInt,
    Paper,
    SafeRelativePath,
    SearchMode,
    Sha256,
    StructuredSearchResponse,
    UsageActual,
    validate_dependency_status_order,
)


SearchErrorCode = Literal[
    "invalid_request",
    "live_not_authorized",
    "config_mismatch",
    "snapshot_unavailable",
    "budget_exhausted",
    "dependency_failure",
    "integrity_failure",
    "validation_attempt_conflict",
    "internal_error",
]


class SearchRequest(DomainModel):
    query_id: NonEmptyStr
    query: NonEmptyStr
    budget_profile: Literal["low", "balanced"] = "balanced"
    include_trace: bool = True
    mode: SearchMode = "replay"


class SnapshotRef(DomainModel):
    entry_id: NonEmptyStr
    dependency: DependencyName
    cache_key: Sha256
    response_sha256: Sha256
    captured_at: datetime
    snapshot_path: SafeRelativePath


class DependencyDiagnostic(DomainModel):
    dependency: DependencyName
    endpoint: NonEmptyStr
    model_id: NonEmptyStr | None
    usage: UsageActual
    latency_ms: NonNegativeInt
    cache_hit: bool
    snapshot_refs: list[SnapshotRef]
    errors: list[ErrorDetail]


class SearchErrorResponse(DomainModel):
    code: SearchErrorCode
    detail: NonEmptyStr
    retryable: bool
    run_id: NonEmptyStr | None


class SearchSuccess(DomainModel):
    kind: Literal["success"] = "success"
    response: StructuredSearchResponse


class SearchFailure(DomainModel):
    kind: Literal["failure"] = "failure"
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    error: SearchErrorResponse
    usage: UsageActual
    stop_reason: NonEmptyStr


SearchOutcome = Annotated[SearchSuccess | SearchFailure, Field(discriminator="kind")]


class SearchExecutionResult(DomainModel):
    outcome: SearchOutcome
    diagnostics: list[DependencyDiagnostic]
    business_result_sha256: Sha256 | None
    retrieved_paper_ids: list[NonEmptyStr] = Field(default_factory=list)
    post_filter_paper_ids: list[NonEmptyStr] = Field(default_factory=list)
    pre_truncation_candidates: list[Paper] = Field(default_factory=list)


class ReadyHealthResponse(DomainModel):
    status: Literal["ready", "degraded"]
    execution_mode: SearchMode
    snapshot_set_id: NonEmptyStr | None
    dependencies: list[DependencyStatus]
    last_authorized_probe_at: datetime | None

    @model_validator(mode="after")
    def validate_dependency_order(self) -> ReadyHealthResponse:
        validate_dependency_status_order(self.dependencies)
        return self
