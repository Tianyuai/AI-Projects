"""Pure conversion from orchestration evidence to the public response model."""

from __future__ import annotations

from datetime import datetime
from typing import cast

from paper_search.domain.models import (
    DependencyErrorCode,
    DependencyName,
    DependencyStatus,
    SearchMode,
    RankedPaper,
    StructuredSearchResponse,
)
from paper_search.pipeline.orchestrator import OrchestratorResult


_DEPENDENCIES: tuple[DependencyName, ...] = (
    "llm",
    "openalex",
    "semantic_scholar",
)
_PUBLIC_ERROR_CODES = frozenset(
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


def _dependency_statuses(
    result: OrchestratorResult,
    *,
    execution_mode: SearchMode,
) -> list[DependencyStatus]:
    statuses: list[DependencyStatus] = []
    for dependency in _DEPENDENCIES:
        matching = [
            item for item in result.diagnostics if item.dependency == dependency
        ]
        errors = [error for item in matching for error in item.errors]
        codes = [
            cast(
                DependencyErrorCode,
                error.code if error.code in _PUBLIC_ERROR_CODES else "provider_error",
            )
            for error in errors
        ]
        if errors:
            state = "degraded"
        elif matching:
            state = "replayed" if execution_mode == "replay" else "ready"
        else:
            state = "degraded"
        statuses.append(
            DependencyStatus(
                dependency=dependency,
                state=state,
                cache_hit=bool(matching) and all(item.cache_hit for item in matching),
                error_codes=codes,
            )
        )
    return statuses


def to_structured_response(
    result: OrchestratorResult,
    *,
    query_id: str,
    git_sha: str,
    run_id: str = "mock-run-1",
    execution_mode: SearchMode = "replay",
    snapshot_set_id: str = "mock-snapshot-v1",
    snapshot_captured_at: datetime | None = None,
    include_trace: bool = True,
) -> StructuredSearchResponse:
    """Preserve ranking and dependency evidence without inventing it."""

    warnings = list(result.warnings)
    planner_fallback = result.planner_status == "rules_fallback"
    if planner_fallback and "planner_rules_fallback" not in warnings:
        warnings.append("planner_rules_fallback")
    fused_by_id = {item.paper.canonical_id: item for item in result.fused_papers}

    def enrich(item: RankedPaper) -> RankedPaper:
        fused = fused_by_id.get(item.paper.canonical_id)
        if fused is None:
            return item
        return item.model_copy(
            update={
                "evidence": item.evidence.model_copy(
                    update={
                        "fusion_score": fused.score,
                        "source_ranks": fused.source_ranks,
                    }
                )
            }
        )

    high_relevance = [enrich(item) for item in result.high_relevance]
    partial_relevance = [enrich(item) for item in result.partial_relevance]
    return StructuredSearchResponse(
        run_id=run_id,
        query_id=query_id,
        execution_mode=execution_mode,
        snapshot_set_id=snapshot_set_id,
        snapshot_captured_at=snapshot_captured_at,
        query_analysis=result.query_analysis,
        selected_paper_ids=[
            item.paper.canonical_id for item in result.fused_papers
        ],
        fused_papers=result.fused_papers,
        high_relevance=high_relevance,
        partial_relevance=partial_relevance,
        citation_edges=result.citation_edges,
        search_trace=result.trace if include_trace else [],
        usage=result.usage,
        stop_reason=result.stop_reason,
        is_partial=result.is_partial or planner_fallback,
        planner_fallback=planner_fallback,
        planner_status=result.planner_status,
        dependency_status=_dependency_statuses(
            result,
            execution_mode=execution_mode,
        ),
        warnings=warnings,
        prompt_version=result.prompt_version,
        config_hash=result.config_hash,
        git_sha=git_sha,
    )
