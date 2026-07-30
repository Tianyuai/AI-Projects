"""Pure conversion from mock orchestration output to the public response model."""

from paper_search.domain.models import DependencyStatus, StructuredSearchResponse
from paper_search.pipeline.orchestrator import MinimalSearchResult


def to_structured_response(
    result: MinimalSearchResult,
    *,
    query_id: str,
    git_sha: str,
) -> StructuredSearchResponse:
    """Preserve known result data without inventing ranking evidence."""
    return StructuredSearchResponse(
        run_id="mock-run-1",
        query_id=query_id,
        execution_mode="replay",
        snapshot_set_id="mock-snapshot-v1",
        snapshot_captured_at=None,
        query_analysis=result.query_analysis,
        selected_paper_ids=[paper.canonical_id for paper in result.papers],
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        search_trace=result.trace,
        usage=result.usage,
        stop_reason=result.stop_reason,
        is_partial=result.is_partial,
        planner_fallback=False,
        planner_status="primary",
        dependency_status=[
            DependencyStatus(
                dependency="llm",
                state="replayed",
                cache_hit=True,
                error_codes=[],
            ),
            DependencyStatus(
                dependency="openalex",
                state="replayed",
                cache_hit=True,
                error_codes=[],
            ),
            DependencyStatus(
                dependency="semantic_scholar",
                state="replayed",
                cache_hit=True,
                error_codes=[],
            ),
        ],
        warnings=result.warnings,
        prompt_version=result.prompt_version,
        config_hash=result.config_hash,
        git_sha=git_sha,
    )
