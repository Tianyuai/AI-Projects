"""Pure conversion from mock orchestration output to the public response model."""

from paper_search.domain.models import StructuredSearchResponse
from paper_search.pipeline.orchestrator import MinimalSearchResult


def to_structured_response(
    result: MinimalSearchResult,
    *,
    query_id: str,
    git_sha: str,
) -> StructuredSearchResponse:
    """Preserve known result data without inventing ranking evidence."""
    return StructuredSearchResponse(
        query_id=query_id,
        query_analysis=result.query_analysis,
        selected_paper_ids=[paper.canonical_id for paper in result.papers],
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        search_trace=result.trace,
        usage=result.usage,
        stop_reason=result.stop_reason,
        is_partial=result.is_partial,
        warnings=result.warnings,
        config_hash=result.config_hash,
        git_sha=git_sha,
    )
