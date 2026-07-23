from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_search.domain.models import (
    Paper,
    QueryAnalysisResult,
    QuerySpec,
    SearchPlan,
    SubQuery,
    UsageActual,
)
from paper_search.pipeline.orchestrator import MinimalSearchResult
from paper_search.pipeline.response import to_structured_response


def _minimal_result() -> MinimalSearchResult:
    query_spec = QuerySpec(
        original_query="graph retrieval",
        research_goal="find graph retrieval papers",
    )
    query_analysis = QueryAnalysisResult(
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
            rationale="synthetic",
        ),
    )
    return MinimalSearchResult(
        query_analysis=query_analysis,
        papers=[
            Paper(
                canonical_id="openalex:W1",
                title="OpenAlex Paper",
                openalex_id="W1",
                sources=["openalex"],
            ),
            Paper(
                canonical_id="s2:S1",
                title="Semantic Scholar Paper",
                semantic_scholar_id="S1",
                sources=["semantic_scholar"],
            ),
        ],
        provider_results={},
        trace=[{"step": "fuse", "count": 2}],
        usage=UsageActual(search_api_calls=2, llm_calls=1, cost_cny=0.1),
        stop_reason="soft_stop",
        is_partial=True,
        warnings=["semantic_scholar: budget unavailable"],
        config_hash="sha256:" + "a" * 64,
        prompt_version="query-analyze-v1",
    )


def test_to_structured_response_preserves_known_fields_without_fabrication() -> None:
    minimal_result = _minimal_result()

    response = to_structured_response(
        minimal_result,
        query_id="query-1",
        git_sha="abc1234",
    )

    assert response.query_id == "query-1"
    assert response.selected_paper_ids == ["openalex:W1", "s2:S1"]
    assert response.query_analysis == minimal_result.query_analysis
    assert response.search_trace == minimal_result.trace
    assert response.usage == minimal_result.usage
    assert response.stop_reason == minimal_result.stop_reason
    assert response.is_partial == minimal_result.is_partial
    assert response.warnings == minimal_result.warnings
    assert response.config_hash == minimal_result.config_hash
    assert response.git_sha == "abc1234"
    assert response.high_relevance == []
    assert response.partial_relevance == []
    assert response.citation_edges == []


@pytest.mark.parametrize(
    ("query_id", "git_sha"),
    [("", "abc1234"), ("query-1", "")],
)
def test_to_structured_response_rejects_blank_identity(
    query_id: str,
    git_sha: str,
) -> None:
    with pytest.raises(ValidationError):
        to_structured_response(
            _minimal_result(),
            query_id=query_id,
            git_sha=git_sha,
        )
