from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from paper_search.application.contracts import DependencyDiagnostic
from paper_search.domain.models import (
    CandidateEvidence,
    Paper,
    QueryAnalysisResult,
    QuerySpec,
    RankedPaper,
    ResolvedCitationEdge,
    SearchPlan,
    SubQuery,
    UsageActual,
)
from paper_search.pipeline.orchestrator import OrchestratorResult
from paper_search.pipeline.response import to_structured_response
from paper_search.ranking.fusion import FusedPaper


def _ranked(paper: Paper) -> RankedPaper:
    return RankedPaper(
        paper=paper,
        evidence=CandidateEvidence(
            paper_id=paper.canonical_id,
            matched_subqueries=["sq-1"],
            matched_constraints=["graph"],
            unmatched_constraints=[],
            filter_reasons=[],
            lexical_score=0.8,
            embedding_score=0.0,
            rerank_score=None,
            constraint_coverage=1.0,
            source_agreement=0.5,
            authority_score=0.5,
            recency_score=0.5,
            final_score=0.8,
            scoring_version="fixture-v1",
            relevance_level="high",
        ),
    )


def _orchestrator_result() -> OrchestratorResult:
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
    openalex = Paper(
        canonical_id="openalex:W1",
        title="OpenAlex Paper",
        openalex_id="W1",
        sources=["openalex"],
    )
    semantic = Paper(
        canonical_id="s2:S1",
        title="Semantic Scholar Paper",
        semantic_scholar_id="S1",
        sources=["semantic_scholar"],
    )
    edge = ResolvedCitationEdge(
        provider="openalex",
        citing_canonical_id="openalex:W1",
        cited_canonical_id="s2:S1",
        source_edge_hash="sha256:" + "e" * 64,
    )
    diagnostics = [
        DependencyDiagnostic(
            dependency=dependency,
            endpoint="/fixture",
            model_id="fixture-v1",
            usage=UsageActual(),
            latency_ms=0,
            cache_hit=True,
            snapshot_refs=[],
            errors=[],
        )
        for dependency in ("llm", "openalex", "semantic_scholar")
    ]
    return OrchestratorResult(
        query_analysis=query_analysis,
        fused_papers=[
            FusedPaper(
                paper=openalex,
                score=0.25,
                source_ranks={"openalex": 1},
            ),
            FusedPaper(
                paper=semantic,
                score=0.20,
                source_ranks={"semantic_scholar": 1},
            ),
        ],
        high_relevance=[_ranked(openalex)],
        partial_relevance=[],
        citation_edges=[edge],
        provider_results={},
        diagnostics=diagnostics,
        planner_status="primary",
        trace=[{"step": "fuse", "count": 2}],
        usage=UsageActual(search_api_calls=2, llm_calls=1, cost_cny=0.1),
        stop_reason="soft_stop",
        is_partial=True,
        warnings=["semantic_scholar: budget unavailable"],
        config_hash="sha256:" + "a" * 64,
        prompt_version="query-analyze-v1",
    )


def test_to_structured_response_preserves_fusion_and_optional_evidence() -> None:
    result = _orchestrator_result()

    response = to_structured_response(
        result,
        query_id="query-1",
        git_sha="abc1234",
        run_id="run-1",
        execution_mode="replay",
        snapshot_set_id="snapshot-set-1",
        snapshot_captured_at=datetime(2026, 8, 2, tzinfo=UTC),
    )

    assert response.query_id == "query-1"
    assert response.run_id == "run-1"
    assert response.execution_mode == "replay"
    assert response.snapshot_set_id == "snapshot-set-1"
    assert response.selected_paper_ids == ["openalex:W1", "s2:S1"]
    assert response.fused_papers == result.fused_papers
    assert response.high_relevance[0].evidence.fusion_score == 0.25
    assert response.high_relevance[0].evidence.source_ranks == {"openalex": 1}
    assert response.partial_relevance[0].paper.canonical_id == "s2:S1"
    assert response.partial_relevance[0].evidence.fusion_score == 0.20
    assert response.partial_relevance[0].evidence.source_ranks == {
        "semantic_scholar": 1
    }
    assert response.query_analysis == result.query_analysis
    assert response.search_trace == result.trace
    assert response.usage == result.usage
    assert response.stop_reason == result.stop_reason
    assert response.is_partial is True
    assert response.planner_status == "primary"
    assert response.planner_fallback is False
    assert [status.dependency for status in response.dependency_status] == [
        "llm",
        "openalex",
        "semantic_scholar",
    ]
    assert [item.paper for item in response.high_relevance] == [
        item.paper for item in result.high_relevance
    ]
    assert [item.paper.canonical_id for item in response.partial_relevance] == [
        "s2:S1"
    ]
    assert response.citation_edges == result.citation_edges
    assert response.prompt_version == result.prompt_version
    assert response.warnings == result.warnings
    assert response.config_hash == result.config_hash
    assert response.git_sha == "abc1234"


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
            _orchestrator_result(),
            query_id=query_id,
            git_sha=git_sha,
        )
