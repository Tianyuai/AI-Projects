from __future__ import annotations

from paper_search.domain.models import Paper
from paper_search.learning.cpu_document_ranker import DocumentCandidateEvidence
from scripts.attribute_cross_vocabulary_misses_with_pasa import attribute_query_miss


def _paper(identifier: str, title: str, abstract: str | None = None) -> Paper:
    return Paper(
        canonical_id=identifier,
        title=title,
        abstract=abstract,
        arxiv_id=(identifier.removeprefix("arxiv:") if identifier.startswith("arxiv:") else None),
        sources=["pasa_paper_database"],
    )


def _candidate(
    index: int,
    title: str,
    *,
    actions: tuple[str, ...] = ("anchor@a",),
) -> DocumentCandidateEvidence:
    return DocumentCandidateEvidence(
        paper=Paper(
            canonical_id=f"openalex:w{index}",
            openalex_id=f"W{index}",
            title=title,
            sources=["openalex"],
        ),
        baseline_score=1.0 / (60 + index),
        source_ranks={action: index for action in actions},
    )


def test_attribution_reports_pasa_unavailable_without_guessing() -> None:
    result = attribute_query_miss(
        query="graph retrieval",
        action_text="graph retrieval alignment",
        candidates=[_candidate(1, "Graph retrieval systems")],
        pasa_gold_papers=[],
        expected_gold_count=1,
        anchors=("graph", "retrieval"),
        expansion_terms=("alignment",),
    )

    assert result["category"] == "pasa_gold_metadata_unavailable"


def test_attribution_detects_openalex_identity_metadata_gap_by_exact_title() -> None:
    result = attribute_query_miss(
        query="graph retrieval",
        action_text="graph retrieval alignment",
        candidates=[_candidate(1, "Graph Retrieval with Alignment")],
        pasa_gold_papers=[_paper("arxiv:2201.00001", "Graph Retrieval with Alignment")],
        expected_gold_count=1,
        anchors=("graph", "retrieval"),
        expansion_terms=("alignment",),
    )

    assert result["category"] == "openalex_metadata_identity_gap"
    assert result["exact_title_alias_candidate_count"] == 1


def test_attribution_separates_lexical_mismatch_from_action_construction_gap() -> None:
    mismatch = attribute_query_miss(
        query="graph retrieval",
        action_text="graph retrieval alignment",
        candidates=[_candidate(1, "Graph retrieval systems")],
        pasa_gold_papers=[
            _paper("arxiv:2201.00002", "Bayesian molecular synthesis")
        ],
        expected_gold_count=1,
        anchors=("graph", "retrieval"),
        expansion_terms=("alignment",),
    )
    construction = attribute_query_miss(
        query="graph retrieval",
        action_text="graph retrieval alignment",
        candidates=[_candidate(1, "Graph retrieval systems")],
        pasa_gold_papers=[
            _paper("arxiv:2201.00003", "Alignment methods for graph retrieval")
        ],
        expected_gold_count=1,
        anchors=("graph", "retrieval"),
        expansion_terms=("alignment",),
    )

    assert mismatch["category"] == "cross_vocabulary_mismatch"
    assert construction["category"] == "action_construction_insufficient"


def test_attribution_finds_cross_action_candidate_title_phrase_in_gold() -> None:
    result = attribute_query_miss(
        query="graph retrieval benchmark",
        action_text="graph retrieval benchmark contrastive",
        candidates=[
            _candidate(
                1,
                "Graph contrastive alignment for retrieval",
                actions=("anchor@a", "semantic@b"),
            ),
            _candidate(
                2,
                "Scalable graph contrastive alignment",
                actions=("boolean@c",),
            ),
        ],
        pasa_gold_papers=[
            _paper(
                "arxiv:2201.00004",
                "Graph Contrastive Alignment for Scientific Search",
            )
        ],
        expected_gold_count=1,
        anchors=("graph", "retrieval", "benchmark"),
        expansion_terms=("contrastive",),
    )

    assert result["category"] == "action_construction_insufficient"
    assert result["candidate_supported_gold_phrases"] == [
        {
            "phrase": "graph contrastive",
            "candidate_support": 2,
            "action_support": 3,
        }
    ]
