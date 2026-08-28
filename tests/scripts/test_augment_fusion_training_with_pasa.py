from __future__ import annotations

from paper_search.domain.models import Paper
from paper_search.learning.cpu_document_ranker import (
    DocumentCandidateEvidence,
    DocumentRankingQuery,
)
from paper_search.learning.gated_feature_fusion_ranker import (
    FUSION_FAMILIES,
    FusionQueryContext,
    GatedFeatureFusionRanker,
)
from scripts.augment_fusion_training_with_pasa import (
    _query_audit,
    _supplement_decision,
)


class _ExplodingBaseline:
    def rank(self, query, candidates):
        raise AssertionError("pair-impossible query must bypass baseline ranking")


class _EmptyContext:
    def for_training_query(self, query):
        return FusionQueryContext()


def test_query_audit_short_circuits_when_no_gold_candidate_exists() -> None:
    ranker = GatedFeatureFusionRanker(
        baseline_ranker=_ExplodingBaseline(),
        context_store=_EmptyContext(),
        feature_families=FUSION_FAMILIES,
        epochs=1,
        dimension=8,
    )
    query = DocumentRankingQuery(
        query_id="q1",
        query="graph retrieval",
        gold_paper_ids=["openalex:W1"],
        candidates=[
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id="openalex:W2",
                    openalex_id="W2",
                    title="negative",
                ),
                baseline_score=1.0,
                source_ranks={"lexical": 1},
            )
        ],
    )

    audit = _query_audit(query, ranker)

    assert audit["positive_candidate_count"] == 0
    assert audit["family_effective_pair_count"] == {
        "entity": 0,
        "hard_constraint": 0,
        "reliability": 0,
        "task_provenance": 0,
    }


def test_query_audit_counts_reliability_pairs_without_running_b0() -> None:
    ranker = GatedFeatureFusionRanker(
        baseline_ranker=_ExplodingBaseline(),
        context_store=_EmptyContext(),
        feature_families=FUSION_FAMILIES,
        epochs=1,
        dimension=8,
    )
    query = DocumentRankingQuery(
        query_id="q2",
        query="graph retrieval",
        gold_paper_ids=["openalex:W1"],
        candidates=[
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id=paper_id,
                    openalex_id=paper_id.removeprefix("openalex:"),
                    title=title,
                ),
                baseline_score=score,
                source_ranks={"lexical": rank},
            )
            for paper_id, title, score, rank in (
                ("openalex:W1", "positive", 1.0, 1),
                ("openalex:W2", "negative one", 0.9, 2),
                ("openalex:W3", "negative two", 0.8, 3),
            )
        ],
    )

    audit = _query_audit(query, ranker)

    assert audit["family_effective_pair_count"]["reliability"] == 2


def test_query_audit_applies_query_balanced_pair_limit() -> None:
    ranker = GatedFeatureFusionRanker(
        baseline_ranker=_ExplodingBaseline(),
        context_store=_EmptyContext(),
        feature_families=FUSION_FAMILIES,
        epochs=1,
        dimension=8,
        max_pairs_per_query_family=1,
    )
    query = DocumentRankingQuery(
        query_id="q3",
        query="graph retrieval",
        gold_paper_ids=["openalex:W1"],
        candidates=[
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id=paper_id,
                    openalex_id=paper_id.removeprefix("openalex:"),
                    title=title,
                ),
                baseline_score=score,
                source_ranks={"lexical": rank},
            )
            for paper_id, title, score, rank in (
                ("openalex:W1", "positive", 1.0, 1),
                ("openalex:W2", "negative one", 0.9, 2),
                ("openalex:W3", "negative two", 0.8, 3),
            )
        ],
    )

    audit = _query_audit(query, ranker)

    assert audit["family_effective_pair_count"]["reliability"] == 1


def test_missing_gold_negation_query_requests_mixed_lexical_supplement() -> None:
    audit = {
        "positive_candidate_count": 0,
        "hard_negative_candidate_count": 10,
        "candidate_count": 10,
        "signals": {
            "method": {"eligible": False, "effective": False},
            "dataset": {"eligible": False, "effective": False},
            "year": {"eligible": False, "effective": False},
            "negation": {"eligible": True, "effective": False},
        },
    }

    mode, reasons = _supplement_decision(
        audit,
        shallow_candidate_threshold=50,
        minimum_hard_negatives=20,
    )

    assert mode == "lexical"
    assert reasons == ["missing_gold_positive", "ineffective_gate:negation"]
