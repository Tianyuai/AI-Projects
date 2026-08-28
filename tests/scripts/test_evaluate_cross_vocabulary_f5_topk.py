from __future__ import annotations

import math

import pytest

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import IdentifierMap
from paper_search.learning.cpu_document_ranker import DocumentCandidateEvidence
from scripts.evaluate_cross_vocabulary_f5_topk import (
    aggregate_comparison,
    audit_candidate_pool_monotonicity,
    ranking_metrics,
)


def _candidate(
    canonical_id: str,
    *,
    arxiv_id: str | None = None,
    doi: str | None = None,
    title: str | None = None,
    source_id: str | None = None,
    source_rank: int = 1,
) -> DocumentCandidateEvidence:
    return DocumentCandidateEvidence(
        paper=Paper(
            canonical_id=canonical_id,
            title=title or canonical_id,
            arxiv_id=arxiv_id,
            doi=doi,
            sources=["openalex"],
        ),
        baseline_score=1.0,
        source_ranks={source_id or f"action:{canonical_id}": source_rank},
    )


def test_ranking_metrics_use_all_gold_aliases_and_binary_ndcg() -> None:
    ranked = [
        _candidate("openalex:w1"),
        _candidate("openalex:w2", arxiv_id="2206.00048"),
        _candidate("openalex:w3"),
        _candidate("doi:10.1/gold"),
    ]

    result = ranking_metrics(
        ["arxiv:2206.00048", "doi:10.1/gold"],
        ranked,
        cutoffs=(1, 5),
    )

    assert result["gold_ranks"] == [2, 4]
    assert result["hit_at_1"] == 0.0
    assert result["hit_at_5"] == 1.0
    assert result["recall_at_1"] == 0.0
    assert result["recall_at_5"] == 1.0
    assert result["mrr"] == 0.5
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(5)
    expected_ideal = 1 + 1 / math.log2(3)
    assert result["ndcg_at_10"] == pytest.approx(expected_dcg / expected_ideal)


def test_ranking_metrics_apply_only_an_explicit_identifier_map() -> None:
    ranked = [
        _candidate(
            "doi:10.1109/cvpr.2018.00454",
            doi="10.1109/cvpr.2018.00454",
        )
    ]
    identifier_map = IdentifierMap.from_bytes(
        b'{"doi:10.1109/cvpr.2018.00454":"arxiv:1706.00384"}\n'
    )

    without_map = ranking_metrics(["arxiv:1706.00384"], ranked, cutoffs=(1,))
    with_map = ranking_metrics(
        ["arxiv:1706.00384"],
        ranked,
        cutoffs=(1,),
        identifier_map=identifier_map,
    )

    assert without_map["gold_ranks"] == []
    assert with_map["gold_ranks"] == [1]


def test_ranking_metrics_count_one_resolved_gold_identity_only_once() -> None:
    ranked = [
        _candidate("doi:10.1000/published-a", doi="10.1000/published-a"),
        _candidate("doi:10.1000/published-b", doi="10.1000/published-b"),
    ]
    identifier_map = IdentifierMap.from_bytes(
        b'{"doi:10.1000/published-a":"arxiv:2201.00001",'
        b'"doi:10.1000/published-b":"arxiv:2201.00001"}\n'
    )

    result = ranking_metrics(
        ["arxiv:2201.00001"],
        ranked,
        cutoffs=(1, 2),
        identifier_map=identifier_map,
    )

    assert result["gold_ranks"] == [1]
    assert result["recall_at_2"] == 1.0
    assert result["ndcg_at_10"] == 1.0


def test_monotonicity_accepts_alias_equivalence_and_rejects_missing_member() -> None:
    baseline = [_candidate("arxiv:2206.00048", arxiv_id="2206.00048")]
    equivalent = _candidate(
        "openalex:w123",
        doi="10.48550/arxiv.2206.00048",
    )

    audit = audit_candidate_pool_monotonicity(baseline, [equivalent])
    assert audit == {"baseline_candidate_count": 1, "missing_member_count": 0}

    preprint = _candidate(
        "doi:10.36227/techrxiv.13708270",
        doi="10.36227/techrxiv.13708270",
        title="Human Action Recognition from Various Data Modalities: A Review",
    )
    published = _candidate(
        "doi:10.1109/tpami.2022.3183112",
        doi="10.1109/tpami.2022.3183112",
        title="Human Action Recognition From Various Data Modalities: A Review",
    )
    assert audit_candidate_pool_monotonicity([preprint], [published]) == {
        "baseline_candidate_count": 1,
        "missing_member_count": 0,
    }

    raw_member = _candidate(
        "openalex:w3211394146",
        title="Do Transformers Really Perform Badly for Graph Representation",
        source_id="semantic-original",
        source_rank=1,
    )
    merged_representative = _candidate(
        "doi:10.48550/arxiv.2106.05234",
        title="Do Transformers Really Perform Bad for Graph Representation?",
        source_id="semantic-original",
        source_rank=1,
    )
    assert audit_candidate_pool_monotonicity(
        [raw_member], [merged_representative]
    ) == {"baseline_candidate_count": 1, "missing_member_count": 0}

    with pytest.raises(ValueError, match="lost baseline candidate"):
        audit_candidate_pool_monotonicity(baseline, [_candidate("openalex:w999")])


def test_aggregate_comparison_reports_topk_direction_and_hit_promotions() -> None:
    rows = [
        {
            "signal": "negation",
            "baseline_candidate_count": 10,
            "augmented_candidate_count": 14,
            "baseline": {
                "gold_ranks": [],
                "mrr": 0.0,
                "ndcg_at_10": 0.0,
                "hit_at_5": 0.0,
                "recall_at_5": 0.0,
            },
            "augmented": {
                "gold_ranks": [3],
                "mrr": 1 / 3,
                "ndcg_at_10": 0.5,
                "hit_at_5": 1.0,
                "recall_at_5": 1.0,
            },
        },
        {
            "signal": "unconstrained",
            "baseline_candidate_count": 20,
            "augmented_candidate_count": 21,
            "baseline": {
                "gold_ranks": [2],
                "mrr": 0.5,
                "ndcg_at_10": 0.6,
                "hit_at_5": 1.0,
                "recall_at_5": 1.0,
            },
            "augmented": {
                "gold_ranks": [7],
                "mrr": 1 / 7,
                "ndcg_at_10": 0.3,
                "hit_at_5": 0.0,
                "recall_at_5": 0.0,
            },
        },
    ]

    summary = aggregate_comparison(rows, cutoffs=(5,))

    assert summary["query_count"] == 2
    assert summary["candidate_pool"]["added_candidate_count"] == 5
    assert summary["candidate_pool"]["gold_hit_promotions"] == 1
    assert summary["candidate_pool"]["gold_hit_regressions"] == 0
    assert summary["direction_at_5"] == {
        "improved_query_count": 1,
        "worsened_query_count": 1,
        "unchanged_query_count": 0,
    }
    assert summary["baseline"]["hit_at_5"] == 0.5
    assert summary["augmented"]["hit_at_5"] == 0.5
    assert summary["delta"]["mrr"] == pytest.approx((1 / 3 + 1 / 7) / 2 - 0.25)
