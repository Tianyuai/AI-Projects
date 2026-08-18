from __future__ import annotations

from paper_search.learning.document_ranking_gate import decide_document_ranking_promotion


def test_document_ranking_gate_allows_small_top50_drop_with_front_gain() -> None:
    decision = decide_document_ranking_promotion(
        baseline={5: 0.05, 10: 0.10, 20: 0.14, 50: 0.20},
        candidate={5: 0.06, 10: 0.12, 20: 0.15, 50: 0.195},
        baseline_by_fold={
            1: {10: 0.10, 20: 0.14, 50: 0.20},
            2: {10: 0.09, 20: 0.13, 50: 0.19},
            3: {10: 0.11, 20: 0.15, 50: 0.21},
        },
        candidate_by_fold={
            1: {10: 0.11, 20: 0.15, 50: 0.195},
            2: {10: 0.10, 20: 0.14, 50: 0.18},
            3: {10: 0.12, 20: 0.16, 50: 0.205},
        },
    )

    assert decision["promote"] is True
    assert decision["failed_conditions"] == []


def test_document_ranking_gate_rejects_large_top50_drop() -> None:
    decision = decide_document_ranking_promotion(
        baseline={5: 0.05, 10: 0.10, 20: 0.14, 50: 0.20},
        candidate={5: 0.07, 10: 0.13, 20: 0.16, 50: 0.18},
        baseline_by_fold={fold: {10: 0.10, 20: 0.14, 50: 0.20} for fold in (1, 2, 3)},
        candidate_by_fold={fold: {10: 0.12, 20: 0.15, 50: 0.18} for fold in (1, 2, 3)},
    )

    assert decision["promote"] is False
    assert "overall_recall_at_50_drop" in decision["failed_conditions"]
