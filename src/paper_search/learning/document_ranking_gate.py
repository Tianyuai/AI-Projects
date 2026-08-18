"""Frozen promotion criteria for CPU document-ranking experiments."""

from __future__ import annotations

from collections.abc import Mapping


def decide_document_ranking_promotion(
    *,
    baseline: Mapping[int, float],
    candidate: Mapping[int, float],
    baseline_by_fold: Mapping[int, Mapping[int, float]],
    candidate_by_fold: Mapping[int, Mapping[int, float]],
    maximum_overall_recall_at_50_drop: float = 0.01,
    maximum_fold_recall_at_50_drop: float = 0.02,
) -> dict[str, object]:
    """Allow small Top-50 noise only when front-rank recall improves."""

    if set(baseline_by_fold) != set(candidate_by_fold) or not baseline_by_fold:
        raise ValueError("baseline and candidate folds must match")
    failed: list[str] = []
    if baseline[50] - candidate[50] > maximum_overall_recall_at_50_drop:
        failed.append("overall_recall_at_50_drop")
    if any(
        baseline_by_fold[fold][50] - candidate_by_fold[fold][50]
        > maximum_fold_recall_at_50_drop
        for fold in baseline_by_fold
    ):
        failed.append("fold_recall_at_50_drop")
    if not (candidate[10] > baseline[10] or candidate[20] > baseline[20]):
        failed.append("no_front_recall_gain")
    non_decreasing_folds = sum(
        candidate_by_fold[fold][10] >= baseline_by_fold[fold][10]
        or candidate_by_fold[fold][20] >= baseline_by_fold[fold][20]
        for fold in baseline_by_fold
    )
    if non_decreasing_folds < 2:
        failed.append("front_recall_unstable_across_folds")
    return {
        "promote": not failed,
        "failed_conditions": failed,
        "front_non_decreasing_fold_count": non_decreasing_folds,
        "maximum_overall_recall_at_50_drop": maximum_overall_recall_at_50_drop,
        "maximum_fold_recall_at_50_drop": maximum_fold_recall_at_50_drop,
    }


__all__ = ["decide_document_ranking_promotion"]
