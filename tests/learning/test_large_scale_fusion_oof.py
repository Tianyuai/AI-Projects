from __future__ import annotations

from paper_search.learning.large_scale_fusion_oof import (
    build_oof_comparison,
    fold_for_query_id,
)


def test_oof_fold_assignment_is_stable_and_three_way() -> None:
    first = [fold_for_query_id(f"query-{index}") for index in range(100)]
    second = [fold_for_query_id(f"query-{index}") for index in range(100)]

    assert first == second
    assert set(first) == {1, 2, 3}


def test_oof_comparison_reports_f4_and_f5_against_b0() -> None:
    rows = [
        {
            "query_id": "q1",
            "fold": 1,
            "gold_count": 1,
            "gold_ranks": {"B0": [6], "F4": [4], "F5": [2]},
        },
        {
            "query_id": "q2",
            "fold": 2,
            "gold_count": 1,
            "gold_ranks": {"B0": [], "F4": [], "F5": [5]},
        },
        {
            "query_id": "q3",
            "fold": 3,
            "gold_count": 1,
            "gold_ranks": {"B0": [1], "F4": [1], "F5": [1]},
        },
    ]

    report = build_oof_comparison(rows, training_query_count=3)

    assert report["query_count"] == 3
    assert report["metrics"]["F5"]["recall_at_5"] == 1.0
    assert report["experiments"]["S4-F4-reliability-3"]["metrics_vs_b0"]
    assert report["experiments"]["S4-F5-gated-fusion-3"]["metrics_vs_f4"]
    assert report["test_partition_touched"] is False
