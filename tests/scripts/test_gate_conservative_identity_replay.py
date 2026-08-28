from __future__ import annotations

import pytest

from scripts.gate_conservative_identity_replay import compare_replays


def _row(query_id: str, baseline: list[int], augmented: list[int]) -> dict[str, object]:
    return {
        "query_id": query_id,
        "baseline_candidate_count": 2,
        "augmented_candidate_count": 3,
        "baseline_top_50": ["doi:10.1/a", "doi:10.1/b"],
        "augmented_top_50": ["doi:10.1/c", "doi:10.1/a"],
        "baseline": {"gold_ranks": baseline},
        "augmented": {"gold_ranks": augmented},
    }


def test_compare_replays_accepts_additive_identity_matches_only() -> None:
    before = {"per_query": [_row("q1", [], [2])]}
    after = {"per_query": [_row("q1", [1], [1, 2])]}

    result = compare_replays(before, after)

    assert result == {
        "query_count": 1,
        "candidate_count_drift_query_count": 0,
        "ranked_sequence_drift_query_count": 0,
        "removed_gold_rank_query_count": 0,
        "newly_resolved_baseline_query_count": 1,
        "newly_resolved_augmented_query_count": 0,
        "gold_rank_changed_query_count": 1,
    }


def test_compare_replays_rejects_ranked_sequence_drift() -> None:
    before = {"per_query": [_row("q1", [], [])]}
    changed = _row("q1", [], [])
    changed["augmented_top_50"] = ["doi:10.1/a", "doi:10.1/c"]

    with pytest.raises(ValueError, match="ranked candidate sequence drift"):
        compare_replays(before, {"per_query": [changed]})
