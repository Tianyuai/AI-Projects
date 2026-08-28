from __future__ import annotations

import pytest

from paper_search.evaluation.delivery_rehearsal import compare_delivery_predictions
from paper_search.evaluation.official_adapter import InternalPredictionRecord


def test_delivery_rehearsal_requires_exact_live_replay_order() -> None:
    live = [
        InternalPredictionRecord(query_id="q1", selected_paper_ids=["openalex:W1"]),
        InternalPredictionRecord(query_id="q2", selected_paper_ids=["arxiv:2"]),
    ]
    replay = [
        InternalPredictionRecord(query_id="q1", selected_paper_ids=["openalex:W1"]),
        InternalPredictionRecord(query_id="q2", selected_paper_ids=["arxiv:2"]),
    ]

    report = compare_delivery_predictions(live, replay)

    assert report["passed"] is True
    assert report["query_count"] == 2
    assert report["identical_query_count"] == 2
    assert report["mismatched_query_ids"] == []
    assert report["live_predictions_sha256"] == report["replay_predictions_sha256"]


def test_delivery_rehearsal_fails_closed_on_ranking_mismatch() -> None:
    live = [
        InternalPredictionRecord(
            query_id="q1", selected_paper_ids=["openalex:W1", "openalex:W2"]
        )
    ]
    replay = [
        InternalPredictionRecord(
            query_id="q1", selected_paper_ids=["openalex:W2", "openalex:W1"]
        )
    ]

    with pytest.raises(ValueError, match="live/replay ranking mismatch"):
        compare_delivery_predictions(live, replay)
