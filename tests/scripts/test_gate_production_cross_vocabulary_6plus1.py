from __future__ import annotations

import pytest

from scripts.gate_production_cross_vocabulary_6plus1 import evaluate_f5_evidence


def _f5_ab() -> dict[str, object]:
    return {
        "comparison": {
            "candidate_pool_only_intervention": True,
            "same_ranker_both_arms": True,
        },
        "by_signal": {
            "unconstrained": {
                "query_count": 121,
                "candidate_pool": {
                    "baseline_gold_hit_query_count": 0,
                    "augmented_gold_hit_query_count": 18,
                    "gold_hit_regressions": 0,
                },
                "direction_at_5": {"worsened_query_count": 0},
                "direction_at_10": {
                    "improved_query_count": 3,
                    "worsened_query_count": 0,
                },
                "direction_at_20": {
                    "improved_query_count": 9,
                    "worsened_query_count": 0,
                },
                "direction_at_50": {
                    "improved_query_count": 13,
                    "worsened_query_count": 0,
                },
            }
        },
        "safety": {
            "candidate_membership_monotonic": True,
            "llm_requests_made": 0,
            "online_requests_made": 0,
            "production_lock_modified": False,
            "test_partition_touched": False,
            "training_started": False,
        },
    }


def test_f5_evidence_gate_accepts_the_sealed_unconstrained_result() -> None:
    result = evaluate_f5_evidence(_f5_ab())

    assert result["passed"] is True
    assert result["unconstrained_gold_hit_query_count"] == 18
    assert result["top_k_improved_query_counts"] == {"10": 3, "20": 9, "50": 13}


def test_f5_evidence_gate_rejects_any_top_k_regression() -> None:
    payload = _f5_ab()
    payload["by_signal"]["unconstrained"]["direction_at_10"][  # type: ignore[index]
        "worsened_query_count"
    ] = 1

    with pytest.raises(ValueError, match="Top-K regression"):
        evaluate_f5_evidence(payload)
