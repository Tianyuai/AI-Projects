from __future__ import annotations

from scripts.gate_low_confidence_llm_supplement import promotion_decision


def _metrics() -> dict[str, int]:
    return {
        "query_count": 24,
        "baseline_gold_pool_hit_query_count": 9,
        "supplemented_gold_pool_hit_query_count": 12,
        "supplemented_gold_pool_improved_query_count": 3,
        "supplemented_gold_pool_regressed_query_count": 0,
        "baseline_top5_hit_query_count": 4,
        "supplemented_top5_hit_query_count": 4,
        "baseline_top10_hit_query_count": 6,
        "supplemented_top10_hit_query_count": 6,
        "baseline_top20_hit_query_count": 6,
        "supplemented_top20_hit_query_count": 7,
        "unconstrained_baseline_top10_hit_query_count": 2,
        "unconstrained_supplemented_top10_hit_query_count": 2,
        "unconstrained_baseline_top20_hit_query_count": 2,
        "unconstrained_supplemented_top20_hit_query_count": 2,
        "baseline_reconstruction_exact_query_count": 24,
        "production_f5_query_count": 24,
        "gold_blind_selection_query_count": 24,
        "independent_quota_query_count": 24,
    }


def test_promotion_requires_baseline_retention_unconstrained_safety_and_gain() -> None:
    passed = promotion_decision(_metrics())
    regressed = promotion_decision(
        {
            **_metrics(),
            "unconstrained_supplemented_top20_hit_query_count": 1,
        }
    )
    no_gain = promotion_decision(
        {
            **_metrics(),
            "supplemented_gold_pool_hit_query_count": 9,
            "supplemented_gold_pool_improved_query_count": 0,
            "supplemented_top20_hit_query_count": 6,
        }
    )

    assert passed["passed"] is True
    assert passed["decision"] == "eligible_for_production_promotion"
    assert regressed["passed"] is False
    assert "unconstrained_top20_regression" in regressed["failed_gates"]
    assert no_gain["passed"] is False
    assert "no_strict_gain" in no_gain["failed_gates"]
