from __future__ import annotations

from paper_search.learning.promotion import (
    ActionRankerEvaluationSummary,
    ActionRankerPromotionCriteria,
    evaluate_action_ranker_promotion,
)


def _passing_summary(**updates: object) -> ActionRankerEvaluationSummary:
    values: dict[str, object] = {
        "evaluated_query_count": 100,
        "expected_query_count": 100,
        "original_query_anchor_rate": 1.0,
        "historical_gold_retention_pre_truncation": 1.0,
        "top50_gold_retention": 0.96,
        "not_retrieved_relative_reduction": 0.11,
        "positive_seed_count": 2,
        "evaluated_seed_count": 3,
        "constraint_retention_delta": 0.0,
        "average_llm_calls_per_query": 0.4,
    }
    values.update(updates)
    return ActionRankerEvaluationSummary.model_validate(values)


def test_default_promotion_criteria_accept_only_a_complete_passing_dev_run() -> None:
    decision = evaluate_action_ranker_promotion(
        _passing_summary(),
        ActionRankerPromotionCriteria(),
    )

    assert decision.promoted is True
    assert decision.failed_gates == []


def test_promotion_reports_every_failed_gate_instead_of_tuning_on_test() -> None:
    decision = evaluate_action_ranker_promotion(
        _passing_summary(
            evaluated_query_count=99,
            original_query_anchor_rate=0.99,
            historical_gold_retention_pre_truncation=0.99,
            top50_gold_retention=0.94,
            not_retrieved_relative_reduction=0.09,
            positive_seed_count=1,
            constraint_retention_delta=-0.01,
            average_llm_calls_per_query=0.51,
        ),
        ActionRankerPromotionCriteria(),
    )

    assert decision.promoted is False
    assert decision.failed_gates == [
        "incomplete_dev_evaluation",
        "original_query_anchor",
        "historical_gold_pre_truncation",
        "top50_gold_retention",
        "not_retrieved_reduction",
        "seed_stability",
        "constraint_retention",
        "llm_call_budget",
    ]
