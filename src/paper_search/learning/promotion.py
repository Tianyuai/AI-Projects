"""Pre-registered development gates for promoting an action ranker."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from paper_search.domain.models import (
    DomainModel,
    StrictNonNegativeFloat,
    StrictNonNegativeInt,
    UnitFloat,
)


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PromotionGate = Literal[
    "incomplete_dev_evaluation",
    "original_query_anchor",
    "historical_gold_pre_truncation",
    "top50_gold_retention",
    "not_retrieved_reduction",
    "seed_stability",
    "constraint_retention",
    "llm_call_budget",
]


class ActionRankerPromotionCriteria(DomainModel):
    minimum_original_query_anchor_rate: UnitFloat = 1.0
    minimum_historical_gold_retention_pre_truncation: UnitFloat = 1.0
    minimum_top50_gold_retention: UnitFloat = 0.95
    minimum_not_retrieved_relative_reduction: StrictNonNegativeFloat = 0.10
    minimum_positive_seed_count: StrictNonNegativeInt = 2
    required_seed_count: StrictNonNegativeInt = 3
    minimum_constraint_retention_delta: FiniteFloat = 0.0
    maximum_average_llm_calls_per_query: StrictNonNegativeFloat = 0.5


class ActionRankerEvaluationSummary(DomainModel):
    evaluation_role: Literal["development"] = "development"
    evaluated_query_count: StrictNonNegativeInt
    expected_query_count: Annotated[int, Field(strict=True, gt=0)]
    original_query_anchor_rate: UnitFloat
    historical_gold_retention_pre_truncation: UnitFloat
    top50_gold_retention: UnitFloat
    not_retrieved_relative_reduction: FiniteFloat
    positive_seed_count: StrictNonNegativeInt
    evaluated_seed_count: StrictNonNegativeInt
    constraint_retention_delta: FiniteFloat
    average_llm_calls_per_query: StrictNonNegativeFloat

    @model_validator(mode="after")
    def validate_counts(self) -> ActionRankerEvaluationSummary:
        if self.positive_seed_count > self.evaluated_seed_count:
            raise ValueError("positive seed count cannot exceed evaluated seeds")
        return self


class ActionRankerPromotionDecision(DomainModel):
    promoted: bool
    failed_gates: list[PromotionGate]


def evaluate_action_ranker_promotion(
    summary: ActionRankerEvaluationSummary,
    criteria: ActionRankerPromotionCriteria,
) -> ActionRankerPromotionDecision:
    summary = ActionRankerEvaluationSummary.model_validate(summary)
    criteria = ActionRankerPromotionCriteria.model_validate(criteria)
    failed: list[PromotionGate] = []
    checks: tuple[tuple[PromotionGate, bool], ...] = (
        (
            "incomplete_dev_evaluation",
            summary.evaluated_query_count == summary.expected_query_count,
        ),
        (
            "original_query_anchor",
            summary.original_query_anchor_rate
            >= criteria.minimum_original_query_anchor_rate,
        ),
        (
            "historical_gold_pre_truncation",
            summary.historical_gold_retention_pre_truncation
            >= criteria.minimum_historical_gold_retention_pre_truncation,
        ),
        (
            "top50_gold_retention",
            summary.top50_gold_retention >= criteria.minimum_top50_gold_retention,
        ),
        (
            "not_retrieved_reduction",
            summary.not_retrieved_relative_reduction
            >= criteria.minimum_not_retrieved_relative_reduction,
        ),
        (
            "seed_stability",
            summary.evaluated_seed_count == criteria.required_seed_count
            and summary.positive_seed_count >= criteria.minimum_positive_seed_count,
        ),
        (
            "constraint_retention",
            summary.constraint_retention_delta
            >= criteria.minimum_constraint_retention_delta,
        ),
        (
            "llm_call_budget",
            summary.average_llm_calls_per_query
            <= criteria.maximum_average_llm_calls_per_query,
        ),
    )
    failed.extend(gate for gate, passed in checks if not passed)
    return ActionRankerPromotionDecision(promoted=not failed, failed_gates=failed)


__all__ = [
    "ActionRankerEvaluationSummary",
    "ActionRankerPromotionCriteria",
    "ActionRankerPromotionDecision",
    "evaluate_action_ranker_promotion",
]
