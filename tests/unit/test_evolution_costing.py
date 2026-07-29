import importlib
from typing import Any

import pytest

from paper_search.domain.models import UsageEstimate
from paper_search.evolution import RoundPlan


def costing_api() -> tuple[type[Any], type[Any]]:
    module = importlib.import_module("paper_search.evolution")
    assert hasattr(
        module, "DeterministicRoundCostEstimator"
    ), "DeterministicRoundCostEstimator must be implemented"
    assert hasattr(module, "RoundCostEstimator"), "RoundCostEstimator must be implemented"
    return module.DeterministicRoundCostEstimator, module.RoundCostEstimator


def round_plan_with_three_queries() -> RoundPlan:
    return RoundPlan(
        round_number=1,
        subqueries=[
            {
                "query_id": f"q{index}",
                "text": f"query {index}",
                "query_type": "decomposed",
                "priority": index,
                "provider_hint": "either",
            }
            for index in range(1, 4)
        ],
    )


def estimator(**updates: object) -> Any:
    estimator_type, _ = costing_api()
    values = {
        "search_calls_per_subquery": 2,
        "llm_calls_per_round": 1,
        "input_tokens_per_subquery": 100,
        "output_tokens_per_subquery": 20,
        "cost_cny_per_subquery": 0.01,
        "elapsed_ms_per_subquery": 500,
    }
    values.update(updates)
    return estimator_type(**values)


def test_estimate_scales_all_usage_dimensions_by_subquery_count() -> None:
    estimate = estimator().estimate(round_plan_with_three_queries(), 0)

    assert estimate == UsageEstimate(
        search_api_calls=6,
        llm_calls=1,
        input_tokens=300,
        output_tokens=60,
        cost_cny=0.03,
        elapsed_ms=1500,
    )


@pytest.mark.parametrize("value", [True, False, 1.5, -1])
@pytest.mark.parametrize(
    "field",
    [
        "search_calls_per_subquery",
        "llm_calls_per_round",
        "input_tokens_per_subquery",
        "output_tokens_per_subquery",
        "elapsed_ms_per_subquery",
    ],
)
def test_integer_numeric_assumptions_must_be_strict_nonnegative_integers(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        estimator(**{field: value})


@pytest.mark.parametrize("value", [True, -0.1, float("inf"), float("nan")])
def test_cost_assumption_must_be_finite_nonnegative_number(value: object) -> None:
    with pytest.raises(ValueError):
        estimator(cost_cny_per_subquery=value)


@pytest.mark.parametrize("value", [True, False, 1.0, -1])
def test_completed_round_count_must_be_nonnegative_integer(value: object) -> None:
    with pytest.raises(ValueError, match="completed_round_count must be a nonnegative integer"):
        estimator().estimate(round_plan_with_three_queries(), value)
