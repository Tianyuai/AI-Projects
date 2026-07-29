"""Deterministic usage estimates for planned evolution rounds."""

from __future__ import annotations

import math
from typing import Protocol

from paper_search.domain.models import UsageEstimate
from paper_search.evolution.models import RoundPlan


class RoundCostEstimator(Protocol):
    """Estimate the provider usage required for a planned evolution round."""

    def estimate(self, plan: RoundPlan, completed_round_count: int) -> UsageEstimate:
        """Return the predicted usage for ``plan``."""


class DeterministicRoundCostEstimator:
    """Estimate round usage from fixed per-subquery and per-round assumptions."""

    def __init__(
        self,
        *,
        search_calls_per_subquery: int,
        llm_calls_per_round: int,
        input_tokens_per_subquery: int,
        output_tokens_per_subquery: int,
        cost_cny_per_subquery: int | float,
        elapsed_ms_per_subquery: int,
    ) -> None:
        self._search_calls_per_subquery = _nonnegative_integer(
            "search_calls_per_subquery", search_calls_per_subquery
        )
        self._llm_calls_per_round = _nonnegative_integer(
            "llm_calls_per_round", llm_calls_per_round
        )
        self._input_tokens_per_subquery = _nonnegative_integer(
            "input_tokens_per_subquery", input_tokens_per_subquery
        )
        self._output_tokens_per_subquery = _nonnegative_integer(
            "output_tokens_per_subquery", output_tokens_per_subquery
        )
        self._cost_cny_per_subquery = _nonnegative_finite_number(
            "cost_cny_per_subquery", cost_cny_per_subquery
        )
        self._elapsed_ms_per_subquery = _nonnegative_integer(
            "elapsed_ms_per_subquery", elapsed_ms_per_subquery
        )

    def estimate(self, plan: RoundPlan, completed_round_count: int) -> UsageEstimate:
        if (
            isinstance(completed_round_count, bool)
            or not isinstance(completed_round_count, int)
            or completed_round_count < 0
        ):
            raise ValueError("completed_round_count must be a nonnegative integer")
        count = len(plan.subqueries)
        return UsageEstimate(
            search_api_calls=self._search_calls_per_subquery * count,
            llm_calls=self._llm_calls_per_round,
            input_tokens=self._input_tokens_per_subquery * count,
            output_tokens=self._output_tokens_per_subquery * count,
            cost_cny=self._cost_cny_per_subquery * count,
            elapsed_ms=self._elapsed_ms_per_subquery * count,
        )


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _nonnegative_finite_number(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)
