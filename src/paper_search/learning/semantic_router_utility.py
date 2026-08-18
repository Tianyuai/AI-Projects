"""Continuous semantic-routing reward labels derived from observed receipts."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr
from paper_search.learning.method_route_labels import MethodRouteLabel


class SemanticUtilityLabel(DomainModel):
    dataset: NonEmptyStr
    split: NonEmptyStr
    role: Literal["training", "development"]
    query_id: NonEmptyStr
    query: NonEmptyStr
    marginal_gold_hit_count: int = Field(strict=True, ge=0)
    search_api_calls: int = Field(strict=True, gt=0)
    marginal_hits_per_call: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_observed_utility(self) -> SemanticUtilityLabel:
        expected = self.marginal_gold_hit_count / self.search_api_calls
        if abs(self.marginal_hits_per_call - expected) > 1e-12:
            raise ValueError("utility must equal marginal Gold hits per API call")
        return self


def semantic_utility_labels(
    rows: list[MethodRouteLabel],
) -> list[SemanticUtilityLabel]:
    """Convert available method labels into continuous observed utility."""
    labels: list[SemanticUtilityLabel] = []
    for raw in sorted(rows, key=lambda item: item.query_id):
        row = MethodRouteLabel.model_validate(raw)
        if row.routing_label == "unavailable":
            continue
        if row.search_api_calls <= 0:
            raise ValueError("available utility labels require positive observed API cost")
        labels.append(
            SemanticUtilityLabel(
                dataset=row.dataset,
                split=row.split,
                role=row.role,
                query_id=row.query_id,
                query=row.query,
                marginal_gold_hit_count=row.marginal_gold_hit_count,
                search_api_calls=row.search_api_calls,
                marginal_hits_per_call=(
                    row.marginal_gold_hit_count / row.search_api_calls
                ),
            )
        )
    return labels


def utility_rank_scores(predictions: list[float]) -> list[float]:
    """Map arbitrary regression outputs to monotonic unit scores for routing."""
    if not predictions or not all(math.isfinite(value) for value in predictions):
        raise ValueError("utility predictions must be non-empty and finite")
    return [
        1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))
        for value in predictions
    ]


def marginal_capture_at_call_reduction(
    rows: list[MethodRouteLabel],
    scores: list[float],
    *,
    target_call_reduction: float,
) -> dict[str, float | int]:
    """Measure captured marginal Gold at a fixed observed API-call budget."""
    if not 0.0 <= target_call_reduction < 1.0:
        raise ValueError("target call reduction must be in [0, 1)")
    if not rows or len(rows) != len(scores):
        raise ValueError("rows and scores must be non-empty and aligned")
    validated = [MethodRouteLabel.model_validate(row) for row in rows]
    if any(
        row.routing_label == "unavailable" or row.search_api_calls <= 0
        for row in validated
    ):
        raise ValueError("fixed-cost comparison requires available observed calls")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("ranking scores must be finite")
    ranked = sorted(
        zip(validated, scores, strict=True),
        key=lambda item: (-item[1], item[0].query_id),
    )
    all_calls = sum(row.search_api_calls for row in validated)
    call_budget = all_calls * (1.0 - target_call_reduction)
    selected: list[MethodRouteLabel] = []
    selected_calls = 0
    for row, _ in ranked:
        if selected_calls + row.search_api_calls > call_budget + 1e-12:
            continue
        selected.append(row)
        selected_calls += row.search_api_calls
    all_marginal = sum(row.marginal_gold_hit_count for row in validated)
    selected_marginal = sum(row.marginal_gold_hit_count for row in selected)
    return {
        "target_call_reduction": target_call_reduction,
        "achieved_call_reduction": 1.0 - selected_calls / all_calls,
        "selected_query_count": len(selected),
        "marginal_gold_capture": (
            selected_marginal / all_marginal if all_marginal else 0.0
        ),
    }


__all__ = [
    "SemanticUtilityLabel",
    "marginal_capture_at_call_reduction",
    "semantic_utility_labels",
    "utility_rank_scores",
]
