"""Predeclared benefit/cost gate for optional retrieval method routers."""

from __future__ import annotations

from pydantic import Field

from paper_search.domain.models import DomainModel, UnitFloat
from paper_search.learning.cpu_baseline import BinaryMetrics, evaluate_probabilities
from paper_search.learning.method_route_labels import MethodRouteLabel


class MethodRouterGate(DomainModel):
    minimum_evaluated_queries: int = Field(strict=True, gt=0)
    minimum_beneficial_queries: int = Field(strict=True, gt=0)
    minimum_availability_rate: UnitFloat
    minimum_beneficial_recall: UnitFloat
    minimum_call_reduction: UnitFloat
    minimum_f1_lift: UnitFloat
    minimum_marginal_gold_capture: UnitFloat


class MethodRouterGateDecision(DomainModel):
    enable: bool
    total_query_count: int = Field(strict=True, gt=0)
    evaluated_query_count: int = Field(strict=True, ge=0)
    beneficial_query_count: int = Field(strict=True, ge=0)
    availability_rate: UnitFloat
    routed: BinaryMetrics
    always_call: BinaryMetrics
    f1_lift: float
    call_reduction: UnitFloat
    marginal_gold_capture: UnitFloat
    failed_conditions: tuple[str, ...]


def assess_method_router(
    rows: list[MethodRouteLabel],
    probabilities: list[float],
    *,
    threshold: float,
    gate: MethodRouterGate,
) -> MethodRouterGateDecision:
    validated = [MethodRouteLabel.model_validate(row) for row in rows]
    if not validated or len(validated) != len(probabilities):
        raise ValueError("labels and probabilities must be non-empty and aligned")
    pairs = [
        (row, probability)
        for row, probability in zip(validated, probabilities, strict=True)
        if row.routing_label != "unavailable"
    ]
    available = [row for row, _ in pairs]
    scores = [score for _, score in pairs]
    labels = [row.routing_label == "beneficial" for row in available]
    routed = evaluate_probabilities(labels, scores, threshold=threshold)
    always_call = evaluate_probabilities(
        labels, [1.0] * len(labels), threshold=0.5
    )
    predicted = [score >= threshold for score in scores]
    all_calls = sum(row.search_api_calls for row in available)
    selected_calls = sum(
        row.search_api_calls
        for row, selected in zip(available, predicted, strict=True)
        if selected
    )
    call_reduction = 1.0 - selected_calls / all_calls if all_calls else 0.0
    all_marginal = sum(row.marginal_gold_hit_count for row in available)
    captured_marginal = sum(
        row.marginal_gold_hit_count
        for row, selected in zip(available, predicted, strict=True)
        if selected
    )
    marginal_capture = (
        captured_marginal / all_marginal if all_marginal else 0.0
    )
    beneficial_count = sum(labels)
    availability_rate = len(available) / len(validated)
    f1_lift = float(routed.f1) - float(always_call.f1)
    checks = {
        "minimum_evaluated_queries": len(available)
        >= gate.minimum_evaluated_queries,
        "minimum_beneficial_queries": beneficial_count
        >= gate.minimum_beneficial_queries,
        "minimum_availability_rate": availability_rate
        >= gate.minimum_availability_rate,
        "minimum_beneficial_recall": routed.recall
        >= gate.minimum_beneficial_recall,
        "minimum_call_reduction": call_reduction >= gate.minimum_call_reduction,
        "minimum_f1_lift": f1_lift >= gate.minimum_f1_lift,
        "minimum_marginal_gold_capture": marginal_capture
        >= gate.minimum_marginal_gold_capture,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return MethodRouterGateDecision(
        enable=not failed,
        total_query_count=len(validated),
        evaluated_query_count=len(available),
        beneficial_query_count=beneficial_count,
        availability_rate=availability_rate,
        routed=routed,
        always_call=always_call,
        f1_lift=f1_lift,
        call_reduction=call_reduction,
        marginal_gold_capture=marginal_capture,
        failed_conditions=failed,
    )


__all__ = [
    "MethodRouterGate",
    "MethodRouterGateDecision",
    "assess_method_router",
]
