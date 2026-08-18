from __future__ import annotations

from paper_search.learning.method_route_labels import MethodRouteLabel
from paper_search.learning.method_router_gate import (
    MethodRouterGate,
    assess_method_router,
)


def _row(index: int, *, beneficial: bool, calls: int = 1) -> MethodRouteLabel:
    return MethodRouteLabel(
        dataset="pasa",
        split="auto_dev",
        role="development",
        method="semantic",
        query_id=f"q{index}",
        query=f"query {index}",
        routing_label="beneficial" if beneficial else "not_beneficial",
        gold_association_count=1,
        marginal_gold_hit_count=1 if beneficial else 0,
        marginal_recall=1.0 if beneficial else 0.0,
        method_action_count=calls,
        search_api_calls=calls,
    )


def test_gate_requires_all_predeclared_benefit_and_cost_conditions() -> None:
    rows = [_row(index, beneficial=index < 2) for index in range(10)]
    probabilities = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    gate = MethodRouterGate(
        minimum_evaluated_queries=10,
        minimum_beneficial_queries=2,
        minimum_availability_rate=0.95,
        minimum_beneficial_recall=0.9,
        minimum_call_reduction=0.25,
        minimum_f1_lift=0.05,
        minimum_marginal_gold_capture=0.9,
    )

    decision = assess_method_router(rows, probabilities, threshold=0.5, gate=gate)

    assert decision.enable is True
    assert decision.call_reduction == 0.8
    assert decision.marginal_gold_capture == 1.0
    assert decision.failed_conditions == ()


def test_gate_fails_closed_when_positive_support_is_too_small() -> None:
    rows = [_row(index, beneficial=index == 0) for index in range(10)]
    gate = MethodRouterGate(
        minimum_evaluated_queries=10,
        minimum_beneficial_queries=2,
        minimum_availability_rate=0.95,
        minimum_beneficial_recall=0.9,
        minimum_call_reduction=0.25,
        minimum_f1_lift=0.05,
        minimum_marginal_gold_capture=0.9,
    )

    decision = assess_method_router(
        rows,
        [0.9] + [0.1] * 9,
        threshold=0.5,
        gate=gate,
    )

    assert decision.enable is False
    assert "minimum_beneficial_queries" in decision.failed_conditions
