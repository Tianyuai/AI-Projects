from __future__ import annotations

import pytest

from paper_search.learning.method_route_labels import MethodRouteLabel
from paper_search.learning.semantic_router_utility import (
    marginal_capture_at_call_reduction,
    semantic_utility_labels,
    utility_rank_scores,
)


def _route_label(
    query_id: str,
    *,
    marginal_hits: int,
    calls: int = 1,
    routing_label: str | None = None,
) -> MethodRouteLabel:
    return MethodRouteLabel(
        dataset="pasa",
        split="auto_train",
        role="training",
        method="semantic",
        query_id=query_id,
        query=f"query {query_id}",
        routing_label=routing_label or ("beneficial" if marginal_hits else "not_beneficial"),
        gold_association_count=4,
        marginal_gold_hit_count=marginal_hits,
        marginal_recall=marginal_hits / 4,
        method_action_count=calls,
        search_api_calls=calls,
    )


def test_semantic_utility_is_marginal_gold_hits_per_observed_call() -> None:
    labels = semantic_utility_labels(
        [_route_label("q1", marginal_hits=2, calls=1), _route_label("q2", marginal_hits=0, calls=1)]
    )

    assert [label.marginal_hits_per_call for label in labels] == [2.0, 0.0]
    assert [label.query_id for label in labels] == ["q1", "q2"]


def test_semantic_utility_excludes_unavailable_observations() -> None:
    labels = semantic_utility_labels(
        [
            _route_label("available", marginal_hits=1),
            _route_label(
                "unavailable",
                marginal_hits=0,
                calls=0,
                routing_label="unavailable",
            ),
        ]
    )

    assert [label.query_id for label in labels] == ["available"]


def test_semantic_utility_rejects_available_rows_without_observed_calls() -> None:
    with pytest.raises(ValueError, match="positive observed API cost"):
        semantic_utility_labels([_route_label("q1", marginal_hits=0, calls=0)])


def test_utility_rank_scores_are_monotonic_unit_values() -> None:
    scores = utility_rank_scores([-2.0, 0.0, 3.0])

    assert 0.0 < scores[0] < scores[1] < scores[2] < 1.0


def test_marginal_capture_compares_rankings_at_fixed_call_reduction() -> None:
    rows = [
        _route_label("three", marginal_hits=3),
        _route_label("two", marginal_hits=2),
        _route_label("one", marginal_hits=1),
        _route_label("zero", marginal_hits=0),
    ]

    result = marginal_capture_at_call_reduction(
        rows,
        [0.9, 0.8, 0.2, 0.1],
        target_call_reduction=0.5,
    )

    assert result["selected_query_count"] == 2
    assert result["achieved_call_reduction"] == 0.5
    assert result["marginal_gold_capture"] == pytest.approx(5 / 6)
