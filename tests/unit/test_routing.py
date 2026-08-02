from __future__ import annotations

import pytest

from paper_search.domain.models import SearchPlan, SubQuery
from paper_search.retrieval.routing import route_baseline_subqueries


def _plan(*hints: str) -> SearchPlan:
    subqueries = []
    for index, hint in enumerate(hints, start=1):
        constraints = [f"constraint-{index // 2}"] if hint == "either" else []
        subqueries.append(
            SubQuery(
                query_id=f"q{index}",
                text=f"query {index}",
                query_type="expanded",
                target_constraints=constraints,
                priority=index,
                provider_hint=hint,
            )
        )
    return SearchPlan(
        subqueries=subqueries,
        inherited_hard_filters={},
        rationale="bounded baseline",
    )


def test_openalex_routes_are_deterministically_bounded_between_three_and_six() -> None:
    routed = route_baseline_subqueries(
        _plan("either", "openalex", "semantic_scholar", "either", "openalex", "either", "either")
    )

    assert [item.subquery_id for item in routed] == ["q1", "q2", "q3", "q4", "q5", "q6"]
    assert all(item.providers[0] == "openalex" for item in routed)


def test_semantic_scholar_is_limited_to_top_two_qualifying_subqueries() -> None:
    routed = route_baseline_subqueries(
        _plan("openalex", "semantic_scholar", "either", "semantic_scholar", "either"),
        max_semantic_scholar_calls=2,
    )

    supplemented = [
        item.subquery_id for item in routed if "semantic_scholar" in item.providers
    ]
    assert supplemented == ["q2", "q4"]
    assert routed[1].routing_reason == "high_priority_supplement"


def test_either_does_not_unconditionally_fan_out_to_both_providers() -> None:
    routed = route_baseline_subqueries(_plan("either", "either", "either", "either"))

    supplemented = [item for item in routed if len(item.providers) == 2]
    assert 1 <= len(supplemented) <= 2
    assert len(supplemented) < len(routed)
    assert all(
        item.routing_reason == "uncovered_constraint_supplement"
        for item in supplemented
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_openalex_calls": 0},
        {"min_openalex_calls": 4, "max_openalex_calls": 3},
        {"max_openalex_calls": 7},
        {"max_semantic_scholar_calls": 3},
    ],
)
def test_routing_rejects_out_of_contract_bounds(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        route_baseline_subqueries(_plan("either", "either", "either"), **kwargs)


@pytest.mark.parametrize("count", [1, 2])
def test_routing_rejects_plan_below_openalex_minimum(count: int) -> None:
    with pytest.raises(ValueError, match="OpenAlex minimum"):
        route_baseline_subqueries(_plan(*(["openalex"] * count)))


@pytest.mark.parametrize(("count", "expected"), [(3, 3), (6, 6), (7, 6)])
def test_routing_default_openalex_boundary_cardinalities(
    count: int,
    expected: int,
) -> None:
    routed = route_baseline_subqueries(_plan(*(["openalex"] * count)))
    assert len(routed) == expected


def test_routing_rejects_duplicate_query_ids() -> None:
    plan = _plan("semantic_scholar", "semantic_scholar", "semantic_scholar")
    duplicate = plan.model_copy(
        update={
            "subqueries": [
                item.model_copy(update={"query_id": "duplicate"})
                for item in plan.subqueries
            ]
        }
    )
    with pytest.raises(ValueError, match="unique"):
        route_baseline_subqueries(duplicate)


def test_priority_ties_use_query_id_as_stable_tiebreaker() -> None:
    plan = _plan("openalex", "openalex", "openalex")
    tied = plan.model_copy(
        update={
            "subqueries": [
                plan.subqueries[0].model_copy(update={"query_id": "q-c", "priority": 1}),
                plan.subqueries[1].model_copy(update={"query_id": "q-a", "priority": 1}),
                plan.subqueries[2].model_copy(update={"query_id": "q-b", "priority": 1}),
            ]
        }
    )
    assert [item.subquery_id for item in route_baseline_subqueries(tied)] == [
        "q-a",
        "q-b",
        "q-c",
    ]


@pytest.mark.parametrize("maximum", [0, 1, 2])
def test_semantic_scholar_supplements_respect_each_supported_maximum(
    maximum: int,
) -> None:
    routed = route_baseline_subqueries(
        _plan("semantic_scholar", "semantic_scholar", "semantic_scholar"),
        max_semantic_scholar_calls=maximum,
    )
    assert sum("semantic_scholar" in item.providers for item in routed) == maximum
