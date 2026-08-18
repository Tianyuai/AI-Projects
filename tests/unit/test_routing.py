from __future__ import annotations

import pytest

from paper_search.domain.models import SearchPlan, SubQuery
from paper_search.retrieval.routing import (
    FixedBudgetOpenAlexPolicy,
    FixedHybridOpenAlexPolicy,
    route_baseline_subqueries,
)


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


def test_fixed_hybrid_policy_builds_lexical_and_semantic_openalex_actions() -> None:
    routes = FixedHybridOpenAlexPolicy(
        min_openalex_calls=3,
        max_openalex_calls=6,
        max_semantic_scholar_calls=2,
    ).route(_plan("either", "openalex", "semantic_scholar"))

    assert [(item.source_subquery_id, item.search_mode) for item in routes] == [
        ("q1", "lexical"),
        ("q1", "semantic"),
        ("q2", "lexical"),
        ("q2", "semantic"),
        ("q3", "lexical"),
        ("q3", "semantic"),
    ]
    assert all(item.providers[0] == "openalex" for item in routes)
    assert len({item.route_id for item in routes}) == 6


def test_fixed_hybrid_policy_uses_s2_only_as_bounded_lexical_fallback() -> None:
    routes = FixedHybridOpenAlexPolicy(
        min_openalex_calls=3,
        max_openalex_calls=6,
        max_semantic_scholar_calls=2,
    ).route(_plan("either", "either", "either"))

    fallback_routes = [
        item for item in routes if "semantic_scholar" in item.providers
    ]
    assert [(item.source_subquery_id, item.search_mode) for item in fallback_routes] == [
        ("q1", "lexical"),
        ("q2", "lexical"),
    ]
    assert all(item.routing_reason == "provider_fallback" for item in fallback_routes)


def test_fixed_budget_policy_reserves_original_anchors_and_structured_budget() -> None:
    plan = SearchPlan(
        subqueries=[
            SubQuery(
                query_id="structured-title",
                text="Graph Diffusion Networks",
                query_type="decomposed",
                action_type="title_search",
                priority=1,
                provider_hint="openalex",
            ),
            SubQuery(
                query_id="duplicate-title",
                text="  graph   diffusion networks ",
                query_type="decomposed",
                action_type="title_search",
                priority=2,
                provider_hint="openalex",
            ),
            SubQuery(
                query_id="structured-facet",
                text="graph diffusion retrieval",
                query_type="decomposed",
                priority=3,
                provider_hint="openalex",
            ),
        ],
        inherited_hard_filters={},
        rationale="test",
    )

    routes = FixedBudgetOpenAlexPolicy(max_openalex_calls=6).route(
        plan,
        original_query="Which paper proposed graph diffusion networks?",
    )

    assert [(item.method, item.action_type, item.search_mode) for item in routes] == [
        ("lexical_original", "text_search", "lexical"),
        ("semantic_original", "text_search", "semantic"),
        ("structured", "title_search", "lexical"),
        ("structured", "text_search", "lexical"),
    ]
    assert len(routes) <= 6
    identities = {
        (item.action_type, item.search_mode, " ".join(item.text.split()).casefold())
        for item in routes
    }
    assert len(identities) == len(routes)


def test_fixed_budget_policy_uses_s2_only_after_original_lexical_failure() -> None:
    routes = FixedBudgetOpenAlexPolicy(
        max_openalex_calls=6,
        semantic_scholar_fallback=True,
    ).route(
        _plan("either", "either", "either"),
        original_query="original query",
    )

    fallback_routes = [
        item for item in routes if "semantic_scholar" in item.providers
    ]
    assert len(fallback_routes) == 1
    assert fallback_routes[0].method == "lexical_original"
    assert fallback_routes[0].search_mode == "lexical"
    assert fallback_routes[0].routing_reason == "provider_fallback"
    assert sum(item.method == "semantic_original" for item in routes) == 1


def test_fixed_budget_policy_keeps_unused_structured_budget() -> None:
    plan = SearchPlan(
        subqueries=[
            SubQuery(
                query_id="only-duplicate",
                text="original query",
                query_type="exact",
                priority=1,
                provider_hint="either",
            )
        ],
        inherited_hard_filters={},
        rationale="test",
    )

    routes = FixedBudgetOpenAlexPolicy(max_openalex_calls=6).route(
        plan,
        original_query="original query",
    )

    assert [item.method for item in routes] == [
        "lexical_original",
        "semantic_original",
    ]
