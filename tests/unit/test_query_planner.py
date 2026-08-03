from paper_search.domain.models import QuerySpec, SearchPlan, SubQuery
from paper_search.query.planner import QueryPlanner


def _spec() -> QuerySpec:
    return QuerySpec(
        original_query="graph retrieval without surveys at NeurIPS from 2021 to 2024",
        research_goal="Find graph retrieval methods",
        topics=["graph retrieval"],
        methods=["message passing"],
        datasets=["OpenGraph"],
        year_from=2021,
        year_to=2024,
        venues=["NeurIPS"],
        must_have=["graph retrieval"],
        exclusions=["surveys"],
    )


def test_finalize_is_deterministic_clipped_and_inherits_hard_constraints() -> None:
    source = SearchPlan(
        subqueries=[
            SubQuery(
                query_id="z",
                text="graph retrieval",
                query_type="exact",
                target_constraints=["graph retrieval"],
                priority=2,
                provider_hint="either",
            ),
            SubQuery(
                query_id="a",
                text="message passing OpenGraph",
                query_type="decomposed",
                target_constraints=["message passing", "OpenGraph"],
                priority=1,
                provider_hint="semantic_scholar",
            ),
            SubQuery(
                query_id="duplicate",
                text=" GRAPH   RETRIEVAL ",
                query_type="expanded",
                target_constraints=["graph retrieval"],
                priority=3,
                provider_hint="openalex",
            ),
            SubQuery(
                query_id="extra-1",
                text="graph neural search",
                query_type="expanded",
                target_constraints=["graph retrieval"],
                priority=4,
                provider_hint="either",
            ),
            SubQuery(
                query_id="extra-2",
                text="scientific graph search",
                query_type="expanded",
                target_constraints=["graph retrieval"],
                priority=5,
                provider_hint="either",
            ),
        ],
        inherited_hard_filters={"year_from": 1900},
        rationale="fixture",
    )
    planner = QueryPlanner()

    first = planner.finalize(_spec(), source, max_subqueries=4)
    second = planner.finalize(_spec(), source, max_subqueries=4)

    assert first == second
    assert 3 <= len(first.subqueries) <= 4
    assert [item.query_id for item in first.subqueries] == ["sq-1", "sq-2", "sq-3", "sq-4"]
    assert [item.text for item in first.subqueries][:2] == [
        "message passing OpenGraph",
        "graph retrieval",
    ]
    assert first.inherited_hard_filters == {
        "year_from": 2021,
        "year_to": 2024,
        "venues": ["NeurIPS"],
    }
    assert _spec().exclusions == ["surveys"]
    assert all(item.target_constraints for item in first.subqueries)


def test_finalize_builds_three_rule_queries_when_model_plan_is_missing() -> None:
    plan = QueryPlanner().finalize(_spec(), None, max_subqueries=5)

    assert len(plan.subqueries) >= 3
    assert plan.subqueries[0].text == _spec().original_query
    assert {item.query_type for item in plan.subqueries} >= {"exact", "decomposed"}
    assert all(item.provider_hint in {"openalex", "semantic_scholar", "either"} for item in plan.subqueries)


def test_finalize_rejects_invalid_max_subqueries() -> None:
    planner = QueryPlanner()

    try:
        planner.finalize(_spec(), None, max_subqueries=2)
    except ValueError as error:
        assert "at least 3" in str(error)
    else:
        raise AssertionError("max_subqueries below three must be rejected")


def test_finalize_clamps_caps_above_five() -> None:
    plan = QueryPlanner().finalize(_spec(), None, max_subqueries=99)

    assert len(plan.subqueries) == 3
