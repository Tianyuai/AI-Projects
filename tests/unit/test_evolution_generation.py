from __future__ import annotations

import asyncio

import pytest

from paper_search.domain.models import QuerySpec
from paper_search.evolution import (
    ConstraintCoverage,
    ConstraintRef,
    CoverageReport,
    NoTargetedQueriesError,
    RoundPlan,
    RuleBasedNextRoundGenerator,
)


def query_spec() -> QuerySpec:
    return QuerySpec(original_query="q", research_goal="Find papers")


def coverage_report(statuses: list[str]) -> CoverageReport:
    items = [
        ConstraintCoverage(
            constraint=ConstraintRef(
                kind="methods", value=f"m{index}", normalized_value=f"m{index}"
            ),
            matched_candidate_ids=[],
            hit_count=0,
            status=status,
        )
        for index, status in enumerate(statuses, start=1)
    ]
    return CoverageReport(
        constraints=items,
        covered_count=statuses.count("covered"),
        low_coverage_count=statuses.count("low_coverage"),
        uncovered_count=statuses.count("uncovered"),
    )


def round_plan(round_number: int, texts: list[str]) -> RoundPlan:
    return RoundPlan(
        round_number=round_number,
        subqueries=[
            {
                "query_id": f"old-{index}",
                "text": text,
                "query_type": "decomposed",
                "priority": index,
                "provider_hint": "either",
            }
            for index, text in enumerate(texts, start=1)
        ],
    )


def test_targets_only_low_and_uncovered_constraints_in_stable_order() -> None:
    report = coverage_report(["covered", "low_coverage", "uncovered"])
    result = asyncio.run(
        RuleBasedNextRoundGenerator().generate(
            spec=query_spec(),
            coverage=report,
            prior_plans=[round_plan(1, ["already used"])],
            round_number=2,
            max_subqueries=2,
        )
    )

    assert result.round_number == 2
    assert [item.target_constraints for item in result.subqueries] == [["m2"], ["m3"]]
    assert [item.query_id for item in result.subqueries] == [
        "evolution-r2-q1",
        "evolution-r2-q2",
    ]


def test_excludes_previous_queries_after_normalizing_whitespace_and_case() -> None:
    report = coverage_report(["uncovered", "uncovered"])
    result = asyncio.run(
        RuleBasedNextRoundGenerator().generate(
            spec=QuerySpec(original_query="q", research_goal="Find   PAPERS"),
            coverage=report,
            prior_plans=[round_plan(1, [" find papers   M1 "])],
            round_number=2,
            max_subqueries=2,
        )
    )

    assert [item.target_constraints for item in result.subqueries] == [["m2"]]


def test_clips_generation_to_max_subqueries() -> None:
    result = asyncio.run(
        RuleBasedNextRoundGenerator().generate(
            spec=query_spec(),
            coverage=coverage_report(["uncovered", "uncovered"]),
            prior_plans=[],
            round_number=2,
            max_subqueries=1,
        )
    )

    assert len(result.subqueries) == 1


def test_raises_when_every_targeted_query_was_already_used() -> None:
    with pytest.raises(NoTargetedQueriesError, match="no unique targeted queries remain"):
        asyncio.run(
            RuleBasedNextRoundGenerator().generate(
                spec=query_spec(),
                coverage=coverage_report(["uncovered"]),
                prior_plans=[round_plan(1, ["find papers m1"])],
                round_number=2,
                max_subqueries=2,
            )
        )


@pytest.mark.parametrize("value", [True, False, 1.0, 0, -1])
def test_max_subqueries_requires_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            RuleBasedNextRoundGenerator().generate(
                spec=query_spec(),
                coverage=coverage_report(["uncovered"]),
                prior_plans=[],
                round_number=2,
                max_subqueries=value,  # type: ignore[arg-type]
            )
        )

