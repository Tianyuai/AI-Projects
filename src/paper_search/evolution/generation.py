from collections.abc import Sequence
from typing import Protocol

from paper_search.domain.models import QuerySpec, SubQuery

from .models import CoverageReport, RoundPlan


class NoTargetedQueriesError(ValueError):
    pass


class NextRoundGenerator(Protocol):
    async def generate(
        self,
        *,
        spec: QuerySpec,
        coverage: CoverageReport,
        prior_plans: Sequence[RoundPlan],
        round_number: int,
        max_subqueries: int,
    ) -> RoundPlan: ...


def _normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class RuleBasedNextRoundGenerator:
    async def generate(
        self,
        *,
        spec: QuerySpec,
        coverage: CoverageReport,
        prior_plans: Sequence[RoundPlan],
        round_number: int,
        max_subqueries: int,
    ) -> RoundPlan:
        _positive_integer(round_number, "round_number")
        _positive_integer(max_subqueries, "max_subqueries")

        used_texts = {
            _normalize(subquery.text)
            for plan in prior_plans
            for subquery in plan.subqueries
        }
        selected: list[SubQuery] = []
        for coverage_item in coverage.constraints:
            if coverage_item.status not in ("low_coverage", "uncovered"):
                continue
            text = f"{spec.research_goal} {coverage_item.constraint.value}"
            if _normalize(text) in used_texts:
                continue
            selected.append(
                SubQuery(
                    query_id=f"evolution-r{round_number}-q{len(selected) + 1}",
                    text=text,
                    query_type="decomposed",
                    target_constraints=[coverage_item.constraint.value],
                    priority=len(selected) + 1,
                    provider_hint="either",
                )
            )
            used_texts.add(_normalize(text))
            if len(selected) == max_subqueries:
                break

        if not selected:
            raise NoTargetedQueriesError("no unique targeted queries remain")
        return RoundPlan(round_number=round_number, subqueries=selected)
