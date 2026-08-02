"""Deterministic bounded baseline routing across scholarly providers."""

from __future__ import annotations

from typing import Literal

from paper_search.domain.models import DomainModel, NonEmptyStr, SearchPlan, SubQuery


class RoutedSubquery(DomainModel):
    subquery_id: NonEmptyStr
    text: NonEmptyStr
    providers: tuple[Literal["openalex"], ...] | tuple[
        Literal["openalex"], Literal["semantic_scholar"]
    ]
    routing_reason: Literal[
        "primary",
        "high_priority_supplement",
        "uncovered_constraint_supplement",
    ]


def _validate_bounds(
    min_openalex_calls: int,
    max_openalex_calls: int,
    max_semantic_scholar_calls: int,
) -> None:
    if type(min_openalex_calls) is not int or not 1 <= min_openalex_calls <= 6:
        raise ValueError("min_openalex_calls must be an integer between 1 and 6")
    if type(max_openalex_calls) is not int or not 1 <= max_openalex_calls <= 6:
        raise ValueError("max_openalex_calls must be an integer between 1 and 6")
    if min_openalex_calls > max_openalex_calls:
        raise ValueError("min_openalex_calls must not exceed max_openalex_calls")
    if (
        type(max_semantic_scholar_calls) is not int
        or not 0 <= max_semantic_scholar_calls <= 2
    ):
        raise ValueError(
            "max_semantic_scholar_calls must be an integer between 0 and 2"
        )


def _ordered(subqueries: list[SubQuery]) -> list[SubQuery]:
    return sorted(subqueries, key=lambda item: (item.priority, item.query_id))


def route_baseline_subqueries(
    plan: SearchPlan,
    *,
    min_openalex_calls: int = 3,
    max_openalex_calls: int = 6,
    max_semantic_scholar_calls: int = 2,
) -> list[RoutedSubquery]:
    """Route a stable OpenAlex baseline with narrowly targeted S2 supplements."""

    _validate_bounds(
        min_openalex_calls,
        max_openalex_calls,
        max_semantic_scholar_calls,
    )
    selected = _ordered(plan.subqueries)[:max_openalex_calls]
    if len(plan.subqueries) >= min_openalex_calls and len(selected) < min_openalex_calls:
        raise ValueError("plan cannot satisfy the OpenAlex minimum")

    supplemented: dict[str, str] = {}
    explicit = [
        item for item in selected if item.provider_hint == "semantic_scholar"
    ]
    for item in explicit[:max_semantic_scholar_calls]:
        supplemented[item.query_id] = "high_priority_supplement"

    remaining = max_semantic_scholar_calls - len(supplemented)
    covered_constraints: set[str] = set()
    for item in selected:
        if item.provider_hint != "either":
            covered_constraints.update(item.target_constraints)
            continue
        new_constraints = set(item.target_constraints).difference(covered_constraints)
        if remaining > 0 and new_constraints:
            supplemented[item.query_id] = "uncovered_constraint_supplement"
            remaining -= 1
        covered_constraints.update(item.target_constraints)

    routed: list[RoutedSubquery] = []
    for item in selected:
        reason = supplemented.get(item.query_id)
        if reason is None:
            routed.append(
                RoutedSubquery(
                    subquery_id=item.query_id,
                    text=item.text,
                    providers=("openalex",),
                    routing_reason="primary",
                )
            )
        else:
            routed.append(
                RoutedSubquery(
                    subquery_id=item.query_id,
                    text=item.text,
                    providers=("openalex", "semantic_scholar"),
                    routing_reason=reason,
                )
            )
    return routed


__all__ = ["RoutedSubquery", "route_baseline_subqueries"]
