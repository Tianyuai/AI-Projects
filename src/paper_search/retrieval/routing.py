"""Deterministic bounded baseline routing across scholarly providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

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


class RetrievalRoute(DomainModel):
    """One provider action emitted by a replaceable retrieval policy."""

    route_id: NonEmptyStr
    source_subquery_id: NonEmptyStr
    text: NonEmptyStr
    action_type: Literal["text_search", "title_search"] = "text_search"
    search_mode: Literal["lexical", "semantic"]
    method: Literal["lexical_original", "semantic_original", "structured"] = (
        "structured"
    )
    providers: tuple[Literal["openalex"], ...] | tuple[
        Literal["openalex"], Literal["semantic_scholar"]
    ]
    routing_reason: Literal["primary", "provider_fallback"]


class RetrievalPolicy(Protocol):
    """Boundary used by orchestration to obtain bounded retrieval actions."""

    def route(
        self,
        plan: SearchPlan,
        *,
        original_query: str | None = None,
    ) -> list[RetrievalRoute]: ...


@dataclass(frozen=True)
class FixedHybridOpenAlexPolicy:
    """Run paired OpenAlex lexical/semantic actions; keep S2 as fallback only."""

    min_openalex_calls: int = 3
    max_openalex_calls: int = 6
    max_semantic_scholar_calls: int = 2

    def __post_init__(self) -> None:
        _validate_bounds(
            self.min_openalex_calls,
            self.max_openalex_calls,
            self.max_semantic_scholar_calls,
        )
        if self.max_openalex_calls < self.min_openalex_calls * 2:
            raise ValueError(
                "fixed hybrid policy requires room for lexical and semantic actions"
            )

    def route(
        self,
        plan: SearchPlan,
        *,
        original_query: str | None = None,
    ) -> list[RetrievalRoute]:
        del original_query
        ordered = _ordered(plan.subqueries)
        query_ids = [item.query_id for item in plan.subqueries]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("routed subquery IDs must be unique")
        selected_count = min(
            len(ordered),
            self.max_openalex_calls // 2,
        )
        if selected_count < self.min_openalex_calls:
            raise ValueError("plan cannot satisfy the OpenAlex minimum")

        routes: list[RetrievalRoute] = []
        fallback_slots = self.max_semantic_scholar_calls
        for item in ordered[:selected_count]:
            lexical_fallback = fallback_slots > 0
            routes.append(
                RetrievalRoute(
                    route_id=f"{item.query_id}:lexical",
                    source_subquery_id=item.query_id,
                    text=item.text,
                    search_mode="lexical",
                    providers=(
                        ("openalex", "semantic_scholar")
                        if lexical_fallback
                        else ("openalex",)
                    ),
                    routing_reason=(
                        "provider_fallback" if lexical_fallback else "primary"
                    ),
                )
            )
            if lexical_fallback:
                fallback_slots -= 1
            routes.append(
                RetrievalRoute(
                    route_id=f"{item.query_id}:semantic",
                    source_subquery_id=item.query_id,
                    text=item.text,
                    search_mode="semantic",
                    providers=("openalex",),
                    routing_reason="primary",
                )
            )
        return routes


@dataclass(frozen=True)
class FixedBudgetOpenAlexPolicy:
    """Reserve two original-query anchors and at most four structured actions."""

    max_openalex_calls: int = 6
    semantic_scholar_fallback: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.max_openalex_calls) is not int
            or not 2 <= self.max_openalex_calls <= 6
        ):
            raise ValueError("max_openalex_calls must be an integer between 2 and 6")
        if type(self.semantic_scholar_fallback) is not bool:
            raise ValueError("semantic_scholar_fallback must be a boolean")

    def route(
        self,
        plan: SearchPlan,
        *,
        original_query: str | None = None,
    ) -> list[RetrievalRoute]:
        query = " ".join((original_query or "").split())
        if not query:
            exact = next(
                (item for item in _ordered(plan.subqueries) if item.query_type == "exact"),
                None,
            )
            source = exact or _ordered(plan.subqueries)[0]
            query = " ".join(source.text.split())
        original_source = next(
            (
                item.query_id
                for item in _ordered(plan.subqueries)
                if _normalized_text(item.text) == _normalized_text(query)
            ),
            "original-query",
        )
        lexical_providers = (
            ("openalex", "semantic_scholar")
            if self.semantic_scholar_fallback
            else ("openalex",)
        )
        routes = [
            RetrievalRoute(
                route_id="original-query:lexical",
                source_subquery_id=original_source,
                text=query,
                action_type="text_search",
                search_mode="lexical",
                method="lexical_original",
                providers=lexical_providers,
                routing_reason=(
                    "provider_fallback"
                    if self.semantic_scholar_fallback
                    else "primary"
                ),
            ),
            RetrievalRoute(
                route_id="original-query:semantic",
                source_subquery_id=original_source,
                text=query,
                action_type="text_search",
                search_mode="semantic",
                method="semantic_original",
                providers=("openalex",),
                routing_reason="primary",
            ),
        ]
        seen = {_route_identity(item) for item in routes}
        structured_limit = min(4, self.max_openalex_calls - len(routes))
        for item in _ordered(plan.subqueries):
            if item.search_mode != "lexical":
                continue
            identity = (
                item.action_type,
                item.search_mode,
                _normalized_text(item.text),
            )
            if identity in seen:
                continue
            seen.add(identity)
            routes.append(
                RetrievalRoute(
                    route_id=f"{item.query_id}:structured",
                    source_subquery_id=item.query_id,
                    text=" ".join(item.text.split()),
                    action_type=item.action_type,
                    search_mode=item.search_mode,
                    method="structured",
                    providers=("openalex",),
                    routing_reason="primary",
                )
            )
            if len(routes) == 2 + structured_limit:
                break
        return routes


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _route_identity(route: RetrievalRoute) -> tuple[str, str, str]:
    return route.action_type, route.search_mode, _normalized_text(route.text)


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
    if len(selected) < min_openalex_calls:
        raise ValueError("plan cannot satisfy the OpenAlex minimum")
    query_ids = [item.query_id for item in plan.subqueries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("routed subquery IDs must be unique")

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


__all__ = [
    "FixedBudgetOpenAlexPolicy",
    "FixedHybridOpenAlexPolicy",
    "RetrievalPolicy",
    "RetrievalRoute",
    "RoutedSubquery",
    "route_baseline_subqueries",
]
