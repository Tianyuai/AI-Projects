"""Deterministic validation and clipping of query plans."""

from __future__ import annotations

import re
from collections.abc import Iterable

from paper_search.domain.models import QuerySpec, SearchPlan, SubQuery


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _hard_filters(spec: QuerySpec) -> dict[str, object]:
    filters: dict[str, object] = {}
    if spec.year_from is not None:
        filters["year_from"] = spec.year_from
    if spec.year_to is not None:
        filters["year_to"] = spec.year_to
    if spec.venues:
        filters["venues"] = list(spec.venues)
    if spec.exclusions:
        filters["exclusions"] = list(spec.exclusions)
    return filters


def _target_constraints(spec: QuerySpec) -> list[str]:
    values = (
        spec.must_have
        + spec.topics
        + spec.methods
        + spec.tasks
        + spec.datasets
        + spec.domains
        + spec.venues
    )
    return _ordered_unique(values) or [spec.research_goal]


def _rule_subqueries(spec: QuerySpec) -> list[SubQuery]:
    constraints = _target_constraints(spec)
    expanded_terms = _ordered_unique(
        spec.topics + spec.methods + spec.tasks + spec.datasets + spec.domains + spec.should_have
    )
    expanded = " ".join(expanded_terms) or f"{spec.original_query} scholarly papers"
    if expanded.casefold() == spec.original_query.casefold():
        expanded = f"{spec.original_query} scholarly papers"
    decomposed_terms = _ordered_unique(spec.must_have + spec.datasets + spec.venues)
    decomposed = " ".join(decomposed_terms) or f"{spec.original_query} methods"
    if decomposed.casefold() in {spec.original_query.casefold(), expanded.casefold()}:
        decomposed = f"{spec.original_query} methods"
    return [
        SubQuery(
            query_id="rule-exact",
            text=spec.original_query,
            query_type="exact",
            target_constraints=constraints,
            priority=1,
            provider_hint="either",
        ),
        SubQuery(
            query_id="rule-expanded",
            text=expanded,
            query_type="expanded",
            target_constraints=constraints,
            priority=2,
            provider_hint="openalex",
        ),
        SubQuery(
            query_id="rule-decomposed",
            text=decomposed,
            query_type="decomposed",
            target_constraints=constraints,
            priority=3,
            provider_hint="semantic_scholar",
        ),
    ]


class QueryPlanner:
    """Canonicalize a model plan or build a deterministic rules-only plan."""

    def finalize(
        self,
        spec: QuerySpec,
        plan: SearchPlan | None,
        *,
        max_subqueries: int = 5,
    ) -> SearchPlan:
        if (
            isinstance(max_subqueries, bool)
            or not isinstance(max_subqueries, int)
            or not 3 <= max_subqueries <= 5
        ):
            raise ValueError("max_subqueries must be an integer between 3 and 5")

        candidates: list[SubQuery] = []
        if plan is not None:
            indexed = list(enumerate(plan.subqueries))
            indexed.sort(key=lambda item: (item[1].priority, item[0], item[1].query_id))
            candidates.extend(item for _, item in indexed)
        if len(candidates) < 3:
            candidates.extend(_rule_subqueries(spec))

        selected: list[SubQuery] = []
        seen_text: set[str] = set()
        for candidate in candidates:
            text = re.sub(r"\s+", " ", candidate.text).strip()
            key = text.casefold()
            if key in seen_text:
                continue
            seen_text.add(key)
            constraints = _ordered_unique(candidate.target_constraints) or _target_constraints(spec)
            selected.append(
                candidate.model_copy(
                    update={
                        "query_id": f"sq-{len(selected) + 1}",
                        "text": text,
                        "target_constraints": constraints,
                        "priority": len(selected) + 1,
                    }
                )
            )
            if len(selected) == max_subqueries:
                break

        if len(selected) < 3:
            raise ValueError("query plan could not produce three distinct subqueries")
        return SearchPlan(
            subqueries=selected,
            inherited_hard_filters=_hard_filters(spec),
            rationale=plan.rationale if plan is not None else "Deterministic rule fallback",
        )
