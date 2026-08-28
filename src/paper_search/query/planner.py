"""Deterministic validation and clipping of query plans."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from paper_search.domain.models import QuerySpec, SearchPlan, SubQuery
from paper_search.query.semantic_actions import (
    PROTECTED_ACTION_PROMPT_VERSION,
    SEMANTIC_ACTION_PROMPT_VERSION,
    filter_lexical_action_candidates,
    filter_semantic_action_candidates,
)
from paper_search.query.protected_action_mix import select_protected_action_mix


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


def _semantic_rule_subqueries(spec: QuerySpec) -> list[SubQuery]:
    """Provide three no-drift fallbacks without reusing unverified model terms."""

    constraints = _target_constraints(spec)
    return [
        SubQuery(
            query_id="semantic-rule-exact",
            text=spec.original_query,
            query_type="exact",
            target_constraints=constraints,
            priority=1,
            provider_hint="either",
            search_mode="lexical",
        ),
        SubQuery(
            query_id="semantic-rule-original-semantic",
            text=spec.original_query,
            query_type="expanded",
            target_constraints=constraints,
            priority=2,
            provider_hint="either",
            search_mode="semantic",
        ),
        SubQuery(
            query_id="semantic-rule-original-title",
            text=spec.original_query,
            query_type="decomposed",
            action_type="title_search",
            target_constraints=constraints,
            priority=3,
            provider_hint="openalex",
            search_mode="lexical",
        ),
    ]


def _protected_rule_subqueries(spec: QuerySpec) -> list[SubQuery]:
    """Bounded lexical fail-safe actions used only to fill the five-slot bank."""

    rules = _rule_subqueries(spec)
    constraints = _target_constraints(spec)
    return [
        *rules,
        SubQuery(
            query_id="protected-rule-original-title",
            text=spec.original_query,
            query_type="decomposed",
            action_type="title_search",
            target_constraints=constraints,
            priority=4,
            provider_hint="openalex",
            search_mode="lexical",
        ),
    ]


class QueryPlanner:
    """Canonicalize a model plan or build a deterministic rules-only plan."""

    def __init__(
        self,
        *,
        prompt_version: str = "query-analyze-v1",
        soft_concept_evidence: Callable[[str], Iterable[str]] | None = None,
    ) -> None:
        self._semantic_actions = prompt_version == SEMANTIC_ACTION_PROMPT_VERSION
        self._protected_actions = prompt_version == PROTECTED_ACTION_PROMPT_VERSION
        self._soft_concept_evidence = soft_concept_evidence

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
            or max_subqueries < 3
        ):
            raise ValueError("max_subqueries must be an integer of at least 3")
        limit = min(max_subqueries, 5)

        candidates: list[SubQuery] = []
        if plan is not None:
            indexed = list(enumerate(plan.subqueries))
            indexed.sort(key=lambda item: (item[1].priority, item[0], item[1].query_id))
            candidates.extend(item for _, item in indexed)
        if self._protected_actions:
            evidence: tuple[str, ...] = ()
            if self._soft_concept_evidence is not None:
                try:
                    evidence = tuple(self._soft_concept_evidence(spec.original_query))
                except Exception:  # fail closed at the local model boundary
                    evidence = ()
            lexical = filter_lexical_action_candidates(
                spec,
                candidates,
                soft_concept_evidence=evidence,
            )
            lexical.extend(_protected_rule_subqueries(spec))
            lexical_plan = SearchPlan(
                subqueries=lexical[:limit],
                inherited_hard_filters={},
                rationale="Protected lexical action bank",
            )
            semantic = filter_semantic_action_candidates(
                spec,
                (item for item in candidates if item.search_mode == "semantic"),
                soft_concept_evidence=evidence,
            )
            semantic_plan = SearchPlan(
                subqueries=semantic or [_semantic_rule_subqueries(spec)[1]],
                inherited_hard_filters={},
                rationale="Single semantic challenger",
            )
            mixed = select_protected_action_mix(
                spec,
                lexical_plan,
                semantic_plan,
                max_planner_actions=limit,
            )
            candidates = [item.action for item in mixed.actions]
        elif self._semantic_actions:
            evidence: tuple[str, ...] = ()
            if self._soft_concept_evidence is not None:
                try:
                    evidence = tuple(self._soft_concept_evidence(spec.original_query))
                except Exception:  # fail closed at the local model boundary
                    evidence = ()
            fallback = _semantic_rule_subqueries(spec)
            candidates = [
                fallback[0],
                *filter_semantic_action_candidates(
                    spec,
                    candidates,
                    soft_concept_evidence=evidence,
                ),
                *fallback[1:],
            ]
        elif len(candidates) < 3:
            candidates.extend(_rule_subqueries(spec))

        selected: list[SubQuery] = []
        seen_text: set[tuple[str, str, str]] = set()
        for candidate in candidates:
            text = re.sub(r"\s+", " ", candidate.text).strip()
            key = (candidate.action_type, candidate.search_mode, text.casefold())
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
            if len(selected) == limit:
                break

        if len(selected) < 3:
            raise ValueError("query plan could not produce three distinct subqueries")
        return SearchPlan(
            subqueries=selected,
            inherited_hard_filters=_hard_filters(spec),
            rationale=plan.rationale if plan is not None else "Deterministic rule fallback",
        )
