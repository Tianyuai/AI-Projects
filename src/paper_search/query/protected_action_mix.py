"""Gold-blind protection of useful lexical actions during semantic replacement."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from paper_search.domain.models import QuerySpec, SearchPlan, SubQuery


_TOKEN = re.compile(r"[A-Za-z0-9]+")
_GENERIC_RETRIEVAL_TERMS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "article",
        "articles",
        "by",
        "find",
        "for",
        "from",
        "introduced",
        "introducing",
        "introduction",
        "of",
        "on",
        "origin",
        "paper",
        "papers",
        "publication",
        "publications",
        "provide",
        "research",
        "responsible",
        "study",
        "studies",
        "the",
        "title",
        "to",
        "using",
        "which",
        "who",
        "work",
        "works",
    }
)


@dataclass(frozen=True)
class SelectedAction:
    """One selected action with its frozen-receipt origin."""

    origin: Literal["production", "semantic"]
    source_query_id: str
    action: SubQuery


@dataclass(frozen=True)
class ProtectedActionMix:
    """Auditable result of a Gold-blind single-slot replacement decision."""

    actions: tuple[SelectedAction, ...]
    protected_production_ids: tuple[str, ...]
    replaced_production_ids: tuple[str, ...]
    added_semantic_ids: tuple[str, ...]


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _technical_terms(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _TOKEN.findall(value.casefold())
        if token not in _GENERIC_RETRIEVAL_TERMS
        and (len(token) >= 3 or any(character.isdigit() for character in token))
    )


def _technical_evidence(value: str) -> set[str]:
    terms = _technical_terms(value)
    evidence = {f"term:{term}" for term in terms}
    evidence.update(
        f"phrase:{left} {right}" for left, right in zip(terms, terms[1:])
    )
    return evidence


def _is_bridge(action: SubQuery) -> bool:
    return "supervised-lexical-bridge" in action.query_id.casefold()


def _ordered(plan: SearchPlan) -> list[SubQuery]:
    return sorted(plan.subqueries, key=lambda item: (item.priority, item.query_id))


def _semantic_candidates(
    spec: QuerySpec,
    plan: SearchPlan,
    production: list[SubQuery],
) -> list[SubQuery]:
    original = _normalized(spec.original_query)
    production_identities = {
        (item.search_mode, _normalized(item.text)) for item in production
    }
    production_texts = {_normalized(item.text) for item in production}
    candidates: list[SubQuery] = []
    seen: set[tuple[str, str]] = set()
    for item in _ordered(plan):
        identity = (item.search_mode, _normalized(item.text))
        if (
            _is_bridge(item)
            or _normalized(item.text) == original
            or (item.action_type == "title_search" and _normalized(item.text) == original)
            or _normalized(item.text) in production_texts
            or identity in production_identities
            or identity in seen
        ):
            continue
        seen.add(identity)
        candidates.append(item)
    return candidates


def _independent_production_ids(actions: list[SubQuery]) -> set[str]:
    evidence = [_technical_evidence(item.text) for item in actions]
    protected: set[str] = set()
    for index, item in enumerate(actions):
        other_evidence = set().union(
            *(value for other_index, value in enumerate(evidence) if other_index != index)
        )
        if evidence[index].difference(other_evidence):
            protected.add(item.query_id)
    return protected


def _redundancy(action: SubQuery, others: list[SubQuery]) -> float:
    terms = set(_technical_terms(action.text))
    if not terms:
        return 1.0
    maximum = 0.0
    for other in others:
        other_terms = set(_technical_terms(other.text))
        union = terms.union(other_terms)
        if union:
            maximum = max(maximum, len(terms.intersection(other_terms)) / len(union))
    return maximum


def _best_novel_candidate(
    candidates: list[SubQuery],
    production: list[SubQuery],
) -> SubQuery | None:
    production_terms = {
        term for item in production for term in _technical_terms(item.text)
    }
    ranked: list[tuple[int, int, int, str, SubQuery]] = []
    for item in candidates:
        terms = set(_technical_terms(item.text))
        novel_count = len(terms.difference(production_terms))
        if novel_count == 0:
            continue
        ranked.append(
            (
                -novel_count,
                len(terms),
                item.priority,
                item.query_id,
                item,
            )
        )
    return min(ranked)[-1] if ranked else None


def select_protected_action_mix(
    spec: QuerySpec,
    production_plan: SearchPlan,
    semantic_plan: SearchPlan,
    *,
    max_planner_actions: int = 5,
) -> ProtectedActionMix:
    """Keep independent lexical coverage and expose only one replacement slot.

    Selection is based exclusively on query/action text and never on candidates or
    relevance labels. Negation queries retain the production plan unchanged.
    """

    if (
        isinstance(max_planner_actions, bool)
        or not isinstance(max_planner_actions, int)
        or max_planner_actions < 1
    ):
        raise ValueError("max_planner_actions must be a positive integer")
    production = [
        item for item in _ordered(production_plan) if not _is_bridge(item)
    ][:max_planner_actions]
    protected = _independent_production_ids(production)
    selected = [
        SelectedAction("production", item.query_id, item) for item in production
    ]
    if spec.exclusions:
        return ProtectedActionMix(
            actions=tuple(selected),
            protected_production_ids=tuple(
                item.query_id for item in production if item.query_id in protected
            ),
            replaced_production_ids=(),
            added_semantic_ids=(),
        )

    candidates = _semantic_candidates(spec, semantic_plan, production)
    free_slots = max_planner_actions - len(selected)
    if free_slots > 0:
        additions = candidates[:free_slots]
        selected.extend(
            SelectedAction("semantic", item.query_id, item) for item in additions
        )
        return ProtectedActionMix(
            actions=tuple(selected),
            protected_production_ids=tuple(
                item.query_id for item in production if item.query_id in protected
            ),
            replaced_production_ids=(),
            added_semantic_ids=tuple(item.query_id for item in additions),
        )

    challenger = _best_novel_candidate(candidates, production)
    replaceable = [item for item in production if item.query_id not in protected]
    if challenger is None or not replaceable:
        return ProtectedActionMix(
            actions=tuple(selected),
            protected_production_ids=tuple(
                item.query_id for item in production if item.query_id in protected
            ),
            replaced_production_ids=(),
            added_semantic_ids=(),
        )

    victim = max(
        replaceable,
        key=lambda item: (
            _redundancy(item, [other for other in production if other is not item]),
            item.priority,
            item.query_id,
        ),
    )
    selected = [
        item
        for item in selected
        if not (
            item.origin == "production" and item.source_query_id == victim.query_id
        )
    ]
    selected.append(SelectedAction("semantic", challenger.query_id, challenger))
    return ProtectedActionMix(
        actions=tuple(selected),
        protected_production_ids=tuple(
            item.query_id for item in production if item.query_id in protected
        ),
        replaced_production_ids=(victim.query_id,),
        added_semantic_ids=(challenger.query_id,),
    )


__all__ = [
    "ProtectedActionMix",
    "SelectedAction",
    "select_protected_action_mix",
]
