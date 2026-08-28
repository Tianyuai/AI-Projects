"""Gold-blind selection of one finalized LLM action as recall supplement."""

from __future__ import annotations

import re
from dataclasses import dataclass

from paper_search.domain.models import QuerySpec, SearchPlan, SubQuery
from paper_search.learning.adaptive_openalex_recall import OpenAlexRecallDecision
from paper_search.query.semantic_actions import preserves_explicit_hard_constraints


_TOKEN = re.compile(r"[A-Za-z0-9]+")
_GENERIC = frozenset(
    {
        "a",
        "an",
        "and",
        "article",
        "articles",
        "find",
        "for",
        "from",
        "of",
        "on",
        "paper",
        "papers",
        "provide",
        "research",
        "study",
        "studies",
        "the",
        "to",
        "using",
        "which",
        "work",
        "works",
    }
)


@dataclass(frozen=True)
class LowConfidenceLLMActionSelection:
    """One supplemental action selected without labels or candidate relevance."""

    source_query_id: str
    action: SubQuery
    novel_phrase_count: int
    novel_term_count: int


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _terms(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _TOKEN.findall(value.casefold())
        if token not in _GENERIC
        and (len(token) >= 3 or any(character.isdigit() for character in token))
    )


def _phrases(value: str) -> set[tuple[str, str]]:
    terms = _terms(value)
    return set(zip(terms, terms[1:]))


def _is_bridge(action: SubQuery) -> bool:
    return "supervised-lexical-bridge" in action.query_id.casefold()


def select_low_confidence_llm_action(
    spec: QuerySpec,
    production_plan: SearchPlan,
    candidate_plan: SearchPlan,
    decision: OpenAlexRecallDecision,
) -> LowConfidenceLLMActionSelection | None:
    """Choose one phrase-novel lexical action while preserving the base plan.

    ``candidate_plan`` must already have passed the v3 hard-constraint and soft
    evidence gates.  This function only decides which safe action earns the
    bounded supplemental slot; it never inspects Gold labels or paper contents.
    """

    if not decision.low_confidence or spec.exclusions:
        return None
    production = [
        item
        for item in production_plan.subqueries
        if not _is_bridge(item) and item.action_type == "text_search"
    ]
    production_identities = {
        (item.search_mode, _normalized(item.text)) for item in production
    }
    production_terms = {term for item in production for term in _terms(item.text)}
    production_phrases = {
        phrase for item in production for phrase in _phrases(item.text)
    }
    original = _normalized(spec.original_query)
    ranked: list[tuple[int, int, int, int, int, str, SubQuery]] = []
    seen: set[tuple[str, str]] = set()
    for item in candidate_plan.subqueries:
        identity = (item.search_mode, _normalized(item.text))
        if (
            _is_bridge(item)
            or item.action_type != "text_search"
            or item.search_mode != "lexical"
            or identity[1] == original
            or identity in production_identities
            or identity in seen
            or not preserves_explicit_hard_constraints(spec, item.text)
        ):
            continue
        seen.add(identity)
        terms = set(_terms(item.text))
        novel_phrases = len(_phrases(item.text).difference(production_phrases))
        novel_terms = len(terms.difference(production_terms))
        if novel_phrases == 0 and novel_terms == 0:
            continue
        ranked.append(
            (
                0 if item.provider_hint == "semantic_scholar" else 1,
                -novel_phrases,
                -novel_terms,
                len(terms),
                item.priority,
                item.query_id,
                item,
            )
        )
    if not ranked:
        return None
    best = min(ranked)
    action = best[-1]
    return LowConfidenceLLMActionSelection(
        source_query_id=action.query_id,
        action=action,
        novel_phrase_count=-best[1],
        novel_term_count=-best[2],
    )


__all__ = [
    "LowConfidenceLLMActionSelection",
    "select_low_confidence_llm_action",
]
