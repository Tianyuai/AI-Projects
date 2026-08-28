"""Gold-blind OpenAlex syntax compiled from the existing LLM QuerySpec."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from paper_search.domain.models import QuerySpec
from paper_search.learning.candidates import query_content_terms
from paper_search.learning.query_constraint_profile import profile_query_constraints
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    TextSearchAction,
    TextSearchPayload,
)


_ALIAS_ACTION_ID = "structured-openalex-entity-alias-v1"
_PHRASE_ACTION_ID = "structured-openalex-phrase-proximity-v1"
_ENTITY_ACTION_ID = "structured-openalex-entity-v1"
_ALIAS_TOKEN = r"[A-Za-z][A-Za-z0-9+_.-]{1,15}"
_UNSAFE_SYNTAX = re.compile(r'["()~*?\\]+')
_TECHNICAL_SINGLE = re.compile(r"^(?=.*(?:[A-Z0-9+_.-]))[A-Za-z0-9+_.-]{2,40}$")
_GENERIC_TERMS = frozenset(
    {
        "analysis",
        "approach",
        "approaches",
        "data",
        "learning",
        "method",
        "methods",
        "model",
        "models",
        "paper",
        "papers",
        "research",
        "study",
        "studies",
        "survey",
        "surveys",
        "system",
        "systems",
    }
)


@dataclass(frozen=True)
class StructuredOpenAlexProposal:
    """One deterministic provider query derived only from parsed query slots."""

    action_id: str
    strategy: str
    query_text: str
    source_slot: str
    source_value: str


def _clean_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(_UNSAFE_SYNTAX.sub(" ", normalized).split()).strip(" ,.;:!")


def _contains_sequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    width = len(needle)
    return bool(width) and any(
        list(haystack[index : index + width]) == list(needle)
        for index in range(len(haystack) - width + 1)
    )


def _grounded_values(spec: QuerySpec) -> list[tuple[str, str]]:
    original_terms = tuple(query_content_terms(spec.original_query))
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    slot_values = (
        ("dataset", spec.datasets),
        ("method", spec.methods),
        ("task", spec.tasks),
        ("must_have", spec.must_have),
    )
    for slot, values in slot_values:
        for raw in values:
            value = _clean_value(raw)
            folded = value.casefold()
            terms = tuple(query_content_terms(value))
            if (
                not value
                or folded in seen
                or len(value) > 80
                or not _contains_sequence(original_terms, terms)
            ):
                continue
            seen.add(folded)
            result.append((slot, value))
    return result


def _explicit_alias(original_query: str, value: str) -> str | None:
    escaped = re.escape(value)
    forward = re.search(
        rf"(?i)(?<![A-Za-z0-9]){escaped}\s*\(\s*({_ALIAS_TOKEN})\s*\)",
        original_query,
    )
    if forward is not None:
        return _clean_value(forward.group(1))
    reverse = re.search(
        rf"(?i)(?<![A-Za-z0-9])({_ALIAS_TOKEN})\s*\(\s*{escaped}\s*\)",
        original_query,
    )
    if reverse is not None:
        return _clean_value(reverse.group(1))
    return None


def _phrase_proposal(slot: str, value: str) -> StructuredOpenAlexProposal | None:
    terms = tuple(query_content_terms(value))
    if not terms or all(term in _GENERIC_TERMS for term in terms):
        return None
    if len(terms) == 1:
        if not _TECHNICAL_SINGLE.fullmatch(value):
            return None
        return StructuredOpenAlexProposal(
            action_id=_ENTITY_ACTION_ID,
            strategy="llm-query-spec:technical-entity-v1",
            query_text=f'"{value}"',
            source_slot=slot,
            source_value=value,
        )
    if len(terms) > 6:
        return None
    return StructuredOpenAlexProposal(
        action_id=_PHRASE_ACTION_ID,
        strategy="llm-query-spec:phrase-proximity-v1",
        query_text=f'"{value}"~5',
        source_slot=slot,
        source_value=value,
    )


def propose_structured_openalex_supplement(
    spec: QuerySpec,
) -> StructuredOpenAlexProposal | None:
    """Compile one grounded action, abstaining for all negation queries."""

    profile = profile_query_constraints(spec)
    if profile.has_negation or profile.is_title_like:
        return None
    grounded = _grounded_values(spec)
    if not grounded:
        return None
    for slot, value in grounded:
        alias = _explicit_alias(spec.original_query, value)
        if alias is not None and alias.casefold() != value.casefold():
            return StructuredOpenAlexProposal(
                action_id=_ALIAS_ACTION_ID,
                strategy="llm-query-spec:entity-alias-v1",
                query_text=f'("{value}" OR "{alias}")',
                source_slot=slot,
                source_value=value,
            )
    for slot, value in grounded:
        proposal = _phrase_proposal(slot, value)
        if proposal is not None:
            return proposal
    return None


def build_structured_openalex_action_batch(
    proposal: StructuredOpenAlexProposal,
) -> RecallActionBatch:
    return RecallActionBatch(
        actions=[
            TextSearchAction(
                action_id=proposal.action_id,
                strategy=proposal.strategy,
                action_type="text_search",
                payload=TextSearchPayload(
                    query_text=proposal.query_text,
                    search_mode="lexical",
                ),
            )
        ]
    )


__all__ = [
    "StructuredOpenAlexProposal",
    "build_structured_openalex_action_batch",
    "propose_structured_openalex_supplement",
]
