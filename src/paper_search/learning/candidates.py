"""Deterministic, bounded action candidates shared by training and inference."""

from __future__ import annotations

import re

from paper_search.domain.models import QuerySpec
from paper_search.learning.contracts import (
    PolicyActionCandidate,
    QueryKind,
)


_STOPWORDS = frozenset(
    {
        "a", "about", "an", "and", "any", "are", "can", "could", "does",
        "find", "for", "from", "give", "have", "in", "is", "me", "mention", "of",
        "on", "paper", "papers", "please", "provide", "research", "show",
        "some", "studies", "study", "that", "the", "to", "what", "which",
        "who", "with", "work", "works", "you", "related",
    }
)
_TITLE_SEEKING = re.compile(
    r"\b(?:which|what)\s+(?:paper|study|work)\b.*\b(?:introduced|proposed|presented|developed)\b",
    flags=re.IGNORECASE,
)
_PARENTHETICAL_ACRONYM = re.compile(
    r"([A-Za-z][A-Za-z -]{2,60}?)\s*\(([A-Z][A-Za-z]{1,10})\)",
)


def _terms(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def query_content_terms(value: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in _terms(value):
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


class DeterministicActionCandidateGenerator:
    def __init__(
        self,
        *,
        max_candidates: int = 8,
        include_semantic_anchor: bool = True,
    ) -> None:
        if not 2 <= max_candidates <= 15:
            raise ValueError("max_candidates must be between 2 and 15")
        self.max_candidates = max_candidates
        self.include_semantic_anchor = include_semantic_anchor

    def generate(
        self,
        query_spec: QuerySpec,
        *,
        query_kind: QueryKind,
    ) -> list[PolicyActionCandidate]:
        query_spec = QuerySpec.model_validate(query_spec)
        original = query_spec.original_query
        content_terms = query_content_terms(original)
        candidates = [
            PolicyActionCandidate(
                action_id="candidate-anchor",
                action_type="text_search",
                text=original,
                origin="original_query",
                provider_hint="either",
            )
        ]
        if self.include_semantic_anchor:
            candidates.append(PolicyActionCandidate(
                action_id="candidate-anchor-semantic",
                action_type="text_search",
                text=original,
                origin="deterministic_rule",
                provider_hint="openalex",
                search_mode="semantic",
            ))
        texts: list[str] = []
        if content_terms:
            texts.append(" ".join(content_terms))
        acronym_match = _PARENTHETICAL_ACRONYM.search(original)
        if acronym_match is not None:
            phrase_terms = query_content_terms(acronym_match.group(1))
            acronym = acronym_match.group(2).casefold()
            if phrase_terms:
                texts.append(" ".join([*phrase_terms, acronym]))
        for task in query_spec.tasks:
            for method in query_spec.methods:
                texts.append(" ".join([task, method]))
            for dataset in query_spec.datasets:
                texts.append(" ".join([task, dataset]))
        if len(content_terms) > 5:
            texts.append(" ".join(content_terms[:5]))
            texts.append(" ".join(content_terms[-5:]))
        if len(content_terms) >= 3:
            texts.append(" ".join(content_terms[:3]))
        for index, text in enumerate(texts, start=1):
            candidates.append(
                PolicyActionCandidate(
                    action_id=f"candidate-text-{index}",
                    action_type="text_search",
                    text=text,
                    origin="deterministic_rule",
                    provider_hint="either",
                )
            )
        if content_terms and (
            query_kind == "navigational" or _TITLE_SEEKING.search(original)
        ):
            candidates.append(
                PolicyActionCandidate(
                    action_id="candidate-title-1",
                    action_type="title_search",
                    text=" ".join(content_terms),
                    origin="deterministic_rule",
                    provider_hint="either",
                )
            )
        deduplicated: list[PolicyActionCandidate] = []
        seen: set[tuple[str, str, str]] = set()
        for candidate in candidates:
            key = (
                candidate.action_type,
                candidate.search_mode,
                " ".join(candidate.text.split()).casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(candidate)
        return deduplicated[: self.max_candidates]


__all__ = ["DeterministicActionCandidateGenerator", "query_content_terms"]
