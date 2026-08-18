"""Deterministic routing before any local model or LLM call."""

from __future__ import annotations

import re

from paper_search.domain.models import DomainModel, QuerySpec
from paper_search.learning.contracts import QueryKind
from paper_search.query.parser import rule_fallback


_IDENTIFIER = re.compile(
    r"(?:\bdoi\s*:\s*|\bdoi\.org/|\b10\.\d{4,9}/\S+|\barxiv\s*:\s*\d{4}\.\d{4,5})",
    flags=re.IGNORECASE,
)
_EXPLICIT_TITLE = re.compile(
    r"(?:paper\s+(?:titled|called)\s+[\"']|(?:find|show|get)\s+(?:me\s+)?(?:the\s+)?[\"'][^\"']+[\"'])",
    flags=re.IGNORECASE,
)
_METADATA = re.compile(
    r"(?:\bpapers?\s+by\b|\bauthored\s+by\b|\bpublished\s+(?:at|in|between|before|after)\b|"
    r"\b(?:cite|cites|citing|cited\s+by)\b|\b(?:journal|conference|venue)\b)",
    flags=re.IGNORECASE,
)


class RoutedQuery(DomainModel):
    query_kind: QueryKind
    query_spec: QuerySpec


class RuleQueryRouter:
    def route(self, query: str) -> RoutedQuery:
        spec = rule_fallback(query)
        normalized = spec.original_query
        if _IDENTIFIER.search(normalized) or _EXPLICIT_TITLE.search(normalized):
            kind: QueryKind = "navigational"
        elif _METADATA.search(normalized) or (
            bool(spec.venues) and spec.year_from is not None
        ):
            kind = "metadata"
        else:
            kind = "semantic"
        return RoutedQuery(query_kind=kind, query_spec=spec)


__all__ = ["RoutedQuery", "RuleQueryRouter"]
