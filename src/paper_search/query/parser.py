"""Schema validation, one repair attempt, and deterministic query fallback."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from paper_search.domain.models import (
    PlannerStatus,
    ProviderResult,
    QueryAnalysisResult,
    QuerySpec,
)
from paper_search.query.planner import QueryPlanner


Repair = Callable[[str], Awaitable[ProviderResult[dict[str, Any]]]]
_KNOWN_VENUES = ("NeurIPS", "ICLR", "ICML", "ACL", "EMNLP", "CVPR")
_MALFORMED_CONTENT_CODES = {"invalid_response", "empty_response", "invalid_json"}


class PlannerDependencyError(RuntimeError):
    """A sanitized transport or authentication failure that forbids fallback."""


class ClassifiedQueryAnalysis(QueryAnalysisResult):
    planner_status: PlannerStatus


def _require_content_failure(result: ProviderResult[dict[str, Any]]) -> None:
    blocking = [
        error.code
        for error in result.errors
        if error.code not in _MALFORMED_CONTENT_CODES
    ]
    if blocking:
        raise PlannerDependencyError("planner dependency failure")


def rule_fallback(query: str) -> QuerySpec:
    """Extract only explicit years, known venues, and `without` exclusions."""
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", normalized)]
    venue_matches = [
        venue
        for venue in _KNOWN_VENUES
        if re.search(rf"\b{re.escape(venue)}\b", normalized, flags=re.IGNORECASE)
    ]
    exclusion_match = re.search(
        r"\bwithout\s+(.+?)(?=\s+(?:at|in|from|between)\b|[,.;]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    exclusions = [exclusion_match.group(1).strip()] if exclusion_match else []
    return QuerySpec(
        original_query=normalized,
        research_goal=normalized,
        topics=[normalized],
        year_from=min(years) if years else None,
        year_to=max(years) if years else None,
        venues=venue_matches,
        must_have=[normalized],
        exclusions=exclusions,
        ambiguities=["rules_only_fallback"],
    )


class QueryParser:
    """Validate a combined query analysis and fall back after one repair."""

    def __init__(self, planner: QueryPlanner) -> None:
        self._planner = planner

    def _validate(
        self,
        query: str,
        data: dict[str, Any],
        *,
        status: PlannerStatus,
    ) -> ClassifiedQueryAnalysis:
        analysis = QueryAnalysisResult.model_validate(data)
        spec = analysis.query_spec.model_copy(update={"original_query": " ".join(query.split())})
        return ClassifiedQueryAnalysis(
            query_spec=spec,
            search_plan=self._planner.finalize(spec, analysis.search_plan),
            planner_status=status,
        )

    async def parse(
        self,
        query: str,
        initial: ProviderResult[dict[str, Any]],
        *,
        repair: Repair | None = None,
    ) -> ClassifiedQueryAnalysis:
        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("query must not be empty")
        _require_content_failure(initial)
        try:
            return self._validate(normalized, initial.data, status="primary")
        except (ValidationError, ValueError):
            pass

        if repair is not None:
            repair_input = json.dumps(initial.data, ensure_ascii=False, sort_keys=True)
            repaired = await repair(repair_input)
            _require_content_failure(repaired)
            try:
                return self._validate(normalized, repaired.data, status="repaired")
            except (ValidationError, ValueError):
                pass

        spec = rule_fallback(normalized)
        return ClassifiedQueryAnalysis(
            query_spec=spec,
            search_plan=self._planner.finalize(spec, None),
            planner_status="rules_fallback",
        )


__all__ = [
    "ClassifiedQueryAnalysis",
    "PlannerDependencyError",
    "QueryParser",
    "rule_fallback",
]
