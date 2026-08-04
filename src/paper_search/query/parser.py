"""Schema validation, one repair attempt, and deterministic query fallback."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
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


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            result.append(normalized)
    return result


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, nested in value.items():
            result.append(
                f"{key}: {nested}" if isinstance(nested, str) else str(key)
            )
        return result
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, Mapping):
                result.extend(_string_list(item))
        return result
    return []


def _year(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def normalize_query_analysis(
    data: Mapping[str, object],
    original_query: str,
) -> dict[str, object]:
    """Map flexible model output onto the strict QueryAnalysisResult contract."""
    raw_spec = data.get("query_spec")
    raw_plan = data.get("search_plan")
    if not isinstance(raw_spec, Mapping) or not isinstance(raw_plan, Mapping):
        raise ValueError("model analysis must include query_spec and search_plan")
    spec = raw_spec
    plan = raw_plan

    research_goal = spec.get("research_goal") or spec.get("intent") or original_query
    if not isinstance(research_goal, str) or not research_goal.strip():
        raise ValueError("model analysis research goal is invalid")

    topics = _ordered_unique(
        _string_list(spec.get("topics"))
        + _string_list(spec.get("core_concepts"))
        + _string_list(spec.get("topic"))
        + _string_list(spec.get("domain"))
    )
    must_have = _ordered_unique(
        _string_list(spec.get("must_have"))
        + _string_list(spec.get("constraints"))
    )
    constraint_values = [
        value
        for value in spec.get("constraints", {}).values()
        if isinstance(value, str) and value.strip()
    ] if isinstance(spec.get("constraints"), Mapping) else []
    must_have = _ordered_unique(must_have + constraint_values)
    topics = _ordered_unique(topics + constraint_values)
    exclusions = _ordered_unique(
        _string_list(spec.get("exclusions"))
        + _string_list(spec.get("excluded_topics"))
    )

    raw_subqueries = plan.get("subqueries")
    if not isinstance(raw_subqueries, list):
        raise ValueError("model search plan must include subqueries")
    subqueries: list[dict[str, object]] = []
    for index, item in enumerate(raw_subqueries):
        if isinstance(item, str):
            text = item
        elif isinstance(item, Mapping):
            text = item.get("text") or item.get("query")
        else:
            text = None
        if not isinstance(text, str) or not text.strip():
            continue
        subqueries.append(
            {
                "query_id": f"sq-{index + 1}",
                "text": " ".join(text.split()),
                "query_type": "expanded",
                "target_constraints": [],
                "priority": index + 1,
                "provider_hint": "either",
            }
        )
    if not subqueries:
        raise ValueError("model search plan requires at least one subquery")

    filters = plan.get("inherited_hard_filters", {})
    if not isinstance(filters, Mapping):
        filters = {}
    rationale = plan.get("rationale") or plan.get("strategy") or "Model analysis"
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("model search plan rationale is invalid")

    return {
        "query_spec": {
            "original_query": " ".join(original_query.split()),
            "research_goal": " ".join(research_goal.split()),
            "topics": topics,
            "methods": _ordered_unique(_string_list(spec.get("methods"))),
            "tasks": _ordered_unique(_string_list(spec.get("tasks"))),
            "datasets": _ordered_unique(_string_list(spec.get("datasets"))),
            "domains": _ordered_unique(
                _string_list(spec.get("domains"))
                + _string_list(spec.get("domain"))
            ),
            "year_from": _year(spec.get("year_from")),
            "year_to": _year(spec.get("year_to")),
            "venues": _ordered_unique(_string_list(spec.get("venues"))),
            "must_have": must_have,
            "should_have": _ordered_unique(_string_list(spec.get("should_have"))),
            "exclusions": exclusions,
            "ambiguities": _ordered_unique(_string_list(spec.get("ambiguities"))),
        },
        "search_plan": {
            "subqueries": subqueries,
            "inherited_hard_filters": dict(filters),
            "rationale": " ".join(rationale.split()),
        },
    }


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

        try:
            normalized_data = normalize_query_analysis(initial.data, normalized)
        except (TypeError, ValueError):
            normalized_data = None
        if normalized_data is not None:
            try:
                return self._validate(
                    normalized,
                    normalized_data,
                    status="primary",
                )
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
