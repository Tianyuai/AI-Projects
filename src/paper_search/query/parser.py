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
        mapped_result: list[str] = []
        for key, nested in value.items():
            mapped_result.append(
                f"{key}: {nested}" if isinstance(nested, str) else str(key)
            )
        return mapped_result
    if isinstance(value, list):
        listed_result: list[str] = []
        for item in value:
            if isinstance(item, str):
                listed_result.append(item)
            elif isinstance(item, Mapping):
                listed_result.extend(_string_list(item))
        return listed_result
    return []


def _constraint_strings(value: object) -> list[str]:
    """Extract constraint text from strings, mappings, or lists of either."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        for key in ("description", "value", "text"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return [nested]
        values = [
            nested
            for nested in value.values()
            if isinstance(nested, str) and nested.strip()
        ]
        return _ordered_unique(_string_list(value) + values)
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_constraint_strings(item))
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


def _explicit_exclusions(query: str) -> list[str]:
    boundary = r"(?=\s+(?:at|in|from|between|but|while)\b|[,.;!?]|$)"
    patterns = (
        rf"\b(?:without|excluding|exclude)\s+(.+?){boundary}",
        rf"\b(?:do|does|did|must|should)\s+not\s+"
        rf"(?:require|use|include|rely\s+on)\s+(.+?){boundary}",
    )
    exclusions: list[str] = []
    for pattern in patterns:
        exclusions.extend(
            match.group(1).strip(" \t\r\n,.;:!?")
            for match in re.finditer(pattern, query, flags=re.IGNORECASE)
        )
    return _ordered_unique(exclusions)


def normalize_query_analysis(
    data: Mapping[str, object],
    original_query: str,
) -> dict[str, object]:
    """Map flexible model output onto the strict QueryAnalysisResult contract."""
    raw_spec = data.get("query_spec")
    raw_plan = data.get("search_plan")
    if not isinstance(raw_spec, Mapping) or not isinstance(raw_plan, Mapping):
        wrapped = data.get("QueryAnalysisResult")
        if not isinstance(wrapped, Mapping):
            wrapped = data.get("query_analysis_result")
        if isinstance(wrapped, Mapping):
            if not isinstance(raw_spec, Mapping):
                raw_spec = wrapped.get("query_spec")
            if not isinstance(raw_plan, Mapping):
                raw_plan = wrapped.get("search_plan")
            if not isinstance(raw_spec, Mapping):
                raw_spec = wrapped.get("QuerySpec")
            if not isinstance(raw_plan, Mapping):
                raw_plan = wrapped.get("SearchPlan")
        if not isinstance(raw_spec, Mapping):
            raw_spec = data.get("QuerySpec")
        if not isinstance(raw_plan, Mapping):
            raw_plan = data.get("SearchPlan")
    if not isinstance(raw_spec, Mapping):
        top_subqueries = data.get("subqueries")
        if isinstance(top_subqueries, list) and top_subqueries:
            raw_spec = {"subqueries": top_subqueries}
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
    constraint_values = _constraint_strings(spec.get("constraints"))
    must_have = _ordered_unique(
        _string_list(spec.get("must_have")) + constraint_values
    )
    topics = _ordered_unique(topics + constraint_values)
    exclusions = _ordered_unique(
        _string_list(spec.get("exclusions"))
        + _string_list(spec.get("excluded_topics"))
        + _string_list(plan.get("exclusions"))
        + _string_list(plan.get("excluded_topics"))
        + _explicit_exclusions(original_query)
    )

    raw_subqueries = plan.get("subqueries")
    if not isinstance(raw_subqueries, list):
        raw_subqueries = plan.get("steps")
    if not isinstance(raw_subqueries, list):
        raw_subqueries = plan.get("queries")
    if not isinstance(raw_subqueries, list):
        raw_subqueries = spec.get("subqueries")
    if not isinstance(raw_subqueries, list):
        raw_subqueries = spec.get("sub_queries")
    if not isinstance(raw_subqueries, list):
        raise ValueError("model search plan must include subqueries")
    subqueries: list[dict[str, object]] = []
    for index, item in enumerate(raw_subqueries):
        text: object = None
        query_type: object = None
        search_mode: object = None
        target_constraints: list[str] = []
        provider_hint: object = "either"
        action_type: object = "text_search"
        priority: object = index + 1
        if isinstance(item, str):
            text = item
        elif isinstance(item, Mapping):
            text = item.get("text") or item.get("query") or item.get("subquery")
            query_type = item.get("query_type")
            search_mode = item.get("search_mode")
            target_constraints = _ordered_unique(
                _constraint_strings(item.get("target_constraints"))
            )
            provider_hint = item.get("provider_hint", "either")
            action_type = item.get("action_type", "text_search")
            raw_priority = item.get("priority")
            if (
                isinstance(raw_priority, int)
                and not isinstance(raw_priority, bool)
                and raw_priority > 0
            ):
                priority = raw_priority
            if not isinstance(text, str) or not text.strip():
                action = item.get("action")
                if isinstance(action, str):
                    match = re.search(r"['\"]([^'\"]{3,})['\"]", action)
                    if match is not None:
                        text = match.group(1)
        else:
            text = None
        if not isinstance(text, str) or not text.strip():
            continue
        normalized_text = " ".join(text.split())
        if query_type not in {"exact", "expanded", "decomposed"}:
            if search_mode in {"exact", "expanded", "decomposed"}:
                query_type = search_mode
                search_mode = "lexical"
            elif normalized_text.casefold() == " ".join(original_query.split()).casefold():
                query_type = "exact"
            else:
                query_type = "expanded"
        if search_mode not in {"lexical", "semantic"}:
            search_mode = "lexical"
        if provider_hint not in {"openalex", "semantic_scholar", "either"}:
            provider_hint = "either"
        if action_type not in {"text_search", "title_search"}:
            action_type = "text_search"
        subqueries.append(
            {
                "query_id": f"sq-{index + 1}",
                "text": normalized_text,
                "query_type": query_type,
                "action_type": action_type,
                "target_constraints": target_constraints,
                "priority": priority,
                "provider_hint": provider_hint,
                "search_mode": search_mode,
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


def extract_explicit_year_bounds(query: str) -> tuple[int | None, int | None]:
    """Extract publication-year bounds only when temporal syntax is explicit."""

    year = r"((?:19|20)\d{2})"
    range_match = re.search(
        rf"\b(?:from\s+{year}\s+to|between\s+{year}\s+and)\s+{year}\b",
        query,
        flags=re.IGNORECASE,
    )
    if range_match is not None:
        values = [int(value) for value in range_match.groups() if value is not None]
        return min(values), max(values)

    lower_match = re.search(
        rf"\b(?:since|after|from)\s+{year}\b",
        query,
        flags=re.IGNORECASE,
    )
    upper_match = re.search(
        rf"\b(?:before|until|through|up\s+to)\s+{year}\b",
        query,
        flags=re.IGNORECASE,
    )
    exact_match = re.search(
        rf"\b(?:published|appeared|released|papers?|studies|work)\s+"
        rf"(?:in|during)\s+{year}\b",
        query,
        flags=re.IGNORECASE,
    )
    if exact_match is not None:
        value = int(exact_match.group(1))
        return value, value
    return (
        int(lower_match.group(1)) if lower_match is not None else None,
        int(upper_match.group(1)) if upper_match is not None else None,
    )


def rule_fallback(query: str) -> QuerySpec:
    """Extract explicit temporal bounds, known venues, and `without` exclusions."""
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    year_from, year_to = extract_explicit_year_bounds(normalized)
    venue_matches = [
        venue
        for venue in _KNOWN_VENUES
        if re.search(rf"\b{re.escape(venue)}\b", normalized, flags=re.IGNORECASE)
    ]
    if year_from is None and year_to is None:
        for venue in venue_matches:
            venue_year = re.search(
                rf"\b{re.escape(venue)}\s+((?:19|20)\d{{2}})\b",
                normalized,
                flags=re.IGNORECASE,
            )
            if venue_year is not None:
                year_from = year_to = int(venue_year.group(1))
                break
    exclusions = _explicit_exclusions(normalized)
    return QuerySpec(
        original_query=normalized,
        research_goal=normalized,
        topics=[normalized],
        year_from=year_from,
        year_to=year_to,
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
        normalized_query = " ".join(query.split())
        spec = analysis.query_spec.model_copy(
            update={
                "original_query": normalized_query,
                "exclusions": _ordered_unique(
                    list(analysis.query_spec.exclusions)
                    + _explicit_exclusions(normalized_query)
                ),
            }
        )
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
            try:
                normalized_repair = normalize_query_analysis(
                    repaired.data,
                    normalized,
                )
            except (TypeError, ValueError):
                normalized_repair = None
            if normalized_repair is not None:
                try:
                    return self._validate(
                        normalized,
                        normalized_repair,
                        status="repaired",
                    )
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
    "extract_explicit_year_bounds",
    "rule_fallback",
]
