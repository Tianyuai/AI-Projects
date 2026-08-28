"""Deterministic multi-label profiles for query-adaptive ranking."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from paper_search.domain.models import QuerySpec


ConstraintLabel = Literal[
    "conceptual",
    "dataset",
    "method",
    "negation",
    "task",
    "title_like",
    "year",
]


class QueryConstraintProfile(BaseModel):
    """Normalized query labels and confidence signals used by feature gates."""

    model_config = ConfigDict(frozen=True)

    labels: list[ConstraintLabel]
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    has_negation: bool = False
    is_title_like: bool = False
    is_conceptual: bool = False
    constraint_count: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)


def _normalize(values: list[str]) -> list[str]:
    return sorted(
        {
            " ".join(value.casefold().split())
            for value in values
            if value.strip()
        }
    )


def _is_title_like(query: str) -> bool:
    stripped = query.strip()
    quoted = (
        len(stripped) >= 2
        and stripped[0] in {'"', "“", "'"}
        and stripped[-1] in {'"', "”", "'"}
    )
    title_prompt = bool(
        re.match(
            r"(?i)^(find|locate|search for)\s+(the\s+)?paper\s+(titled|named)\b",
            stripped,
        )
    )
    return quoted or title_prompt


def _query_has_negation(query: str) -> bool:
    lowered = query.casefold()
    patterns = (
        r"\bwithout\b",
        r"\bexcluding\b",
        r"\bexclude\b",
        r"\bnot using\b",
        r"排除",
        r"不使用",
        r"不包含",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def profile_query_constraints(spec: QuerySpec) -> QueryConstraintProfile:
    """Create a conservative profile without network or model calls."""

    methods = _normalize(list(spec.methods))
    datasets = _normalize(list(spec.datasets))
    tasks = _normalize(list(spec.tasks))
    exclusions = _normalize(list(spec.exclusions))
    has_negation = bool(exclusions) or _query_has_negation(spec.original_query)
    is_title_like = _is_title_like(spec.original_query)
    has_structured = bool(
        methods
        or datasets
        or tasks
        or spec.year_from is not None
        or spec.year_to is not None
        or has_negation
    )
    question_like = bool(
        re.match(
            r"(?i)^(how|why|what|which|when|where|can|does|do)\b",
            spec.original_query.strip(),
        )
    )
    is_conceptual = not has_structured and not is_title_like and (
        bool(spec.topics) or question_like
    )

    labels: list[ConstraintLabel] = []
    if methods:
        labels.append("method")
    if datasets:
        labels.append("dataset")
    if tasks:
        labels.append("task")
    if spec.year_from is not None or spec.year_to is not None:
        labels.append("year")
    if has_negation:
        labels.append("negation")
    if is_title_like:
        labels.append("title_like")
    if is_conceptual:
        labels.append("conceptual")
    labels.sort()
    confidence = min(0.95, 0.4 + 0.1 * len(labels)) if labels else 0.25

    return QueryConstraintProfile(
        labels=labels,
        methods=methods,
        datasets=datasets,
        tasks=tasks,
        exclusions=exclusions,
        year_from=spec.year_from,
        year_to=spec.year_to,
        has_negation=has_negation,
        is_title_like=is_title_like,
        is_conceptual=is_conceptual,
        constraint_count=len(labels),
        confidence=confidence,
    )


__all__ = ["QueryConstraintProfile", "profile_query_constraints"]
