"""Offline injected constraint reranking with bounded public metadata."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, Paper, UnitFloat


ConstraintRerankStatus = Literal["applied", "degraded"]
ConstraintEvaluator = Callable[[Paper, tuple[str, ...]], object]
_SAFE_WARNING_CODES = frozenset({"rerank_unavailable"})
_RELEVANCE_WEIGHT = 0.7
_COVERAGE_WEIGHT = 0.3


class ConstraintAssessment(DomainModel):
    matched_constraint_count: int = Field(ge=0)
    unmatched_constraint_count: int = Field(ge=0)
    relevance_score: UnitFloat
    constraint_coverage: UnitFloat


class ConstraintScoredPaper(DomainModel):
    paper: Paper
    score: UnitFloat
    assessment: ConstraintAssessment


class ConstraintRerankResult(DomainModel):
    ranked: list[ConstraintScoredPaper]
    status: ConstraintRerankStatus
    processed_count: int = Field(ge=0)
    truncated: bool
    batch_count: int = Field(ge=0)
    warnings: list[str]


def _normalize_constraint(text: str) -> str:
    return " ".join(text.split())


def _normalize_constraints(constraints: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        normalized
        for normalized in (_normalize_constraint(constraint) for constraint in constraints)
        if normalized
    )


def _empty_assessment() -> ConstraintAssessment:
    return ConstraintAssessment(
        matched_constraint_count=0,
        unmatched_constraint_count=0,
        relevance_score=0.0,
        constraint_coverage=0.0,
    )


def _zero_ranked(papers: Sequence[Paper]) -> list[ConstraintScoredPaper]:
    empty = _empty_assessment()
    return [
        ConstraintScoredPaper(
            paper=paper,
            score=0.0,
            assessment=empty,
        )
        for paper in papers
    ]


def _read_field(source: object, *names: str) -> object:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        raise KeyError(names[0])
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    raise AttributeError(names[0])


def _coerce_count(value: object) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError("constraint counts must be nonnegative integers or sequences")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("constraint counts must be nonnegative")
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return len(value)
    raise TypeError("constraint counts must be nonnegative integers or sequences")


def _coerce_relevance_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("relevance_score must be numeric")
    relevance_score = float(value)
    if not math.isfinite(relevance_score) or not 0.0 <= relevance_score <= 1.0:
        raise ValueError("relevance_score must be between 0 and 1")
    return relevance_score


def _coerce_assessment(raw: object, total_constraints: int) -> ConstraintAssessment:
    if isinstance(raw, ConstraintAssessment):
        matched_count = raw.matched_constraint_count
        unmatched_count = raw.unmatched_constraint_count
        relevance_score = raw.relevance_score
    else:
        matched_count = _coerce_count(
            _read_field(raw, "matched_constraint_count", "matched_constraints")
        )
        unmatched_count = _coerce_count(
            _read_field(raw, "unmatched_constraint_count", "unmatched_constraints")
        )
        relevance_score = _coerce_relevance_score(_read_field(raw, "relevance_score"))
    if matched_count + unmatched_count != total_constraints:
        raise ValueError("constraint counts must account for every normalized constraint")
    coverage = 0.0 if total_constraints == 0 else matched_count / total_constraints
    return ConstraintAssessment(
        matched_constraint_count=matched_count,
        unmatched_constraint_count=unmatched_count,
        relevance_score=relevance_score,
        constraint_coverage=coverage,
    )


def _score_assessment(assessment: ConstraintAssessment) -> float:
    return (
        _RELEVANCE_WEIGHT * assessment.relevance_score
        + _COVERAGE_WEIGHT * assessment.constraint_coverage
    )


def _sanitize_warnings(warnings: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    for warning in warnings:
        code = warning.strip()
        if code in _SAFE_WARNING_CODES:
            sanitized.append(code)
    return sanitized


class ConstraintReranker:
    def __init__(
        self,
        evaluator: ConstraintEvaluator,
        max_candidates: int = 30,
        batch_size: int = 15,
        max_batches: int = 2,
    ) -> None:
        for name, value in (
            ("max_candidates", max_candidates),
            ("batch_size", batch_size),
            ("max_batches", max_batches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if batch_size > 15:
            raise ValueError("batch_size must not exceed 15")
        if max_batches > 2:
            raise ValueError("max_batches must not exceed 2")
        self._evaluator = evaluator
        self._max_candidates = max_candidates
        self._batch_size = batch_size
        self._max_batches = max_batches

    def rerank(
        self,
        papers: Sequence[Paper],
        constraints: Sequence[str],
    ) -> ConstraintRerankResult:
        if not papers:
            return ConstraintRerankResult(
                ranked=[],
                status="applied",
                processed_count=0,
                truncated=False,
                batch_count=0,
                warnings=[],
            )

        normalized_constraints = _normalize_constraints(constraints)
        if not normalized_constraints:
            return ConstraintRerankResult(
                ranked=_zero_ranked(papers),
                status="applied",
                processed_count=0,
                truncated=False,
                batch_count=0,
                warnings=[],
            )

        process_limit = min(self._max_candidates, self._batch_size * self._max_batches)
        processed_papers = list(papers[:process_limit])

        try:
            ranked_items: list[tuple[int, ConstraintScoredPaper]] = []
            for index, paper in enumerate(processed_papers):
                assessment = _coerce_assessment(
                    self._evaluator(paper, normalized_constraints),
                    len(normalized_constraints),
                )
                ranked_items.append(
                    (
                        index,
                        ConstraintScoredPaper(
                            paper=paper,
                            score=_score_assessment(assessment),
                            assessment=assessment,
                        ),
                    )
                )
        except Exception:  # noqa: BLE001
            return ConstraintRerankResult(
                ranked=_zero_ranked(papers),
                status="degraded",
                processed_count=0,
                truncated=False,
                batch_count=0,
                warnings=_sanitize_warnings(["rerank_unavailable"]),
            )

        ranked_items.sort(key=lambda item: (-item[1].score, item[0]))
        return ConstraintRerankResult(
            ranked=[item for _, item in ranked_items],
            status="applied",
            processed_count=len(processed_papers),
            truncated=len(processed_papers) < len(papers),
            batch_count=math.ceil(len(processed_papers) / self._batch_size),
            warnings=[],
        )
