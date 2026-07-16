"""Deterministic keyword coverage and lexical ranking."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from pydantic import Field
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from paper_search.domain.models import DomainModel, Paper, QuerySpec
from paper_search.processing import AcceptedPaper


SCORING_VERSION = "week1-lexical-v1"


class LexicalScore(DomainModel):
    """Auditable components of a candidate's lexical score."""

    paper: Paper
    bm25_score: float = Field(allow_inf_nan=False)
    normalized_bm25: float = Field(ge=0, le=1, allow_inf_nan=False)
    keyword_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    uncertainty_multiplier: float = Field(ge=0, le=1, allow_inf_nan=False)
    final_score: float = Field(ge=0, le=1, allow_inf_nan=False)


def tokenize_text(value: str) -> list[str]:
    """Normalize text and return Unicode alphanumeric tokens in source order."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def _query_tokens(query: QuerySpec) -> list[str]:
    values = (
        [query.original_query]
        + query.topics
        + query.methods
        + query.tasks
        + query.datasets
        + query.domains
        + query.must_have
        + query.should_have
    )
    return tokenize_text(" ".join(values))


def _document_tokens(paper: Paper) -> list[str]:
    return tokenize_text(" ".join(value for value in (paper.title, paper.abstract) if value))


def rank_lexically(
    query: QuerySpec,
    candidates: Sequence[AcceptedPaper],
) -> list[LexicalScore]:
    """Rank candidates by BM25, query-token coverage, and uncertainty."""
    if not candidates:
        return []

    query_tokens = _query_tokens(query)
    document_token_lists = [_document_tokens(candidate.paper) for candidate in candidates]
    if any(document_token_lists):
        bm25 = BM25Okapi(document_token_lists)
        raw_scores = [float(value) for value in bm25.get_scores(query_tokens)]
    else:
        raw_scores = [0.0] * len(document_token_lists)
    minimum = min(raw_scores)
    maximum = max(raw_scores)
    normalized_scores = [
        (score - minimum) / (maximum - minimum) if maximum > minimum else 0.0
        for score in raw_scores
    ]

    unique_query_tokens = set(query_tokens)
    indexed_scores: list[tuple[int, LexicalScore]] = []
    for index, (candidate, raw_score, normalized_score) in enumerate(
        zip(candidates, raw_scores, normalized_scores, strict=True)
    ):
        document_tokens = set(document_token_lists[index])
        coverage = (
            len(unique_query_tokens.intersection(document_tokens)) / len(unique_query_tokens)
            if unique_query_tokens
            else 0.0
        )
        final_score = (0.7 * normalized_score + 0.3 * coverage) * candidate.score_multiplier
        indexed_scores.append(
            (
                index,
                LexicalScore(
                    paper=candidate.paper,
                    bm25_score=raw_score,
                    normalized_bm25=normalized_score,
                    keyword_coverage=coverage,
                    uncertainty_multiplier=candidate.score_multiplier,
                    final_score=final_score,
                ),
            )
        )
    indexed_scores.sort(
        key=lambda item: (
            -item[1].final_score,
            -item[1].keyword_coverage,
            -item[1].bm25_score,
            item[0],
            item[1].paper.canonical_id,
        )
    )
    return [score for _, score in indexed_scores]
