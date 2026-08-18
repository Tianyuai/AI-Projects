"""Small, isolated development-set closed-loop metrics."""

from __future__ import annotations

from paper_search.domain.models import DomainModel, NonEmptyStr, NonNegativeInt


class DevelopmentQueryScore(DomainModel):
    query_id: NonEmptyStr
    gold_count: NonNegativeInt
    candidate_count: NonNegativeInt
    final_count: NonNegativeInt
    candidate_hit_count: NonNegativeInt
    final_hit_count: NonNegativeInt
    candidate_oracle_recall: float
    final_recall: float
    oracle_final_gap: float


class DevelopmentClosedLoopSummary(DomainModel):
    query_count: NonNegativeInt
    completed_query_count: NonNegativeInt
    candidate_oracle_macro_recall: float
    final_macro_recall: float
    oracle_final_macro_gap: float
    queries: list[DevelopmentQueryScore]


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def score_development_query(
    *,
    query_id: str,
    gold_paper_ids: list[str],
    candidate_paper_ids: list[str],
    final_paper_ids: list[str],
) -> DevelopmentQueryScore:
    """Score one completed query without consulting any test partition."""

    gold = set(_ordered_unique(gold_paper_ids))
    if not gold:
        raise ValueError("development query gold must not be empty")
    candidates = _ordered_unique(candidate_paper_ids)
    final = _ordered_unique(final_paper_ids)
    candidate_hits = len(gold.intersection(candidates))
    final_hits = len(gold.intersection(final))
    oracle = candidate_hits / len(gold)
    final_recall = final_hits / len(gold)
    return DevelopmentQueryScore(
        query_id=query_id,
        gold_count=len(gold),
        candidate_count=len(candidates),
        final_count=len(final),
        candidate_hit_count=candidate_hits,
        final_hit_count=final_hits,
        candidate_oracle_recall=oracle,
        final_recall=final_recall,
        oracle_final_gap=oracle - final_recall,
    )


def aggregate_development_closed_loop(
    queries: list[DevelopmentQueryScore],
) -> DevelopmentClosedLoopSummary:
    """Aggregate only completed development-query executions."""

    count = len(queries)
    if count == 0:
        return DevelopmentClosedLoopSummary(
            query_count=0,
            completed_query_count=0,
            candidate_oracle_macro_recall=0.0,
            final_macro_recall=0.0,
            oracle_final_macro_gap=0.0,
            queries=[],
        )
    oracle = sum(item.candidate_oracle_recall for item in queries) / count
    final = sum(item.final_recall for item in queries) / count
    return DevelopmentClosedLoopSummary(
        query_count=count,
        completed_query_count=count,
        candidate_oracle_macro_recall=oracle,
        final_macro_recall=final,
        oracle_final_macro_gap=oracle - final,
        queries=queries,
    )


__all__ = [
    "DevelopmentClosedLoopSummary",
    "DevelopmentQueryScore",
    "aggregate_development_closed_loop",
    "score_development_query",
]
