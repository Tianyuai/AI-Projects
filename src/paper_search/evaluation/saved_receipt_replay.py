"""Deterministic production-style replay over saved retrieval action receipts."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from paper_search.domain.models import (
    DomainModel,
    NonEmptyStr,
    NonNegativeInt,
    Paper,
    ProviderResult,
    UsageActual,
)
from paper_search.evaluation.dataset import normalize_paper_id
from paper_search.evaluation.predictions import paper_evaluation_id
from paper_search.processing.deduplicate import deduplicate_papers
from paper_search.processing.filter import apply_hard_filters
from paper_search.query.parser import rule_fallback
from paper_search.ranking.fusion import fuse_provider_results


class SavedQueryReplay(DomainModel):
    query_id: NonEmptyStr
    fold: int | None = Field(default=None, ge=1)
    gold_count: NonNegativeInt
    raw_candidate_count: NonNegativeInt
    candidate_count: NonNegativeInt
    raw_candidate_oracle_recall: float = Field(ge=0, le=1)
    candidate_oracle_recall: float = Field(ge=0, le=1)
    ranked_paper_ids: list[NonEmptyStr]
    recall_at: dict[int, float]


class SavedReplayMetrics(DomainModel):
    query_count: NonNegativeInt
    raw_candidate_oracle_macro_recall: float = Field(ge=0, le=1)
    candidate_oracle_macro_recall: float = Field(ge=0, le=1)
    macro_recall_at: dict[int, float]
    hit_query_at: dict[int, NonNegativeInt]
    gold_hits_at: dict[int, NonNegativeInt]


class SavedReplaySummary(DomainModel):
    overall: SavedReplayMetrics
    by_fold: dict[int, SavedReplayMetrics]
    queries: list[SavedQueryReplay]


def _provider_result(data: list[Paper], index: int) -> ProviderResult[list[Paper]]:
    return ProviderResult(
        data=data,
        usage=UsageActual(),
        provenance={
            "provider": f"action-{index:02d}",
            "endpoint": "saved-receipt-replay",
            "model_id": "saved-receipt-replay-v1",
            "requested_at": "2026-08-18T00:00:00+08:00",
            "response_hash": "sha256:" + f"{index:064x}"[-64:],
        },
        cache_hit=True,
        latency_ms=0,
        errors=[],
    )


def _evaluation_ids(papers: Sequence[Paper]) -> list[str]:
    return list(dict.fromkeys(paper_evaluation_id(paper) for paper in papers))


def replay_saved_query(
    *,
    query_id: str,
    query: str,
    gold_paper_ids: Sequence[str],
    action_results: Sequence[tuple[str, Sequence[Paper]]],
    cutoffs: Sequence[int] = (5, 10, 20, 50),
    fold: int | None = None,
) -> SavedQueryReplay:
    """Replay one query using action-level RRF, hard filters, and fixed cutoffs."""

    normalized_gold = {
        normalize_paper_id(identifier) for identifier in gold_paper_ids
    }
    if not normalized_gold:
        raise ValueError("saved replay gold must not be empty")
    normalized_cutoffs = tuple(dict.fromkeys(cutoffs))
    if not normalized_cutoffs or any(cutoff <= 0 for cutoff in normalized_cutoffs):
        raise ValueError("cutoffs must be positive")
    action_ids = [action_id for action_id, _papers in action_results]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("action ids must be unique")

    flattened = [paper for _action_id, papers in action_results for paper in papers]
    raw_candidates = deduplicate_papers(flattened).papers if flattened else []
    accepted_ids = {
        item.paper.canonical_id
        for item in apply_hard_filters(raw_candidates, rule_fallback(query)).accepted
    }
    fusion_input = {
        action_id: _provider_result(list(papers), index)
        for index, (action_id, papers) in enumerate(action_results, start=1)
    }
    fused = fuse_provider_results(fusion_input, method="rrf", rrf_k=60)
    ranked = [item.paper for item in fused if item.paper.canonical_id in accepted_ids]
    raw_ids = set(_evaluation_ids(raw_candidates))
    ranked_ids = _evaluation_ids(ranked)

    return SavedQueryReplay(
        query_id=query_id,
        fold=fold,
        gold_count=len(normalized_gold),
        raw_candidate_count=len(raw_ids),
        candidate_count=len(ranked_ids),
        raw_candidate_oracle_recall=len(normalized_gold.intersection(raw_ids))
        / len(normalized_gold),
        candidate_oracle_recall=len(normalized_gold.intersection(ranked_ids))
        / len(normalized_gold),
        ranked_paper_ids=ranked_ids,
        recall_at={
            cutoff: len(normalized_gold.intersection(ranked_ids[:cutoff]))
            / len(normalized_gold)
            for cutoff in normalized_cutoffs
        },
    )


def _aggregate(
    queries: Sequence[SavedQueryReplay], cutoffs: Sequence[int]
) -> SavedReplayMetrics:
    count = len(queries)
    if count == 0:
        return SavedReplayMetrics(
            query_count=0,
            raw_candidate_oracle_macro_recall=0.0,
            candidate_oracle_macro_recall=0.0,
            macro_recall_at={cutoff: 0.0 for cutoff in cutoffs},
            hit_query_at={cutoff: 0 for cutoff in cutoffs},
            gold_hits_at={cutoff: 0 for cutoff in cutoffs},
        )
    return SavedReplayMetrics(
        query_count=count,
        raw_candidate_oracle_macro_recall=sum(
            item.raw_candidate_oracle_recall for item in queries
        )
        / count,
        candidate_oracle_macro_recall=sum(
            item.candidate_oracle_recall for item in queries
        )
        / count,
        macro_recall_at={
            cutoff: sum(item.recall_at[cutoff] for item in queries) / count
            for cutoff in cutoffs
        },
        hit_query_at={
            cutoff: sum(item.recall_at[cutoff] > 0 for item in queries)
            for cutoff in cutoffs
        },
        gold_hits_at={
            cutoff: sum(
                round(item.recall_at[cutoff] * item.gold_count) for item in queries
            )
            for cutoff in cutoffs
        },
    )


def aggregate_saved_replays(
    queries: Sequence[SavedQueryReplay],
    *,
    cutoffs: Sequence[int] = (5, 10, 20, 50),
) -> SavedReplaySummary:
    """Aggregate overall and per-fold replay metrics."""

    normalized_cutoffs = tuple(dict.fromkeys(cutoffs))
    by_fold_values: dict[int, list[SavedQueryReplay]] = {}
    for query in queries:
        if query.fold is not None:
            by_fold_values.setdefault(query.fold, []).append(query)
    return SavedReplaySummary(
        overall=_aggregate(queries, normalized_cutoffs),
        by_fold={
            fold: _aggregate(items, normalized_cutoffs)
            for fold, items in sorted(by_fold_values.items())
        },
        queries=list(queries),
    )


__all__ = [
    "SavedQueryReplay",
    "SavedReplayMetrics",
    "SavedReplaySummary",
    "aggregate_saved_replays",
    "replay_saved_query",
]
