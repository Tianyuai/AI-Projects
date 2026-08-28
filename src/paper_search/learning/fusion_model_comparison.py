"""Paired frozen-set comparison for production document rankers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from paper_search.evaluation.delivery_rehearsal import (
    compare_delivery_predictions,
)
from paper_search.evaluation.official_adapter import InternalPredictionRecord
from paper_search.evaluation.predictions import paper_evaluation_id
from paper_search.learning.cpu_document_ranker import (
    DocumentCandidateEvidence,
    DocumentRankingQuery,
)


def _rankings(
    rows: Sequence[tuple[int, DocumentRankingQuery]], ranker: Any
) -> dict[str, list[DocumentCandidateEvidence]]:
    return {
        query.query_id: list(ranker.rank(query.query, query.candidates))
        for _fold, query in rows
    }


def _metrics(
    rows: Sequence[tuple[int, DocumentRankingQuery]],
    rankings: Mapping[str, Sequence[DocumentCandidateEvidence]],
    cutoffs: Sequence[int],
) -> dict[str, object]:
    recall = {cutoff: 0.0 for cutoff in cutoffs}
    hits = {cutoff: 0 for cutoff in cutoffs}
    reciprocal_rank = 0.0
    ndcg_at_10 = 0.0
    for _fold, query in rows:
        gold = set(query.gold_paper_ids)
        ranked = [
            paper_evaluation_id(candidate.paper)
            for candidate in rankings[query.query_id]
        ]
        for cutoff in cutoffs:
            count = len(gold.intersection(ranked[:cutoff]))
            recall[cutoff] += count / len(gold)
            hits[cutoff] += int(count > 0)
        positions = [index for index, paper_id in enumerate(ranked, 1) if paper_id in gold]
        if positions:
            reciprocal_rank += 1.0 / min(positions)
        dcg = sum(
            1.0 / math.log2(index + 1)
            for index, paper_id in enumerate(ranked[:10], 1)
            if paper_id in gold
        )
        ideal = sum(
            1.0 / math.log2(index + 1)
            for index in range(1, min(len(gold), 10) + 1)
        )
        ndcg_at_10 += dcg / ideal
    count = len(rows)
    return {
        "query_count": count,
        "macro_recall_at": {
            cutoff: recall[cutoff] / count for cutoff in cutoffs
        },
        "hit_query_at": hits,
        "mrr": reciprocal_rank / count,
        "ndcg_at_10": ndcg_at_10 / count,
    }


def evaluate_fusion_model_set(
    folded_queries: Sequence[tuple[int, DocumentRankingQuery]],
    *,
    rankers: Mapping[str, Any],
    replay_rankers: Mapping[str, Any],
    gate_model: str,
    production_model: str,
    cutoffs: Sequence[int] = (5, 10, 20, 50),
) -> dict[str, object]:
    """Compare frozen rankers and fail closed on reloaded-artifact divergence."""

    rows = [
        (fold, DocumentRankingQuery.model_validate(query))
        for fold, query in folded_queries
    ]
    if not rows or sorted({fold for fold, _query in rows}) != [1, 2, 3]:
        raise ValueError("fusion comparison requires all three frozen folds")
    query_ids = [query.query_id for _fold, query in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("fusion comparison query ids must be unique")
    if (
        "B0" not in rankers
        or gate_model not in rankers
        or production_model not in rankers
    ):
        raise ValueError(
            "fusion comparison requires B0, the gate model, and the production model"
        )
    if gate_model == production_model:
        raise ValueError("gate model and production model must differ")
    if set(rankers) != set(replay_rankers):
        raise ValueError("live/replay ranker sets differ")
    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not {10, 20, 50}.issubset(normalized_cutoffs):
        raise ValueError("fusion comparison requires cutoffs 10, 20, and 50")

    live = {name: _rankings(rows, ranker) for name, ranker in rankers.items()}
    replay = {
        name: _rankings(rows, ranker) for name, ranker in replay_rankers.items()
    }
    models = {
        name: _metrics(rows, live[name], normalized_cutoffs) for name in rankers
    }
    baseline_recall = models["B0"]["macro_recall_at"]
    gate_recall = models[gate_model]["macro_recall_at"]
    deltas = {
        cutoff: gate_recall[cutoff] - baseline_recall[cutoff]
        for cutoff in normalized_cutoffs
    }
    production_metrics = models[production_model]
    production_deltas = {
        "mrr": models[gate_model]["mrr"] - production_metrics["mrr"],
        "ndcg_at_10": (
            models[gate_model]["ndcg_at_10"] - production_metrics["ndcg_at_10"]
        ),
        **{
            f"recall_at_{cutoff}": (
                gate_recall[cutoff]
                - production_metrics["macro_recall_at"][cutoff]
            )
            for cutoff in normalized_cutoffs
        },
    }
    candidate_identity = all(
        {paper_evaluation_id(candidate.paper) for candidate in query.candidates}
        == {
            paper_evaluation_id(candidate.paper)
            for candidate in ranking[query.query_id]
        }
        for _fold, query in rows
        for ranking in live.values()
    )
    gate_live = [
        InternalPredictionRecord(
            query_id=query.query_id,
            selected_paper_ids=[
                paper_evaluation_id(candidate.paper)
                for candidate in live[gate_model][query.query_id]
            ],
        )
        for _fold, query in rows
    ]
    gate_replay = [
        InternalPredictionRecord(
            query_id=query.query_id,
            selected_paper_ids=[
                paper_evaluation_id(candidate.paper)
                for candidate in replay[gate_model][query.query_id]
            ],
        )
        for _fold, query in rows
    ]
    parity = compare_delivery_predictions(gate_live, gate_replay)
    return {
        "schema_version": "fusion-training-package-comparison-v2",
        "query_count": len(rows),
        "models": models,
        "deltas": {
            name: {
                cutoff: models[name]["macro_recall_at"][cutoff]
                - baseline_recall[cutoff]
                for cutoff in normalized_cutoffs
            }
            for name in rankers
            if name != "B0"
        },
        "auto_dev_gate": {
            "policy": (
                "nondecreasing-vs-B0-recall-and-vs-production-all-primary-"
                "metrics-with-strict-gain"
            ),
            "gate_model": gate_model,
            "production_model": production_model,
            "deltas_vs_B0": deltas,
            "deltas_vs_production": production_deltas,
            "passed": (
                candidate_identity
                and all(value >= 0.0 for value in deltas.values())
                and all(value >= 0.0 for value in production_deltas.values())
                and any(value > 0.0 for value in production_deltas.values())
            ),
        },
        "live_replay_gate": parity,
        "candidate_pool_identity_unchanged": candidate_identity,
        "test_partition_touched": False,
    }


__all__ = ["evaluate_fusion_model_set"]
