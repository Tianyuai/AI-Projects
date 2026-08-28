"""Streaming metrics for disk-backed large-scale F4/F5 OOF evaluation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence


_METRICS = ("recall_at_5", "recall_at_10", "recall_at_50", "mrr", "ndcg_at_10")


def fold_for_query_id(query_id: str) -> int:
    """Assign a stable three-way fold without consulting labels or Gold papers."""

    digest = hashlib.sha256(query_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 3 + 1


def _metrics(rows: Sequence[Mapping[str, object]], model: str) -> dict[str, float | int]:
    if not rows:
        raise ValueError("OOF metrics require rows")
    totals = {5: 0.0, 10: 0.0, 50: 0.0}
    reciprocal_rank = 0.0
    ndcg = 0.0
    for row in rows:
        gold_count = int(row["gold_count"])
        if gold_count <= 0:
            raise ValueError("OOF row requires Gold papers")
        all_ranks = row.get("gold_ranks")
        if not isinstance(all_ranks, Mapping):
            raise ValueError("OOF row is missing model ranks")
        raw_ranks = all_ranks.get(model)
        if not isinstance(raw_ranks, list):
            raise ValueError(f"OOF row is missing {model} ranks")
        ranks = sorted(int(rank) for rank in raw_ranks)
        for cutoff in totals:
            totals[cutoff] += sum(rank <= cutoff for rank in ranks) / gold_count
        if ranks:
            reciprocal_rank += 1.0 / ranks[0]
        dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 10)
        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(gold_count, 10) + 1)
        )
        ndcg += dcg / ideal
    count = len(rows)
    return {
        "query_count": count,
        "recall_at_5": totals[5] / count,
        "recall_at_10": totals[10] / count,
        "recall_at_50": totals[50] / count,
        "mrr": reciprocal_rank / count,
        "ndcg_at_10": ndcg / count,
    }


def _comparison(
    baseline: Mapping[str, float | int], candidate: Mapping[str, float | int]
) -> dict[str, object]:
    if baseline["query_count"] != candidate["query_count"]:
        raise ValueError("OOF metric query counts differ")
    return {
        "query_count": int(baseline["query_count"]),
        **{
            metric: {
                "baseline": float(baseline[metric]),
                "candidate": float(candidate[metric]),
                "delta": float(candidate[metric]) - float(baseline[metric]),
            }
            for metric in _METRICS
        },
    }


def build_oof_comparison(
    rows: Sequence[Mapping[str, object]], *, training_query_count: int
) -> dict[str, object]:
    """Aggregate compact held-out Gold ranks into promotion-compatible evidence."""

    validated = list(rows)
    if len(validated) != training_query_count:
        raise ValueError("OOF row count does not match the training package")
    ids = [str(row["query_id"]) for row in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("OOF query ids must be unique")
    folds = sorted({int(row["fold"]) for row in validated})
    if folds != [1, 2, 3]:
        raise ValueError("OOF evidence requires folds 1, 2, and 3")
    models = ("B0", "F4", "F5")
    overall = {model: _metrics(validated, model) for model in models}
    by_fold = {
        str(fold): {
            model: _metrics(
                [row for row in validated if int(row["fold"]) == fold], model
            )
            for model in models
        }
        for fold in folds
    }

    def comparison(candidate: str, baseline: str) -> dict[str, object]:
        return {
            "overall": _comparison(overall[baseline], overall[candidate]),
            "folds": {
                str(fold): _comparison(
                    by_fold[str(fold)][baseline], by_fold[str(fold)][candidate]
                )
                for fold in folds
            },
        }

    return {
        "schema_version": "large-scale-gated-fusion-oof-v1",
        "training_query_count": training_query_count,
        "query_count": len(validated),
        "fold_assignment": "sha256(query_id) % 3 + 1",
        "fold_counts": {
            str(fold): sum(int(row["fold"]) == fold for row in validated)
            for fold in folds
        },
        "metrics": overall,
        "metrics_by_fold": by_fold,
        "experiments": {
            f"S4-F4-reliability-{training_query_count}": {
                "metrics_vs_b0": comparison("F4", "B0")
            },
            f"S4-F5-gated-fusion-{training_query_count}": {
                "metrics_vs_b0": comparison("F5", "B0"),
                "metrics_vs_f4": comparison("F5", "F4"),
            },
        },
        "candidate_pool_identity_unchanged": True,
        "development_labels_used_for_training": False,
        "online_requests_made": 0,
        "test_partition_touched": False,
    }


__all__ = ["build_oof_comparison", "fold_for_query_id"]
