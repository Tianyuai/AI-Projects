"""Training-only candidate-depth and cross-action overlap ablations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import normalize_paper_id
from paper_search.evaluation.predictions import paper_evaluation_id


def _paper_ids(papers: Sequence[Paper]) -> list[str]:
    return list(dict.fromkeys(paper_evaluation_id(paper) for paper in papers))


def analyze_receipt_depths(
    *,
    queries: Sequence[Mapping[str, Any]],
    depths: Sequence[int] = (5, 10, 20, 30, 40, 50),
) -> dict[str, object]:
    """Compare fixed per-action depths against the deepest frozen reference."""

    normalized_depths = tuple(sorted(set(depths)))
    if not queries or not normalized_depths or normalized_depths[0] <= 0:
        raise ValueError("queries and positive depths are required")
    seen_query_ids: set[str] = set()
    prepared: list[dict[str, Any]] = []
    for raw in queries:
        query_id = str(raw["query_id"])
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query id: {query_id}")
        seen_query_ids.add(query_id)
        gold = {normalize_paper_id(value) for value in raw["gold_paper_ids"]}
        if not gold:
            raise ValueError(f"missing Gold for {query_id}")
        actions = [
            (str(action_id), _paper_ids(papers))
            for action_id, papers in raw["actions"]
        ]
        prepared.append(
            {"query_id": query_id, "fold": int(raw["fold"]), "gold": gold, "actions": actions}
        )

    reference_depth = normalized_depths[-1]

    def projected(depth: int) -> dict[str, tuple[set[str], int, int]]:
        output: dict[str, tuple[set[str], int, int]] = {}
        for row in prepared:
            streams = [ids[:depth] for _action_id, ids in row["actions"]]
            raw_count = sum(len(ids) for ids in streams)
            unique = set().union(*(set(ids) for ids in streams)) if streams else set()
            output[row["query_id"]] = (row["gold"] & unique, raw_count, len(unique))
        return output

    reference = projected(reference_depth)
    depth_rows: list[dict[str, object]] = []
    for depth in normalized_depths:
        values = projected(depth)
        raw_count = sum(value[1] for value in values.values())
        unique_count = sum(value[2] for value in values.values())
        folds: dict[str, dict[str, int | float]] = {}
        for fold in sorted({row["fold"] for row in prepared}):
            fold_rows = [row for row in prepared if row["fold"] == fold]
            folds[str(fold)] = {
                "query_count": len(fold_rows),
                "macro_recall": sum(
                    len(values[row["query_id"]][0]) / len(row["gold"])
                    for row in fold_rows
                )
                / len(fold_rows),
                "gold_hit_count": sum(
                    len(values[row["query_id"]][0]) for row in fold_rows
                ),
            }
        depth_rows.append(
            {
                "depth_per_action": depth,
                "raw_candidate_count": raw_count,
                "unique_candidate_count": unique_count,
                "duplicate_rate": (raw_count - unique_count) / raw_count
                if raw_count
                else 0.0,
                "hit_query_count": sum(bool(value[0]) for value in values.values()),
                "gold_hit_count": sum(len(value[0]) for value in values.values()),
                "macro_recall": sum(
                    len(values[row["query_id"]][0]) / len(row["gold"])
                    for row in prepared
                )
                / len(prepared),
                "zero_recall_loss": all(
                    values[query_id][0] == reference[query_id][0]
                    for query_id in values
                ),
                "folds": folds,
            }
        )

    maximum_positions = max(len(row["actions"]) for row in prepared)
    position_rows: list[dict[str, int | float]] = []
    for position in range(maximum_positions):
        eligible = 0
        candidate_count = 0
        overlap_count = 0
        novel_count = 0
        new_gold_count = 0
        for row in prepared:
            if position >= len(row["actions"]):
                continue
            eligible += 1
            current = set(row["actions"][position][1][:reference_depth])
            previous = set().union(
                *(set(ids[:reference_depth]) for _action_id, ids in row["actions"][:position])
            ) if position else set()
            candidate_count += len(current)
            overlap_count += len(current & previous)
            novel_count += len(current - previous)
            new_gold_count += len((current - previous) & row["gold"])
        position_rows.append(
            {
                "action_position": position + 1,
                "eligible_query_count": eligible,
                "candidate_count": candidate_count,
                "overlap_candidate_count": overlap_count,
                "novel_candidate_count": novel_count,
                "overlap_rate": overlap_count / candidate_count if candidate_count else 0.0,
                "new_gold_hit_count": new_gold_count,
            }
        )

    return {
        "schema_version": "receipt-depth-overlap-ablation-v1",
        "query_count": len(prepared),
        "reference_depth_per_action": reference_depth,
        "depths": depth_rows,
        "action_positions": position_rows,
        "test_partition_touched": False,
    }


__all__ = ["analyze_receipt_depths"]
