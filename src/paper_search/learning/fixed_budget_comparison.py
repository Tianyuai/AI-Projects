"""Paired promotion metrics for the fixed-budget OpenAlex candidate policy."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _action_identity(action: Mapping[str, Any]) -> tuple[str, str, str] | None:
    action_type = action.get("action_type")
    payload = action.get("payload")
    if action_type not in {"text_search", "title_search"} or not isinstance(
        payload, Mapping
    ):
        return None
    if action_type == "text_search":
        text = payload.get("query_text")
        mode = payload.get("search_mode", "lexical")
    else:
        text = payload.get("title_text")
        mode = "lexical"
    if not isinstance(text, str) or mode not in {"lexical", "semantic"}:
        raise ValueError("action has an invalid method identity")
    return str(action_type), str(mode), _normalized(text)


def _metrics(
    query_ids: Sequence[str],
    rows: Mapping[str, Mapping[str, Any]],
    actions: Mapping[str, Sequence[Mapping[str, Any]]],
    raw_hit_counts: Mapping[str, int],
) -> dict[str, int | float]:
    if not query_ids:
        raise ValueError("paired metrics require queries")
    action_count = sum(len(actions[query_id]) for query_id in query_ids)
    unique_candidates = sum(int(rows[query_id]["candidate_count"]) for query_id in query_ids)
    raw_hits = sum(raw_hit_counts[query_id] for query_id in query_ids)
    gold_hits = sum(int(rows[query_id]["gold_hit_count"]) for query_id in query_ids)
    return {
        "query_count": len(query_ids),
        "candidate_oracle_macro_recall": sum(
            float(rows[query_id]["candidate_recall"]) for query_id in query_ids
        )
        / len(query_ids),
        "hit_query_count": sum(
            int(rows[query_id]["gold_hit_count"]) > 0 for query_id in query_ids
        ),
        "gold_hit_count": gold_hits,
        "action_count": action_count,
        "unique_candidate_count": unique_candidates,
        "raw_candidate_hit_count": raw_hits,
        "duplicate_rate": (
            (raw_hits - unique_candidates) / raw_hits if raw_hits else 0.0
        ),
        "gold_hits_per_action": gold_hits / action_count if action_count else 0.0,
    }


def evaluate_paired_candidate_pools(
    *,
    fold_by_query: Mapping[str, int],
    query_text_by_id: Mapping[str, str],
    baseline_rows: Mapping[str, Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    baseline_actions: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_actions: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_raw_hit_counts: Mapping[str, int],
    candidate_raw_hit_counts: Mapping[str, int],
    promotion_gate: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    expected = set(fold_by_query)
    inputs = (
        query_text_by_id,
        baseline_rows,
        candidate_rows,
        baseline_actions,
        candidate_actions,
        baseline_raw_hit_counts,
        candidate_raw_hit_counts,
    )
    if any(set(item) != expected for item in inputs):
        raise ValueError("paired comparison inputs cover different query IDs")
    if set(fold_by_query.values()) != {1, 2, 3}:
        raise ValueError("paired comparison requires all three folds")

    query_ids = sorted(expected)
    baseline = _metrics(
        query_ids, baseline_rows, baseline_actions, baseline_raw_hit_counts
    )
    candidate = _metrics(
        query_ids, candidate_rows, candidate_actions, candidate_raw_hit_counts
    )
    duplicate_action_count = 0
    anchor_failure_count = 0
    maximum_actions = 0
    for query_id in query_ids:
        actions = candidate_actions[query_id]
        maximum_actions = max(maximum_actions, len(actions))
        identities = [
            identity
            for action in actions
            if (identity := _action_identity(action)) is not None
        ]
        duplicate_action_count += len(identities) - len(set(identities))
        query = _normalized(query_text_by_id[query_id])
        if identities.count(("text_search", "lexical", query)) != 1 or identities.count(
            ("text_search", "semantic", query)
        ) != 1:
            anchor_failure_count += 1

    folds: dict[str, dict[str, object]] = {}
    improved = 0
    declined = 0
    for fold in (1, 2, 3):
        fold_ids = sorted(
            query_id for query_id, value in fold_by_query.items() if value == fold
        )
        baseline_fold = _metrics(
            fold_ids, baseline_rows, baseline_actions, baseline_raw_hit_counts
        )
        candidate_fold = _metrics(
            fold_ids, candidate_rows, candidate_actions, candidate_raw_hit_counts
        )
        delta = float(candidate_fold["candidate_oracle_macro_recall"]) - float(
            baseline_fold["candidate_oracle_macro_recall"]
        )
        improved += delta > 0
        declined += delta < 0
        folds[str(fold)] = {
            "baseline": baseline_fold,
            "candidate": candidate_fold,
            "candidate_oracle_macro_recall_delta": delta,
        }

    overall_delta = float(candidate["candidate_oracle_macro_recall"]) - float(
        baseline["candidate_oracle_macro_recall"]
    )
    gate = promotion_gate or {}
    maximum_allowed_actions = int(
        gate.get("maximum_openalex_actions_per_query", 6)
    )
    minimum_improved_folds = int(gate.get("minimum_improved_folds", 2))
    maximum_declined_folds = int(gate.get("maximum_declined_folds", 0))
    marginal_gold_hits = int(candidate["gold_hit_count"]) - int(
        baseline["gold_hit_count"]
    )
    added_actions = int(candidate["action_count"]) - int(baseline["action_count"])
    marginal_efficiency = (
        marginal_gold_hits / added_actions if added_actions > 0 else None
    )

    failed_conditions: list[str] = []
    if maximum_actions > maximum_allowed_actions:
        failed_conditions.append("maximum_openalex_actions_per_query")
    if duplicate_action_count:
        failed_conditions.append("no_duplicate_actions")
    if anchor_failure_count:
        failed_conditions.append("required_original_anchors")
    if gate.get("require_overall_candidate_oracle_strict_improvement", True) and overall_delta <= 0:
        failed_conditions.append("overall_candidate_oracle_strict_improvement")
    if improved < minimum_improved_folds:
        failed_conditions.append("minimum_improved_folds")
    if declined > maximum_declined_folds:
        failed_conditions.append("no_declined_folds")
    if gate.get("require_hit_query_count_non_decrease", False) and int(
        candidate["hit_query_count"]
    ) < int(baseline["hit_query_count"]):
        failed_conditions.append("hit_query_count_non_decrease")
    if gate.get("require_gold_hit_count_non_decrease", False) and int(
        candidate["gold_hit_count"]
    ) < int(baseline["gold_hit_count"]):
        failed_conditions.append("gold_hit_count_non_decrease")
    if gate.get("require_duplicate_rate_non_increase", False) and float(
        candidate["duplicate_rate"]
    ) > float(baseline["duplicate_rate"]):
        failed_conditions.append("duplicate_rate_non_increase")
    minimum_marginal_efficiency = gate.get(
        "minimum_marginal_gold_hits_per_added_action"
    )
    if minimum_marginal_efficiency is not None and (
        marginal_efficiency is None
        or marginal_efficiency < float(minimum_marginal_efficiency)
    ):
        failed_conditions.append("minimum_marginal_gold_hits_per_added_action")
    return {
        "schema_version": "fixed-budget-openalex-paired-result-v1",
        "query_count": len(query_ids),
        "baseline": baseline,
        "candidate": candidate,
        "candidate_oracle_macro_recall_delta": overall_delta,
        "marginal": {
            "gold_hit_count": marginal_gold_hits,
            "added_action_count": added_actions,
            "gold_hits_per_added_action": marginal_efficiency,
        },
        "folds": folds,
        "integrity": {
            "maximum_action_count": maximum_actions,
            "duplicate_action_count": duplicate_action_count,
            "anchor_failure_count": anchor_failure_count,
        },
        "decision": {
            "promote": not failed_conditions,
            "improved_fold_count": improved,
            "declined_fold_count": declined,
            "failed_conditions": failed_conditions,
        },
        "test_partition_touched": False,
    }


__all__ = ["evaluate_paired_candidate_pools"]
