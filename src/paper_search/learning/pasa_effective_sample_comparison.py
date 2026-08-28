"""Prospective OpenAlex+PASA coverage accounting without candidate rehydration."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any


_SIGNALS = ("task_provenance", "method", "dataset", "year", "negation")


def _coverage_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {key: int(after[key]) - int(before[key]) for key in before}


def compare_effective_sample_coverage(
    *,
    query_rows: Sequence[Mapping[str, Any]],
    gold_ids_by_query: Mapping[str, Sequence[str]],
    strict_ready_query_ids: Collection[str],
    pasa_available_gold_ids: Collection[str],
    signal_eligibility_by_query: Mapping[str, Mapping[str, bool]],
    shallow_candidate_threshold: int,
    minimum_hard_negatives: int,
    hard_negative_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare pair-feasibility capacity when PASA contains missing Gold.

    PASA Gold is injected only for queries with zero OpenAlex Gold hits.  This
    makes the added positive and pair counts exact without needing the full
    candidate identities.  The result is an availability upper bound, not a
    trainable package: a safe supplement must mix PASA lexical candidates with
    Gold so that source provenance cannot reveal the label.  Entity and hard-
    constraint signals are reported as gate-eligible pair-feasible queries,
    not as effective feature contrasts.
    """

    if min(
        shallow_candidate_threshold,
        minimum_hard_negatives,
        hard_negative_limit,
    ) <= 0:
        raise ValueError("coverage thresholds must be positive")
    strict_ids = set(strict_ready_query_ids)
    available_gold = set(pasa_available_gold_ids)
    rows_by_query = {str(row["query_id"]): row for row in query_rows}
    missing_rows = strict_ids - set(rows_by_query)
    if missing_rows:
        raise ValueError(
            f"strict-ready queries missing from audit rows: {len(missing_rows)}"
        )

    base = {
        "gold_hit_query_count": 0,
        "positive_candidate_count": 0,
        "positive_and_hard_negative_query_count": 0,
        "reliability_pair_count": 0,
        "task_provenance_pair_count": 0,
    }
    after = dict(base)
    signal_counts = {
        signal: {
            "eligible_query_count": 0,
            "base_pair_feasible_query_count": 0,
            "openalex_pasa_pair_feasible_query_count": 0,
            "rescued_pair_feasible_query_count": 0,
        }
        for signal in _SIGNALS
    }
    queue: list[dict[str, Any]] = []
    rescued_query_count = 0
    direct_gold_candidate_count = 0
    unresolved_missing_gold_count = 0

    for query_id in sorted(strict_ids):
        row = rows_by_query[query_id]
        candidate_count = int(row["candidate_count"])
        base_gold_count = int(row["gold_hit_count"])
        hard_negative_count = int(row["hard_negative_candidate_count"])
        eligible = signal_eligibility_by_query.get(query_id, {})
        query_gold_ids = gold_ids_by_query.get(query_id, ())
        pasa_gold_count = sum(gold_id in available_gold for gold_id in query_gold_ids)

        # Exact injection is deliberately restricted to missing-positive queries.
        after_gold_count = (
            base_gold_count
            if base_gold_count > 0
            else pasa_gold_count
        )
        base_pair_feasible = base_gold_count > 0 and hard_negative_count > 0
        after_pair_feasible = after_gold_count > 0 and hard_negative_count > 0
        negative_limit = min(hard_negative_count, hard_negative_limit)
        base_pair_count = base_gold_count * negative_limit
        after_pair_count = after_gold_count * negative_limit
        task_eligible = bool(eligible.get("task_provenance", False))

        base["gold_hit_query_count"] += base_gold_count > 0
        base["positive_candidate_count"] += base_gold_count
        base["positive_and_hard_negative_query_count"] += base_pair_feasible
        base["reliability_pair_count"] += base_pair_count
        base["task_provenance_pair_count"] += base_pair_count if task_eligible else 0
        after["gold_hit_query_count"] += after_gold_count > 0
        after["positive_candidate_count"] += after_gold_count
        after["positive_and_hard_negative_query_count"] += after_pair_feasible
        after["reliability_pair_count"] += after_pair_count
        after["task_provenance_pair_count"] += after_pair_count if task_eligible else 0

        for signal in _SIGNALS:
            if not bool(eligible.get(signal, False)):
                continue
            counts = signal_counts[signal]
            counts["eligible_query_count"] += 1
            counts["base_pair_feasible_query_count"] += base_pair_feasible
            counts["openalex_pasa_pair_feasible_query_count"] += after_pair_feasible
            counts["rescued_pair_feasible_query_count"] += (
                not base_pair_feasible and after_pair_feasible
            )

        reasons: list[str] = []
        recommended_action: str | None = None
        if base_gold_count == 0:
            reasons.append("missing_gold_positive")
            if pasa_gold_count > 0 and hard_negative_count > 0:
                recommended_action = "pasa_mixed_lexical_gold_supplement"
                rescued_query_count += 1
                direct_gold_candidate_count += pasa_gold_count
            elif pasa_gold_count > 0:
                reasons.append("missing_hard_negative")
                recommended_action = "targeted_pasa_lexical_negative_then_gold_injection"
            else:
                unresolved_missing_gold_count += 1
                recommended_action = "defer_openalex_high_recall"
        elif hard_negative_count < minimum_hard_negatives:
            reasons.append("missing_hard_negative")
            recommended_action = "targeted_pasa_lexical_negative"
        if candidate_count < shallow_candidate_threshold:
            reasons.append("shallow_candidate_pool")
            if recommended_action is None:
                recommended_action = "targeted_pasa_lexical_negative"

        if recommended_action is not None:
            queue.append(
                {
                    "query_id": query_id,
                    "base_candidate_count": candidate_count,
                    "base_gold_hit_count": base_gold_count,
                    "base_hard_negative_candidate_count": hard_negative_count,
                    "pasa_available_gold_count": pasa_gold_count,
                    "eligible_signals": [
                        signal for signal in _SIGNALS if bool(eligible.get(signal, False))
                    ],
                    "reasons": reasons,
                    "recommended_action": recommended_action,
                }
            )

    summary = {
        "schema_version": "openalex-pasa-effective-sample-comparison-v2",
        "projection": {
            "kind": "pasa_gold_availability_upper_bound",
            "safe_training_package_materialized": False,
            "requires_mixed_positive_negative_pasa_action": True,
            "reason": "gold-only source provenance would leak the training label",
        },
        "strict_ready_query_count": len(strict_ids),
        "base": base,
        "openalex_pasa": after,
        "delta": _coverage_delta(base, after),
        "signals": signal_counts,
        "pasa": {
            "rescued_pair_feasible_query_count": rescued_query_count,
            "direct_gold_candidate_count_for_rescues": direct_gold_candidate_count,
            "unresolved_missing_gold_query_count": unresolved_missing_gold_count,
            "strict_ready_ceiling_unchanged": True,
            "lexical_search_executed": False,
        },
        "network_request_count": 0,
        "llm_request_count": 0,
        "training_started": False,
        "production_lock_modified": False,
        "test_partition_touched": False,
    }
    return summary, queue


__all__ = ["compare_effective_sample_coverage"]
