"""Audit and freeze queries that genuinely train gated fusion families."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from paper_search.evaluation.predictions import (
    paper_evaluation_id,
    paper_matches_evaluation_ids,
)
from paper_search.learning.cpu_document_ranker import DocumentRankingQuery
from paper_search.learning.gated_feature_fusion_ranker import (
    bounded_hard_constraint_preference_pairs,
    bounded_entity_preference_pairs,
    bounded_preference_pairs,
    FusionQueryContext,
    GatedFeatureFusionRanker,
    gated_family_candidate_features,
    gated_family_eligibility,
    entity_pair_signals,
    hard_constraint_pair_signals,
    training_candidate_eligible_for_family,
    training_candidate_source_features_suppressed,
)


def _sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _pair_evidence(family: str, features: Mapping[str, float]) -> dict[str, float]:
    if family == "entity":
        return {
            key: value
            for key, value in features.items()
            if key.endswith("-text-match")
        }
    if family == "hard_constraint":
        return {
            key: value
            for key, value in features.items()
            if key.startswith("hard-year-")
            or key in {"hard-negation-conflict", "hard-negation-clean"}
        }
    return dict(features)


def audit_fusion_query_activation(
    query: DocumentRankingQuery,
    ranker: GatedFeatureFusionRanker,
    *,
    context: FusionQueryContext | None = None,
) -> dict[str, Any]:
    """Count only Gold/negative pairs with distinct family feature evidence."""

    validated = DocumentRankingQuery.model_validate(query)
    resolved = context or ranker.context_store.for_training_query(validated.query)
    baseline = list(ranker.baseline_ranker.rank(validated.query, validated.candidates))
    gold_ids = set(validated.gold_paper_ids)
    families: dict[str, dict[str, object]] = {}
    for family in sorted(ranker.feature_families):
        eligible = gated_family_eligibility(resolved, family, gated=ranker.gated)
        if not eligible:
            families[family] = {
                "gate_eligible": False,
                "effective_pair_count": 0,
                "signal_effective_pair_count": {},
                "reason": "gate_inactive",
                "candidate_evidence_sha256": None,
                "selected_pair_evidence_sha256": None,
            }
            continue
        positives: list[dict[str, float]] = []
        negatives: list[dict[str, float]] = []
        evidence_rows: list[dict[str, object]] = []
        for baseline_rank, candidate in enumerate(baseline, start=1):
            is_gold = paper_matches_evaluation_ids(candidate.paper, gold_ids)
            if not training_candidate_eligible_for_family(
                candidate, family, is_gold=is_gold
            ):
                continue
            features = gated_family_candidate_features(
                resolved,
                candidate,
                baseline_rank=baseline_rank,
                family=family,
                constraint_text_evidence=True,
                query=validated.query,
                suppress_source_features=(
                    family == "entity"
                    and training_candidate_source_features_suppressed(
                        candidate, is_gold=is_gold
                    )
                ),
                publication_year_evidence_policy=(
                    ranker.publication_year_evidence_policy
                ),
                method_usage_evidence_schema_version=(
                    ranker.method_usage_evidence_schema_version
                ),
            )
            pair_evidence = _pair_evidence(family, features)
            evidence_rows.append(
                {
                    "paper_id": paper_evaluation_id(candidate.paper),
                    "is_gold": is_gold,
                    "features": dict(sorted(features.items())),
                }
            )
            if is_gold:
                positives.append(pair_evidence)
            elif family == "hard_constraint":
                negatives.append(pair_evidence)
            elif len(negatives) < ranker.hard_negative_limit:
                negatives.append(pair_evidence)
        preference_pairs: Any
        if family == "hard_constraint":
            preference_pairs = bounded_hard_constraint_preference_pairs(
                resolved,
                positives,
                negatives,
                hard_negative_limit=ranker.hard_negative_limit,
                pair_limit=ranker._pair_budget(family),
            )
        elif family == "entity":
            preference_pairs = bounded_entity_preference_pairs(
                resolved,
                positives,
                negatives,
                hard_negative_limit=ranker.hard_negative_limit,
                pair_limit=ranker._pair_budget(family),
            )
        else:
            preference_pairs = bounded_preference_pairs(
                positives,
                negatives,
                ranker._pair_budget(family),
            )
        pair_count = len(preference_pairs)
        signal_pair_count: dict[str, int] = {}
        selected_pair_rows: list[dict[str, object]] = []
        if family in {"entity", "hard_constraint"}:
            for positive, negative in preference_pairs:
                signals = (
                    hard_constraint_pair_signals(resolved, positive, negative)
                    if family == "hard_constraint"
                    else entity_pair_signals(resolved, positive, negative)
                )
                for signal in signals:
                    signal_pair_count[signal] = signal_pair_count.get(signal, 0) + 1
                selected_pair_rows.append(
                    {
                        "positive": dict(sorted(positive.items())),
                        "negative": dict(sorted(negative.items())),
                        "signals": sorted(signals),
                    }
                )
        else:
            selected_pair_rows = [
                {
                    "positive": dict(sorted(positive.items())),
                    "negative": dict(sorted(negative.items())),
                    "signals": [family],
                }
                for positive, negative in preference_pairs
            ]
        families[family] = {
            "gate_eligible": True,
            "effective_pair_count": pair_count,
            "signal_effective_pair_count": dict(sorted(signal_pair_count.items())),
            "reason": "effective" if pair_count else "no_feature_contrast",
            "candidate_evidence_sha256": _sha256(evidence_rows),
            "selected_pair_evidence_sha256": _sha256(selected_pair_rows),
        }
    return {
        "query_id": validated.query_id,
        "query_sha256": "sha256:"
        + hashlib.sha256(validated.query.encode("utf-8")).hexdigest(),
        "gold_count": len(validated.gold_paper_ids),
        "candidate_count": len(validated.candidates),
        "families": families,
    }


def build_activation_freeze(
    queries: Sequence[DocumentRankingQuery],
    ranker: GatedFeatureFusionRanker,
) -> dict[str, Any]:
    """Build an immutable-ready activation inventory and candidate backfill queue."""

    validated = [DocumentRankingQuery.model_validate(query) for query in queries]
    ids = [query.query_id for query in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("activation freeze query ids must be unique")
    audits = [audit_fusion_query_activation(query, ranker) for query in validated]
    coverage: dict[str, dict[str, int]] = {}
    selected: dict[str, list[str]] = {}
    backfill: list[dict[str, object]] = []
    for family in sorted(ranker.feature_families):
        eligible = [row for row in audits if row["families"][family]["gate_eligible"]]
        effective = [
            row
            for row in eligible
            if int(row["families"][family]["effective_pair_count"]) > 0
        ]
        coverage[family] = {
            "eligible_query_count": len(eligible),
            "effective_query_count": len(effective),
            "effective_pair_count": sum(
                int(row["families"][family]["effective_pair_count"])
                for row in effective
            ),
        }
        selected[family] = sorted(str(row["query_id"]) for row in effective)
    for row in audits:
        missing = sorted(
            family
            for family, values in row["families"].items()
            if values["gate_eligible"]
            and int(values["effective_pair_count"]) == 0
        )
        if missing:
            backfill.append(
                {
                    "query_id": row["query_id"],
                    "families": missing,
                    "reason": "no_feature_contrast",
                }
            )
    identity = {
        "query_ids": sorted(ids),
        "coverage": coverage,
        "selected_query_ids_by_family": selected,
        "candidate_backfill_queue": backfill,
    }
    return {
        "schema_version": "fusion-activation-freeze-v1",
        "query_count": len(validated),
        "coverage": coverage,
        "selected_query_ids_by_family": selected,
        "candidate_backfill_queue": backfill,
        "audits": audits,
        "freeze_sha256": _sha256(identity),
        "online_requests_made": 0,
        "test_partition_touched": False,
    }


__all__ = ["audit_fusion_query_activation", "build_activation_freeze"]
