"""Training-only audit for information compressed by binary route labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from paper_search.learning.method_route_labels import MethodRouteLabel
from paper_search.learning.provider_action_labels import ProviderActionLabel


@dataclass(frozen=True)
class LabelCompressionCriteria:
    minimum_overall_examples_per_strength: int = 30
    minimum_examples_per_strength_per_stratum: int = 10
    minimum_qualifying_strata_per_family: int = 2
    minimum_qualifying_families: int = 2

    def __post_init__(self) -> None:
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError("label compression criteria must be positive")


def baseline_gold_hit_counts(
    rows: Sequence[ProviderActionLabel],
) -> dict[str, int]:
    """Union Gold hits across the available lexical actions for each query."""
    grouped_ids: dict[str, set[str]] = defaultdict(set)
    seen_queries: set[str] = set()
    for raw in rows:
        row = ProviderActionLabel.model_validate(raw)
        if row.action.search_mode == "semantic":
            raise ValueError("baseline audit accepts lexical actions only")
        if row.retrieval_status != "available":
            raise ValueError(f"baseline action is unavailable: {row.query_id}")
        seen_queries.add(row.query_id)
        grouped_ids[row.query_id].update(row.gold_hit_ids)
    return {query_id: len(grouped_ids[query_id]) for query_id in sorted(seen_queries)}


def _lexical_coverage(row: MethodRouteLabel, baseline_hits: int) -> str:
    if baseline_hits < 0 or baseline_hits > row.gold_association_count:
        raise ValueError(f"invalid baseline Gold hit count: {row.query_id}")
    if baseline_hits == 0:
        return "none"
    if baseline_hits == row.gold_association_count:
        return "complete"
    return "partial"


def audit_binary_label_compression(
    rows: Sequence[MethodRouteLabel],
    *,
    metadata: Mapping[str, Mapping[str, Any]],
    baseline_gold_hit_counts: Mapping[str, int],
    criteria: LabelCompressionCriteria,
) -> dict[str, Any]:
    """Check whether one positive label repeatedly hides different hit rewards."""
    validated = [MethodRouteLabel.model_validate(row) for row in rows]
    if not validated:
        raise ValueError("label compression audit requires rows")
    query_ids = {row.query_id for row in validated}
    if query_ids != set(metadata) or query_ids != set(baseline_gold_hit_counts):
        raise ValueError("audit metadata and baseline coverage must match labels")

    family_values: dict[str, dict[str, list[MethodRouteLabel]]] = {
        family: defaultdict(list)
        for family in (
            "intent_family",
            "length_bucket",
            "gold_count_bucket",
            "lexical_coverage",
        )
    }
    for row in validated:
        row_metadata = metadata[row.query_id]
        for family in ("intent_family", "length_bucket", "gold_count_bucket"):
            if family not in row_metadata:
                raise ValueError(f"missing {family} metadata: {row.query_id}")
            family_values[family][str(row_metadata[family])].append(row)
        family_values["lexical_coverage"][
            _lexical_coverage(row, baseline_gold_hit_counts[row.query_id])
        ].append(row)

    beneficial = [row for row in validated if row.routing_label == "beneficial"]
    overall_strength_counts = {
        "multi_hit": sum(row.marginal_gold_hit_count >= 2 for row in beneficial),
        "single_hit": sum(row.marginal_gold_hit_count == 1 for row in beneficial),
    }
    overall_recall_shape_counts = {
        "complete_marginal_recall": sum(
            float(row.marginal_recall) == 1.0 for row in beneficial
        ),
        "partial_marginal_recall": sum(
            float(row.marginal_recall) < 1.0 for row in beneficial
        ),
    }

    families: dict[str, list[dict[str, Any]]] = {}
    qualifying_families: list[str] = []
    for family, grouped in family_values.items():
        strata = []
        for value, group in sorted(grouped.items()):
            positive = [row for row in group if row.routing_label == "beneficial"]
            single = sum(row.marginal_gold_hit_count == 1 for row in positive)
            multi = sum(row.marginal_gold_hit_count >= 2 for row in positive)
            strata.append(
                {
                    "value": value,
                    "query_count": len(group),
                    "beneficial_query_count": len(positive),
                    "single_hit_count": single,
                    "multi_hit_count": multi,
                    "partial_marginal_recall_count": sum(
                        float(row.marginal_recall) < 1.0 for row in positive
                    ),
                    "complete_marginal_recall_count": sum(
                        float(row.marginal_recall) == 1.0 for row in positive
                    ),
                    "qualifies": min(single, multi)
                    >= criteria.minimum_examples_per_strength_per_stratum,
                }
            )
        families[family] = strata
        qualifying_strata = sum(bool(row["qualifies"]) for row in strata)
        if qualifying_strata >= criteria.minimum_qualifying_strata_per_family:
            qualifying_families.append(family)

    overall_qualified = (
        min(overall_strength_counts.values())
        >= criteria.minimum_overall_examples_per_strength
    )
    return {
        "query_count": len(validated),
        "beneficial_query_count": len(beneficial),
        "observed_api_call_distribution": {
            str(calls): count
            for calls, count in sorted(
                Counter(row.search_api_calls for row in validated).items()
            )
        },
        "overall_strength_counts": overall_strength_counts,
        "overall_recall_shape_counts": overall_recall_shape_counts,
        "families": families,
        "qualifying_families": qualifying_families,
        "criteria": asdict(criteria),
        "evidence_crosses_strata": (
            overall_qualified
            and len(qualifying_families) >= criteria.minimum_qualifying_families
        ),
    }


__all__ = [
    "LabelCompressionCriteria",
    "audit_binary_label_compression",
    "baseline_gold_hit_counts",
]
