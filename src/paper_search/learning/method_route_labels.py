"""Gold-blind method-routing supervision derived from provider receipts."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import Field, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, UnitFloat
from paper_search.learning.provider_action_labels import (
    UNAVAILABLE_PROVIDER_ERROR_CODES,
    ProviderActionLabel,
)
from paper_search.learning.graph_method_labels import GraphMethodLabel


MethodName = Literal["semantic", "graph"]
RoutingLabel = Literal["beneficial", "not_beneficial", "unavailable"]


class MethodRouteLabel(DomainModel):
    dataset: NonEmptyStr
    split: NonEmptyStr
    role: Literal["training", "development"]
    method: MethodName
    query_id: NonEmptyStr
    query: NonEmptyStr
    routing_label: RoutingLabel
    gold_association_count: int = Field(strict=True, gt=0)
    marginal_gold_hit_count: int = Field(strict=True, ge=0)
    marginal_recall: UnitFloat
    seed_count: int = Field(default=0, strict=True, ge=0)
    method_action_count: int = Field(default=1, strict=True, ge=0)
    search_api_calls: int = Field(default=0, strict=True, ge=0)

    @model_validator(mode="after")
    def validate_marginal_reward(self) -> MethodRouteLabel:
        beneficial = self.routing_label == "beneficial"
        if beneficial != (self.marginal_gold_hit_count > 0):
            raise ValueError("beneficial label must match positive marginal Gold hits")
        expected = self.marginal_gold_hit_count / self.gold_association_count
        if abs(float(self.marginal_recall) - expected) > 1e-12:
            raise ValueError("marginal recall must match marginal Gold hits")
        return self


def semantic_method_labels(
    rows: list[ProviderActionLabel],
    *,
    provider: Literal["openalex"] = "openalex",
) -> list[MethodRouteLabel]:
    """Aggregate repeated semantic receipts into one method label per query."""
    selected = [
        ProviderActionLabel.model_validate(row)
        for row in rows
        if row.provider == provider and row.action.search_mode == "semantic"
    ]
    labels: list[MethodRouteLabel] = []
    grouped: dict[str, list[ProviderActionLabel]] = defaultdict(list)
    for row in selected:
        grouped[row.query_id].append(row)
    for query_id, group in sorted(grouped.items()):
        first = group[0]
        if any(
            (row.dataset, row.split, row.role, row.query)
            != (first.dataset, first.split, first.role, first.query)
            for row in group[1:]
        ):
            raise ValueError(f"inconsistent semantic receipts for query: {query_id}")
        available_rows = [
            row
            for row in group
            if row.retrieval_status == "available"
            and not UNAVAILABLE_PROVIDER_ERROR_CODES.intersection(row.error_codes)
        ]
        marginal = max(
            (int(row.novel_over_anchor_hit_count or 0) for row in available_rows),
            default=0,
        )
        gold_counts = {
            int(row.gold_association_count)
            for row in available_rows
            if row.gold_association_count is not None
        }
        if len(gold_counts) > 1:
            raise ValueError(f"inconsistent Gold counts for query: {query_id}")
        gold_count = next(iter(gold_counts), 1)
        if not available_rows:
            routing_label: RoutingLabel = "unavailable"
        elif marginal > 0:
            routing_label = "beneficial"
        else:
            routing_label = "not_beneficial"
        labels.append(
            MethodRouteLabel(
                dataset=first.dataset,
                split=first.split,
                role=first.role,
                method="semantic",
                query_id=query_id,
                query=first.query,
                routing_label=routing_label,
                gold_association_count=gold_count,
                marginal_gold_hit_count=marginal,
                marginal_recall=marginal / gold_count,
                method_action_count=len(available_rows),
                search_api_calls=len(available_rows),
            )
        )
    return labels


def paired_semantic_method_labels(
    *,
    baseline_rows: list[ProviderActionLabel],
    semantic_rows: list[ProviderActionLabel],
    provider: Literal["openalex"] = "openalex",
) -> list[MethodRouteLabel]:
    """Measure semantic Gold gain over the union of a production baseline pool."""
    baseline_by_query: dict[str, list[ProviderActionLabel]] = defaultdict(list)
    semantic_by_query: dict[str, list[ProviderActionLabel]] = defaultdict(list)
    for raw in baseline_rows:
        row = ProviderActionLabel.model_validate(raw)
        if row.provider == provider and row.action.search_mode != "semantic":
            baseline_by_query[row.query_id].append(row)
    for raw in semantic_rows:
        row = ProviderActionLabel.model_validate(raw)
        if row.provider == provider and row.action.search_mode == "semantic":
            semantic_by_query[row.query_id].append(row)
    if set(baseline_by_query) != set(semantic_by_query):
        raise ValueError("paired baseline and semantic query coverage must match")

    labels: list[MethodRouteLabel] = []
    for query_id in sorted(semantic_by_query):
        baseline = baseline_by_query[query_id]
        semantic = semantic_by_query[query_id]
        first = semantic[0]
        combined = [*baseline, *semantic]
        if any(
            (row.dataset, row.split, row.role, row.query)
            != (first.dataset, first.split, first.role, first.query)
            for row in combined
        ):
            raise ValueError(f"inconsistent paired receipts for query: {query_id}")
        gold_counts = {
            int(row.gold_association_count)
            for row in combined
            if row.gold_association_count is not None
        }
        if len(gold_counts) != 1:
            raise ValueError(f"inconsistent paired Gold counts for query: {query_id}")
        gold_count = next(iter(gold_counts))
        baseline_available = all(
            row.retrieval_status == "available"
            and not UNAVAILABLE_PROVIDER_ERROR_CODES.intersection(row.error_codes)
            for row in baseline
        )
        semantic_available_rows = [
            row
            for row in semantic
            if row.retrieval_status == "available"
            and not UNAVAILABLE_PROVIDER_ERROR_CODES.intersection(row.error_codes)
        ]
        semantic_available = len(semantic_available_rows) == len(semantic)
        marginal_ids: set[str] = set()
        if baseline_available and semantic_available:
            baseline_hits = set().union(*(set(row.gold_hit_ids) for row in baseline))
            semantic_hits = set().union(
                *(set(row.gold_hit_ids) for row in semantic_available_rows)
            )
            marginal_ids = semantic_hits.difference(baseline_hits)
            routing_label: RoutingLabel = (
                "beneficial" if marginal_ids else "not_beneficial"
            )
        else:
            routing_label = "unavailable"
        labels.append(
            MethodRouteLabel(
                dataset=first.dataset,
                split=first.split,
                role=first.role,
                method="semantic",
                query_id=query_id,
                query=first.query,
                routing_label=routing_label,
                gold_association_count=gold_count,
                marginal_gold_hit_count=len(marginal_ids),
                marginal_recall=len(marginal_ids) / gold_count,
                method_action_count=len(semantic_available_rows),
                search_api_calls=len(semantic_available_rows),
            )
        )
    return labels


def graph_method_route_labels(
    rows: list[GraphMethodLabel],
) -> list[MethodRouteLabel]:
    labels: list[MethodRouteLabel] = []
    for raw in sorted(rows, key=lambda item: item.query_id):
        row = GraphMethodLabel.model_validate(raw)
        labels.append(
            MethodRouteLabel(
                dataset=row.dataset,
                split=row.split,
                role=row.role,
                method="graph",
                query_id=row.query_id,
                query=row.query,
                routing_label=row.routing_label,
                gold_association_count=row.gold_association_count,
                marginal_gold_hit_count=len(row.graph_marginal_gold_hit_ids),
                marginal_recall=row.graph_marginal_recall,
                seed_count=row.seed_count,
                method_action_count=row.graph_action_count,
                search_api_calls=row.search_api_calls,
            )
        )
    return labels


__all__ = [
    "MethodName",
    "MethodRouteLabel",
    "paired_semantic_method_labels",
    "RoutingLabel",
    "graph_method_route_labels",
    "semantic_method_labels",
]
