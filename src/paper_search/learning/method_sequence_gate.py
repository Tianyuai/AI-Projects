"""Frozen paired promotion gate for sequential retrieval-method additions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import Field, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, UnitFloat
from paper_search.learning.graph_method_labels import GraphMethodLabel
from paper_search.learning.provider_action_labels import ProviderActionLabel


class QueryMethodEvidence(DomainModel):
    query_id: NonEmptyStr
    fold: int = Field(strict=True, gt=0)
    intent_family: NonEmptyStr
    length_bucket: NonEmptyStr
    gold_count_bucket: NonEmptyStr
    gold_association_count: int = Field(strict=True, gt=0)
    base_gold_hit_ids: tuple[str, ...] = ()
    semantic_gold_hit_ids: tuple[str, ...] = ()
    graph_gold_hit_ids: tuple[str, ...] = ()
    base_action_count: int = Field(strict=True, ge=0)
    semantic_action_count: int = Field(strict=True, ge=0)
    graph_action_count: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def validate_hit_counts(self) -> QueryMethodEvidence:
        combined = {
            *self.base_gold_hit_ids,
            *self.semantic_gold_hit_ids,
            *self.graph_gold_hit_ids,
        }
        if len(combined) > self.gold_association_count:
            raise ValueError("method hits exceed Gold association count")
        return self


class MethodSequenceGate(DomainModel):
    minimum_beneficial_queries: int = Field(strict=True, gt=0)
    minimum_positive_folds: int = Field(strict=True, gt=0)
    minimum_positive_intent_families: int = Field(strict=True, gt=0)
    minimum_positive_length_buckets: int = Field(strict=True, gt=0)
    minimum_positive_gold_count_buckets: int = Field(strict=True, gt=0)
    minimum_efficiency_ratio_to_baseline: float = Field(ge=0.0)


class BaselineSequenceEvidence(DomainModel):
    query_count: int = Field(strict=True, gt=0)
    hit_query_count: int = Field(strict=True, ge=0)
    gold_hit_count: int = Field(strict=True, ge=0)
    action_count: int = Field(strict=True, gt=0)
    macro_recall: UnitFloat
    gold_hits_per_action: float = Field(ge=0.0)


class MethodStageDecision(DomainModel):
    method: Literal["semantic", "graph"]
    promote: bool
    beneficial_query_count: int = Field(strict=True, ge=0)
    newly_hit_query_count: int = Field(strict=True, ge=0)
    new_gold_hit_count: int = Field(strict=True, ge=0)
    incremental_action_count: int = Field(strict=True, ge=0)
    macro_recall_before: UnitFloat
    macro_recall_after: UnitFloat
    macro_recall_delta: float = Field(ge=0.0)
    positive_fold_count: int = Field(strict=True, ge=0)
    positive_intent_family_count: int = Field(strict=True, ge=0)
    positive_length_bucket_count: int = Field(strict=True, ge=0)
    positive_gold_count_bucket_count: int = Field(strict=True, ge=0)
    marginal_gold_hits_per_action: float = Field(ge=0.0)
    marginal_efficiency_ratio_to_baseline: float = Field(ge=0.0)
    failed_conditions: tuple[str, ...]


class MethodSequenceDecision(DomainModel):
    baseline: BaselineSequenceEvidence
    semantic: MethodStageDecision
    graph: MethodStageDecision


def build_method_sequence_evidence(
    *,
    frozen_rows: list[dict[str, Any]],
    base_labels: list[dict[str, Any]],
    semantic_labels: list[dict[str, Any]],
    graph_labels: list[dict[str, Any]],
) -> list[QueryMethodEvidence]:
    strata = {str(row["query_id"]): row for row in frozen_rows}
    base_grouped: dict[str, list[ProviderActionLabel]] = defaultdict(list)
    for raw in base_labels:
        row = ProviderActionLabel.model_validate(raw)
        base_grouped[row.query_id].append(row)
    semantic_grouped: dict[str, list[ProviderActionLabel]] = defaultdict(list)
    for raw in semantic_labels:
        row = ProviderActionLabel.model_validate(raw)
        if row.action.action_id == "semantic-backfill-original":
            semantic_grouped[row.query_id].append(row)
    graph_by_query = {
        row.query_id: row
        for row in (GraphMethodLabel.model_validate(raw) for raw in graph_labels)
    }
    expected = set(strata)
    if set(base_grouped) != expected:
        raise ValueError("baseline labels do not match the frozen query set")
    if set(semantic_grouped) != expected:
        raise ValueError("semantic labels do not match the frozen query set")
    if set(graph_by_query) != expected:
        raise ValueError("graph labels do not match the frozen query set")

    evidence: list[QueryMethodEvidence] = []
    for query_id in sorted(expected):
        frozen = strata[query_id]
        base = base_grouped[query_id]
        semantic = semantic_grouped[query_id]
        graph = graph_by_query[query_id]
        available_counts = {
            int(row.gold_association_count)
            for row in [*base, *semantic]
            if row.gold_association_count is not None
        }
        available_counts.add(graph.gold_association_count)
        if len(available_counts) != 1:
            raise ValueError(f"inconsistent Gold counts for query: {query_id}")
        evidence.append(
            QueryMethodEvidence(
                query_id=query_id,
                fold=int(frozen["fold"]),
                intent_family=str(frozen["intent_family"]),
                length_bucket=str(frozen["length_bucket"]),
                gold_count_bucket=str(frozen["gold_count_bucket"]),
                gold_association_count=next(iter(available_counts)),
                base_gold_hit_ids=tuple(
                    sorted(
                        set().union(
                            *(set(row.gold_hit_ids) for row in base)
                        )
                    )
                ),
                semantic_gold_hit_ids=tuple(
                    sorted(
                        set().union(
                            *(set(row.gold_hit_ids) for row in semantic)
                        )
                    )
                ),
                graph_gold_hit_ids=tuple(sorted(graph.graph_gold_hit_ids)),
                base_action_count=len(base),
                semantic_action_count=len(semantic),
                graph_action_count=graph.graph_action_count,
            )
        )
    return evidence


def _positive_group_count(
    rows: list[QueryMethodEvidence], deltas: list[float], field: str
) -> int:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, delta in zip(rows, deltas, strict=True):
        grouped[str(getattr(row, field))].append(delta)
    return sum(sum(values) > 0 for values in grouped.values())


def _stage(
    rows: list[QueryMethodEvidence],
    *,
    method: Literal["semantic", "graph"],
    before: list[set[str]],
    additions: list[set[str]],
    action_counts: list[int],
    baseline_efficiency: float,
    gate: MethodSequenceGate,
) -> tuple[MethodStageDecision, list[set[str]]]:
    after = [prior | added for prior, added in zip(before, additions, strict=True)]
    new_hits = [current - prior for prior, current in zip(before, after, strict=True)]
    deltas = [
        len(new) / row.gold_association_count
        for row, new in zip(rows, new_hits, strict=True)
    ]
    recalls_before = [
        len(hits) / row.gold_association_count
        for row, hits in zip(rows, before, strict=True)
    ]
    recalls_after = [
        len(hits) / row.gold_association_count
        for row, hits in zip(rows, after, strict=True)
    ]
    action_count = sum(action_counts)
    new_hit_count = sum(len(values) for values in new_hits)
    efficiency = new_hit_count / action_count if action_count else 0.0
    efficiency_ratio = efficiency / baseline_efficiency
    metrics = {
        "minimum_beneficial_queries": sum(bool(values) for values in new_hits),
        "minimum_positive_folds": _positive_group_count(rows, deltas, "fold"),
        "minimum_positive_intent_families": _positive_group_count(
            rows, deltas, "intent_family"
        ),
        "minimum_positive_length_buckets": _positive_group_count(
            rows, deltas, "length_bucket"
        ),
        "minimum_positive_gold_count_buckets": _positive_group_count(
            rows, deltas, "gold_count_bucket"
        ),
    }
    minimums = {
        "minimum_beneficial_queries": gate.minimum_beneficial_queries,
        "minimum_positive_folds": gate.minimum_positive_folds,
        "minimum_positive_intent_families": gate.minimum_positive_intent_families,
        "minimum_positive_length_buckets": gate.minimum_positive_length_buckets,
        "minimum_positive_gold_count_buckets": gate.minimum_positive_gold_count_buckets,
    }
    failed = [
        name for name, value in metrics.items() if value < minimums[name]
    ]
    if efficiency_ratio < gate.minimum_efficiency_ratio_to_baseline:
        failed.append("minimum_efficiency_ratio_to_baseline")
    macro_before = sum(recalls_before) / len(rows)
    macro_after = sum(recalls_after) / len(rows)
    if macro_after <= macro_before:
        failed.append("positive_macro_recall_delta")
    decision = MethodStageDecision(
        method=method,
        promote=not failed,
        beneficial_query_count=metrics["minimum_beneficial_queries"],
        newly_hit_query_count=sum(
            not prior and bool(current)
            for prior, current in zip(before, after, strict=True)
        ),
        new_gold_hit_count=new_hit_count,
        incremental_action_count=action_count,
        macro_recall_before=macro_before,
        macro_recall_after=macro_after,
        macro_recall_delta=macro_after - macro_before,
        positive_fold_count=metrics["minimum_positive_folds"],
        positive_intent_family_count=metrics[
            "minimum_positive_intent_families"
        ],
        positive_length_bucket_count=metrics["minimum_positive_length_buckets"],
        positive_gold_count_bucket_count=metrics[
            "minimum_positive_gold_count_buckets"
        ],
        marginal_gold_hits_per_action=efficiency,
        marginal_efficiency_ratio_to_baseline=efficiency_ratio,
        failed_conditions=tuple(failed),
    )
    return decision, after


def assess_method_sequence(
    evidence: list[QueryMethodEvidence], gate: MethodSequenceGate
) -> MethodSequenceDecision:
    rows = [QueryMethodEvidence.model_validate(row) for row in evidence]
    if not rows:
        raise ValueError("method sequence evidence is empty")
    if len({row.query_id for row in rows}) != len(rows):
        raise ValueError("method sequence query IDs must be unique")
    base = [set(row.base_gold_hit_ids) for row in rows]
    base_actions = sum(row.base_action_count for row in rows)
    if base_actions <= 0:
        raise ValueError("baseline action count must be positive")
    base_gold_hits = sum(len(values) for values in base)
    baseline_efficiency = base_gold_hits / base_actions
    if baseline_efficiency <= 0:
        raise ValueError("baseline must contain at least one Gold hit")
    baseline = BaselineSequenceEvidence(
        query_count=len(rows),
        hit_query_count=sum(bool(values) for values in base),
        gold_hit_count=base_gold_hits,
        action_count=base_actions,
        macro_recall=sum(
            len(values) / row.gold_association_count
            for row, values in zip(rows, base, strict=True)
        )
        / len(rows),
        gold_hits_per_action=baseline_efficiency,
    )
    semantic, after_semantic = _stage(
        rows,
        method="semantic",
        before=base,
        additions=[set(row.semantic_gold_hit_ids) for row in rows],
        action_counts=[row.semantic_action_count for row in rows],
        baseline_efficiency=baseline_efficiency,
        gate=gate,
    )
    graph, _ = _stage(
        rows,
        method="graph",
        before=after_semantic,
        additions=[set(row.graph_gold_hit_ids) for row in rows],
        action_counts=[row.graph_action_count for row in rows],
        baseline_efficiency=baseline_efficiency,
        gate=gate,
    )
    return MethodSequenceDecision(
        baseline=baseline,
        semantic=semantic,
        graph=graph,
    )


__all__ = [
    "BaselineSequenceEvidence",
    "MethodSequenceDecision",
    "MethodSequenceGate",
    "MethodStageDecision",
    "QueryMethodEvidence",
    "assess_method_sequence",
    "build_method_sequence_evidence",
]
