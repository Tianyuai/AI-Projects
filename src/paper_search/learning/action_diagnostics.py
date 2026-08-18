"""Oracle upper-bound diagnostics for bounded action selection."""

from __future__ import annotations

from itertools import combinations
from typing import cast

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, UnitFloat
from paper_search.learning.provider_action_labels import (
    Provider,
    ProviderActionLabel,
)


ActionProviderPair = tuple[NonEmptyStr, Provider]


class ActionSelectionDiagnostic(DomainModel):
    query_id: NonEmptyStr
    gold_association_count: int = Field(strict=True, gt=0)
    available_action_provider_count: int = Field(strict=True, ge=0)
    selected_action_provider_pairs: tuple[ActionProviderPair, ...]
    selected_gold_hit_count: int = Field(strict=True, ge=0)
    selected_recall: UnitFloat
    oracle_action_provider_pairs: tuple[ActionProviderPair, ...]
    oracle_gold_hit_count: int = Field(strict=True, ge=0)
    oracle_recall: UnitFloat
    selection_gap: UnitFloat


def _pair(label: ProviderActionLabel) -> ActionProviderPair:
    return label.action.action_id, label.provider


def _union_hits(
    pairs: tuple[ActionProviderPair, ...],
    hits_by_pair: dict[ActionProviderPair, frozenset[str]],
) -> frozenset[str]:
    return frozenset().union(*(hits_by_pair[pair] for pair in pairs)) if pairs else frozenset()


def diagnose_action_selection(
    labels: list[ProviderActionLabel],
    *,
    selected_action_provider_pairs: list[tuple[str, str]],
    max_actions: int,
) -> ActionSelectionDiagnostic:
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")
    available = [
        ProviderActionLabel.model_validate(label)
        for label in labels
        if label.retrieval_status == "available"
    ]
    if not available:
        raise ValueError("Oracle diagnostics require available provider actions")
    query_ids = {label.query_id for label in available}
    association_counts = {label.gold_association_count for label in available}
    if len(query_ids) != 1 or len(association_counts) != 1:
        raise ValueError("Oracle diagnostics require one consistent query")
    gold_count = next(iter(association_counts))
    assert gold_count is not None
    hits_by_pair = {
        _pair(label): frozenset(label.gold_hit_ids)
        for label in available
    }
    if len(hits_by_pair) != len(available):
        raise ValueError("action-provider pairs must be unique")
    selected_items: list[ActionProviderPair] = []
    for action_id, raw_provider in selected_action_provider_pairs:
        if raw_provider not in {"openalex", "semantic_scholar"}:
            raise ValueError("selected provider is invalid")
        selected_items.append((action_id, cast(Provider, raw_provider)))
    selected = tuple(selected_items)
    if len(selected) > max_actions or any(pair not in hits_by_pair for pair in selected):
        raise ValueError("selected actions exceed or fall outside available actions")
    ordered_pairs = tuple(sorted(hits_by_pair))
    candidates = [
        pair_set
        for size in range(1, min(max_actions, len(ordered_pairs)) + 1)
        for pair_set in combinations(ordered_pairs, size)
    ]
    oracle = min(
        candidates,
        key=lambda pair_set: (
            -len(_union_hits(pair_set, hits_by_pair)),
            len(pair_set),
            pair_set,
        ),
    )
    selected_hits = _union_hits(selected, hits_by_pair)
    oracle_hits = _union_hits(oracle, hits_by_pair)
    selected_recall = len(selected_hits) / gold_count
    oracle_recall = len(oracle_hits) / gold_count
    return ActionSelectionDiagnostic(
        query_id=next(iter(query_ids)),
        gold_association_count=gold_count,
        available_action_provider_count=len(available),
        selected_action_provider_pairs=selected,
        selected_gold_hit_count=len(selected_hits),
        selected_recall=selected_recall,
        oracle_action_provider_pairs=oracle,
        oracle_gold_hit_count=len(oracle_hits),
        oracle_recall=oracle_recall,
        selection_gap=oracle_recall - selected_recall,
    )


__all__ = ["ActionSelectionDiagnostic", "diagnose_action_selection"]
