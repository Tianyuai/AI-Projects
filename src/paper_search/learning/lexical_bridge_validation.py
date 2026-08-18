"""Precommitted validation contract for supervised lexical bridging."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class BridgeFoldResult:
    fold: int
    query_count: int
    improved_query_count: int
    mean_potential_delta: float
    negative_stratum_count: int


@dataclass(frozen=True)
class BridgeTrainingGate:
    passed: bool
    minimum_improvement_rate: float
    improvement_rate: float
    all_folds_positive: bool
    has_negative_strata: bool


def deterministic_bridge_fold(query_id: str, *, folds: int = 3) -> int:
    """Assign an identifier to a stable one-based fold using SHA-256."""
    if folds < 2:
        raise ValueError("fold count must be at least two")
    digest = int(sha256(query_id.encode("utf-8")).hexdigest(), 16)
    return digest % folds + 1


def bridge_training_gate(
    fold_results: list[BridgeFoldResult],
    *,
    minimum_improvement_rate: float = 2 / 9,
) -> BridgeTrainingGate:
    """Apply the frozen 20/90-equivalent training promotion gate."""
    if {item.fold for item in fold_results} != {1, 2, 3}:
        raise ValueError("training gate requires exactly folds 1, 2, and 3")
    query_count = sum(item.query_count for item in fold_results)
    if query_count <= 0 or any(item.query_count <= 0 for item in fold_results):
        raise ValueError("every fold must contain queries")
    improvement_rate = (
        sum(item.improved_query_count for item in fold_results) / query_count
    )
    all_folds_positive = all(item.mean_potential_delta > 0 for item in fold_results)
    has_negative_strata = any(item.negative_stratum_count > 0 for item in fold_results)
    return BridgeTrainingGate(
        passed=(
            improvement_rate >= minimum_improvement_rate
            and all_folds_positive
            and not has_negative_strata
        ),
        minimum_improvement_rate=minimum_improvement_rate,
        improvement_rate=improvement_rate,
        all_folds_positive=all_folds_positive,
        has_negative_strata=has_negative_strata,
    )


__all__ = [
    "BridgeFoldResult",
    "BridgeTrainingGate",
    "bridge_training_gate",
    "deterministic_bridge_fold",
]
