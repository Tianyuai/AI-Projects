"""Production-anchored fusion weight updates and conservative gating."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set

import numpy as np


PRIMARY_METRICS = (
    "mrr",
    "ndcg_at_10",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "recall_at_50",
)


def blend_anchored_family_weights(
    production_weights: Mapping[str, np.ndarray],
    candidate_weights: Mapping[str, np.ndarray],
    *,
    alpha: float,
    trainable_families: Set[str],
) -> dict[str, np.ndarray]:
    """Interpolate selected families while copying all others from production."""

    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("anchored fusion alpha must be between zero and one")
    if set(production_weights) != set(candidate_weights):
        raise ValueError("anchored fusion artifacts must contain the same families")
    if not trainable_families or not set(trainable_families).issubset(
        production_weights
    ):
        raise ValueError("anchored fusion trainable families are invalid")

    output: dict[str, np.ndarray] = {}
    for family in sorted(production_weights):
        production = np.asarray(production_weights[family], dtype=np.float64)
        candidate = np.asarray(candidate_weights[family], dtype=np.float64)
        if production.ndim != 1 or candidate.shape != production.shape:
            raise ValueError(f"anchored fusion family shape mismatch: {family}")
        if not np.isfinite(production).all() or not np.isfinite(candidate).all():
            raise ValueError(f"anchored fusion family weights are not finite: {family}")
        if family in trainable_families:
            output[family] = production + alpha * (candidate - production)
        else:
            output[family] = production.copy()
    return output


def scale_anchored_family_weights(
    production_weights: Mapping[str, np.ndarray],
    *,
    family: str,
    scale: float,
) -> dict[str, np.ndarray]:
    """Scale one production family while copying every other family exactly."""

    if family not in production_weights:
        raise ValueError("anchored fusion calibrated family is missing")
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("anchored fusion family scale must be between zero and one")
    output: dict[str, np.ndarray] = {}
    for name in sorted(production_weights):
        values = np.asarray(production_weights[name], dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"anchored fusion family weights are invalid: {name}")
        output[name] = values.copy()
    output[family] *= scale
    return output


def new_only_query_ids(
    production_query_ids: Set[str], expanded_query_ids: Set[str]
) -> frozenset[str]:
    """Return expanded queries unseen by the production training package."""

    production = frozenset(production_query_ids)
    expanded = frozenset(expanded_query_ids)
    if not production.issubset(expanded):
        raise ValueError("production query ids are not a subset of expanded query ids")
    output = expanded - production
    if not output:
        raise ValueError("expanded package contains no new-only queries")
    return output


def new_only_batch_indexes(
    expanded_query_ids: Sequence[str],
    batch_query_counts: Sequence[int],
    new_query_ids: Set[str],
) -> frozenset[int]:
    """Return only shard indexes whose frozen query slices contain new queries."""

    if not batch_query_counts or any(count <= 0 for count in batch_query_counts):
        raise ValueError("anchored fusion batch query counts are invalid")
    if sum(batch_query_counts) != len(expanded_query_ids):
        raise ValueError("anchored fusion batch query counts do not cover queries")
    new_ids = frozenset(new_query_ids)
    if not new_ids or not new_ids.issubset(expanded_query_ids):
        raise ValueError("anchored fusion new query ids are invalid")

    output: set[int] = set()
    offset = 0
    seen: set[str] = set()
    for index, count in enumerate(batch_query_counts):
        batch_ids = frozenset(expanded_query_ids[offset : offset + count])
        matched = batch_ids & new_ids
        if matched:
            output.add(index)
            seen.update(matched)
        offset += count
    if seen != new_ids:
        raise ValueError("anchored fusion batch slices missed new query ids")
    return frozenset(output)


def select_conservative_alpha(
    *,
    production_metrics: Mapping[str, float | int],
    b0_metrics: Mapping[str, float | int],
    candidate_metrics_by_alpha: Mapping[float, Mapping[str, float | int]],
    tolerance: float = 1e-12,
) -> float | None:
    """Choose the smallest alpha with no primary regression and a strict gain."""

    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("anchored fusion tolerance must be finite and nonnegative")
    if not candidate_metrics_by_alpha:
        raise ValueError("anchored fusion candidates are empty")

    def checked(metrics: Mapping[str, float | int], label: str) -> dict[str, float]:
        missing = [metric for metric in PRIMARY_METRICS if metric not in metrics]
        if missing:
            raise ValueError(f"{label} is missing metric: {missing[0]}")
        values = {metric: float(metrics[metric]) for metric in PRIMARY_METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"{label} contains a non-finite metric")
        return values

    production = checked(production_metrics, "production metrics")
    b0 = checked(b0_metrics, "B0 metrics")
    for alpha in sorted(candidate_metrics_by_alpha):
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("anchored fusion candidate alpha is invalid")
        candidate = checked(
            candidate_metrics_by_alpha[alpha], f"candidate metrics for alpha={alpha}"
        )
        if any(
            candidate[metric] + tolerance < max(production[metric], b0[metric])
            for metric in PRIMARY_METRICS
        ):
            continue
        if any(
            candidate[metric] > production[metric] + tolerance
            for metric in PRIMARY_METRICS
        ):
            return float(alpha)
    return None


def select_conservative_scale(
    *,
    production_metrics: Mapping[str, float | int],
    b0_metrics: Mapping[str, float | int],
    candidate_metrics_by_scale: Mapping[float, Mapping[str, float | int]],
    tolerance: float = 1e-12,
) -> float | None:
    """Choose the least production-changing scale that clears the strict gate."""

    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("anchored fusion tolerance must be finite and nonnegative")
    if not candidate_metrics_by_scale:
        raise ValueError("anchored fusion scale candidates are empty")

    def checked(metrics: Mapping[str, float | int], label: str) -> dict[str, float]:
        missing = [metric for metric in PRIMARY_METRICS if metric not in metrics]
        if missing:
            raise ValueError(f"{label} is missing metric: {missing[0]}")
        values = {metric: float(metrics[metric]) for metric in PRIMARY_METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"{label} contains a non-finite metric")
        return values

    production = checked(production_metrics, "production metrics")
    b0 = checked(b0_metrics, "B0 metrics")
    for scale in sorted(candidate_metrics_by_scale, reverse=True):
        if not math.isfinite(scale) or not 0.0 <= scale < 1.0:
            raise ValueError("anchored fusion candidate scale is invalid")
        candidate = checked(
            candidate_metrics_by_scale[scale],
            f"candidate metrics for scale={scale}",
        )
        if any(
            candidate[metric] + tolerance < max(production[metric], b0[metric])
            for metric in PRIMARY_METRICS
        ):
            continue
        if any(
            candidate[metric] > production[metric] + tolerance
            for metric in PRIMARY_METRICS
        ):
            return float(scale)
    return None


__all__ = [
    "PRIMARY_METRICS",
    "blend_anchored_family_weights",
    "new_only_batch_indexes",
    "new_only_query_ids",
    "scale_anchored_family_weights",
    "select_conservative_alpha",
    "select_conservative_scale",
]
