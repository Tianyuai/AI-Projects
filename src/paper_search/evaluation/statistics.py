"""Deterministic bootstrap statistics for aggregate offline evaluation metrics."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


_DEFAULT_RESAMPLES = 1000
_DEFAULT_CONFIDENCE_LEVEL = 0.95


class BootstrapInterval(BaseModel):
    """Observed mean with a deterministic inclusive bootstrap percentile interval."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    observed_mean: float = Field(allow_inf_nan=False)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    confidence_level: float = Field(
        default=_DEFAULT_CONFIDENCE_LEVEL,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    resamples: int = Field(default=_DEFAULT_RESAMPLES, strict=True, ge=_DEFAULT_RESAMPLES)
    seed: int | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "BootstrapInterval":
        if self.lower > self.upper:
            raise ValueError("bootstrap interval bounds are invalid")
        if not self.lower <= self.observed_mean <= self.upper:
            raise ValueError("observed mean must fall within the bootstrap interval")
        return self


class MacroF1Comparison(BaseModel):
    """Aggregate paired macro-F1 comparison without identifiers or raw labels."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    baseline_mean: float = Field(allow_inf_nan=False)
    candidate_mean: float = Field(allow_inf_nan=False)
    delta_observed: float = Field(allow_inf_nan=False)
    delta_interval: BootstrapInterval


def _validate_resamples(resamples: int) -> None:
    if isinstance(resamples, bool) or resamples < _DEFAULT_RESAMPLES:
        raise ValueError("resamples must be an integer greater than or equal to 1000")


def _normalize_values(values: Sequence[float], *, field_name: str) -> list[float]:
    normalized_values = [float(value) for value in values]
    if not normalized_values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not math.isfinite(value) for value in normalized_values):
        raise ValueError(f"{field_name} must contain only finite values")
    return normalized_values


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _inclusive_percentile_index(probability: float, sample_count: int, *, upper: bool) -> int:
    raw_index = probability * (sample_count - 1)
    return math.ceil(raw_index) if upper else math.floor(raw_index)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    resamples: int = _DEFAULT_RESAMPLES,
    seed: int | None = None,
) -> BootstrapInterval:
    """Bootstrap the mean of aggregate scalar values with deterministic sampling."""

    _validate_resamples(resamples)
    normalized_values = _normalize_values(values, field_name="values")

    rng = random.Random(seed)
    sample_size = len(normalized_values)
    observed_mean = _mean(normalized_values)
    bootstrap_means = []
    for _ in range(resamples):
        resampled_total = 0.0
        for _ in range(sample_size):
            resampled_total += normalized_values[rng.randrange(sample_size)]
        bootstrap_means.append(resampled_total / sample_size)
    bootstrap_means.sort()

    lower_index = _inclusive_percentile_index(0.025, resamples, upper=False)
    upper_index = _inclusive_percentile_index(0.975, resamples, upper=True)

    return BootstrapInterval(
        observed_mean=observed_mean,
        lower=bootstrap_means[lower_index],
        upper=bootstrap_means[upper_index],
        resamples=resamples,
        seed=seed,
    )


def compare_macro_f1(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    resamples: int = _DEFAULT_RESAMPLES,
    seed: int | None = None,
) -> MacroF1Comparison:
    """Compare candidate and baseline macro-F1 using paired bootstrap deltas."""

    baseline = _normalize_values(baseline_values, field_name="baseline_values")
    candidate = _normalize_values(candidate_values, field_name="candidate_values")
    if len(baseline) != len(candidate):
        raise ValueError("baseline_values and candidate_values must have equal lengths")

    delta_values = [candidate_value - baseline_value for baseline_value, candidate_value in zip(baseline, candidate)]
    delta_interval = bootstrap_mean_interval(delta_values, resamples=resamples, seed=seed)

    baseline_mean = _mean(baseline)
    candidate_mean = _mean(candidate)
    return MacroF1Comparison(
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        delta_observed=candidate_mean - baseline_mean,
        delta_interval=delta_interval,
    )
