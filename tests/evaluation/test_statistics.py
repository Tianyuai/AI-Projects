import math

import pytest

import paper_search.evaluation as evaluation
from paper_search.evaluation.statistics import (
    BootstrapInterval,
    MacroF1Comparison,
    bootstrap_mean_interval,
    compare_macro_f1,
)


def test_bootstrap_interval_is_seeded_and_contains_observed_mean() -> None:
    first = bootstrap_mean_interval([0.0, 0.5, 1.0], resamples=1000, seed=20260728)
    second = bootstrap_mean_interval([0.0, 0.5, 1.0], resamples=1000, seed=20260728)

    assert first == second
    assert isinstance(first, BootstrapInterval)
    assert first.observed_mean == pytest.approx(0.5)
    assert first.lower <= first.observed_mean <= first.upper


def test_compare_macro_f1_uses_paired_deltas_without_identifiers() -> None:
    result = compare_macro_f1([0.0, 1.0], [0.5, 1.0], resamples=1000, seed=7)

    assert isinstance(result, MacroF1Comparison)
    assert result.baseline_mean == pytest.approx(0.5)
    assert result.candidate_mean == pytest.approx(0.75)
    assert result.delta_observed == pytest.approx(0.25)
    assert result.delta_interval.observed_mean == pytest.approx(result.delta_observed)
    assert "query_id" not in result.model_dump_json()


def test_public_package_exports_statistics_symbols() -> None:
    assert evaluation.BootstrapInterval is BootstrapInterval
    assert evaluation.MacroF1Comparison is MacroF1Comparison
    assert evaluation.bootstrap_mean_interval is bootstrap_mean_interval
    assert evaluation.compare_macro_f1 is compare_macro_f1


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.0, math.inf],
        [0.0, math.nan],
    ],
)
def test_bootstrap_mean_interval_rejects_empty_and_nonfinite_values(
    values: list[float],
) -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_interval(values, resamples=1000, seed=1)


def test_bootstrap_mean_interval_rejects_fewer_than_1000_resamples() -> None:
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_mean_interval([0.0, 1.0], resamples=999, seed=1)


@pytest.mark.parametrize(
    ("baseline_values", "candidate_values"),
    [
        ([], []),
        ([0.0], [0.0, 1.0]),
        ([0.0, math.inf], [0.5, 1.0]),
        ([0.0, 1.0], [0.5, math.nan]),
    ],
)
def test_compare_macro_f1_rejects_invalid_paired_inputs(
    baseline_values: list[float],
    candidate_values: list[float],
) -> None:
    with pytest.raises(ValueError):
        compare_macro_f1(
            baseline_values,
            candidate_values,
            resamples=1000,
            seed=11,
        )
