from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from paper_search.control.pricing import AUTHORITATIVE_QUALITY_CATALOG
from paper_search.evaluation.promotion import (
    PromotionEvidence,
    evaluate_promotion_evidence,
)
from paper_search.evaluation.statistics import BootstrapInterval, MacroF1Comparison


POLICY_HASH = "sha256:" + "a" * 64
ZERO_HASH = "sha256:" + "0" * 64


def _comparison(
    *,
    delta: float,
    lower: float,
    upper: float,
    baseline_mean: float = 0.5,
    resamples: int = 1000,
) -> MacroF1Comparison:
    return MacroF1Comparison(
        baseline_mean=baseline_mean,
        candidate_mean=baseline_mean + delta,
        delta_observed=delta,
        delta_interval=BootstrapInterval(
            observed_mean=delta,
            lower=lower,
            upper=upper,
            resamples=resamples,
        ),
    )


def _dev_comparisons() -> list[MacroF1Comparison]:
    return [
        _comparison(delta=0.10, lower=0.05, upper=0.15),
        _comparison(delta=0.11, lower=0.06, upper=0.16),
        _comparison(delta=0.12, lower=0.07, upper=0.17),
    ]


def test_promotion_evidence_requires_exactly_three_unique_dev_run_ids() -> None:
    with pytest.raises(ValidationError):
        PromotionEvidence(
            experiment="embedding",
            dev_run_ids=("dev-1", "dev-2"),
            median_macro_f1_delta=Decimal("0.11"),
            bootstrap_samples=1000,
            bootstrap_95_lower_bound=Decimal("0.05"),
            validation_macro_f1_drop=Decimal("-0.01"),
            policy_sha256=POLICY_HASH,
            passed=True,
        )
    with pytest.raises(ValidationError):
        PromotionEvidence(
            experiment="embedding",
            dev_run_ids=("dev-1", "dev-1", "dev-2"),
            median_macro_f1_delta=Decimal("0.11"),
            bootstrap_samples=1000,
            bootstrap_95_lower_bound=Decimal("0.05"),
            validation_macro_f1_drop=Decimal("-0.01"),
            policy_sha256=POLICY_HASH,
            passed=True,
        )


def test_promotion_evidence_rejects_placeholder_policy_hash() -> None:
    with pytest.raises(ValidationError):
        PromotionEvidence(
            experiment="embedding",
            dev_run_ids=("dev-1", "dev-2", "dev-3"),
            median_macro_f1_delta=Decimal("0.11"),
            bootstrap_samples=1000,
            bootstrap_95_lower_bound=Decimal("0.05"),
            validation_macro_f1_drop=Decimal("-0.01"),
            policy_sha256=ZERO_HASH,
            passed=True,
        )


def test_promotion_evidence_rejects_inconsistent_passed_flag() -> None:
    with pytest.raises(ValidationError):
        PromotionEvidence(
            experiment="embedding",
            dev_run_ids=("dev-1", "dev-2", "dev-3"),
            median_macro_f1_delta=Decimal("0.001"),
            bootstrap_samples=1000,
            bootstrap_95_lower_bound=Decimal("-0.02"),
            validation_macro_f1_drop=Decimal("0.05"),
            policy_sha256=POLICY_HASH,
            passed=True,
        )


def test_evaluate_promotion_evidence_computes_evidence_fields() -> None:
    evidence = evaluate_promotion_evidence(
        experiment="embedding",
        dev_run_ids=("dev-1", "dev-2", "dev-3"),
        dev_comparisons=_dev_comparisons(),
        validation_comparison=_comparison(delta=0.01, lower=-0.005, upper=0.03),
        policy_sha256=POLICY_HASH,
    )

    assert evidence.median_macro_f1_delta == Decimal("0.11")
    assert evidence.bootstrap_95_lower_bound == Decimal("0.05")
    assert evidence.validation_macro_f1_drop == Decimal("-0.01")
    assert evidence.bootstrap_samples == 1000
    assert evidence.policy_sha256 == POLICY_HASH
    assert evidence.passed is True


def test_evaluate_promotion_evidence_passes_at_authoritative_thresholds() -> None:
    evidence = evaluate_promotion_evidence(
        experiment="embedding",
        dev_run_ids=("dev-1", "dev-2", "dev-3"),
        dev_comparisons=[
            _comparison(delta=0.01, lower=-0.005, upper=0.02),
            _comparison(delta=0.011, lower=-0.005, upper=0.02),
            _comparison(delta=0.012, lower=-0.004, upper=0.02),
        ],
        validation_comparison=_comparison(delta=0.01, lower=0.0, upper=0.02),
        policy_sha256=POLICY_HASH,
    )

    assert evidence.passed is True


def test_evaluate_promotion_evidence_fails_when_median_below_threshold() -> None:
    evidence = evaluate_promotion_evidence(
        experiment="embedding",
        dev_run_ids=("dev-1", "dev-2", "dev-3"),
        dev_comparisons=[
            _comparison(delta=0.001, lower=-0.005, upper=0.02),
            _comparison(delta=0.002, lower=-0.005, upper=0.02),
            _comparison(delta=0.003, lower=-0.005, upper=0.02),
        ],
        validation_comparison=_comparison(delta=0.01, lower=0.0, upper=0.02),
        policy_sha256=POLICY_HASH,
    )

    assert evidence.passed is False


def test_evaluate_promotion_evidence_fails_when_lower_bound_below_threshold() -> None:
    evidence = evaluate_promotion_evidence(
        experiment="embedding",
        dev_run_ids=("dev-1", "dev-2", "dev-3"),
        dev_comparisons=[
            _comparison(delta=0.10, lower=0.05, upper=0.15),
            _comparison(delta=0.11, lower=-0.02, upper=0.16),
            _comparison(delta=0.12, lower=0.07, upper=0.17),
        ],
        validation_comparison=_comparison(delta=0.01, lower=0.0, upper=0.02),
        policy_sha256=POLICY_HASH,
    )

    assert evidence.passed is False


def test_evaluate_promotion_evidence_fails_when_validation_drop_above_threshold() -> None:
    evidence = evaluate_promotion_evidence(
        experiment="embedding",
        dev_run_ids=("dev-1", "dev-2", "dev-3"),
        dev_comparisons=_dev_comparisons(),
        validation_comparison=_comparison(delta=-0.02, lower=-0.05, upper=0.01),
        policy_sha256=POLICY_HASH,
    )

    assert evidence.passed is False


def test_evaluate_promotion_evidence_rejects_wrong_run_count() -> None:
    with pytest.raises(ValueError, match="three"):
        evaluate_promotion_evidence(
            experiment="embedding",
            dev_run_ids=("dev-1", "dev-2"),
            dev_comparisons=_dev_comparisons(),
            validation_comparison=_comparison(delta=0.01, lower=0.0, upper=0.02),
            policy_sha256=POLICY_HASH,
        )


def test_evaluate_promotion_evidence_rejects_non_1000_resamples() -> None:
    comparisons = [
        _comparison(delta=0.10, lower=0.05, upper=0.15),
        _comparison(delta=0.11, lower=0.06, upper=0.16),
        _comparison(delta=0.12, lower=0.07, upper=0.17, resamples=1001),
    ]
    with pytest.raises(ValueError, match="1000"):
        evaluate_promotion_evidence(
            experiment="embedding",
            dev_run_ids=("dev-1", "dev-2", "dev-3"),
            dev_comparisons=comparisons,
            validation_comparison=_comparison(delta=0.01, lower=0.0, upper=0.02),
            policy_sha256=POLICY_HASH,
        )


def test_promotion_thresholds_are_bound_to_authoritative_catalog() -> None:
    catalog = {row.rule_id: row for row in AUTHORITATIVE_QUALITY_CATALOG}
    assert catalog["promotion-median-macro-f1-delta"].threshold == Decimal("0.01")
    assert catalog["promotion-bootstrap-lower-bound"].threshold == Decimal("-0.005")
    assert catalog["promotion-validation-macro-f1-drop"].threshold == Decimal("0.01")
    assert catalog["promotion-bootstrap-samples"].threshold == 1000
