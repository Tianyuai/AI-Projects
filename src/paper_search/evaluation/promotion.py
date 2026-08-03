"""Promotion evidence contract bound to the committed quality policy."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from statistics import median
from typing import Literal, Self

from pydantic import model_validator

from paper_search.application.experiments import ExperimentName
from paper_search.control.pricing import AUTHORITATIVE_QUALITY_CATALOG
from paper_search.domain.models import DomainModel, NonEmptyStr, Sha256
from paper_search.evaluation.statistics import MacroF1Comparison


_MEDIAN_DELTA_RULE = "promotion-median-macro-f1-delta"
_LOWER_BOUND_RULE = "promotion-bootstrap-lower-bound"
_VALIDATION_DROP_RULE = "promotion-validation-macro-f1-drop"
_SAMPLES_RULE = "promotion-bootstrap-samples"
_REQUIRED_RESAMPLES = 1000


def _policy_threshold(rule_id: str) -> Decimal:
    for row in AUTHORITATIVE_QUALITY_CATALOG:
        if row.rule_id == rule_id:
            if isinstance(row.threshold, Decimal):
                return row.threshold
            return Decimal(row.threshold)
    raise ValueError(f"promotion policy rule is missing: {rule_id}")


def _rule_operator(rule_id: str) -> str:
    for row in AUTHORITATIVE_QUALITY_CATALOG:
        if row.rule_id == rule_id:
            return row.operator
    raise ValueError(f"promotion policy rule is missing: {rule_id}")


def _meets_threshold(value: Decimal, rule_id: str) -> bool:
    threshold = _policy_threshold(rule_id)
    operator = _rule_operator(rule_id)
    if operator == "gte":
        return value >= threshold
    if operator == "lte":
        return value <= threshold
    if operator == "eq":
        return value == threshold
    if operator == "gt":
        return value > threshold
    raise ValueError(f"unsupported promotion operator: {operator}")


class PromotionEvidence(DomainModel):
    """One complete, policy-bound optional-module promotion evidence bundle."""

    experiment: ExperimentName
    dev_run_ids: tuple[NonEmptyStr, NonEmptyStr, NonEmptyStr]
    median_macro_f1_delta: Decimal
    bootstrap_samples: Literal[1000] = 1000
    bootstrap_95_lower_bound: Decimal
    validation_macro_f1_drop: Decimal
    policy_sha256: Sha256
    passed: bool

    @model_validator(mode="after")
    def validate_promotion_evidence(self) -> Self:
        if len(set(self.dev_run_ids)) != 3:
            raise ValueError("dev_run_ids must contain exactly three unique run IDs")
        if self.policy_sha256 == "sha256:" + "0" * 64:
            raise ValueError("promotion evidence requires a real policy hash")
        for value in (
            self.median_macro_f1_delta,
            self.bootstrap_95_lower_bound,
            self.validation_macro_f1_drop,
        ):
            if not value.is_finite():
                raise ValueError("promotion evidence values must be finite")
        expected = (
            _meets_threshold(
                self.median_macro_f1_delta,
                _MEDIAN_DELTA_RULE,
            )
            and _meets_threshold(
                self.bootstrap_95_lower_bound,
                _LOWER_BOUND_RULE,
            )
            and _meets_threshold(
                self.validation_macro_f1_drop,
                _VALIDATION_DROP_RULE,
            )
            and self.bootstrap_samples
            == _policy_threshold(_SAMPLES_RULE)
        )
        if self.passed != expected:
            raise ValueError(
                "promotion passed flag does not match the committed policy"
            )
        return self


def evaluate_promotion_evidence(
    *,
    experiment: ExperimentName,
    dev_run_ids: Sequence[str],
    dev_comparisons: Sequence[MacroF1Comparison],
    validation_comparison: MacroF1Comparison,
    policy_sha256: Sha256,
) -> PromotionEvidence:
    """Aggregate three dev runs and one validation run into promotion evidence."""

    if len(dev_run_ids) != 3 or len(set(dev_run_ids)) != 3:
        raise ValueError("promotion evidence requires exactly three unique dev run IDs")
    if len(dev_comparisons) != 3:
        raise ValueError("promotion evidence requires exactly three dev comparisons")
    comparisons = list(dev_comparisons)
    for comparison in comparisons:
        if comparison.delta_interval.resamples != _REQUIRED_RESAMPLES:
            raise ValueError("promotion comparisons require 1000 bootstrap resamples")
    if validation_comparison.delta_interval.resamples != _REQUIRED_RESAMPLES:
        raise ValueError("promotion validation requires 1000 bootstrap resamples")

    median_delta = Decimal(str(median(item.delta_observed for item in comparisons)))
    lower_bound = Decimal(
        str(min(item.delta_interval.lower for item in comparisons))
    )
    validation_drop = Decimal(str(validation_comparison.baseline_mean)) - Decimal(
        str(validation_comparison.candidate_mean)
    )
    passed = (
        _meets_threshold(median_delta, _MEDIAN_DELTA_RULE)
        and _meets_threshold(lower_bound, _LOWER_BOUND_RULE)
        and _meets_threshold(validation_drop, _VALIDATION_DROP_RULE)
    )
    return PromotionEvidence(
        experiment=experiment,
        dev_run_ids=tuple(dev_run_ids),
        median_macro_f1_delta=median_delta,
        bootstrap_samples=_REQUIRED_RESAMPLES,
        bootstrap_95_lower_bound=lower_bound,
        validation_macro_f1_drop=validation_drop,
        policy_sha256=policy_sha256,
        passed=passed,
    )


__all__ = ["PromotionEvidence", "evaluate_promotion_evidence"]
