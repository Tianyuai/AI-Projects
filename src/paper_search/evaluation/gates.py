"""Deterministic formal-validity and baseline-quality Gate evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Literal

from pydantic import Field

from paper_search.control.ledger import LedgerReport
from paper_search.control.pricing import QualityGatePolicy, QualityGateRule
from paper_search.domain.models import DomainModel, NonEmptyStr
from paper_search.evaluation.dataset import EvaluationQuery, PredictionRecord
from paper_search.evaluation.execution_adapter import EvaluationFailureRecord
from paper_search.evaluation.metrics import EvaluationResult, MetricMeasure


_HARD_FAILED_QUERY_PREFIX = "hard_failed_query:"


class MeasureValue(DomainModel):
    numerator: Decimal
    denominator: Decimal = Field(ge=0)
    value: Decimal | None


class GateCheck(DomainModel):
    rule_id: NonEmptyStr
    classification: Literal[
        "formal_validity", "baseline_quality", "reporting_only", "promotion"
    ]
    applies: bool
    measure: MeasureValue
    operator: Literal["eq", "gt", "gte", "lte"]
    threshold: Decimal | int
    passed: bool | None


class GateEvaluation(DomainModel):
    split: Literal["dev", "validation"]
    formal_valid: bool
    quality_passed: bool
    gate_result: Literal["passed", "failed"]
    checks: list[GateCheck]


def _measure(
    numerator: Decimal | int,
    denominator: Decimal | int,
    value: Decimal | int | None,
) -> MeasureValue:
    return MeasureValue(
        numerator=Decimal(numerator),
        denominator=Decimal(denominator),
        value=None if value is None else Decimal(value),
    )


def _from_metric(measure: MetricMeasure) -> MeasureValue:
    return MeasureValue.model_validate(measure.model_dump())


def _prediction_cardinality(
    frozen_queries: Sequence[EvaluationQuery],
    predictions: Sequence[PredictionRecord],
) -> MeasureValue:
    expected_ids = [query.query_id for query in frozen_queries]
    actual_ids = [prediction.query_id for prediction in predictions]
    denominator = len(expected_ids)
    matching = sum(
        expected == actual
        for expected, actual in zip(expected_ids, actual_ids, strict=False)
    )
    valid = len(actual_ids) == denominator and matching == denominator
    return _measure(
        denominator if valid else 0,
        denominator,
        1 if valid else 0,
    )


def _failure_cardinality(
    frozen_queries: Sequence[EvaluationQuery],
    failures: Sequence[EvaluationFailureRecord],
    expected_hard_failures: MeasureValue | None,
    audit_measures: Mapping[str, MeasureValue],
) -> MeasureValue:
    frozen_ids = [query.query_id for query in frozen_queries]
    failure_ids = [failure.query_id for failure in failures]
    if expected_hard_failures is None:
        return _measure(0, 0, None)
    expected_count = expected_hard_failures.numerator
    expected_denominator = Decimal(len(frozen_queries))
    expected_value = (
        expected_count / expected_denominator if expected_denominator else None
    )
    if (
        not expected_count.is_finite()
        or not expected_hard_failures.denominator.is_finite()
        or expected_count != expected_count.to_integral_value()
        or expected_count < 0
        or expected_count > expected_denominator
        or expected_hard_failures.denominator != expected_denominator
        or expected_hard_failures.value != expected_value
    ):
        return _measure(0, 0, None)
    denominator = int(expected_count)
    marked_ids = {
        name.removeprefix(_HARD_FAILED_QUERY_PREFIX): measure
        for name, measure in audit_measures.items()
        if name.startswith(_HARD_FAILED_QUERY_PREFIX)
    }
    markers_valid = (
        len(marked_ids) == denominator
        and set(marked_ids) <= set(frozen_ids)
        and all(
            measure == _measure(1, 1, 1) for measure in marked_ids.values()
        )
    )
    if denominator == 0:
        return _measure(0, 0, None)
    expected_ids = [query_id for query_id in frozen_ids if query_id in marked_ids]
    valid = (
        markers_valid
        and failure_ids == expected_ids
        and len(failure_ids) == denominator
    )
    return _measure(denominator if valid else 0, denominator, 1 if valid else 0)


def _split_from_frozen_queries(
    frozen_queries: Sequence[EvaluationQuery],
) -> Literal["dev", "validation"]:
    splits = {query.metadata.get("split") for query in frozen_queries}
    if splits == {"dev"}:
        return "dev"
    if splits == {"validation"}:
        return "validation"
    raise ValueError("frozen queries must declare one consistent dev or validation split")


def _compare(rule: QualityGateRule, value: Decimal) -> bool:
    threshold = Decimal(rule.threshold)
    if rule.operator == "eq":
        return value == threshold
    if rule.operator == "gt":
        return value > threshold
    if rule.operator == "gte":
        return value >= threshold
    return value <= threshold


def evaluate_gates(
    *,
    frozen_queries: Sequence[EvaluationQuery],
    predictions: Sequence[PredictionRecord],
    failures: Sequence[EvaluationFailureRecord],
    metrics: EvaluationResult,
    audit_measures: Mapping[str, MeasureValue],
    ledger_report: LedgerReport,
    policy: QualityGatePolicy,
) -> GateEvaluation:
    """Evaluate every policy row while only enforcing applicable formal/baseline rows."""
    split = _split_from_frozen_queries(frozen_queries)
    available: dict[str, MeasureValue] = {
        name: _from_metric(measure) for name, measure in metrics.measures.items()
    }
    available.update(audit_measures)
    available["predictions_per_frozen_query_in_order"] = _prediction_cardinality(
        frozen_queries, predictions
    )
    available["supplemental_failure_records_per_hard_failed_query"] = (
        _failure_cardinality(
            frozen_queries,
            failures,
            audit_measures.get("hard_failure_rate"),
            audit_measures,
        )
    )
    available["budget_ledgers_over_hard_cap"] = _measure(
        0 if ledger_report.within_caps else 1,
        1,
        0 if ledger_report.within_caps else 1,
    )

    checks: list[GateCheck] = []
    for rule in policy.rules:
        applies = split in rule.applies_to
        measure = available.get(rule.measure, _measure(0, 0, None))
        passed: bool | None = None
        if applies:
            no_hard_failures = (
                rule.measure == "supplemental_failure_records_per_hard_failed_query"
                and (hard_failure_rate := audit_measures.get("hard_failure_rate"))
                is not None
                and hard_failure_rate.numerator == 0
                and hard_failure_rate.denominator == Decimal(len(frozen_queries))
                and hard_failure_rate.value == 0
                and not failures
                and not any(
                    name.startswith(_HARD_FAILED_QUERY_PREFIX)
                    for name in audit_measures
                )
            )
            if no_hard_failures:
                passed = True
            elif measure.value is None or (
                measure.denominator == 0
                and rule.measure
                not in {"supplemental_failure_records_per_hard_failed_query"}
            ):
                passed = None if rule.classification == "reporting_only" else False
            else:
                passed = _compare(rule, measure.value)
        checks.append(
            GateCheck(
                rule_id=rule.rule_id,
                classification=rule.classification,
                applies=applies,
                measure=measure,
                operator=rule.operator,
                threshold=rule.threshold,
                passed=passed,
            )
        )

    formal_valid = all(
        check.passed is True
        for check in checks
        if check.applies and check.classification == "formal_validity"
    )
    quality_passed = all(
        check.passed is True
        for check in checks
        if check.applies and check.classification == "baseline_quality"
    )
    return GateEvaluation(
        split=split,
        formal_valid=formal_valid,
        quality_passed=quality_passed,
        gate_result="passed" if formal_valid and quality_passed else "failed",
        checks=checks,
    )
