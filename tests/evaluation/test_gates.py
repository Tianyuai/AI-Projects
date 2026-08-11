from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from paper_search.control.ledger import LedgerReport
from paper_search.control.pricing import load_quality_gate_policy
from paper_search.domain.models import UsageActual, UsageEstimate
from paper_search.evaluation.dataset import EvaluationQuery, PredictionRecord
from paper_search.evaluation.execution_adapter import EvaluationFailureRecord
from paper_search.evaluation.gates import (
    MeasureValue,
    compare_quality_gate_rule,
    evaluate_gates,
)
from paper_search.evaluation.metrics import evaluate


POLICY = load_quality_gate_policy(Path("configs/quality_gates_v1.yaml"))


def test_public_gate_comparator_preserves_required_boundaries() -> None:
    rules = {rule.rule_id: rule for rule in POLICY.rules}

    assert compare_quality_gate_rule(
        rules["hard-filter-recall-loss"], Decimal("0.02")
    )
    assert not compare_quality_gate_rule(
        rules["hard-filter-recall-loss"], Decimal("0.0201")
    )
    assert compare_quality_gate_rule(
        rules["macro-recall-positive"], Decimal("0.0001")
    )
    assert not compare_quality_gate_rule(
        rules["macro-recall-positive"], Decimal("0")
    )


def _measure(value: str, denominator: str = "100") -> MeasureValue:
    denominator_value = Decimal(denominator)
    return MeasureValue(
        numerator=Decimal(value) * denominator_value,
        denominator=denominator_value,
        value=Decimal(value),
    )


def _ledger(*, within_caps: bool = True) -> LedgerReport:
    return LedgerReport(
        run_id="run-1",
        reserved=UsageEstimate(cost_cny=Decimal("0")),
        actual=UsageActual(cost_cny=Decimal("0")),
        run_cap_cny=Decimal("18"),
        project_actual_cny=Decimal("0"),
        project_soft_stop_cny=Decimal("160"),
        project_hard_cap_cny=Decimal("200"),
        within_caps=within_caps,
    )


def _inputs() -> tuple[list[EvaluationQuery], list[PredictionRecord]]:
    return (
        [
            EvaluationQuery(
                query_id="q1",
                query="one",
                relevant_paper_ids=["openalex:W1"],
                metadata={"split": "dev"},
            )
        ],
        [PredictionRecord(query_id="q1", predicted_paper_ids=["openalex:W1"])],
    )


def _failure(query_id: str = "q1") -> EvaluationFailureRecord:
    return EvaluationFailureRecord(
        query_id=query_id,
        run_id=f"run-{query_id}",
        error_code="dependency_failure",
        retryable=True,
        stop_reason="dependency_failure",
        usage=UsageActual(),
        dependency_error_codes=[],
        diagnostics=[],
        diagnostics_sha256="sha256:" + "a" * 64,
    )


def _audits(**changes: MeasureValue) -> dict[str, MeasureValue]:
    values = {
        "integrity_failures": _measure("0", "1"),
        "provenance_failures": _measure("0", "1"),
        "sanitization_failures": _measure("0", "1"),
        "unaccounted_usage_failures": _measure("0", "1"),
        "valid_model_produced_query_analysis_rate": _measure("0.99"),
        "parseable_configured_retrieval_response_rate": _measure("0.95"),
        "hard_filter_absolute_recall_loss": _measure("0.02"),
        "hard_failure_rate": _measure("0", "1"),
    }
    values.update(changes)
    return values


def _evaluate(
    *,
    frozen: list[EvaluationQuery] | None = None,
    predictions: list[PredictionRecord] | None = None,
    failures: list[EvaluationFailureRecord] | None = None,
    audits: dict[str, MeasureValue] | None = None,
    within_caps: bool = True,
):
    default_frozen, default_predictions = _inputs()
    selected_frozen = default_frozen if frozen is None else frozen
    selected_predictions = default_predictions if predictions is None else predictions
    try:
        metric_result = evaluate(selected_frozen, selected_predictions)
    except ValueError:
        metric_result = evaluate(selected_frozen, [])
    return evaluate_gates(
        frozen_queries=selected_frozen,
        predictions=selected_predictions,
        failures=[] if failures is None else failures,
        metrics=metric_result,
        audit_measures=_audits() if audits is None else audits,
        ledger_report=_ledger(within_caps=within_caps),
        policy=POLICY,
    )


def _check(evaluation, rule_id: str):
    return next(check for check in evaluation.checks if check.rule_id == rule_id)


@pytest.mark.parametrize(
    ("rule_id", "measure", "boundary", "failing"),
    [
        ("model-produced-analysis-rate", "valid_model_produced_query_analysis_rate", "0.99", "0.9899"),
        ("retrieval-response-rate", "parseable_configured_retrieval_response_rate", "0.95", "0.9499"),
        ("hard-filter-recall-loss", "hard_filter_absolute_recall_loss", "0.02", "0.0201"),
    ],
)
def test_baseline_boundaries_pass_and_adjacent_values_fail(
    rule_id: str,
    measure: str,
    boundary: str,
    failing: str,
) -> None:
    passed = _evaluate(audits=_audits(**{measure: _measure(boundary)}))
    failed = _evaluate(audits=_audits(**{measure: _measure(failing)}))

    assert _check(passed, rule_id).passed is True
    assert _check(failed, rule_id).passed is False
    assert failed.quality_passed is False
    assert failed.gate_result == "failed"


def test_strict_positive_recall_boundaries() -> None:
    frozen, predictions = _inputs()
    positive = _evaluate()
    zero_metrics = evaluate(
        frozen,
        [PredictionRecord(query_id="q1", predicted_paper_ids=[])],
    )
    zero = evaluate_gates(
        frozen_queries=frozen,
        predictions=predictions,
        failures=[],
        metrics=zero_metrics,
        audit_measures=_audits(),
        ledger_report=_ledger(),
        policy=POLICY,
    )

    assert _check(positive, "macro-recall-positive").passed is True
    assert _check(positive, "micro-recall-positive").passed is True
    assert _check(zero, "macro-recall-positive").passed is False
    assert _check(zero, "micro-recall-positive").passed is False


@pytest.mark.parametrize(
    "measure",
    ["integrity_failures", "provenance_failures", "sanitization_failures", "unaccounted_usage_failures"],
)
def test_each_formal_audit_failure_invalidates_run(measure: str) -> None:
    evaluation = _evaluate(audits=_audits(**{measure: _measure("1", "1")}))

    assert evaluation.formal_valid is False
    assert evaluation.gate_result == "failed"


def test_prediction_order_and_cardinality_are_formal_inputs() -> None:
    frozen = [
        EvaluationQuery(query_id="q1", query="one", metadata={"split": "dev"}),
        EvaluationQuery(query_id="q2", query="two", metadata={"split": "dev"}),
    ]
    missing = [PredictionRecord(query_id="q1")]
    reordered = [PredictionRecord(query_id="q2"), PredictionRecord(query_id="q1")]

    extra = [
        PredictionRecord(query_id="q1"),
        PredictionRecord(query_id="q2"),
        PredictionRecord(query_id="extra"),
    ]
    duplicate = [PredictionRecord(query_id="q1"), PredictionRecord(query_id="q1")]

    for predictions in (missing, reordered, extra, duplicate):
        evaluation = _evaluate(frozen=frozen, predictions=predictions)
        check = _check(evaluation, "prediction-cardinality")
        assert check.passed is False
        assert evaluation.formal_valid is False


def test_hard_failure_requires_exactly_one_supplemental_record_per_query() -> None:
    one_expected = _audits(hard_failure_rate=_measure("1", "1"))
    one_expected["hard_failed_query:q1"] = _measure("1", "1")
    missing = _evaluate(failures=[], audits=one_expected)
    single = _evaluate(failures=[_failure()], audits=one_expected)
    duplicate = _evaluate(failures=[_failure(), _failure()], audits=one_expected)
    unknown = _evaluate(failures=[_failure("unknown")], audits=one_expected)

    assert _check(missing, "hard-failure-cardinality").passed is False
    assert _check(single, "hard-failure-cardinality").passed is True
    assert _check(duplicate, "hard-failure-cardinality").passed is False
    assert _check(unknown, "hard-failure-cardinality").passed is False


def test_two_expected_hard_failures_require_exact_linked_frozen_order() -> None:
    frozen = [
        EvaluationQuery(query_id="q1", query="one", metadata={"split": "dev"}),
        EvaluationQuery(query_id="q2", query="two", metadata={"split": "dev"}),
    ]
    audits = _audits(hard_failure_rate=_measure("1", "2"))
    audits["hard_failed_query:q1"] = _measure("1", "1")
    audits["hard_failed_query:q2"] = _measure("1", "1")

    ordered = _evaluate(
        frozen=frozen,
        failures=[_failure("q1"), _failure("q2")],
        audits=audits,
    )
    invalid_sets = (
        [_failure("q2"), _failure("q1")],
        [_failure("q1")],
        [_failure("q1"), _failure("q1")],
        [_failure("q1"), _failure("q2"), _failure("unknown")],
    )

    assert _check(ordered, "hard-failure-cardinality").passed is True
    for failures in invalid_sets:
        evaluation = _evaluate(frozen=frozen, failures=failures, audits=audits)
        assert _check(evaluation, "hard-failure-cardinality").passed is False
        assert evaluation.formal_valid is False


def test_nonzero_hard_failure_rate_requires_identity_bearing_audit_markers() -> None:
    missing_identity = _evaluate(
        failures=[_failure()],
        audits=_audits(hard_failure_rate=_measure("1", "1")),
    )

    assert _check(missing_identity, "hard-failure-cardinality").passed is False


def test_hard_failure_cardinality_rejects_inconsistent_authoritative_rate() -> None:
    invalid = _evaluate(
        failures=[],
        audits=_audits(hard_failure_rate=_measure("0", "0")),
    )

    assert _check(invalid, "hard-failure-cardinality").passed is False
    assert invalid.formal_valid is False


def test_zero_expected_hard_failures_rejects_an_extra_failure_record() -> None:
    evaluation = _evaluate(failures=[_failure()])

    check = _check(evaluation, "hard-failure-cardinality")
    assert check.measure.value is None
    assert check.passed is False
    assert evaluation.formal_valid is False
    assert evaluation.gate_result == "failed"


def test_ledger_over_cap_is_formal_failure() -> None:
    evaluation = _evaluate(within_caps=False)

    assert _check(evaluation, "budget-ledgers-within-cap").passed is False
    assert evaluation.formal_valid is False


def test_zero_denominator_applied_audit_is_invalid() -> None:
    evaluation = _evaluate(
        audits=_audits(valid_model_produced_query_analysis_rate=_measure("0", "0"))
    )

    assert _check(evaluation, "model-produced-analysis-rate").passed is False
    assert evaluation.quality_passed is False


def test_frozen_audit_thresholds_are_reported_but_not_applied_to_dev() -> None:
    evaluation = _evaluate()

    strong = _check(evaluation, "strong-constraint-recall")
    fuzzy = _check(evaluation, "fuzzy-merge-accuracy")
    denominator = _check(evaluation, "fuzzy-merge-audit-denominator")
    assert (strong.threshold, strong.applies, strong.passed) == (Decimal("0.90"), False, None)
    assert (fuzzy.threshold, fuzzy.applies, fuzzy.passed) == (Decimal("0.98"), False, None)
    assert (denominator.threshold, denominator.applies, denominator.passed) == (0, False, None)


def test_reporting_f1_check_does_not_control_gate_result() -> None:
    evaluation = _evaluate()
    check = _check(evaluation, "report-macro-f1")

    assert check.classification == "reporting_only"
    assert check.applies is True
    assert evaluation.gate_result == "passed"


def test_complete_run_can_be_formally_valid_but_quality_failed() -> None:
    evaluation = _evaluate(
        audits=_audits(valid_model_produced_query_analysis_rate=_measure("0.98"))
    )

    assert evaluation.formal_valid is True
    assert evaluation.quality_passed is False
    assert evaluation.gate_result == "failed"


def test_all_policy_rows_are_reported_in_policy_order() -> None:
    evaluation = _evaluate()

    assert [check.rule_id for check in evaluation.checks] == [rule.rule_id for rule in POLICY.rules]
    assert all(check.measure.denominator >= 0 for check in evaluation.checks)
