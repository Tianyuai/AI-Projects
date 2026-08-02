"""Shared reconstruction of formal Gate audit measures from published evidence."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from paper_search.control.ledger import LedgerReport
from paper_search.evaluation.business_results import BusinessResultRecord
from paper_search.evaluation.dataset import EvaluationQuery
from paper_search.evaluation.execution_adapter import (
    EvaluationExecutionRecord,
    EvaluationFailureRecord,
)
from paper_search.evaluation.gates import MeasureValue


def _measure(numerator: int | Decimal, denominator: int | Decimal) -> MeasureValue:
    numerator_value = Decimal(numerator)
    denominator_value = Decimal(denominator)
    return MeasureValue(
        numerator=numerator_value,
        denominator=denominator_value,
        value=(
            numerator_value / denominator_value if denominator_value else None
        ),
    )


def formal_audit_measures(
    *,
    frozen_queries: Sequence[EvaluationQuery],
    executions: Sequence[EvaluationExecutionRecord],
    business_results: Sequence[BusinessResultRecord],
    failures: Sequence[EvaluationFailureRecord],
    ledger_report: LedgerReport,
) -> dict[str, MeasureValue]:
    """Derive all applicable enforced and core reporting measures."""
    count = len(frozen_queries)
    diagnostics = [item for execution in executions for item in execution.diagnostics]
    error_codes = [error.code for item in diagnostics for error in item.errors]
    hard_failure_count = len(failures)
    actual_matches = all(
        getattr(ledger_report.actual, field)
        == sum(getattr(execution.usage, field) for execution in executions)
        for field in (
            "search_api_calls",
            "llm_calls",
            "input_tokens",
            "output_tokens",
            "elapsed_ms",
        )
    )
    execution_costs = [execution.usage.cost_cny for execution in executions]
    expected_cost = (
        sum((value for value in execution_costs if value is not None), Decimal("0"))
        if all(value is not None for value in execution_costs)
        else None
    )
    actual_matches = actual_matches and ledger_report.actual.cost_cny == expected_cost
    clean_diagnostics = sum(
        item.endpoint == "dependency"
        and all(
            error.request_id is None
            and error.provider == item.dependency
            and error.message == "Dependency execution reported an error"
            for error in item.errors
        )
        for item in diagnostics
    )
    latency_values = sorted(execution.usage.elapsed_ms for execution in executions)
    p50 = latency_values[(len(latency_values) - 1) // 2] if latency_values else 0
    p95 = latency_values[max(0, (95 * len(latency_values) + 99) // 100 - 1)] if latency_values else 0
    measures = {
        "integrity_failures": _measure(
            sum(failure.error_code == "integrity_failure" for failure in failures),
            1,
        ),
        "provenance_failures": _measure(
            sum(code in {"integrity_failure", "snapshot_unavailable"} for code in error_codes),
            1,
        ),
        "sanitization_failures": _measure(len(diagnostics) - clean_diagnostics, 1),
        "unaccounted_usage_failures": _measure(0 if actual_matches else 1, 1),
        "valid_model_produced_query_analysis_rate": _measure(
            sum(
                record.query_analysis is not None and record.planner_status == "primary"
                for record in business_results
            ),
            count,
        ),
        "parseable_configured_retrieval_response_rate": _measure(
            sum(not item.errors for item in diagnostics),
            len(diagnostics),
        ),
        "hard_filter_absolute_recall_loss": _measure(
            sum(
                any(warning.startswith("hard_filter_recall_loss") for warning in record.warnings)
                for record in business_results
            ),
            count,
        ),
        "hard_failure_rate": _measure(hard_failure_count, count),
        "partial_result_rate": _measure(
            sum(record.is_partial for record in business_results), count
        ),
        "planner_fallback_rate": _measure(
            sum(record.planner_fallback for record in business_results), count
        ),
        "latency_p50_ms": _measure(p50, 1),
        "latency_p95_ms": _measure(p95, 1),
        "external_calls": _measure(ledger_report.actual.search_api_calls, 1),
        "actual_tokens": _measure(
            ledger_report.actual.input_tokens + ledger_report.actual.output_tokens, 1
        ),
        "valued_cost_cny": _measure(ledger_report.actual.cost_cny or 0, 1),
        "cache_hit_rate": _measure(
            sum(item.cache_hit for item in diagnostics), len(diagnostics)
        ),
    }
    measures.update(
        {
            f"hard_failed_query:{failure.query_id}": _measure(1, 1)
            for failure in failures
        }
    )
    return measures


__all__ = ["formal_audit_measures"]
