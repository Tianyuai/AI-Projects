"""Shared reconstruction of formal Gate audit measures from published evidence."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from paper_search.control.ledger import LedgerReport
from paper_search.control.pricing import QualityGatePolicy
from paper_search.evaluation.business_results import BusinessResultRecord
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    IdentifierMap,
    PredictionRecord,
    normalize_paper_id,
)
from paper_search.evaluation.execution_adapter import (
    EvaluationExecutionRecord,
    EvaluationFailureRecord,
)
from paper_search.evaluation.gates import MeasureValue
from paper_search.evaluation.metrics import EvaluationResult, evaluate


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


def _has_parseable_openalex(execution: EvaluationExecutionRecord) -> bool:
    diagnostics = [
        item for item in execution.diagnostics if item.dependency == "openalex"
    ]
    return bool(diagnostics) and all(not item.errors for item in diagnostics)


def formal_audit_measures(
    *,
    frozen_queries: Sequence[EvaluationQuery],
    executions: Sequence[EvaluationExecutionRecord],
    business_results: Sequence[BusinessResultRecord],
    failures: Sequence[EvaluationFailureRecord],
    ledger_report: LedgerReport,
    identifier_map: IdentifierMap | None = None,
    metrics: EvaluationResult | None = None,
) -> dict[str, MeasureValue]:
    """Derive all applicable enforced and core reporting measures."""
    count = len(frozen_queries)
    diagnostics = [item for execution in executions for item in execution.diagnostics]
    error_codes = [error.code for item in diagnostics for error in item.errors]
    hard_failure_count = len(failures)
    resolve = identifier_map.resolve if identifier_map is not None else lambda value: value
    execution_by_query = {execution.query_id: execution for execution in executions}
    relevant_count = 0
    retrieved_relevant_count = 0
    post_filter_relevant_count = 0
    for query in frozen_queries:
        relevant = {resolve(identifier) for identifier in query.relevant_paper_ids}
        relevant_count += len(relevant)
        execution = execution_by_query.get(query.query_id)
        if execution is None:
            continue
        parseable = _has_parseable_openalex(execution)
        retrieved = (
            {resolve(identifier) for identifier in execution.retrieved_paper_ids}
            if parseable
            else set()
        )
        post_filter = (
            {resolve(identifier) for identifier in execution.post_filter_paper_ids}
            if parseable
            else set()
        )
        retrieved_relevant_count += len(relevant & retrieved)
        post_filter_relevant_count += len(relevant & post_filter)
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
    cached_latency_values = sorted(
        item.latency_ms for item in diagnostics if item.cache_hit
    )
    cached_p50 = (
        cached_latency_values[(len(cached_latency_values) - 1) // 2]
        if cached_latency_values
        else 0
    )

    def canonical_id(identifier: str) -> bool:
        try:
            return normalize_paper_id(identifier) == identifier
        except ValueError:
            return False

    structured_by_query = {record.query_id: record for record in business_results}
    schema_valid_queries = sum(
        query.query_id in structured_by_query for query in frozen_queries
    )
    valid_link_queries = 0
    reason_complete_queries = 0
    verifiable_edge_queries = 0
    fabricated_count = 0
    for query in frozen_queries:
        record = structured_by_query.get(query.query_id)
        execution = execution_by_query.get(query.query_id)
        if record is None:
            continue
        ranked = [*record.high_relevance, *record.partial_relevance]
        ranked_ids = {item.paper.canonical_id for item in ranked}
        linked_ids = [
            *record.selected_paper_ids,
            *ranked_ids,
            *(edge.citing_canonical_id for edge in record.citation_edges),
            *(edge.cited_canonical_id for edge in record.citation_edges),
        ]
        links_valid = all(canonical_id(identifier) for identifier in linked_ids)
        valid_link_queries += links_valid
        reason_complete_queries += set(record.selected_paper_ids) <= ranked_ids
        edges_valid = all(
            canonical_id(edge.citing_canonical_id)
            and canonical_id(edge.cited_canonical_id)
            and bool(edge.source_edge_hash)
            for edge in record.citation_edges
        )
        verifiable_edge_queries += edges_valid
        post_filter_ids = set(execution.post_filter_paper_ids) if execution else set()
        fabricated_count += sum(
            not canonical_id(identifier) or identifier not in post_filter_ids
            for identifier in record.selected_paper_ids
        )
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
            retrieved_relevant_count,
            relevant_count,
        ),
        "hard_filter_absolute_recall_loss": _measure(
            retrieved_relevant_count - post_filter_relevant_count,
            relevant_count,
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
        "schema_valid_rate": _measure(schema_valid_queries, count),
        "valid_paper_link_rate": _measure(valid_link_queries, count),
        "reason_complete_rate": _measure(reason_complete_queries, count),
        "verifiable_citation_edge_rate": _measure(verifiable_edge_queries, count),
        "fabricated_paper_or_relation_count": _measure(fabricated_count, 1),
        "cached_repeat_latency_p50_ms": _measure(cached_p50, 1),
    }
    if metrics is not None:
        splits = {query.metadata.get("split") for query in frozen_queries}
        if len(splits) != 1 or not splits <= {"dev", "validation"}:
            raise ValueError("formal reporting requires one dev or validation split")
        split = str(next(iter(splits)))
        macro_f1 = metrics.measures["macro_f1"]
        measures[f"{split}_macro_f1"] = MeasureValue.model_validate(macro_f1.model_dump())
        raw_predictions = [
            PredictionRecord(
                query_id=query.query_id,
                predicted_paper_ids=(
                    execution_by_query[query.query_id].retrieved_paper_ids
                    if query.query_id in execution_by_query
                    and _has_parseable_openalex(execution_by_query[query.query_id])
                    else []
                ),
            )
            for query in frozen_queries
        ]
        raw_macro_f1 = evaluate(
            frozen_queries, raw_predictions, id_map=identifier_map
        ).measures["macro_f1"].value
        delta = (macro_f1.value or Decimal(0)) - (raw_macro_f1 or Decimal(0))
        measures[f"{split}_macro_f1_delta_vs_raw_openalex"] = _measure(delta, 1)
    measures.update(
        {
            f"hard_failed_query:{failure.query_id}": _measure(1, 1)
            for failure in failures
        }
    )
    return measures


def complete_policy_measures(
    measures: dict[str, MeasureValue],
    *,
    policy: QualityGatePolicy,
    split: str,
) -> dict[str, MeasureValue]:
    """Preserve evidence without manufacturing rows that can mask real metrics."""
    completed = measures.copy()
    del policy, split
    return completed


__all__ = ["complete_policy_measures", "formal_audit_measures"]
