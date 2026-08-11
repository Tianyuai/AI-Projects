"""Pure offline reconstruction, projection, and gate evaluation for Query Evolution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import Field, model_validator

from paper_search.domain.models import DomainModel, Paper, ProviderResult, QuerySpec, SearchPlan, UsageActual, UsageEstimate
from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap, PredictionRecord
from paper_search.evaluation.metrics import MetricSummary, evaluate
from paper_search.evaluation.ranking_metrics import RankingSummary, evaluate_ranking
from paper_search.processing.filter import apply_hard_filters
from paper_search.ranking.fusion import fuse_provider_results

QueryTerminal = Literal[
    "generated",
    "no_op",
    "integrity_failure",
    "dependency_failure",
    "accounting_failure",
    "snapshot_failure",
    "cancelled",
    "not_scheduled",
]
CaptureReplayMatch = Literal["matched", "mismatched", "not_evaluated"]
GateStatus = Literal["passed", "failed", "not_evaluated"]
RunReason = Literal[
    "preflight_failed",
    "generation_failed",
    "dependency_failed",
    "accounting_failed",
    "snapshot_failed",
    "replay_mismatch",
    "cancelled",
    "gate_b_failed",
    "gate_c_failed",
]


class ReplayProvider(Protocol):
    def search(self, query: str, filters: dict[str, object], limit: int) -> ProviderResult[list[Paper]]: ...


class FrozenQueryRecord(DomainModel):
    query_id: str = Field(min_length=1)
    query_spec: QuerySpec
    search_plan: SearchPlan | None = None
    baseline_results: list[ProviderResult[list[Paper]]]
    retrieved_paper_ids: list[str]
    source_index: int = Field(strict=True, ge=0)


class FrozenProbeInputs(DomainModel):
    queries: list[FrozenQueryRecord]
    source_run_id: str = Field(min_length=1)
    source_hashes: dict[str, str]
    expected_query_count: int | None = Field(default=None, strict=True, ge=0)
    expected_total_selected: int | None = Field(default=None, strict=True, ge=0)


class FrozenProbeBaseline(DomainModel):
    queries: list[FrozenQueryRecord]
    source_run_id: str
    source_hashes: dict[str, str]
    expected_query_count: int | None = None
    expected_total_selected: int | None = None

    @model_validator(mode="after")
    def validate_order(self) -> FrozenProbeBaseline:
        if [record.source_index for record in self.queries] != list(range(len(self.queries))):
            raise ValueError("frozen query order is not contiguous")
        if len({record.query_id for record in self.queries}) != len(self.queries):
            raise ValueError("frozen query IDs must be unique")
        if self.expected_query_count is not None and self.query_count != self.expected_query_count:
            raise ValueError("frozen query count does not match expected denominator")
        if self.expected_total_selected is not None and self.total_selected != self.expected_total_selected:
            raise ValueError("frozen selected total does not match expected denominator")
        return self

    @property
    def query_count(self) -> int:
        return len(self.queries)

    @property
    def query_ids(self) -> tuple[str, ...]:
        return tuple(record.query_id for record in self.queries)

    @property
    def total_selected(self) -> int:
        return sum(
            len(
                _project_query(
                    record.query_spec,
                    record.baseline_results,
                    (),
                    retrieved_paper_ids=record.retrieved_paper_ids,
                ).top50_ids
            )
            for record in self.queries
        )


class QueryProjection(DomainModel):
    candidate_papers: list[Paper]
    retrieved_ids: list[str]
    post_filter_ids: tuple[str, ...]
    top50_ids: list[str]
    fusion_sources: tuple[str, ...]
    hard_filter_rejections: int = Field(strict=True, ge=0)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(paper.canonical_id for paper in self.candidate_papers)


class ProbeProjection(DomainModel):
    by_query: dict[str, QueryProjection]


class ProbeIntegrity(DomainModel):
    capture_replay_match: CaptureReplayMatch = "matched"
    locked_query_count: int | None = Field(default=None, strict=True, ge=0)
    terminal_count: int | None = Field(default=None, strict=True, ge=0)
    integrity_failures: int = Field(default=0, strict=True, ge=0)
    provenance_failures: int = Field(default=0, strict=True, ge=0)
    unaccounted_usage_failures: int = Field(default=0, strict=True, ge=0)
    request_failures: int = Field(default=0, strict=True, ge=0)
    source_hash_mismatches: int = Field(default=0, strict=True, ge=0)
    availability_hash_mismatch: bool = False
    lock_hash_mismatch: bool = False
    post_seal_gold_hash_mismatch: bool = False
    limits_respected: bool = True
    aggregate_only: bool = True
    exact_baseline: bool = True
    balanced_production_estimate: Decimal | None = Field(default=None, ge=Decimal("0"))
    run_reason: RunReason | None = None
    warnings: list[str] = Field(default_factory=list)


class ProbeEvaluation(DomainModel):
    baseline_metrics: MetricSummary
    candidate_metrics: MetricSummary
    baseline_ranking: RankingSummary
    candidate_ranking: RankingSummary
    baseline_candidate_gold_count: int = Field(strict=True, ge=0)
    candidate_candidate_gold_count: int = Field(strict=True, ge=0)
    baseline_top50_gold_count: int = Field(strict=True, ge=0)
    candidate_top50_gold_count: int = Field(strict=True, ge=0)
    newly_retrieved_count: int = Field(strict=True, ge=0)
    prior_candidate_gold_retained: bool
    prior_top50_gold_retained: bool
    baseline_hard_filter_rejections: int = Field(strict=True, ge=0)
    candidate_hard_filter_rejections: int = Field(strict=True, ge=0)
    metric_deltas: dict[str, float]
    gate_a: GateStatus
    gate_b: GateStatus
    gate_c: GateStatus
    run_reason: RunReason | None = None
    warnings: list[str] = Field(default_factory=list)
    balanced_production_estimate: Decimal | None = None

    @model_validator(mode="after")
    def validate_finite_deltas(self) -> ProbeEvaluation:
        if any(not math.isfinite(value) for value in self.metric_deltas.values()):
            raise ValueError("metric deltas must be finite")
        return self


class PublicProbeReport(DomainModel):
    schema_version: Literal["query-evolution-probe-report-v1"] = "query-evolution-probe-report-v1"
    gate_a: GateStatus
    gate_b: GateStatus
    gate_c: GateStatus
    run_reason: RunReason | None
    baseline_candidate_gold_count: int = Field(strict=True, ge=0)
    candidate_candidate_gold_count: int = Field(strict=True, ge=0)
    baseline_top50_gold_count: int = Field(strict=True, ge=0)
    candidate_top50_gold_count: int = Field(strict=True, ge=0)
    newly_retrieved_count: int = Field(strict=True, ge=0)
    metric_deltas: dict[str, float]
    balanced_production_estimate: Decimal | None
    warnings: list[str]

    @model_validator(mode="after")
    def validate_finite_values(self) -> PublicProbeReport:
        if any(not math.isfinite(value) for value in self.metric_deltas.values()):
            raise ValueError("public metric deltas must be finite")
        return self


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def calculate_production_estimates(
    actuals_by_operation: Mapping[str, Sequence[UsageActual]],
    *,
    scheduled_by_operation: Mapping[str, Sequence[bool]] | None = None,
) -> dict[str, UsageEstimate]:
    """Bound each operation/dimension by max(actual, ceil(p95 * 1.2))."""
    estimates: dict[str, UsageEstimate] = {}
    dimensions = ("llm_calls", "search_api_calls", "input_tokens", "output_tokens", "elapsed_ms")
    for operation, actuals in actuals_by_operation.items():
        scheduled = scheduled_by_operation.get(operation) if scheduled_by_operation else None
        selected = [
            actual
            for index, actual in enumerate(actuals)
            if scheduled is None or index < len(scheduled) and scheduled[index]
        ]
        if not selected:
            estimates[operation] = UsageEstimate()
            continue
        values: dict[str, int] = {}
        for dimension in dimensions:
            samples = [float(getattr(actual, dimension)) for actual in selected]
            values[dimension] = max(max(int(sample) for sample in samples), math.ceil(_p95(samples) * 1.2))
        costs = [actual.cost_cny for actual in selected if actual.cost_cny is not None]
        cost: Decimal | None = None
        if costs:
            p95_cost = Decimal(str(_p95([float(value) for value in costs]))) * Decimal("1.2")
            cost = max(max(costs), p95_cost).quantize(Decimal("0.000001"))
        estimates[operation] = UsageEstimate(**values, cost_cny=cost)
    return estimates


def _offline_provider_result(papers: Sequence[Paper]) -> ProviderResult[list[Paper]]:
    return ProviderResult(
        data=list(papers),
        usage=UsageActual(),
        provenance={
            "provider": "offline",
            "endpoint": "offline",
            "model_id": "offline",
            "requested_at": "2026-08-10T00:00:00+00:00",
            "response_hash": "sha256:" + "0" * 64,
        },
        cache_hit=True,
        latency_ms=0,
        errors=[],
    )


def offline_provider_result(papers: Sequence[Paper]) -> ProviderResult[list[Paper]]:
    """Expose the existing deterministic offline result adapter."""
    return _offline_provider_result(papers)


def _project_query(
    spec: QuerySpec,
    baseline_results: Sequence[ProviderResult[list[Paper]]],
    additions: Sequence[ProviderResult[list[Paper]]],
    *,
    retrieved_paper_ids: Sequence[str] | None = None,
) -> QueryProjection:
    ordered: list[Paper] = []
    seen: set[str] = set()
    for result in (*baseline_results, *additions):
        for paper in result.data:
            if paper.canonical_id not in seen:
                seen.add(paper.canonical_id)
                ordered.append(paper)
    filtered = apply_hard_filters(ordered, spec)
    post_filter_ids = tuple(item.paper.canonical_id for item in filtered.accepted)
    accepted = set(post_filter_ids)
    fused = fuse_provider_results({"openalex": _offline_provider_result(ordered)}, method="rrf")
    retrieved: list[str] = []
    seen_retrieved: set[str] = set()
    baseline_retrieved = retrieved_paper_ids if retrieved_paper_ids is not None else (
        paper.canonical_id for result in baseline_results for paper in result.data
    )
    for identifier in baseline_retrieved:
        if identifier not in seen_retrieved:
            seen_retrieved.add(identifier)
            retrieved.append(identifier)
    for result in additions:
        for paper in result.data:
            if paper.canonical_id not in seen_retrieved:
                seen_retrieved.add(paper.canonical_id)
                retrieved.append(paper.canonical_id)
    return QueryProjection(
        candidate_papers=ordered,
        retrieved_ids=retrieved,
        post_filter_ids=post_filter_ids,
        top50_ids=[item.paper.canonical_id for item in fused if item.paper.canonical_id in accepted][:50],
        fusion_sources=("openalex",),
        hard_filter_rejections=len(filtered.rejected),
    )


def project_openalex_stream(
    spec: QuerySpec,
    baseline: Sequence[ProviderResult[list[Paper]]],
    additions: Sequence[ProviderResult[list[Paper]]],
) -> QueryProjection:
    """Public production-equivalent projection helper for offline evaluation."""
    return _project_query(spec, baseline, additions)


def reconstruct_frozen_baseline(
    inputs: FrozenProbeInputs | FrozenProbeBaseline,
    replay_provider: ReplayProvider | None,
) -> FrozenProbeBaseline:
    """Validate and project frozen OpenAlex results without network or file I/O."""
    del replay_provider
    if isinstance(inputs, FrozenProbeBaseline):
        return inputs
    if not inputs.queries:
        raise ValueError("frozen inputs must contain queries")
    return FrozenProbeBaseline(
        queries=inputs.queries,
        source_run_id=inputs.source_run_id,
        source_hashes=inputs.source_hashes,
        expected_query_count=inputs.expected_query_count,
        expected_total_selected=inputs.expected_total_selected,
    )


def _availability_ids(availability: Mapping[str, object], query_id: str) -> set[str]:
    value = availability.get(query_id)
    if isinstance(value, Mapping):
        return {key for key, status in value.items() if status in {True, "available"}}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item) for item in value}
    return {key for key, status in availability.items() if status in {True, "available"}}


def select_probe_query_ids(
    baseline: FrozenProbeBaseline,
    gold: Sequence[EvaluationQuery],
    availability: Mapping[str, object],
) -> tuple[str, ...]:
    """Select available-but-not-retrieved gold associations in frozen order."""
    gold_by_id = {record.query_id: record for record in gold}
    selected: list[str] = []
    for record in baseline.queries:
        gold_record = gold_by_id.get(record.query_id)
        if gold_record is None:
            continue
        retrieved = set(
            _project_query(
                record.query_spec,
                record.baseline_results,
                [],
                retrieved_paper_ids=record.retrieved_paper_ids,
            ).retrieved_ids
        )
        available = _availability_ids(availability, record.query_id)
        if any(identifier in available and identifier not in retrieved for identifier in gold_record.relevant_paper_ids):
            selected.append(record.query_id)
    return tuple(selected)


def merge_probe_results(
    baseline: FrozenProbeBaseline,
    additions: Mapping[str, Sequence[ProviderResult[list[Paper]]]],
) -> ProbeProjection:
    """Append generated search slots after the frozen baseline in query order."""
    projected: dict[str, QueryProjection] = {}
    for record in baseline.queries:
        projected[record.query_id] = _project_query(
            record.query_spec,
            record.baseline_results,
            additions.get(record.query_id, ()),
            retrieved_paper_ids=record.retrieved_paper_ids,
        )
    unknown = set(additions).difference(projected)
    if unknown:
        raise ValueError(f"probe additions contain unknown queries: {sorted(unknown)}")
    return ProbeProjection(by_query=projected)


def _predictions(
    query_ids: Sequence[str],
    projections: Mapping[str, QueryProjection],
) -> list[PredictionRecord]:
    return [
        PredictionRecord(query_id=query_id, predicted_paper_ids=list(projections[query_id].top50_ids))
        for query_id in query_ids
    ]


def _resolved(identifier: str, id_map: IdentifierMap | None) -> str:
    return id_map.resolve(identifier) if id_map is not None else identifier


def _gold_associations(
    gold: Sequence[EvaluationQuery],
    identifiers_by_query: Mapping[str, Sequence[str]],
    id_map: IdentifierMap | None,
) -> set[tuple[str, str]]:
    associations: set[tuple[str, str]] = set()
    for record in gold:
        retrieved = {_resolved(identifier, id_map) for identifier in identifiers_by_query[record.query_id]}
        associations.update(
            (record.query_id, resolved)
            for identifier in record.relevant_paper_ids
            if (resolved := _resolved(identifier, id_map)) in retrieved
        )
    return associations


def count_gold_associations(
    gold: Sequence[EvaluationQuery],
    identifiers_by_query: Mapping[str, Sequence[str]],
    id_map: IdentifierMap | None,
) -> int:
    """Count unique resolved ``(query_id, paper_id)`` gold associations."""
    return len(_gold_associations(gold, identifiers_by_query, id_map))


def _retains_prior_gold(
    gold: Sequence[EvaluationQuery],
    baseline: Mapping[str, QueryProjection],
    candidate: Mapping[str, QueryProjection],
    id_map: IdentifierMap | None,
    *,
    top50: bool,
) -> bool:
    for record in gold:
        before = set(baseline[record.query_id].top50_ids if top50 else baseline[record.query_id].retrieved_ids)
        after = set(candidate[record.query_id].top50_ids if top50 else candidate[record.query_id].retrieved_ids)
        before_resolved = {_resolved(value, id_map) for value in before}
        after_resolved = {_resolved(value, id_map) for value in after}
        prior_gold = {
            _resolved(identifier, id_map)
            for identifier in record.relevant_paper_ids
            if _resolved(identifier, id_map) in before_resolved
        }
        if not prior_gold.issubset(after_resolved):
            return False
    return True


def _delta(candidate: float, baseline: float) -> float:
    return candidate - baseline


def _gate_a(baseline: FrozenProbeBaseline, integrity: ProbeIntegrity) -> bool:
    if baseline.query_count != 60 or baseline.expected_query_count != 60:
        return False
    if baseline.total_selected != 2910 or baseline.expected_total_selected != 2910:
        return False
    probe_counts_match = integrity.locked_query_count is None and integrity.terminal_count is None
    if integrity.locked_query_count is not None and integrity.terminal_count is not None:
        probe_counts_match = integrity.locked_query_count > 0 and integrity.locked_query_count == integrity.terminal_count
    return all(
        (
            integrity.exact_baseline,
            integrity.capture_replay_match == "matched",
            integrity.integrity_failures == 0,
            integrity.provenance_failures == 0,
            integrity.unaccounted_usage_failures == 0,
            integrity.request_failures == 0,
            integrity.source_hash_mismatches == 0,
            not integrity.availability_hash_mismatch,
            not integrity.lock_hash_mismatch,
            not integrity.post_seal_gold_hash_mismatch,
            integrity.limits_respected,
            integrity.aggregate_only,
            probe_counts_match,
        )
    )


def evaluate_probe(
    baseline: FrozenProbeBaseline,
    projection: ProbeProjection,
    gold: Sequence[EvaluationQuery],
    id_map: IdentifierMap | None,
    integrity: ProbeIntegrity | Mapping[str, object],
) -> ProbeEvaluation:
    """Score baseline and candidate projections, then apply strict Gate A/B/C."""
    checked_integrity = integrity if isinstance(integrity, ProbeIntegrity) else ProbeIntegrity.model_validate(integrity)
    if set(projection.by_query) != set(baseline.query_ids):
        raise ValueError("projection query set does not match frozen baseline")
    baseline_projection = merge_probe_results(baseline, {})
    query_ids = baseline.query_ids
    baseline_metrics = evaluate(gold, _predictions(query_ids, baseline_projection.by_query), id_map=id_map)
    candidate_metrics = evaluate(gold, _predictions(query_ids, projection.by_query), id_map=id_map)
    baseline_ranking = evaluate_ranking(gold, _predictions(query_ids, baseline_projection.by_query), id_map=id_map)
    candidate_ranking = evaluate_ranking(gold, _predictions(query_ids, projection.by_query), id_map=id_map)

    baseline_retrieved = {query_id: item.retrieved_ids for query_id, item in baseline_projection.by_query.items()}
    candidate_retrieved = {query_id: item.retrieved_ids for query_id, item in projection.by_query.items()}
    baseline_top50 = {query_id: item.top50_ids for query_id, item in baseline_projection.by_query.items()}
    candidate_top50 = {query_id: item.top50_ids for query_id, item in projection.by_query.items()}
    baseline_gold_pairs = _gold_associations(gold, baseline_retrieved, id_map)
    candidate_gold_pairs = _gold_associations(gold, candidate_retrieved, id_map)
    baseline_top50_pairs = _gold_associations(gold, baseline_top50, id_map)
    candidate_top50_pairs = _gold_associations(gold, candidate_top50, id_map)
    baseline_candidate_gold = len(baseline_gold_pairs)
    candidate_candidate_gold = len(candidate_gold_pairs)
    baseline_top50_gold = len(baseline_top50_pairs)
    candidate_top50_gold = len(candidate_top50_pairs)
    metric_deltas = {
        "macro_f1": _delta(candidate_metrics.summary.macro_f1, baseline_metrics.summary.macro_f1),
        "macro_recall": _delta(candidate_metrics.summary.macro_recall, baseline_metrics.summary.macro_recall),
        "macro_mrr": _delta(candidate_ranking.summary.macro_mrr, baseline_ranking.summary.macro_mrr),
        "macro_ndcg": _delta(candidate_ranking.summary.macro_ndcg, baseline_ranking.summary.macro_ndcg),
    }
    prior_candidate_retained = baseline_gold_pairs.issubset(candidate_gold_pairs)
    prior_top50_retained = _retains_prior_gold(gold, baseline_projection.by_query, projection.by_query, id_map, top50=True)
    gate_a_passed = _gate_a(baseline, checked_integrity)
    gate_b_passed = gate_a_passed and candidate_candidate_gold > 14 and len(candidate_gold_pairs - baseline_gold_pairs) >= 1 and prior_candidate_retained
    baseline_rejections = sum(item.hard_filter_rejections for item in baseline_projection.by_query.values())
    candidate_rejections = sum(item.hard_filter_rejections for item in projection.by_query.values())
    estimate = checked_integrity.balanced_production_estimate
    gate_c_passed = gate_b_passed and candidate_top50_gold > 8 and prior_top50_retained and metric_deltas["macro_f1"] >= 0.01 and all(metric_deltas[name] >= 0 for name in ("macro_recall", "macro_mrr", "macro_ndcg")) and candidate_rejections <= baseline_rejections and estimate is not None and estimate > 0
    return ProbeEvaluation(
        baseline_metrics=baseline_metrics.summary,
        candidate_metrics=candidate_metrics.summary,
        baseline_ranking=baseline_ranking.summary,
        candidate_ranking=candidate_ranking.summary,
        baseline_candidate_gold_count=baseline_candidate_gold,
        candidate_candidate_gold_count=candidate_candidate_gold,
        baseline_top50_gold_count=baseline_top50_gold,
        candidate_top50_gold_count=candidate_top50_gold,
        newly_retrieved_count=len(candidate_gold_pairs - baseline_gold_pairs),
        prior_candidate_gold_retained=prior_candidate_retained,
        prior_top50_gold_retained=prior_top50_retained,
        baseline_hard_filter_rejections=baseline_rejections,
        candidate_hard_filter_rejections=candidate_rejections,
        metric_deltas=metric_deltas,
        gate_a="passed" if gate_a_passed else "failed",
        gate_b="passed" if gate_b_passed else "failed" if gate_a_passed else "not_evaluated",
        gate_c="passed" if gate_c_passed else "failed" if gate_b_passed else "not_evaluated",
        run_reason=checked_integrity.run_reason,
        warnings=list(checked_integrity.warnings),
        balanced_production_estimate=estimate,
    )


def public_probe_report(evaluation: ProbeEvaluation) -> PublicProbeReport:
    """Return a report containing only aggregate metrics and fixed statuses."""
    return PublicProbeReport(
        gate_a=evaluation.gate_a,
        gate_b=evaluation.gate_b,
        gate_c=evaluation.gate_c,
        run_reason=evaluation.run_reason,
        baseline_candidate_gold_count=evaluation.baseline_candidate_gold_count,
        candidate_candidate_gold_count=evaluation.candidate_candidate_gold_count,
        baseline_top50_gold_count=evaluation.baseline_top50_gold_count,
        candidate_top50_gold_count=evaluation.candidate_top50_gold_count,
        newly_retrieved_count=evaluation.newly_retrieved_count,
        metric_deltas=evaluation.metric_deltas,
        balanced_production_estimate=evaluation.balanced_production_estimate,
        warnings=evaluation.warnings,
    )


__all__ = [
    "CaptureReplayMatch",
    "FrozenProbeBaseline",
    "FrozenProbeInputs",
    "FrozenQueryRecord",
    "GateStatus",
    "ProbeEvaluation",
    "ProbeIntegrity",
    "ProbeProjection",
    "PublicProbeReport",
    "QueryProjection",
    "QueryTerminal",
    "count_gold_associations",
    "evaluate_probe",
    "merge_probe_results",
    "offline_provider_result",
    "calculate_production_estimates",
    "project_openalex_stream",
    "public_probe_report",
    "reconstruct_frozen_baseline",
    "select_probe_query_ids",
]
