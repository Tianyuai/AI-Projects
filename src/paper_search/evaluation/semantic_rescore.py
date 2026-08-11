"""Source-neutral identifier-semantic rescoring and bottleneck decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator, model_validator

from paper_search.control.pricing import QualityGatePolicy, QualityGateRule
from paper_search.domain.models import (
    DomainModel,
    NonEmptyStr,
    NonNegativeInt,
    Sha256,
)
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    IdentifierMap,
    PredictionRecord,
    normalize_paper_id,
)
from paper_search.evaluation.gates import compare_quality_gate_rule
from paper_search.evaluation.identifier_semantics import arxiv_anchor
from paper_search.evaluation.metrics import MetricMeasure, evaluate
from paper_search.evaluation.ranking_metrics import evaluate_ranking


SourceLabel = Literal[
    "formal_baseline_2026_08_10",
    "formal_baseline_2026_08_09",
    "legacy_title_2026_08_05",
    "query_evolution_prompt_v2",
]
SourceKind = Literal["formal_run", "legacy_hash_bound_run", "sealed_probe"]
SourceVerificationStatus = Literal[
    "formal_validated",
    "legacy_hash_bound",
    "probe_verified",
]
CaptureReplayStatus = Literal["not_applicable", "matched"]
LossStage = Literal["not_retrieved", "filtered_out", "ranked_outside_top50"]
NextDirection = Literal["retrieval_query", "retention_filter", "ranking_selector"]

_FIXED_SOURCE_ORDER: tuple[SourceLabel, ...] = (
    "formal_baseline_2026_08_10",
    "formal_baseline_2026_08_09",
    "legacy_title_2026_08_05",
    "query_evolution_prompt_v2",
)
_QUALITY_MEASURES = (
    "hard_filter_absolute_recall_loss",
    "macro_identifier_map_recall",
    "micro_identifier_map_recall",
)


class SourceProjection(DomainModel):
    """The raw, sealed source values needed for source-neutral scoring."""

    label: SourceLabel
    kind: SourceKind
    verification_status: SourceVerificationStatus
    capture_replay_status: CaptureReplayStatus
    binding_hashes: dict[NonEmptyStr, Sha256] = Field(min_length=1)
    query_ids: tuple[NonEmptyStr, ...]
    retrieved_paper_ids: dict[NonEmptyStr, tuple[NonEmptyStr, ...]]
    post_filter_paper_ids: dict[NonEmptyStr, tuple[NonEmptyStr, ...]]
    selected_paper_ids: dict[NonEmptyStr, tuple[NonEmptyStr, ...]]

    @field_validator(
        "retrieved_paper_ids", "post_filter_paper_ids", "selected_paper_ids"
    )
    @classmethod
    def normalize_stage_identifiers(
        cls, values: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        return {
            query_id: tuple(normalize_paper_id(identifier) for identifier in identifiers)
            for query_id, identifiers in values.items()
        }

    @model_validator(mode="after")
    def validate_projection(self) -> SourceProjection:
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("source query IDs must be unique")
        expected_ids = set(self.query_ids)
        for stage_name, values in (
            ("retrieved", self.retrieved_paper_ids),
            ("post-filter", self.post_filter_paper_ids),
            ("selected", self.selected_paper_ids),
        ):
            if set(values) != expected_ids:
                raise ValueError(f"{stage_name} query IDs must match source query IDs")
        expected_metadata = {
            "formal_baseline_2026_08_10": (
                "formal_run",
                "formal_validated",
                "not_applicable",
            ),
            "formal_baseline_2026_08_09": (
                "formal_run",
                "formal_validated",
                "not_applicable",
            ),
            "legacy_title_2026_08_05": (
                "legacy_hash_bound_run",
                "legacy_hash_bound",
                "not_applicable",
            ),
            "query_evolution_prompt_v2": (
                "sealed_probe",
                "probe_verified",
                "matched",
            ),
        }
        if (self.kind, self.verification_status, self.capture_replay_status) != (
            expected_metadata[self.label]
        ):
            raise ValueError("source metadata does not match the fixed source label")
        return self


class PipelineStageCounts(DomainModel):
    total_gold_associations: NonNegativeInt
    not_retrieved: NonNegativeInt
    filtered_out: NonNegativeInt
    ranked_outside_top50: NonNegativeInt
    selected_top50: NonNegativeInt

    @model_validator(mode="after")
    def validate_conservation(self) -> PipelineStageCounts:
        if self.total_gold_associations != (
            self.not_retrieved
            + self.filtered_out
            + self.ranked_outside_top50
            + self.selected_top50
        ):
            raise ValueError("pipeline stages must conserve gold associations")
        return self


class MetricQualityCheck(DomainModel):
    rule_id: NonEmptyStr
    measure: NonEmptyStr
    numerator: Decimal
    denominator: Decimal = Field(ge=0)
    value: Decimal | None
    operator: Literal["eq", "gt", "gte", "lte"]
    threshold: Decimal | int
    passed: bool


class SemanticRescoreRun(DomainModel):
    label: SourceLabel
    kind: SourceKind
    verification_status: SourceVerificationStatus
    capture_replay_status: CaptureReplayStatus
    binding_hashes: dict[NonEmptyStr, Sha256]
    true_positive_count: NonNegativeInt
    macro_f1: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    micro_recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_mrr: float = Field(ge=0, le=1, allow_inf_nan=False)
    macro_ndcg: float = Field(ge=0, le=1, allow_inf_nan=False)
    direct_same_arxiv_hit_count: NonNegativeInt
    pipeline_stages: PipelineStageCounts
    metric_quality_checks: tuple[MetricQualityCheck, ...] = ()

    @model_validator(mode="after")
    def validate_quality_check_scope(self) -> SemanticRescoreRun:
        expected_rule_ids = (
            "hard-filter-recall-loss",
            "macro-recall-positive",
            "micro-recall-positive",
        )
        actual_rule_ids = tuple(check.rule_id for check in self.metric_quality_checks)
        if self.kind == "formal_run" and actual_rule_ids != expected_rule_ids:
            raise ValueError("formal runs require the three metric quality checks")
        if self.kind != "formal_run" and actual_rule_ids:
            raise ValueError("legacy and probe runs cannot have metric quality checks")
        return self


class GenerationHashes(DomainModel):
    public_audit_sha256: Sha256
    gold_sha256: Sha256
    identity_evidence_sha256: Sha256
    snapshot_manifest_sha256: Sha256
    private_audit_sha256: Sha256
    candidate_map_sha256: Sha256


class RescoreDecision(DomainModel):
    designated_source: Literal["formal_baseline_2026_08_10"]
    primary_loss_stage: LossStage | None
    next_direction: NextDirection | None
    reason_codes: tuple[Literal["largest_loss_tie", "source_sensitivity"], ...]

    @model_validator(mode="after")
    def validate_decision(self) -> RescoreDecision:
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("decision reason codes must be unique")
        expected_order = tuple(
            code
            for code in ("largest_loss_tie", "source_sensitivity")
            if code in self.reason_codes
        )
        if self.reason_codes != expected_order:
            raise ValueError("decision reason codes must use fixed order")
        if self.primary_loss_stage is None:
            if self.next_direction is not None or self.reason_codes != ("largest_loss_tie",):
                raise ValueError("tied decisions cannot recommend a direction")
        elif "source_sensitivity" in self.reason_codes:
            if self.next_direction is not None:
                raise ValueError("source-sensitive decisions cannot recommend a direction")
        elif self.next_direction != _direction_for_stage(self.primary_loss_stage):
            raise ValueError("decision direction must match the primary loss stage")
        return self


class SemanticRescoreReport(DomainModel):
    schema_version: Literal["identifier-semantic-rescore-v2"] = (
        "identifier-semantic-rescore-v2"
    )
    scope: Literal["dev"] = "dev"
    status: Literal["passed"] = "passed"
    generation_hashes: GenerationHashes
    quality_policy_sha256: Sha256
    total_gold_associations: NonNegativeInt
    runs: tuple[SemanticRescoreRun, ...] = Field(min_length=4, max_length=4)
    decision: RescoreDecision

    @model_validator(mode="after")
    def validate_fixed_runs(self) -> SemanticRescoreReport:
        if tuple(run.label for run in self.runs) != _FIXED_SOURCE_ORDER:
            raise ValueError("rescore rows must use the fixed order")
        if any(
            run.pipeline_stages.total_gold_associations
            != self.total_gold_associations
            for run in self.runs
        ):
            raise ValueError("rescore rows must have the same total_gold_associations")
        return self


def _direction_for_stage(stage: LossStage) -> NextDirection:
    if stage == "not_retrieved":
        return "retrieval_query"
    if stage == "filtered_out":
        return "retention_filter"
    return "ranking_selector"


def _resolved_stage_sets(
    source: SourceProjection, identifier_map: IdentifierMap
) -> dict[str, tuple[set[str], set[str], set[str]]]:
    resolved: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for query_id in source.query_ids:
        retrieved = {identifier_map.resolve(value) for value in source.retrieved_paper_ids[query_id]}
        post_filter = {
            identifier_map.resolve(value) for value in source.post_filter_paper_ids[query_id]
        }
        selected = {identifier_map.resolve(value) for value in source.selected_paper_ids[query_id]}
        if not selected <= post_filter:
            raise ValueError("selected IDs must be a subset of post-filter IDs after resolution")
        if not post_filter <= retrieved:
            raise ValueError("post-filter IDs must be a subset of retrieved IDs after resolution")
        resolved[query_id] = (retrieved, post_filter, selected)
    return resolved


def _metric_check(
    rule: QualityGateRule, measure: MetricMeasure
) -> MetricQualityCheck:
    passed = measure.value is not None and compare_quality_gate_rule(rule, measure.value)
    return MetricQualityCheck(
        rule_id=rule.rule_id,
        measure=rule.measure,
        numerator=measure.numerator,
        denominator=measure.denominator,
        value=measure.value,
        operator=rule.operator,
        threshold=rule.threshold,
        passed=passed,
    )


def _formal_quality_checks(
    policy: QualityGatePolicy,
    *,
    filtered_out: int,
    total: int,
    metrics: Mapping[str, MetricMeasure],
) -> tuple[MetricQualityCheck, ...]:
    measures: dict[str, MetricMeasure] = dict(metrics)
    measures["hard_filter_absolute_recall_loss"] = MetricMeasure(
        numerator=Decimal(filtered_out),
        denominator=Decimal(total),
        value=Decimal(filtered_out) / Decimal(total) if total else None,
    )
    checks: list[MetricQualityCheck] = []
    for measure_name in _QUALITY_MEASURES:
        applicable = [
            rule
            for rule in policy.rules
            if rule.measure == measure_name
            and rule.classification == "baseline_quality"
            and "dev" in rule.applies_to
        ]
        if len(applicable) != 1:
            raise ValueError(f"expected exactly one applicable policy rule for {measure_name}")
        checks.append(_metric_check(applicable[0], measures[measure_name]))
    return tuple(checks)


def score_source(
    gold: Sequence[EvaluationQuery],
    identifier_map: IdentifierMap,
    source: SourceProjection,
    *,
    policy: QualityGatePolicy,
) -> SemanticRescoreRun:
    """Score one validated source without reading or modifying external state."""
    expected_query_ids = tuple(query.query_id for query in gold)
    if len(expected_query_ids) != len(set(expected_query_ids)):
        raise ValueError("gold query IDs must be unique")
    if source.query_ids != expected_query_ids:
        raise ValueError("source query IDs must exactly match gold query IDs")

    resolved_stages = _resolved_stage_sets(source, identifier_map)
    stage_counts = {
        "not_retrieved": 0,
        "filtered_out": 0,
        "ranked_outside_top50": 0,
        "selected_top50": 0,
    }
    direct_hits = 0
    predictions: list[PredictionRecord] = []
    for query in gold:
        query_id = query.query_id
        retrieved, post_filter, selected = resolved_stages[query_id]
        gold_terminals = {identifier_map.resolve(value) for value in query.relevant_paper_ids}
        for terminal in gold_terminals:
            if terminal in selected:
                stage_counts["selected_top50"] += 1
            elif terminal in post_filter:
                stage_counts["ranked_outside_top50"] += 1
            elif terminal in retrieved:
                stage_counts["filtered_out"] += 1
            else:
                stage_counts["not_retrieved"] += 1
        raw_selected = set(source.selected_paper_ids[query_id])
        direct_anchors = {
            arxiv_anchor(identifier)
            for identifier in query.relevant_paper_ids
            if identifier.startswith("arxiv:")
        }
        direct_hits += len(direct_anchors & raw_selected)
        predictions.append(
            PredictionRecord(
                query_id=query_id,
                predicted_paper_ids=list(source.selected_paper_ids[query_id]),
            )
        )

    metrics = evaluate(gold, predictions, id_map=identifier_map)
    ranking = evaluate_ranking(gold, predictions, id_map=identifier_map)
    pipeline_stages = PipelineStageCounts(
        total_gold_associations=sum(stage_counts.values()), **stage_counts
    )
    quality_checks = (
        _formal_quality_checks(
            policy,
            filtered_out=stage_counts["filtered_out"],
            total=pipeline_stages.total_gold_associations,
            metrics=metrics.measures,
        )
        if source.kind == "formal_run"
        else ()
    )
    return SemanticRescoreRun(
        label=source.label,
        kind=source.kind,
        verification_status=source.verification_status,
        capture_replay_status=source.capture_replay_status,
        binding_hashes=source.binding_hashes,
        true_positive_count=sum(
            measure.true_positive_count for measure in metrics.per_query.values()
        ),
        macro_f1=metrics.summary.macro_f1,
        macro_recall=metrics.summary.macro_recall,
        micro_recall=metrics.summary.micro_recall,
        macro_mrr=ranking.summary.macro_mrr,
        macro_ndcg=ranking.summary.macro_ndcg,
        direct_same_arxiv_hit_count=direct_hits,
        pipeline_stages=pipeline_stages,
        metric_quality_checks=quality_checks,
    )


def _unique_loss_stage(run: SemanticRescoreRun) -> LossStage | None:
    losses: dict[LossStage, int] = {
        "not_retrieved": run.pipeline_stages.not_retrieved,
        "filtered_out": run.pipeline_stages.filtered_out,
        "ranked_outside_top50": run.pipeline_stages.ranked_outside_top50,
    }
    maximum = max(losses.values())
    stages = [stage for stage, value in losses.items() if value == maximum]
    return stages[0] if len(stages) == 1 else None


def decide_bottleneck(runs: Sequence[SemanticRescoreRun]) -> RescoreDecision:
    """Derive the constrained experiment direction from valid rescore rows."""
    designated = next(
        (run for run in runs if run.label == "formal_baseline_2026_08_10"), None
    )
    if designated is None:
        raise ValueError("designated source is required for bottleneck decision")
    primary = _unique_loss_stage(designated)
    if primary is None:
        return RescoreDecision(
            designated_source="formal_baseline_2026_08_10",
            primary_loss_stage=None,
            next_direction=None,
            reason_codes=("largest_loss_tie",),
        )
    source_sensitive = any(
        (stage := _unique_loss_stage(run)) is not None and stage != primary
        for run in runs
        if run.label != designated.label
    )
    return RescoreDecision(
        designated_source="formal_baseline_2026_08_10",
        primary_loss_stage=primary,
        next_direction=None if source_sensitive else _direction_for_stage(primary),
        reason_codes=("source_sensitivity",) if source_sensitive else (),
    )


def build_rescore_report(
    *,
    gold: Sequence[EvaluationQuery],
    identifier_map: IdentifierMap,
    sources: Sequence[SourceProjection],
    policy: QualityGatePolicy,
    generation_hashes: GenerationHashes,
    quality_policy_sha256: str,
) -> SemanticRescoreReport:
    """Build the fixed aggregate report from already-verified in-memory inputs."""
    if tuple(source.label for source in sources) != _FIXED_SOURCE_ORDER:
        raise ValueError("sources must use the fixed order")
    runs = tuple(
        score_source(gold, identifier_map, source, policy=policy) for source in sources
    )
    total_gold_associations = runs[0].pipeline_stages.total_gold_associations
    if any(
        run.pipeline_stages.total_gold_associations != total_gold_associations
        for run in runs
    ):
        raise ValueError("sources must have equal gold association denominators")
    return SemanticRescoreReport(
        generation_hashes=generation_hashes,
        quality_policy_sha256=quality_policy_sha256,
        total_gold_associations=total_gold_associations,
        runs=runs,
        decision=decide_bottleneck(runs),
    )
