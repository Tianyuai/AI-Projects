"""Stable public result and comparison contracts for recall canaries."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, Sha256, UsageActual
from paper_search.recall_experiments.identity import LiveRuntimeIdentity
from paper_search.recall_experiments.contracts import RecallSearchAction


EvaluationStatus = Literal["available", "not_available"]


class CanaryPerQueryResult(DomainModel):
    query_id: NonEmptyStr
    candidate_pool_ids: list[NonEmptyStr]
    candidate_count: int = Field(ge=0)
    evaluation_status: EvaluationStatus
    gold_hit_ids: list[NonEmptyStr]
    gold_association_count: int | None = Field(default=None, ge=1)
    gold_hit_count: int | None = Field(default=None, ge=0)
    candidate_recall: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_evaluation_shape(self) -> CanaryPerQueryResult:
        metrics = (self.gold_association_count, self.gold_hit_count, self.candidate_recall)
        if self.evaluation_status == "available" and any(item is None for item in metrics):
            raise ValueError("available evaluation requires all Gold metrics")
        if self.evaluation_status == "not_available" and (
            any(item is not None for item in metrics) or self.gold_hit_ids
        ):
            raise ValueError("unavailable evaluation forbids Gold values")
        if self.candidate_count != len(self.candidate_pool_ids):
            raise ValueError("candidate_count must match candidate_pool_ids")
        return self


class CanaryRecallResult(DomainModel):
    candidate_pool_policy_version: NonEmptyStr
    evaluation_status: EvaluationStatus
    per_query: list[CanaryPerQueryResult]
    gold_association_count: int | None = Field(default=None, ge=1)
    gold_hit_count: int | None = Field(default=None, ge=0)
    macro_candidate_recall: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_evaluation_shape(self) -> CanaryRecallResult:
        metrics = (
            self.gold_association_count,
            self.gold_hit_count,
            self.macro_candidate_recall,
        )
        if self.evaluation_status == "available" and any(item is None for item in metrics):
            raise ValueError("available evaluation requires all aggregate metrics")
        if self.evaluation_status == "not_available" and any(
            item is not None for item in metrics
        ):
            raise ValueError("unavailable evaluation forbids aggregate Gold values")
        if any(row.evaluation_status != self.evaluation_status for row in self.per_query):
            raise ValueError("all query evaluation states must match the aggregate")
        return self


class CanaryPerQueryComparison(DomainModel):
    query_id: NonEmptyStr
    current_candidate_count: int = Field(ge=0)
    baseline_candidate_count: int = Field(ge=0)
    intersection_count: int = Field(ge=0)
    jaccard: float = Field(ge=0, le=1)
    added_gold_hit_ids: list[NonEmptyStr]
    lost_gold_hit_ids: list[NonEmptyStr]


class CanaryComparison(DomainModel):
    evidence_level: Literal["strict", "exploratory"]
    per_query: list[CanaryPerQueryComparison]


class CanaryExecutionIdentity(DomainModel):
    identity_schema_version: Literal["recall-canary-execution-identity-v1"]
    method_id: NonEmptyStr
    recipe_sha256: Sha256
    input_sha256: Sha256
    identifier_map_sha256: Sha256 | None
    generator_type: Literal[
        "manual_actions",
        "fixed_actions",
        "deepseek_prompt",
        "local_cpu",
        "local_cpu_fallback",
    ]
    generator_model: NonEmptyStr | None
    prompt_sha256: Sha256 | None
    actions_sha256: Sha256 | None
    allowed_actions: tuple[Literal["text_search", "title_search", "citation_expand"], ...]
    max_total_actions: int = Field(strict=True, gt=0)
    max_results_per_action: int = Field(strict=True, gt=0)
    candidate_pool_policy_version: Literal[
        "production-dedup-v1", "canonical-id-first-v1"
    ]
    runtime: LiveRuntimeIdentity
    snapshot_manifest_sha256: Sha256
    snapshot_set_id: Sha256

    @model_validator(mode="after")
    def validate_generator_material(self) -> CanaryExecutionIdentity:
        if not self.allowed_actions or len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("canary identity requires unique allowed actions")
        if self.generator_type == "deepseek_prompt":
            if self.generator_model is None or self.prompt_sha256 is None or self.actions_sha256:
                raise ValueError("DeepSeek canary identity requires model/prompt and no action source")
            if self.generator_model != self.runtime.dependencies["llm"].model:
                raise ValueError("canary generator and runtime models do not match")
        elif self.generator_type == "local_cpu":
            if self.generator_model is None or self.prompt_sha256 is not None or not self.actions_sha256:
                raise ValueError("local CPU canary identity requires model and weight source")
        elif self.generator_type == "local_cpu_fallback":
            if self.generator_model is None or self.prompt_sha256 is None or not self.actions_sha256:
                raise ValueError("hybrid CPU canary identity requires model, prompt, and weights")
        elif self.generator_model is not None or self.prompt_sha256 is not None or not self.actions_sha256:
            raise ValueError("fixed/manual canary identity requires only an action source")
        return self


class CanaryInputIdentity(DomainModel):
    input_kind: Literal["single", "jsonl", "frozen"]
    input_sha256: Sha256
    evaluation_status: EvaluationStatus
    query_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def validate_query_ids(self) -> CanaryInputIdentity:
        if not self.query_ids or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("canary input identity requires unique query IDs")
        return self


class CanaryReport(DomainModel):
    schema_version: Literal["recall-canary-report-v1"] = "recall-canary-report-v1"
    run_id: NonEmptyStr
    input: CanaryInputIdentity
    execution_identity: CanaryExecutionIdentity
    actions_by_query: dict[NonEmptyStr, list[RecallSearchAction]]
    usage: UsageActual
    result: CanaryRecallResult
    comparison: CanaryComparison | None = None


def compare_canary_results(
    current: CanaryRecallResult,
    baseline: CanaryRecallResult,
    *,
    identities_match: bool,
) -> CanaryComparison:
    current_by_id = {row.query_id: row for row in current.per_query}
    baseline_by_id = {row.query_id: row for row in baseline.per_query}
    if set(current_by_id) != set(baseline_by_id):
        raise ValueError("canary comparison query IDs do not match")
    rows: list[CanaryPerQueryComparison] = []
    for query_id in current_by_id:
        current_row = current_by_id[query_id]
        baseline_row = baseline_by_id[query_id]
        current_ids = set(current_row.candidate_pool_ids)
        baseline_ids = set(baseline_row.candidate_pool_ids)
        union = current_ids.union(baseline_ids)
        current_hits = set(current_row.gold_hit_ids)
        baseline_hits = set(baseline_row.gold_hit_ids)
        rows.append(
            CanaryPerQueryComparison(
                query_id=query_id,
                current_candidate_count=len(current_ids),
                baseline_candidate_count=len(baseline_ids),
                intersection_count=len(current_ids.intersection(baseline_ids)),
                jaccard=len(current_ids.intersection(baseline_ids)) / len(union) if union else 1.0,
                added_gold_hit_ids=sorted(current_hits.difference(baseline_hits)),
                lost_gold_hit_ids=sorted(baseline_hits.difference(current_hits)),
            )
        )
    return CanaryComparison(
        evidence_level="strict" if identities_match else "exploratory",
        per_query=rows,
    )


__all__ = [
    "CanaryComparison",
    "CanaryExecutionIdentity",
    "CanaryInputIdentity",
    "CanaryReport",
    "CanaryPerQueryComparison",
    "CanaryPerQueryResult",
    "CanaryRecallResult",
    "compare_canary_results",
]
