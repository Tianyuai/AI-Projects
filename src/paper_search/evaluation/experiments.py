"""Privacy-safe aggregate experiment records for offline evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_search.storage.experiment import ExperimentRecordStore


class ExperimentAggregate(BaseModel):
    """Safe aggregate metrics and usage totals for one experiment run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_count: int = Field(strict=True, gt=0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    macro_recall: float = Field(ge=0.0, le=1.0)
    search_api_calls: int = Field(strict=True, ge=0)
    llm_calls: int = Field(strict=True, ge=0)
    input_tokens: int = Field(strict=True, ge=0)
    output_tokens: int = Field(strict=True, ge=0)
    cost_cny: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    failure_count: int = Field(strict=True, ge=0)
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _compute_failure_rate(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        payload = dict(data)
        query_count = payload.get("query_count")
        failure_count = payload.get("failure_count")
        if (
            isinstance(query_count, int)
            and not isinstance(query_count, bool)
            and isinstance(failure_count, int)
            and not isinstance(failure_count, bool)
        ):
            if query_count <= 0:
                return payload
            if failure_count > query_count:
                raise ValueError("failure_count cannot exceed query_count")
            payload["failure_rate"] = failure_count / query_count
        return payload


class ExperimentRecord(BaseModel):
    """Immutable metadata-only experiment record safe to persist publicly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    git_sha: str = Field(min_length=1)
    split: str = Field(min_length=1)
    phase: Literal["tuning", "selection_only"]
    annotation_policy: Literal["owner_only_provisional"] = "owner_only_provisional"
    modules: dict[str, bool]
    prompt_versions: dict[str, str]
    model_metadata: dict[str, str]
    aggregate: ExperimentAggregate
    artifact_hashes: dict[str, str]

    @model_validator(mode="after")
    def _validate_split_phase(self) -> "ExperimentRecord":
        if self.split == "validation" and self.phase != "selection_only":
            raise ValueError("validation experiments must use phase 'selection_only'")
        return self


def build_experiment_record(
    *,
    run_id: str,
    config_hash: str,
    git_sha: str,
    split: str,
    phase: Literal["tuning", "selection_only"],
    modules: Mapping[str, bool],
    prompt_versions: Mapping[str, str],
    model_metadata: Mapping[str, str],
    aggregate: ExperimentAggregate,
    artifact_hashes: Mapping[str, str],
) -> ExperimentRecord:
    """Build one safe, immutable experiment record."""

    return ExperimentRecord(
        run_id=run_id,
        config_hash=config_hash,
        git_sha=git_sha,
        split=split,
        phase=phase,
        modules=dict(modules),
        prompt_versions=dict(prompt_versions),
        model_metadata=dict(model_metadata),
        aggregate=aggregate,
        artifact_hashes=dict(artifact_hashes),
    )


def write_experiment_record(store: ExperimentRecordStore, record: ExperimentRecord) -> Path:
    """Write one canonical experiment record through the immutable store."""

    return store.write(record.run_id, record.model_dump(mode="json"))
