"""Privacy-safe aggregate experiment records for offline evaluation runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_search.storage.experiment import ExperimentRecordStore


_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SECRET_TOKEN = re.compile(r"sk-[A-Za-z0-9_-]{20,}\Z")


def _looks_secret(value: str) -> bool:
    upper_value = value.upper()
    return bool(_SECRET_TOKEN.fullmatch(value)) or "PRIVATE KEY" in upper_value or "PRIVATE_KEY" in upper_value


def _validate_identifier_mapping(values: Mapping[str, str], field_name: str) -> None:
    for key, value in values.items():
        if not _SAFE_IDENTIFIER.fullmatch(key):
            raise ValueError(f"{field_name} keys must be safe identifiers")
        if _looks_secret(value) or not _SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError(f"{field_name} values must be safe identifiers")


def _validate_hash_mapping(values: Mapping[str, str], field_name: str) -> None:
    for key, value in values.items():
        if not _SAFE_IDENTIFIER.fullmatch(key):
            raise ValueError(f"{field_name} keys must be safe identifiers")
        if not _SHA256_HASH.fullmatch(value):
            raise ValueError(f"{field_name} values must be canonical sha256 hashes")


class ExperimentAggregate(BaseModel):
    """Safe aggregate metrics and usage totals for one experiment run."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

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

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

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
        if self.phase == "tuning" and self.split != "dev":
            raise ValueError("tuning experiments must use split 'dev'")
        if self.split == "validation" and self.phase != "selection_only":
            raise ValueError("validation experiments must use phase 'selection_only'")
        _validate_identifier_mapping(self.prompt_versions, "prompt_versions")
        _validate_identifier_mapping(self.model_metadata, "model_metadata")
        _validate_hash_mapping(self.artifact_hashes, "artifact_hashes")
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
