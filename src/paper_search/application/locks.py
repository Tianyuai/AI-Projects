"""Exact, content-addressed input-lock contracts for replay and live capture."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, PositiveInt, TypeAdapter, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, SafeRelativePath, Sha256


class ArtifactBinding(DomainModel):
    path: SafeRelativePath
    sha256: Sha256


class FrozenDataBinding(DomainModel):
    manifest: ArtifactBinding
    identifier_map: ArtifactBinding
    split: Literal["smoke", "dev", "validation"]
    query_count: PositiveInt
    partition_sha256: Sha256


class CapturePolicyBinding(DomainModel):
    snapshot_schema: Literal["dependency-snapshot-v2"]
    capture_policy_sha256: Sha256


class TimeoutBinding(DomainModel):
    connect_seconds: Literal[5]
    read_seconds: Literal[20]
    write_seconds: Literal[20]
    pool_seconds: Literal[5]


class RetryBinding(DomainModel):
    max_attempts: Literal[3]
    retryable_statuses: tuple[Literal[429], Literal["5xx"]]
    retry_timeouts: Literal[True]
    backoff_rule: Literal["min(8,2^retry_index)+jitter[0,1)"]


class PlannerBinding(DomainModel):
    prompt_config: ArtifactBinding
    normal_subqueries_min: Literal[3]
    normal_subqueries_max: Literal[5]
    configured_subqueries_max: Literal[6]
    repair_attempts: Literal[1]
    rules_fallback_enabled: Literal[True]


class RetrievalBinding(DomainModel):
    openalex_endpoint: Literal["/works"]
    semantic_scholar_endpoint: Literal["/graph/v1/paper/search"]
    openalex_calls_min: Literal[3]
    openalex_calls_max: Literal[6]
    semantic_scholar_calls_max: Literal[2]
    max_results_per_subquery: Literal[50]
    max_raw_candidates: Literal[300]
    max_deduplicated_candidates: Literal[200]
    max_output_papers: Literal[50]


class BaselineOptionalModules(DomainModel):
    embedding: Literal[False]
    citation_expansion: Literal[False]
    constraint_reranking: Literal[False]
    fixed_two_round: Literal[False]
    adaptive_evolution: Literal[False]


class BaselineBinding(DomainModel):
    primary_model: Literal["qwen3.7-plus"]
    fallback_model: Literal["qwen3.6-flash"]
    prompt_version: Literal["query-analyze-v1"]
    strategy: Literal["fixed-one-round"]
    planner: PlannerBinding
    retrieval: RetrievalBinding
    timeout: TimeoutBinding
    retry: RetryBinding
    optional_modules: BaselineOptionalModules


class _LiveInputLockBase(DomainModel):
    schema_version: Literal["integrated-lock-v1"]
    created_at: datetime
    source_git_sha: NonEmptyStr
    runtime_allow_live: Literal[True]
    frozen_data: FrozenDataBinding
    baseline: BaselineBinding
    budget_config: ArtifactBinding
    pricing_policy: ArtifactBinding
    quality_gates: ArtifactBinding
    capture_policy: CapturePolicyBinding
    approval_ref: NonEmptyStr


class CandidateLock(_LiveInputLockBase):
    lock_kind: Literal["candidate"]

    @model_validator(mode="after")
    def validate_candidate_split(self) -> CandidateLock:
        if self.frozen_data.split != "dev":
            raise ValueError("candidate lock frozen_data.split must be dev")
        return self


class ValidationLock(_LiveInputLockBase):
    lock_kind: Literal["validation"]
    frozen_data: FrozenDataBinding
    promoted_from_dev_run_id: NonEmptyStr
    promoted_from_dev_run_sha256: Sha256

    @model_validator(mode="after")
    def validate_validation_split(self) -> ValidationLock:
        if self.frozen_data.split != "validation":
            raise ValueError("validation lock frozen_data.split must be validation")
        return self


class ReplayLock(DomainModel):
    schema_version: Literal["integrated-lock-v1"]
    lock_kind: Literal["replay"]
    created_at: datetime
    source_capture_run_id: NonEmptyStr
    source_git_sha: NonEmptyStr
    runtime_allow_live: bool
    frozen_data: FrozenDataBinding
    baseline: BaselineBinding
    budget_config: ArtifactBinding
    pricing_policy: ArtifactBinding
    quality_gates: ArtifactBinding
    capture_policy: CapturePolicyBinding
    snapshot_set_id: NonEmptyStr
    snapshot_manifest_sha256: Sha256


InputLock = Annotated[
    CandidateLock | ValidationLock | ReplayLock,
    Field(discriminator="lock_kind"),
]

_INPUT_LOCK_ADAPTER: TypeAdapter[CandidateLock | ValidationLock | ReplayLock] = TypeAdapter(
    InputLock
)


def _artifact_bindings(lock: InputLock) -> tuple[ArtifactBinding, ...]:
    return (
        lock.frozen_data.manifest,
        lock.frozen_data.identifier_map,
        lock.baseline.planner.prompt_config,
        lock.budget_config,
        lock.pricing_policy,
        lock.quality_gates,
    )


def _confined_artifact_path(artifact_root: Path, relative_path: SafeRelativePath) -> Path:
    root = artifact_root.resolve(strict=True)
    try:
        candidate = (root / relative_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"lock artifact does not exist: {relative_path}") from error
    if not candidate.is_relative_to(root):
        raise ValueError(f"lock artifact escapes artifact root: {relative_path}")
    return candidate


def _verify_artifact_bindings(lock: InputLock, artifact_root: Path) -> None:
    bytes_by_path: dict[Path, bytes] = {}
    for binding in _artifact_bindings(lock):
        path = _confined_artifact_path(artifact_root, binding.path)
        payload = bytes_by_path.get(path)
        if payload is None:
            payload = path.read_bytes()
            bytes_by_path[path] = payload
        actual_sha256 = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if actual_sha256 != binding.sha256:
            raise ValueError(f"lock artifact hash mismatch: {binding.path}")


def load_input_lock(path: Path, *, artifact_root: Path) -> InputLock:
    """Load one exact lock and verify each referenced artifact from one byte read."""

    try:
        raw = yaml.safe_load(path.read_bytes())
    except yaml.YAMLError as error:
        raise ValueError(f"invalid lock YAML: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"lock file must contain a mapping: {path}")
    lock = _INPUT_LOCK_ADAPTER.validate_python(raw)
    _verify_artifact_bindings(lock, artifact_root)
    return lock


def canonical_lock_bytes(lock: InputLock) -> bytes:
    """Return sorted, compact JSON bytes for a lock's immutable identity."""

    return json.dumps(
        lock.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def lock_sha256(lock: InputLock) -> Sha256:
    """Return the canonical immutable identity of an input lock."""

    return f"sha256:{hashlib.sha256(canonical_lock_bytes(lock)).hexdigest()}"
