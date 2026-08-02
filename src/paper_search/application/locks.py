"""Exact, content-addressed input-lock contracts for replay and live capture."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

import yaml
from pydantic import ConfigDict, Field, PositiveInt, TypeAdapter, model_validator

from paper_search.domain.models import (
    DomainModel,
    NonEmptyStr,
    NonNegativeInt,
    SafeRelativePath,
    Sha256,
)


_EMPTY_RECEIPTS_SHA256 = (
    "sha256:37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"
)


class _LockModel(DomainModel):
    """Strict model base for serialized lock identities only."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactBinding(_LockModel):
    path: SafeRelativePath
    sha256: Sha256


class FrozenDataBinding(_LockModel):
    manifest: ArtifactBinding
    identifier_map: ArtifactBinding
    split: Literal["smoke", "dev", "validation"]
    query_count: PositiveInt
    partition_sha256: Sha256


class CapturePolicyBinding(_LockModel):
    snapshot_schema: Literal["dependency-snapshot-v2"]
    capture_policy_sha256: Sha256


class ProjectLedgerBinding(_LockModel):
    receipt_count: NonNegativeInt = 0
    receipts_sha256: Sha256 = _EMPTY_RECEIPTS_SHA256


class TimeoutBinding(_LockModel):
    connect_seconds: Literal[5]
    read_seconds: Literal[20]
    write_seconds: Literal[20]
    pool_seconds: Literal[5]


class RetryBinding(_LockModel):
    max_attempts: Literal[3]
    retryable_statuses: tuple[Literal[429], Literal["5xx"]]
    retry_timeouts: Literal[True]
    backoff_rule: Literal["min(8,2^retry_index)+jitter[0,1)"]

    @model_validator(mode="before")
    @classmethod
    def normalize_yaml_sequence(cls, value: object) -> object:
        """Accept YAML's sequence representation without coercing scalar values."""

        if isinstance(value, dict) and isinstance(value.get("retryable_statuses"), list):
            return {**value, "retryable_statuses": tuple(value["retryable_statuses"])}
        return value


class PlannerBinding(_LockModel):
    prompt_config: ArtifactBinding
    normal_subqueries_min: Literal[3]
    normal_subqueries_max: Literal[5]
    configured_subqueries_max: Literal[6]
    repair_attempts: Literal[1]
    rules_fallback_enabled: Literal[True]

    @model_validator(mode="after")
    def validate_prompt_config_sha256(self) -> PlannerBinding:
        if self.prompt_config.sha256 == "sha256:" + "0" * 64:
            raise ValueError("prompt config SHA-256 must be nonzero")
        return self


class RetrievalBinding(_LockModel):
    openalex_endpoint: Literal["/works"]
    semantic_scholar_endpoint: Literal["/graph/v1/paper/search"]
    openalex_calls_min: Literal[3]
    openalex_calls_max: Literal[6]
    semantic_scholar_calls_max: Literal[2]
    max_results_per_subquery: Literal[50]
    max_raw_candidates: Literal[300]
    max_deduplicated_candidates: Literal[200]
    max_output_papers: Literal[50]


class BaselineOptionalModules(_LockModel):
    embedding: Literal[False]
    citation_expansion: Literal[False]
    constraint_reranking: Literal[False]
    fixed_two_round: Literal[False]
    adaptive_evolution: Literal[False]


class BaselineBinding(_LockModel):
    primary_model: Literal["qwen3.7-plus"]
    fallback_model: Literal["qwen3.6-flash"]
    prompt_version: Literal["query-analyze-v1"]
    strategy: Literal["fixed-one-round"]
    planner: PlannerBinding
    retrieval: RetrievalBinding
    timeout: TimeoutBinding
    retry: RetryBinding
    optional_modules: BaselineOptionalModules


class _LiveInputLockBase(_LockModel):
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
    project_ledger: ProjectLedgerBinding = ProjectLedgerBinding()
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


class ReplayLock(_LockModel):
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
    project_ledger: ProjectLedgerBinding = ProjectLedgerBinding()
    snapshot_set_id: NonEmptyStr
    snapshot_manifest_sha256: Sha256


InputLock = Annotated[
    CandidateLock | ValidationLock | ReplayLock,
    Field(discriminator="lock_kind"),
]


@dataclass(frozen=True)
class VerifiedInputLock:
    """One parsed lock plus the exact artifact bytes verified with it."""

    lock: CandidateLock | ValidationLock | ReplayLock
    artifact_bytes: Mapping[SafeRelativePath, bytes]

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


def _preflight_artifact_root(artifact_root: Path, relative_path: SafeRelativePath) -> Path:
    """Reject immediately-invalid paths without treating this check as approval."""

    root = artifact_root.resolve(strict=True)
    try:
        candidate = (root / relative_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"lock artifact does not exist: {relative_path}") from error
    if not candidate.is_relative_to(root):
        raise ValueError(f"lock artifact escapes artifact root: {relative_path}")
    return root


def _read_all(file_descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(file_descriptor, 64 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _read_confined_bytes_posix(root: Path, relative_path: SafeRelativePath) -> bytes:
    parts = PurePosixPath(relative_path).parts
    if not parts:
        raise ValueError(f"lock artifact does not exist: {relative_path}")
    directory_flag = getattr(os, "O_DIRECTORY", None)
    nofollow_flag = getattr(os, "O_NOFOLLOW", None)
    if directory_flag is None or nofollow_flag is None:
        raise OSError("safe descriptor traversal is unavailable on this platform")
    directory_flags = os.O_RDONLY | directory_flag | nofollow_flag
    root_descriptor = os.open(root, directory_flags)
    directory_descriptor = root_descriptor
    directory_descriptors = [root_descriptor]
    file_descriptor: int | None = None
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=directory_descriptor)
            directory_descriptor = next_descriptor
            directory_descriptors.append(directory_descriptor)
        file_descriptor = os.open(
            parts[-1], os.O_RDONLY | nofollow_flag, dir_fd=directory_descriptor
        )
        return _read_all(file_descriptor)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)


def _windows_open_handle(path: Path, *, directory: bool) -> int:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    flags = 0x00000080 | (0x02000000 if directory else 0)
    handle = create_file(str(path), 0x80000000, 0x00000001, None, 3, flags, None)
    if handle == wintypes.HANDLE(-1).value:
        raise OSError(ctypes.get_last_error(), f"could not open lock artifact: {path}")
    return int(handle)


def _windows_close_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.CloseHandle(wintypes.HANDLE(handle)):
        raise OSError(ctypes.get_last_error(), "could not close lock artifact handle")


def _windows_final_path(handle: int) -> str:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    final_path = kernel32.GetFinalPathNameByHandleW
    final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    final_path.restype = wintypes.DWORD
    size = 260
    while True:
        buffer = ctypes.create_unicode_buffer(size)
        result = final_path(wintypes.HANDLE(handle), buffer, size, 0)
        if result == 0:
            raise OSError(ctypes.get_last_error(), "could not resolve lock artifact handle")
        if result < size:
            return buffer.value
        size = result + 1


def _windows_path_is_beneath(root: str, target: str) -> bool:
    import ntpath

    normalized_root = ntpath.normcase(ntpath.normpath(root))
    normalized_target = ntpath.normcase(ntpath.normpath(target))
    try:
        return ntpath.commonpath([normalized_root, normalized_target]) == normalized_root
    except ValueError:
        return False


def _read_confined_bytes_windows(root: Path, relative_path: SafeRelativePath) -> bytes:
    import msvcrt

    root_handle = _windows_open_handle(root, directory=True)
    file_handle: int | None = None
    try:
        root_path = _windows_final_path(root_handle)
        file_handle = _windows_open_handle(root / relative_path, directory=False)
        target_path = _windows_final_path(file_handle)
        if not _windows_path_is_beneath(root_path, target_path):
            raise ValueError(f"lock artifact escapes artifact root: {relative_path}")
        file_descriptor = msvcrt.open_osfhandle(file_handle, os.O_RDONLY | os.O_BINARY)
        file_handle = None
        try:
            return _read_all(file_descriptor)
        finally:
            os.close(file_descriptor)
    finally:
        if file_handle is not None:
            _windows_close_handle(file_handle)
        _windows_close_handle(root_handle)


def _read_confined_bytes(artifact_root: Path, relative_path: SafeRelativePath) -> bytes:
    """Read one artifact once after handle-bound confinement verification."""

    root = _preflight_artifact_root(artifact_root, relative_path)
    try:
        if os.name == "nt":
            return _read_confined_bytes_windows(root, relative_path)
        return _read_confined_bytes_posix(root, relative_path)
    except OSError as error:
        raise ValueError(f"could not safely read lock artifact: {relative_path}") from error


def _verify_artifact_bindings(
    lock: InputLock,
    artifact_root: Path,
) -> dict[SafeRelativePath, bytes]:
    bytes_by_path: dict[SafeRelativePath, bytes] = {}
    for binding in _artifact_bindings(lock):
        payload = bytes_by_path.get(binding.path)
        if payload is None:
            payload = _read_confined_bytes(artifact_root, binding.path)
            bytes_by_path[binding.path] = payload
        actual_sha256 = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if actual_sha256 != binding.sha256:
            raise ValueError(f"lock artifact hash mismatch: {binding.path}")
    return bytes_by_path


def load_input_lock(path: Path, *, artifact_root: Path) -> InputLock:
    """Load one exact lock and verify each referenced artifact from one byte read."""

    return load_verified_input_lock(path, artifact_root=artifact_root).lock


def load_verified_input_lock(
    path: Path,
    *,
    artifact_root: Path,
) -> VerifiedInputLock:
    """Read one lock and retain the exact bytes of every verified dependency."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read input lock: {path}") from error
    return load_verified_input_lock_bytes(payload, artifact_root=artifact_root)


def load_input_lock_bytes(payload: bytes, *, artifact_root: Path) -> InputLock:
    """Validate already-read lock bytes and every content-addressed dependency."""

    return load_verified_input_lock_bytes(payload, artifact_root=artifact_root).lock


def load_verified_input_lock_bytes(
    payload: bytes,
    *,
    artifact_root: Path,
) -> VerifiedInputLock:
    """Validate one byte snapshot and retain its verified dependency snapshots."""

    try:
        raw = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise ValueError("invalid lock YAML") from error
    if not isinstance(raw, dict):
        raise ValueError("lock file must contain a mapping")
    lock = _INPUT_LOCK_ADAPTER.validate_python(raw)
    artifact_bytes = _verify_artifact_bindings(lock, artifact_root)
    return VerifiedInputLock(
        lock=lock,
        artifact_bytes=MappingProxyType(artifact_bytes),
    )


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
