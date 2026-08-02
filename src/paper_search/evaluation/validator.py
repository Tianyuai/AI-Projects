"""Fail-closed validation and comparison for formal run directories."""

from __future__ import annotations

import json
from pathlib import Path
import yaml
from pydantic import TypeAdapter

from paper_search.application.artifacts import RunManifest
from paper_search.application.locks import InputLock, ReplayLock, lock_sha256
from paper_search.control.ledger import LedgerReport
from paper_search.domain.models import DomainModel, NonEmptyStr, SafeRelativePath
from paper_search.evaluation.business_results import (
    BusinessResultRecord,
    business_result_sha256,
)
from paper_search.evaluation.execution_adapter import (
    EvaluationExecutionRecord,
    EvaluationFailureRecord,
)
from paper_search.evaluation.metrics import EvaluationResult
from paper_search.evaluation.official_adapter import InternalPredictionRecord
from paper_search.storage.dependency_snapshot import (
    DependencySnapshotManifestV2,
    DependencySnapshotReader,
)


_LOCK_ADAPTER: TypeAdapter[InputLock] = TypeAdapter(InputLock)
_REQUIRED_FILES = frozenset(
    {
        "business-results.jsonl",
        "config.lock.yaml",
        "executions.jsonl",
        "failures.jsonl",
        "metrics.json",
        "predictions.jsonl",
        "replay.lock.yaml",
        "run.json",
        "snapshot-manifest.json",
        "usage.json",
    }
)
_SECRET_MARKERS = (
    b"authorization",
    b"api_key",
    b"api-key",
    b"credential",
    b"bearer ",
)


class ValidationIssue(DomainModel):
    code: NonEmptyStr
    artifact: SafeRelativePath
    detail: NonEmptyStr


class RunValidationResult(DomainModel):
    valid: bool
    run_id: NonEmptyStr | None
    issues: list[ValidationIssue]


def _issue(code: str, artifact: str, detail: str) -> ValidationIssue:
    return ValidationIssue(code=code, artifact=artifact, detail=detail)


def _jsonl(path: Path, model: type[DomainModel]) -> list[DomainModel]:
    records: list[DomainModel] = []
    for line in path.read_bytes().splitlines():
        if not line:
            raise ValueError("blank JSONL line")
        records.append(model.model_validate_json(line))
    return records


def _validate(path: Path) -> tuple[RunValidationResult, bytes | None, str | None]:
    unavailable = RunValidationResult(
        valid=False,
        run_id=None,
        issues=[
            _issue(
                "run_directory_unavailable",
                "run.json",
                "Required formal run evidence is unavailable",
            )
        ],
    )
    try:
        root = path.resolve(strict=True)
    except OSError:
        return unavailable, None, None
    if not root.is_dir() or root.is_symlink():
        return unavailable, None, None

    issues: list[ValidationIssue] = []
    run_id: str | None = None
    business_bytes: bytes | None = None
    mode: str | None = None
    try:
        children = list(root.iterdir())
        if any(child.is_symlink() for child in children):
            issues.append(_issue("path_escape", "run.json", "Artifact paths must be confined"))
        names = {child.name for child in children}
        allowed = set(_REQUIRED_FILES) | {"snapshots"}
        if not _REQUIRED_FILES <= names or names - allowed:
            issues.append(_issue("file_set_invalid", "run.json", "Formal file set is invalid"))

        run_bytes = (root / "run.json").read_bytes()
        manifest = RunManifest.model_validate_json(run_bytes)
        run_id = manifest.run_id
        mode = manifest.execution_mode
        if manifest.status != "complete":
            issues.append(_issue("run_not_complete", "run.json", "Formal run is not complete"))
        has_snapshots = (root / "snapshots").is_dir()
        if (mode == "live") != has_snapshots:
            issues.append(_issue("mode_tree_invalid", "run.json", "Run tree does not match execution mode"))

        lock_bytes = (root / "config.lock.yaml").read_bytes()
        lock = _LOCK_ADAPTER.validate_python(yaml.safe_load(lock_bytes))
        import hashlib

        exact_lock_sha = f"sha256:{hashlib.sha256(lock_bytes).hexdigest()}"
        if (
            exact_lock_sha != manifest.input_lock_sha256
            or lock_sha256(lock) != manifest.config_hash
            or lock.source_git_sha != manifest.source_git_sha
            or lock.frozen_data.split != manifest.split
            or lock.frozen_data.manifest.sha256 != manifest.frozen_manifest_sha256
            or lock.frozen_data.partition_sha256 != manifest.partition_sha256
            or lock.frozen_data.identifier_map.sha256 != manifest.identifier_map_sha256
            or lock.baseline.prompt_version != manifest.prompt_version
        ):
            issues.append(_issue("lock_binding_invalid", "config.lock.yaml", "Input lock binding is invalid"))

        replay_bytes = (root / "replay.lock.yaml").read_bytes()
        replay_lock = _LOCK_ADAPTER.validate_python(yaml.safe_load(replay_bytes))
        if not isinstance(replay_lock, ReplayLock):
            raise ValueError("replay lock has wrong kind")
        snapshot_bytes = (root / "snapshot-manifest.json").read_bytes()
        snapshot = DependencySnapshotManifestV2.model_validate_json(snapshot_bytes)
        snapshot_root = root / "snapshots" if mode == "live" else root.parent / replay_lock.source_capture_run_id / "snapshots"
        reader_manifest = snapshot_root / "snapshot-manifest.json"
        if reader_manifest.read_bytes() != snapshot_bytes:
            raise ValueError("snapshot manifest exact bytes differ")
        reader = DependencySnapshotReader(
            reader_manifest,
            snapshot_manifest_sha256=replay_lock.snapshot_manifest_sha256,
            snapshot_set_id=replay_lock.snapshot_set_id,
        )
        if (
            replay_lock.snapshot_set_id != manifest.snapshot_set_id
            or replay_lock.snapshot_manifest_sha256 != manifest.snapshot_manifest_sha256
            or snapshot.snapshot_set_id != manifest.snapshot_set_id
        ):
            raise ValueError("snapshot binding mismatch")
        for entry in snapshot.entries:
            reader.read(entry.request)

        predictions = _jsonl(root / "predictions.jsonl", InternalPredictionRecord)
        executions = _jsonl(root / "executions.jsonl", EvaluationExecutionRecord)
        business = _jsonl(root / "business-results.jsonl", BusinessResultRecord)
        failures = _jsonl(root / "failures.jsonl", EvaluationFailureRecord)
        metrics = EvaluationResult.model_validate_json((root / "metrics.json").read_bytes())
        usage = LedgerReport.model_validate_json((root / "usage.json").read_bytes())
        query_ids = [getattr(record, "query_id") for record in predictions]
        if not (
            query_ids
            == [getattr(record, "query_id") for record in executions]
            == [getattr(record, "query_id") for record in business]
            == list(metrics.per_query)
            and len(query_ids) == lock.frozen_data.query_count
        ):
            issues.append(_issue("query_coverage_invalid", "predictions.jsonl", "Ordered query coverage is invalid"))
        expected_failures = [
            record.query_id
            for record in executions
            if isinstance(record, EvaluationExecutionRecord) and record.outcome_kind == "failure"
        ]
        if [getattr(record, "query_id") for record in failures] != expected_failures:
            issues.append(_issue("failure_coverage_invalid", "failures.jsonl", "Hard-failure coverage is invalid"))
        for execution, business_record in zip(executions, business, strict=True):
            if (
                isinstance(execution, EvaluationExecutionRecord)
                and isinstance(business_record, BusinessResultRecord)
                and execution.business_result_sha256 != business_result_sha256(business_record)
            ):
                issues.append(_issue("business_hash_invalid", "business-results.jsonl", "Business projection hash is invalid"))
                break
        if usage.run_id != manifest.run_id or not usage.within_caps:
            issues.append(_issue("ledger_invalid", "usage.json", "Budget ledger is not closed within caps"))
        for child in children:
            if child.is_file() and any(marker in child.read_bytes().lower() for marker in _SECRET_MARKERS):
                issues.append(_issue("sanitization_invalid", child.name, "Artifact contains prohibited private fields"))
                break
        business_bytes = (root / "business-results.jsonl").read_bytes()
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        issues.append(_issue("artifact_invalid", "run.json", "Formal artifact validation failed"))
    return RunValidationResult(valid=not issues, run_id=run_id, issues=issues), business_bytes, mode


def validate_run_directory(path: Path) -> RunValidationResult:
    return _validate(path)[0]


def verify_run_command(path: Path) -> int:
    result = validate_run_directory(path)
    print(json.dumps({"valid": result.valid, "run_id": result.run_id, "issue_codes": [issue.code for issue in result.issues]}, sort_keys=True))
    return 0 if result.valid else 3


def compare_replay_command(capture_path: Path, replay_path: Path) -> int:
    capture, capture_business, capture_mode = _validate(capture_path)
    replay, replay_business, replay_mode = _validate(replay_path)
    if not capture.valid or not replay.valid:
        print(json.dumps({"equivalent": False, "issue_codes": ["artifact_invalid"]}, sort_keys=True))
        return 3
    equivalent = (
        capture_mode == "live"
        and replay_mode == "replay"
        and capture_business == replay_business
    )
    print(json.dumps({"equivalent": equivalent, "capture_run_id": capture.run_id, "replay_run_id": replay.run_id}, sort_keys=True))
    return 0 if equivalent else 4


__all__ = [
    "RunValidationResult",
    "ValidationIssue",
    "compare_replay_command",
    "validate_run_directory",
    "verify_run_command",
]
