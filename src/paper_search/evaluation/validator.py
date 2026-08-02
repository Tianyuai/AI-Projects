"""Fail-closed validation and comparison for formal run directories."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from paper_search.application.artifacts import RunManifest
from paper_search.application.locks import (
    CandidateLock,
    InputLock,
    ReplayLock,
    ValidationLock,
    lock_sha256,
)
from paper_search.control.ledger import (
    DEV_RUN_CAP_CNY,
    PROJECT_HARD_CAP_CNY,
    PROJECT_SOFT_STOP_CNY,
    REQUEST_HARD_CAP_CNY,
    VALIDATION_RUN_CAP_CNY,
    LedgerReport,
)
from paper_search.control.pricing import QualityGatePolicy, parse_quality_gate_policy_bytes
from paper_search.domain.models import DomainModel, NonEmptyStr, SafeRelativePath
from paper_search.evaluation.business_results import (
    BusinessResultRecord,
    business_result_sha256,
)
from paper_search.evaluation.execution_adapter import (
    EvaluationExecutionRecord,
    EvaluationFailureRecord,
)
from paper_search.evaluation.freeze_schema import FreezeManifestV2
from paper_search.evaluation.formal_evidence import (
    complete_policy_measures,
    formal_audit_measures,
)
from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap
from paper_search.evaluation.gates import evaluate_gates
from paper_search.evaluation.metrics import EvaluationResult, evaluate
from paper_search.evaluation.official_adapter import (
    InternalPredictionRecord,
    adapt_prediction_record,
)
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
    b"password",
    b"private_key",
    b"access_token",
)
_RUN_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$")
_PRIVATE_PATH = re.compile(
    rb"(?i)(?:[a-z]:(?:\\+|/)(?:users|documents and settings)(?:\\+|/)|/(?:home|users)/[^/\s]+/|https?://[^/\s:@]+:[^/\s@]+@)"
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


def _bound_path(relative: str, *, root: Path) -> Path:
    candidate = root / relative
    current = root
    for component in Path(relative).parts:
        current /= component
        if current.is_symlink():
            raise ValueError("bound artifact path is invalid")
    path = candidate.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("bound artifact path is invalid")
    return path


def _frozen_evidence(
    lock: InputLock,
) -> tuple[list[EvaluationQuery], IdentifierMap, QualityGatePolicy]:
    artifact_root = Path.cwd().resolve()
    manifest_path = _bound_path(lock.frozen_data.manifest.path, root=artifact_root)
    manifest_bytes = manifest_path.read_bytes()
    if f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}" != lock.frozen_data.manifest.sha256:
        raise ValueError("frozen manifest hash mismatch")
    manifest = FreezeManifestV2.model_validate_json(manifest_bytes)
    if manifest.identifier_map.sha256 != lock.frozen_data.identifier_map.sha256:
        raise ValueError("frozen identifier-map binding mismatch")
    partition = next(item for item in manifest.partitions if item.name == lock.frozen_data.split)
    if (
        partition.sha256 != lock.frozen_data.partition_sha256
        or partition.query_count != lock.frozen_data.query_count
    ):
        raise ValueError("frozen partition binding mismatch")
    data_root = manifest_path.parent
    partition_path = _bound_path(partition.path, root=data_root)
    partition_bytes = partition_path.read_bytes()
    if f"sha256:{hashlib.sha256(partition_bytes).hexdigest()}" != partition.sha256:
        raise ValueError("frozen partition hash mismatch")
    queries = [EvaluationQuery.model_validate_json(line) for line in partition_bytes.splitlines() if line]
    if len(queries) != lock.frozen_data.query_count or len({item.query_id for item in queries}) != len(queries):
        raise ValueError("frozen query coverage is invalid")
    identifier_path = _bound_path(lock.frozen_data.identifier_map.path, root=artifact_root)
    identifier_bytes = identifier_path.read_bytes()
    if f"sha256:{hashlib.sha256(identifier_bytes).hexdigest()}" != lock.frozen_data.identifier_map.sha256:
        raise ValueError("identifier map hash mismatch")
    gate_path = _bound_path(lock.quality_gates.path, root=artifact_root)
    gate_bytes = gate_path.read_bytes()
    if f"sha256:{hashlib.sha256(gate_bytes).hexdigest()}" != lock.quality_gates.sha256:
        raise ValueError("Gate policy hash mismatch")
    return queries, IdentifierMap.from_bytes(identifier_bytes), parse_quality_gate_policy_bytes(gate_bytes)


def _replay_inherits_live_lock(
    replay: ReplayLock,
    live: CandidateLock | ValidationLock,
    manifest: RunManifest,
) -> bool:
    return (
        replay.source_capture_run_id == manifest.run_id
        and replay.created_at == manifest.started_at
        and replay.source_git_sha == live.source_git_sha
        and replay.runtime_allow_live == live.runtime_allow_live
        and replay.frozen_data == live.frozen_data
        and replay.baseline == live.baseline
        and replay.budget_config == live.budget_config
        and replay.pricing_policy == live.pricing_policy
        and replay.quality_gates == live.quality_gates
        and replay.capture_policy == live.capture_policy
        and replay.snapshot_set_id == manifest.snapshot_set_id
        and replay.snapshot_manifest_sha256 == manifest.snapshot_manifest_sha256
    )


def _diagnostics_sha256(diagnostics: object) -> str:
    encoded = (
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _sum_usage(records: Sequence[object]) -> dict[str, int | Decimal | None]:
    usages = [getattr(record, "usage") for record in records]
    costs = [usage.cost_cny for usage in usages]
    return {
        "search_api_calls": sum(usage.search_api_calls for usage in usages),
        "llm_calls": sum(usage.llm_calls for usage in usages),
        "input_tokens": sum(usage.input_tokens for usage in usages),
        "output_tokens": sum(usage.output_tokens for usage in usages),
        "elapsed_ms": sum(usage.elapsed_ms for usage in usages),
        "cost_cny": (
            sum((cost for cost in costs if cost is not None), Decimal("0"))
            if all(cost is not None for cost in costs)
            else None
        ),
    }


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
    if path.is_symlink():
        return unavailable, None, None
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
        frozen_queries, identifier_map, gate_policy = _frozen_evidence(lock)

        replay_bytes = (root / "replay.lock.yaml").read_bytes()
        replay_lock = _LOCK_ADAPTER.validate_python(yaml.safe_load(replay_bytes))
        if not isinstance(replay_lock, ReplayLock):
            raise ValueError("replay lock has wrong kind")
        if mode == "replay" and replay_lock != lock:
            raise ValueError("replay lock does not match exact run input lock")
        if mode == "live" and (
            not isinstance(lock, (CandidateLock, ValidationLock))
            or not _replay_inherits_live_lock(replay_lock, lock, manifest)
        ):
            issues.append(
                _issue(
                    "replay_binding_invalid",
                    "replay.lock.yaml",
                    "Replay lock does not inherit the live input lock",
                )
            )
        snapshot_bytes = (root / "snapshot-manifest.json").read_bytes()
        snapshot = DependencySnapshotManifestV2.model_validate_json(snapshot_bytes)
        if _RUN_ID.fullmatch(replay_lock.source_capture_run_id) is None:
            raise ValueError("replay source run ID is invalid")
        snapshot_root = root / "snapshots" if mode == "live" else root.parent / replay_lock.source_capture_run_id / "snapshots"
        snapshot_root = snapshot_root.resolve(strict=True)
        if mode == "replay" and not snapshot_root.is_relative_to(root.parent.resolve()):
            raise ValueError("replay source path escapes artifact root")
        if mode == "replay":
            source_root = snapshot_root.parent
            source_manifest = RunManifest.model_validate_json(
                (source_root / "run.json").read_bytes()
            )
            if (
                source_manifest.status != "complete"
                or source_manifest.execution_mode != "live"
                or source_manifest.run_id != replay_lock.source_capture_run_id
            ):
                issues.append(
                    _issue(
                        "source_capture_invalid",
                        "replay.lock.yaml",
                        "Replay source capture is not a complete live run",
                    )
                )
            else:
                source_result, _, source_mode = _validate(source_root)
                if not source_result.valid or source_mode != "live":
                    issues.append(
                        _issue(
                            "source_capture_invalid",
                            "replay.lock.yaml",
                            "Replay source capture validation failed",
                        )
                    )
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
        if mode == "live":
            expected_snapshot_files = {
                "snapshot-manifest.json",
                *(entry.response_path for entry in snapshot.entries),
            }
            actual_snapshot_files = {
                item.relative_to(snapshot_root).as_posix()
                for item in snapshot_root.rglob("*")
                if item.is_file()
            }
            if actual_snapshot_files != expected_snapshot_files:
                issues.append(
                    _issue(
                        "snapshot_tree_invalid",
                        "snapshots/snapshot-manifest.json",
                        "Snapshot file set is invalid",
                    )
                )

        predictions = _jsonl(root / "predictions.jsonl", InternalPredictionRecord)
        executions = _jsonl(root / "executions.jsonl", EvaluationExecutionRecord)
        business = _jsonl(root / "business-results.jsonl", BusinessResultRecord)
        failures = _jsonl(root / "failures.jsonl", EvaluationFailureRecord)
        metrics = EvaluationResult.model_validate_json((root / "metrics.json").read_bytes())
        usage = LedgerReport.model_validate_json((root / "usage.json").read_bytes())
        query_ids = [getattr(record, "query_id") for record in predictions]
        frozen_ids = [query.query_id for query in frozen_queries]
        if not (
            query_ids
            == [getattr(record, "query_id") for record in executions]
            == [getattr(record, "query_id") for record in business]
            == list(metrics.per_query)
            and query_ids == frozen_ids
        ):
            issues.append(_issue("query_coverage_invalid", "predictions.jsonl", "Ordered query coverage is invalid"))
        expected_failures = [
            record.query_id
            for record in executions
            if isinstance(record, EvaluationExecutionRecord) and record.outcome_kind == "failure"
        ]
        if [getattr(record, "query_id") for record in failures] != expected_failures:
            issues.append(_issue("failure_coverage_invalid", "failures.jsonl", "Hard-failure coverage is invalid"))
        if any(
            isinstance(execution, EvaluationExecutionRecord)
            and execution.run_id != manifest.run_id
            for execution in executions
        ) or any(
            isinstance(failure, EvaluationFailureRecord)
            and failure.run_id != manifest.run_id
            for failure in failures
        ):
            issues.append(_issue("run_binding_invalid", "executions.jsonl", "Record run IDs are invalid"))
        for execution, business_record in zip(executions, business, strict=True):
            if (
                isinstance(execution, EvaluationExecutionRecord)
                and isinstance(business_record, BusinessResultRecord)
                and execution.business_result_sha256 != business_result_sha256(business_record)
            ):
                issues.append(_issue("business_hash_invalid", "business-results.jsonl", "Business projection hash is invalid"))
                break
        official_predictions = [
            adapt_prediction_record(record)
            for record in predictions
            if isinstance(record, InternalPredictionRecord)
        ]
        recomputed_metrics = evaluate(
            frozen_queries,
            official_predictions,
            id_map=identifier_map,
        )
        if metrics != recomputed_metrics:
            issues.append(_issue("metrics_invalid", "metrics.json", "Stored metrics do not match frozen evidence"))
        execution_records = [
            record for record in executions if isinstance(record, EvaluationExecutionRecord)
        ]
        expected_usage = _sum_usage(execution_records)
        run_receipts = [receipt for receipt in usage.receipts if receipt.run_id == manifest.run_id]
        receipt_by_query = {receipt.query_id: receipt for receipt in run_receipts}
        expected_run_cap = (
            DEV_RUN_CAP_CNY if manifest.split == "dev" else VALIDATION_RUN_CAP_CNY
        )
        project_actual = sum(
            (
                receipt.actual.cost_cny
                for receipt in usage.receipts
                if receipt.state in {"settled", "failed"}
                and receipt.actual is not None
                and receipt.actual.cost_cny is not None
            ),
            Decimal("0"),
        )
        project_reserved = sum(
            (
                receipt.estimate.cost_cny or Decimal("0")
                for receipt in usage.receipts
                if receipt.state == "reserved" and receipt.actual is None
            ),
            Decimal("0"),
        )
        costs_known = all(
            receipt.actual is not None and receipt.actual.cost_cny is not None
            for receipt in usage.receipts
            if receipt.state in {"settled", "failed"}
        )
        recomputed_within_caps = (
            costs_known
            and all(
                receipt.actual is None
                or receipt.actual.cost_cny is None
                or receipt.actual.cost_cny <= REQUEST_HARD_CAP_CNY
                for receipt in run_receipts
            )
            and Decimal(expected_usage["cost_cny"] or 0) <= expected_run_cap
            and project_actual + project_reserved <= PROJECT_HARD_CAP_CNY
        )
        ledger_valid = (
            usage.run_id == manifest.run_id
            and usage.reserved == type(usage.reserved)(cost_cny=Decimal("0"))
            and usage.run_cap_cny == expected_run_cap
            and usage.project_soft_stop_cny == PROJECT_SOFT_STOP_CNY
            and usage.project_hard_cap_cny == PROJECT_HARD_CAP_CNY
            and usage.project_actual_cny == project_actual
            and usage.within_caps == recomputed_within_caps
            and usage.within_caps
            and len(run_receipts) == len(usage.receipts)
            and [receipt.query_id for receipt in run_receipts] == query_ids
            and len(receipt_by_query) == len(run_receipts)
            and all(receipt.state in {"settled", "failed"} for receipt in run_receipts)
            and all(
                receipt_by_query[execution.query_id].actual == execution.usage
                for execution in execution_records
                if execution.query_id in receipt_by_query
            )
            and all(
                getattr(usage.actual, field) == value
                for field, value in expected_usage.items()
            )
        )
        if not ledger_valid:
            issues.append(_issue("ledger_invalid", "usage.json", "Budget ledger evidence is invalid"))
        failure_records = [
            record for record in failures if isinstance(record, EvaluationFailureRecord)
        ]
        failure_count = len(failure_records)
        audit = complete_policy_measures(
            formal_audit_measures(
                frozen_queries=frozen_queries,
                executions=execution_records,
                business_results=[
                    record
                    for record in business
                    if isinstance(record, BusinessResultRecord)
                ],
                failures=failure_records,
                ledger_report=usage,
            ),
            policy=gate_policy,
            split=manifest.split,
        )
        recomputed_gate = evaluate_gates(
            frozen_queries=frozen_queries,
            predictions=official_predictions,
            failures=failure_records,
            metrics=recomputed_metrics,
            audit_measures=audit,
            ledger_report=usage,
            policy=gate_policy,
        )
        if recomputed_gate.gate_result != manifest.gate_result:
            issues.append(_issue("gate_invalid", "run.json", "Stored Gate result does not match policy"))
        if manifest.failure_count != failure_count:
            issues.append(_issue("failure_coverage_invalid", "run.json", "Failure count is invalid"))
        failure_by_query = {record.query_id: record for record in failure_records}
        if any(
            failure.diagnostics_sha256
            != _diagnostics_sha256(
                [item.model_dump(mode="json") for item in failure.diagnostics]
            )
            for failure in failure_records
        ):
            issues.append(
                _issue(
                    "diagnostic_hash_invalid",
                    "failures.jsonl",
                    "Diagnostic digest does not match canonical diagnostics",
                )
            )
        for execution in executions:
            if not isinstance(execution, EvaluationExecutionRecord):
                continue
            failure = failure_by_query.get(execution.query_id)
            if execution.outcome_kind == "failure" and (
                failure is None
                or failure.usage != execution.usage
                or failure.stop_reason != execution.stop_reason
                or failure.diagnostics != execution.diagnostics
            ):
                issues.append(_issue("failure_binding_invalid", "failures.jsonl", "Failure record binding is invalid"))
                break
        diagnostics = [
            diagnostic
            for execution in executions
            if isinstance(execution, EvaluationExecutionRecord)
            for diagnostic in execution.diagnostics
        ]
        entry_by_key = {
            (entry.request.dependency, entry.cache_key): entry
            for entry in snapshot.entries
        }
        seen_snapshot_keys: list[tuple[str, str]] = []
        snapshot_refs_valid = True
        for diagnostic in diagnostics:
            for ref in diagnostic.snapshot_refs:
                key = (ref.dependency, ref.cache_key)
                bound_entry = entry_by_key.get(key)
                seen_snapshot_keys.append(key)
                if (
                    bound_entry is None
                    or diagnostic.dependency != ref.dependency
                    or ref.entry_id != bound_entry.entry_id
                    or ref.response_sha256 != bound_entry.response_sha256
                    or ref.captured_at != bound_entry.captured_at
                    or ref.snapshot_path != bound_entry.response_path
                ):
                    snapshot_refs_valid = False
        if (
            not snapshot_refs_valid
            or len(seen_snapshot_keys) != len(set(seen_snapshot_keys))
            or set(seen_snapshot_keys) != set(entry_by_key)
        ):
            issues.append(
                _issue(
                    "snapshot_ref_invalid",
                    "executions.jsonl",
                    "Execution snapshot references do not bind the sealed manifest",
                )
            )
        if any(
            diagnostic.endpoint != "dependency"
            or any(
                error.request_id is not None
                or error.message != "Dependency execution reported an error"
                or error.provider != diagnostic.dependency
                for error in diagnostic.errors
            )
            for diagnostic in diagnostics
        ):
            issues.append(_issue("sanitization_invalid", "executions.jsonl", "Diagnostics are not sanitized"))
        for child in root.rglob("*"):
            if child.is_file() and (
                any(marker in child.read_bytes().lower() for marker in _SECRET_MARKERS)
                or _PRIVATE_PATH.search(child.read_bytes()) is not None
            ):
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
    try:
        capture_manifest = RunManifest.model_validate_json(
            (capture_path / "run.json").read_bytes()
        )
        replay_manifest = RunManifest.model_validate_json(
            (replay_path / "run.json").read_bytes()
        )
        capture_replay_lock = _LOCK_ADAPTER.validate_python(
            yaml.safe_load((capture_path / "replay.lock.yaml").read_bytes())
        )
        replay_input_lock = _LOCK_ADAPTER.validate_python(
            yaml.safe_load((replay_path / "config.lock.yaml").read_bytes())
        )
        linkage_valid = (
            isinstance(capture_replay_lock, ReplayLock)
            and capture_replay_lock == replay_input_lock
            and capture_replay_lock.source_capture_run_id == capture_manifest.run_id
            and capture_manifest.source_git_sha == replay_manifest.source_git_sha
            and capture_manifest.frozen_manifest_sha256
            == replay_manifest.frozen_manifest_sha256
            and capture_manifest.partition_sha256 == replay_manifest.partition_sha256
            and capture_manifest.identifier_map_sha256
            == replay_manifest.identifier_map_sha256
            and capture_manifest.prompt_version == replay_manifest.prompt_version
            and capture_manifest.snapshot_set_id == replay_manifest.snapshot_set_id
            and capture_manifest.snapshot_manifest_sha256
            == replay_manifest.snapshot_manifest_sha256
        )
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        linkage_valid = False
    equivalent = (
        capture_mode == "live"
        and replay_mode == "replay"
        and linkage_valid
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
