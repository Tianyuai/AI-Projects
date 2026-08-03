"""Atomic, minimal smoke-run capture artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Callable, Literal, Protocol, cast

import yaml
from pydantic import TypeAdapter

from paper_search.application.contracts import (
    SearchErrorCode,
    SearchExecutionResult,
    SearchSuccess,
)
from paper_search.application.locks import (
    CandidateLock,
    InputLock,
    ReplayLock,
    ValidationLock,
    lock_sha256,
)
from paper_search.control.ledger import LedgerReport
from paper_search.domain.models import (
    DependencyStatus,
    DomainModel,
    NonEmptyStr,
    NonNegativeInt,
    SearchMode,
    Sha256,
    UsageActual,
)
from paper_search.evaluation.business_results import BusinessResultRecord
from paper_search.evaluation.business_results import business_result_sha256
from paper_search.evaluation.execution_adapter import (
    EvaluationExecutionRecord,
    EvaluationFailureRecord,
)
from paper_search.evaluation.gates import GateEvaluation
from paper_search.evaluation.metrics import EvaluationResult
from paper_search.evaluation.official_adapter import InternalPredictionRecord
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencySnapshotManifestV2,
    DependencySnapshotReader,
)


_RUN_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$")
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_INPUT_LOCK_ADAPTER: TypeAdapter[InputLock] = TypeAdapter(InputLock)


class _AsyncClient(Protocol):
    async def aclose(self) -> None: ...


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )


def _compact_model_bytes(record: DomainModel) -> bytes:
    return (
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _model_json_bytes(record: DomainModel) -> bytes:
    return (
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class RunManifest(DomainModel):
    schema_version: Literal["formal-run-v1"] = "formal-run-v1"
    run_id: NonEmptyStr
    status: Literal["incomplete", "failed", "interrupted", "complete"]
    gate_result: Literal["passed", "failed", "not_applicable"]
    execution_mode: SearchMode
    split: Literal["smoke", "dev", "validation"]
    frozen_manifest_sha256: Sha256
    partition_sha256: Sha256
    identifier_map_sha256: Sha256
    source_git_sha: NonEmptyStr
    tracked_source_dirty: bool
    config_hash: Sha256
    input_lock_sha256: Sha256
    prompt_version: NonEmptyStr
    snapshot_set_id: NonEmptyStr
    snapshot_manifest_sha256: Sha256
    experiment_name: NonEmptyStr
    optional_modules: dict[NonEmptyStr, bool]
    started_at: datetime
    ended_at: datetime | None
    readiness_summary: list[DependencyStatus]
    failure_count: NonNegativeInt
    project_receipt_count: NonNegativeInt = 0
    project_receipts_sha256: Sha256 = (
        "sha256:37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"
    )


class FormalRunWorkspace:
    """Stage one formal run and publish it with a single directory rename."""

    _JSONL_FILES = {
        "prediction": "predictions.jsonl",
        "execution": "executions.jsonl",
        "business_result": "business-results.jsonl",
        "failure": "failures.jsonl",
    }

    def __init__(
        self,
        *,
        runs_root: Path,
        manifest: RunManifest,
        input_lock_bytes: bytes,
        nonce_factory: Callable[[], str],
        clock: Callable[[], datetime],
        validator: Callable[[Path], None] | None = None,
        writer: Callable[[Path, bytes], None] = _atomic_write,
        publisher: Callable[[Path, Path], None] = os.replace,
        replay_snapshot_root: Path | None = None,
    ) -> None:
        if not _is_valid_run_id(manifest.run_id):
            raise ValueError("run_id is invalid")
        if (
            manifest.status != "incomplete"
            or manifest.gate_result != "not_applicable"
            or manifest.ended_at is not None
        ):
            raise ValueError("formal workspace requires an incomplete manifest")
        self._runs_root = runs_root.resolve()
        self._runs_root.mkdir(parents=True, exist_ok=True)
        if os.stat(self._runs_root).st_dev != os.stat(self._runs_root.parent).st_dev:
            raise ValueError("formal workspace paths must use the same filesystem")
        self._manifest = manifest
        self._input_lock_bytes = bytes(input_lock_bytes)
        if _sha256(self._input_lock_bytes) != manifest.input_lock_sha256:
            raise ValueError("input lock sha256 does not match exact input bytes")
        self._input_lock = _parse_lock(self._input_lock_bytes)
        expected_mode: SearchMode = (
            "replay" if isinstance(self._input_lock, ReplayLock) else "live"
        )
        if manifest.execution_mode != expected_mode:
            raise ValueError("formal run execution mode does not match input lock")
        if (
            manifest.split != self._input_lock.frozen_data.split
            or manifest.frozen_manifest_sha256
            != self._input_lock.frozen_data.manifest.sha256
            or manifest.partition_sha256
            != self._input_lock.frozen_data.partition_sha256
            or manifest.identifier_map_sha256
            != self._input_lock.frozen_data.identifier_map.sha256
            or manifest.source_git_sha != self._input_lock.source_git_sha
            or manifest.config_hash != lock_sha256(self._input_lock)
            or manifest.prompt_version != self._input_lock.baseline.prompt_version
        ):
            raise ValueError("formal run manifest does not match input lock bindings")
        self._validator = validator or (lambda path: None)
        self._writer = writer
        self._publisher = publisher
        self._clock = clock
        nonce = nonce_factory()
        if not nonce or not re.fullmatch(r"[A-Za-z0-9._-]+", nonce):
            raise ValueError("workspace nonce is invalid")
        self._work_dir = self._runs_root / f".incomplete-{manifest.run_id}-{nonce}"
        self._complete_dir = self._runs_root / manifest.run_id
        self._failed_dir = self._runs_root / "_failed" / manifest.run_id
        if self._complete_dir.exists() or self._failed_dir.exists():
            raise FileExistsError("formal run destination already exists")
        self._work_dir.mkdir(exist_ok=False)
        self._records: dict[str, list[DomainModel]] = {
            name: [] for name in self._JSONL_FILES
        }
        self._seen: dict[str, set[str]] = {name: set() for name in self._JSONL_FILES}
        self._terminal = False
        self._metrics_written = False
        self._usage_written = False
        self._metrics: EvaluationResult | None = None
        self._snapshot_store: DependencyCaptureStore | None = None
        self._replay_snapshot_root: Path | None = None
        self._replay_snapshot_manifest: DependencySnapshotManifestV2 | None = None
        self._replay_snapshot_bytes: bytes | None = None
        self._write(self._work_dir / "config.lock.yaml", self._input_lock_bytes)
        self._write_manifest()
        for filename in self._JSONL_FILES.values():
            self._write(self._work_dir / filename, b"")
        if manifest.execution_mode == "live":
            if replay_snapshot_root is not None:
                raise ValueError("live workspace cannot accept replay snapshots")
            self._snapshot_store = DependencyCaptureStore(
                self._work_dir / "snapshots",
                clock=clock,
            )
            self._snapshot_store.root.mkdir(exist_ok=False)
        elif replay_snapshot_root is not None:
            if not isinstance(self._input_lock, ReplayLock):
                raise ValueError("replay snapshots require a replay input lock")
            source_root = replay_snapshot_root.resolve(strict=True)
            manifest_path = source_root / "snapshot-manifest.json"
            try:
                snapshot_bytes = manifest_path.read_bytes()
                snapshot_manifest = DependencySnapshotManifestV2.model_validate_json(
                    snapshot_bytes
                )
                reader = DependencySnapshotReader(
                    manifest_path,
                    snapshot_manifest_sha256=self._input_lock.snapshot_manifest_sha256,
                    snapshot_set_id=self._input_lock.snapshot_set_id,
                )
                for entry in snapshot_manifest.entries:
                    reader.read(entry.request)
            except (OSError, ValueError) as error:
                raise ValueError("replay snapshot evidence is invalid") from error
            self._replay_snapshot_manifest = snapshot_manifest
            self._replay_snapshot_bytes = snapshot_bytes
            self._replay_snapshot_root = source_root

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @property
    def snapshot_store(self) -> DependencyCaptureStore:
        if self._snapshot_store is None:
            raise RuntimeError("snapshot capture is available only for live workspaces")
        return self._snapshot_store

    def seal_snapshots(self) -> DependencySnapshotManifestV2:
        self._ensure_active()
        if self._snapshot_store is None:
            raise RuntimeError("snapshot capture is available only for live workspaces")
        manifest = self._snapshot_store.seal()
        manifest_bytes = self._snapshot_store.manifest_path.read_bytes()
        self._manifest = self._manifest.model_copy(
            update={
                "snapshot_set_id": manifest.snapshot_set_id,
                "snapshot_manifest_sha256": _sha256(manifest_bytes),
            }
        )
        self._write_manifest()
        return manifest

    def _ensure_active(self) -> None:
        if self._terminal:
            raise RuntimeError("formal workspace is already terminal")

    def _write(self, path: Path, payload: bytes) -> None:
        self._writer(path, payload)

    def _write_manifest(self) -> None:
        self._write(self._work_dir / "run.json", _model_json_bytes(self._manifest))

    def _append(self, kind: str, record: DomainModel) -> None:
        self._ensure_active()
        query_id = getattr(record, "query_id")
        if query_id in self._seen[kind]:
            raise ValueError(f"duplicate query_id in {self._JSONL_FILES[kind]}: {query_id}")
        records = [*self._records[kind], record]
        payload = b"".join(_compact_model_bytes(item) for item in records)
        self._write(self._work_dir / self._JSONL_FILES[kind], payload)
        self._records[kind] = records
        self._seen[kind].add(query_id)

    def write_prediction(self, record: InternalPredictionRecord) -> None:
        self._append("prediction", record)

    def write_execution(self, record: EvaluationExecutionRecord) -> None:
        if record.run_id != self._manifest.run_id:
            raise ValueError("execution run_id does not match formal workspace")
        self._append("execution", record)

    def write_business_result(self, record: BusinessResultRecord) -> None:
        self._append("business_result", record)

    def write_failure(self, record: EvaluationFailureRecord) -> None:
        if record.run_id != self._manifest.run_id:
            raise ValueError("failure run_id does not match formal workspace")
        self._append("failure", record)
        self._manifest = self._manifest.model_copy(
            update={"failure_count": len(self._records["failure"])}
        )
        self._write_manifest()

    def write_metrics(self, metrics: EvaluationResult) -> None:
        self._ensure_active()
        if self._metrics_written:
            raise RuntimeError("metrics are already written")
        self._write(self._work_dir / "metrics.json", _model_json_bytes(metrics))
        self._metrics_written = True
        self._metrics = metrics

    def write_usage(self, report: LedgerReport) -> None:
        self._ensure_active()
        if self._usage_written:
            raise RuntimeError("usage is already written")
        if report.run_id != self._manifest.run_id:
            raise ValueError("usage run_id does not match formal workspace")
        self._write(self._work_dir / "usage.json", _model_json_bytes(report))
        self._usage_written = True

    def bind_ledger_checkpoint(self, report: LedgerReport) -> None:
        self._ensure_active()
        if report.run_id != self._manifest.run_id:
            raise ValueError("ledger checkpoint run ID does not match formal run")
        self._manifest = self._manifest.model_copy(
            update={
                "project_receipt_count": report.project_receipt_count,
                "project_receipts_sha256": report.project_receipts_sha256,
            }
        )
        self._write_manifest()

    def _publish(self, destination: Path) -> Path:
        if destination.exists():
            raise FileExistsError("formal run destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._publisher(self._work_dir, destination)
        self._terminal = True
        _fsync_directory(destination.parent)
        return destination

    def _validate_ordered_records(self) -> None:
        prediction_records = cast(
            list[InternalPredictionRecord],
            self._records["prediction"],
        )
        prediction_ids = [
            record.query_id for record in prediction_records
        ]
        execution_records = cast(
            list[EvaluationExecutionRecord],
            self._records["execution"],
        )
        execution_ids = [record.query_id for record in execution_records]
        business_records = cast(
            list[BusinessResultRecord],
            self._records["business_result"],
        )
        business_ids = [record.query_id for record in business_records]
        failure_records = cast(
            list[EvaluationFailureRecord],
            self._records["failure"],
        )
        failure_ids = [record.query_id for record in failure_records]
        metrics_ids = list(self._metrics.per_query) if self._metrics is not None else []
        if not (
            prediction_ids
            == execution_ids
            == business_ids
            == metrics_ids
            and len(prediction_ids) == self._input_lock.frozen_data.query_count
        ):
            raise ValueError("formal artifacts have inconsistent ordered query coverage")
        expected_failure_ids = [
            record.query_id
            for record in execution_records
            if isinstance(record, EvaluationExecutionRecord)
            and record.outcome_kind == "failure"
        ]
        if (
            failure_ids != expected_failure_ids
            or self._manifest.failure_count != len(failure_ids)
        ):
            raise ValueError("formal artifacts have inconsistent failure coverage")
        for execution, business in zip(
            execution_records,
            business_records,
            strict=True,
        ):
            if (
                isinstance(execution, EvaluationExecutionRecord)
                and isinstance(business, BusinessResultRecord)
                and execution.business_result_sha256
                != business_result_sha256(business)
            ):
                raise ValueError("formal business result hash does not match execution")

    def _revalidate_replay_source(self) -> None:
        if (
            self._replay_snapshot_root is None
            or self._replay_snapshot_manifest is None
            or self._replay_snapshot_bytes is None
            or not isinstance(self._input_lock, ReplayLock)
        ):
            raise ValueError("replay run requires its exact verified snapshot evidence")
        manifest_path = self._replay_snapshot_root / "snapshot-manifest.json"
        try:
            if manifest_path.read_bytes() != self._replay_snapshot_bytes:
                raise ValueError("replay snapshot manifest changed after verification")
        except OSError as error:
            raise ValueError("replay snapshot manifest is unavailable") from error
        reader = DependencySnapshotReader(
            manifest_path,
            snapshot_manifest_sha256=self._input_lock.snapshot_manifest_sha256,
            snapshot_set_id=self._input_lock.snapshot_set_id,
        )
        for entry in self._replay_snapshot_manifest.entries:
            reader.read(entry.request)

    def finalize(
        self,
        *,
        gate_evaluation: GateEvaluation,
        replay_lock: ReplayLock,
        snapshot_manifest: DependencySnapshotManifestV2,
    ) -> Path:
        self._ensure_active()
        if snapshot_manifest is None:
            raise RuntimeError("sealed snapshot manifest is required")
        if not self._metrics_written or not self._usage_written:
            raise RuntimeError("metrics and usage must be written before finalization")
        if gate_evaluation.split != self._manifest.split:
            raise ValueError("Gate split does not match formal run")
        self._validate_ordered_records()
        if isinstance(self._input_lock, ReplayLock):
            self._revalidate_replay_source()
            if (
                self._replay_snapshot_manifest is None
                or self._replay_snapshot_bytes is None
                or snapshot_manifest != self._replay_snapshot_manifest
            ):
                raise ValueError("replay run requires its exact verified snapshot evidence")
            snapshot_bytes = self._replay_snapshot_bytes
        else:
            if self._snapshot_store is None:
                raise RuntimeError("sealed snapshot manifest is required")
            try:
                sealed_sha256 = self._snapshot_store.manifest_sha256
            except RuntimeError:
                raise RuntimeError("sealed snapshot manifest is required") from None
            snapshot_bytes = self._snapshot_store.manifest_path.read_bytes()
            if _sha256(snapshot_bytes) != sealed_sha256:
                raise ValueError("sealed snapshot manifest changed after sealing")
            try:
                captured_manifest = DependencySnapshotManifestV2.model_validate_json(
                    snapshot_bytes
                )
            except ValueError as error:
                raise ValueError("sealed snapshot manifest is invalid") from error
            if captured_manifest != snapshot_manifest:
                raise ValueError("sealed snapshot manifest does not match capture")
        snapshot_sha256 = _sha256(snapshot_bytes)
        if (
            replay_lock.snapshot_set_id != snapshot_manifest.snapshot_set_id
            or replay_lock.snapshot_set_id != self._manifest.snapshot_set_id
            or replay_lock.snapshot_manifest_sha256
            != self._manifest.snapshot_manifest_sha256
            or snapshot_sha256 != replay_lock.snapshot_manifest_sha256
        ):
            raise ValueError("snapshot manifest sha256 or identity does not match formal run")
        if isinstance(self._input_lock, ReplayLock):
            if replay_lock != self._input_lock:
                raise ValueError("replay run must retain its bound replay lock")
            replay_bytes = self._input_lock_bytes
        else:
            if replay_lock.source_capture_run_id != self._manifest.run_id:
                raise ValueError("capture replay lock does not bind the formal run")
            replay_bytes = _replay_lock_bytes(replay_lock)
        if not isinstance(self._input_lock, ReplayLock):
            reader = DependencySnapshotReader(
                self._work_dir / "snapshots" / "snapshot-manifest.json",
                snapshot_manifest_sha256=replay_lock.snapshot_manifest_sha256,
                snapshot_set_id=replay_lock.snapshot_set_id,
            )
            for entry in snapshot_manifest.entries:
                reader.read(entry.request)
        if self._complete_dir.exists():
            raise FileExistsError("formal run destination already exists")
        self._write(self._work_dir / "replay.lock.yaml", replay_bytes)
        self._write(
            self._work_dir / "snapshot-manifest.json",
            snapshot_bytes,
        )
        self._write(
            self._work_dir / "gates.json",
            _model_json_bytes(gate_evaluation),
        )
        self._manifest = self._manifest.model_copy(
            update={
                "status": "complete",
                "gate_result": gate_evaluation.gate_result,
                "ended_at": self._clock(),
            }
        )
        self._write_manifest()
        self._validator(self._work_dir)
        return self._publish(self._complete_dir)

    def fail(self, reason: SearchErrorCode) -> Path:
        del reason
        self._ensure_active()
        self._manifest = self._manifest.model_copy(
            update={
                "status": "failed",
                "gate_result": "not_applicable",
                "ended_at": self._clock(),
            }
        )
        self._write_manifest()
        return self._publish(self._failed_dir)

    def interrupt(self) -> Path:
        self._ensure_active()
        self._manifest = self._manifest.model_copy(
            update={
                "status": "interrupted",
                "gate_result": "not_applicable",
                "ended_at": self._clock(),
            }
        )
        self._write_manifest()
        return self._publish(self._failed_dir)


def _parse_lock(payload: bytes) -> CandidateLock | ValidationLock | ReplayLock:
    try:
        raw = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise ValueError("invalid input lock YAML") from error
    if not isinstance(raw, dict):
        raise ValueError("input lock must contain a mapping")
    return _INPUT_LOCK_ADAPTER.validate_python(raw)


def _replay_lock_bytes(lock: ReplayLock) -> bytes:
    return yaml.safe_dump(
        lock.model_dump(mode="python"),
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")


def _is_valid_run_id(run_id: str) -> bool:
    if _RUN_ID.fullmatch(run_id) is None:
        return False
    return run_id.split(".", maxsplit=1)[0].upper() not in _WINDOWS_RESERVED


class CaptureSession:
    """Own one incomplete smoke directory until failure or atomic publication."""

    def __init__(
        self,
        *,
        output_root: Path,
        run_id: str,
        input_lock_bytes: bytes,
        on_terminal: Callable[[str, CaptureSession], None] | None = None,
    ) -> None:
        if not _is_valid_run_id(run_id):
            raise ValueError("run_id is invalid")
        self._output_root = output_root.resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id
        self._input_lock_bytes = bytes(input_lock_bytes)
        self._input_lock = _parse_lock(self._input_lock_bytes)
        staging_key = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
        self._work_dir = self._output_root / f".{staging_key}.incomplete"
        self._final_dir = self._output_root / run_id
        self._failed_dir = self._output_root / f"{run_id}.failed"
        if os.name == "nt" and len(str(self._final_dir)) > 180:
            raise ValueError("run_id is invalid for the output path")
        if self._final_dir.exists() or self._failed_dir.exists():
            raise FileExistsError("run artifact already exists")
        self._work_dir.mkdir(exist_ok=False)
        self._snapshot_store = DependencyCaptureStore(self._work_dir)
        self._execution: SearchExecutionResult | None = None
        self._expected_replay_lock: ReplayLock | None = None
        self._sealed_manifest_bytes: bytes | None = None
        self._sealed = False
        self._published = False
        self._claimed = False
        self._on_terminal = on_terminal
        _atomic_write(self._work_dir / "config.lock.yaml", self._input_lock_bytes)
        self._write_run(status="incomplete", error_code=None)

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @property
    def snapshot_store(self) -> DependencyCaptureStore:
        return self._snapshot_store

    @property
    def run_id(self) -> str:
        return self._run_id

    def claim_snapshot_store(self) -> DependencyCaptureStore:
        if self._claimed:
            raise RuntimeError("capture snapshot store is already claimed")
        self._claimed = True
        return self._snapshot_store

    def _run_bytes(
        self,
        *,
        status: str,
        error_code: SearchErrorCode | None,
    ) -> bytes:
        return _json_bytes(
            {
                "business_result_sha256": (
                    self._execution.business_result_sha256
                    if self._execution is not None
                    else None
                ),
                "error_code": error_code,
                "run_id": self._run_id,
                "status": status,
            }
        )

    def _write_run(
        self,
        *,
        status: str,
        error_code: SearchErrorCode | None,
    ) -> None:
        _atomic_write(
            self._work_dir / "run.json",
            self._run_bytes(status=status, error_code=error_code),
        )

    @staticmethod
    def _usage(result: SearchExecutionResult) -> UsageActual:
        return (
            result.outcome.response.usage
            if isinstance(result.outcome, SearchSuccess)
            else result.outcome.usage
        )

    def _execution_bytes(self, result: SearchExecutionResult) -> bytes:
        usage = self._usage(result)
        if isinstance(result.outcome, SearchSuccess):
            response = result.outcome.response
            payload: dict[str, object] = {
                "business_result_sha256": result.business_result_sha256,
                "config_hash": response.config_hash,
                "outcome": result.outcome.kind,
                "query_id": response.query_id,
                "schema_version": "smoke-execution-v1",
                "stop_reason": response.stop_reason,
                "usage": usage.model_dump(mode="json"),
            }
        else:
            payload = {
                "business_result_sha256": result.business_result_sha256,
                "config_hash": None,
                "outcome": result.outcome.kind,
                "query_id": result.outcome.query_id,
                "schema_version": "smoke-execution-v1",
                "stop_reason": result.outcome.stop_reason,
                "usage": usage.model_dump(mode="json"),
            }
        return _json_bytes(payload)

    def record_execution(self, result: SearchExecutionResult) -> None:
        if self._execution is not None:
            raise RuntimeError("execution is already recorded")
        if self._published:
            raise RuntimeError("capture is already published")
        outcome_run_id = (
            result.outcome.response.run_id
            if isinstance(result.outcome, SearchSuccess)
            else result.outcome.run_id
        )
        if outcome_run_id != self._run_id:
            raise ValueError("execution run_id does not match capture session")
        if (
            isinstance(result.outcome, SearchSuccess)
            and result.outcome.response.config_hash != lock_sha256(self._input_lock)
        ):
            raise ValueError("execution config does not match captured input lock")
        self._execution = result
        _atomic_write(
            self._work_dir / "execution.json",
            self._execution_bytes(result),
        )
        usage = self._usage(result)
        _atomic_write(
            self._work_dir / "usage.json",
            _json_bytes(usage.model_dump(mode="json")),
        )
        self._write_run(status="incomplete", error_code=None)

    def seal(self) -> tuple[DependencySnapshotManifestV2, ReplayLock]:
        if self._sealed:
            raise RuntimeError("capture is already sealed")
        if self._execution is None or not isinstance(self._execution.outcome, SearchSuccess):
            raise RuntimeError("successful execution must be recorded before sealing")
        if isinstance(self._input_lock, ReplayLock):
            raise RuntimeError("a replay input lock cannot emit another replay lock")
        if self._snapshot_store.manifest_path.exists():
            raise RuntimeError("capture session must own snapshot sealing")
        manifest = self._snapshot_store.seal()
        replay_lock = ReplayLock(
            schema_version=self._input_lock.schema_version,
            lock_kind="replay",
            created_at=manifest.sealed_at,
            source_capture_run_id=self._run_id,
            source_git_sha=self._input_lock.source_git_sha,
            runtime_allow_live=self._input_lock.runtime_allow_live,
            frozen_data=self._input_lock.frozen_data,
            baseline=self._input_lock.baseline,
            budget_config=self._input_lock.budget_config,
            pricing_policy=self._input_lock.pricing_policy,
            quality_gates=self._input_lock.quality_gates,
            capture_policy=self._input_lock.capture_policy,
            project_ledger=self._input_lock.project_ledger,
            snapshot_set_id=manifest.snapshot_set_id,
            snapshot_manifest_sha256=self._snapshot_store.manifest_sha256,
        )
        replay_bytes = _replay_lock_bytes(replay_lock)
        _atomic_write(self._work_dir / "replay.lock.yaml", replay_bytes)
        self._expected_replay_lock = replay_lock
        self._sealed_manifest_bytes = self._snapshot_store.manifest_path.read_bytes()
        self._sealed = True
        return manifest, replay_lock

    def _validate_canonical_evidence(self, *, status: str) -> None:
        if self._execution is None:
            raise RuntimeError("execution is unavailable")
        if (self._work_dir / "config.lock.yaml").read_bytes() != self._input_lock_bytes:
            raise ValueError("captured input lock bytes changed")
        expected = {
            "execution.json": self._execution_bytes(self._execution),
            "usage.json": _json_bytes(
                self._usage(self._execution).model_dump(mode="json")
            ),
            "run.json": self._run_bytes(status=status, error_code=None),
        }
        for name, content in expected.items():
            try:
                actual = (self._work_dir / name).read_bytes()
            except OSError as error:
                raise ValueError("captured smoke evidence changed") from error
            if actual != content:
                raise ValueError("captured smoke evidence changed")

    def _validate_for_publication(self, *, status: str) -> None:
        self._validate_canonical_evidence(status=status)
        if isinstance(self._input_lock, ReplayLock):
            return

        if self._expected_replay_lock is None or self._sealed_manifest_bytes is None:
            raise RuntimeError("sealed capture evidence is unavailable")
        try:
            replay_bytes = (self._work_dir / "replay.lock.yaml").read_bytes()
        except OSError as error:
            raise ValueError("sealed replay lock is unavailable") from error
        if replay_bytes != _replay_lock_bytes(self._expected_replay_lock):
            raise ValueError("sealed replay lock changed")
        replay_lock = _parse_lock(replay_bytes)
        if not isinstance(replay_lock, ReplayLock):
            raise ValueError("sealed replay lock is invalid")
        manifest_path = self._work_dir / "snapshot-manifest.json"
        try:
            manifest_bytes = manifest_path.read_bytes()
        except (OSError, ValueError) as error:
            raise ValueError("sealed snapshot manifest is invalid") from error
        if manifest_bytes != self._sealed_manifest_bytes:
            raise ValueError("sealed snapshot manifest changed")
        manifest = DependencySnapshotManifestV2.model_validate_json(manifest_bytes)
        reader = DependencySnapshotReader(
            manifest_path,
            snapshot_manifest_sha256=replay_lock.snapshot_manifest_sha256,
            snapshot_set_id=replay_lock.snapshot_set_id,
        )
        for entry in manifest.entries:
            reader.read(entry.request)

    def publish(self) -> Path:
        if self._execution is None or not isinstance(self._execution.outcome, SearchSuccess):
            raise RuntimeError("successful execution must be recorded before publication")
        if not isinstance(self._input_lock, ReplayLock) and not self._sealed:
            raise RuntimeError("live capture must be sealed before publication")
        if self._published:
            raise RuntimeError("capture is already published")
        if self._final_dir.exists():
            raise FileExistsError("final run artifact already exists")
        self._validate_for_publication(status="incomplete")
        self._write_run(status="complete", error_code=None)
        self._validate_for_publication(status="complete")
        os.replace(self._work_dir, self._final_dir)
        self._published = True
        if self._on_terminal is not None:
            self._on_terminal(self._run_id, self)
        return self._final_dir

    def fail(self, code: SearchErrorCode) -> Path:
        if self._published:
            raise RuntimeError("capture is already published")
        if self._failed_dir.exists():
            raise FileExistsError("failed run artifact already exists")
        replay_lock_path = self._work_dir / "replay.lock.yaml"
        if replay_lock_path.exists():
            replay_lock_path.unlink()
        self._write_run(status="failed", error_code=code)
        os.replace(self._work_dir, self._failed_dir)
        self._published = True
        if self._on_terminal is not None:
            self._on_terminal(self._run_id, self)
        return self._failed_dir


@dataclass(frozen=True)
class ArtifactFactory:
    """Create atomic smoke sessions and own request-local live clients."""

    output_root: Path
    _clients: list[_AsyncClient] = field(default_factory=list, repr=False, compare=False)
    _sessions: dict[str, CaptureSession] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _session_lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def _release_session(self, run_id: str, session: CaptureSession) -> None:
        session_key = run_id.casefold()
        with self._session_lock:
            if self._sessions.get(session_key) is session:
                del self._sessions[session_key]

    def start_capture(
        self,
        *,
        run_id: str,
        input_lock_bytes: bytes,
    ) -> CaptureSession:
        session_key = run_id.casefold()
        with self._session_lock:
            if session_key in self._sessions:
                raise ValueError("capture run_id is already active")
        session = CaptureSession(
            output_root=self.output_root,
            run_id=run_id,
            input_lock_bytes=input_lock_bytes,
            on_terminal=self._release_session,
        )
        with self._session_lock:
            if session_key in self._sessions:
                raise ValueError("capture run_id is already active")
            self._sessions[session_key] = session
        return session

    def has_capture_session(self, *, run_id: str) -> bool:
        with self._session_lock:
            session = self._sessions.get(run_id.casefold())
            return session is not None and session.run_id == run_id

    def start_dependency_capture(self, *, run_id: str) -> DependencyCaptureStore:
        with self._session_lock:
            session = self._sessions.get(run_id.casefold())
            if session is not None:
                if session.run_id != run_id:
                    raise RuntimeError("capture session does not match execution run_id")
                return session.claim_snapshot_store()
            if self._sessions:
                raise RuntimeError("capture session does not match execution run_id")
        run_key = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        return DependencyCaptureStore(
            self.output_root / "captures" / run_key / "dependency-snapshot"
        )

    def register_client(self, client: _AsyncClient) -> None:
        self._clients.append(client)

    def release_client(self, client: _AsyncClient) -> None:
        try:
            self._clients.remove(client)
        except ValueError:
            pass

    async def aclose(self) -> None:
        clients = list(self._clients)
        self._clients.clear()
        for client in clients:
            await client.aclose()


__all__ = [
    "ArtifactFactory",
    "CaptureSession",
    "FormalRunWorkspace",
    "RunManifest",
]
