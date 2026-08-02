"""Atomic, minimal smoke-run capture artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Protocol

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
from paper_search.domain.models import UsageActual
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
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )


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
        with self._session_lock:
            if self._sessions.get(run_id) is session:
                del self._sessions[run_id]

    def start_capture(
        self,
        *,
        run_id: str,
        input_lock_bytes: bytes,
    ) -> CaptureSession:
        with self._session_lock:
            if run_id in self._sessions:
                raise ValueError("capture run_id is already active")
        session = CaptureSession(
            output_root=self.output_root,
            run_id=run_id,
            input_lock_bytes=input_lock_bytes,
            on_terminal=self._release_session,
        )
        with self._session_lock:
            if run_id in self._sessions:
                raise ValueError("capture run_id is already active")
            self._sessions[run_id] = session
        return session

    def has_capture_session(self, *, run_id: str) -> bool:
        with self._session_lock:
            return run_id in self._sessions

    def start_dependency_capture(self, *, run_id: str) -> DependencyCaptureStore:
        with self._session_lock:
            session = self._sessions.get(run_id)
            if session is not None:
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


__all__ = ["ArtifactFactory", "CaptureSession"]
