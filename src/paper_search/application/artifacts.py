"""Atomic, minimal smoke-run capture artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

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
)
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencySnapshotManifestV2,
    DependencySnapshotReader,
)


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
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


class CaptureSession:
    """Own one incomplete smoke directory until failure or atomic publication."""

    def __init__(
        self,
        *,
        output_root: Path,
        run_id: str,
        input_lock_bytes: bytes,
    ) -> None:
        if not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
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
        if self._final_dir.exists() or self._failed_dir.exists():
            raise FileExistsError("run artifact already exists")
        self._work_dir.mkdir(exist_ok=False)
        self._snapshot_store = DependencyCaptureStore(self._work_dir)
        self._execution: SearchExecutionResult | None = None
        self._sealed = False
        self._published = False
        self._claimed = False
        _atomic_write(self._work_dir / "config.lock.yaml", self._input_lock_bytes)
        self._write_run(status="incomplete", error_code=None)

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @property
    def snapshot_store(self) -> DependencyCaptureStore:
        return self._snapshot_store

    def claim_snapshot_store(self) -> DependencyCaptureStore:
        if self._claimed:
            raise RuntimeError("capture snapshot store is already claimed")
        self._claimed = True
        return self._snapshot_store

    def _write_run(
        self,
        *,
        status: str,
        error_code: SearchErrorCode | None,
    ) -> None:
        _atomic_write(
            self._work_dir / "run.json",
            _json_bytes(
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
            ),
        )

    def record_execution(self, result: SearchExecutionResult) -> None:
        if self._execution is not None:
            raise RuntimeError("execution is already recorded")
        if self._published:
            raise RuntimeError("capture is already published")
        self._execution = result
        _atomic_write(
            self._work_dir / "execution.json",
            _json_bytes(
                {
                    "business_result_sha256": result.business_result_sha256,
                    "outcome": result.outcome.kind,
                }
            ),
        )
        usage = (
            result.outcome.response.usage
            if isinstance(result.outcome, SearchSuccess)
            else result.outcome.usage
        )
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
            manifest = DependencySnapshotManifestV2.model_validate_json(
                self._snapshot_store.manifest_path.read_bytes()
            )
        else:
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
        _atomic_write(self._work_dir / "replay.lock.yaml", _replay_lock_bytes(replay_lock))
        self._sealed = True
        return manifest, replay_lock

    def _validate_for_publication(self) -> None:
        if (self._work_dir / "config.lock.yaml").read_bytes() != self._input_lock_bytes:
            raise ValueError("captured input lock bytes changed")
        for name in ("execution.json", "usage.json", "run.json"):
            try:
                parsed = json.loads((self._work_dir / name).read_bytes())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid smoke artifact: {name}") from error
            if not isinstance(parsed, dict):
                raise ValueError(f"invalid smoke artifact: {name}")
        if isinstance(self._input_lock, ReplayLock):
            return

        try:
            replay_lock = _parse_lock((self._work_dir / "replay.lock.yaml").read_bytes())
        except OSError as error:
            raise ValueError("sealed replay lock is unavailable") from error
        if not isinstance(replay_lock, ReplayLock):
            raise ValueError("sealed replay lock is invalid")
        manifest_path = self._work_dir / "snapshot-manifest.json"
        try:
            manifest = DependencySnapshotManifestV2.model_validate_json(
                manifest_path.read_bytes()
            )
        except (OSError, ValueError) as error:
            raise ValueError("sealed snapshot manifest is invalid") from error
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
        self._validate_for_publication()
        self._write_run(status="complete", error_code=None)
        os.replace(self._work_dir, self._final_dir)
        self._published = True
        return self._final_dir

    def fail(self, code: SearchErrorCode) -> Path:
        if self._published:
            raise RuntimeError("capture is already published")
        if self._failed_dir.exists():
            raise FileExistsError("failed run artifact already exists")
        self._write_run(status="failed", error_code=code)
        os.replace(self._work_dir, self._failed_dir)
        self._published = True
        return self._failed_dir


@dataclass(frozen=True)
class ArtifactFactory:
    """Create atomic smoke sessions and own request-local live clients."""

    output_root: Path
    _clients: list[_AsyncClient] = field(default_factory=list, repr=False, compare=False)
    _sessions: list[CaptureSession] = field(default_factory=list, repr=False, compare=False)

    def start_capture(
        self,
        *,
        run_id: str,
        input_lock_bytes: bytes,
    ) -> CaptureSession:
        session = CaptureSession(
            output_root=self.output_root,
            run_id=run_id,
            input_lock_bytes=input_lock_bytes,
        )
        self._sessions.append(session)
        return session

    def start_dependency_capture(self, *, run_id: str) -> DependencyCaptureStore:
        for session in reversed(self._sessions):
            try:
                return session.claim_snapshot_store()
            except RuntimeError:
                continue
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
