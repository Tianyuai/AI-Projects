"""Irrevocable, filesystem-backed live validation-attempt claims."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar
from uuid import uuid4

from pydantic import TypeAdapter, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, SearchMode, Sha256


_SHA_ADAPTER: TypeAdapter[str] = TypeAdapter(Sha256)
ResultT = TypeVar("ResultT")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _claim_bytes(claim: ValidationAttemptClaim) -> bytes:
    return (
        json.dumps(
            claim.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


class ValidationAttemptClaim(DomainModel):
    schema_version: Literal["validation-attempt-v1"] = "validation-attempt-v1"
    validation_lock_sha256: Sha256
    run_id: NonEmptyStr
    claimed_at: datetime
    state: Literal["claimed", "complete", "failed", "interrupted"]
    completed_at: datetime | None
    incident_ref: NonEmptyStr | None

    @model_validator(mode="after")
    def validate_state(self) -> ValidationAttemptClaim:
        if self.claimed_at.tzinfo is None:
            raise ValueError("claimed_at must include a timezone")
        if self.state == "claimed":
            if self.completed_at is not None or self.incident_ref is not None:
                raise ValueError("claimed attempt cannot have terminal metadata")
            return self
        if self.completed_at is None or self.completed_at.tzinfo is None:
            raise ValueError("terminal attempt requires timezone-aware completed_at")
        if self.completed_at < self.claimed_at:
            raise ValueError("completed_at cannot precede claimed_at")
        if self.state == "complete" and self.incident_ref is not None:
            raise ValueError("complete attempt cannot have incident_ref")
        if self.state in {"failed", "interrupted"} and self.incident_ref is None:
            raise ValueError("failed or interrupted attempt requires incident_ref")
        return self


class ValidationAttemptConflictError(RuntimeError):
    code: Literal["validation_attempt_conflict"] = "validation_attempt_conflict"


class ValidationAttemptStore:
    """Consume each validation lock hash at most once, across restarts."""

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root.resolve()
        self._attempts_root = self._artifact_root / "validation-attempts"

    @staticmethod
    def _digest(validation_lock_sha256: str) -> str:
        validated = _SHA_ADAPTER.validate_python(validation_lock_sha256)
        return validated.removeprefix("sha256:")

    def _path(self, validation_lock_sha256: str) -> Path:
        return self._attempts_root / f"{self._digest(validation_lock_sha256)}.claim"

    def claim(
        self,
        *,
        validation_lock_sha256: Sha256,
        run_id: str,
        claimed_at: datetime,
    ) -> ValidationAttemptClaim:
        claim = ValidationAttemptClaim(
            validation_lock_sha256=validation_lock_sha256,
            run_id=run_id,
            claimed_at=claimed_at,
            state="claimed",
            completed_at=None,
            incident_ref=None,
        )
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._attempts_root.mkdir(parents=True, exist_ok=True)
        if os.stat(self._artifact_root).st_dev != os.stat(self._attempts_root).st_dev:
            raise ValueError("validation attempt paths must use the same filesystem")
        path = self._path(validation_lock_sha256)
        try:
            with path.open("xb") as target:
                target.write(_claim_bytes(claim))
                target.flush()
                os.fsync(target.fileno())
        except FileExistsError:
            raise ValidationAttemptConflictError(
                "validation lock already has an irrevocable attempt"
            ) from None
        _fsync_directory(self._attempts_root)
        return claim

    def read(self, validation_lock_sha256: Sha256) -> ValidationAttemptClaim:
        path = self._path(validation_lock_sha256)
        try:
            payload = path.read_bytes()
            claim = ValidationAttemptClaim.model_validate_json(payload)
        except (OSError, ValueError):
            raise ValueError("malformed validation attempt claim") from None
        if claim.validation_lock_sha256 != validation_lock_sha256:
            raise ValueError("malformed validation attempt claim")
        return claim

    def transition(
        self,
        *,
        validation_lock_sha256: Sha256,
        target: Literal["complete", "failed", "interrupted"],
        completed_at: datetime,
        incident_ref: str | None = None,
    ) -> ValidationAttemptClaim:
        path = self._path(validation_lock_sha256)
        transition_lock = path.with_name(f".{path.name}.transition")
        try:
            lock_handle = transition_lock.open("xb")
        except FileExistsError:
            raise ValidationAttemptConflictError(
                "validation attempt transition is already in progress"
            ) from None
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with lock_handle:
                lock_handle.flush()
                os.fsync(lock_handle.fileno())
            current = self.read(validation_lock_sha256)
            if current.state != "claimed":
                raise ValidationAttemptConflictError(
                    "validation attempt is already terminal"
                )
            transitioned = ValidationAttemptClaim(
                validation_lock_sha256=current.validation_lock_sha256,
                run_id=current.run_id,
                claimed_at=current.claimed_at,
                state=target,
                completed_at=completed_at,
                incident_ref=incident_ref,
            )
            with temporary.open("xb") as output:
                output.write(_claim_bytes(transitioned))
                output.flush()
                os.fsync(output.fileno())
            if self.read(validation_lock_sha256) != current:
                raise ValidationAttemptConflictError(
                    "validation attempt changed during transition"
                )
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
            transition_lock.unlink(missing_ok=True)
            _fsync_directory(path.parent)
        return transitioned


def dispatch_with_validation_claim(
    *,
    execution_mode: SearchMode,
    offline_preflight: Callable[[], object],
    reserve_run_budget: Callable[[], object],
    store: ValidationAttemptStore,
    validation_lock_sha256: Sha256,
    run_id: str,
    claimed_at: datetime,
    dispatch: Callable[[], ResultT],
    on_claim: Callable[[], object] | None = None,
) -> ResultT:
    """Order offline checks and budget reserve before the first live dispatch."""
    offline_preflight()
    reserve_run_budget()
    if execution_mode == "replay":
        return dispatch()
    store.claim(
        validation_lock_sha256=validation_lock_sha256,
        run_id=run_id,
        claimed_at=claimed_at,
    )
    if on_claim is not None:
        on_claim()
    try:
        return dispatch()
    except (KeyboardInterrupt, asyncio.CancelledError):
        store.transition(
            validation_lock_sha256=validation_lock_sha256,
            target="interrupted",
            completed_at=datetime.now(UTC),
            incident_ref=f"automatic-interruption:{run_id}",
        )
        raise


__all__ = [
    "ValidationAttemptClaim",
    "ValidationAttemptConflictError",
    "ValidationAttemptStore",
    "dispatch_with_validation_claim",
]
