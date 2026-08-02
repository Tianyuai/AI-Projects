"""Irrevocable, filesystem-backed live validation-attempt claims."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar
from uuid import uuid4

from pydantic import TypeAdapter, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, SearchMode, Sha256


_SHA_ADAPTER: TypeAdapter[str] = TypeAdapter(Sha256)
ResultT = TypeVar("ResultT")
_SUPERSEDES_PATTERN = re.compile(
    r"^supersedes:(sha256:[0-9a-f]{64}):(.+)$",
    flags=re.DOTALL,
)


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
            if self.completed_at is not None:
                raise ValueError("claimed attempt cannot have terminal metadata")
            if (
                self.incident_ref is not None
                and _SUPERSEDES_PATTERN.fullmatch(self.incident_ref) is None
            ):
                raise ValueError("claimed attempt has malformed supersedes binding")
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

    def _terminal_path(self, validation_lock_sha256: str) -> Path:
        return self._attempts_root / f".{self._digest(validation_lock_sha256)}.terminal"

    def _superseded_path(self, validation_lock_sha256: str) -> Path:
        return self._attempts_root / f".{self._digest(validation_lock_sha256)}.superseded"

    @staticmethod
    def _read_claim_path(
        path: Path,
        validation_lock_sha256: Sha256,
    ) -> ValidationAttemptClaim:
        try:
            payload = path.read_bytes()
            claim = ValidationAttemptClaim.model_validate_json(payload)
        except (OSError, ValueError):
            raise ValueError("malformed validation attempt claim") from None
        if claim.validation_lock_sha256 != validation_lock_sha256:
            raise ValueError("malformed validation attempt claim")
        return claim

    def _existing_claims(self) -> list[ValidationAttemptClaim]:
        claims: list[ValidationAttemptClaim] = []
        for path in self._attempts_root.glob("*.claim"):
            try:
                digest = path.stem
                expected = _SHA_ADAPTER.validate_python(f"sha256:{digest}")
            except ValueError:
                raise ValueError("malformed validation attempt claim path") from None
            claims.append(self.read(expected))
        return claims

    def claim(
        self,
        *,
        validation_lock_sha256: Sha256,
        run_id: str,
        claimed_at: datetime,
        supersedes_validation_lock_sha256: Sha256 | None = None,
        incident_ref: str | None = None,
    ) -> ValidationAttemptClaim:
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._attempts_root.mkdir(parents=True, exist_ok=True)
        if os.stat(self._artifact_root).st_dev != os.stat(self._attempts_root).st_dev:
            raise ValueError("validation attempt paths must use the same filesystem")
        path = self._path(validation_lock_sha256)
        if path.exists():
            raise ValidationAttemptConflictError(
                "validation lock already has an irrevocable attempt"
            )
        existing = self._existing_claims()
        supersedes_binding: str | None = None
        supersession_bytes: bytes | None = None
        supersession_path: Path | None = None
        if existing:
            if supersedes_validation_lock_sha256 is None or not incident_ref:
                raise ValidationAttemptConflictError(
                    "replacement attempt must declare what it supersedes"
                )
            predecessor = max(
                existing,
                key=lambda item: (item.claimed_at, item.validation_lock_sha256),
            )
            if (
                predecessor.validation_lock_sha256
                != supersedes_validation_lock_sha256
                or predecessor.validation_lock_sha256 == validation_lock_sha256
                or predecessor.state not in {"failed", "interrupted"}
                or predecessor.incident_ref != incident_ref
                or predecessor.completed_at is None
                or claimed_at <= predecessor.completed_at
            ):
                raise ValidationAttemptConflictError(
                    "replacement attempt does not bind the superseded incident"
                )
            supersedes_binding = (
                f"supersedes:{supersedes_validation_lock_sha256}:{incident_ref}"
            )
            supersession_bytes = (
                json.dumps(
                    {
                        "incident_ref": incident_ref,
                        "run_id": run_id,
                        "superseded_validation_lock_sha256": (
                            supersedes_validation_lock_sha256
                        ),
                        "validation_lock_sha256": validation_lock_sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            supersession_path = self._superseded_path(
                supersedes_validation_lock_sha256
            )
        elif supersedes_validation_lock_sha256 is not None or incident_ref is not None:
            raise ValidationAttemptConflictError(
                "initial attempt cannot declare a superseded incident"
            )
        claim = ValidationAttemptClaim(
            validation_lock_sha256=validation_lock_sha256,
            run_id=run_id,
            claimed_at=claimed_at,
            state="claimed",
            completed_at=None,
            incident_ref=supersedes_binding,
        )
        if supersession_path is not None and supersession_bytes is not None:
            temporary = supersession_path.with_name(
                f".{supersession_path.name}.{uuid4().hex}.tmp"
            )
            try:
                with temporary.open("xb") as target:
                    target.write(supersession_bytes)
                    target.flush()
                    os.fsync(target.fileno())
                try:
                    os.link(temporary, supersession_path)
                except FileExistsError:
                    try:
                        reserved = supersession_path.read_bytes()
                    except OSError:
                        reserved = b""
                    if reserved != supersession_bytes:
                        raise ValidationAttemptConflictError(
                            "prior incident already has a superseding attempt"
                        ) from None
                _fsync_directory(self._attempts_root)
            finally:
                temporary.unlink(missing_ok=True)
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
        terminal_path = self._terminal_path(validation_lock_sha256)
        if terminal_path.exists():
            terminal = self._read_claim_path(terminal_path, validation_lock_sha256)
            current = self._read_claim_path(path, validation_lock_sha256)
            if current.state == "claimed":
                if (
                    current.run_id != terminal.run_id
                    or current.claimed_at != terminal.claimed_at
                    or terminal.state == "claimed"
                ):
                    raise ValueError("malformed validation attempt transition")
                os.replace(terminal_path, path)
                _fsync_directory(path.parent)
                return terminal
            if current != terminal:
                raise ValueError("malformed validation attempt transition")
            terminal_path.unlink()
            _fsync_directory(path.parent)
            return current
        return self._read_claim_path(path, validation_lock_sha256)

    def transition(
        self,
        *,
        validation_lock_sha256: Sha256,
        target: Literal["complete", "failed", "interrupted"],
        completed_at: datetime,
        incident_ref: str | None = None,
    ) -> ValidationAttemptClaim:
        path = self._path(validation_lock_sha256)
        terminal_path = self._terminal_path(validation_lock_sha256)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
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
            try:
                os.link(temporary, terminal_path)
            except FileExistsError:
                raise ValidationAttemptConflictError(
                    "validation attempt transition is already in progress"
                ) from None
            latest = self._read_claim_path(path, validation_lock_sha256)
            if latest != current:
                terminal_path.unlink(missing_ok=True)
                raise ValidationAttemptConflictError(
                    "validation attempt changed during transition"
                )
            os.replace(terminal_path, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
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

    def transition_created_claim(
        target: Literal["failed", "interrupted"],
        incident_ref: str,
    ) -> None:
        try:
            current = store.read(validation_lock_sha256)
            if current.run_id != run_id or current.state != "claimed":
                return
            store.transition(
                validation_lock_sha256=validation_lock_sha256,
                target=target,
                completed_at=max(datetime.now(UTC), claimed_at),
                incident_ref=incident_ref,
            )
        except (OSError, RuntimeError, ValueError):
            return

    try:
        store.claim(
            validation_lock_sha256=validation_lock_sha256,
            run_id=run_id,
            claimed_at=claimed_at,
        )
        if on_claim is not None:
            on_claim()
        return dispatch()
    except (KeyboardInterrupt, asyncio.CancelledError):
        transition_created_claim(
            "interrupted",
            f"automatic-interruption:{run_id}",
        )
        raise
    except Exception:
        transition_created_claim(
            "failed",
            f"automatic-failure:{run_id}",
        )
        raise


__all__ = [
    "ValidationAttemptClaim",
    "ValidationAttemptConflictError",
    "ValidationAttemptStore",
    "dispatch_with_validation_claim",
]
