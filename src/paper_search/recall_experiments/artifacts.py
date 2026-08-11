"""Immutable, canonical artifacts for recall-experiment executions."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml

from paper_search.domain.models import DomainModel
from paper_search.recall_experiments.generation.base import GenerationResult


AttemptStatus = Literal["running", "succeeded", "failed"]
_SENSITIVE_KEY = re.compile(r"authorization|api[_-]?key|token|secret|password", re.IGNORECASE)
_AUTHORIZATION_VALUE = re.compile(r"(?:authorization\s*:\s*|bearer\s+)\S+", re.IGNORECASE)


class ArtifactAttemptMetadata(DomainModel):
    attempt_id: str
    attempt_status: AttemptStatus
    valid_repeat_ordinal: int | None


class RecallArtifactWriter:
    """Publish a new run once; every individual artifact is append-only."""

    def __init__(self, output_root: str | Path) -> None:
        self._output_root = Path(output_root).resolve()
        self._run_path: Path | None = None

    @property
    def run_path(self) -> Path:
        if self._run_path is None:
            raise RuntimeError("a run must be started before writing artifacts")
        return self._run_path

    def start_run(
        self,
        run_id: str,
        *,
        recipe_lock: bytes | str | Mapping[str, object],
        sample_manifest: Mapping[str, object],
    ) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run ID must be a single path component")
        self._output_root.mkdir(parents=True, exist_ok=True)
        run_path = self._output_root / run_id
        if run_path.exists():
            raise FileExistsError(run_path)
        if self._run_path is not None:
            raise RuntimeError("this artifact writer already owns a run")
        run_path.mkdir()
        try:
            self._publish(run_path / "recipe.lock.yaml", _recipe_bytes(recipe_lock))
            self._publish(run_path / "sample-manifest.json", _canonical_json(sample_manifest))
        except Exception:
            # The directory remains as immutable evidence that creation began;
            # it must never be reused for a later execution.
            self._run_path = run_path
            raise
        self._run_path = run_path
        return run_path

    def write_generation(
        self,
        attempt_id: str,
        query_id: str,
        generation: GenerationResult | Mapping[str, object],
        *,
        attempt_status: AttemptStatus,
        valid_repeat_ordinal: int | None,
    ) -> Path:
        if isinstance(generation, GenerationResult):
            if generation.query_id != query_id:
                raise ValueError("generation result query ID does not match artifact query ID")
            payload: Mapping[str, object] = generation.action_batch.model_dump(mode="json")
        else:
            payload = generation
        return self._write_attempt_json(
            "generation", attempt_id, query_id, payload, attempt_status, valid_repeat_ordinal
        )

    def write_retrieval(
        self,
        attempt_id: str,
        query_id: str,
        retrieval: Mapping[str, object],
        *,
        attempt_status: AttemptStatus,
        valid_repeat_ordinal: int | None,
        errors: list[Mapping[str, object]] | None = None,
    ) -> Path:
        payload = dict(retrieval)
        if errors:
            payload["errors"] = _sanitize(errors)
        return self._write_attempt_json(
            "retrieval", attempt_id, query_id, payload, attempt_status, valid_repeat_ordinal
        )

    def write_candidate_pool(
        self,
        attempt_id: str,
        query_id: str,
        pool: Mapping[str, object],
        *,
        attempt_status: AttemptStatus,
        valid_repeat_ordinal: int | None,
    ) -> Path:
        return self._write_attempt_json(
            "candidate-pools", attempt_id, query_id, pool, attempt_status, valid_repeat_ordinal
        )

    def write_report(self, report: Mapping[str, object]) -> Path:
        path = self.run_path / "recall-report.json"
        self._publish(path, _canonical_json(_sanitize(report)))
        return path

    def _write_attempt_json(
        self,
        category: str,
        attempt_id: str,
        query_id: str,
        payload: Mapping[str, object],
        attempt_status: AttemptStatus,
        valid_repeat_ordinal: int | None,
    ) -> Path:
        if not attempt_id or not query_id:
            raise ValueError("attempt and query IDs must not be empty")
        metadata = ArtifactAttemptMetadata(
            attempt_id=attempt_id,
            attempt_status=attempt_status,
            valid_repeat_ordinal=valid_repeat_ordinal,
        ).model_dump(mode="json")
        document = {**payload, "query_id": query_id, **metadata}
        path = self.run_path / category / attempt_id / f"{query_id}.json"
        self._publish(path, _canonical_json(_sanitize(document)))
        return path

    @staticmethod
    def _publish(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise
            finally:
                temporary.unlink(missing_ok=True)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def _recipe_bytes(value: bytes | str | Mapping[str, object]) -> bytes:
    if isinstance(value, bytes):
        value.decode("utf-8")
        return value if value.endswith(b"\n") else value + b"\n"
    if isinstance(value, str):
        value.encode("utf-8").decode("utf-8")
        return value.encode("utf-8") if value.endswith("\n") else (value + "\n").encode("utf-8")
    return yaml.safe_dump(dict(value), allow_unicode=True, sort_keys=True).encode("utf-8")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _sanitize(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, str) and _AUTHORIZATION_VALUE.search(value):
        return "[REDACTED]"
    return value


__all__ = ["ArtifactAttemptMetadata", "AttemptStatus", "RecallArtifactWriter"]
