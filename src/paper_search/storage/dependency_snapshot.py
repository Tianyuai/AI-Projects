"""Sealed, content-verified snapshots for external dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import ValidationError, model_validator

from paper_search.application.contracts import SnapshotRef
from paper_search.domain.models import (
    DependencyName,
    DomainModel,
    NonEmptyStr,
    SafeRelativePath,
    Sha256,
    UsageActual,
)
from paper_search.storage.cache import make_cache_key


Clock = Callable[[], datetime]
_SECRET_FIELD_NAME = re.compile(
    r"(?:authorization|api[-_]?key|auth[-_]?token|access[-_]?token|cookie|secret)",
    re.IGNORECASE,
)
_SECRET_HEADER_VALUE = re.compile(
    r"(?:\bbearer\s+|\bsk-[A-Za-z0-9]|\bgh[pousr]_[A-Za-z0-9]"
    r"|\bgithub_pat_[A-Za-z0-9]|\bxox[baprs]-|\bsecret\b"
    r"|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_SAFE_HEADER_NAMES = frozenset(
    {
        "content-type",
        "x-request-id",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-credits-used",
        "x-ratelimit-reset",
    }
)
_SAFE_MODEL_OR_ADAPTER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
_CANONICAL_REQUEST_FIELDS: dict[tuple[str, str], frozenset[str]] = {
    ("llm", "generate_json"): frozenset(
        {
            "payload",
            "prompt_artifact_sha256",
            "prompt_name",
            "prompt_version",
        }
    ),
    ("openalex", "search"): frozenset(
        {
            "cursor",
            "filter",
            "filters",
            "limit",
            "mailto",
            "per_page",
            "query",
            "search",
            "select",
        }
    ),
    ("semantic_scholar", "search"): frozenset(
        {"fields", "filters", "limit", "offset", "query", "venue", "year"}
    ),
    ("semantic_scholar", "batch"): frozenset({"fields", "ids"}),
    ("semantic_scholar", "citations"): frozenset(
        {"fields", "limit", "offset", "paper_id"}
    ),
    ("semantic_scholar", "references"): frozenset(
        {"fields", "limit", "offset", "paper_id"}
    ),
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _reject_secret_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _SECRET_FIELD_NAME.search(str(key)):
                raise ValueError("canonical request contains a secret-shaped field")
            _reject_secret_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_secret_keys(nested)


def _sanitize_error_bytes(value: bytes) -> bytes:
    """Drop provider error bytes that may contain credential-shaped data."""
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        return b""
    if _SECRET_HEADER_VALUE.search(text) or _SECRET_FIELD_NAME.search(text):
        return b""
    return value


class DependencyRequestIdentity(DomainModel):
    schema_version: Literal["dependency-request-v1"] = "dependency-request-v1"
    dependency: DependencyName
    operation: NonEmptyStr
    method: Literal["GET", "POST"]
    endpoint: NonEmptyStr
    model_or_adapter: NonEmptyStr
    canonical_request_sha256: Sha256

    @model_validator(mode="after")
    def validate_safe_binding(self) -> Self:
        if (
            _SAFE_MODEL_OR_ADAPTER.fullmatch(self.model_or_adapter) is None
            or _SECRET_HEADER_VALUE.search(self.model_or_adapter) is not None
        ):
            raise ValueError("model or adapter identifier is not safe")
        valid_binding = False
        if self.dependency == "llm":
            valid_binding = (
                self.operation == "generate_json"
                and self.method == "POST"
                and self.endpoint == "/chat/completions"
            )
        elif self.dependency == "openalex":
            valid_binding = (
                self.operation == "search"
                and self.method == "GET"
                and self.endpoint == "/works"
            )
        elif self.dependency == "semantic_scholar":
            valid_binding = (
                self.operation == "search"
                and self.method == "GET"
                and self.endpoint == "/paper/search"
            ) or (
                self.operation == "batch"
                and self.method == "POST"
                and self.endpoint == "/paper/batch"
            )
            if self.operation in {"citations", "references"}:
                valid_binding = self.method == "GET" and re.fullmatch(
                    rf"/paper/[^/]+/{self.operation}", self.endpoint
                ) is not None
        if not valid_binding:
            raise ValueError("dependency request binding is not allowlisted")
        return self

    @classmethod
    def from_canonical_request(
        cls,
        *,
        dependency: DependencyName,
        operation: str,
        method: Literal["GET", "POST"],
        endpoint: str,
        model_or_adapter: str,
        canonical_request: Mapping[str, object],
    ) -> DependencyRequestIdentity:
        """Build an identity from an explicitly safe, canonical request object."""
        allowed_fields = _CANONICAL_REQUEST_FIELDS.get((dependency, operation))
        if allowed_fields is None:
            raise ValueError("canonical request operation is not allowlisted")
        unknown_fields = set(canonical_request).difference(allowed_fields)
        if unknown_fields:
            raise ValueError(
                f"canonical request field is not allowlisted: {sorted(unknown_fields)}"
            )
        if (dependency, operation) == ("llm", "generate_json"):
            missing_fields = allowed_fields.difference(canonical_request)
            if missing_fields:
                raise ValueError(
                    "canonical LLM request field is missing: "
                    f"{sorted(missing_fields)}"
                )
            prompt_sha256 = canonical_request["prompt_artifact_sha256"]
            if (
                not isinstance(prompt_sha256, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", prompt_sha256) is None
                or prompt_sha256 == "sha256:" + "0" * 64
            ):
                raise ValueError("canonical LLM prompt artifact SHA-256 is invalid")
        _reject_secret_keys(canonical_request)
        return cls(
            dependency=dependency,
            operation=operation,
            method=method,
            endpoint=endpoint,
            model_or_adapter=model_or_adapter,
            canonical_request_sha256=_sha256(_canonical_json_bytes(canonical_request)),
        )


class SnapshotErrorV2(DomainModel):
    code: NonEmptyStr
    message: NonEmptyStr
    retryable: bool


class SnapshotEntryV2(DomainModel):
    entry_id: NonEmptyStr
    request: DependencyRequestIdentity
    cache_key: Sha256
    response_sha256: Sha256
    captured_at: datetime
    response_path: SafeRelativePath
    safe_headers: dict[NonEmptyStr, NonEmptyStr]
    error: SnapshotErrorV2 | None = None
    usage: UsageActual | None = None


class DependencySnapshotManifestV2(DomainModel):
    schema_version: Literal["dependency-snapshot-v2"] = "dependency-snapshot-v2"
    snapshot_set_id: Sha256
    sealed_at: datetime
    entries: list[SnapshotEntryV2]


class SnapshotRead(DomainModel):
    ref: SnapshotRef
    response_bytes: bytes
    error: SnapshotErrorV2 | None = None
    usage: UsageActual | None = None


def _identity_cache_key(identity: DependencyRequestIdentity) -> str:
    return _sha256(_canonical_json_bytes(identity.model_dump(mode="json")))


def _response_path(identity: DependencyRequestIdentity, cache_key: str) -> str:
    return (
        Path("responses")
        / identity.dependency
        / f"{cache_key.removeprefix('sha256:')}.bin"
    ).as_posix()


def _entry_metadata_bytes(entries: list[SnapshotEntryV2]) -> bytes:
    return _canonical_json_bytes(
        [entry.model_dump(mode="json", exclude_none=True) for entry in entries]
    )


def _manifest_bytes(manifest: DependencySnapshotManifestV2) -> bytes:
    return (
        json.dumps(
            manifest.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sanitize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for name, value in headers.items():
        normalized = name.casefold()
        if normalized not in _SAFE_HEADER_NAMES:
            raise ValueError("safe header name is not allowlisted")
        if _SECRET_FIELD_NAME.search(normalized) or _SECRET_HEADER_VALUE.search(value):
            raise ValueError("safe header contains secret-shaped data")
        sanitized[normalized] = value
    return sanitized


def _atomic_new_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite sealed snapshot file: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _snapshot_ref(entry: SnapshotEntryV2) -> SnapshotRef:
    return SnapshotRef(
        entry_id=entry.entry_id,
        dependency=entry.request.dependency,
        cache_key=entry.cache_key,
        response_sha256=entry.response_sha256,
        captured_at=entry.captured_at,
        snapshot_path=entry.response_path,
    )


class DependencyCaptureStore:
    """Stage successful response bytes and seal them into one immutable set."""

    def __init__(self, root: str | Path, *, clock: Clock = _utc_now) -> None:
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "snapshot-manifest.json"
        self._clock = clock
        self._entries: list[SnapshotEntryV2] = []
        self._cache_keys: set[str] = set()
        self._sealed = self.manifest_path.exists()
        self._manifest_sha256: str | None = None

    @property
    def manifest_sha256(self) -> Sha256:
        if self._manifest_sha256 is None:
            raise RuntimeError("snapshot store is not sealed")
        return self._manifest_sha256

    def stage_success(
        self,
        identity: DependencyRequestIdentity,
        *,
        response_bytes: bytes,
        safe_headers: Mapping[str, str],
        captured_at: datetime,
    ) -> SnapshotRef:
        if self._sealed:
            raise RuntimeError("snapshot store is sealed")
        cache_key = _identity_cache_key(identity)
        if cache_key in self._cache_keys:
            raise ValueError("duplicate cache key")

        response_sha256 = _sha256(response_bytes)
        entry_id = _sha256(
            _canonical_json_bytes(
                {"cache_key": cache_key, "response_sha256": response_sha256}
            )
        )
        response_path = _response_path(identity, cache_key)
        entry = SnapshotEntryV2(
            entry_id=entry_id,
            request=identity,
            cache_key=cache_key,
            response_sha256=response_sha256,
            captured_at=captured_at,
            response_path=response_path,
            safe_headers=_sanitize_headers(safe_headers),
        )
        _atomic_new_file(self.root / response_path, response_bytes)
        self._entries.append(entry)
        self._cache_keys.add(cache_key)
        return _snapshot_ref(entry)

    def stage_error(
        self,
        identity: DependencyRequestIdentity,
        *,
        error_code: str,
        message: str,
        retryable: bool,
        response_bytes: bytes,
        safe_headers: Mapping[str, str],
        captured_at: datetime,
    ) -> SnapshotRef:
        if self._sealed:
            raise RuntimeError("snapshot store is sealed")
        cache_key = _identity_cache_key(identity)
        if cache_key in self._cache_keys:
            raise ValueError("duplicate cache key")

        stored_bytes = _sanitize_error_bytes(response_bytes)
        response_sha256 = _sha256(stored_bytes)
        entry_id = _sha256(
            _canonical_json_bytes(
                {
                    "cache_key": cache_key,
                    "error_code": error_code,
                    "response_sha256": response_sha256,
                }
            )
        )
        response_path = _response_path(identity, cache_key)
        entry = SnapshotEntryV2(
            entry_id=entry_id,
            request=identity,
            cache_key=cache_key,
            response_sha256=response_sha256,
            captured_at=captured_at,
            response_path=response_path,
            safe_headers=_sanitize_headers(safe_headers),
            error=SnapshotErrorV2(
                code=error_code,
                message=message,
                retryable=retryable,
            ),
        )
        _atomic_new_file(self.root / response_path, stored_bytes)
        self._entries.append(entry)
        self._cache_keys.add(cache_key)
        return _snapshot_ref(entry)

    def annotate_usage(
        self,
        entry_id: str,
        usage: UsageActual,
    ) -> None:
        """Attach the settled usage of one captured call to its snapshot entry."""
        if self._sealed:
            raise RuntimeError("snapshot store is sealed")
        for index, entry in enumerate(self._entries):
            if entry.entry_id != entry_id:
                continue
            if entry.usage is not None:
                raise ValueError("snapshot usage is already annotated")
            self._entries[index] = entry.model_copy(
                update={"usage": UsageActual.model_validate(usage.model_dump(mode="python"))}
            )
            return
        raise KeyError(f"snapshot entry is unavailable: {entry_id}")

    def seal(self) -> DependencySnapshotManifestV2:
        if self._sealed:
            raise RuntimeError("snapshot store is sealed")
        entries = sorted(
            self._entries,
            key=lambda entry: (
                entry.request.dependency,
                entry.cache_key,
                entry.entry_id,
            ),
        )
        if len({entry.cache_key for entry in entries}) != len(entries):
            raise ValueError("duplicate cache key")
        manifest = DependencySnapshotManifestV2(
            snapshot_set_id=_sha256(_entry_metadata_bytes(entries)),
            sealed_at=self._clock(),
            entries=entries,
        )
        content = _manifest_bytes(manifest)
        _atomic_new_file(self.manifest_path, content)
        self._manifest_sha256 = _sha256(content)
        self._sealed = True
        return manifest


class DependencySnapshotReader:
    """Read an immutable manifest index without owning a live dependency client."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        snapshot_manifest_sha256: Sha256,
        snapshot_set_id: Sha256 | None = None,
        manifest_bytes: bytes | None = None,
    ) -> None:
        self._manifest_path = Path(manifest_path)
        if not self._manifest_path.is_file():
            raise FileNotFoundError("sealed manifest is unavailable")
        content = (
            manifest_bytes
            if manifest_bytes is not None
            else self._manifest_path.read_bytes()
        )
        if _sha256(content) != snapshot_manifest_sha256:
            raise ValueError("snapshot manifest hash does not match lock")
        try:
            manifest = DependencySnapshotManifestV2.model_validate_json(content)
        except ValidationError as error:
            raise ValueError("invalid dependency snapshot manifest") from error
        entries = manifest.entries
        if len({entry.cache_key for entry in entries}) != len(entries):
            raise ValueError("duplicate cache key")
        ordered_entries = sorted(
            entries,
            key=lambda entry: (
                entry.request.dependency,
                entry.cache_key,
                entry.entry_id,
            ),
        )
        if entries != ordered_entries:
            raise ValueError("snapshot entries are not in canonical order")
        for entry in entries:
            if entry.cache_key != _identity_cache_key(entry.request):
                raise ValueError("snapshot cache key does not match request")
            if entry.response_path != _response_path(entry.request, entry.cache_key):
                raise ValueError("snapshot response path is not canonical")
        if manifest.snapshot_set_id != _sha256(_entry_metadata_bytes(entries)):
            raise ValueError("snapshot set identity mismatch")
        if snapshot_set_id is not None and manifest.snapshot_set_id != snapshot_set_id:
            raise ValueError("snapshot set identity does not match lock")
        self._root = self._manifest_path.parent.resolve()
        published_root = self._root / "snapshots"
        if published_root.is_dir():
            self._root = published_root.resolve()
        self._snapshot_set_id = manifest.snapshot_set_id
        self._entries = {entry.cache_key: entry for entry in entries}

    @property
    def snapshot_set_id(self) -> Sha256:
        return self._snapshot_set_id

    def read(self, identity: DependencyRequestIdentity) -> SnapshotRead:
        cache_key = _identity_cache_key(identity)
        entry = self._entries.get(cache_key)
        if entry is None:
            raise KeyError("snapshot unavailable")
        path = self._root / entry.response_path
        resolved = path.resolve()
        if self._root not in resolved.parents:
            raise ValueError("snapshot response path escapes its root")
        relative_parts = Path(entry.response_path).parts
        cursor = self._root
        for part in relative_parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError("snapshot response path contains a symlink")
        try:
            response_bytes = path.read_bytes()
        except OSError as error:
            raise ValueError("snapshot response is unavailable") from error
        if _sha256(response_bytes) != entry.response_sha256:
            raise ValueError("snapshot response hash mismatch")
        return SnapshotRead(
            ref=_snapshot_ref(entry),
            response_bytes=response_bytes,
            error=entry.error,
            usage=entry.usage,
        )


def migrate_v1_to_v2(
    manifest_path: str | Path,
    destination: str | Path,
    *,
    sealed_at: datetime,
) -> DependencySnapshotManifestV2:
    """Migrate only unambiguous Provider V1 entries into a sealed V2 store."""
    source_manifest = Path(manifest_path)
    try:
        payload = json.loads(source_manifest.read_bytes())
        entries = payload["entries"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("invalid V1 snapshot manifest") from error
    if payload.get("contract_version") != "provider-snapshot-v1" or not isinstance(
        entries, list
    ):
        raise ValueError("invalid V1 snapshot manifest")

    validated: list[
        tuple[DependencyRequestIdentity, bytes, Mapping[str, str], datetime]
    ] = []
    for raw_entry in entries:
        try:
            if not isinstance(raw_entry, dict):
                raise TypeError
            provider = raw_entry["provider"]
            if provider not in ("openalex", "semantic_scholar"):
                raise ValueError
            endpoint = raw_entry["endpoint"]
            adapter = raw_entry["cache_version"]
            params = raw_entry["params"]
            if not isinstance(endpoint, str) or not isinstance(params, dict):
                raise ValueError
            operation, method = _infer_v1_operation(provider, endpoint)
            if raw_entry["cache_key"] != make_cache_key(
                provider, endpoint, params, adapter
            ):
                raise ValueError
            response_hash = raw_entry["response_hash"]
            if raw_entry["snapshot_sha256"] != response_hash:
                raise ValueError
            identity = DependencyRequestIdentity.from_canonical_request(
                dependency=provider,
                operation=operation,
                method=method,
                endpoint=endpoint,
                model_or_adapter=adapter,
                canonical_request=params,
            )
            relative = SnapshotRef(
                entry_id="v1-validation",
                dependency=provider,
                cache_key=_sha256(b"v1-validation"),
                response_sha256=response_hash,
                captured_at=raw_entry["requested_at"],
                snapshot_path=raw_entry["snapshot_path"],
            ).snapshot_path
            response_path = source_manifest.parent / relative
            resolved = response_path.resolve()
            source_root = source_manifest.parent.resolve()
            if source_root not in resolved.parents or response_path.is_symlink():
                raise ValueError
            response_bytes = response_path.read_bytes()
            if _sha256(response_bytes) != response_hash:
                raise ValueError
            safe_headers = raw_entry.get("safe_headers", {})
            if not isinstance(safe_headers, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in safe_headers.items()
            ):
                raise ValueError
            captured_at = datetime.fromisoformat(
                str(raw_entry["requested_at"]).replace("Z", "+00:00")
            )
        except (KeyError, OSError, TypeError, ValueError, ValidationError) as error:
            raise ValueError("ambiguous V1 snapshot entry") from error
        validated.append((identity, response_bytes, safe_headers, captured_at))

    store = DependencyCaptureStore(destination, clock=lambda: sealed_at)
    for identity, response_bytes, safe_headers, captured_at in validated:
        store.stage_success(
            identity,
            response_bytes=response_bytes,
            safe_headers=safe_headers,
            captured_at=captured_at,
        )
    return store.seal()


def _infer_v1_operation(
    provider: object, endpoint: str
) -> tuple[str, Literal["GET", "POST"]]:
    if provider == "openalex" and endpoint == "/works":
        return "search", "GET"
    if provider == "semantic_scholar":
        if endpoint == "/paper/search":
            return "search", "GET"
        if endpoint == "/paper/batch":
            return "batch", "POST"
        match = re.fullmatch(r"/paper/[^/]+/(citations|references)", endpoint)
        if match is not None:
            return match.group(1), "GET"
    raise ValueError("ambiguous V1 provider endpoint")


__all__ = [
    "DependencyCaptureStore",
    "DependencyRequestIdentity",
    "DependencySnapshotManifestV2",
    "DependencySnapshotReader",
    "SnapshotErrorV2",
    "SnapshotEntryV2",
    "SnapshotRead",
]
