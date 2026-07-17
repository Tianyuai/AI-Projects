"""SQLite-backed raw provider response cache."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


Clock = Callable[[], datetime]
SAFE_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "x-request-id",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-credits-used",
        "x-ratelimit-reset",
    }
)


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


def _frozen_write(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"refusing to overwrite frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def canonical_request_params(params: Mapping[str, object]) -> dict[str, str]:
    """Return sorted string parameters with credentials removed."""
    return {
        key: str(value)
        for key, value in sorted(params.items())
        if key.casefold() != "api_key"
    }


def make_cache_key(
    provider: str,
    endpoint: str,
    params: Mapping[str, object],
    cache_version: str,
) -> str:
    """Hash the stable non-secret request identity."""
    payload = {
        "cache_version": cache_version,
        "endpoint": endpoint,
        "params": canonical_request_params(params),
        "provider": provider,
    }
    return _sha256(_canonical_json_bytes(payload))


@dataclass(frozen=True)
class CachedResponse:
    cache_key: str
    provider: str
    endpoint: str
    cache_version: str
    params: dict[str, str]
    status_code: int
    raw_response: bytes
    response_hash: str
    safe_headers: dict[str, str]
    requested_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class PreparedSnapshot:
    """Validated cached rows and exact bytes ready for one formal snapshot write."""

    responses: tuple[CachedResponse, ...]
    files: tuple[tuple[Path, bytes], ...]
    manifest_content: bytes


class SQLiteResponseCache:
    """Store replayable successful responses and short provider cooldowns."""

    def __init__(self, path: str | Path, *, clock: Clock = _utc_now) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS responses (
                        cache_key TEXT PRIMARY KEY,
                        provider TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        cache_version TEXT NOT NULL DEFAULT 'legacy',
                        params_json TEXT NOT NULL,
                        status_code INTEGER NOT NULL,
                        raw_response BLOB NOT NULL,
                        response_hash TEXT NOT NULL,
                        safe_headers_json TEXT NOT NULL,
                        requested_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS cooldowns (
                        cache_key TEXT PRIMARY KEY,
                        cooldown_until TEXT NOT NULL
                    );
                    """
                )
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(responses)")
                }
                if "cache_version" not in columns:
                    connection.execute(
                        "ALTER TABLE responses ADD COLUMN "
                        "cache_version TEXT NOT NULL DEFAULT 'legacy'"
                    )

    def put_response(
        self,
        *,
        key: str,
        provider: str,
        endpoint: str,
        cache_version: str,
        params: Mapping[str, object],
        raw_response: bytes,
        requested_at: datetime,
        ttl: timedelta,
        safe_headers: Mapping[str, str],
    ) -> None:
        """Atomically store one successful response."""
        sanitized_headers = {
            key.casefold(): value
            for key, value in safe_headers.items()
            if key.casefold() in SAFE_RESPONSE_HEADERS
        }
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO responses (
                        cache_key, provider, endpoint, cache_version, params_json,
                        status_code, raw_response, response_hash,
                        safe_headers_json, requested_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, 200, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        provider,
                        endpoint,
                        cache_version,
                        _canonical_json_bytes(canonical_request_params(params)).decode(
                            "utf-8"
                        ),
                        raw_response,
                        _sha256(raw_response),
                        _canonical_json_bytes(sanitized_headers).decode("utf-8"),
                        requested_at.isoformat(),
                        (requested_at + ttl).isoformat(),
                    ),
                )

    def get_response(self, key: str) -> CachedResponse | None:
        """Return an unexpired response without deleting expired history."""
        cached = self._get_response_any(key)
        if cached is None or cached.expires_at <= self._clock():
            return None
        return cached

    def get_snapshot_response(self, key: str) -> CachedResponse | None:
        """Return a stored response for formal snapshot validation, regardless of TTL."""
        return self._get_response_any(key)

    def _get_response_any(self, key: str) -> CachedResponse | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM responses WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        return CachedResponse(
            cache_key=row["cache_key"],
            provider=row["provider"],
            endpoint=row["endpoint"],
            cache_version=row["cache_version"],
            params=json.loads(row["params_json"]),
            status_code=row["status_code"],
            raw_response=row["raw_response"],
            response_hash=row["response_hash"],
            safe_headers=json.loads(row["safe_headers_json"]),
            requested_at=datetime.fromisoformat(row["requested_at"]),
            expires_at=expires_at,
        )

    def set_cooldown(self, key: str, until: datetime) -> None:
        """Record a provider cooldown without persisting an error body."""
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO cooldowns "
                    "(cache_key, cooldown_until) VALUES (?, ?)",
                    (key, until.isoformat()),
                )

    def get_cooldown(self, key: str) -> datetime | None:
        """Return an active cooldown deadline."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT cooldown_until FROM cooldowns WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        until = datetime.fromisoformat(row["cooldown_until"])
        return until if until > self._clock() else None

    def prepare_snapshot(self, cache_keys: Sequence[str]) -> PreparedSnapshot:
        """Load and validate exact cached bytes without writing formal artifacts."""
        if len(set(cache_keys)) != len(cache_keys):
            raise ValueError("snapshot cache keys must be unique")

        entries: list[dict[str, object]] = []
        files: list[tuple[Path, bytes]] = []
        responses: list[CachedResponse] = []
        for index, key in enumerate(cache_keys, start=1):
            cached = self._get_response_any(key)
            if cached is None:
                raise KeyError(f"unknown cache key: {key}")
            if _sha256(cached.raw_response) != cached.response_hash:
                raise ValueError("cached response bytes do not match response hash")
            relative_path = Path("snapshots") / f"openalex-{index:04d}.json"
            files.append((relative_path, cached.raw_response))
            responses.append(cached)
            entries.append(
                {
                    "cache_key": cached.cache_key,
                    "cache_version": cached.cache_version,
                    "endpoint": cached.endpoint,
                    "params": cached.params,
                    "provider": cached.provider,
                    "requested_at": cached.requested_at.isoformat(),
                    "response_hash": cached.response_hash,
                    "snapshot_path": relative_path.as_posix(),
                    "snapshot_sha256": _sha256(cached.raw_response),
                }
            )

        manifest_content = json.dumps(
            {"contract_version": "provider-snapshot-v1", "entries": entries},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        return PreparedSnapshot(
            responses=tuple(responses),
            files=tuple(files),
            manifest_content=manifest_content,
        )

    def write_snapshot(self, prepared: PreparedSnapshot, run_dir: Path) -> Path:
        """Freeze one previously prepared snapshot without re-reading the cache."""
        root = run_dir.resolve()
        manifest_path = root / "snapshot_manifest.json"
        files: list[tuple[Path, bytes]] = []
        seen_paths: set[Path] = set()
        for relative, content in prepared.files:
            if relative.is_absolute():
                raise ValueError("snapshot path must stay under the run directory")
            path = (root / relative).resolve()
            if root not in path.parents or path == manifest_path or path in seen_paths:
                raise ValueError("snapshot path must stay under the run directory")
            seen_paths.add(path)
            files.append((path, content))

        for path, content in [*files, (manifest_path, prepared.manifest_content)]:
            if path.exists() and path.read_bytes() != content:
                raise FileExistsError(f"refusing to overwrite frozen file: {path}")
        for path, content in files:
            _frozen_write(path, content)
        _frozen_write(manifest_path, prepared.manifest_content)
        return manifest_path

    def export_snapshot(self, cache_keys: Sequence[str], run_dir: Path) -> Path:
        """Freeze exact cached response bytes and an ordered manifest."""
        return self.write_snapshot(self.prepare_snapshot(cache_keys), run_dir)


def validate_snapshot_manifest(path: Path) -> None:
    """Verify every frozen response referenced by a snapshot manifest."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid snapshot manifest") from error
    if not isinstance(payload, dict) or payload.get("contract_version") != (
        "provider-snapshot-v1"
    ):
        raise ValueError("invalid snapshot manifest contract")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("snapshot manifest entries must be a list")

    root = path.parent.resolve()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("snapshot manifest entry must be an object")
        relative = entry.get("snapshot_path")
        expected = entry.get("snapshot_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("snapshot manifest entry is incomplete")
        snapshot = (root / relative).resolve()
        if root not in snapshot.parents:
            raise ValueError("snapshot path escapes the run directory")
        try:
            actual = _sha256(snapshot.read_bytes())
        except OSError as error:
            raise ValueError("snapshot file is missing") from error
        if actual != expected or actual != entry.get("response_hash"):
            raise ValueError("snapshot hash mismatch")
