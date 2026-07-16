"""SQLite-backed raw provider response cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping
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
    params: dict[str, str]
    status_code: int
    raw_response: bytes
    response_hash: str
    safe_headers: dict[str, str]
    requested_at: datetime
    expires_at: datetime


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
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
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

    def put_response(
        self,
        *,
        key: str,
        provider: str,
        endpoint: str,
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
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO responses (
                    cache_key, provider, endpoint, params_json, status_code,
                    raw_response, response_hash, safe_headers_json,
                    requested_at, expires_at
                ) VALUES (?, ?, ?, ?, 200, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    provider,
                    endpoint,
                    _canonical_json_bytes(canonical_request_params(params)).decode("utf-8"),
                    raw_response,
                    _sha256(raw_response),
                    _canonical_json_bytes(sanitized_headers).decode("utf-8"),
                    requested_at.isoformat(),
                    (requested_at + ttl).isoformat(),
                ),
            )

    def get_response(self, key: str) -> CachedResponse | None:
        """Return an unexpired response without deleting expired history."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM responses WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at <= self._clock():
            return None
        return CachedResponse(
            cache_key=row["cache_key"],
            provider=row["provider"],
            endpoint=row["endpoint"],
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
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO cooldowns (cache_key, cooldown_until) VALUES (?, ?)",
                (key, until.isoformat()),
            )

    def get_cooldown(self, key: str) -> datetime | None:
        """Return an active cooldown deadline."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cooldown_until FROM cooldowns WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        until = datetime.fromisoformat(row["cooldown_until"])
        return until if until > self._clock() else None
