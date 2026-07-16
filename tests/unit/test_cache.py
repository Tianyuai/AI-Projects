from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from paper_search.storage.cache import (
    SQLiteResponseCache,
    make_cache_key,
    validate_snapshot_manifest,
)


class MutableClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


def test_cache_key_is_order_independent_and_excludes_api_key() -> None:
    first = make_cache_key(
        "openalex",
        "/works",
        {"search": "rag", "api_key": "secret-value", "per_page": 2},
        "v1",
    )
    second = make_cache_key(
        "openalex",
        "/works",
        {"per_page": 2, "search": "rag"},
        "v1",
    )

    assert first == second
    assert first.startswith("sha256:")
    assert "secret-value" not in first


def test_success_response_round_trips_safe_metadata(tmp_path: Path) -> None:
    now = datetime(2026, 7, 16, tzinfo=UTC)
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite3", clock=lambda: now)

    cache.put_response(
        key="key-1",
        provider="openalex",
        endpoint="/works",
        params={"search": "rag", "api_key": "secret-value"},
        raw_response=b'{"results":[]}',
        requested_at=now,
        ttl=timedelta(days=7),
        safe_headers={"x-request-id": "request-1"},
    )

    cached = cache.get_response("key-1")
    assert cached is not None
    assert cached.raw_response == b'{"results":[]}'
    assert cached.params == {"search": "rag"}
    assert cached.safe_headers == {"x-request-id": "request-1"}
    assert cached.response_hash.startswith("sha256:")
    assert b"secret-value" not in (tmp_path / "cache.sqlite3").read_bytes()


def test_success_response_expires_after_seven_days(tmp_path: Path) -> None:
    clock = MutableClock(datetime(2026, 7, 16, tzinfo=UTC))
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite3", clock=clock)
    cache.put_response(
        key="key-1",
        provider="openalex",
        endpoint="/works",
        params={"search": "rag"},
        raw_response=b"{}",
        requested_at=clock(),
        ttl=timedelta(days=7),
        safe_headers={},
    )

    assert cache.get_response("key-1") is not None
    clock.advance(timedelta(days=7, microseconds=1))
    assert cache.get_response("key-1") is None


def test_cooldown_expires_at_boundary(tmp_path: Path) -> None:
    clock = MutableClock(datetime(2026, 7, 16, tzinfo=UTC))
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite3", clock=clock)
    until = clock() + timedelta(seconds=60)

    cache.set_cooldown("key-1", until)

    assert cache.get_cooldown("key-1") == until
    clock.advance(timedelta(seconds=60))
    assert cache.get_cooldown("key-1") is None


def test_missing_cache_entries_return_none(tmp_path: Path) -> None:
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite3")

    assert cache.get_response("missing") is None
    assert cache.get_cooldown("missing") is None


def test_cache_connections_do_not_lock_database_file(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"
    cache = SQLiteResponseCache(path)

    assert cache.get_response("missing") is None
    cache.set_cooldown("key-1", datetime(2026, 7, 16, tzinfo=UTC))

    path.rename(tmp_path / "renamed-cache.sqlite3")


def populated_cache(tmp_path: Path) -> SQLiteResponseCache:
    now = datetime(2026, 7, 16, tzinfo=UTC)
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite3", clock=lambda: now)
    for index, key in enumerate(("page-1", "page-2"), start=1):
        cache.put_response(
            key=key,
            provider="openalex",
            endpoint="/works",
            params={"search": "rag", "cursor": f"cursor-{index}"},
            raw_response=json.dumps({"page": index}, sort_keys=True).encode(),
            requested_at=now + timedelta(seconds=index),
            ttl=timedelta(days=7),
            safe_headers={"x-request-id": f"request-{index}"},
        )
    return cache


def test_snapshot_export_is_deterministic_and_validated(tmp_path: Path) -> None:
    cache = populated_cache(tmp_path)
    manifest = cache.export_snapshot(["page-1", "page-2"], tmp_path / "run")
    first = manifest.read_bytes()

    payload = json.loads(first)
    assert payload["contract_version"] == "provider-snapshot-v1"
    assert [entry["cache_key"] for entry in payload["entries"]] == ["page-1", "page-2"]
    validate_snapshot_manifest(manifest)

    repeated = cache.export_snapshot(["page-1", "page-2"], tmp_path / "run")
    assert repeated.read_bytes() == first


def test_snapshot_preserves_exact_response_bytes(tmp_path: Path) -> None:
    cache = populated_cache(tmp_path)
    manifest = cache.export_snapshot(["page-1"], tmp_path / "run")
    entry = json.loads(manifest.read_bytes())["entries"][0]

    assert (manifest.parent / entry["snapshot_path"]).read_bytes() == b'{"page": 1}'


def test_snapshot_refuses_different_existing_content(tmp_path: Path) -> None:
    cache = populated_cache(tmp_path)
    cache.export_snapshot(["page-1"], tmp_path / "run")

    with pytest.raises(FileExistsError):
        cache.export_snapshot(["page-2"], tmp_path / "run")


def test_snapshot_validation_detects_tampering(tmp_path: Path) -> None:
    cache = populated_cache(tmp_path)
    manifest = cache.export_snapshot(["page-1"], tmp_path / "run")
    entry = json.loads(manifest.read_bytes())["entries"][0]
    (manifest.parent / entry["snapshot_path"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash"):
        validate_snapshot_manifest(manifest)
