from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from paper_search.storage.cache import (
    SQLiteResponseCache,
    make_cache_key,
    validate_snapshot_manifest,
)
from paper_search.storage.dependency_snapshot import migrate_v1_to_v2


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
        cache_version="v2",
        params={"search": "rag", "api_key": "secret-value"},
        raw_response=b'{"results":[]}',
        requested_at=now,
        ttl=timedelta(days=7),
        safe_headers={"x-request-id": "request-1"},
    )

    cached = cache.get_response("key-1")
    assert cached is not None
    assert cached.cache_version == "v2"
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
        cache_version="v1",
        params={"search": "rag"},
        raw_response=b"{}",
        requested_at=clock(),
        ttl=timedelta(days=7),
        safe_headers={},
    )

    assert cache.get_response("key-1") is not None
    clock.advance(timedelta(days=7, microseconds=1))
    assert cache.get_response("key-1") is None


def test_snapshot_response_reads_expired_history(tmp_path: Path) -> None:
    clock = MutableClock(datetime(2026, 7, 16, tzinfo=UTC))
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite3", clock=clock)
    cache.put_response(
        key="key-1",
        provider="openalex",
        endpoint="/works",
        cache_version="v1",
        params={"search": "rag"},
        raw_response=b"{}",
        requested_at=clock(),
        ttl=timedelta(days=7),
        safe_headers={},
    )
    clock.advance(timedelta(days=8))

    cached = cache.get_snapshot_response("key-1")

    assert cached is not None
    assert cached.raw_response == b"{}"


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


def test_initialization_migrates_legacy_response_table(tmp_path: Path) -> None:
    path = tmp_path / "legacy-cache.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE responses (
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
            )
            """
        )

    SQLiteResponseCache(path)

    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(responses)")
        }
    assert "cache_version" in columns


def populated_cache(tmp_path: Path) -> SQLiteResponseCache:
    now = datetime(2026, 7, 16, tzinfo=UTC)
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite3", clock=lambda: now)
    for index, key in enumerate(("page-1", "page-2"), start=1):
        cache.put_response(
            key=key,
            provider="openalex",
            endpoint="/works",
            cache_version="v1",
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
    assert [entry["cache_version"] for entry in payload["entries"]] == ["v1", "v1"]
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


@pytest.mark.parametrize("relative", [Path("../outside.json"), Path("C:/outside.json")])
def test_prepared_snapshot_write_rejects_paths_outside_run_directory(
    tmp_path: Path,
    relative: Path,
) -> None:
    cache = populated_cache(tmp_path)
    prepared = cache.prepare_snapshot(["page-1"])
    malicious = replace(prepared, files=((relative, b"malicious"),))

    with pytest.raises(ValueError, match="snapshot path"):
        cache.write_snapshot(malicious, tmp_path / "run")


def test_v1_provider_snapshot_migration_requires_complete_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v1"
    source.mkdir()
    (source / "response.json").write_bytes(b"{}")
    complete_entry = {
        "provider": "openalex",
        "operation": "search",
        "method": "GET",
        "endpoint": "/works",
        "cache_version": "adapter-v1",
        "params": {"search": "rag", "per_page": "2"},
        "requested_at": "2026-08-01T12:00:00Z",
        "response_hash": (
            "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        ),
        "snapshot_path": "response.json",
    }
    manifest = source / "snapshot_manifest.json"
    manifest.write_text(
        json.dumps(
            {"contract_version": "provider-snapshot-v1", "entries": [complete_entry]}
        ),
        encoding="utf-8",
    )

    migrated = migrate_v1_to_v2(
        manifest, tmp_path / "v2", sealed_at=datetime(2026, 8, 1, tzinfo=UTC)
    )
    assert migrated.entries[0].request.dependency == "openalex"
    assert migrated.entries[0].request.endpoint == "/works"

    del complete_entry["operation"]
    manifest.write_text(
        json.dumps(
            {"contract_version": "provider-snapshot-v1", "entries": [complete_entry]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ambiguous V1 snapshot entry"):
        migrate_v1_to_v2(
            manifest,
            tmp_path / "ambiguous-v2",
            sealed_at=datetime(2026, 8, 1, tzinfo=UTC),
        )
