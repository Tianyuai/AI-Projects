from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from paper_search.storage.cache import SQLiteResponseCache, make_cache_key


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
