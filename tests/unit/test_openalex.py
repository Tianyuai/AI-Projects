from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from paper_search.domain.models import BudgetReservation, ProviderResult, UsageEstimate
from paper_search.retrieval.openalex import OPENALEX_SELECT_FIELDS, OpenAlexProvider
from paper_search.storage.cache import SQLiteResponseCache


FIXTURE_ROOT = Path("tests/fixtures/openalex")
API_KEY = "test-key-not-a-real-secret"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURE_ROOT / name).read_bytes()


def reservation(calls: int) -> BudgetReservation:
    return BudgetReservation(
        reservation_id="reservation-1",
        action="openalex-search",
        reserved=UsageEstimate(search_api_calls=calls),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )


async def run_search(
    cache: SQLiteResponseCache,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    query: str = "RAG",
    filters: dict[str, object] | None = None,
    limit: int = 2,
    calls: int = 1,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> ProviderResult[list[Any]]:
    async with httpx.AsyncClient(
        base_url="https://api.openalex.org",
        transport=httpx.MockTransport(handler),
    ) as client:
        provider_kwargs: dict[str, object] = {}
        if sleep is not None:
            provider_kwargs.update(sleep=sleep, jitter=lambda: 0.0)
        provider = OpenAlexProvider(
            client=client,
            cache=cache,
            api_key=API_KEY,
            **provider_kwargs,
        )
        return await provider.search(query, filters or {}, limit, reservation(calls))


def test_search_builds_safe_bounded_request_and_maps_results(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=fixture_bytes("works_page_1.json"))

    result = asyncio.run(
        run_search(
            SQLiteResponseCache(tmp_path / "cache.sqlite3"),
            handler,
            filters={"year_from": 2020, "year_to": 2024},
        )
    )

    assert [paper.openalex_id for paper in result.data] == ["W123", "W124"]
    params = seen[0].url.params
    assert params["search"] == "RAG"
    assert params["per_page"] == "2"
    assert params["cursor"] == "*"
    assert params["select"] == OPENALEX_SELECT_FIELDS
    assert params["filter"] == (
        "from_publication_date:2020-01-01,to_publication_date:2024-12-31"
    )
    assert params["api_key"] == API_KEY
    assert result.usage.search_api_calls == 1
    assert result.cache_hit is False
    assert API_KEY not in json.dumps(result.provenance)


def test_search_pages_until_limit(tmp_path: Path) -> None:
    seen_cursors: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params["cursor"]
        seen_cursors.append(cursor)
        fixture = "works_page_1.json" if cursor == "*" else "works_page_2.json"
        return httpx.Response(200, content=fixture_bytes(fixture))

    result = asyncio.run(
        run_search(
            SQLiteResponseCache(tmp_path / "cache.sqlite3"),
            handler,
            limit=3,
            calls=2,
        )
    )

    assert seen_cursors == ["*", "cursor-page-2"]
    assert [paper.openalex_id for paper in result.data] == ["W123", "W124", "W126"]
    assert result.usage.search_api_calls == 2
    assert len(json.loads(result.provenance["cache_keys"])) == 2


def test_empty_search_returns_successful_empty_result(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=fixture_bytes("works_empty.json"), request=request)

    result = asyncio.run(
        run_search(SQLiteResponseCache(tmp_path / "cache.sqlite3"), handler, limit=5)
    )

    assert result.data == []
    assert result.errors == []
    assert result.usage.search_api_calls == 1


def test_single_page_response_hash_is_the_raw_response_hash(tmp_path: Path) -> None:
    raw = fixture_bytes("works_empty.json")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw, request=request)

    result = asyncio.run(
        run_search(SQLiteResponseCache(tmp_path / "cache.sqlite3"), handler, limit=5)
    )

    assert result.provenance["response_hash"] == (
        "sha256:" + hashlib.sha256(raw).hexdigest()
    )


def test_search_replays_cache_without_network_call(tmp_path: Path) -> None:
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite3")

    def online(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=fixture_bytes("works_page_1.json"), request=request)

    first = asyncio.run(run_search(cache, online))

    def offline(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"network must not be called: {request.url.path}")

    replay = asyncio.run(run_search(cache, offline, calls=0))

    assert replay.data == first.data
    assert replay.cache_hit is True
    assert replay.usage.search_api_calls == 0


@pytest.mark.parametrize(
    ("query", "filters", "limit"),
    [
        (" ", {}, 1),
        ("valid", {"unknown": 1}, 1),
        ("valid", {}, 0),
        ("valid", {}, 301),
        ("valid", {"year_from": True}, 1),
    ],
)
def test_search_rejects_invalid_inputs(
    tmp_path: Path,
    query: str,
    filters: dict[str, object],
    limit: int,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"invalid input must not call network: {request.url}")

    with pytest.raises(ValueError):
        asyncio.run(
            run_search(
                SQLiteResponseCache(tmp_path / "cache.sqlite3"),
                handler,
                query=query,
                filters=filters,
                limit=limit,
            )
        )


class ScriptedHandler:
    def __init__(self, outcomes: list[bytes | int | Exception]) -> None:
        self.outcomes = outcomes
        self.attempts = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        outcome = self.outcomes[self.attempts]
        self.attempts += 1
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, int):
            return httpx.Response(
                outcome,
                json={"error": f"status-{outcome}"},
                headers={"x-request-id": "request-error"},
                request=request,
            )
        return httpx.Response(200, content=outcome, request=request)


def test_429_retries_three_times_sets_cooldown_and_then_uses_zero_calls(
    tmp_path: Path,
) -> None:
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite3")
    scripted = ScriptedHandler([429, 429, 429])
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    result = asyncio.run(run_search(cache, scripted, calls=3, sleep=fake_sleep))

    assert scripted.attempts == 3
    assert sleeps == [1.0, 2.0]
    assert result.errors[-1].code == "rate_limited"
    assert result.errors[-1].request_id == "request-error"
    assert result.usage.search_api_calls == 3

    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"cooldown must prevent network: {request.url}")

    cooled = asyncio.run(run_search(cache, forbidden, calls=0, sleep=fake_sleep))
    assert cooled.errors[-1].code == "rate_limited"
    assert cooled.usage.search_api_calls == 0


def test_timeout_then_success_counts_both_attempts(tmp_path: Path) -> None:
    scripted = ScriptedHandler(
        [httpx.ReadTimeout("slow"), fixture_bytes("works_page_1.json")]
    )

    async def no_wait(delay: float) -> None:
        assert delay == 1.0

    result = asyncio.run(
        run_search(
            SQLiteResponseCache(tmp_path / "cache.sqlite3"),
            scripted,
            calls=2,
            sleep=no_wait,
        )
    )

    assert scripted.attempts == 2
    assert result.usage.search_api_calls == 2
    assert len(result.data) == 2


def test_connection_failure_is_bounded_and_structured(tmp_path: Path) -> None:
    scripted = ScriptedHandler(
        [httpx.ConnectError("offline"), httpx.ConnectError("offline")]
    )

    async def no_wait(delay: float) -> None:
        assert delay == 1.0

    result = asyncio.run(
        run_search(
            SQLiteResponseCache(tmp_path / "cache.sqlite3"),
            scripted,
            calls=2,
            sleep=no_wait,
        )
    )

    assert scripted.attempts == 2
    assert result.data == []
    assert result.errors[-1].code == "network_error"
    assert result.errors[-1].retryable is True
    assert API_KEY not in result.model_dump_json()


def test_repeated_cursor_stops_before_replaying_same_page(tmp_path: Path) -> None:
    class GuardedCache(SQLiteResponseCache):
        reads = 0

        def get_response(self, key: str) -> Any:
            self.reads += 1
            if self.reads > 1:
                raise AssertionError("repeated cursor must stop before a second cache read")
            return super().get_response(key)

    payload = json.dumps(
        {"meta": {"next_cursor": "*"}, "results": [{"id": None, "title": None}]}
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload, request=request)

    result = asyncio.run(run_search(GuardedCache(tmp_path / "cache.sqlite3"), handler))

    assert result.data == []
    assert result.errors[-1].code == "pagination_cycle"
    assert result.usage.search_api_calls == 1


def test_second_page_failure_returns_first_page_and_structured_error(tmp_path: Path) -> None:
    scripted = ScriptedHandler([fixture_bytes("works_page_1.json"), 500, 500, 500])

    async def no_wait(delay: float) -> None:
        assert delay in {1.0, 2.0}

    result = asyncio.run(
        run_search(
            SQLiteResponseCache(tmp_path / "cache.sqlite3"),
            scripted,
            limit=3,
            calls=4,
            sleep=no_wait,
        )
    )

    assert [paper.openalex_id for paper in result.data] == ["W123", "W124"]
    assert result.errors[-1].code == "server_error"
    assert result.usage.search_api_calls == 4


def test_budget_exhaustion_returns_error_without_network(tmp_path: Path) -> None:
    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"budget exhaustion must prevent network: {request.url}")

    result = asyncio.run(
        run_search(SQLiteResponseCache(tmp_path / "cache.sqlite3"), forbidden, calls=0)
    )

    assert result.data == []
    assert result.errors[-1].code == "budget_exhausted"
    assert result.usage.search_api_calls == 0


@pytest.mark.parametrize(
    ("outcome", "expected_code"),
    [
        (400, "invalid_request"),
        (403, "authentication_error"),
        (b"not-json", "invalid_response"),
        (b'{"meta": {}}', "invalid_response"),
    ],
)
def test_nonretryable_failures_return_structured_errors(
    tmp_path: Path,
    outcome: bytes | int,
    expected_code: str,
) -> None:
    scripted = ScriptedHandler([outcome])

    result = asyncio.run(
        run_search(SQLiteResponseCache(tmp_path / "cache.sqlite3"), scripted, calls=1)
    )

    assert scripted.attempts == 1
    assert result.data == []
    assert result.errors[-1].code == expected_code
    assert API_KEY not in result.model_dump_json()


def test_invalid_work_is_skipped_without_losing_valid_sibling(tmp_path: Path) -> None:
    scripted = ScriptedHandler([fixture_bytes("works_invalid_record.json")])

    result = asyncio.run(
        run_search(SQLiteResponseCache(tmp_path / "cache.sqlite3"), scripted, calls=1)
    )

    assert [paper.openalex_id for paper in result.data] == ["W127"]
    assert result.errors[0].code == "invalid_work"


def test_task3_public_api_exports() -> None:
    import paper_search.processing as processing
    import paper_search.retrieval as retrieval
    import paper_search.storage as storage

    assert retrieval.OpenAlexProvider is OpenAlexProvider
    assert processing.normalize_openalex_work.__name__ == "normalize_openalex_work"
    assert storage.SQLiteResponseCache is SQLiteResponseCache
    assert storage.validate_snapshot_manifest.__name__ == "validate_snapshot_manifest"
