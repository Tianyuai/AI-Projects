from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
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
) -> ProviderResult[list[Any]]:
    async with httpx.AsyncClient(
        base_url="https://api.openalex.org",
        transport=httpx.MockTransport(handler),
    ) as client:
        provider = OpenAlexProvider(client=client, cache=cache, api_key=API_KEY)
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
