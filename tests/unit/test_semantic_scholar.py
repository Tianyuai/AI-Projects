from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from paper_search.domain.models import BudgetReservation, ProviderPaperId, UsageEstimate
from paper_search.retrieval.semantic_scholar import (
    SemanticScholarProvider,
    decode_semantic_scholar_search,
)
from paper_search.storage.cache import SQLiteResponseCache


FIXTURES = Path("tests/fixtures/semantic_scholar")
API_KEY = "synthetic-test-key"


def _bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _reservation(calls: int = 1) -> BudgetReservation:
    return BudgetReservation(
        reservation_id=f"s2-{calls}",
        action="semantic_scholar",
        reserved=UsageEstimate(search_api_calls=calls),
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )


async def _provider(
    tmp_path: Path,
    handler: httpx.MockTransport,
) -> tuple[SemanticScholarProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=handler)
    return (
        SemanticScholarProvider(
            client=client,
            cache=SQLiteResponseCache(tmp_path / "cache.sqlite3"),
            api_key=API_KEY,
            clock=lambda: datetime(2026, 7, 23, tzinfo=UTC),
        ),
        client,
    )


def test_search_maps_complete_and_missing_fields_and_preserves_identity(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_bytes("search.json"), request=request)

    async def run() -> object:
        provider, client = await _provider(tmp_path, httpx.MockTransport(handler))
        async with client:
            return await provider.search(
                "graph retrieval",
                {"year_from": 2020, "year_to": 2025},
                2,
                _reservation(),
            )

    result = asyncio.run(run())

    assert [paper.semantic_scholar_id for paper in result.data] == ["S2-001", "S2-002"]
    assert result.data[0].canonical_id == "doi:10.9999/synthetic.001"
    assert result.data[0].sources == ["semantic_scholar"]
    assert result.data[1].canonical_id == "s2:S2-002"
    assert result.data[1].abstract is None
    assert seen[0].url.host == "api.semanticscholar.org"
    assert seen[0].headers["x-api-key"] == API_KEY
    assert result.provenance["provider"] == "semantic_scholar"
    assert API_KEY not in result.model_dump_json()


def test_batch_details_skips_null_entry_and_maps_relation_ids(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert json.loads(request.content) == {"ids": ["S2-001", "missing"]}
        return httpx.Response(200, content=_bytes("batch.json"), request=request)

    async def run() -> object:
        provider, client = await _provider(tmp_path, httpx.MockTransport(handler))
        async with client:
            return await provider.batch_details(["S2-001", "missing"], _reservation())

    result = asyncio.run(run())

    assert [paper.semantic_scholar_id for paper in result.data] == ["S2-001"]
    assert result.data[0].reference_ids == [
        ProviderPaperId(provider="semantic_scholar", value="S2-010")
    ]
    assert result.data[0].cited_by_ids == [
        ProviderPaperId(provider="semantic_scholar", value="S2-020")
    ]
    assert result.errors[0].code == "missing_record"


@pytest.mark.parametrize(
    ("method", "fixture", "neighbor_id", "citing", "cited"),
    [
        ("references", "references.json", "S2-010", "S2-001", "S2-010"),
        ("citations", "citations.json", "S2-020", "S2-020", "S2-001"),
    ],
)
def test_reference_and_citation_contracts(
    tmp_path: Path,
    method: str,
    fixture: str,
    neighbor_id: str,
    citing: str,
    cited: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_bytes(fixture), request=request)

    async def run() -> object:
        provider, client = await _provider(tmp_path, httpx.MockTransport(handler))
        async with client:
            operation = getattr(provider, method)
            return await operation(
                ProviderPaperId(provider="semantic_scholar", value="S2-001"),
                10,
                _reservation(),
            )

    result = asyncio.run(run())

    assert [paper.semantic_scholar_id for paper in result.data.papers] == [neighbor_id]
    edge = result.data.raw_edges[0]
    assert edge.provider == "semantic_scholar"
    assert edge.citing_provider_id.value == citing
    assert edge.cited_provider_id.value == cited


def test_empty_search_is_successful(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"total": 0, "offset": 0, "next": None, "data": []},
            request=request,
        )

    async def run() -> object:
        provider, client = await _provider(tmp_path, httpx.MockTransport(handler))
        async with client:
            return await provider.search("none", {}, 5, _reservation())

    result = asyncio.run(run())
    assert result.data == []
    assert result.errors == []


def test_invalid_record_does_not_drop_valid_sibling(tmp_path: Path) -> None:
    payload = json.loads(_bytes("search.json"))
    payload["data"].insert(0, {"paperId": None, "title": None})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    async def run() -> object:
        provider, client = await _provider(tmp_path, httpx.MockTransport(handler))
        async with client:
            return await provider.search("graph", {}, 3, _reservation())

    result = asyncio.run(run())
    assert len(result.data) == 2
    assert result.errors[0].code == "invalid_record"


@pytest.mark.parametrize(
    ("outcome", "code", "retryable"),
    [
        (429, "rate_limited", True),
        (503, "server_error", True),
        (400, "invalid_request", False),
        (httpx.ReadTimeout("slow"), "timeout", True),
    ],
)
def test_provider_failures_are_structured_and_safe(
    tmp_path: Path,
    outcome: int | Exception,
    code: str,
    retryable: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, json={"error": "synthetic"}, request=request)

    async def run() -> object:
        provider, client = await _provider(tmp_path, httpx.MockTransport(handler))
        async with client:
            return await provider.search("graph", {}, 2, _reservation())

    result = asyncio.run(run())
    assert result.data == []
    assert result.errors[-1].code == code
    assert result.errors[-1].retryable is retryable
    assert result.usage.search_api_calls == 1
    assert API_KEY not in result.model_dump_json()


def test_zero_call_reservation_prevents_transport(tmp_path: Path) -> None:
    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"transport must not be called: {request.url}")

    async def run() -> object:
        provider, client = await _provider(tmp_path, httpx.MockTransport(forbidden))
        async with client:
            return await provider.search("graph", {}, 2, _reservation(0))

    result = asyncio.run(run())
    assert result.errors[0].code == "budget_exhausted"
    assert result.usage.search_api_calls == 0


def test_semantic_scholar_decoder_is_pure_and_deterministic() -> None:
    raw = _bytes("search.json")

    first = decode_semantic_scholar_search(raw, limit=2)
    second = decode_semantic_scholar_search(raw, limit=2)

    assert first == second
    assert [paper.semantic_scholar_id for paper in first.papers] == ["S2-001", "S2-002"]
