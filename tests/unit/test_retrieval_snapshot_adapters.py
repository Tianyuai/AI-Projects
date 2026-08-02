from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx

from paper_search.control.pricing import ActualCostPricer, load_pricing_policy
from paper_search.domain.models import (
    BudgetReservation,
    ProviderPaperId,
    UsageActual,
    UsageEstimate,
)
from paper_search.retrieval.snapshot_adapters import (
    LiveCaptureSearchProvider,
    ReplaySearchProvider,
)
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencySnapshotReader,
)


CAPTURED_AT = datetime(2026, 8, 2, 3, 4, 5, tzinfo=UTC)
PRICING = Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")
OPENALEX = Path("tests/fixtures/openalex")
S2 = Path("tests/fixtures/semantic_scholar")


class SettlementSpy:
    def __init__(self) -> None:
        self.settled: list[tuple[BudgetReservation, UsageActual]] = []
        self.dispatched: list[str] = []

    def mark_dispatched(self, reservation: BudgetReservation) -> None:
        self.dispatched.append(reservation.reservation_id)

    def settle(self, reservation: BudgetReservation, actual: UsageActual) -> None:
        self.settled.append((reservation, actual))

    def fail_closed(self, reservation: BudgetReservation, actual: UsageActual) -> None:
        raise AssertionError((reservation, actual))


def _reservation(calls: int = 3) -> BudgetReservation:
    return BudgetReservation(
        reservation_id="provider-request",
        action="provider.search",
        reserved=UsageEstimate(search_api_calls=calls, cost_cny=Decimal("0.01")),
        expires_at=CAPTURED_AT + timedelta(minutes=5),
    )


def _pricer() -> ActualCostPricer:
    return ActualCostPricer(load_pricing_policy(PRICING), valued_at=CAPTURED_AT)


def _reader(
    store: DependencyCaptureStore,
    *,
    snapshot_set_id: str,
) -> DependencySnapshotReader:
    return DependencySnapshotReader(
        store.manifest_path,
        snapshot_manifest_sha256=store.manifest_sha256,
        snapshot_set_id=snapshot_set_id,
    )


async def _capture(
    tmp_path: Path,
    *,
    dependency: str,
    handler: Callable[[httpx.Request], httpx.Response],
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> tuple[LiveCaptureSearchProvider, DependencyCaptureStore, SettlementSpy, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    store = DependencyCaptureStore(tmp_path / "snapshot", clock=lambda: CAPTURED_AT)
    controller = SettlementSpy()
    provider = LiveCaptureSearchProvider(
        dependency=dependency,
        client=client,
        capture_store=store,
        pricer=_pricer(),
        controller=controller,
        api_key="synthetic-key",
        clock=lambda: CAPTURED_AT,
        sleep=sleep,
        jitter=lambda: 0.0,
    )
    return provider, store, controller, client


def test_openalex_live_and_replay_are_identical_and_refs_are_page_ordered(tmp_path: Path) -> None:
    cursors: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params["cursor"]
        cursors.append(cursor)
        name = "works_page_1.json" if cursor == "*" else "works_page_2.json"
        return httpx.Response(200, content=(OPENALEX / name).read_bytes(), request=request)

    async def run() -> tuple[object, object]:
        live, store, controller, client = await _capture(
            tmp_path, dependency="openalex", handler=handler
        )
        async with client:
            captured = await live.search("RAG", {}, 3, _reservation())
        manifest = store.seal()
        replay = ReplaySearchProvider(
            dependency="openalex",
            reader=_reader(store, snapshot_set_id=manifest.snapshot_set_id),
            clock=lambda: CAPTURED_AT,
        )
        replayed = await replay.search("RAG", {}, 3, _reservation(0))
        assert controller.settled[0][1].cost_cny == Decimal("0.000100")
        return captured, replayed

    live_result, replay_result = asyncio.run(run())

    assert cursors == ["*", "cursor-page-2"]
    assert replay_result.data == live_result.data
    assert replay_result.usage == UsageActual()
    refs = json.loads(live_result.provenance["snapshot_refs"])
    expected_hashes = [
        "sha256:"
        + hashlib.sha256((OPENALEX / name).read_bytes()).hexdigest()
        for name in ("works_page_1.json", "works_page_2.json")
    ]
    assert [ref["response_sha256"] for ref in refs] == expected_hashes


def test_semantic_scholar_get_and_post_have_distinct_identities(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        fixture = "search.json" if request.method == "GET" else "batch.json"
        return httpx.Response(200, content=(S2 / fixture).read_bytes(), request=request)

    async def run() -> list[object]:
        live, store, _, client = await _capture(
            tmp_path, dependency="semantic_scholar", handler=handler
        )
        async with client:
            await live.search("graph retrieval", {}, 2, _reservation())
            await live.batch_details(["S2-001", "missing"], _reservation())
        return store.seal().entries

    entries = asyncio.run(run())

    assert {(entry.request.method, entry.request.endpoint) for entry in entries} == {
        ("GET", "/paper/search"),
        ("POST", "/paper/batch"),
    }
    assert entries[0].cache_key != entries[1].cache_key


def test_semantic_scholar_live_and_replay_normalize_identically(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(S2 / "search.json").read_bytes(),
            request=request,
        )

    async def run() -> tuple[object, object]:
        live, store, _, client = await _capture(
            tmp_path,
            dependency="semantic_scholar",
            handler=handler,
        )
        async with client:
            captured = await live.search("graph retrieval", {}, 2, _reservation())
        manifest = store.seal()
        replay = ReplaySearchProvider(
            dependency="semantic_scholar",
            reader=_reader(store, snapshot_set_id=manifest.snapshot_set_id),
            clock=lambda: CAPTURED_AT,
        )
        replayed = await replay.search("graph retrieval", {}, 2, _reservation(0))
        return captured, replayed

    captured, replayed = asyncio.run(run())

    assert replayed.data == captured.data
    assert replayed.errors == captured.errors
    assert replayed.usage == UsageActual()


def test_capture_identity_is_safe_and_excludes_credentials(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=(S2 / "search.json").read_bytes(), request=request)

    async def run() -> object:
        live, store, _, client = await _capture(
            tmp_path, dependency="semantic_scholar", handler=handler
        )
        async with client:
            result = await live.search("graph retrieval", {}, 2, _reservation())
        return result, store.seal()

    result, manifest = asyncio.run(run())
    serialized = manifest.model_dump_json() + result.model_dump_json()
    assert "synthetic-key" not in serialized
    assert "api_key" not in serialized.casefold()


def test_failed_response_body_is_not_captured_and_retries_are_accounted(tmp_path: Path) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, content=b"sensitive failed body", request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def run() -> tuple[object, object]:
        live, store, controller, client = await _capture(
            tmp_path,
            dependency="semantic_scholar",
            handler=handler,
            sleep=fake_sleep,
        )
        async with client:
            result = await live.search("graph", {}, 2, _reservation())
        return result, (store, controller)

    result, (store, controller) = asyncio.run(run())

    assert attempts == 3
    assert sleeps == [1.0, 2.0]
    assert result.usage.search_api_calls == 3
    assert controller.settled[0][1].cost_cny == Decimal("0.000180")
    assert store.seal().entries == []
    assert b"sensitive failed body" not in store.manifest_path.read_bytes()


def test_replay_miss_is_structured_and_has_no_network_dependency(tmp_path: Path) -> None:
    store = DependencyCaptureStore(tmp_path / "empty", clock=lambda: CAPTURED_AT)
    manifest = store.seal()
    replay = ReplaySearchProvider(
        dependency="semantic_scholar",
        reader=_reader(store, snapshot_set_id=manifest.snapshot_set_id),
        clock=lambda: CAPTURED_AT,
    )

    result = asyncio.run(replay.search("missing", {}, 1, _reservation(0)))

    assert result.data == []
    assert result.errors[0].code == "snapshot_unavailable"
    assert result.usage == UsageActual()


def test_replay_references_uses_snapshot_only(tmp_path: Path) -> None:
    paper_id = ProviderPaperId(provider="semantic_scholar", value="S2-001")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=(S2 / "references.json").read_bytes(), request=request)

    async def run() -> tuple[object, object]:
        live, store, _, client = await _capture(
            tmp_path, dependency="semantic_scholar", handler=handler
        )
        async with client:
            captured = await live.references(paper_id, 10, _reservation())
        manifest = store.seal()
        replay = ReplaySearchProvider(
            dependency="semantic_scholar",
            reader=_reader(store, snapshot_set_id=manifest.snapshot_set_id),
            clock=lambda: CAPTURED_AT,
        )
        return captured, await replay.references(paper_id, 10, _reservation(0))

    captured, replayed = asyncio.run(run())
    assert replayed.data == captured.data
    assert replayed.cache_hit is True


def test_expired_deadline_prevents_dispatch_and_accounts_zero_calls(tmp_path: Path) -> None:
    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"expired request must not reach transport: {request.url}")

    async def run() -> tuple[object, SettlementSpy]:
        live, _, controller, client = await _capture(
            tmp_path,
            dependency="semantic_scholar",
            handler=forbidden,
        )
        expired = _reservation().model_copy(update={"expires_at": CAPTURED_AT})
        async with client:
            result = await live.search("graph", {}, 2, expired)
        return result, controller

    result, controller = asyncio.run(run())

    assert result.errors[0].code == "budget_exhausted"
    assert result.usage.search_api_calls == 0
    assert controller.dispatched == []
    assert controller.settled[0][1].cost_cny == Decimal("0")
