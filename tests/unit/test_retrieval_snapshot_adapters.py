from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from paper_search.control.budget import HardBudgetController
from paper_search.control.pricing import ActualCostPricer, load_pricing_policy
from paper_search.domain.models import (
    BudgetReservation,
    ProviderPaperId,
    UsageActual,
    UsageEstimate,
    SearchBudget,
)
from paper_search.retrieval.snapshot_adapters import (
    LiveCaptureSearchProvider,
    ProviderAdapterError,
    ReplaySearchProvider,
)
from paper_search.retrieval.openalex import OPENALEX_SELECT_FIELDS
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencyRequestIdentity,
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


def _openalex_identity() -> DependencyRequestIdentity:
    return DependencyRequestIdentity.from_canonical_request(
        dependency="openalex",
        operation="search",
        method="GET",
        endpoint="/works",
        model_or_adapter="openalex-works-v1",
        canonical_request={
            "query": "RAG",
            "filters": {},
            "limit": 3,
            "cursor": "*",
            "per_page": 3,
            "select": OPENALEX_SELECT_FIELDS,
        },
    )


def _real_controller_reservation(
    *, estimate_cost: str = "0.01"
) -> tuple[HardBudgetController, BudgetReservation]:
    budget = SearchBudget(
        max_search_api_calls=4,
        target_search_api_calls=1,
        max_llm_calls=1,
        target_llm_calls=0,
        max_total_tokens=1,
        max_cost_cny=1.0,
        max_elapsed_seconds=120,
        soft_deadline_seconds=100,
    )
    controller = HardBudgetController(
        budget,
        formal_live=True,
        reservation_ttl_seconds=60,
        clock=lambda: CAPTURED_AT,
    )
    reservation = controller.reserve(
        "provider.search",
        UsageEstimate(
            search_api_calls=3,
            cost_cny=Decimal(estimate_cost),
            elapsed_ms=60_000,
        ),
    )
    return controller, reservation


async def _capture(
    tmp_path: Path,
    *,
    dependency: str,
    handler: Callable[[httpx.Request], httpx.Response],
    sleep: Callable[[float], Awaitable[None]] | None = None,
    mailto: str | None = None,
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
        mailto=mailto,
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


def test_openalex_mailto_is_sent_but_never_enters_snapshot_identity(
    tmp_path: Path,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            content=(OPENALEX / "works_page_1.json").read_bytes(),
            request=request,
        )

    async def run() -> object:
        live, store, _, client = await _capture(
            tmp_path,
            dependency="openalex",
            handler=handler,
            mailto="team@example.com",
        )
        async with client:
            result = await live.search("RAG", {}, 3, _reservation())
        manifest = store.seal()
        return result, manifest

    result, manifest = asyncio.run(run())
    serialized = manifest.model_dump_json() + result.model_dump_json()

    assert seen[0].url.params["mailto"] == "team@example.com"
    assert "team@example.com" not in serialized
    assert "mailto" not in serialized.casefold()


def test_failed_response_is_captured_as_error_and_retries_are_accounted(
    tmp_path: Path,
) -> None:
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
    manifest = store.seal()
    assert len(manifest.entries) == 1
    assert manifest.entries[0].error is not None
    assert manifest.entries[0].error.code == "server_error"
    assert manifest.entries[0].error.retryable is True
    assert manifest.entries[0].response_sha256 == (
        "sha256:" + hashlib.sha256(b"sensitive failed body").hexdigest()
    )


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


@pytest.mark.parametrize("fault", ["capture", "decode"])
def test_dispatched_faults_fail_closed_with_valued_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    controller, reservation = _real_controller_reservation()

    class FailingStore(DependencyCaptureStore):
        def stage_success(self, *args: object, **kwargs: object) -> object:
            raise OSError("synthetic snapshot failure")

    store: DependencyCaptureStore = DependencyCaptureStore(tmp_path / "snapshot")
    if fault == "capture":
        store = FailingStore(tmp_path / "snapshot")
    else:
        def fail_decode(content: bytes, *, limit: int) -> object:
            del content, limit
            raise RuntimeError("synthetic decoder failure")

        monkeypatch.setattr(
            "paper_search.retrieval.snapshot_adapters.decode_semantic_scholar_search",
            fail_decode,
        )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=(S2 / "search.json").read_bytes(), request=request)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = LiveCaptureSearchProvider(
                dependency="semantic_scholar",
                client=client,
                capture_store=store,
                pricer=_pricer(),
                controller=controller,
                clock=lambda: CAPTURED_AT,
            )
            with pytest.raises(ProviderAdapterError, match="provider live capture failed"):
                await provider.search("graph", {}, 2, reservation)

    asyncio.run(run())

    assert controller.committed_usage.search_api_calls == 1
    assert controller.known_committed_cost_cny == Decimal("0.000060")
    assert controller.unknown_cost_actions == []
    assert controller.stop_status() == "hard_stop"


def test_settlement_failure_fail_closes_with_known_valued_cost(tmp_path: Path) -> None:
    controller, reservation = _real_controller_reservation(
        estimate_cost="0.000001"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=(S2 / "search.json").read_bytes(), request=request)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = LiveCaptureSearchProvider(
                dependency="semantic_scholar",
                client=client,
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=_pricer(),
                controller=controller,
                clock=lambda: CAPTURED_AT,
            )
            with pytest.raises(ProviderAdapterError):
                await provider.search("graph", {}, 2, reservation)

    asyncio.run(run())

    assert controller.committed_usage.search_api_calls == 1
    assert controller.known_committed_cost_cny == Decimal("0.000060")
    assert controller.unknown_cost_actions == []


def test_pricing_failure_fail_closes_with_unknown_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, reservation = _real_controller_reservation()

    def fail_pricing(*args: object, **kwargs: object) -> UsageActual:
        del args, kwargs
        raise RuntimeError("synthetic pricing failure")

    monkeypatch.setattr(ActualCostPricer, "value_actual", fail_pricing)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=(S2 / "search.json").read_bytes(), request=request)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = LiveCaptureSearchProvider(
                dependency="semantic_scholar",
                client=client,
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=_pricer(),
                controller=controller,
                clock=lambda: CAPTURED_AT,
            )
            with pytest.raises(ProviderAdapterError, match="provider live capture failed"):
                await provider.search("graph", {}, 2, reservation)

    asyncio.run(run())

    assert controller.committed_usage.search_api_calls == 1
    assert controller.committed_usage.cost_cny is None
    assert controller.known_committed_cost_cny == Decimal("0")
    assert controller.unknown_cost_actions == ["provider.search"]
    assert controller.stop_status() == "hard_stop"


def test_cancellation_during_retry_sleep_terminalizes_prior_attempt(tmp_path: Path) -> None:
    controller, reservation = _real_controller_reservation()
    sleep_started = asyncio.Event()
    never = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"not captured", request=request)

    async def blocking_sleep(delay: float) -> None:
        assert delay == 1.0
        sleep_started.set()
        await never.wait()

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = LiveCaptureSearchProvider(
                dependency="semantic_scholar",
                client=client,
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=_pricer(),
                controller=controller,
                clock=lambda: CAPTURED_AT,
                sleep=blocking_sleep,
                jitter=lambda: 0.0,
            )
            task = asyncio.create_task(provider.search("graph", {}, 2, reservation))
            await sleep_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())

    assert controller.committed_usage.search_api_calls == 1
    assert controller.known_committed_cost_cny == Decimal("0.000060")
    assert controller.stop_status() == "hard_stop"


def test_retry_is_not_started_when_backoff_exceeds_remaining_deadline(
    tmp_path: Path,
) -> None:
    settled = SettlementSpy()
    reservation = _reservation().model_copy(
        update={"expires_at": datetime.now(UTC) + timedelta(milliseconds=20)}
    )
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"not captured", request=request)

    async def forbidden_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = LiveCaptureSearchProvider(
                dependency="semantic_scholar",
                client=client,
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=_pricer(),
                controller=settled,
                clock=lambda: datetime.now(UTC),
                sleep=forbidden_sleep,
                jitter=lambda: 0.0,
            )
            return await provider.search("graph", {}, 2, reservation)

    result = asyncio.run(run())

    assert sleeps == []
    assert result.errors[-1].code == "timeout"
    assert result.usage.search_api_calls == 1


def test_total_wall_deadline_rejects_response_that_crosses_deadline(tmp_path: Path) -> None:
    settled = SettlementSpy()
    now = datetime.now(UTC)
    reservation = _reservation().model_copy(
        update={"expires_at": now + timedelta(milliseconds=20)}
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, content=(S2 / "search.json").read_bytes(), request=request)

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = LiveCaptureSearchProvider(
                dependency="semantic_scholar",
                client=client,
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=_pricer(),
                controller=settled,
                clock=lambda: datetime.now(UTC),
                jitter=lambda: 0.0,
            )
            return await provider.search("graph", {}, 2, reservation)

    result = asyncio.run(run())
    assert result.data == []
    assert result.errors[-1].code == "timeout"
    assert result.usage.search_api_calls == 1


def test_http_200_invalid_bytes_are_captured_before_decode_and_replayed(
    tmp_path: Path,
) -> None:
    raw = b"not-json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw, request=request)

    async def run() -> tuple[object, object, object]:
        live, store, _, client = await _capture(
            tmp_path,
            dependency="openalex",
            handler=handler,
        )
        async with client:
            captured = await live.search("RAG", {}, 1, _reservation())
        manifest = store.seal()
        replay = ReplaySearchProvider(
            dependency="openalex",
            reader=_reader(store, snapshot_set_id=manifest.snapshot_set_id),
            clock=lambda: CAPTURED_AT,
        )
        replayed = await replay.search("RAG", {}, 1, _reservation(0))
        return captured, replayed, manifest

    captured, replayed, manifest = asyncio.run(run())
    assert len(manifest.entries) == 1
    assert captured.errors[-1].code == "invalid_response"
    assert replayed.errors[-1].code == "invalid_response"
    assert replayed.errors[-1].code != "snapshot_unavailable"
    expected_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
    captured_refs = json.loads(captured.provenance["snapshot_refs"])
    replayed_refs = json.loads(replayed.provenance["snapshot_refs"])
    assert replayed.cache_hit is True
    assert len(captured_refs) == len(replayed_refs) == 1
    assert replayed_refs == captured_refs
    assert captured.provenance["response_hash"] == expected_hash
    assert replayed.provenance["response_hash"] == expected_hash


def test_openalex_error_is_captured_and_replayed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            text='{"message":"Too Many Requests"}',
            request=request,
        )

    async def run() -> tuple[object, object]:
        live, store, _, client = await _capture(
            tmp_path,
            dependency="openalex",
            handler=handler,
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
        return captured, replayed

    live_result, replay_result = asyncio.run(run())

    assert [error.code for error in live_result.errors] == ["rate_limited"]
    assert [error.code for error in replay_result.errors] == ["rate_limited"]
    assert live_result.data == []
    assert replay_result.data == []
    assert live_result.provenance["snapshot_refs"]
    assert replay_result.provenance["snapshot_refs"] == live_result.provenance["snapshot_refs"]


def test_replay_reproduces_staged_provider_error(tmp_path: Path) -> None:
    store = DependencyCaptureStore(tmp_path / "snapshot", clock=lambda: CAPTURED_AT)
    store.stage_error(
        _openalex_identity(),
        error_code="rate_limited",
        message="openalex request was rate limited",
        retryable=True,
        response_bytes=b'{"message":"Too Many Requests"}',
        safe_headers={"content-type": "application/json"},
        captured_at=CAPTURED_AT,
    )
    manifest = store.seal()
    replay = ReplaySearchProvider(
        dependency="openalex",
        reader=_reader(store, snapshot_set_id=manifest.snapshot_set_id),
        clock=lambda: CAPTURED_AT,
    )

    result = asyncio.run(replay.search("RAG", {}, 3, _reservation(0)))

    assert result.data == []
    assert [error.code for error in result.errors] == ["rate_limited"]
    assert result.provenance["snapshot_refs"]
