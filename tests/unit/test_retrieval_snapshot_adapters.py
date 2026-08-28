from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import FrozenInstanceError
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
    SearchAttemptQuotaExceededError,
    _attempt_timeout,
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

    @property
    def policy_fingerprint(self) -> str:
        return "sha256:" + "c" * 64

    @property
    def formal_live(self) -> bool:
        return True

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


def _formal_controller(
    *,
    reservation_ttl_seconds: int = 60,
    max_search_api_calls: int = 4,
) -> HardBudgetController:
    return HardBudgetController(
        SearchBudget(
            max_search_api_calls=max_search_api_calls,
            target_search_api_calls=1,
            max_llm_calls=1,
            target_llm_calls=0,
            max_total_tokens=1,
            max_cost_cny=1.0,
            max_elapsed_seconds=120,
            soft_deadline_seconds=100,
        ),
        formal_live=True,
        reservation_ttl_seconds=reservation_ttl_seconds,
        clock=lambda: CAPTURED_AT,
    )


def test_openalex_live_capture_identity_is_derived_and_credential_free(
    tmp_path: Path,
) -> None:
    requests = 0

    def no_request(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise AssertionError(request)

    async def run() -> None:
        store = DependencyCaptureStore(tmp_path / "private-capture-root")
        pricer = _pricer()
        controller = _formal_controller()
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(no_request)) as first_client,
            httpx.AsyncClient(transport=httpx.MockTransport(no_request)) as second_client,
        ):
            first = LiveCaptureSearchProvider(
                dependency="openalex",
                client=first_client,
                capture_store=store,
                pricer=pricer,
                controller=controller,
                api_key="secret-one",
                mailto="private@example.invalid",
            )
            second = LiveCaptureSearchProvider(
                dependency="openalex",
                client=second_client,
                capture_store=store,
                pricer=pricer,
                controller=controller,
                api_key="secret-two",
                mailto="other@example.invalid",
            )
            assert first.live_identity_evidence == second.live_identity_evidence
            payload = first.live_identity_evidence.model_dump_json()
            assert "secret" not in payload
            assert "example.invalid" not in payload
            assert "private-capture-root" not in payload
            assert first.live_pricer is pricer
            assert first.live_controller is controller

    asyncio.run(run())
    assert requests == 0


def test_search_provider_descriptor_changes_with_dependency_or_adapter_version(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = DependencyCaptureStore(tmp_path / "snapshot")
        pricer = _pricer()
        controller = _formal_controller()
        transport = httpx.MockTransport(lambda request: pytest.fail(str(request)))
        async with httpx.AsyncClient(transport=transport) as client:
            openalex = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=store,
                pricer=pricer,
                controller=controller,
            )
            semantic = LiveCaptureSearchProvider(
                dependency="semantic_scholar",
                client=client,
                capture_store=store,
                pricer=pricer,
                controller=controller,
            )
            changed = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=store,
                pricer=pricer,
                controller=controller,
                adapter_version="openalex-works-v2",
            )
            assert openalex.live_identity_evidence.provider != (
                semantic.live_identity_evidence.provider
            )
            assert openalex.live_identity_evidence.provider != (
                changed.live_identity_evidence.provider
            )
            assert semantic.live_identity_evidence.provider.operations == (
                "search",
                "batch",
                "references",
                "citations",
            )

    asyncio.run(run())


def test_search_provider_evidence_changes_with_pricing_budget_or_ttl(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        store = DependencyCaptureStore(tmp_path / "snapshot")
        policy = load_pricing_policy(PRICING)
        changed_policy = policy.model_copy(
            update={"source_identity": "operator-verified-changed-policy"}
        )
        pricers = (
            ActualCostPricer(policy, valued_at=CAPTURED_AT),
            ActualCostPricer(changed_policy, valued_at=CAPTURED_AT),
        )
        controllers = (
            _formal_controller(),
            _formal_controller(max_search_api_calls=5),
            _formal_controller(reservation_ttl_seconds=61),
        )
        transport = httpx.MockTransport(lambda request: pytest.fail(str(request)))
        async with httpx.AsyncClient(transport=transport) as client:
            baseline = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=store,
                pricer=pricers[0],
                controller=controllers[0],
            ).live_identity_evidence
            changed_pricing = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=store,
                pricer=pricers[1],
                controller=controllers[0],
            ).live_identity_evidence
            changed_budget = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=store,
                pricer=pricers[0],
                controller=controllers[1],
            ).live_identity_evidence
            changed_ttl = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=store,
                pricer=pricers[0],
                controller=controllers[2],
            ).live_identity_evidence
            assert baseline.pricing_policy_sha256 != changed_pricing.pricing_policy_sha256
            assert baseline.controller_policy_sha256 != (
                changed_budget.controller_policy_sha256
            )
            assert baseline.controller_policy_sha256 != changed_ttl.controller_policy_sha256

    asyncio.run(run())


def test_search_provider_rejects_nonformal_controller_before_dispatch(
    tmp_path: Path,
) -> None:
    requests = 0

    def no_request(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        raise AssertionError(request)

    controller = _formal_controller()
    controller.formal_live = False

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(no_request)
        ) as client:
            with pytest.raises(ValueError, match="formal-live"):
                LiveCaptureSearchProvider(
                    dependency="openalex",
                    client=client,
                    capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                    pricer=_pricer(),
                    controller=controller,
                )

    asyncio.run(run())
    assert requests == 0


def test_search_transport_config_tampering_fails_before_credential_or_query_dispatch(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def attacker(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError("attacker transport received a request")

    async def run() -> None:
        controller = _formal_controller()
        reservation = controller.reserve(
            "provider.search",
            UsageEstimate(search_api_calls=1, cost_cny=Decimal("0.01")),
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(attacker)
        ) as client:
            provider = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=_pricer(),
                controller=controller,
                api_key="PRIVATE API KEY",
                mailto="private@example.invalid",
                clock=lambda: CAPTURED_AT,
            )
            with pytest.raises(FrozenInstanceError):
                provider._transport_config.base_url = (  # type: ignore[attr-defined,misc]
                    "https://attacker.invalid"
                )
            object.__setattr__(
                provider._transport_config,  # type: ignore[attr-defined]
                "base_url",
                "https://attacker.invalid",
            )
            with pytest.raises(ValueError, match="transport configuration"):
                await provider.search("PRIVATE QUERY", {}, 1, reservation)

    asyncio.run(run())
    assert requests == []


async def _capture(
    tmp_path: Path,
    *,
    dependency: str,
    handler: Callable[[httpx.Request], httpx.Response],
    sleep: Callable[[float], Awaitable[None]] | None = None,
    mailto: str | None = None,
    additional_api_keys: Sequence[str] = (),
    minimum_request_interval_seconds: float = 0.0,
    attempt_gate: object | None = None,
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
        additional_api_keys=additional_api_keys,
        mailto=mailto,
        clock=lambda: CAPTURED_AT,
        sleep=sleep,
        jitter=lambda: 0.0,
        minimum_request_interval_seconds=minimum_request_interval_seconds,
        attempt_gate=attempt_gate,
    )
    return provider, store, controller, client


def test_openalex_attempt_gate_stops_before_network_dispatch(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    class ExhaustedGate:
        def claim_attempt(self) -> int:
            raise SearchAttemptQuotaExceededError("daily key cap exhausted")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=(OPENALEX / "works_page_1.json").read_bytes(),
            request=request,
        )

    async def run() -> object:
        provider, _, _, client = await _capture(
            tmp_path,
            dependency="openalex",
            handler=handler,
            attempt_gate=ExhaustedGate(),
        )
        async with client:
            return await provider.search("RAG", {}, 1, _reservation())

    result = asyncio.run(run())

    assert requests == []
    assert result.usage.search_api_calls == 0
    assert [error.code for error in result.errors] == ["budget_exhausted"]


def test_semantic_scholar_live_requests_are_paced_after_each_attempt(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        events.append("request")
        return httpx.Response(
            200,
            content=(S2 / "search.json").read_bytes(),
            request=request,
        )

    async def paced_sleep(delay: float) -> None:
        events.append(f"sleep:{delay}")

    async def run() -> None:
        provider, _, _, client = await _capture(
            tmp_path,
            dependency="semantic_scholar",
            handler=handler,
            sleep=paced_sleep,
            minimum_request_interval_seconds=1.1,
        )
        async with client:
            await provider.search("graph retrieval", {}, 2, _reservation())
            await provider.search("citation retrieval", {}, 2, _reservation())

    asyncio.run(run())

    assert events == [
        "request",
        "sleep:1.1",
        "request",
        "sleep:1.1",
    ]


def test_request_pacing_serializes_concurrent_provider_attempts(tmp_path: Path) -> None:
    requests: list[str] = []
    first_sleep_started = asyncio.Event()
    release_first_sleep = asyncio.Event()
    sleep_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            content=(OPENALEX / "works_page_1.json").read_bytes(),
            request=request,
        )

    async def paced_sleep(_: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            first_sleep_started.set()
            await release_first_sleep.wait()

    async def run() -> None:
        provider, _, _, client = await _capture(
            tmp_path,
            dependency="openalex",
            handler=handler,
            sleep=paced_sleep,
            minimum_request_interval_seconds=0.2,
        )
        async with client:
            tasks = [
                asyncio.create_task(provider.search(query, {}, 1, _reservation()))
                for query in ("graph retrieval", "citation retrieval")
            ]
            await first_sleep_started.wait()
            await asyncio.sleep(0)
            assert len(requests) == 1
            release_first_sleep.set()
            await asyncio.gather(*tasks)

    asyncio.run(run())
    assert len(requests) == 2


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
    refs = json.loads(live_result.provenance["snapshot_refs"])
    expected_hashes = [
        "sha256:"
        + hashlib.sha256((OPENALEX / name).read_bytes()).hexdigest()
        for name in ("works_page_1.json", "works_page_2.json")
    ]
    assert [ref["response_sha256"] for ref in refs] == expected_hashes


def test_openalex_live_continuation_fetches_only_the_frozen_next_page(
    tmp_path: Path,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            content=(OPENALEX / "works_page_2.json").read_bytes(),
            request=request,
        )

    async def run() -> object:
        provider, _, _, client = await _capture(
            tmp_path, dependency="openalex", handler=handler
        )
        async with client:
            return await provider.search_continuation(
                "RAG",
                {},
                cursor="cursor-page-2",
                limit=50,
                reservation=_reservation(),
            )

    result = asyncio.run(run())

    assert len(seen) == 1
    assert seen[0].url.params["cursor"] == "cursor-page-2"
    assert seen[0].url.params["per_page"] == "50"
    assert all("gold" not in key.casefold() for key in seen[0].url.params.keys())
    assert [paper.openalex_id for paper in result.data] == ["W126"]
    assert result.usage.search_api_calls == 1


def test_openalex_live_continuation_rejects_first_page_cursor(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            content=(OPENALEX / "works_page_1.json").read_bytes(),
            request=request,
        )

    async def run() -> None:
        provider, _, _, client = await _capture(
            tmp_path, dependency="openalex", handler=handler
        )
        async with client:
            with pytest.raises(ValueError, match="continuation cursor"):
                await provider.search_continuation(
                    "RAG",
                    {},
                    cursor="*",
                    limit=50,
                    reservation=_reservation(),
                )

    asyncio.run(run())
    assert seen == []


def test_openalex_live_continuation_never_retries_a_failed_page(
    tmp_path: Path,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(500, content=b"{}", request=request)

    async def run() -> object:
        provider, _, _, client = await _capture(
            tmp_path, dependency="openalex", handler=handler
        )
        async with client:
            return await provider.search_continuation(
                "RAG",
                {},
                cursor="cursor-page-2",
                limit=50,
                reservation=_reservation(),
            )

    result = asyncio.run(run())

    assert len(seen) == 1
    assert result.usage.search_api_calls == 1
    assert [error.code for error in result.errors] == ["server_error"]


def test_openalex_semantic_live_capture_and_replay_share_search_mode_identity(
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

    async def run() -> tuple[object, object]:
        live, store, _, client = await _capture(
            tmp_path, dependency="openalex", handler=handler
        )
        filters = {"_search_mode": "semantic", "year_from": 2020}
        async with client:
            captured = await live.search(
                "predicting toxicity from molecular structure",
                filters,
                2,
                _reservation(),
            )
        manifest = store.seal()
        replay = ReplaySearchProvider(
            dependency="openalex",
            reader=_reader(store, snapshot_set_id=manifest.snapshot_set_id),
            clock=lambda: CAPTURED_AT,
        )
        replayed = await replay.search(
            "predicting toxicity from molecular structure",
            filters,
            2,
            _reservation(),
        )
        return captured, replayed

    captured, replayed = asyncio.run(run())

    assert "search" not in seen[0].url.params
    assert seen[0].url.params["search.semantic"] == (
        "predicting toxicity from molecular structure"
    )
    assert "cursor" not in seen[0].url.params
    assert seen[0].url.params["page"] == "1"
    assert captured.data == replayed.data


def test_replay_reports_captured_usage_instead_of_zero(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params["cursor"]
        name = "works_page_1.json" if cursor == "*" else "works_page_2.json"
        return httpx.Response(200, content=(OPENALEX / name).read_bytes(), request=request)

    async def run() -> tuple[object, object]:
        live, store, _, client = await _capture(
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
        return captured, replayed

    captured, replayed = asyncio.run(run())

    assert captured.usage.search_api_calls == 2
    assert captured.usage != UsageActual()
    assert replayed.usage == captured.usage


def test_replay_commits_the_same_budget_usage_as_live(tmp_path: Path) -> None:
    budget = SearchBudget(
        max_search_api_calls=3,
        target_search_api_calls=1,
        max_llm_calls=1,
        target_llm_calls=0,
        max_total_tokens=1,
        max_cost_cny=1.0,
        max_elapsed_seconds=120,
        soft_deadline_seconds=100,
    )
    estimate = UsageEstimate(
        search_api_calls=3,
        cost_cny=Decimal("0.01"),
        elapsed_ms=60_000,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(OPENALEX / "works_page_1.json").read_bytes(),
            request=request,
        )

    async def run() -> tuple[UsageActual, UsageActual, str, str]:
        live_controller = HardBudgetController(
            budget,
            formal_live=True,
            clock=lambda: CAPTURED_AT,
        )
        reservation = live_controller.reserve("provider.search", estimate)
        store = DependencyCaptureStore(
            tmp_path / "snapshot",
            clock=lambda: CAPTURED_AT,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=store,
                pricer=_pricer(),
                controller=live_controller,
                api_key="synthetic-key",
                clock=lambda: CAPTURED_AT,
            )
            await provider.search("RAG", {}, 3, reservation)
        manifest = store.seal()

        replay_controller = HardBudgetController(
            budget,
            formal_live=False,
            clock=lambda: CAPTURED_AT,
        )
        replay_reservation = replay_controller.reserve("provider.search", estimate)
        replay = ReplaySearchProvider(
            dependency="openalex",
            reader=_reader(store, snapshot_set_id=manifest.snapshot_set_id),
            clock=lambda: CAPTURED_AT,
        )
        replayed = await replay.search("RAG", {}, 3, replay_reservation)
        replay_controller.settle(replay_reservation, replayed.usage)
        return (
            live_controller.committed_usage,
            replay_controller.committed_usage,
            live_controller.stop_status(),
            replay_controller.stop_status(),
        )

    live_committed, replay_committed, live_status, replay_status = asyncio.run(run())

    assert live_committed == replay_committed
    assert live_status == replay_status


def test_openalex_rotates_to_next_key_when_quota_is_exhausted(
    tmp_path: Path,
) -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_key = request.url.params["api_key"]
        seen.append((api_key, request.url.params["cursor"]))
        if api_key == "synthetic-key":
            return httpx.Response(
                429,
                headers={"x-ratelimit-remaining": "0"},
                request=request,
            )
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
            additional_api_keys=["fallback-key"],
        )
        async with client:
            result = await live.search("RAG", {}, 3, _reservation())
        return result, store.seal()

    result, manifest = asyncio.run(run())

    assert [api_key for api_key, _ in seen] == [
        "synthetic-key",
        "fallback-key",
        "fallback-key",
    ]
    assert [cursor for _, cursor in seen] == ["*", "*", "cursor-page-2"]
    assert result.errors == []
    captured_hashes = {
        entry.request.canonical_request_sha256 for entry in manifest.entries
    }
    assert _openalex_identity().canonical_request_sha256 in captured_hashes


def test_openalex_rotates_key_when_zero_budget_is_reported_only_in_json(
    tmp_path: Path,
) -> None:
    seen_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_key = request.url.params["api_key"]
        seen_keys.append(api_key)
        if api_key == "synthetic-key":
            return httpx.Response(
                429,
                json={
                    "error": "Rate limit exceeded",
                    "message": "Insufficient budget.",
                    "dailyRemainingUsd": 0,
                    "creditsRemaining": 0,
                },
                request=request,
            )
        return httpx.Response(
            200,
            content=(OPENALEX / "works_page_1.json").read_bytes(),
            request=request,
        )

    async def run() -> object:
        live, _, _, client = await _capture(
            tmp_path,
            dependency="openalex",
            handler=handler,
            additional_api_keys=["fallback-key"],
        )
        async with client:
            return await live.search("RAG", {}, 1, _reservation())

    result = asyncio.run(run())

    assert seen_keys == ["synthetic-key", "fallback-key"]
    assert result.errors == []


def test_openalex_single_key_stops_immediately_when_daily_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            429,
            json={
                "message": "Insufficient budget.",
                "dailyRemainingUsd": 0,
                "prepaidRemainingUsd": 0,
            },
            request=request,
        )

    async def run() -> tuple[object, object]:
        live, _, _, client = await _capture(
            tmp_path,
            dependency="openalex",
            handler=handler,
        )
        async with client:
            first = await live.search("RAG", {}, 1, _reservation())
            second = await live.search("graph retrieval", {}, 1, _reservation())
        return first, second

    first, second = asyncio.run(run())

    assert len(requests) == 1
    assert [error.code for error in first.errors] == ["quota_exhausted"]
    assert [error.code for error in second.errors] == ["quota_exhausted"]


def test_openalex_can_rotate_past_three_exhausted_keys(tmp_path: Path) -> None:
    seen_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_key = request.url.params["api_key"]
        seen_keys.append(api_key)
        if api_key != "funded-key":
            return httpx.Response(
                429,
                json={
                    "message": "Insufficient budget.",
                    "dailyRemainingUsd": 0,
                },
                request=request,
            )
        return httpx.Response(
            200,
            content=(OPENALEX / "works_page_1.json").read_bytes(),
            request=request,
        )

    async def run() -> object:
        live, _, _, client = await _capture(
            tmp_path,
            dependency="openalex",
            handler=handler,
            additional_api_keys=["empty-2", "empty-3", "funded-key"],
        )
        async with client:
            return await live.search("RAG", {}, 1, _reservation(4))

    result = asyncio.run(run())

    assert seen_keys == ["synthetic-key", "empty-2", "empty-3", "funded-key"]
    assert result.errors == []


@pytest.mark.parametrize(
    ("remaining", "read_timeout", "expected"),
    [
        (120.0, 20.0, 20.0),
        (5.0, 20.0, 5.0),
        (120.0, None, 120.0),
        (120.0, 0.0, 120.0),
    ],
)
def test_attempt_timeout_is_bounded_by_read_timeout(
    remaining: float,
    read_timeout: float | None,
    expected: float,
) -> None:
    assert _attempt_timeout(remaining, read_timeout) == expected


def test_openalex_does_not_rotate_on_transient_rate_limit(
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_key = request.url.params["api_key"]
        seen.append(api_key)
        if len(seen) == 1:
            return httpx.Response(
                429,
                headers={"x-ratelimit-remaining": "9950"},
                request=request,
            )
        return httpx.Response(
            200,
            content=(OPENALEX / "works_page_1.json").read_bytes(),
            request=request,
        )

    async def run() -> object:
        live, _, _, client = await _capture(
            tmp_path,
            dependency="openalex",
            handler=handler,
            additional_api_keys=["fallback-key"],
        )
        async with client:
            return await live.search("RAG", {}, 3, _reservation())

    result = asyncio.run(run())

    assert seen == ["synthetic-key", "synthetic-key", "synthetic-key"]
    assert result.errors == []


def test_openalex_rotates_when_remaining_credits_cannot_afford_search(
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_key = request.url.params["api_key"]
        seen.append(api_key)
        if api_key == "synthetic-key":
            return httpx.Response(
                429,
                headers={"x-ratelimit-remaining": "6"},
                request=request,
            )
        return httpx.Response(
            200,
            content=(OPENALEX / "works_page_1.json").read_bytes(),
            request=request,
        )

    async def run() -> object:
        live, _, _, client = await _capture(
            tmp_path,
            dependency="openalex",
            handler=handler,
            additional_api_keys=["fallback-key"],
        )
        async with client:
            return await live.search("RAG", {}, 3, _reservation())

    result = asyncio.run(run())

    assert seen[:2] == ["synthetic-key", "fallback-key"]
    assert result.errors == []


def test_openalex_key_rotation_persists_for_followup_requests(
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_key = request.url.params["api_key"]
        seen.append(api_key)
        if api_key == "synthetic-key":
            return httpx.Response(
                429,
                headers={"x-ratelimit-remaining": "0"},
                request=request,
            )
        return httpx.Response(
            200,
            content=(OPENALEX / "works_page_1.json").read_bytes(),
            request=request,
        )

    async def run() -> object:
        live, _, _, client = await _capture(
            tmp_path,
            dependency="openalex",
            handler=handler,
            additional_api_keys=["fallback-key"],
        )
        async with client:
            await live.search("RAG", {}, 3, _reservation())
            await live.search("graph retrieval", {}, 3, _reservation())

    asyncio.run(run())

    assert seen == [
        "synthetic-key",
        "fallback-key",
        "fallback-key",
        "fallback-key",
        "fallback-key",
    ]


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
    assert replayed.usage == captured.usage
    assert replayed.usage != UsageActual()


def test_semantic_scholar_semantic_mode_live_and_replay_share_identity(
    tmp_path: Path,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
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
        filters = {"_search_mode": "semantic", "year_from": 2020}
        async with client:
            captured = await live.search("graph retrieval", filters, 2, _reservation())
        manifest = store.seal()
        replay = ReplaySearchProvider(
            dependency="semantic_scholar",
            reader=_reader(store, snapshot_set_id=manifest.snapshot_set_id),
            clock=lambda: CAPTURED_AT,
        )
        replayed = await replay.search(
            "graph retrieval",
            filters,
            2,
            _reservation(0),
        )
        return captured, replayed

    captured, replayed = asyncio.run(run())

    assert seen[0].url.params["year"] == "2020-"
    assert "_search_mode" not in seen[0].url.params
    assert captured.data == replayed.data
    assert captured.provenance["search_mode"] == "semantic"
    assert replayed.provenance["search_mode"] == "semantic"


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


def test_semantic_scholar_rate_limit_uses_extended_backoff(tmp_path: Path) -> None:
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=b"rate limited", request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def run() -> None:
        live, _, _, client = await _capture(
            tmp_path,
            dependency="semantic_scholar",
            handler=handler,
            sleep=fake_sleep,
        )
        async with client:
            await live.search("graph", {}, 2, _reservation())

    asyncio.run(run())

    assert sleeps == [60.0, 60.0]


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


def test_openalex_references_fetches_seed_edges_then_batches_works(
    tmp_path: Path,
) -> None:
    paper_id = ProviderPaperId(provider="openalex", value="W1000000001")
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/works/W1000000001":
            return httpx.Response(
                200,
                json={
                    "id": "https://openalex.org/W1000000001",
                    "referenced_works": [
                        "https://openalex.org/W2000000002",
                        "https://openalex.org/W3000000003",
                    ],
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "meta": {"next_cursor": None},
                "results": [
                    {
                        "id": "https://openalex.org/W2000000002",
                        "title": "First cited work",
                    },
                    {
                        "id": "https://openalex.org/W3000000003",
                        "title": "Second cited work",
                    },
                ],
            },
            request=request,
        )

    async def run() -> object:
        live, _, _, client = await _capture(
            tmp_path, dependency="openalex", handler=handler
        )
        async with client:
            return await live.references(paper_id, 2, _reservation(2))

    result = asyncio.run(run())

    assert paths == ["/works/W1000000001", "/works"]
    assert [paper.openalex_id for paper in result.data.papers] == [
        "W2000000002",
        "W3000000003",
    ]
    assert {
        (edge.citing_provider_id.value, edge.cited_provider_id.value)
        for edge in result.data.raw_edges
    } == {
        ("W1000000001", "W2000000002"),
        ("W1000000001", "W3000000003"),
    }
    assert result.usage.search_api_calls == 2


def test_openalex_citations_filters_for_works_that_cite_the_seed(
    tmp_path: Path,
) -> None:
    paper_id = ProviderPaperId(provider="openalex", value="W1000000001")
    filters: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        filters.append(request.url.params.get("filter"))
        return httpx.Response(
            200,
            json={
                "meta": {"next_cursor": None},
                "results": [
                    {
                        "id": "https://openalex.org/W4000000004",
                        "title": "A citing work",
                    }
                ],
            },
            request=request,
        )

    async def run() -> object:
        live, _, _, client = await _capture(
            tmp_path, dependency="openalex", handler=handler
        )
        async with client:
            return await live.citations(paper_id, 1, _reservation(1))

    result = asyncio.run(run())

    assert filters == ["cites:W1000000001"]
    assert [paper.openalex_id for paper in result.data.papers] == ["W4000000004"]
    assert [
        (edge.citing_provider_id.value, edge.cited_provider_id.value)
        for edge in result.data.raw_edges
    ] == [("W4000000004", "W1000000001")]
    assert result.usage.search_api_calls == 1


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
            expected_type = "OSError" if fault == "capture" else "RuntimeError"
            with pytest.raises(
                ProviderAdapterError,
                match=rf"provider live capture failed \({expected_type}\)",
            ):
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
        update={"expires_at": datetime.now(UTC) + timedelta(milliseconds=100)}
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
        update={"expires_at": now + timedelta(milliseconds=100)}
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
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
