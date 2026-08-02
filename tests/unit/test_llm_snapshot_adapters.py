from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from paper_search.control.pricing import ActualCostPricer, load_pricing_policy
from paper_search.control.budget import HardBudgetController, ReservationError
from paper_search.domain.models import (
    BudgetReservation,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.llm.client import LLMResponseDecoder, OpenAICompatibleLLMClient
from paper_search.llm.snapshot_adapters import (
    HardBudgetSettlementAdapter,
    LLMAdapterError,
    LiveCaptureLLMAnalyzer,
    ReplayLLMAnalyzer,
)
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencySnapshotReader,
)


CAPTURED_AT = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
PRICING_FIXTURE = Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")


def _reservation() -> BudgetReservation:
    return BudgetReservation(
        reservation_id="llm-live-1",
        action="query.analyze",
        reserved=UsageEstimate(
            llm_calls=3,
            input_tokens=100,
            output_tokens=100,
            cost_cny=Decimal("0.30"),
        ),
        expires_at=CAPTURED_AT + timedelta(minutes=1),
    )


def _response_bytes(
    data: dict[str, object], *, input_tokens: int = 5, output_tokens: int = 3
) -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"content": json.dumps(data)}}],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            },
        },
        separators=(",", ":"),
    ).encode()


class SettlementRecorder:
    def __init__(self) -> None:
        self.settled: list[tuple[BudgetReservation, UsageActual]] = []
        self.failed: list[tuple[BudgetReservation, UsageActual]] = []

    def settle(self, reservation: BudgetReservation, actual: UsageActual) -> None:
        self.settled.append((reservation, actual))

    def fail_closed(
        self, reservation: BudgetReservation, actual: UsageActual
    ) -> None:
        self.failed.append((reservation, actual))


def _pricer() -> ActualCostPricer:
    return ActualCostPricer(
        load_pricing_policy(PRICING_FIXTURE), valued_at=CAPTURED_AT
    )


async def _live(
    tmp_path: Path,
    transport: httpx.MockTransport,
    controller: object,
    reservation: BudgetReservation | None = None,
) -> tuple[object, DependencyCaptureStore]:
    store = DependencyCaptureStore(tmp_path / "snapshot", clock=lambda: CAPTURED_AT)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenAICompatibleLLMClient(
            client=http_client,
            base_url="https://llm.example.test/v1",
            model="qwen-test-v1",
            api_key="unit-test-secret",
            clock=lambda: CAPTURED_AT,
        )
        analyzer = LiveCaptureLLMAnalyzer(
            client=client,
            capture_store=store,
            pricer=_pricer(),
            controller=controller,  # type: ignore[arg-type]
            clock=lambda: CAPTURED_AT,
        )
        result = await analyzer.generate_json(
            prompt_name="query_analyze",
            payload={"query": "graph retrieval"},
            reservation=reservation or _reservation(),
        )
    return result, store


def _budget_controller(*, formal_live: bool = True) -> HardBudgetController:
    return HardBudgetController(
        SearchBudget(
            max_llm_calls=10,
            target_llm_calls=3,
            max_total_tokens=1_000,
            max_cost_cny=1.0,
            max_elapsed_seconds=120,
            soft_deadline_seconds=110,
        ),
        formal_live=formal_live,
        clock=lambda: CAPTURED_AT,
    )


def _controller_reservation(controller: HardBudgetController) -> BudgetReservation:
    return controller.reserve(
        "query.analyze",
        UsageEstimate(
            llm_calls=3,
            input_tokens=100,
            output_tokens=100,
            cost_cny=Decimal("0.30"),
            elapsed_ms=10_000,
        ),
    )


def test_real_budget_controller_success_settles_once(tmp_path: Path) -> None:
    controller = _budget_controller()
    settlement = HardBudgetSettlementAdapter(controller)
    reservation = _controller_reservation(controller)

    result, _ = asyncio.run(
        _live(
            tmp_path,
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=_response_bytes({"ok": True}), request=request
                )
            ),
            settlement,
            reservation,
        )
    )

    assert controller.committed_usage == result.usage
    assert settlement.committed_usage == result.usage
    assert controller.reserved_usage.llm_calls == 0
    with pytest.raises(ReservationError, match="reservation is unknown"):
        settlement.settle(reservation, result.usage)


def test_real_budget_controller_terminal_failure_records_usage_and_hard_stops(
    tmp_path: Path,
) -> None:
    class BrokenPricer:
        def value_actual(self, **_: object) -> UsageActual:
            raise ValueError("sensitive pricing failure")

    controller = _budget_controller()
    settlement = HardBudgetSettlementAdapter(controller)
    reservation = _controller_reservation(controller)
    store = DependencyCaptureStore(tmp_path / "snapshot")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=_response_bytes({"ok": True}), request=request
                )
            )
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://llm.example.test/v1",
                    model="qwen-test-v1",
                    api_key="unit-test-secret",
                ),
                capture_store=store,
                pricer=BrokenPricer(),  # type: ignore[arg-type]
                controller=settlement,
            )
            with pytest.raises(LLMAdapterError, match="live capture failed"):
                await analyzer.generate_json(
                    prompt_name="query_analyze",
                    payload={"query": "x"},
                    reservation=reservation,
                )

    asyncio.run(run())

    assert controller.stop_status() == "hard_stop"
    assert controller.reserved_usage.llm_calls == 0
    assert settlement.committed_usage.llm_calls == 1
    with pytest.raises(ReservationError, match="reservation is unknown"):
        settlement.fail_closed(reservation, UsageActual(llm_calls=1))


def test_cancellation_after_dispatch_fail_closes_then_reraises_cancelled_error(
    tmp_path: Path,
) -> None:
    controller = _budget_controller()
    settlement = HardBudgetSettlementAdapter(controller)
    reservation = _controller_reservation(controller)

    async def run() -> None:
        dispatched = asyncio.Event()
        never_complete = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            dispatched.set()
            await never_complete.wait()
            return httpx.Response(200, content=_response_bytes({}), request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://llm.example.test/v1",
                    model="qwen-test-v1",
                    api_key="unit-test-secret",
                ),
                capture_store=DependencyCaptureStore(tmp_path / "cancelled"),
                pricer=_pricer(),
                controller=settlement,
            )
            task = asyncio.create_task(
                analyzer.generate_json(
                    prompt_name="query_analyze",
                    payload={"query": "x"},
                    reservation=reservation,
                )
            )
            await dispatched.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())

    assert controller.stop_status() == "hard_stop"
    assert controller.reserved_usage.llm_calls == 0
    assert settlement.committed_usage.llm_calls == 1


def test_unexpected_exception_after_dispatch_commits_conservative_usage(
    tmp_path: Path,
) -> None:
    controller = _budget_controller()
    reservation = _controller_reservation(controller)

    def handler(_: httpx.Request) -> httpx.Response:
        raise RuntimeError("sensitive response boundary failure")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://llm.example.test/v1",
                    model="qwen-test-v1",
                    api_key="unit-test-secret",
                ),
                capture_store=DependencyCaptureStore(tmp_path / "unexpected"),
                pricer=_pricer(),
                controller=HardBudgetSettlementAdapter(controller),
            )
            with pytest.raises(LLMAdapterError) as error:
                await analyzer.generate_json(
                    prompt_name="query_analyze",
                    payload={"query": "x"},
                    reservation=reservation,
                )
            assert str(error.value) == "LLM live capture failed"
            assert "sensitive" not in repr(error.value)

    asyncio.run(run())

    assert controller.stop_status() == "hard_stop"
    assert controller.reserved_usage == UsageEstimate()
    assert controller.committed_usage.llm_calls == 1
    assert controller.committed_usage.cost_cny is None
    assert controller.unknown_cost_actions == ["query.analyze"]


def test_usage_measurement_failure_after_response_commits_one_unknown_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _budget_controller()
    settlement = HardBudgetSettlementAdapter(controller)
    reservation = _controller_reservation(controller)

    def fail_measurement(_: bytes) -> UsageActual:
        raise RecursionError("sensitive nested response failure")

    monkeypatch.setattr(
        "paper_search.llm.snapshot_adapters.usage_from_response_bytes",
        fail_measurement,
    )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=_response_bytes({"ok": True}), request=request
                )
            )
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://llm.example.test/v1",
                    model="qwen-test-v1",
                    api_key="unit-test-secret",
                ),
                capture_store=DependencyCaptureStore(tmp_path / "measurement-failure"),
                pricer=_pricer(),
                controller=settlement,
            )
            with pytest.raises(LLMAdapterError) as error:
                await analyzer.generate_json(
                    prompt_name="query_analyze",
                    payload={"query": "x"},
                    reservation=reservation,
                )
            assert str(error.value) == "LLM live capture failed"
            assert error.value.__cause__ is None
            assert "sensitive" not in repr(error.value)

    asyncio.run(run())

    assert controller.stop_status() == "hard_stop"
    assert controller.reserved_usage == UsageEstimate()
    assert controller.committed_usage.llm_calls == 1
    assert controller.committed_usage.cost_cny is None
    assert controller.unknown_cost_actions == ["query.analyze"]
    with pytest.raises(ReservationError, match="reservation is unknown"):
        settlement.fail_closed(reservation, UsageActual(llm_calls=1))


def test_live_capture_builds_safe_identity_and_stages_exact_success_bytes(
    tmp_path: Path,
) -> None:
    raw = _response_bytes({"ok": True})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw, request=request)

    controller = SettlementRecorder()
    result, store = asyncio.run(
        _live(tmp_path, httpx.MockTransport(handler), controller)
    )
    manifest = store.seal()
    entry = manifest.entries[0]

    assert result.data == {"ok": True}
    assert (store.root / entry.response_path).read_bytes() == raw
    assert entry.request.dependency == "llm"
    assert entry.request.endpoint == "/chat/completions"
    assert entry.request.model_or_adapter == "qwen-test-v1"
    assert "secret" not in entry.model_dump_json().casefold()
    assert len(controller.settled) == 1


def test_replay_uses_identical_decoder_zero_cost_and_snapshot_provenance(
    tmp_path: Path,
) -> None:
    raw = _response_bytes({"ok": True})
    controller = SettlementRecorder()
    live_result, store = asyncio.run(
        _live(
            tmp_path,
            httpx.MockTransport(
                lambda request: httpx.Response(200, content=raw, request=request)
            ),
            controller,
        )
    )
    manifest = store.seal()
    reader = DependencySnapshotReader(
        store.manifest_path,
        snapshot_manifest_sha256=store.manifest_sha256,
        snapshot_set_id=manifest.snapshot_set_id,
    )
    replay = ReplayLLMAnalyzer(
        reader=reader,
        model_id="qwen-test-v1",
        clock=lambda: CAPTURED_AT,
    )

    replay_result = asyncio.run(
        replay.generate_json(
            prompt_name="query_analyze",
            payload={"query": "graph retrieval"},
            reservation=_reservation(),
        )
    )

    assert replay_result.data == live_result.data
    assert replay_result.cache_hit is True
    assert replay_result.usage == UsageActual()
    assert replay_result.provenance["snapshot_set_id"] == manifest.snapshot_set_id
    assert "snapshot_entry_id" in replay_result.provenance


def test_replay_miss_is_snapshot_unavailable_without_network_fallback(
    tmp_path: Path,
) -> None:
    store = DependencyCaptureStore(tmp_path / "empty", clock=lambda: CAPTURED_AT)
    manifest = store.seal()
    reader = DependencySnapshotReader(
        store.manifest_path,
        snapshot_manifest_sha256=store.manifest_sha256,
    )
    replay = ReplayLLMAnalyzer(
        reader=reader,
        model_id="qwen-test-v1",
        clock=lambda: CAPTURED_AT,
    )

    result = asyncio.run(
        replay.generate_json(
            prompt_name="query_analyze",
            payload={"query": "missing"},
            reservation=_reservation(),
        )
    )

    assert manifest.entries == []
    assert result.errors[0].code == "snapshot_unavailable"
    assert result.usage == UsageActual()


def test_replay_rejects_secret_shaped_model_identifier(tmp_path: Path) -> None:
    store = DependencyCaptureStore(tmp_path / "empty-model-check")
    store.seal()
    reader = DependencySnapshotReader(
        store.manifest_path,
        snapshot_manifest_sha256=store.manifest_sha256,
    )

    with pytest.raises(ValueError, match="model identifier is not safe"):
        ReplayLLMAnalyzer(reader=reader, model_id="sk-live-replay-secret")


def test_injected_decoder_cannot_bypass_replay_prompt_version_validation(
    tmp_path: Path,
) -> None:
    store = DependencyCaptureStore(tmp_path / "empty-prompt-check")
    store.seal()
    reader = DependencySnapshotReader(
        store.manifest_path,
        snapshot_manifest_sha256=store.manifest_sha256,
    )
    secret_prompt_version = "query-sk-live-replay-secret"

    with pytest.raises(ValueError, match="prompt version is not safe") as error:
        ReplayLLMAnalyzer(
            reader=reader,
            model_id="qwen-test-v1",
            prompt_version=secret_prompt_version,
            decoder=LLMResponseDecoder(prompt_version="query-analyze-v1"),
        )

    assert secret_prompt_version not in str(error.value)


@pytest.mark.parametrize(
    ("status_code", "expected_code", "expected_calls"),
    [(401, "authentication_error", 1), (400, "invalid_request", 1)],
)
def test_nonretryable_http_errors_are_sanitized_and_not_captured(
    tmp_path: Path,
    status_code: int,
    expected_code: str,
    expected_calls: int,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status_code,
            content=b'{"error":"Bearer top-secret"}',
            headers={"x-request-id": "sk-live-request-secret"},
            request=request,
        )

    controller = SettlementRecorder()
    result, store = asyncio.run(
        _live(tmp_path, httpx.MockTransport(handler), controller)
    )

    assert calls == expected_calls
    assert result.errors[0].code == expected_code
    assert "top-secret" not in result.model_dump_json()
    assert "sk-live-request-secret" not in result.model_dump_json()
    assert not (store.root / "responses").exists()
    assert len(controller.settled) == 1


def test_timeout_and_429_are_retried_at_most_three_total_attempts(
    tmp_path: Path,
) -> None:
    attempts = 0
    raw = _response_bytes({"ok": True}, input_tokens=5, output_tokens=3)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("secret timeout detail", request=request)
        if attempts == 2:
            return httpx.Response(
                429,
                content=b'{"usage":{"prompt_tokens":1,"completion_tokens":0}}',
                request=request,
            )
        return httpx.Response(200, content=raw, request=request)

    controller = SettlementRecorder()
    result, _ = asyncio.run(_live(tmp_path, httpx.MockTransport(handler), controller))

    assert attempts == 3
    assert result.data == {"ok": True}
    assert result.usage.llm_calls == 3
    assert result.usage.input_tokens == 6
    assert result.usage.output_tokens == 3
    assert result.usage.cost_cny == Decimal("0.000321")
    assert controller.settled == [(_reservation(), result.usage)]


def test_three_timeouts_return_fixed_error_and_settle_once(tmp_path: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("never expose me", request=request)

    controller = SettlementRecorder()
    result, _ = asyncio.run(_live(tmp_path, httpx.MockTransport(handler), controller))

    assert attempts == 3
    assert result.errors[0].code == "timeout"
    assert "never expose me" not in result.model_dump_json()
    assert len(controller.settled) == 1
    assert controller.settled[0][1].llm_calls == 3


def test_staging_failure_commits_all_valued_attempts_to_real_controller(
    tmp_path: Path,
) -> None:
    class BrokenCaptureStore:
        def stage_success(self, *_: object, **__: object) -> object:
            raise OSError("sensitive staging failure")

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                content=b'{"usage":{"prompt_tokens":1,"completion_tokens":0}}',
                request=request,
            )
        return httpx.Response(
            200,
            content=_response_bytes({"ok": True}, input_tokens=5, output_tokens=3),
            request=request,
        )

    controller = _budget_controller()
    settlement = HardBudgetSettlementAdapter(controller)
    reservation = _controller_reservation(controller)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://llm.example.test/v1",
                    model="qwen-test-v1",
                    api_key="unit-test-secret",
                ),
                capture_store=BrokenCaptureStore(),  # type: ignore[arg-type]
                pricer=_pricer(),
                controller=settlement,
            )
            with pytest.raises(LLMAdapterError, match="live capture failed") as error:
                await analyzer.generate_json(
                    prompt_name="query_analyze",
                    payload={"query": "x"},
                    reservation=reservation,
                )
            assert "sensitive" not in str(error.value)

    asyncio.run(run())

    assert attempts == 2
    assert controller.stop_status() == "hard_stop"
    assert controller.reserved_usage == UsageEstimate()
    assert controller.committed_usage.llm_calls == 2
    assert controller.committed_usage.input_tokens == 6
    assert controller.committed_usage.output_tokens == 3
    assert controller.committed_usage.cost_cny == Decimal("0.000221")


def test_pricing_failure_preserves_prior_valued_attempts_in_real_controller(
    tmp_path: Path,
) -> None:
    delegate = _pricer()

    class SecondAttemptPricingFailure:
        attempts = 0

        def value_actual(self, **kwargs: object) -> UsageActual:
            self.attempts += 1
            if self.attempts == 2:
                raise ValueError("sensitive pricing failure")
            return delegate.value_actual(**kwargs)  # type: ignore[arg-type]

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                content=b'{"usage":{"prompt_tokens":1,"completion_tokens":0}}',
                request=request,
            )
        return httpx.Response(
            200,
            content=_response_bytes({"ok": True}, input_tokens=5, output_tokens=3),
            request=request,
        )

    controller = _budget_controller()
    reservation = _controller_reservation(controller)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://llm.example.test/v1",
                    model="qwen-test-v1",
                    api_key="unit-test-secret",
                ),
                capture_store=DependencyCaptureStore(tmp_path / "partial-pricing"),
                pricer=SecondAttemptPricingFailure(),  # type: ignore[arg-type]
                controller=HardBudgetSettlementAdapter(controller),
            )
            with pytest.raises(LLMAdapterError, match="live capture failed"):
                await analyzer.generate_json(
                    prompt_name="query_analyze",
                    payload={"query": "x"},
                    reservation=reservation,
                )

    asyncio.run(run())

    assert attempts == 2
    assert controller.committed_usage.llm_calls == 2
    assert controller.committed_usage.input_tokens == 6
    assert controller.committed_usage.output_tokens == 3
    assert controller.committed_usage.cost_cny is None
    assert controller.known_committed_cost_cny == Decimal("0.000102")
    assert controller.unknown_cost_actions == ["query.analyze"]


def test_terminal_internal_failure_records_accumulated_usage_fail_closed(
    tmp_path: Path,
) -> None:
    class BrokenPricer:
        def value_actual(self, **_: object) -> UsageActual:
            raise ValueError("sensitive pricing failure")

    controller = SettlementRecorder()
    store = DependencyCaptureStore(tmp_path / "snapshot")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=_response_bytes({"ok": True}), request=request
                )
            )
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://llm.example.test/v1",
                    model="qwen-test-v1",
                    api_key="unit-test-secret",
                ),
                capture_store=store,
                pricer=BrokenPricer(),  # type: ignore[arg-type]
                controller=controller,
            )
            with pytest.raises(LLMAdapterError, match="live capture failed") as error:
                await analyzer.generate_json(
                    prompt_name="query_analyze",
                    payload={"query": "x"},
                    reservation=_reservation(),
                )
            assert "sensitive" not in str(error.value)

    asyncio.run(run())
    assert controller.settled == []
    assert len(controller.failed) == 1
    assert controller.failed[0][1].llm_calls == 1


@pytest.mark.parametrize("settlement_error", [ReservationError, TypeError])
def test_terminal_settlement_failure_is_fixed_no_chain_adapter_error(
    tmp_path: Path,
    settlement_error: type[Exception],
) -> None:
    class BrokenPricer:
        def value_actual(self, **_: object) -> UsageActual:
            raise ValueError("sensitive pricing failure")

    class FailingSettlementController:
        def settle(self, *_: object) -> None:
            raise AssertionError("settle must not be called")

        def fail_closed(self, *_: object) -> None:
            raise settlement_error("sensitive settlement failure")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=_response_bytes({"ok": True}), request=request
                )
            )
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://llm.example.test/v1",
                    model="qwen-test-v1",
                    api_key="unit-test-secret",
                ),
                capture_store=DependencyCaptureStore(tmp_path / "terminal-mask"),
                pricer=BrokenPricer(),  # type: ignore[arg-type]
                controller=FailingSettlementController(),  # type: ignore[arg-type]
            )
            with pytest.raises(LLMAdapterError) as error:
                await analyzer.generate_json(
                    prompt_name="query_analyze",
                    payload={"query": "x"},
                    reservation=_reservation(),
                )
            assert str(error.value) == "LLM live capture failed"
            assert error.value.__cause__ is None
            assert "sensitive" not in repr(error.value)

    asyncio.run(run())


@pytest.mark.parametrize("settlement_error", [ReservationError, TypeError])
def test_cancellation_preserves_cancelled_error_when_terminal_settlement_fails(
    tmp_path: Path,
    settlement_error: type[Exception],
) -> None:
    class FailingSettlementController:
        def settle(self, *_: object) -> None:
            raise AssertionError("settle must not be called")

        def fail_closed(self, *_: object) -> None:
            raise settlement_error("sensitive settlement failure")

    async def run() -> None:
        dispatched = asyncio.Event()
        never_complete = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            dispatched.set()
            await never_complete.wait()
            return httpx.Response(200, content=_response_bytes({}), request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://llm.example.test/v1",
                    model="qwen-test-v1",
                    api_key="unit-test-secret",
                ),
                capture_store=DependencyCaptureStore(tmp_path / "cancel-mask"),
                pricer=_pricer(),
                controller=FailingSettlementController(),  # type: ignore[arg-type]
            )
            task = asyncio.create_task(
                analyzer.generate_json(
                    prompt_name="query_analyze",
                    payload={"query": "x"},
                    reservation=_reservation(),
                )
            )
            await dispatched.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as error:
                await task
            assert error.value.__cause__ is None
            assert "sensitive" not in repr(error.value)

    asyncio.run(run())
