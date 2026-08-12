"""Contract tests for reservation-owning LLM recall generation backends."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from paper_search.control.budget import HardBudgetController
from paper_search.control.pricing import ActualCostPricer, load_pricing_policy
from paper_search.domain.models import ErrorDetail, ProviderResult, SearchBudget, UsageActual, UsageEstimate
from paper_search.recall_experiments.generation.backends import (
    BudgetedLLMBackend,
    LLMGenerationRequest,
)
from paper_search.recall_experiments.identity import (
    LiveDependencyIdentity,
    validate_scheme_b_dependency_identity,
)
from paper_search.llm.client import OpenAICompatibleLLMClient
from paper_search.llm.snapshot_adapters import LiveCaptureLLMAnalyzer
from paper_search.storage.dependency_snapshot import DependencyCaptureStore


def _budget() -> SearchBudget:
    return SearchBudget(
        max_search_api_calls=10,
        max_llm_calls=10,
        max_total_tokens=10_000,
        max_cost_cny=Decimal("10.00"),
        max_elapsed_seconds=60,
        soft_deadline_seconds=30,
        max_citation_seeds=2,
    )


def _result(*, errors: list[ErrorDetail] | None = None) -> ProviderResult[dict[str, object]]:
    return ProviderResult(
        data={"actions": []},
        usage=UsageActual(
            llm_calls=1,
            input_tokens=9,
            output_tokens=4,
            elapsed_ms=8,
            cost_cny=Decimal("0.01"),
        ),
        provenance={
            "provider": "llm",
            "endpoint": "/chat/completions",
            "model_id": "fake-model",
            "requested_at": datetime.now(UTC).isoformat(),
            "response_hash": "sha256:test",
        },
        cache_hit=True,
        latency_ms=8,
        errors=errors or [],
    )


class _FakeAnalyzer:
    def __init__(self, results: list[ProviderResult[dict[str, object]]]) -> None:
        self.results = iter(results)
        self.reservations = []

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: object,
    ) -> ProviderResult[dict[str, object]]:
        assert prompt_name == "recall_actions"
        assert payload == {"query": "offline"}
        self.reservations.append(reservation)
        return next(self.results)


class _CancellingAnalyzer:
    def __init__(self) -> None:
        self.reservation: object | None = None

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: object,
    ) -> ProviderResult[dict[str, object]]:
        self.reservation = reservation
        raise asyncio.CancelledError()


def _estimate() -> UsageEstimate:
    return UsageEstimate(
        llm_calls=1,
        input_tokens=10,
        output_tokens=10,
        elapsed_ms=20,
        cost_cny=Decimal("0.02"),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "openai_compatible"},
        {"adapter": "evil-json"},
        {"version": "openai-compatible-client-v2"},
        {"model": "unexpected-model"},
        {"endpoints": ("https://api.deepseek.com/chat/completions",)},
        {"operations": ("generate_json", "exfiltrate")},
    ],
)
def test_scheme_b_llm_identity_rejects_every_unadmitted_surface_change(
    changes: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "identity_schema_version": "live-dependency-runtime-identity-v1",
        "provider": "deepseek",
        "dependency": "llm",
        "adapter": "openai-compatible-json",
        "model": "deepseek-v4-flash",
        "version": "openai-compatible-client-v1",
        "endpoints": ("https://api.deepseek.com/v1/chat/completions",),
        "operations": ("generate_json",),
        "pricing_policy_sha256": "sha256:" + "a" * 64,
        "controller_policy_sha256": "sha256:" + "b" * 64,
    }
    payload.update(changes)
    identity = LiveDependencyIdentity.model_validate(payload)

    with pytest.raises(ValueError, match="LLM surface"):
        validate_scheme_b_dependency_identity("llm", identity)


def test_live_llm_backend_binds_analyzer_identity_and_exact_objects(
    tmp_path: Path,
) -> None:
    controller = HardBudgetController(_budget(), formal_live=True)
    pricer = ActualCostPricer(
        load_pricing_policy(Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")),
        valued_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://api.deepseek.com/v1",
                    model="deepseek-v4-flash",
                    api_key="private-key",
                ),
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=pricer,
                controller=controller,
                prompt_artifact_sha256="sha256:" + "a" * 64,
            )
            backend = BudgetedLLMBackend(
                analyzer=analyzer,
                controller=controller,
                initial_estimate=_estimate(),
                repair_estimate=_estimate(),
            )
            assert backend.dependency_identity.dependency == "llm"
            assert backend.dependency_identity.provider == "deepseek"
            assert backend.dependency_identity.model == "deepseek-v4-flash"
            assert backend.live_pricer is pricer
            assert backend.live_controller is controller

    asyncio.run(run())


def test_live_llm_backend_rejects_mismatched_controller(tmp_path: Path) -> None:
    controller = HardBudgetController(_budget(), formal_live=True)
    other_controller = HardBudgetController(_budget(), formal_live=True)
    pricer = ActualCostPricer(
        load_pricing_policy(Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")),
        valued_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://api.deepseek.com",
                    model="deepseek-test-v1",
                    api_key="private-key",
                ),
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=pricer,
                controller=controller,
                prompt_artifact_sha256="sha256:" + "a" * 64,
            )
            with pytest.raises(ValueError, match="controller"):
                BudgetedLLMBackend(
                    analyzer=analyzer,
                    controller=other_controller,
                    initial_estimate=_estimate(),
                    repair_estimate=_estimate(),
                )

    asyncio.run(run())


def test_live_llm_backend_rejects_unadmitted_model_surface(tmp_path: Path) -> None:
    controller = HardBudgetController(_budget(), formal_live=True)
    pricer = ActualCostPricer(
        load_pricing_policy(Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")),
        valued_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
        ) as http_client:
            analyzer = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://api.deepseek.com/v1",
                    model="deepseek-test-v1",
                    api_key="private-key",
                ),
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=pricer,
                controller=controller,
                prompt_artifact_sha256="sha256:" + "a" * 64,
            )
            with pytest.raises(ValueError, match="LLM surface"):
                BudgetedLLMBackend(
                    analyzer=analyzer,
                    controller=controller,
                    initial_estimate=_estimate(),
                    repair_estimate=_estimate(),
                )

    asyncio.run(run())


def test_live_llm_backend_rejects_caller_duck_with_exact_live_evidence(
    tmp_path: Path,
) -> None:
    controller = HardBudgetController(_budget(), formal_live=True)
    policy = load_pricing_policy(
        Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")
    ).model_copy(
        update={
            "rates": [
                rate.model_copy(update={"model_or_adapter": "deepseek-v4-flash"})
                if rate.dependency == "llm"
                else rate
                for rate in load_pricing_policy(
                    Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")
                ).rates
            ]
        }
    )
    pricer = ActualCostPricer(policy, valued_at=datetime(2026, 8, 1, tzinfo=UTC))

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
        ) as http_client:
            trusted = LiveCaptureLLMAnalyzer(
                client=OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url="https://api.deepseek.com/v1",
                    model="deepseek-v4-flash",
                    api_key="private-key",
                ),
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=pricer,
                controller=controller,
                prompt_artifact_sha256="sha256:" + "a" * 64,
            )

            class ExactDuck:
                live_identity_evidence = trusted.live_identity_evidence
                live_controller = controller
                live_pricer = pricer

                async def generate_json(self, **_kwargs: object) -> object:
                    raise AssertionError("duck analyzer dispatched")

            with pytest.raises(ValueError, match="trusted live LLM analyzer"):
                BudgetedLLMBackend(
                    analyzer=ExactDuck(),  # type: ignore[arg-type]
                    controller=controller,
                    initial_estimate=_estimate(),
                    repair_estimate=_estimate(),
                )

    asyncio.run(run())


def test_initial_and_repair_calls_receive_independent_terminal_reservations() -> None:
    controller = HardBudgetController(_budget())
    analyzer = _FakeAnalyzer([_result(), _result()])
    backend = BudgetedLLMBackend(
        analyzer=analyzer,
        controller=controller,
        initial_estimate=_estimate(),
        repair_estimate=_estimate(),
    )
    request = LLMGenerationRequest(prompt_name="recall_actions", payload={"query": "offline"})

    initial = asyncio.run(backend.generate(request, "initial"))
    repair = asyncio.run(backend.generate(request, "repair"))

    assert initial.data == {"actions": []}
    assert repair.data == {"actions": []}
    assert len(analyzer.reservations) == 2
    assert analyzer.reservations[0].reservation_id != analyzer.reservations[1].reservation_id
    assert [item["action"] for item in controller.export_state()["terminal_outcomes"]] == [
        "recall.generate.initial",
        "recall.generate.repair",
    ]
    assert all(
        controller.terminal_outcome(reservation) == ("settled", _result().usage)
        for reservation in analyzer.reservations
    )


def test_invalid_json_is_repairable_without_becoming_infrastructure_failure() -> None:
    invalid_json = ErrorDetail(
        code="invalid_json",
        message="response is not JSON",
        retryable=False,
        provider="llm",
    )
    backend = BudgetedLLMBackend(
        analyzer=_FakeAnalyzer([_result(errors=[invalid_json])]),
        controller=HardBudgetController(_budget()),
        initial_estimate=_estimate(),
        repair_estimate=_estimate(),
    )
    request = LLMGenerationRequest(prompt_name="recall_actions", payload={"query": "offline"})

    observed = asyncio.run(backend.generate(request, "initial"))

    assert observed.errors[0].code == "invalid_json"
    assert observed.repairable is True
    assert observed.infrastructure_failure is False


def test_any_infrastructure_error_prevents_semantic_repair() -> None:
    invalid_json = ErrorDetail(
        code="invalid_json",
        message="response is not JSON",
        retryable=False,
        provider="llm",
    )
    network_error = ErrorDetail(
        code="network_error",
        message="connection failed",
        retryable=True,
        provider="llm",
    )
    backend = BudgetedLLMBackend(
        analyzer=_FakeAnalyzer([_result(errors=[invalid_json, network_error])]),
        controller=HardBudgetController(_budget()),
        initial_estimate=_estimate(),
        repair_estimate=_estimate(),
    )
    request = LLMGenerationRequest(prompt_name="recall_actions", payload={"query": "offline"})

    observed = asyncio.run(backend.generate(request, "initial"))

    assert observed.infrastructure_failure is True
    assert observed.repairable is False


@pytest.mark.parametrize("code", ["invalid_request", "unknown_provider_failure"])
def test_all_non_json_analyzer_errors_are_infrastructure_failures(code: str) -> None:
    error = ErrorDetail(
        code=code,
        message="provider rejected the request",
        retryable=False,
        provider="llm",
    )
    backend = BudgetedLLMBackend(
        analyzer=_FakeAnalyzer([_result(errors=[error])]),
        controller=HardBudgetController(_budget()),
        initial_estimate=_estimate(),
        repair_estimate=_estimate(),
    )
    request = LLMGenerationRequest(prompt_name="recall_actions", payload={"query": "offline"})

    observed = asyncio.run(backend.generate(request, "initial"))

    assert observed.infrastructure_failure is True
    assert observed.repairable is False


def test_cancelled_generation_releases_its_undispatched_reservation() -> None:
    controller = HardBudgetController(_budget())
    analyzer = _CancellingAnalyzer()
    backend = BudgetedLLMBackend(
        analyzer=analyzer,
        controller=controller,
        initial_estimate=_estimate(),
        repair_estimate=_estimate(),
    )
    request = LLMGenerationRequest(prompt_name="recall_actions", payload={"query": "offline"})

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(backend.generate(request, "initial"))

    assert controller.export_state()["reservations"] == []


def test_accounting_mismatch_is_a_terminal_infrastructure_failure() -> None:
    controller = HardBudgetController(_budget())
    oversized = _result().model_copy(
        update={
            "usage": UsageActual(
                llm_calls=1,
                input_tokens=99,
                output_tokens=4,
                elapsed_ms=8,
                cost_cny=Decimal("0.01"),
            )
        }
    )
    analyzer = _FakeAnalyzer([oversized])
    backend = BudgetedLLMBackend(
        analyzer=analyzer,
        controller=controller,
        initial_estimate=_estimate(),
        repair_estimate=_estimate(),
    )
    request = LLMGenerationRequest(prompt_name="recall_actions", payload={"query": "offline"})

    observed = asyncio.run(backend.generate(request, "initial"))

    assert observed.errors[0].code == "accounting_failure"
    assert observed.infrastructure_failure is True
    assert observed.repairable is False
    assert controller.terminal_outcome(analyzer.reservations[0]) == ("failed", oversized.usage)


@pytest.mark.parametrize(
    "code",
    ["authentication_error", "rate_limited", "network_error", "snapshot_unavailable"],
)
def test_infrastructure_errors_are_not_semantic_repairs(code: str) -> None:
    error = ErrorDetail(
        code=code,
        message="offline failure",
        retryable=code in {"rate_limited", "network_error"},
        provider="llm",
    )
    backend = BudgetedLLMBackend(
        analyzer=_FakeAnalyzer([_result(errors=[error])]),
        controller=HardBudgetController(_budget()),
        initial_estimate=_estimate(),
        repair_estimate=_estimate(),
    )
    request = LLMGenerationRequest(prompt_name="recall_actions", payload={"query": "offline"})

    observed = asyncio.run(backend.generate(request, "initial"))

    assert observed.errors == [error]
    assert observed.infrastructure_failure is True
    assert observed.repairable is False
