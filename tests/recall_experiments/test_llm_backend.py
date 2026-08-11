"""Contract tests for reservation-owning LLM recall generation backends."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import ErrorDetail, ProviderResult, SearchBudget, UsageActual, UsageEstimate
from paper_search.recall_experiments.generation.backends import (
    BudgetedLLMBackend,
    LLMGenerationRequest,
)


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
