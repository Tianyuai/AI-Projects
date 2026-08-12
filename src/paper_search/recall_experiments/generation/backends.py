"""Budget-owning adapters for replaceable recall-action LLM analyzers.

This module deliberately accepts an already composed analyzer.  It never reads
credentials, constructs a client, or selects live versus replay execution.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from paper_search.control.budget import (
    BudgetExceededError,
    HardBudgetController,
    ReservationError,
)
from paper_search.control.pricing import ActualCostPricer
from paper_search.domain.models import (
    BudgetReservation,
    DomainModel,
    ErrorDetail,
    ProviderResult,
    UsageActual,
    UsageEstimate,
    Sha256,
)
from paper_search.live_identity import LiveDependencyEvidence
from paper_search.llm.snapshot_adapters import LiveCaptureLLMAnalyzer
from paper_search.recall_experiments.identity import (
    LiveDependencyIdentity,
    dependency_identity_from_evidence,
    validate_scheme_b_dependency_identity,
)


LLMCallKind = Literal["initial", "repair"]


class LLMGenerationRequest(DomainModel):
    """The complete input visible to one recall-action generation call."""

    prompt_name: str
    payload: dict[str, object]
    prompt_instructions: str | None = None
    prompt_bytes: bytes | None = None
    prompt_artifact_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_prompt_identity(self) -> LLMGenerationRequest:
        values = (self.prompt_instructions, self.prompt_bytes, self.prompt_artifact_sha256)
        if any(value is not None for value in values) and any(value is None for value in values):
            raise ValueError("prompt instructions, bytes, and SHA-256 must be bound together")
        if self.prompt_bytes is not None:
            digest = "sha256:" + hashlib.sha256(self.prompt_bytes).hexdigest()
            if digest != self.prompt_artifact_sha256:
                raise ValueError("prompt SHA-256 does not match exact prompt bytes")
        return self


class LLMBackendResult(DomainModel):
    """Provider-neutral generation outcome consumed by action generators."""

    data: dict[str, object] = Field(default_factory=dict)
    usage: UsageActual = Field(default_factory=UsageActual)
    provenance: dict[str, str] = Field(default_factory=dict)
    errors: list[ErrorDetail] = Field(default_factory=list)
    infrastructure_failure: bool = False
    repairable: bool = False


class LLMBackend(Protocol):
    async def generate(
        self,
        request: LLMGenerationRequest,
        call_kind: LLMCallKind,
    ) -> LLMBackendResult: ...


class _Analyzer(Protocol):
    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
        prompt_instructions: str | None = None,
        prompt_artifact_sha256: str | None = None,
    ) -> ProviderResult[dict[str, Any]]: ...


def _as_result(result: ProviderResult[dict[str, Any]]) -> LLMBackendResult:
    code = result.errors[0].code if result.errors else None
    repairable = bool(result.errors) and all(
        error.code == "invalid_json" for error in result.errors
    )
    infrastructure_failure = bool(result.errors) and not repairable
    return LLMBackendResult(
        data=dict(result.data),
        usage=result.usage,
        provenance=dict(result.provenance),
        errors=list(result.errors),
        infrastructure_failure=infrastructure_failure,
        repairable=code == "invalid_json" and repairable,
    )


def _error_result(
    *,
    code: str,
    message: str,
    usage: UsageActual | None = None,
) -> LLMBackendResult:
    return LLMBackendResult(
        usage=usage or UsageActual(),
        provenance={"provider": "llm-backend"},
        errors=[
            ErrorDetail(
                code=code,
                message=message,
                retryable=False,
                provider="llm",
            )
        ],
        infrastructure_failure=True,
    )


class BudgetedLLMBackend:
    """Reserve exactly one budget receipt for each analyzer generation call."""

    def __init__(
        self,
        *,
        analyzer: _Analyzer,
        controller: HardBudgetController,
        initial_estimate: UsageEstimate,
        repair_estimate: UsageEstimate,
    ) -> None:
        if initial_estimate.llm_calls < 1 or repair_estimate.llm_calls < 1:
            raise ValueError("LLM backend estimates must reserve at least one LLM call")
        self._analyzer = analyzer
        self._controller = controller
        self._estimates: Mapping[LLMCallKind, UsageEstimate] = {
            "initial": initial_estimate,
            "repair": repair_estimate,
        }
        self._dependency_identity: LiveDependencyIdentity | None = None
        self._live_pricer: ActualCostPricer | None = None
        evidence = getattr(analyzer, "live_identity_evidence", None)
        if evidence is not None:
            if not isinstance(analyzer, LiveCaptureLLMAnalyzer):
                raise ValueError("trusted live LLM analyzer required")
            if not isinstance(evidence, LiveDependencyEvidence):
                raise ValueError("live identity evidence is invalid")
            if getattr(analyzer, "live_controller", None) is not controller:
                raise ValueError("live analyzer controller does not match backend controller")
            if controller.formal_live is not True:
                raise ValueError("live analyzer controller must use formal-live enforcement")
            pricer = getattr(analyzer, "live_pricer", None)
            if not isinstance(pricer, ActualCostPricer):
                raise ValueError("live analyzer pricer is invalid")
            identity = validate_scheme_b_dependency_identity(
                "llm",
                dependency_identity_from_evidence(evidence),
            )
            if identity.dependency != "llm":
                raise ValueError("live LLM dependency identity is invalid")
            self._dependency_identity = identity
            self._live_pricer = pricer

    @property
    def dependency_identity(self) -> LiveDependencyIdentity:
        if self._dependency_identity is None:
            raise ValueError("live identity unavailable")
        return self._dependency_identity

    @property
    def live_pricer(self) -> ActualCostPricer:
        if self._live_pricer is None:
            raise ValueError("live identity unavailable")
        return self._live_pricer

    @property
    def live_controller(self) -> HardBudgetController:
        if self._dependency_identity is None:
            raise ValueError("live identity unavailable")
        return self._controller

    def _settle_or_verify(
        self,
        reservation: BudgetReservation,
        actual: UsageActual,
    ) -> None:
        terminal = self._controller.terminal_outcome(reservation)
        if terminal is None:
            self._controller.settle(reservation, actual)
            return
        mode, recorded = terminal
        if mode != "settled" or recorded != actual:
            raise ReservationError("LLM settlement receipt does not match result")

    def _finalize_exception(self, reservation: BudgetReservation) -> None:
        if self._controller.terminal_outcome(reservation) is not None:
            return
        try:
            self._controller.release(reservation)
        except ReservationError:
            self._controller.fail_closed(reservation, UsageActual())

    def _fail_accounting(self, reservation: BudgetReservation, actual: UsageActual) -> None:
        if self._controller.terminal_outcome(reservation) is not None:
            return
        self._controller.fail_closed(reservation, actual)

    async def generate(
        self,
        request: LLMGenerationRequest,
        call_kind: LLMCallKind,
    ) -> LLMBackendResult:
        estimate = self._estimates[call_kind]
        try:
            reservation = self._controller.reserve(
                f"recall.generate.{call_kind}", estimate
            )
        except BudgetExceededError as error:
            return _error_result(code="budget_exhausted", message=str(error))
        except ReservationError as error:
            return _error_result(code="accounting_failure", message=str(error))

        result: ProviderResult[dict[str, Any]] | None = None
        try:
            result = await self._analyzer.generate_json(
                prompt_name=request.prompt_name,
                payload=request.payload,
                reservation=reservation,
                **_prompt_identity_kwargs(request),
            )
            self._settle_or_verify(reservation, result.usage)
        except asyncio.CancelledError:
            self._finalize_exception(reservation)
            raise
        except ReservationError as error:
            if result is None:
                self._finalize_exception(reservation)
            else:
                self._fail_accounting(reservation, result.usage)
            return _error_result(code="accounting_failure", message=str(error))
        except Exception as error:
            self._finalize_exception(reservation)
            return _error_result(code="provider_error", message=str(error))
        backend_result = _as_result(result)
        provenance = dict(backend_result.provenance)
        provenance["backend_call_id"] = reservation.reservation_id
        return backend_result.model_copy(update={"provenance": provenance})


def _prompt_identity_kwargs(request: LLMGenerationRequest) -> dict[str, str]:
    if request.prompt_instructions is None:
        return {}
    assert request.prompt_artifact_sha256 is not None
    return {
        "prompt_instructions": request.prompt_instructions,
        "prompt_artifact_sha256": request.prompt_artifact_sha256,
    }


__all__ = [
    "BudgetedLLMBackend",
    "LLMBackend",
    "LLMBackendResult",
    "LLMCallKind",
    "LLMGenerationRequest",
]
