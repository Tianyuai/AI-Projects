"""Priced live capture and structurally offline replay adapters for LLM calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

import httpx

from paper_search.application.contracts import SnapshotRef
from paper_search.control.budget import HardBudgetController
from paper_search.control.pricing import ActualCostPricer
from paper_search.domain.models import (
    BudgetReservation,
    ErrorDetail,
    ProviderResult,
    UsageActual,
)
from paper_search.errors import ProtectedExecutionError
from paper_search.llm.client import (
    LLMResponseDecoder,
    OpenAICompatibleLLMClient,
    sanitize_request_id,
    usage_from_response_bytes,
    validate_model_id,
    validate_prompt_artifact_sha256,
    validate_prompt_version,
)
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencyRequestIdentity,
    DependencySnapshotReader,
)


Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


class RequestSettlementController(Protocol):
    def settle(self, reservation: BudgetReservation, actual: UsageActual) -> None: ...

    def fail_closed(
        self, reservation: BudgetReservation, actual: UsageActual
    ) -> None: ...


class LLMAdapterError(ProtectedExecutionError):
    """A fixed, credential-safe terminal adapter failure."""

    search_error_code = "dependency_failure"


class HardBudgetSettlementAdapter:
    """Expose the analyzer settlement contract over one authoritative controller."""

    def __init__(self, controller: HardBudgetController) -> None:
        self._controller = controller

    @property
    def committed_usage(self) -> UsageActual:
        return self._controller.committed_usage

    def stop_status(self) -> str:
        return self._controller.stop_status()

    def mark_dispatched(self, reservation: BudgetReservation) -> None:
        self._controller.mark_dispatched(reservation)

    def settle(self, reservation: BudgetReservation, actual: UsageActual) -> None:
        self._controller.settle(reservation, actual)

    def fail_closed(
        self, reservation: BudgetReservation, actual: UsageActual
    ) -> None:
        self._controller.fail_closed(reservation, actual)

    def fail_closed_attempts(
        self,
        reservation: BudgetReservation,
        actuals: list[UsageActual],
    ) -> None:
        self._controller.fail_closed_attempts(reservation, actuals)


def _identity(
    client: OpenAICompatibleLLMClient,
    *,
    prompt_name: str,
    payload: dict[str, object],
    prompt_artifact_sha256: str,
) -> DependencyRequestIdentity:
    return DependencyRequestIdentity.from_canonical_request(
        dependency="llm",
        operation="generate_json",
        method="POST",
        endpoint=client.endpoint,
        model_or_adapter=client.model_id,
        canonical_request=client.canonical_identity_request(
            prompt_name=prompt_name,
            payload=payload,
            prompt_artifact_sha256=prompt_artifact_sha256,
        ),
    )


def _replay_identity(
    *,
    model_id: str,
    prompt_name: str,
    payload: dict[str, object],
    prompt_artifact_sha256: str,
    prompt_version: str,
) -> DependencyRequestIdentity:
    canonical_request = {
        "prompt_name": prompt_name,
        "payload": payload,
        "prompt_artifact_sha256": prompt_artifact_sha256,
        "prompt_version": prompt_version,
    }
    return DependencyRequestIdentity.from_canonical_request(
        dependency="llm",
        operation="generate_json",
        method="POST",
        endpoint="/chat/completions",
        model_or_adapter=model_id,
        canonical_request=canonical_request,
    )


def _aggregate(usages: list[UsageActual]) -> UsageActual:
    costs = [usage.cost_cny for usage in usages]
    cost = None if any(item is None for item in costs) else sum(
        (item for item in costs if item is not None), Decimal("0")
    )
    return UsageActual(
        search_api_calls=sum(item.search_api_calls for item in usages),
        llm_calls=sum(item.llm_calls for item in usages),
        input_tokens=sum(item.input_tokens for item in usages),
        output_tokens=sum(item.output_tokens for item in usages),
        cost_cny=cost,
        elapsed_ms=sum(item.elapsed_ms for item in usages),
    )


def _terminal_attempts(
    measured_attempts: list[UsageActual], valued_attempts: list[UsageActual]
) -> list[UsageActual]:
    return [*valued_attempts, *measured_attempts[len(valued_attempts) :]]


def _fail_closed_terminal(
    controller: RequestSettlementController,
    reservation: BudgetReservation,
    measured_attempts: list[UsageActual],
    valued_attempts: list[UsageActual],
) -> None:
    attempts = _terminal_attempts(measured_attempts, valued_attempts)
    batch_settlement = getattr(controller, "fail_closed_attempts", None)
    if callable(batch_settlement):
        batch_settlement(reservation, attempts)
        return
    controller.fail_closed(reservation, _aggregate(attempts))


def _error_result(
    *,
    code: str,
    message: str,
    retryable: bool,
    model_id: str,
    requested_at: datetime,
    response_bytes: bytes,
    prompt_version: str,
    usage: UsageActual,
    request_id: str | None = None,
    snapshot_ref: SnapshotRef | None = None,
) -> ProviderResult[dict[str, Any]]:
    provenance: dict[str, str] = {
        "provider": "llm",
        "endpoint": "/chat/completions",
        "model_id": model_id,
        "requested_at": requested_at.isoformat(),
        "response_hash": _sha256(response_bytes),
        "prompt_version": prompt_version,
        "request_id": request_id or "unavailable",
    }
    if snapshot_ref is not None:
        provenance.update(
            {
                "snapshot_entry_id": snapshot_ref.entry_id,
                "snapshot_cache_key": snapshot_ref.cache_key,
                "snapshot_response_sha256": snapshot_ref.response_sha256,
                "snapshot_path": snapshot_ref.snapshot_path,
                "snapshot_refs": json.dumps(
                    [snapshot_ref.model_dump(mode="json")],
                    separators=(",", ":"),
                ),
            }
        )
    return ProviderResult[dict[str, Any]](
        data={},
        usage=usage,
        provenance=provenance,
        cache_hit=False,
        latency_ms=usage.elapsed_ms,
        errors=[
            ErrorDetail(
                code=code,
                message=message,
                retryable=retryable,
                provider="llm",
                request_id=request_id,
            )
        ],
    )


def _status_error(status_code: int) -> tuple[str, str, bool]:
    if status_code in (401, 403):
        return "authentication_error", "LLM authentication failed", False
    if status_code == 429:
        return "rate_limited", "LLM request was rate limited", True
    if 500 <= status_code <= 599:
        return "server_error", "LLM provider failed", True
    return "invalid_request", "LLM request was rejected", False


class LiveCaptureLLMAnalyzer:
    """Retry, price, settle, and capture one live LLM request."""

    def __init__(
        self,
        *,
        client: OpenAICompatibleLLMClient,
        capture_store: DependencyCaptureStore,
        pricer: ActualCostPricer,
        controller: RequestSettlementController,
        prompt_artifact_sha256: str,
        decoder: LLMResponseDecoder | None = None,
        clock: Clock = _utc_now,
        prompt_instructions: str | None = None,
    ) -> None:
        self._client = client
        self._capture_store = capture_store
        self._pricer = pricer
        self._controller = controller
        self._prompt_artifact_sha256 = validate_prompt_artifact_sha256(
            prompt_artifact_sha256
        )
        self._decoder = decoder or LLMResponseDecoder(
            prompt_version=client.prompt_version
        )
        self._prompt_instructions = prompt_instructions
        self._clock = clock

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]:
        if reservation.reserved.llm_calls < 1:
            raise ValueError("reservation must include one LLM call")
        requested_at = self._clock()
        identity = _identity(
            self._client,
            prompt_name=prompt_name,
            payload=payload,
            prompt_artifact_sha256=self._prompt_artifact_sha256,
        )
        measured_attempts: list[UsageActual] = []
        valued_attempts: list[UsageActual] = []
        terminal: ProviderResult[dict[str, Any]] | None = None
        in_flight_started: float | None = None
        dispatch_unaccounted = False

        try:
            for attempt in range(3):
                started = time.perf_counter()
                response: httpx.Response | None = None
                mark_dispatched = getattr(
                    self._controller, "mark_dispatched", None
                )
                if callable(mark_dispatched):
                    mark_dispatched(reservation)
                in_flight_started = started
                dispatch_unaccounted = True
                try:
                    response = await self._client.request_response(
                        prompt_name=prompt_name,
                        payload=payload,
                        prompt_instructions=self._prompt_instructions,
                    )
                except httpx.TimeoutException:
                    code, message, retryable = (
                        "timeout",
                        "LLM request timed out",
                        True,
                    )
                    response_bytes = b""
                    request_id = None
                except httpx.RequestError:
                    code, message, retryable = (
                        "network_error",
                        "LLM network request failed",
                        True,
                    )
                    response_bytes = b""
                    request_id = None
                else:
                    response_bytes = response.content
                    request_id = sanitize_request_id(
                        response.headers.get("x-request-id")
                    )
                    code, message, retryable = _status_error(response.status_code)
                elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
                measured = usage_from_response_bytes(response_bytes).model_copy(
                    update={"elapsed_ms": elapsed_ms}
                )
                measured_attempts.append(measured)
                dispatch_unaccounted = False
                in_flight_started = None
                valued_attempts.append(
                    self._pricer.value_actual(
                        dependency="llm",
                        model_or_adapter=self._client.model_id,
                        usage=measured,
                    )
                )
                accumulated = _aggregate(valued_attempts)

                if response is not None and response.status_code == 200:
                    decoded = self._decoder.decode(
                        response_bytes,
                        model_id=self._client.model_id,
                        captured_at=requested_at,
                        cache_hit=False,
                        snapshot_ref=None,
                    )
                    if decoded.errors:
                        decoded_error_headers: dict[str, str] = {}
                        content_type = response.headers.get("content-type")
                        if content_type is not None:
                            decoded_error_headers["content-type"] = content_type
                        if request_id is not None:
                            decoded_error_headers["x-request-id"] = request_id
                        snapshot_ref = self._capture_store.stage_error(
                            identity,
                            error_code=decoded.errors[0].code,
                            message=decoded.errors[0].message,
                            retryable=decoded.errors[0].retryable,
                            response_bytes=response_bytes,
                            safe_headers=decoded_error_headers,
                            captured_at=requested_at,
                        )
                        self._capture_store.annotate_usage(
                            snapshot_ref.entry_id,
                            accumulated,
                        )
                        decoded_with_ref = self._decoder.decode(
                            response_bytes,
                            model_id=self._client.model_id,
                            captured_at=requested_at,
                            cache_hit=False,
                            snapshot_ref=snapshot_ref,
                        )
                        terminal = decoded_with_ref.model_copy(
                            update={
                                "usage": accumulated,
                                "latency_ms": accumulated.elapsed_ms,
                            }
                        )
                    else:
                        success_headers: dict[str, str] = {}
                        content_type = response.headers.get("content-type")
                        if content_type is not None:
                            success_headers["content-type"] = content_type
                        if request_id is not None:
                            success_headers["x-request-id"] = request_id
                        snapshot_ref = self._capture_store.stage_success(
                            identity,
                            response_bytes=response_bytes,
                            safe_headers=success_headers,
                            captured_at=requested_at,
                        )
                        self._capture_store.annotate_usage(
                            snapshot_ref.entry_id,
                            accumulated,
                        )
                        decoded_with_ref = self._decoder.decode(
                            response_bytes,
                            model_id=self._client.model_id,
                            captured_at=requested_at,
                            cache_hit=False,
                            snapshot_ref=snapshot_ref,
                        )
                        terminal = decoded_with_ref.model_copy(
                            update={
                                "usage": accumulated,
                                "latency_ms": accumulated.elapsed_ms,
                            }
                        )
                    break

                terminal = _error_result(
                    code=code,
                    message=message,
                    retryable=retryable,
                    model_id=self._client.model_id,
                    requested_at=requested_at,
                    response_bytes=response_bytes,
                    prompt_version=self._client.prompt_version,
                    usage=accumulated,
                    request_id=request_id,
                )
                if not retryable or attempt == 2:
                    break

            if terminal is None:
                raise RuntimeError("missing terminal LLM result")
            if response is not None and response.status_code != 200:
                http_error_headers: dict[str, str] = {}
                content_type = response.headers.get("content-type")
                if content_type is not None:
                    http_error_headers["content-type"] = content_type
                if request_id is not None:
                    http_error_headers["x-request-id"] = request_id
                snapshot_ref = self._capture_store.stage_error(
                    identity,
                    error_code=code,
                    message=message,
                    retryable=retryable,
                    response_bytes=response_bytes,
                    safe_headers=http_error_headers,
                    captured_at=requested_at,
                )
                self._capture_store.annotate_usage(
                    snapshot_ref.entry_id,
                    accumulated,
                )
                terminal = _error_result(
                    code=code,
                    message=message,
                    retryable=retryable,
                    model_id=self._client.model_id,
                    requested_at=requested_at,
                    response_bytes=response_bytes,
                    prompt_version=self._client.prompt_version,
                    usage=accumulated,
                    request_id=request_id,
                    snapshot_ref=snapshot_ref,
                )

            self._controller.settle(reservation, terminal.usage)
            return terminal
        except asyncio.CancelledError as cancellation:
            if dispatch_unaccounted and in_flight_started is not None:
                measured_attempts.append(
                    UsageActual(
                        llm_calls=1,
                        elapsed_ms=max(
                            0,
                            round(
                                (time.perf_counter() - in_flight_started) * 1000
                            ),
                        ),
                    )
                )
            try:
                _fail_closed_terminal(
                    self._controller,
                    reservation,
                    measured_attempts,
                    valued_attempts,
                )
            except Exception:
                pass
            raise cancellation from None
        except Exception:
            if dispatch_unaccounted and in_flight_started is not None:
                measured_attempts.append(
                    UsageActual(
                        llm_calls=1,
                        elapsed_ms=max(
                            0,
                            round(
                                (time.perf_counter() - in_flight_started) * 1000
                            ),
                        ),
                    )
                )
            try:
                _fail_closed_terminal(
                    self._controller,
                    reservation,
                    measured_attempts,
                    valued_attempts,
                )
            except Exception:
                pass
            raise LLMAdapterError("LLM live capture failed") from None


class ReplayLLMAnalyzer:
    """Decode an LLM response from an immutable snapshot with no live client."""

    def __init__(
        self,
        *,
        reader: DependencySnapshotReader,
        model_id: str,
        prompt_artifact_sha256: str,
        prompt_version: str = "query-analyze-v1",
        decoder: LLMResponseDecoder | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        self._reader = reader
        self._model_id = validate_model_id(model_id)
        self._prompt_artifact_sha256 = validate_prompt_artifact_sha256(
            prompt_artifact_sha256
        )
        self._prompt_version = validate_prompt_version(prompt_version)
        self._decoder = decoder or LLMResponseDecoder(
            prompt_version=self._prompt_version
        )
        self._clock = clock

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]:
        del reservation
        identity = _replay_identity(
            model_id=self._model_id,
            prompt_name=prompt_name,
            payload=payload,
            prompt_artifact_sha256=self._prompt_artifact_sha256,
            prompt_version=self._prompt_version,
        )
        try:
            snapshot = self._reader.read(identity)
        except KeyError:
            return _error_result(
                code="snapshot_unavailable",
                message="LLM snapshot is unavailable",
                retryable=False,
                model_id=self._model_id,
                requested_at=self._clock(),
                response_bytes=b"",
                prompt_version=self._prompt_version,
                usage=UsageActual(),
            )
        if snapshot.error is not None:
            return _error_result(
                code=snapshot.error.code,
                message=snapshot.error.message,
                retryable=snapshot.error.retryable,
                model_id=self._model_id,
                requested_at=self._clock(),
                response_bytes=snapshot.response_bytes,
                prompt_version=self._prompt_version,
                usage=snapshot.usage or UsageActual(),
                snapshot_ref=snapshot.ref,
            )
        decoded = self._decoder.decode(
            snapshot.response_bytes,
            model_id=self._model_id,
            captured_at=snapshot.ref.captured_at,
            cache_hit=True,
            snapshot_ref=snapshot.ref,
        )
        provenance = dict(decoded.provenance)
        provenance["snapshot_set_id"] = self._reader.snapshot_set_id
        return decoded.model_copy(
            update={
                "usage": snapshot.usage or UsageActual(),
                "provenance": provenance,
            }
        )


__all__ = [
    "LLMAdapterError",
    "LLMResponseDecoder",
    "HardBudgetSettlementAdapter",
    "LiveCaptureLLMAnalyzer",
    "ReplayLLMAnalyzer",
]
