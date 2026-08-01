"""Priced live capture and structurally offline replay adapters for LLM calls."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

import httpx

from paper_search.control.pricing import ActualCostPricer
from paper_search.domain.models import (
    BudgetReservation,
    ErrorDetail,
    ProviderResult,
    UsageActual,
)
from paper_search.llm.client import (
    LLMResponseDecoder,
    OpenAICompatibleLLMClient,
    sanitize_request_id,
    usage_from_response_bytes,
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


class LLMAdapterError(RuntimeError):
    """A fixed, credential-safe terminal adapter failure."""


def _identity(
    client: OpenAICompatibleLLMClient,
    *,
    prompt_name: str,
    payload: dict[str, object],
) -> DependencyRequestIdentity:
    return DependencyRequestIdentity.from_canonical_request(
        dependency="llm",
        operation="generate_json",
        method="POST",
        endpoint=client.endpoint,
        model_or_adapter=client.model_id,
        canonical_request=client.canonical_request(
            prompt_name=prompt_name,
            payload=payload,
        ),
    )


def _replay_identity(
    *, model_id: str, prompt_name: str, payload: dict[str, object]
) -> DependencyRequestIdentity:
    canonical_request = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {"prompt_name": prompt_name, "payload": payload},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
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
) -> ProviderResult[dict[str, Any]]:
    return ProviderResult[dict[str, Any]](
        data={},
        usage=usage,
        provenance={
            "provider": "llm",
            "endpoint": "/chat/completions",
            "model_id": model_id,
            "requested_at": requested_at.isoformat(),
            "response_hash": _sha256(response_bytes),
            "prompt_version": prompt_version,
            "request_id": request_id or "unavailable",
        },
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
        decoder: LLMResponseDecoder | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        self._client = client
        self._capture_store = capture_store
        self._pricer = pricer
        self._controller = controller
        self._decoder = decoder or LLMResponseDecoder(
            prompt_version=client.prompt_version
        )
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
        identity = _identity(self._client, prompt_name=prompt_name, payload=payload)
        measured_attempts: list[UsageActual] = []
        valued_attempts: list[UsageActual] = []
        terminal: ProviderResult[dict[str, Any]] | None = None

        try:
            for attempt in range(3):
                started = time.perf_counter()
                response: httpx.Response | None = None
                try:
                    response = await self._client.request_response(
                        prompt_name=prompt_name,
                        payload=payload,
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
                        terminal = decoded.model_copy(
                            update={
                                "usage": accumulated,
                                "latency_ms": accumulated.elapsed_ms,
                            }
                        )
                    else:
                        safe_headers: dict[str, str] = {}
                        content_type = response.headers.get("content-type")
                        if content_type is not None:
                            safe_headers["content-type"] = content_type
                        if request_id is not None:
                            safe_headers["x-request-id"] = request_id
                        snapshot_ref = self._capture_store.stage_success(
                            identity,
                            response_bytes=response_bytes,
                            safe_headers=safe_headers,
                            captured_at=requested_at,
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
            self._controller.settle(reservation, terminal.usage)
            return terminal
        except Exception:
            measured_total = _aggregate(measured_attempts)
            try:
                self._controller.fail_closed(reservation, measured_total)
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
        prompt_version: str = "query-analyze-v1",
        decoder: LLMResponseDecoder | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        self._reader = reader
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._decoder = decoder or LLMResponseDecoder(prompt_version=prompt_version)
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
            update={"usage": UsageActual(), "provenance": provenance}
        )


__all__ = [
    "LLMAdapterError",
    "LLMResponseDecoder",
    "LiveCaptureLLMAnalyzer",
    "ReplayLLMAnalyzer",
]
