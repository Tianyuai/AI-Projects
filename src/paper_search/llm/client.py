"""Safe OpenAI-compatible structured-response adapter."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import httpx

from paper_search.domain.models import (
    BudgetReservation,
    ErrorDetail,
    ProviderResult,
    UsageActual,
)


Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _hash(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


class OpenAICompatibleLLMClient:
    """Generate one JSON object through an injected asynchronous HTTP client."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        api_key: str,
        prompt_version: str = "query-analyze-v1",
        clock: Clock = _utc_now,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        if not normalized_url.startswith("https://"):
            raise ValueError("LLM base_url must use HTTPS")
        if not model.strip() or not api_key:
            raise ValueError("LLM model and API key must not be empty")
        self._client = client
        self._endpoint = f"{normalized_url}/chat/completions"
        self._model = model.strip()
        self._api_key = api_key
        self._prompt_version = prompt_version
        self._clock = clock

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]:
        """Return decoded JSON or a structured, credential-safe error."""
        if reservation.reserved.llm_calls < 1:
            raise ValueError("reservation must include one LLM call")
        requested_at = self._clock()
        started = time.perf_counter()
        raw_response = b""
        data: dict[str, Any] = {}
        errors: list[ErrorDetail] = []
        input_tokens = 0
        output_tokens = 0
        request_id = "unavailable"

        try:
            response = await self._client.post(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
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
                },
                follow_redirects=False,
            )
        except httpx.TimeoutException:
            errors.append(
                ErrorDetail(
                    code="timeout",
                    message="LLM request timed out",
                    retryable=True,
                    provider="llm",
                )
            )
        except httpx.RequestError:
            errors.append(
                ErrorDetail(
                    code="network_error",
                    message="LLM network request failed",
                    retryable=True,
                    provider="llm",
                )
            )
        else:
            raw_response = response.content
            header_request_id = response.headers.get("x-request-id")
            if header_request_id and header_request_id.strip():
                request_id = header_request_id
            if response.status_code != 200:
                errors.append(
                    ErrorDetail(
                        code="provider_error",
                        message=f"LLM provider returned HTTP {response.status_code}",
                        retryable=response.status_code == 429 or response.status_code >= 500,
                        provider="llm",
                        request_id=None if request_id == "unavailable" else request_id,
                    )
                )
            else:
                try:
                    envelope = response.json()
                except json.JSONDecodeError:
                    envelope = None
                if not isinstance(envelope, Mapping):
                    errors.append(
                        ErrorDetail(
                            code="invalid_response",
                            message="LLM provider returned an invalid response envelope",
                            retryable=False,
                            provider="llm",
                        )
                    )
                else:
                    usage = envelope.get("usage")
                    if isinstance(usage, Mapping):
                        input_tokens = _nonnegative_int(usage.get("prompt_tokens"))
                        output_tokens = _nonnegative_int(usage.get("completion_tokens"))
                    choices = envelope.get("choices")
                    content: object = None
                    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                        message = choices[0].get("message")
                        if isinstance(message, Mapping):
                            content = message.get("content")
                    if not isinstance(content, str) or not content.strip():
                        errors.append(
                            ErrorDetail(
                                code="empty_response",
                                message="LLM response did not contain assistant content",
                                retryable=False,
                                provider="llm",
                            )
                        )
                    else:
                        try:
                            decoded = json.loads(content)
                        except json.JSONDecodeError:
                            decoded = None
                        if not isinstance(decoded, dict):
                            errors.append(
                                ErrorDetail(
                                    code="invalid_json",
                                    message="LLM assistant content is not a JSON object",
                                    retryable=False,
                                    provider="llm",
                                )
                            )
                        else:
                            data = decoded

        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        return ProviderResult[dict[str, Any]](
            data=data,
            usage=UsageActual(
                llm_calls=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_cny=None,
                elapsed_ms=elapsed_ms,
            ),
            provenance={
                "provider": "llm",
                "endpoint": "/chat/completions",
                "model_id": self._model,
                "requested_at": requested_at.isoformat(),
                "response_hash": _hash(raw_response),
                "prompt_version": self._prompt_version,
                "request_id": request_id,
            },
            cache_hit=False,
            latency_ms=elapsed_ms,
            errors=errors,
        )
