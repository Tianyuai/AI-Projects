"""Safe OpenAI-compatible transport and pure structured-response decoding."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from paper_search.application.contracts import SnapshotRef
from paper_search.domain.models import (
    BudgetReservation,
    ErrorDetail,
    ProviderResult,
    UsageActual,
)
from paper_search.live_identity import LiveProviderDescriptor


Clock = Callable[[], datetime]
_SAFE_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
_SAFE_PROMPT_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
_SAFE_PROMPT_ARTIFACT_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_SECRET_SHAPE = re.compile(
    r"(?:\bsk-[A-Za-z0-9]|\bgh[pousr]_[A-Za-z0-9]|\bgithub_pat_"
    r"|\bxox[baprs]-|\bsecret\b|\bbearer\b)",
    re.IGNORECASE,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _hash(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def sanitize_request_id(value: str | None) -> str | None:
    """Retain only an irreversible identity for a provider-controlled request ID."""
    if value is None or not value or len(value) > 1024:
        return None
    return _hash(value.encode("utf-8"))


def validate_model_id(value: str) -> str:
    normalized = value.strip()
    if (
        _SAFE_MODEL_ID.fullmatch(normalized) is None
        or _SECRET_SHAPE.search(normalized) is not None
    ):
        raise ValueError("LLM model identifier is not safe")
    return normalized


def validate_prompt_version(value: str) -> str:
    normalized = value.strip()
    if (
        _SAFE_PROMPT_VERSION.fullmatch(normalized) is None
        or _SECRET_SHAPE.search(normalized) is not None
    ):
        raise ValueError("LLM prompt version is not safe")
    return normalized


def validate_prompt_artifact_sha256(value: str) -> str:
    normalized = value.strip()
    if (
        _SAFE_PROMPT_ARTIFACT_SHA256.fullmatch(normalized) is None
        or normalized == "sha256:" + "0" * 64
    ):
        raise ValueError("LLM prompt artifact SHA-256 is invalid")
    return normalized


def _normalized_transport_endpoint(base_url: str) -> tuple[str, str]:
    if base_url != base_url.strip():
        raise ValueError("LLM base_url is invalid")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("LLM base_url is invalid") from error
    hostname = parsed.hostname
    if (
        parsed.scheme != "https"
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port is not None
    ):
        raise ValueError("LLM base_url is invalid")
    try:
        hostname.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("LLM base_url is invalid") from error
    hostname = hostname.casefold()
    path = parsed.path.rstrip("/")
    if hostname == "api.deepseek.com":
        if path not in {"", "/v1"}:
            raise ValueError("LLM base_url path is not approved")
        provider = "deepseek"
    elif hostname == "dashscope.aliyuncs.com":
        if path != "/compatible-mode/v1":
            raise ValueError("LLM base_url path is not approved")
        provider = "dashscope"
    else:
        if "deepseek" in hostname or "dashscope" in hostname:
            raise ValueError("LLM base_url hostname is not approved")
        provider = "openai_compatible"
    return f"https://{hostname}{path}/chat/completions", provider


@dataclass(frozen=True, slots=True)
class _LLMTransportConfig:
    descriptor: LiveProviderDescriptor

    @property
    def endpoint(self) -> str:
        return self.descriptor.endpoints[0]

    @property
    def provider(self) -> str:
        return self.descriptor.provider

    @property
    def model(self) -> str:
        model = self.descriptor.model
        if model is None:
            raise ValueError("LLM transport model is missing")
        return model

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.descriptor.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


def usage_from_response_bytes(response_bytes: bytes) -> UsageActual:
    """Measure reported token usage without interpreting assistant content."""
    input_tokens = 0
    output_tokens = 0
    try:
        envelope = json.loads(response_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        envelope = None
    if isinstance(envelope, Mapping):
        usage = envelope.get("usage")
        if isinstance(usage, Mapping):
            input_tokens = _nonnegative_int(usage.get("prompt_tokens"))
            output_tokens = _nonnegative_int(usage.get("completion_tokens"))
    return UsageActual(
        llm_calls=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _base_provenance(
    *, model_id: str, captured_at: datetime, response_bytes: bytes
) -> dict[str, str]:
    return {
        "provider": "llm",
        "endpoint": "/chat/completions",
        "model_id": model_id,
        "requested_at": captured_at.isoformat(),
        "response_hash": _hash(response_bytes),
    }


class LLMResponseDecoder:
    """Decode one exact OpenAI-compatible response without transport or credentials."""

    def __init__(self, *, prompt_version: str = "query-analyze-v1") -> None:
        self._prompt_version = validate_prompt_version(prompt_version)

    def decode(
        self,
        response_bytes: bytes,
        *,
        model_id: str,
        captured_at: datetime,
        cache_hit: bool,
        snapshot_ref: SnapshotRef | None,
    ) -> ProviderResult[dict[str, Any]]:
        safe_model_id = validate_model_id(model_id)
        data: dict[str, Any] = {}
        errors: list[ErrorDetail] = []
        try:
            envelope = json.loads(response_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
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

        provenance = _base_provenance(
            model_id=safe_model_id,
            captured_at=captured_at,
            response_bytes=response_bytes,
        )
        provenance["prompt_version"] = self._prompt_version
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
            data=data,
            usage=usage_from_response_bytes(response_bytes),
            provenance=provenance,
            cache_hit=cache_hit,
            latency_ms=0,
            errors=errors,
        )


def _transport_error_result(
    *,
    code: str,
    message: str,
    retryable: bool,
    model_id: str,
    prompt_version: str,
    requested_at: datetime,
    response_bytes: bytes = b"",
    request_id: str | None = None,
    latency_ms: int = 0,
) -> ProviderResult[dict[str, Any]]:
    provenance = _base_provenance(
        model_id=model_id,
        captured_at=requested_at,
        response_bytes=response_bytes,
    )
    provenance["prompt_version"] = prompt_version
    provenance["request_id"] = request_id or "unavailable"
    return ProviderResult[dict[str, Any]](
        data={},
        usage=usage_from_response_bytes(response_bytes),
        provenance=provenance,
        cache_hit=False,
        latency_ms=latency_ms,
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
        if not model.strip() or not api_key:
            raise ValueError("LLM model and API key must not be empty")
        transport_endpoint, provider = _normalized_transport_endpoint(base_url)
        self._client = client
        self._transport_config = _LLMTransportConfig(
            descriptor=LiveProviderDescriptor(
                identity_schema_version="live-provider-descriptor-v1",
                provider=provider,
                dependency="llm",
                adapter="openai-compatible-json",
                version="openai-compatible-client-v1",
                model=validate_model_id(model),
                endpoints=(transport_endpoint,),
                operations=("generate_json",),
            )
        )
        self._transport_config_sha256 = _hash(
            self._transport_config.canonical_bytes()
        )
        self._api_key = api_key
        self._prompt_version = validate_prompt_version(prompt_version)
        self._clock = clock
        self._decoder = LLMResponseDecoder(prompt_version=prompt_version)

    def _validated_transport_config(self) -> _LLMTransportConfig:
        config = self._transport_config
        if _hash(config.canonical_bytes()) != self._transport_config_sha256:
            raise ValueError("LLM transport configuration was modified")
        return config

    @property
    def live_provider_descriptor(self) -> LiveProviderDescriptor:
        return self._validated_transport_config().descriptor

    @property
    def endpoint(self) -> str:
        return "/chat/completions"

    @property
    def model_id(self) -> str:
        return self._validated_transport_config().model

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def canonical_request(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        prompt_instructions: str | None = None,
    ) -> dict[str, object]:
        config = self._validated_transport_config()
        request: dict[str, object] = {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        prompt_instructions
                        if prompt_instructions is not None
                        else "Respond with a JSON object."
                    ),
                },
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
        if config.provider == "dashscope":
            request["enable_thinking"] = False
        elif config.provider == "deepseek":
            request["thinking"] = {"type": "disabled"}
        return request

    def canonical_identity_request(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        prompt_artifact_sha256: str,
    ) -> dict[str, object]:
        """Return only approved identity fields, excluding transport metadata."""
        return {
            "prompt_name": prompt_name,
            "payload": payload,
            "prompt_artifact_sha256": validate_prompt_artifact_sha256(
                prompt_artifact_sha256
            ),
            "prompt_version": self._prompt_version,
        }

    async def request_response(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        prompt_instructions: str | None = None,
    ) -> httpx.Response:
        config = self._validated_transport_config()
        return await self._client.post(
            config.endpoint,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=self.canonical_request(
                prompt_name=prompt_name,
                payload=payload,
                prompt_instructions=prompt_instructions,
            ),
            follow_redirects=False,
        )

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
        prompt_instructions: str | None = None,
    ) -> ProviderResult[dict[str, Any]]:
        """Return decoded JSON or a structured, credential-safe error."""
        if reservation.reserved.llm_calls < 1:
            raise ValueError("reservation must include one LLM call")
        requested_at = self._clock()
        started = time.perf_counter()
        try:
            response = await self.request_response(
                prompt_name=prompt_name,
                payload=payload,
                prompt_instructions=prompt_instructions,
            )
        except httpx.TimeoutException:
            elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
            return _transport_error_result(
                code="timeout",
                message="LLM request timed out",
                retryable=True,
                model_id=self.model_id,
                prompt_version=self._prompt_version,
                requested_at=requested_at,
                latency_ms=elapsed_ms,
            )
        except httpx.RequestError:
            elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
            return _transport_error_result(
                code="network_error",
                message="LLM network request failed",
                retryable=True,
                model_id=self.model_id,
                prompt_version=self._prompt_version,
                requested_at=requested_at,
                latency_ms=elapsed_ms,
            )

        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        request_id = sanitize_request_id(response.headers.get("x-request-id"))
        if response.status_code != 200:
            return _transport_error_result(
                code="provider_error",
                message=f"LLM provider returned HTTP {response.status_code}",
                retryable=response.status_code == 429 or response.status_code >= 500,
                model_id=self.model_id,
                prompt_version=self._prompt_version,
                requested_at=requested_at,
                response_bytes=response.content,
                request_id=request_id,
                latency_ms=elapsed_ms,
            )
        decoded = self._decoder.decode(
            response.content,
            model_id=self.model_id,
            captured_at=requested_at,
            cache_hit=False,
            snapshot_ref=None,
        )
        provenance = dict(decoded.provenance)
        provenance["request_id"] = request_id or "unavailable"
        return decoded.model_copy(
            update={"provenance": provenance, "latency_ms": elapsed_ms}
        )


__all__ = ["LLMResponseDecoder", "OpenAICompatibleLLMClient"]
