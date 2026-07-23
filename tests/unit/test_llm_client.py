from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx

from paper_search.domain.models import BudgetReservation, UsageEstimate
from paper_search.llm.client import OpenAICompatibleLLMClient


API_KEY = "unit-test-secret"


def _reservation() -> BudgetReservation:
    return BudgetReservation(
        reservation_id="llm-1",
        action="query.analyze",
        reserved=UsageEstimate(
            llm_calls=1,
            input_tokens=100,
            output_tokens=100,
            cost_cny=0.05,
        ),
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )


def test_generate_json_returns_data_usage_and_safe_provenance() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "fixture-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"query_spec": {"ok": True}}),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                },
            },
            headers={"x-request-id": "request-1"},
            request=request,
        )

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLMClient(
                client=client,
                base_url="https://llm.example.test/v1",
                model="fixture-model",
                api_key=API_KEY,
                prompt_version="query-analyze-v1",
                clock=lambda: datetime(2026, 7, 23, tzinfo=UTC),
            )
            return await llm.generate_json(
                prompt_name="query_analyze",
                payload={"query": "graph retrieval"},
                reservation=_reservation(),
            )

    result = asyncio.run(run())

    assert result.data == {"query_spec": {"ok": True}}
    assert result.usage.llm_calls == 1
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 7
    assert result.usage.cost_cny is None
    assert result.provenance["model_id"] == "fixture-model"
    assert result.provenance["prompt_version"] == "query-analyze-v1"
    assert result.provenance["request_id"] == "request-1"
    assert result.provenance["response_hash"].startswith("sha256:")
    assert seen[0].url == httpx.URL("https://llm.example.test/v1/chat/completions")
    assert seen[0].headers["authorization"] == f"Bearer {API_KEY}"
    assert API_KEY not in result.model_dump_json()


def test_invalid_json_is_a_structured_error_without_credential_leakage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "fixture-model",
                "choices": [{"message": {"role": "assistant", "content": "{broken"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
            request=request,
        )

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLMClient(
                client=client,
                base_url="https://llm.example.test/v1",
                model="fixture-model",
                api_key=API_KEY,
            )
            return await llm.generate_json(
                prompt_name="query_analyze",
                payload={"authorization": f"Bearer {API_KEY}"},
                reservation=_reservation(),
            )

    result = asyncio.run(run())

    assert result.data == {}
    assert result.errors[0].code == "invalid_json"
    assert API_KEY not in result.model_dump_json()
    assert "authorization" not in result.model_dump_json().casefold()


def test_empty_response_is_a_structured_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "fixture-model", "choices": [], "usage": {}},
            request=request,
        )

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLMClient(
                client=client,
                base_url="https://llm.example.test/v1",
                model="fixture-model",
                api_key=API_KEY,
            )
            return await llm.generate_json(
                prompt_name="query_analyze",
                payload={"query": "x"},
                reservation=_reservation(),
            )

    result = asyncio.run(run())

    assert result.data == {}
    assert result.errors[0].code == "empty_response"
    assert result.usage.llm_calls == 1


def test_timeout_is_bounded_and_structured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLMClient(
                client=client,
                base_url="https://llm.example.test/v1",
                model="fixture-model",
                api_key=API_KEY,
            )
            return await llm.generate_json(
                prompt_name="query_analyze",
                payload={"query": "x"},
                reservation=_reservation(),
            )

    result = asyncio.run(run())

    assert result.data == {}
    assert result.errors[0].code == "timeout"
    assert result.errors[0].retryable is True
    assert result.usage.llm_calls == 1
    assert API_KEY not in result.model_dump_json()
