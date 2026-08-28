from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from paper_search.domain.models import BudgetReservation, UsageEstimate
from paper_search.llm.client import (
    LLMResponseDecoder,
    OpenAICompatibleLLMClient,
    usage_from_response_bytes,
)


API_KEY = "unit-test-secret"


def test_usage_parser_preserves_deepseek_cache_hit_and_miss_tokens() -> None:
    usage = usage_from_response_bytes(
        json.dumps(
            {
                "usage": {
                    "prompt_tokens": 12,
                    "prompt_cache_hit_tokens": 7,
                    "prompt_cache_miss_tokens": 5,
                    "completion_tokens": 3,
                }
            }
        ).encode("utf-8")
    )

    assert usage.input_tokens == 12
    assert usage.cached_input_tokens == 7
    assert usage.uncached_input_tokens == 5


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


def test_deepseek_client_descriptor_binds_endpoint_model_and_adapter_not_key() -> None:
    def make_client(
        http_client: httpx.AsyncClient, *, base_url: str, model: str, key: str
    ) -> OpenAICompatibleLLMClient:
        return OpenAICompatibleLLMClient(
            client=http_client,
            base_url=base_url,
            model=model,
            api_key=key,
        )

    async def run() -> None:
        transport = httpx.MockTransport(lambda request: pytest.fail(str(request)))
        async with httpx.AsyncClient(transport=transport) as http_client:
            first = make_client(
                http_client,
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                key="credential-one",
            )
            second = make_client(
                http_client,
                base_url="https://api.deepseek.com/",
                model="deepseek-chat",
                key="credential-two",
            )
            assert first.live_provider_descriptor == second.live_provider_descriptor
            descriptor = first.live_provider_descriptor
            assert descriptor.provider == "deepseek"
            assert descriptor.model == "deepseek-chat"
            assert descriptor.endpoints == (
                "https://api.deepseek.com/chat/completions",
            )
            assert descriptor.operations == ("generate_json",)
            serialized = descriptor.model_dump_json()
            assert "credential-one" not in serialized
            assert "credential-two" not in serialized

    asyncio.run(run())


def test_llm_descriptor_changes_with_endpoint_or_model() -> None:
    async def run() -> None:
        transport = httpx.MockTransport(lambda request: pytest.fail(str(request)))
        async with httpx.AsyncClient(transport=transport) as http_client:
            def make(base_url: str, model: str) -> OpenAICompatibleLLMClient:
                return OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url=base_url,
                    model=model,
                    api_key="credential",
                )

            baseline = make("https://api.deepseek.com", "deepseek-chat")
            assert baseline.live_provider_descriptor != make(
                "https://api.deepseek.com", "deepseek-reasoner"
            ).live_provider_descriptor
            dashscope = make(
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "deepseek-v3",
            ).live_provider_descriptor
            assert baseline.live_provider_descriptor != dashscope
            assert dashscope.provider == "dashscope"

    asyncio.run(run())


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.deepseek.com.evil.invalid/v1",
        "https://api.deepseek.com@evil.invalid/v1",
        "https://api.deepseek.com/v1?redirect=https://evil.invalid",
        "https://api.deepseek.com/v1#private-fragment",
        "https://dashscope.aliyuncs.com.evil.invalid/compatible-mode/v1",
        "https://dashscope.aliyuncs.com/not-approved",
    ],
)
def test_llm_client_rejects_spoofed_or_ambiguous_provider_urls_before_dispatch(
    base_url: str,
) -> None:
    requests: list[httpx.Request] = []

    def no_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError(request)

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(no_request)
        ) as http_client:
            with pytest.raises(ValueError, match="base_url"):
                OpenAICompatibleLLMClient(
                    client=http_client,
                    base_url=base_url,
                    model="deepseek-chat",
                    api_key="PRIVATE API KEY",
                )

    asyncio.run(run())
    assert requests == []


def test_llm_client_structurally_normalizes_approved_deepseek_origin_and_path() -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
        ) as http_client:
            client = OpenAICompatibleLLMClient(
                client=http_client,
                base_url="https://API.DEEPSEEK.COM/v1/",
                model="deepseek-chat",
                api_key="PRIVATE API KEY",
            )
            assert client.live_provider_descriptor.provider == "deepseek"
            assert client.live_provider_descriptor.endpoints == (
                "https://api.deepseek.com/v1/chat/completions",
            )

    asyncio.run(run())


def test_llm_transport_config_tampering_fails_before_secret_or_prompt_dispatch() -> None:
    requests: list[httpx.Request] = []

    def attacker(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError("attacker transport received a request")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(attacker)
        ) as http_client:
            client = OpenAICompatibleLLMClient(
                client=http_client,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat",
                api_key="PRIVATE API KEY",
            )
            with pytest.raises(FrozenInstanceError):
                client._transport_config.descriptor = (  # type: ignore[attr-defined,misc]
                    client.live_provider_descriptor
                )
            object.__setattr__(
                client._transport_config.descriptor,  # type: ignore[attr-defined]
                "endpoints",
                ("https://attacker.invalid/chat/completions",),
            )
            with pytest.raises(ValueError, match="transport configuration"):
                await client.request_response(
                    prompt_name="PRIVATE PROMPT",
                    payload={"query": "PRIVATE QUERY"},
                )

    asyncio.run(run())
    assert requests == []


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
    assert result.provenance["request_id"].startswith("sha256:")
    assert "request-1" not in result.provenance["request_id"]
    assert result.provenance["response_hash"].startswith("sha256:")
    assert seen[0].url == httpx.URL("https://llm.example.test/v1/chat/completions")
    assert seen[0].headers["authorization"] == f"Bearer {API_KEY}"


def test_canonical_request_includes_bound_prompt_instructions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={}, request=request)

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLMClient(
                client=client,
                base_url="https://llm.example.test/v1",
                model="fixture-model",
                api_key=API_KEY,
            )
            return llm.canonical_request(
                prompt_name="query_analyze",
                payload={"query": "graph retrieval"},
                prompt_instructions=(
                    "Respond with a JSON object.\n"
                    "- Return one QuerySpec and one SearchPlan."
                ),
            )

    request = asyncio.run(run())

    system = request["messages"][0]
    assert system["role"] == "system"
    assert "Return one QuerySpec and one SearchPlan" in system["content"]
    assert system["content"] != "Respond with a JSON object."


def test_generate_json_sends_bound_prompt_instructions() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"query_spec": {"ok": True}}),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
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
                payload={"query": "graph retrieval"},
                reservation=_reservation(),
                prompt_instructions=(
                    "Respond with a JSON object.\n- Return one QuerySpec."
                ),
            )

    asyncio.run(run())

    payload = json.loads(seen[0].content)
    assert payload["messages"][0]["role"] == "system"
    assert "Return one QuerySpec" in payload["messages"][0]["content"]


def test_json_object_request_contains_json_hint_in_messages() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "fixture-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"query_spec": {"ok": True}}),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
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
                prompt_version="query-analyze-v1",
            )
            return await llm.generate_json(
                prompt_name="query_analyze",
                payload={"query": "graph retrieval"},
                reservation=_reservation(),
            )

    asyncio.run(run())

    body = json.loads(seen[0].content)
    assert body["response_format"] == {"type": "json_object"}
    messages_text = " ".join(
        str(message.get("content", "")) for message in body["messages"]
    )
    assert "json" in messages_text.casefold()


def test_llm_decoder_records_snapshot_refs_in_provenance() -> None:
    from paper_search.application.contracts import SnapshotRef

    decoder = LLMResponseDecoder(prompt_version="query-analyze-v1")
    ref = SnapshotRef(
        entry_id="entry-1",
        dependency="llm",
        cache_key="sha256:" + "a" * 64,
        response_sha256="sha256:" + "b" * 64,
        captured_at=datetime(2026, 8, 3, tzinfo=UTC),
        snapshot_path="responses/llm/entry-1.bin",
    )
    response = (
        b'{"choices":[{"message":{"content":"{\\"query_spec\\":{}}"}}],'
        b'"usage":{"prompt_tokens":1,"completion_tokens":1}}'
    )

    result = decoder.decode(
        response,
        model_id="deepseek-v4-flash",
        captured_at=datetime(2026, 8, 3, tzinfo=UTC),
        cache_hit=False,
        snapshot_ref=ref,
    )

    assert json.loads(result.provenance["snapshot_refs"]) == [
        ref.model_dump(mode="json")
    ]


def test_dashscope_json_request_disables_thinking_and_stays_bounded() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "fixture-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"query_spec": {"ok": True}}),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            request=request,
        )

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLMClient(
                client=client,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model="fixture-model",
                api_key=API_KEY,
                prompt_version="query-analyze-v1",
            )
            return await llm.generate_json(
                prompt_name="query_analyze",
                payload={"query": "graph retrieval"},
                reservation=_reservation(),
            )

    asyncio.run(run())

    body = json.loads(seen[0].content)
    assert body["enable_thinking"] is False
    assert "max_tokens" not in body or body["max_tokens"] > 0


def test_deepseek_json_request_disables_thinking() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "fixture-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"query_spec": {"ok": True}}),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            request=request,
        )

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLMClient(
                client=client,
                base_url="https://api.deepseek.com/v1",
                model="fixture-model",
                api_key=API_KEY,
                prompt_version="query-analyze-v1",
            )
            return await llm.generate_json(
                prompt_name="query_analyze",
                payload={"query": "graph retrieval"},
                reservation=_reservation(),
            )

    asyncio.run(run())

    body = json.loads(seen[0].content)
    assert body["thinking"] == {"type": "disabled"}
    assert "enable_thinking" not in body


def test_non_dashscope_json_request_omits_dashscope_specific_fields() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "fixture-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"query_spec": {"ok": True}}),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
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
                prompt_version="query-analyze-v1",
            )
            return await llm.generate_json(
                prompt_name="query_analyze",
                payload={"query": "graph retrieval"},
                reservation=_reservation(),
            )

    asyncio.run(run())

    body = json.loads(seen[0].content)
    assert "enable_thinking" not in body
    assert "thinking" not in body


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


def test_pure_decoder_does_not_require_transport_or_credentials() -> None:
    raw = json.dumps(
        {
            "choices": [{"message": {"content": json.dumps({"ok": True})}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
    ).encode()

    result = LLMResponseDecoder().decode(
        raw,
        model_id="fixture-model",
        captured_at=datetime(2026, 8, 1, tzinfo=UTC),
        cache_hit=True,
        snapshot_ref=None,
    )

    assert result.data == {"ok": True}
    assert result.cache_hit is True
    assert result.usage.input_tokens == 2
    assert result.usage.output_tokens == 1


def test_pure_decoder_rejects_secret_shaped_model_identifier_without_echo() -> None:
    secret_model = "sk-live-model-secret"

    with pytest.raises(ValueError, match="model identifier is not safe") as error:
        LLMResponseDecoder().decode(
            b"{}",
            model_id=secret_model,
            captured_at=datetime(2026, 8, 1, tzinfo=UTC),
            cache_hit=True,
            snapshot_ref=None,
        )

    assert secret_model not in str(error.value)


def test_pure_decoder_rejects_secret_shaped_prompt_version_without_echo() -> None:
    secret_prompt_version = "query-sk-live-prompt-secret"

    with pytest.raises(ValueError, match="prompt version is not safe") as error:
        LLMResponseDecoder(prompt_version=secret_prompt_version)

    assert secret_prompt_version not in str(error.value)


def test_secret_shaped_model_identifier_is_rejected_without_echo() -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={}, request=request)
            )
        ) as client:
            with pytest.raises(ValueError, match="model identifier is not safe") as error:
                OpenAICompatibleLLMClient(
                    client=client,
                    base_url="https://llm.example.test/v1",
                    model="sk-live-model-secret",
                    api_key=API_KEY,
                )
            assert "sk-live-model-secret" not in str(error.value)

    asyncio.run(run())
