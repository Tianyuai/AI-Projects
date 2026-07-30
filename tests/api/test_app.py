from __future__ import annotations

import asyncio
import importlib
from typing import Any, cast

import httpx
import pytest

from paper_search.api.app import create_app
from paper_search.api.contracts import SearchRequest
from paper_search.domain.models import (
    QueryAnalysisResult,
    QuerySpec,
    SearchPlan,
    StructuredSearchResponse,
    SubQuery,
    UsageActual,
)


def _structured_response(query_id: str = "q1") -> StructuredSearchResponse:
    spec = QuerySpec(
        original_query="graph retrieval",
        research_goal="find graph retrieval papers",
    )
    return StructuredSearchResponse(
        query_id=query_id,
        query_analysis=QueryAnalysisResult(
            query_spec=spec,
            search_plan=SearchPlan(
                subqueries=[
                    SubQuery(
                        query_id="sq-1",
                        text="graph retrieval",
                        query_type="exact",
                        target_constraints=[],
                        priority=1,
                        provider_hint="either",
                    )
                ],
                inherited_hard_filters={},
                rationale="synthetic",
            ),
        ),
        selected_paper_ids=["openalex:W1"],
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        search_trace=[{"step": "fuse", "count": 1}],
        usage=UsageActual(search_api_calls=1),
        stop_reason="completed",
        is_partial=False,
        warnings=[],
        config_hash="sha256:" + "a" * 64,
        git_sha="abc1234",
    )


class RecordingService:
    def __init__(self, *, raises: bool = False) -> None:
        self.raises = raises
        self.requests: list[SearchRequest] = []

    async def __call__(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResponse:
        self.requests.append(request)
        if self.raises:
            raise RuntimeError("private failure detail")
        return _structured_response(request.query_id)


async def _request(
    application: object,
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        return await client.request(method, path, json=json)


def test_live_is_always_ok() -> None:
    response = asyncio.run(_request(create_app(), "GET", "/health/live"))

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_requires_service_and_all_nonempty_provider_states() -> None:
    ready = asyncio.run(
        _request(
            create_app(
                RecordingService(),
                readiness_probe=lambda: {
                    "semantic_scholar": True,
                    "openalex": True,
                },
            ),
            "GET",
            "/health/ready",
        )
    )

    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "execution_mode": "replay",
        "snapshot_set_id": "mock-snapshot-v1",
        "dependencies": [
            {"dependency": "llm", "state": "ready", "cache_hit": False, "error_codes": []},
            {"dependency": "openalex", "state": "ready", "cache_hit": False, "error_codes": []},
            {
                "dependency": "semantic_scholar",
                "state": "ready",
                "cache_hit": False,
                "error_codes": [],
            },
        ],
        "last_authorized_probe_at": None,
    }
    assert [item["dependency"] for item in ready.json()["dependencies"]] == [
        "llm",
        "openalex",
        "semantic_scholar",
    ]

    degraded = asyncio.run(
        _request(create_app(), "GET", "/health/ready")
    )

    assert degraded.status_code == 503
    assert degraded.json() == {
        "status": "degraded",
        "execution_mode": "replay",
        "snapshot_set_id": "mock-snapshot-v1",
        "dependencies": [
            {"dependency": "llm", "state": "failed", "cache_hit": False, "error_codes": []},
            {"dependency": "openalex", "state": "failed", "cache_hit": False, "error_codes": []},
            {
                "dependency": "semantic_scholar",
                "state": "failed",
                "cache_hit": False,
                "error_codes": [],
            },
        ],
        "last_authorized_probe_at": None,
    }


def test_ready_reports_false_provider_and_probe_failure_as_degraded() -> None:
    unavailable = asyncio.run(
        _request(
            create_app(
                RecordingService(),
                readiness_probe=lambda: {
                    "openalex": True,
                    "semantic_scholar": False,
                },
            ),
            "GET",
            "/health/ready",
        )
    )

    def fail_probe() -> dict[str, bool]:
        raise RuntimeError("private readiness detail")

    failed = asyncio.run(
        _request(
            create_app(RecordingService(), readiness_probe=fail_probe),
            "GET",
            "/health/ready",
        )
    )

    assert unavailable.status_code == 503
    assert unavailable.json() == {
        "status": "degraded",
        "execution_mode": "replay",
        "snapshot_set_id": "mock-snapshot-v1",
        "dependencies": [
            {"dependency": "llm", "state": "ready", "cache_hit": False, "error_codes": []},
            {"dependency": "openalex", "state": "ready", "cache_hit": False, "error_codes": []},
            {
                "dependency": "semantic_scholar",
                "state": "degraded",
                "cache_hit": False,
                "error_codes": [],
            },
        ],
        "last_authorized_probe_at": None,
    }
    assert failed.status_code == 503
    assert failed.json()["status"] == "degraded"
    assert [item["state"] for item in failed.json()["dependencies"]] == [
        "ready",
        "failed",
        "failed",
    ]
    assert "private readiness detail" not in failed.text


@pytest.mark.parametrize(
    "providers",
    [
        {"": True},
        {1: True},
        {"openalex": "false"},
        {"openalex": 1},
        {"openalex": True, 1: False},
    ],
)
def test_ready_fails_closed_for_malformed_probe_mapping(
    providers: object,
) -> None:
    response = asyncio.run(
        _request(
            create_app(
                RecordingService(),
                readiness_probe=cast(
                    Any,
                    lambda: providers,
                ),
            ),
            "GET",
            "/health/ready",
        )
    )

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert [item["state"] for item in response.json()["dependencies"]] == [
        "ready",
        "failed",
        "failed",
    ]


def test_default_module_app_is_explicitly_degraded() -> None:
    api_module = importlib.import_module("paper_search.api.app")

    response = asyncio.run(
        _request(api_module.app, "GET", "/health/ready")
    )

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert [item["state"] for item in response.json()["dependencies"]] == [
        "failed",
        "failed",
        "failed",
    ]


def test_search_validates_request_and_serializes_fixed_response() -> None:
    service = RecordingService()
    application = create_app(
        service,
        readiness_probe=lambda: {"openalex": True},
    )

    response = asyncio.run(
        _request(
            application,
            "POST",
            "/v1/search",
            json={
                "query_id": " q1 ",
                "query": " graph retrieval ",
                "budget_profile": "low",
                "include_trace": False,
            },
        )
    )

    assert response.status_code == 200
    assert StructuredSearchResponse.model_validate(response.json()).query_id == "q1"
    assert service.requests == [
        SearchRequest(
            query_id="q1",
            query="graph retrieval",
            budget_profile="low",
            include_trace=False,
        )
    ]


def test_invalid_search_request_does_not_call_service() -> None:
    service = RecordingService()
    application = create_app(service)

    response = asyncio.run(
        _request(
            application,
            "POST",
            "/v1/search",
            json={"query_id": "q1", "query": "valid", "extra": True},
        )
    )

    assert response.status_code == 422
    assert service.requests == []


def test_search_unavailable_and_service_failure_return_constant_safe_503() -> None:
    unavailable = asyncio.run(
        _request(
            create_app(),
            "POST",
            "/v1/search",
            json={"query_id": "q1", "query": "graph retrieval"},
        )
    )
    failed = asyncio.run(
        _request(
            create_app(RecordingService(raises=True)),
            "POST",
            "/v1/search",
            json={"query_id": "q1", "query": "graph retrieval"},
        )
    )

    expected = {"detail": "search temporarily unavailable"}
    assert unavailable.status_code == 503
    assert unavailable.json() == expected
    assert failed.status_code == 503
    assert failed.json() == expected
    assert "private failure detail" not in failed.text


def test_search_openapi_declares_fixed_503_json_schema() -> None:
    schema = create_app().openapi()

    unavailable = schema["paths"]["/v1/search"]["post"]["responses"]["503"]

    assert unavailable["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/UnavailableResponse"
    }
