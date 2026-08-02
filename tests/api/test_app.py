from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, cast

import httpx
import pytest

from paper_search.api.app import create_app
from paper_search.api.contracts import ReadyHealthResponse, SearchRequest
from paper_search.api.routing import SearchServiceRouter
from paper_search.application.contracts import (
    SearchErrorResponse,
    SearchExecutionResult,
    SearchFailure,
    SearchSuccess,
)
from paper_search.domain.models import (
    DependencyStatus,
    QueryAnalysisResult,
    QuerySpec,
    SearchPlan,
    StructuredSearchResponse,
    SubQuery,
    UsageActual,
)


_SAFE_DETAILS = {
    "invalid_request": "The search request is invalid",
    "live_not_authorized": "Live search is not authorized",
    "config_mismatch": "The requested mode does not match the application binding",
    "validation_attempt_conflict": "The validation attempt conflicts with prior state",
    "budget_exhausted": "The search budget was exhausted",
    "snapshot_unavailable": "Required replay data is unavailable",
    "dependency_failure": "A required search dependency failed",
    "integrity_failure": "Search integrity validation failed",
    "internal_error": "The search could not be completed",
}


def _structured_response(query_id: str = "q1") -> StructuredSearchResponse:
    return StructuredSearchResponse(
        run_id="app-run-1",
        query_id=query_id,
        execution_mode="replay",
        snapshot_set_id="app-snapshot-v1",
        snapshot_captured_at=None,
        query_analysis=QueryAnalysisResult(
            query_spec=QuerySpec(
                original_query="graph retrieval",
                research_goal="find graph retrieval papers",
            ),
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
        selected_paper_ids=[],
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        search_trace=[],
        usage=UsageActual(),
        stop_reason="completed",
        is_partial=False,
        planner_fallback=False,
        planner_status="primary",
        dependency_status=[
            DependencyStatus(
                dependency="llm", state="replayed", cache_hit=True, error_codes=[]
            ),
            DependencyStatus(
                dependency="openalex", state="replayed", cache_hit=True, error_codes=[]
            ),
            DependencyStatus(
                dependency="semantic_scholar",
                state="replayed",
                cache_hit=True,
                error_codes=[],
            ),
        ],
        warnings=[],
        prompt_version="query-analyze-v1",
        config_hash="sha256:" + "a" * 64,
        git_sha="abc1234",
    )


def _readiness(
    *,
    observed_at: datetime | None = None,
) -> ReadyHealthResponse:
    return ReadyHealthResponse(
        status="ready",
        execution_mode="replay",
        snapshot_set_id="bound-snapshot-v1",
        dependencies=[
            DependencyStatus(
                dependency="llm", state="replayed", cache_hit=True, error_codes=[]
            ),
            DependencyStatus(
                dependency="openalex", state="replayed", cache_hit=True, error_codes=[]
            ),
            DependencyStatus(
                dependency="semantic_scholar",
                state="replayed",
                cache_hit=True,
                error_codes=[],
            ),
        ],
        last_authorized_probe_at=observed_at,
    )


class OutcomeService:
    def __init__(self, result: SearchExecutionResult | Exception) -> None:
        self.result = result
        self.requests: list[SearchRequest] = []

    async def execute(self, request: SearchRequest) -> SearchExecutionResult:
        self.requests.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _router(result: SearchExecutionResult | Exception) -> tuple[SearchServiceRouter, OutcomeService]:
    service = OutcomeService(result)
    return SearchServiceRouter(replay_service=service, readiness=_readiness()), service


def _execution_success(*, partial: bool = False) -> SearchExecutionResult:
    return SearchExecutionResult(
        outcome=SearchSuccess(
            response=_structured_response().model_copy(update={"is_partial": partial})
        ),
        diagnostics=[],
        business_result_sha256=None,
    )


def _execution_failure(code: str) -> SearchExecutionResult:
    return SearchExecutionResult(
        outcome=SearchFailure(
            query_id="q1",
            run_id="router-run-1",
            error=SearchErrorResponse(
                code=cast(Any, code),
                detail="private failure detail",
                retryable=False,
                run_id="router-run-1",
            ),
            usage=UsageActual(),
            stop_reason=code,
        ),
        diagnostics=[],
        business_result_sha256=None,
    )


async def _request(
    application: object,
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, json=json)


def test_live_is_always_ok() -> None:
    response = asyncio.run(_request(create_app(), "GET", "/health/live"))

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_search_request_returns_fixed_safe_400_without_service_call() -> None:
    router, service = _router(_execution_success())

    response = asyncio.run(
        _request(
            create_app(router),
            "POST",
            "/v1/search",
            json={"query_id": "q1", "query": "valid", "extra": True},
        )
    )

    assert response.status_code == 400
    assert response.json() == {
        "code": "invalid_request",
        "detail": _SAFE_DETAILS["invalid_request"],
        "retryable": False,
        "run_id": None,
    }
    assert service.requests == []


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("invalid_request", 400),
        ("live_not_authorized", 403),
        ("config_mismatch", 409),
        ("validation_attempt_conflict", 409),
        ("budget_exhausted", 429),
        ("snapshot_unavailable", 503),
        ("dependency_failure", 503),
        ("integrity_failure", 500),
        ("internal_error", 500),
    ],
)
def test_search_maps_each_typed_failure_to_its_stable_safe_http_body(
    code: str,
    status: int,
) -> None:
    router, _ = _router(_execution_failure(code))

    response = asyncio.run(
        _request(
            create_app(router),
            "POST",
            "/v1/search",
            json={"query_id": "q1", "query": "graph retrieval"},
        )
    )

    assert response.status_code == status
    assert response.json() == {
        "code": code,
        "detail": _SAFE_DETAILS[code],
        "retryable": code in {"snapshot_unavailable", "dependency_failure"},
        "run_id": "router-run-1",
    }
    assert "private failure detail" not in response.text


def test_search_maps_unexpected_exception_to_fixed_internal_error() -> None:
    router, _ = _router(RuntimeError("private exception detail"))

    response = asyncio.run(
        _request(
            create_app(router),
            "POST",
            "/v1/search",
            json={"query_id": "q1", "query": "graph retrieval"},
        )
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "detail": _SAFE_DETAILS["internal_error"],
        "retryable": False,
        "run_id": None,
    }
    assert "private exception detail" not in response.text


def test_partial_success_is_a_200_structured_response() -> None:
    router, _ = _router(_execution_success(partial=True))

    response = asyncio.run(
        _request(
            create_app(router),
            "POST",
            "/v1/search",
            json={"query_id": "q1", "query": "graph retrieval"},
        )
    )

    assert response.status_code == 200
    assert StructuredSearchResponse.model_validate(response.json()).is_partial is True


def test_ready_returns_cached_mode_snapshot_and_dependency_state_without_calls() -> None:
    observed_at = datetime(2026, 8, 3, 9, 30, tzinfo=UTC)
    service = OutcomeService(_execution_success())
    router = SearchServiceRouter(replay_service=service, readiness=_readiness(observed_at=observed_at))

    response = asyncio.run(_request(create_app(router), "GET", "/health/ready"))

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "execution_mode": "replay",
        "snapshot_set_id": "bound-snapshot-v1",
        "dependencies": [
            {"dependency": "llm", "state": "replayed", "cache_hit": True, "error_codes": []},
            {"dependency": "openalex", "state": "replayed", "cache_hit": True, "error_codes": []},
            {
                "dependency": "semantic_scholar",
                "state": "replayed",
                "cache_hit": True,
                "error_codes": [],
            },
        ],
        "last_authorized_probe_at": "2026-08-03T09:30:00Z",
    }
    assert service.requests == []


def test_search_openapi_declares_typed_error_json_schema() -> None:
    schema = create_app().openapi()

    typed_error = schema["paths"]["/v1/search"]["post"]["responses"]["503"]

    assert typed_error["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SearchErrorResponse"
    }
