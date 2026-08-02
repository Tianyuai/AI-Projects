from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
import pytest

from paper_search.api.app import create_app
from paper_search.api.contracts import BudgetProfile, SearchRequest
from paper_search.api.routing import SearchServiceRouter
from paper_search.api.service import MockApiSearchService
from paper_search.application.contracts import (
    ReadyHealthResponse,
    SearchExecutionResult,
    SearchSuccess,
)
from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import (
    DependencyStatus,
    ErrorDetail,
    Paper,
    ProviderResult,
    SearchBudget,
    StructuredSearchResponse,
    UsageActual,
    UsageEstimate,
)
from paper_search.pipeline.orchestrator import MockSearchOrchestrator


Scenario = Literal[
    "success",
    "empty",
    "provider_failure",
    "analysis_failure",
    "budget_exhausted",
    "soft_stop",
]


def _result(
    provider: str,
    data: Any,
    usage: UsageActual,
    *,
    failed: bool = False,
) -> ProviderResult[Any]:
    return ProviderResult[Any](
        data=data,
        usage=usage,
        provenance={
            "provider": provider,
            "endpoint": "/synthetic",
            "model_id": "fixture",
            "requested_at": datetime(2026, 7, 24, tzinfo=UTC).isoformat(),
            "response_hash": f"sha256:{provider}",
        },
        cache_hit=False,
        latency_ms=0,
        errors=(
            [
                ErrorDetail(
                    code="synthetic_failure",
                    message="synthetic",
                    retryable=True,
                    provider=provider,
                )
            ]
            if failed
            else []
        ),
    )


class ScenarioAnalyzer:
    def __init__(self, scenario: Scenario, events: list[str]) -> None:
        self.scenario = scenario
        self.events = events

    async def __call__(
        self,
        query: str,
        _: object,
    ) -> ProviderResult[dict[str, object]]:
        self.events.append("analyze")
        if self.scenario == "analysis_failure":
            raise TimeoutError("synthetic analyzer timeout")
        elapsed_ms = 1_000 if self.scenario == "soft_stop" else 0
        return _result(
            "llm",
            {
                "query_spec": {
                    "original_query": query,
                    "research_goal": "find papers",
                },
                "search_plan": {
                    "subqueries": [
                        {
                            "query_id": "sq-1",
                            "text": query,
                            "query_type": "exact",
                            "target_constraints": [],
                            "priority": 1,
                            "provider_hint": "either",
                        }
                    ],
                    "inherited_hard_filters": {},
                    "rationale": "synthetic",
                },
            },
            UsageActual(
                llm_calls=1,
                cost_cny=0.1,
                elapsed_ms=elapsed_ms,
            ),
        )


class ScenarioProvider:
    def __init__(
        self,
        name: Literal["openalex", "semantic_scholar"],
        scenario: Scenario,
        events: list[str],
    ) -> None:
        self.name = name
        self.scenario = scenario
        self.events = events

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: object,
    ) -> ProviderResult[list[Paper]]:
        assert query
        assert filters == {}
        assert limit == 5
        assert reservation is not None
        self.events.append(self.name)
        failed = self.scenario == "provider_failure" and self.name == "openalex"
        empty = self.scenario == "empty"
        paper = Paper(
            canonical_id=(
                "openalex:W1"
                if self.name == "openalex"
                else "s2:S1"
            ),
            title=f"{self.name} paper",
            openalex_id="W1" if self.name == "openalex" else None,
            semantic_scholar_id=(
                "S1" if self.name == "semantic_scholar" else None
            ),
            sources=[self.name],
        )
        return _result(
            self.name,
            [] if failed or empty else [paper],
            UsageActual(search_api_calls=1),
            failed=failed,
        )


def _budget(scenario: Scenario) -> SearchBudget:
    max_search_api_calls = 7 if scenario == "provider_failure" else 4
    target_search_api_calls = 1
    if scenario == "budget_exhausted":
        max_search_api_calls = 0
        target_search_api_calls = 0
    return SearchBudget(
        max_search_api_calls=max_search_api_calls,
        target_search_api_calls=target_search_api_calls,
        max_llm_calls=2,
        target_llm_calls=1,
        max_total_tokens=100,
        max_cost_cny=1.0,
        max_elapsed_seconds=2,
        soft_deadline_seconds=1,
    )


class ScenarioFactory:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.profiles: list[BudgetProfile] = []
        self.controllers: list[HardBudgetController] = []
        self.events: list[list[str]] = []

    def __call__(
        self,
        profile: BudgetProfile,
    ) -> MockSearchOrchestrator:
        events: list[str] = []
        controller = HardBudgetController(_budget(self.scenario))
        providers: dict[str, ScenarioProvider] = {
            "openalex": ScenarioProvider(
                "openalex",
                self.scenario,
                events,
            )
        }
        if self.scenario == "provider_failure":
            providers["semantic_scholar"] = ScenarioProvider(
                "semantic_scholar",
                self.scenario,
                events,
            )
        elapsed_ms = 1_000 if self.scenario == "soft_stop" else 0
        orchestrator = MockSearchOrchestrator(
            controller=controller,
            analyzer=ScenarioAnalyzer(self.scenario, events),
            providers=providers,
            config_hash="sha256:" + "b" * 64,
            prompt_version="query-analyze-v1",
            analysis_estimate=UsageEstimate(
                llm_calls=1,
                cost_cny=0.1,
                elapsed_ms=elapsed_ms,
            ),
            provider_estimate=UsageEstimate(search_api_calls=1),
        )
        self.profiles.append(profile)
        self.controllers.append(controller)
        self.events.append(events)
        return orchestrator


async def _post(
    application: object,
    *,
    query_id: str = "q1",
    mode: Literal["replay", "live"] = "replay",
) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        return await client.post(
            "/v1/search",
            json={
                "query_id": query_id,
                "query": "graph retrieval",
                "budget_profile": "low",
                "mode": mode,
            },
        )


@pytest.mark.parametrize(
    (
        "scenario",
        "stop_reason",
        "is_partial",
        "selected_paper_ids",
        "expected_events",
    ),
    [
        (
            "success",
            "completed",
            False,
            ["openalex:W1"],
            ["analyze", "openalex", "openalex"],
        ),
        (
            "empty",
            "completed",
            False,
            [],
            ["analyze", "openalex", "openalex"],
        ),
        (
            "provider_failure",
            "completed",
            True,
            ["s2:S1"],
            [
                "analyze",
                "openalex",
                "semantic_scholar",
                "openalex",
                "semantic_scholar",
            ],
        ),
        ("analysis_failure", "hard_stop", True, [], ["analyze"]),
        ("budget_exhausted", "hard_stop", True, [], []),
        ("soft_stop", "soft_stop", True, [], ["analyze"]),
    ],
)
def test_mock_api_covers_required_end_to_end_scenarios(
    scenario: Scenario,
    stop_reason: str,
    is_partial: bool,
    selected_paper_ids: list[str],
    expected_events: list[str],
) -> None:
    factory = ScenarioFactory(scenario)
    service = MockApiSearchService(
        factory,
        git_sha="abc1234",
        max_provider_results=5,
    )
    application = create_app(
        SearchServiceRouter(
            replay_service=service,
            readiness=_router_readiness(),
        )
    )

    raw_response = asyncio.run(_post(application))
    response = StructuredSearchResponse.model_validate(raw_response.json())

    assert raw_response.status_code == 200
    assert response.stop_reason == stop_reason
    assert response.is_partial is is_partial
    assert response.selected_paper_ids == selected_paper_ids
    assert factory.profiles == ["low"]
    assert factory.events == [expected_events]


def test_api_requests_use_fresh_budget_controllers() -> None:
    factory = ScenarioFactory("success")
    service = MockApiSearchService(
        factory,
        git_sha="abc1234",
        max_provider_results=5,
    )
    application = create_app(
        SearchServiceRouter(
            replay_service=service,
            readiness=_router_readiness(),
        )
    )

    first = StructuredSearchResponse.model_validate(
        asyncio.run(_post(application, query_id="q1")).json()
    )
    second = StructuredSearchResponse.model_validate(
        asyncio.run(_post(application, query_id="q2")).json()
    )

    assert first.stop_reason == "completed"
    assert second.stop_reason == "completed"
    assert len(factory.controllers) == 2
    assert factory.controllers[0] is not factory.controllers[1]
    assert [controller.committed_usage.search_api_calls for controller in factory.controllers] == [
        2,
        2,
    ]


def _router_readiness() -> ReadyHealthResponse:
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
        last_authorized_probe_at=None,
    )


class _RouterService:
    def __init__(self, label: str, events: list[str]) -> None:
        self.label = label
        self.events = events

    async def execute(self, request: object) -> SearchExecutionResult:
        self.events.append(f"execute:{self.label}")
        response = StructuredSearchResponse.model_validate(
            {
                **_structured_success_response().model_dump(mode="python"),
                "query_id": getattr(request, "query_id"),
            }
        )
        return SearchExecutionResult(
            outcome=SearchSuccess(response=response),
            diagnostics=[],
            business_result_sha256=None,
        )



class _LiveRouterService:
    def __init__(
        self,
        events: list[str],
        *,
        execution_mode: Literal["replay", "live"] = "live",
        publication_error: bool = False,
    ) -> None:
        self._events = events
        self._execution_mode = execution_mode
        self._publication_error = publication_error

    async def execute_and_publish(self, request: SearchRequest) -> SearchExecutionResult:
        self._events.append("execute:live")
        if self._publication_error:
            self._events.append("publish:live")
            raise RuntimeError("private publication failure")
        response = _structured_success_response().model_copy(
            update={
                "execution_mode": self._execution_mode,
                "query_id": request.query_id,
            }
        )
        execution = SearchExecutionResult(
            outcome=SearchSuccess(response=response),
            diagnostics=[],
            business_result_sha256=None,
        )
        self._events.append("publish:live")
        return execution


def _structured_success_response() -> StructuredSearchResponse:
    return StructuredSearchResponse.model_validate(
        {
            "run_id": "router-run-1",
            "query_id": "q1",
            "execution_mode": "replay",
            "snapshot_set_id": "bound-snapshot-v1",
            "snapshot_captured_at": None,
            "query_analysis": {
                "query_spec": {
                    "original_query": "graph retrieval",
                    "research_goal": "find papers",
                },
                "search_plan": {
                    "subqueries": [
                        {
                            "query_id": "sq-1",
                            "text": "graph retrieval",
                            "query_type": "exact",
                            "target_constraints": [],
                            "priority": 1,
                            "provider_hint": "either",
                        }
                    ],
                    "inherited_hard_filters": {},
                    "rationale": "synthetic",
                },
            },
            "selected_paper_ids": [],
            "high_relevance": [],
            "partial_relevance": [],
            "citation_edges": [],
            "search_trace": [],
            "usage": {},
            "stop_reason": "completed",
            "is_partial": False,
            "planner_fallback": False,
            "planner_status": "primary",
            "dependency_status": [
                {"dependency": "llm", "state": "replayed", "cache_hit": True, "error_codes": []},
                {"dependency": "openalex", "state": "replayed", "cache_hit": True, "error_codes": []},
                {
                    "dependency": "semantic_scholar",
                    "state": "replayed",
                    "cache_hit": True,
                    "error_codes": [],
                },
            ],
            "warnings": [],
            "prompt_version": "query-analyze-v1",
            "config_hash": "sha256:" + "a" * 64,
            "git_sha": "abc1234",
        }
    )


def test_router_binds_replay_and_requires_all_live_authorization_keys() -> None:
    events: list[str] = []
    replay = _RouterService("replay", events)
    live = _LiveRouterService(events)
    factory_calls: list[str] = []

    def live_factory() -> _LiveRouterService:
        factory_calls.append("live")
        return live

    router = SearchServiceRouter(
        replay_service=replay,
        readiness=_router_readiness(),
        live_service_factory=live_factory,
        runtime_allow_live=True,
        server_live_authorized=True,
    )
    replay_result = asyncio.run(
        router.execute(
            SearchRequest(query_id="replay", query="graph retrieval", mode="replay")
        )
    )
    live_result = asyncio.run(
        router.execute(
            SearchRequest(query_id="live", query="graph retrieval", mode="live")
        )
    )

    assert isinstance(replay_result.outcome, SearchSuccess)
    assert isinstance(live_result.outcome, SearchSuccess)
    assert events == ["execute:replay", "execute:live", "publish:live"]
    assert factory_calls == ["live"]

    missing_factory = SearchServiceRouter(
        replay_service=replay,
        readiness=_router_readiness(),
        live_service_factory=None,
        runtime_allow_live=True,
        server_live_authorized=True,
    )
    lock_disallowed = SearchServiceRouter(
        replay_service=replay,
        readiness=_router_readiness(),
        live_service_factory=live_factory,
        runtime_allow_live=False,
        server_live_authorized=True,
    )
    server_disallowed = SearchServiceRouter(
        replay_service=replay,
        readiness=_router_readiness(),
        live_service_factory=live_factory,
        runtime_allow_live=True,
        server_live_authorized=False,
    )
    for unavailable in (missing_factory, lock_disallowed, server_disallowed):
        result = asyncio.run(
            unavailable.execute(
                SearchRequest(query_id="live-denied", query="graph retrieval", mode="live")
            )
        )
        assert result.outcome.kind == "failure"
        assert result.outcome.error.code == "live_not_authorized"


def test_live_publication_failure_never_returns_http_success() -> None:
    events: list[str] = []
    live = _LiveRouterService(events, publication_error=True)
    router = SearchServiceRouter(
        replay_service=_RouterService("replay", events),
        readiness=_router_readiness(),
        live_service_factory=lambda: live,
        runtime_allow_live=True,
        server_live_authorized=True,
    )

    response = asyncio.run(_post(create_app(router), mode="live"))

    assert response.status_code == 500
    assert events == ["execute:live", "publish:live"]
    assert "private publication failure" not in response.text


def test_live_response_with_replay_mode_becomes_integrity_failure() -> None:
    events: list[str] = []
    live = _LiveRouterService(events, execution_mode="replay")
    router = SearchServiceRouter(
        replay_service=_RouterService("replay", events),
        readiness=_router_readiness(),
        live_service_factory=lambda: live,
        runtime_allow_live=True,
        server_live_authorized=True,
    )

    result = asyncio.run(
        router.execute(SearchRequest(query_id="q1", query="graph retrieval", mode="live"))
    )

    assert result.outcome.kind == "failure"
    assert result.outcome.error.code == "integrity_failure"
    assert events == ["execute:live", "publish:live"]
