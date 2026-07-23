from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import ErrorDetail, Paper, ProviderResult, SearchBudget, UsageActual, UsageEstimate
from paper_search.pipeline.orchestrator import MockSearchOrchestrator


def _budget(**updates: object) -> SearchBudget:
    values = {
        "max_search_api_calls": 6,
        "target_search_api_calls": 1,
        "max_llm_calls": 2,
        "target_llm_calls": 1,
        "max_total_tokens": 100,
        "max_cost_cny": 1.0,
        "max_elapsed_seconds": 2,
        "soft_deadline_seconds": 1,
    }
    values.update(updates)
    return SearchBudget.model_validate(values)


def _result(provider: str, data: Any, usage: UsageActual, *, failed: bool = False) -> ProviderResult[Any]:
    return ProviderResult[Any](
        data=data,
        usage=usage,
        provenance={
            "provider": provider,
            "endpoint": "/synthetic",
            "model_id": "fixture",
            "requested_at": datetime(2026, 7, 23, tzinfo=UTC).isoformat(),
            "response_hash": f"sha256:{provider}",
        },
        cache_hit=False,
        latency_ms=0,
        errors=(
            [ErrorDetail(code="timeout", message="synthetic", retryable=True, provider=provider)]
            if failed
            else []
        ),
    )


class FakeAnalyzer:
    def __init__(self, events: list[str], *, elapsed_ms: int = 0) -> None:
        self.events = events
        self.elapsed_ms = elapsed_ms

    async def __call__(self, query: str, _: object) -> ProviderResult[dict[str, object]]:
        self.events.append("analyze")
        return _result(
            "llm",
            {
                "query_spec": {"original_query": query, "research_goal": "find papers"},
                "search_plan": {
                "subqueries": [
                    {
                        "query_id": "model-1",
                        "text": f"{query} openalex",
                        "query_type": "exact",
                        "target_constraints": ["papers"],
                        "priority": 1,
                        "provider_hint": "openalex",
                    },
                    {
                        "query_id": "model-2",
                        "text": f"{query} semantic",
                        "query_type": "decomposed",
                        "target_constraints": ["papers"],
                        "priority": 2,
                        "provider_hint": "semantic_scholar",
                    },
                    {
                        "query_id": "model-3",
                        "text": query,
                        "query_type": "expanded",
                        "target_constraints": ["papers"],
                        "priority": 3,
                        "provider_hint": "either",
                    },
                ],
                    "inherited_hard_filters": {},
                    "rationale": "fixture",
                },
            },
            UsageActual(llm_calls=1, cost_cny=0.1, elapsed_ms=self.elapsed_ms),
        )


class FailedAnalyzer:
    def __init__(self, events: list[str], *, raises: bool = False) -> None:
        self.events = events
        self.raises = raises

    async def __call__(
        self, query: str, _: object
    ) -> ProviderResult[dict[str, object]]:
        self.events.append("analyze")
        if self.raises:
            raise TimeoutError("synthetic analyzer timeout")
        return _result(
            "llm",
            {},
            UsageActual(llm_calls=1, cost_cny=0.1),
            failed=True,
        )


class FakeProvider:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        failed: bool = False,
        empty: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.failed = failed
        self.empty = empty

    async def search(self, query: str, filters: dict[str, object], limit: int, reservation: object) -> ProviderResult[list[Paper]]:
        assert query
        assert filters == {}
        assert limit == 5
        assert reservation is not None
        self.events.append(self.name)
        paper = Paper(
            canonical_id="openalex:W1" if self.name == "openalex" else "s2:S1",
            title=f"{self.name} paper",
            openalex_id="W1" if self.name == "openalex" else None,
            semantic_scholar_id="S1" if self.name != "openalex" else None,
            sources=[self.name],
        )
        return _result(
            self.name,
            [] if self.failed or self.empty else [paper],
            UsageActual(search_api_calls=1),
            failed=self.failed,
        )


def test_orchestrator_orders_budgeted_mock_pipeline_and_records_trace() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events), "semantic_scholar": FakeProvider("semantic_scholar", events)},
        config_hash="sha256:" + "b" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert events == ["analyze", "openalex", "semantic_scholar", "openalex", "semantic_scholar"]
    assert [paper.canonical_id for paper in result.papers] == ["openalex:W1", "s2:S1"]
    assert [item["step"] for item in result.trace] == [
        "analyze",
        "retrieve",
        "retrieve",
        "retrieve",
        "retrieve",
        "deduplicate",
        "filter",
        "fuse",
    ]
    assert set(result.provider_results) == {"openalex", "semantic_scholar"}
    assert result.config_hash == "sha256:" + "b" * 64
    assert result.prompt_version == "query-analyze-v1"
    assert result.stop_reason == "completed"


def test_orchestrator_uses_rule_fallback_for_structured_analyzer_error() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FailedAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "e" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert result.query_analysis.query_spec.ambiguities == ["rules_only_fallback"]
    assert result.warnings[0] == "analysis: analyzer returned errors"
    assert result.is_partial is True
    assert "openalex" in events


def test_orchestrator_fails_closed_on_analyzer_exception_without_calling_provider() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget())
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FailedAnalyzer(events, raises=True),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "f" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert events == ["analyze"]
    assert result.query_analysis.query_spec.ambiguities == ["rules_only_fallback"]
    assert result.stop_reason == "hard_stop"
    assert result.is_partial is True
    assert result.warnings == ["analysis: dependency failure"]
    assert controller.stop_status() == "hard_stop"


def test_orchestrator_treats_all_empty_provider_results_as_completed() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events, empty=True)},
        config_hash="sha256:" + "1" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert result.papers == []
    assert result.stop_reason == "completed"
    assert result.is_partial is False
    assert result.warnings == []


def test_orchestrator_soft_stop_prevents_provider_calls() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events, elapsed_ms=1_000),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "2" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(
            llm_calls=1,
            cost_cny=0.1,
            elapsed_ms=1_000,
        ),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert events == ["analyze"]
    assert result.stop_reason == "soft_stop"
    assert result.is_partial is True
    assert result.warnings == ["openalex: budget unavailable"]


def test_orchestrator_returns_sibling_result_on_provider_failure_and_skips_calls_on_budget_stop() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget(max_search_api_calls=1, target_search_api_calls=1))
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events, failed=True), "semantic_scholar": FakeProvider("semantic_scholar", events)},
        config_hash="sha256:" + "c" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert events == ["analyze", "openalex"]
    assert result.papers == []
    assert result.is_partial is True
    assert result.stop_reason == "hard_stop"
    assert result.warnings == [
        "openalex: provider returned errors",
        "semantic_scholar: budget unavailable",
    ]


class OverrunProvider(FakeProvider):
    async def search(self, query: str, filters: dict[str, object], limit: int, reservation: object) -> ProviderResult[list[Paper]]:
        result = await super().search(query, filters, limit, reservation)
        return result.model_copy(update={"usage": UsageActual(search_api_calls=2)})


def test_orchestrator_fails_closed_when_a_provider_exceeds_its_reservation() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget())
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FakeAnalyzer(events),
        providers={"openalex": OverrunProvider("openalex", events)},
        config_hash="sha256:" + "d" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    try:
        asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))
    except RuntimeError:
        pass
    else:
        raise AssertionError("over-reservation usage must fail the orchestration")

    assert controller.stop_status() == "hard_stop"
