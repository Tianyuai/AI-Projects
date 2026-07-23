from __future__ import annotations

import asyncio

import pytest

from paper_search.api.contracts import BudgetProfile, SearchRequest
from paper_search.api.service import MockApiSearchService
from paper_search.domain.models import (
    Paper,
    QueryAnalysisResult,
    QuerySpec,
    SearchPlan,
    SubQuery,
    UsageActual,
)
from paper_search.pipeline.orchestrator import MinimalSearchResult


def _minimal_result() -> MinimalSearchResult:
    spec = QuerySpec(
        original_query="graph retrieval",
        research_goal="find graph retrieval papers",
    )
    analysis = QueryAnalysisResult(
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
    )
    return MinimalSearchResult(
        query_analysis=analysis,
        papers=[
            Paper(
                canonical_id="openalex:W1",
                title="Synthetic Paper",
                openalex_id="W1",
                sources=["openalex"],
            )
        ],
        provider_results={},
        trace=[{"step": "fuse", "count": 1}],
        usage=UsageActual(search_api_calls=1, llm_calls=1, cost_cny=0.1),
        stop_reason="completed",
        is_partial=False,
        warnings=[],
        config_hash="sha256:" + "a" * 64,
        prompt_version="query-analyze-v1",
    )


class RecordingOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def run(
        self,
        query: str,
        *,
        max_provider_results: int,
    ) -> MinimalSearchResult:
        self.calls.append((query, max_provider_results))
        return _minimal_result()


class RecordingFactory:
    def __init__(self) -> None:
        self.profiles: list[BudgetProfile] = []
        self.instances: list[RecordingOrchestrator] = []

    def __call__(self, profile: BudgetProfile) -> RecordingOrchestrator:
        instance = RecordingOrchestrator()
        self.profiles.append(profile)
        self.instances.append(instance)
        return instance


def test_service_forwards_profile_and_identity_and_suppresses_trace() -> None:
    factory = RecordingFactory()
    service = MockApiSearchService(
        factory,
        git_sha=" abc1234 ",
        max_provider_results=7,
    )

    response = asyncio.run(
        service(
            SearchRequest(
                query_id="q1",
                query="graph retrieval",
                budget_profile="low",
                include_trace=False,
            )
        )
    )

    assert factory.profiles == ["low"]
    assert factory.instances[0].calls == [("graph retrieval", 7)]
    assert response.query_id == "q1"
    assert response.git_sha == "abc1234"
    assert response.selected_paper_ids == ["openalex:W1"]
    assert response.search_trace == []


def test_service_creates_a_fresh_orchestrator_for_every_request() -> None:
    factory = RecordingFactory()
    service = MockApiSearchService(
        factory,
        git_sha="abc1234",
        max_provider_results=5,
    )

    first = asyncio.run(service(SearchRequest(query_id="q1", query="one")))
    second = asyncio.run(service(SearchRequest(query_id="q2", query="two")))

    assert factory.profiles == ["balanced", "balanced"]
    assert len(factory.instances) == 2
    assert factory.instances[0] is not factory.instances[1]
    assert factory.instances[0].calls == [("one", 5)]
    assert factory.instances[1].calls == [("two", 5)]
    assert first.query_id == "q1"
    assert second.query_id == "q2"
    assert second.search_trace == [{"step": "fuse", "count": 1}]


@pytest.mark.parametrize(
    ("git_sha", "max_provider_results", "message"),
    [
        ("", 5, "git_sha must not be blank"),
        ("abc1234", 0, "max_provider_results must be positive"),
        ("abc1234", -1, "max_provider_results must be positive"),
    ],
)
def test_service_rejects_invalid_fixed_composition(
    git_sha: str,
    max_provider_results: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MockApiSearchService(
            RecordingFactory(),
            git_sha=git_sha,
            max_provider_results=max_provider_results,
        )
