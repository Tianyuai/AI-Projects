"""Fixed offline dependencies for the Task 8C synthetic baseline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypeVar

from paper_search.api.contracts import BudgetProfile
from paper_search.api.service import MockApiSearchService
from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import (
    BudgetReservation,
    CitationExpansion,
    Paper,
    ProviderPaperId,
    ProviderResult,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.pipeline.orchestrator import MockSearchOrchestrator


_REQUESTED_AT = datetime(2026, 7, 24, tzinfo=UTC).isoformat()
_CONFIG_HASH = "sha256:" + "c" * 64
ResultT = TypeVar("ResultT")


def _result(
    provider: str,
    data: ResultT,
    usage: UsageActual,
) -> ProviderResult[ResultT]:
    return ProviderResult[ResultT](
        data=data,
        usage=usage,
        provenance={
            "provider": provider,
            "endpoint": "/synthetic",
            "model_id": "task8c-fixed-mock",
            "requested_at": _REQUESTED_AT,
            "response_hash": f"sha256:task8c-{provider}",
        },
        cache_hit=False,
        latency_ms=0,
        errors=[],
    )


class SyntheticAnalyzer:
    async def __call__(
        self,
        query: str,
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, object]]:
        del reservation
        return _result(
            "llm",
            {
                "query_spec": {
                    "original_query": query,
                    "research_goal": "synthetic baseline",
                },
                "search_plan": {
                    "subqueries": [
                        {
                            "query_id": "synthetic-sq-1",
                            "text": query,
                            "query_type": "exact",
                            "target_constraints": [],
                            "priority": 1,
                            "provider_hint": "either",
                        }
                    ],
                    "inherited_hard_filters": {},
                    "rationale": "fixed synthetic plan",
                },
            },
            UsageActual(llm_calls=1, cost_cny=0.1),
        )


class SyntheticProvider:
    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        del filters, limit, reservation
        papers = (
            []
            if "zero-result" in query
            else [
                Paper(
                    canonical_id="openalex:W100",
                    title="Synthetic Graph Retrieval Paper",
                    openalex_id="W100",
                    sources=["openalex"],
                )
            ]
        )
        return _result(
            "openalex",
            papers,
            UsageActual(search_api_calls=1),
        )

    async def references(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        del paper_id, limit, reservation
        return _result(
            "openalex",
            CitationExpansion(papers=[], raw_edges=[]),
            UsageActual(),
        )

    async def citations(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        del paper_id, limit, reservation
        return _result(
            "openalex",
            CitationExpansion(papers=[], raw_edges=[]),
            UsageActual(),
        )


def _budget(profile: BudgetProfile) -> SearchBudget:
    max_search_calls = 4 if profile == "balanced" else 3
    return SearchBudget(
        max_search_api_calls=max_search_calls,
        target_search_api_calls=1,
        max_llm_calls=2,
        target_llm_calls=1,
        max_total_tokens=100,
        max_cost_cny=1.0,
        max_elapsed_seconds=2,
        soft_deadline_seconds=1,
    )


class SyntheticOrchestratorFactory:
    def __init__(self) -> None:
        self.controllers: list[HardBudgetController] = []

    def __call__(
        self,
        profile: BudgetProfile,
    ) -> MockSearchOrchestrator:
        controller = HardBudgetController(_budget(profile))
        self.controllers.append(controller)
        return MockSearchOrchestrator(
            controller=controller,
            analyzer=SyntheticAnalyzer(),
            providers={"openalex": SyntheticProvider()},
            config_hash=_CONFIG_HASH,
            prompt_version="task8c-synthetic-v1",
            analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
            provider_estimate=UsageEstimate(search_api_calls=1),
        )


def build_synthetic_search_service(
    *,
    factory: SyntheticOrchestratorFactory | None = None,
) -> MockApiSearchService:
    """Build a service containing no real provider or environment boundary."""
    selected_factory = factory or SyntheticOrchestratorFactory()
    return MockApiSearchService(
        selected_factory,
        git_sha="synthetic-task8c",
        max_provider_results=5,
    )
