"""Request-scoped composition for the mock search orchestrator."""

from collections.abc import Callable
from typing import Protocol

from paper_search.api.contracts import BudgetProfile, SearchRequest
from paper_search.application import StructuredSearchResponse
from paper_search.application.contracts import (
    SearchExecutionResult,
    SearchSuccess,
)
from paper_search.pipeline.orchestrator import MinimalSearchResult
from paper_search.pipeline.response import to_structured_response


class SearchOrchestrator(Protocol):
    async def run(
        self,
        query: str,
        *,
        max_provider_results: int,
    ) -> MinimalSearchResult: ...


OrchestratorFactory = Callable[[BudgetProfile], SearchOrchestrator]


class SearchExecutionService(Protocol):
    """The canonical application service shape consumed by API routing."""

    async def execute(self, request: SearchRequest) -> SearchExecutionResult: ...


class RequestScopedLiveSearchService(Protocol):
    """Complete one live execution and its validated atomic publication."""

    async def execute_and_publish(
        self,
        request: SearchRequest,
    ) -> SearchExecutionResult: ...


class MockApiSearchService:
    def __init__(
        self,
        orchestrator_factory: OrchestratorFactory,
        *,
        git_sha: str,
        max_provider_results: int,
    ) -> None:
        if not git_sha.strip():
            raise ValueError("git_sha must not be blank")
        if max_provider_results <= 0:
            raise ValueError("max_provider_results must be positive")
        self._orchestrator_factory = orchestrator_factory
        self._git_sha = git_sha.strip()
        self._max_provider_results = max_provider_results

    async def __call__(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResponse:
        orchestrator = self._orchestrator_factory(request.budget_profile)
        result = await orchestrator.run(
            request.query,
            max_provider_results=self._max_provider_results,
        )
        response = to_structured_response(
            result,
            query_id=request.query_id,
            git_sha=self._git_sha,
        )
        if not request.include_trace:
            return response.model_copy(update={"search_trace": []})
        return response

    async def execute(self, request: SearchRequest) -> SearchExecutionResult:
        """Expose the mock path through the same canonical result contract."""
        return SearchExecutionResult(
            outcome=SearchSuccess(response=await self(request)),
            diagnostics=[],
            business_result_sha256=None,
        )
