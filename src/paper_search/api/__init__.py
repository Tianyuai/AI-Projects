"""Dependency-injected HTTP contracts for offline mock search."""

from paper_search.api.app import app, create_app
from paper_search.api.contracts import (
    BudgetProfile,
    LiveHealthResponse,
    ProviderHealthStatus,
    ReadyHealthResponse,
    SearchRequest,
    UnavailableResponse,
)
from paper_search.api.service import (
    MockApiSearchService,
    OrchestratorFactory,
    RequestScopedLiveSearchService,
    SearchExecutionService,
    SearchOrchestrator,
)
from paper_search.api.routing import SearchServiceRouter


__all__ = [
    "BudgetProfile",
    "LiveHealthResponse",
    "MockApiSearchService",
    "OrchestratorFactory",
    "ProviderHealthStatus",
    "RequestScopedLiveSearchService",
    "ReadyHealthResponse",
    "SearchOrchestrator",
    "SearchRequest",
    "SearchExecutionService",
    "SearchServiceRouter",
    "UnavailableResponse",
    "app",
    "create_app",
]
