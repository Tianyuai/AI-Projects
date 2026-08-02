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
    LiveSearchService,
    MockApiSearchService,
    OrchestratorFactory,
    SearchExecutionService,
    SearchOrchestrator,
)
from paper_search.api.routing import SearchServiceRouter


__all__ = [
    "BudgetProfile",
    "LiveHealthResponse",
    "LiveSearchService",
    "MockApiSearchService",
    "OrchestratorFactory",
    "ProviderHealthStatus",
    "ReadyHealthResponse",
    "SearchOrchestrator",
    "SearchRequest",
    "SearchExecutionService",
    "SearchServiceRouter",
    "UnavailableResponse",
    "app",
    "create_app",
]
