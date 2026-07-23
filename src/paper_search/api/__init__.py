"""Dependency-injected HTTP contracts for offline mock search."""

from paper_search.api.app import (
    ReadinessProbe,
    SearchService,
    app,
    create_app,
)
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
    SearchOrchestrator,
)


__all__ = [
    "BudgetProfile",
    "LiveHealthResponse",
    "MockApiSearchService",
    "OrchestratorFactory",
    "ProviderHealthStatus",
    "ReadinessProbe",
    "ReadyHealthResponse",
    "SearchOrchestrator",
    "SearchRequest",
    "SearchService",
    "UnavailableResponse",
    "app",
    "create_app",
]
