"""Compatibility exports for strict application-layer HTTP contracts."""

from typing import Literal, TypeAlias

from paper_search.application.contracts import ReadyHealthResponse, SearchRequest
from paper_search.domain.models import DomainModel

__all__ = [
    "BudgetProfile",
    "LiveHealthResponse",
    "ProviderHealthStatus",
    "ReadyHealthResponse",
    "SearchRequest",
    "UnavailableResponse",
]


BudgetProfile: TypeAlias = Literal["low", "balanced"]
ProviderHealthStatus: TypeAlias = Literal["ready", "degraded"]

class LiveHealthResponse(DomainModel):
    status: Literal["ok"] = "ok"

class UnavailableResponse(DomainModel):
    detail: Literal["search temporarily unavailable"] = (
        "search temporarily unavailable"
    )
