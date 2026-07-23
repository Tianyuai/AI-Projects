"""Strict HTTP request and health contracts."""

from typing import Literal, TypeAlias

from paper_search.domain.models import DomainModel, NonEmptyStr


BudgetProfile: TypeAlias = Literal["low", "balanced"]
ProviderHealthStatus: TypeAlias = Literal["ready", "degraded"]


class SearchRequest(DomainModel):
    query_id: NonEmptyStr
    query: NonEmptyStr
    budget_profile: BudgetProfile = "balanced"
    include_trace: bool = True


class LiveHealthResponse(DomainModel):
    status: Literal["ok"] = "ok"


class ReadyHealthResponse(DomainModel):
    status: Literal["ready", "degraded"]
    providers: dict[NonEmptyStr, ProviderHealthStatus]


class UnavailableResponse(DomainModel):
    detail: Literal["search temporarily unavailable"] = (
        "search temporarily unavailable"
    )
