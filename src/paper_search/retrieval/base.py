"""Shared scholarly search provider protocol."""

from __future__ import annotations

from typing import Protocol

from paper_search.domain.models import (
    BudgetReservation,
    CitationExpansion,
    Paper,
    ProviderPaperId,
    ProviderResult,
)


class SearchProvider(Protocol):
    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]: ...

    async def references(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]: ...

    async def citations(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]: ...
