"""Async, budgeted one-hop citation expansion over a scholarly provider."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

from pydantic import Field

from paper_search.application.contracts import DependencyDiagnostic, SnapshotRef
from paper_search.application.experiments import OptionalStageUnavailableError
from paper_search.control.budget import HardBudgetController, ReservationError
from paper_search.domain.models import (
    BudgetReservation,
    CitationExpansion,
    ErrorDetail,
    Paper,
    ProviderPaperId,
    ProviderResult,
    UsageActual,
    UsageEstimate,
)
from paper_search.graph.citation_expand import (
    CitationExpansionResult,
    expand_one_hop,
)
from paper_search.retrieval.base import SearchProvider


class AsyncCitationExpansionStage(Protocol):
    async def expand(
        self,
        seeds: list[Paper],
        *,
        controller: HardBudgetController,
    ) -> CitationExpansionResult: ...


class CitationExpansionUnavailableError(OptionalStageUnavailableError):
    """The provider could not produce a trustworthy citation expansion."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: DependencyDiagnostic | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


class ProviderCitationExpansionResult(CitationExpansionResult):
    snapshot_refs: list[SnapshotRef] = Field(default_factory=list)
    diagnostics: list[DependencyDiagnostic] = Field(default_factory=list)


def _settle_or_verify(
    controller: HardBudgetController,
    reservation: BudgetReservation,
    result: ProviderResult[Any],
) -> None:
    terminal = controller.terminal_outcome(reservation)
    if terminal is None:
        controller.settle(reservation, result.usage)
        return
    mode, recorded = terminal
    if mode != "settled" or recorded != result.usage:
        raise ReservationError("citation settlement receipt does not match result")


def _snapshot_refs(result: ProviderResult[Any]) -> list[SnapshotRef]:
    encoded = result.provenance.get("snapshot_refs", "[]")
    try:
        raw_refs = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("invalid citation snapshot provenance") from error
    if not isinstance(raw_refs, list):
        raise ValueError("invalid citation snapshot provenance")
    return [SnapshotRef.model_validate(raw) for raw in raw_refs]


def _provider_id_map(papers: list[Paper]) -> dict[ProviderPaperId, str]:
    mapping: dict[ProviderPaperId, str] = {}
    for paper in papers:
        if paper.openalex_id is not None:
            mapping[ProviderPaperId(provider="openalex", value=paper.openalex_id)] = (
                paper.canonical_id
            )
        if paper.semantic_scholar_id is not None:
            mapping[
                ProviderPaperId(
                    provider="semantic_scholar",
                    value=paper.semantic_scholar_id,
                )
            ] = paper.canonical_id
    return mapping


def _diagnostic(result: ProviderResult[Any]) -> DependencyDiagnostic:
    return DependencyDiagnostic(
        dependency="semantic_scholar",
        endpoint="citation_expansion",
        model_id=None,
        usage=result.usage,
        latency_ms=result.latency_ms,
        cache_hit=result.cache_hit,
        snapshot_refs=_snapshot_refs(result),
        errors=[
            ErrorDetail(
                code=error.code,
                message="Citation dependency reported an error",
                retryable=error.retryable,
                provider="semantic_scholar",
            )
            for error in result.errors
        ],
    )


class ProviderCitationExpansionStage:
    def __init__(
        self,
        *,
        provider: SearchProvider,
        call_estimate: UsageEstimate,
        per_direction_limit: int = 2,
        max_expanded: int = 2,
        action_id: str | None = None,
    ) -> None:
        if (
            type(per_direction_limit) is not int
            or not 1 <= per_direction_limit <= 100
        ):
            raise ValueError("per_direction_limit must be between 1 and 100")
        if type(max_expanded) is not int or max_expanded <= 0:
            raise ValueError("max_expanded must be a positive integer")
        if call_estimate.search_api_calls < 1:
            raise ValueError("citation calls require a search API estimate")
        if action_id is not None and not action_id.strip():
            raise ValueError("citation action ID must not be empty")
        self._provider = provider
        self._call_estimate = call_estimate
        self._per_direction_limit = per_direction_limit
        self._max_expanded = max_expanded
        self._action_id = action_id or "semantic_scholar"

    async def _call(
        self,
        *,
        direction: str,
        paper_id: ProviderPaperId,
        controller: HardBudgetController,
    ) -> ProviderResult[CitationExpansion]:
        reservation = controller.reserve(
            f"{self._action_id}.{direction}:{paper_id.value}",
            self._call_estimate,
        )
        result: ProviderResult[CitationExpansion] | None = None
        try:
            if direction == "references":
                result = await self._provider.references(
                    paper_id,
                    self._per_direction_limit,
                    reservation,
                )
            else:
                result = await self._provider.citations(
                    paper_id,
                    self._per_direction_limit,
                    reservation,
                )
            _settle_or_verify(controller, reservation, result)
        except asyncio.CancelledError:
            if controller.terminal_outcome(reservation) is None:
                controller.fail_closed(reservation, UsageActual())
            raise
        except ReservationError:
            if controller.terminal_outcome(reservation) is None:
                controller.fail_closed(
                    reservation,
                    result.usage if result is not None else UsageActual(),
                )
            raise
        except Exception:
            if controller.terminal_outcome(reservation) is None:
                controller.release(reservation)
            raise
        diagnostic = _diagnostic(result)
        if result.errors:
            raise CitationExpansionUnavailableError(
                "citation provider returned structured errors",
                diagnostic=diagnostic,
            )
        return result

    async def expand(
        self,
        seeds: list[Paper],
        *,
        controller: HardBudgetController,
    ) -> CitationExpansionResult:
        if not seeds:
            raise ValueError("seeds must not be empty")
        active_seeds = seeds[: controller.budget.max_citation_seeds]
        if not active_seeds:
            return ProviderCitationExpansionResult(
                papers=list(seeds),
                edges=[],
                skipped_edge_count=0,
                truncated=False,
                warnings=[],
                snapshot_refs=[],
                diagnostics=[],
            )
        expansions: list[CitationExpansion] = []
        refs: list[SnapshotRef] = []
        diagnostics: list[DependencyDiagnostic] = []
        for seed in active_seeds:
            if seed.semantic_scholar_id is None:
                continue
            paper_id = ProviderPaperId(
                provider="semantic_scholar",
                value=seed.semantic_scholar_id,
            )
            for direction in ("references", "citations"):
                result = await self._call(
                    direction=direction,
                    paper_id=paper_id,
                    controller=controller,
                )
                expansions.append(result.data)
                refs.extend(_snapshot_refs(result))
                diagnostics.append(_diagnostic(result))
        if not expansions:
            return ProviderCitationExpansionResult(
                papers=list(seeds),
                edges=[],
                skipped_edge_count=0,
                truncated=False,
                warnings=[],
                snapshot_refs=[],
                diagnostics=[],
            )
        expanded_papers = [
            paper for expansion in expansions for paper in expansion.papers
        ]
        combined = CitationExpansion(
            papers=expanded_papers,
            raw_edges=[
                edge for expansion in expansions for edge in expansion.raw_edges
            ],
        )
        resolved = expand_one_hop(
            active_seeds,
            combined,
            _provider_id_map([*active_seeds, *expanded_papers]),
            max_seeds=len(active_seeds),
            max_expanded=self._max_expanded,
        )
        return ProviderCitationExpansionResult(
            **resolved.model_dump(mode="python"),
            snapshot_refs=refs,
            diagnostics=diagnostics,
        )


__all__ = [
    "AsyncCitationExpansionStage",
    "CitationExpansionUnavailableError",
    "ProviderCitationExpansionResult",
    "ProviderCitationExpansionStage",
]
