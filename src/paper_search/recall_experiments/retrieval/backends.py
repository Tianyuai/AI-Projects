"""Replaceable budget-owning retrieval adapters for recall experiments."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import Field

from paper_search.control.budget import (
    BudgetExceededError,
    HardBudgetController,
    ReservationError,
)
from paper_search.control.pricing import ActualCostPricer
from paper_search.domain.models import (
    BudgetReservation,
    CitationExpansion,
    DomainModel,
    ErrorDetail,
    Paper,
    ProviderPaperId,
    ProviderResult,
    UsageActual,
    UsageEstimate,
)
from paper_search.graph.provider_stage import (
    CitationExpansionUnavailableError,
    ProviderCitationExpansionStage,
)
from paper_search.live_identity import LiveDependencyEvidence
from paper_search.recall_experiments.contracts import (
    CitationDirection,
    CitationExpandAction,
    RecallSearchAction,
    RetrievalActionResult,
    RetrievalExecutionContext,
    TextSearchAction,
    TitleSearchAction,
)
from paper_search.recall_experiments.identity import (
    LiveDependencyIdentity,
    dependency_identity_from_evidence,
    validate_scheme_b_dependency_identity,
)
from paper_search.retrieval.base import SearchProvider
from paper_search.retrieval.snapshot_adapters import LiveCaptureSearchProvider


_INFRASTRUCTURE_ERROR_CODES = frozenset(
    {
        "authentication_error",
        "accounting_failure",
        "budget_exhausted",
        "network_error",
        "provider_error",
        "rate_limited",
        "server_error",
        "snapshot_unavailable",
        "timeout",
    }
)


class BackendSearchResult(DomainModel):
    hits: list[Paper] = Field(default_factory=list)
    usage: UsageActual = Field(default_factory=UsageActual)
    provenance: dict[str, str] = Field(default_factory=dict)
    errors: list[ErrorDetail] = Field(default_factory=list)
    infrastructure_failure: bool = False


class BackendCitationResult(BackendSearchResult):
    direction: CitationDirection


class SearchBackend(Protocol):
    async def search(
        self,
        action_id: str,
        query: str,
        filters: dict[str, object],
        limit: int,
    ) -> BackendSearchResult: ...


class CitationBackend(Protocol):
    async def expand(
        self,
        action_id: str,
        seed: Paper,
        direction: CitationDirection,
        limit: int,
    ) -> BackendCitationResult: ...


def _infrastructure_failure(errors: list[ErrorDetail]) -> bool:
    return any(error.code in _INFRASTRUCTURE_ERROR_CODES for error in errors)


def _backend_error(*, code: str, message: str, provider: str) -> ErrorDetail:
    return ErrorDetail(code=code, message=message, retryable=False, provider=provider)


def _aggregate_usage(usages: list[UsageActual]) -> UsageActual:
    costs = [usage.cost_cny for usage in usages]
    cost = (
        None
        if any(item is None for item in costs)
        else sum((item for item in costs if item is not None), Decimal("0"))
    )
    return UsageActual(
        search_api_calls=sum(usage.search_api_calls for usage in usages),
        llm_calls=sum(usage.llm_calls for usage in usages),
        input_tokens=sum(usage.input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        elapsed_ms=sum(usage.elapsed_ms for usage in usages),
        cost_cny=cost,
    )


def _citation_failure(
    *,
    direction: CitationDirection,
    error: CitationExpansionUnavailableError,
) -> BackendCitationResult:
    diagnostics = list(error.diagnostics)
    if not diagnostics:
        return BackendCitationResult(
            direction=direction,
            errors=[
                _backend_error(
                    code="provider_error",
                    message=str(error),
                    provider="semantic_scholar",
                )
            ],
            infrastructure_failure=True,
        )
    snapshot_refs = [
        ref.model_dump(mode="json")
        for diagnostic in diagnostics
        for ref in diagnostic.snapshot_refs
    ]
    errors = [item for diagnostic in diagnostics for item in diagnostic.errors]
    return BackendCitationResult(
        direction=direction,
        usage=_aggregate_usage([diagnostic.usage for diagnostic in diagnostics]),
        provenance={
            "provider": "semantic_scholar",
            "stage": "provider_citation_expand",
            "snapshot_refs": json.dumps(snapshot_refs, separators=(",", ":")),
        },
        errors=errors,
        infrastructure_failure=_infrastructure_failure(errors),
    )
class _BudgetedProviderCall:
    def __init__(
        self,
        *,
        controller: HardBudgetController,
        call_estimate: UsageEstimate,
    ) -> None:
        if call_estimate.search_api_calls < 1:
            raise ValueError("retrieval backend calls require a search API estimate")
        self._controller = controller
        self._call_estimate = call_estimate
        self._dependency_identity: LiveDependencyIdentity | None = None
        self._live_pricer: ActualCostPricer | None = None

    def _bind_live_provider(
        self,
        provider: object,
        *,
        dependency: Literal["openalex", "semantic_scholar"],
        role: Literal["search", "citation"],
    ) -> None:
        evidence = getattr(provider, "live_identity_evidence", None)
        if evidence is None:
            return
        if not isinstance(provider, LiveCaptureSearchProvider):
            raise ValueError("trusted live search provider required")
        if not isinstance(evidence, LiveDependencyEvidence):
            raise ValueError("live identity evidence is invalid")
        provider_controller = getattr(provider, "live_controller", None)
        provider_pricer = getattr(provider, "live_pricer", None)
        if provider_controller is not self._controller:
            raise ValueError("live provider controller does not match backend controller")
        if self._controller.formal_live is not True:
            raise ValueError("live provider controller must use formal-live enforcement")
        if not isinstance(provider_pricer, ActualCostPricer):
            raise ValueError("live provider pricer is invalid")
        identity = validate_scheme_b_dependency_identity(
            role,
            dependency_identity_from_evidence(evidence),
        )
        if identity.dependency != dependency:
            raise ValueError(f"live {role} dependency identity is invalid")
        self._dependency_identity = identity
        self._live_pricer = provider_pricer

    @property
    def dependency_identity(self) -> LiveDependencyIdentity:
        if self._dependency_identity is None:
            raise ValueError("live identity unavailable")
        return self._dependency_identity

    @property
    def live_pricer(self) -> ActualCostPricer:
        if self._live_pricer is None:
            raise ValueError("live identity unavailable")
        return self._live_pricer

    @property
    def live_controller(self) -> HardBudgetController:
        if self._dependency_identity is None:
            raise ValueError("live identity unavailable")
        return self._controller

    def _reserve(self, action: str) -> BudgetReservation:
        return self._controller.reserve(action, self._call_estimate)

    def _settle_or_verify(
        self,
        reservation: BudgetReservation,
        actual: UsageActual,
    ) -> None:
        terminal = self._controller.terminal_outcome(reservation)
        if terminal is None:
            self._controller.settle(reservation, actual)
            return
        mode, recorded = terminal
        if mode != "settled" or recorded != actual:
            raise ReservationError("retrieval settlement receipt does not match result")

    def _finalize_exception(self, reservation: BudgetReservation) -> None:
        if self._controller.terminal_outcome(reservation) is not None:
            return
        try:
            self._controller.release(reservation)
        except ReservationError:
            self._controller.fail_closed(reservation, UsageActual())

    def _fail_accounting(self, reservation: BudgetReservation, actual: UsageActual) -> None:
        if self._controller.terminal_outcome(reservation) is None:
            self._controller.fail_closed(reservation, actual)


class BudgetedSearchBackend(_BudgetedProviderCall):
    """Use a supplied search provider while owning the logical call receipt."""

    def __init__(
        self,
        *,
        provider: SearchProvider,
        controller: HardBudgetController,
        call_estimate: UsageEstimate,
    ) -> None:
        super().__init__(controller=controller, call_estimate=call_estimate)
        self._provider = provider
        self._bind_live_provider(provider, dependency="openalex", role="search")

    async def search(
        self,
        action_id: str,
        query: str,
        filters: dict[str, object],
        limit: int,
    ) -> BackendSearchResult:
        try:
            reservation = self._reserve(action_id)
        except BudgetExceededError as error:
            return BackendSearchResult(
                errors=[_backend_error(code="budget_exhausted", message=str(error), provider="search")],
                infrastructure_failure=True,
            )
        except ReservationError as error:
            return BackendSearchResult(
                errors=[_backend_error(code="accounting_failure", message=str(error), provider="search")],
                infrastructure_failure=True,
            )
        result: ProviderResult[list[Paper]] | None = None
        try:
            result = await self._provider.search(query, filters, limit, reservation)
            self._settle_or_verify(reservation, result.usage)
        except asyncio.CancelledError:
            self._finalize_exception(reservation)
            raise
        except ReservationError as error:
            if result is None:
                self._finalize_exception(reservation)
            else:
                self._fail_accounting(reservation, result.usage)
            return BackendSearchResult(
                errors=[_backend_error(code="accounting_failure", message=str(error), provider="search")],
                infrastructure_failure=True,
            )
        except Exception as error:
            self._finalize_exception(reservation)
            return BackendSearchResult(
                errors=[_backend_error(code="provider_error", message=str(error), provider="search")],
                infrastructure_failure=True,
            )
        return BackendSearchResult(
            hits=list(result.data),
            usage=result.usage,
            provenance=dict(result.provenance),
            errors=list(result.errors),
            infrastructure_failure=_infrastructure_failure(result.errors),
        )


class BudgetedCitationBackend(_BudgetedProviderCall):
    """Adapt Semantic Scholar citation calls without constructing provider clients."""

    def __init__(
        self,
        *,
        provider: SearchProvider,
        controller: HardBudgetController,
        call_estimate: UsageEstimate,
    ) -> None:
        super().__init__(controller=controller, call_estimate=call_estimate)
        self._provider = provider
        self._bind_live_provider(
            provider,
            dependency="semantic_scholar",
            role="citation",
        )

    @staticmethod
    def _seed_id(seed: Paper) -> ProviderPaperId | None:
        if seed.semantic_scholar_id is None:
            return None
        return ProviderPaperId(provider="semantic_scholar", value=seed.semantic_scholar_id)

    async def _one_direction(
        self,
        *,
        action_id: str,
        seed_id: ProviderPaperId,
        direction: Literal["references", "citations"],
        limit: int,
    ) -> ProviderResult[CitationExpansion] | BackendCitationResult:
        result: ProviderResult[CitationExpansion] | None = None
        try:
            reservation = self._reserve(f"{action_id}.{direction}")
        except BudgetExceededError as error:
            return BackendCitationResult(
                direction=direction,
                errors=[_backend_error(code="budget_exhausted", message=str(error), provider="semantic_scholar")],
                infrastructure_failure=True,
            )
        except ReservationError as error:
            return BackendCitationResult(
                direction=direction,
                errors=[_backend_error(code="accounting_failure", message=str(error), provider="semantic_scholar")],
                infrastructure_failure=True,
            )
        try:
            if direction == "references":
                result = await self._provider.references(seed_id, limit, reservation)
            else:
                result = await self._provider.citations(seed_id, limit, reservation)
            self._settle_or_verify(reservation, result.usage)
            return result
        except asyncio.CancelledError:
            self._finalize_exception(reservation)
            raise
        except ReservationError as error:
            if result is None:
                self._finalize_exception(reservation)
            else:
                self._fail_accounting(reservation, result.usage)
            return BackendCitationResult(
                direction=direction,
                errors=[_backend_error(code="accounting_failure", message=str(error), provider="semantic_scholar")],
                infrastructure_failure=True,
            )
        except Exception as error:
            self._finalize_exception(reservation)
            return BackendCitationResult(
                direction=direction,
                errors=[_backend_error(code="provider_error", message=str(error), provider="semantic_scholar")],
                infrastructure_failure=True,
            )

    async def expand(
        self,
        action_id: str,
        seed: Paper,
        direction: CitationDirection,
        limit: int,
    ) -> BackendCitationResult:
        seed_id = self._seed_id(seed)
        if seed_id is None:
            return BackendCitationResult(
                direction=direction,
                errors=[
                    _backend_error(
                        code="seed_unavailable",
                        message="citation expansion requires a Semantic Scholar seed ID",
                        provider="semantic_scholar",
                    )
                ],
            )
        if direction == "both":
            # The established stage owns its two provider reservations and
            # resolves graph edges.  We only adapt its normalized papers here.
            stage = ProviderCitationExpansionStage(
                provider=self._provider,
                call_estimate=self._call_estimate,
                per_direction_limit=limit,
                max_expanded=limit,
                action_id=action_id,
            )
            try:
                expanded = await stage.expand([seed], controller=self._controller)
            except asyncio.CancelledError:
                raise
            except CitationExpansionUnavailableError as error:
                return _citation_failure(direction=direction, error=error)
            except ReservationError as error:
                return BackendCitationResult(
                    direction=direction,
                    errors=[
                        _backend_error(
                            code="accounting_failure",
                            message=str(error),
                            provider="semantic_scholar",
                        )
                    ],
                    infrastructure_failure=True,
                )
            except Exception as error:
                return BackendCitationResult(
                    direction=direction,
                    errors=[_backend_error(code="provider_error", message=str(error), provider="semantic_scholar")],
                    infrastructure_failure=True,
                )
            diagnostics = list(getattr(expanded, "diagnostics", []))
            errors = [error for diagnostic in diagnostics for error in diagnostic.errors]
            snapshot_refs = [
                ref.model_dump(mode="json")
                for diagnostic in diagnostics
                for ref in diagnostic.snapshot_refs
            ]
            return BackendCitationResult(
                direction=direction,
                hits=[paper for paper in expanded.papers if paper.canonical_id != seed.canonical_id],
                usage=_aggregate_usage([diagnostic.usage for diagnostic in diagnostics]),
                provenance={
                    "provider": "semantic_scholar",
                    "stage": "provider_citation_expand",
                    "snapshot_refs": json.dumps(snapshot_refs, separators=(",", ":")),
                },
                errors=errors,
                infrastructure_failure=_infrastructure_failure(errors),
            )

        result = await self._one_direction(
            action_id=action_id,
            seed_id=seed_id,
            direction=direction,
            limit=limit,
        )
        if isinstance(result, BackendCitationResult):
            return result
        return BackendCitationResult(
            direction=direction,
            hits=list(result.data.papers),
            usage=result.usage,
            provenance=dict(result.provenance),
            errors=list(result.errors),
            infrastructure_failure=_infrastructure_failure(result.errors),
        )


class SearchActionHandler:
    """Execute text or title actions through an injected search backend."""

    def __init__(self, *, backend: SearchBackend) -> None:
        self._backend = backend

    async def execute(
        self,
        action: RecallSearchAction,
        context: RetrievalExecutionContext,
    ) -> RetrievalActionResult:
        if isinstance(action, TextSearchAction):
            query = action.payload.query_text
        elif isinstance(action, TitleSearchAction):
            query = action.payload.title_text
        else:
            raise TypeError("search action handler does not support citation actions")
        result = await self._backend.search(
            action.action_id,
            query,
            dict(context.provider_filters),
            context.max_results_per_action,
        )
        return RetrievalActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            hits=result.hits,
            usage=result.usage,
            provenance=result.provenance,
            errors=result.errors,
            infrastructure_failure=result.infrastructure_failure,
        )


class CitationActionHandler:
    """Execute one citation action against a frozen, normalized seed candidate."""

    def __init__(self, *, backend: CitationBackend) -> None:
        self._backend = backend

    async def execute(
        self,
        action: RecallSearchAction,
        context: RetrievalExecutionContext,
    ) -> RetrievalActionResult:
        if not isinstance(action, CitationExpandAction):
            raise TypeError("citation action handler requires a citation action")
        seed = next(
            (
                candidate.paper
                for candidate in context.seed_candidates
                if candidate.paper.canonical_id == action.payload.seed_canonical_id
            ),
            None,
        )
        if seed is None:
            return RetrievalActionResult(
                action_id=action.action_id,
                action_type=action.action_type,
                errors=[
                    _backend_error(
                        code="seed_unavailable",
                        message="citation action seed is absent from the execution context",
                        provider="semantic_scholar",
                    )
                ],
            )
        result = await self._backend.expand(
            action.action_id,
            seed,
            action.payload.direction,
            action.payload.limit,
        )
        return RetrievalActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            hits=result.hits,
            usage=result.usage,
            provenance=result.provenance,
            errors=result.errors,
            infrastructure_failure=result.infrastructure_failure,
        )


__all__ = [
    "BackendCitationResult",
    "BackendSearchResult",
    "BudgetedCitationBackend",
    "BudgetedSearchBackend",
    "CitationActionHandler",
    "CitationBackend",
    "SearchActionHandler",
    "SearchBackend",
]
