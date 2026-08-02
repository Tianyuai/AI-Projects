"""Mode-aware routing over the canonical application service boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final
from uuid import uuid4

from paper_search.api.service import LiveSearchService, SearchExecutionService
from paper_search.application.contracts import (
    ReadyHealthResponse,
    SearchErrorCode,
    SearchErrorResponse,
    SearchExecutionResult,
    SearchFailure,
    SearchRequest,
    SearchSuccess,
)
from paper_search.domain.models import UsageActual


HTTP_STATUS_BY_SEARCH_ERROR: Final[dict[SearchErrorCode, int]] = {
    "invalid_request": 400,
    "live_not_authorized": 403,
    "config_mismatch": 409,
    "validation_attempt_conflict": 409,
    "budget_exhausted": 429,
    "snapshot_unavailable": 503,
    "dependency_failure": 503,
    "integrity_failure": 500,
    "internal_error": 500,
}

SAFE_SEARCH_ERROR_DETAILS: Final[dict[SearchErrorCode, str]] = {
    "invalid_request": "The search request is invalid",
    "live_not_authorized": "Live search is not authorized",
    "config_mismatch": "The requested mode does not match the application binding",
    "validation_attempt_conflict": "The validation attempt conflicts with prior state",
    "budget_exhausted": "The search budget was exhausted",
    "snapshot_unavailable": "Required replay data is unavailable",
    "dependency_failure": "A required search dependency failed",
    "integrity_failure": "Search integrity validation failed",
    "internal_error": "The search could not be completed",
}

_RETRYABLE_CODES: Final[frozenset[SearchErrorCode]] = frozenset(
    {"snapshot_unavailable", "dependency_failure"}
)


def safe_error_response(
    code: SearchErrorCode,
    *,
    run_id: str | None,
) -> SearchErrorResponse:
    """Build the only public error representation accepted by the API."""
    return SearchErrorResponse(
        code=code,
        detail=SAFE_SEARCH_ERROR_DETAILS[code],
        retryable=code in _RETRYABLE_CODES,
        run_id=run_id,
    )


class SearchServiceRouter:
    """Keep replay process-bound while creating live services per request."""

    def __init__(
        self,
        *,
        replay_service: SearchExecutionService,
        readiness: ReadyHealthResponse,
        live_service_factory: Callable[[], LiveSearchService] | None = None,
        server_live_authorized: bool = False,
        run_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._replay_service = replay_service
        self._readiness = readiness
        self._live_service_factory = live_service_factory
        self._server_live_authorized = server_live_authorized
        self._run_id_factory = run_id_factory or (lambda: str(uuid4()))

    def readiness(self) -> ReadyHealthResponse:
        """Return cached safe state without making a dependency probe."""
        return self._readiness

    async def execute(self, request: SearchRequest) -> SearchExecutionResult:
        if request.mode == "replay":
            return await self._replay_service.execute(request)
        if not self._server_live_authorized or self._live_service_factory is None:
            return self._live_not_authorized(request)

        live_service = self._live_service_factory()
        execution = await live_service.execute(request)
        if isinstance(execution.outcome, SearchSuccess):
            await live_service.publish(execution)
        return execution

    def _live_not_authorized(self, request: SearchRequest) -> SearchExecutionResult:
        run_id = self._run_id_factory()
        return SearchExecutionResult(
            outcome=SearchFailure(
                query_id=request.query_id,
                run_id=run_id,
                error=safe_error_response("live_not_authorized", run_id=run_id),
                usage=UsageActual(),
                stop_reason="live_not_authorized",
            ),
            diagnostics=[],
            business_result_sha256=None,
        )
