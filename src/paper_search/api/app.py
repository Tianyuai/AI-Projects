"""FastAPI endpoints over the mode-aware search service router."""

from __future__ import annotations

from typing import Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from paper_search.api.contracts import (
    LiveHealthResponse,
    ReadyHealthResponse,
    SearchRequest,
)
from paper_search.api.routing import (
    HTTP_STATUS_BY_SEARCH_ERROR,
    SearchServiceRouter,
    safe_error_response,
)
from paper_search.application import (
    DependencyStatus,
    SearchErrorCode,
    SearchErrorResponse,
    SearchSuccess,
    StructuredSearchResponse,
)


_DEFAULT_READINESS: Final[ReadyHealthResponse] = ReadyHealthResponse(
    status="degraded",
    execution_mode="replay",
    snapshot_set_id=None,
    dependencies=[
        DependencyStatus(
            dependency=dependency,
            state="failed",
            cache_hit=False,
            error_codes=[],
        )
        for dependency in ("llm", "openalex", "semantic_scholar")
    ],
    last_authorized_probe_at=None,
)


def _error_response(
    code: SearchErrorCode,
    *,
    run_id: str | None = None,
) -> JSONResponse:
    error = safe_error_response(code, run_id=run_id)
    return JSONResponse(
        status_code=HTTP_STATUS_BY_SEARCH_ERROR[error.code],
        content=error.model_dump(mode="json"),
    )


def create_app(
    service_router: SearchServiceRouter | None = None,
) -> FastAPI:
    """Create an API that exposes only typed, safe application outcomes."""
    application = FastAPI()

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
        return _error_response("invalid_request")

    @application.get("/health/live", response_model=LiveHealthResponse)
    async def live() -> LiveHealthResponse:
        return LiveHealthResponse()

    @application.get(
        "/health/ready",
        response_model=ReadyHealthResponse,
        responses={503: {"model": ReadyHealthResponse}},
    )
    async def ready() -> ReadyHealthResponse | JSONResponse:
        response = (
            _DEFAULT_READINESS
            if service_router is None
            else service_router.readiness()
        )
        if response.status == "ready":
            return response
        return JSONResponse(status_code=503, content=response.model_dump(mode="json"))

    @application.post(
        "/v1/search",
        response_model=StructuredSearchResponse,
        responses={
            status: {"model": SearchErrorResponse}
            for status in set(HTTP_STATUS_BY_SEARCH_ERROR.values())
        },
    )
    async def search(request: SearchRequest) -> StructuredSearchResponse | JSONResponse:
        if service_router is None:
            return _error_response("internal_error")
        try:
            execution = await service_router.execute(request)
        except Exception:  # noqa: BLE001
            return _error_response("internal_error")
        if isinstance(execution.outcome, SearchSuccess):
            return execution.outcome.response
        return _error_response(
            execution.outcome.error.code,
            run_id=execution.outcome.error.run_id,
        )

    return application


app = create_app()
