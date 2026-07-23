"""Dependency-injected FastAPI routes for offline mock search."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from paper_search.api.contracts import (
    LiveHealthResponse,
    ReadyHealthResponse,
    SearchRequest,
)
from paper_search.domain.models import StructuredSearchResponse


_UNAVAILABLE_DETAIL = "search temporarily unavailable"


class SearchService(Protocol):
    async def __call__(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResponse: ...


ReadinessProbe = Callable[[], Mapping[str, bool]]


def create_app(
    search_service: SearchService | None = None,
    *,
    readiness_probe: ReadinessProbe | None = None,
) -> FastAPI:
    """Create an ASGI app whose external boundaries are explicitly injected."""
    application = FastAPI()

    @application.get(
        "/health/live",
        response_model=LiveHealthResponse,
    )
    async def live() -> LiveHealthResponse:
        return LiveHealthResponse()

    @application.get(
        "/health/ready",
        response_model=ReadyHealthResponse,
        responses={503: {"model": ReadyHealthResponse}},
    )
    async def ready() -> ReadyHealthResponse | JSONResponse:
        try:
            raw_providers = (
                dict(readiness_probe())
                if readiness_probe is not None
                else {}
            )
        except Exception:
            raw_providers = {}
        providers = {
            name: "ready" if available else "degraded"
            for name, available in sorted(raw_providers.items())
        }
        is_ready = (
            search_service is not None
            and bool(providers)
            and all(status == "ready" for status in providers.values())
        )
        response = ReadyHealthResponse(
            status="ready" if is_ready else "degraded",
            providers=providers,
        )
        if is_ready:
            return response
        return JSONResponse(
            status_code=503,
            content=response.model_dump(mode="json"),
        )

    @application.post(
        "/v1/search",
        response_model=StructuredSearchResponse,
        responses={503: {"description": _UNAVAILABLE_DETAIL}},
    )
    async def search(
        request: SearchRequest,
    ) -> StructuredSearchResponse | JSONResponse:
        if search_service is None:
            return JSONResponse(
                status_code=503,
                content={"detail": _UNAVAILABLE_DETAIL},
            )
        try:
            return await search_service(request)
        except Exception:
            return JSONResponse(
                status_code=503,
                content={"detail": _UNAVAILABLE_DETAIL},
            )

    return application


app = create_app()
