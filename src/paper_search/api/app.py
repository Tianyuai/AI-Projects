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
    UnavailableResponse,
)
from paper_search.application import DependencyStatus, StructuredSearchResponse


_UNAVAILABLE_RESPONSE = UnavailableResponse()


class SearchService(Protocol):
    async def __call__(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResponse: ...


ReadinessProbe = Callable[[], Mapping[str, bool]]

_DEPENDENCIES = ("llm", "openalex", "semantic_scholar")
_MOCK_SNAPSHOT_SET_ID = "mock-snapshot-v1"


def _provider_statuses(
    readiness_probe: ReadinessProbe | None,
) -> dict[str, bool] | None:
    if readiness_probe is None:
        return None
    try:
        raw_providers = dict(readiness_probe())
        normalized: list[tuple[str, bool]] = []
        for name, available in raw_providers.items():
            if (
                type(name) is not str
                or not name.strip()
                or type(available) is not bool
                or name.strip() not in _DEPENDENCIES
            ):
                raise ValueError("invalid readiness probe mapping")
            normalized.append((name.strip(), available))
        if len({name for name, _ in normalized}) != len(normalized):
            raise ValueError("duplicate normalized provider name")
        return dict(normalized)
    except Exception:
        return None


def _dependency_statuses(
    search_service: SearchService | None,
    readiness_probe: ReadinessProbe | None,
) -> list[DependencyStatus]:
    probe = _provider_statuses(readiness_probe)
    def state_for(name: str) -> str:
        if search_service is None:
            return "failed"
        if name == "llm":
            return "ready"
        if probe is None:
            return "failed"
        return "ready" if probe.get(name, False) else "degraded"

    return [
        DependencyStatus(
            dependency=name,
            state=state_for(name),
            cache_hit=False,
            error_codes=[],
        )
        for name in _DEPENDENCIES
    ]


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
        dependencies = _dependency_statuses(search_service, readiness_probe)
        is_ready = (
            search_service is not None
            and all(item.state == "ready" for item in dependencies)
        )
        response = ReadyHealthResponse(
            status="ready" if is_ready else "degraded",
            execution_mode="replay",
            snapshot_set_id=_MOCK_SNAPSHOT_SET_ID,
            dependencies=dependencies,
            last_authorized_probe_at=None,
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
        responses={503: {"model": UnavailableResponse}},
    )
    async def search(
        request: SearchRequest,
    ) -> StructuredSearchResponse | JSONResponse:
        if search_service is None:
            return JSONResponse(
                status_code=503,
                content=_UNAVAILABLE_RESPONSE.model_dump(mode="json"),
            )
        try:
            return await search_service(request)
        except Exception:
            return JSONResponse(
                status_code=503,
                content=_UNAVAILABLE_RESPONSE.model_dump(mode="json"),
            )

    return application


app = create_app()
