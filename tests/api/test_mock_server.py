from __future__ import annotations

import asyncio

import httpx

from paper_search.api.mock_server import create_mock_app, mock_readiness


async def _request(application: object, method: str, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path)


def test_mock_readiness_is_fixed_and_complete() -> None:
    assert mock_readiness() == {
        "openalex": True,
        "semantic_scholar": True,
    }


def test_create_mock_app_reports_both_fixed_providers_ready() -> None:
    response = asyncio.run(_request(create_mock_app(), "GET", "/health/ready"))

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "providers": {
            "openalex": "ready",
            "semantic_scholar": "ready",
        },
    }
