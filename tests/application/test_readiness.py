"""Tests for the bounded authorized live readiness probe and mapping."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from paper_search.application.readiness import (
    AuthorizedCapability,
    AuthorizedReadinessEvidence,
    build_live_readiness,
    load_live_readiness,
    probe_live_readiness,
    write_authorized_readiness,
)


_NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC)


def _evidence(*states: str) -> AuthorizedReadinessEvidence:
    return AuthorizedReadinessEvidence(
        schema_version="gate0-readiness-v1",
        generated_at=_NOW,
        capabilities=[
            AuthorizedCapability(name=name, state=state, observed_at=_NOW)
            for name, state in zip(
                ("llm", "openalex", "semantic_scholar"),
                states,
                strict=True,
            )
        ],
    )


def test_build_fresh_all_ready_is_ready() -> None:
    evidence = _evidence("ready", "ready", "ready")
    response = build_live_readiness(evidence, _NOW + timedelta(minutes=1))

    assert response.status == "ready"
    assert response.last_authorized_probe_at == _NOW
    assert [status.state for status in response.dependencies] == [
        "ready",
        "ready",
        "ready",
    ]


def test_build_stale_evidence_is_degraded() -> None:
    evidence = _evidence("ready", "ready", "ready")
    response = build_live_readiness(
        evidence,
        _NOW + timedelta(minutes=16),
    )

    assert response.status == "degraded"
    assert response.last_authorized_probe_at == _NOW


def test_build_degraded_capability_is_degraded() -> None:
    evidence = _evidence("ready", "degraded", "ready")
    response = build_live_readiness(evidence, _NOW + timedelta(minutes=1))

    assert response.status == "degraded"


def test_build_ignores_unused_degraded_capability() -> None:
    evidence = _evidence("ready", "ready", "degraded")
    response = build_live_readiness(
        evidence,
        _NOW + timedelta(minutes=1),
        required_dependencies=("llm", "openalex"),
    )

    assert response.status == "ready"
    assert [status.state for status in response.dependencies] == [
        "ready",
        "ready",
        "degraded",
    ]


def test_build_requires_each_used_capability() -> None:
    evidence = _evidence("ready", "ready", "degraded")
    response = build_live_readiness(
        evidence,
        _NOW + timedelta(minutes=1),
        required_dependencies=("llm", "openalex", "semantic_scholar"),
    )

    assert response.status == "degraded"


def test_probe_maps_mock_provider_responses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.deepseek.com":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "{}"}}]},
            )
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json={"results": [{"id": "W1"}]})
        if request.url.host == "api.semanticscholar.org":
            return httpx.Response(200, json={"data": [{"paperId": "S1"}]})
        return httpx.Response(404)

    async def run() -> AuthorizedReadinessEvidence:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await probe_live_readiness(
                llm_api_key="secret",
                openalex_api_key="secret",
                semantic_scholar_api_key="secret",
                client=client,
                clock=lambda: _NOW,
            )

    evidence = asyncio.run(run())

    assert {
        capability.name: capability.state for capability in evidence.capabilities
    } == {
        "llm": "ready",
        "openalex": "ready",
        "semantic_scholar": "ready",
    }


def test_probe_maps_rate_limit_to_degraded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.deepseek.com":
            return httpx.Response(429, text="rate limited")
        if request.url.host == "api.openalex.org":
            return httpx.Response(500, text="server error")
        if request.url.host == "api.semanticscholar.org":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    async def run() -> AuthorizedReadinessEvidence:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await probe_live_readiness(
                llm_api_key="secret",
                openalex_api_key="secret",
                semantic_scholar_api_key="secret",
                client=client,
                clock=lambda: _NOW,
            )

    evidence = asyncio.run(run())

    assert {
        capability.name: capability.state for capability in evidence.capabilities
    } == {
        "llm": "degraded",
        "openalex": "degraded",
        "semantic_scholar": "ready",
    }


def test_load_missing_readiness_returns_none(tmp_path: Path) -> None:
    assert load_live_readiness(tmp_path) is None


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    evidence = _evidence("ready", "ready", "ready")
    path = (
        tmp_path
        / "data"
        / "annotation_work"
        / "provider_readiness.live.json"
    )
    write_authorized_readiness(path, evidence)

    loaded = load_live_readiness(tmp_path)
    assert loaded == evidence
