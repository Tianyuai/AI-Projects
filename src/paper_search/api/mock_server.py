"""Loopback-only offline server for the fixed Week 2 synthetic mock stack."""

from __future__ import annotations

from fastapi import FastAPI

from paper_search.api.app import create_app
from paper_search.evaluation.synthetic_mocks import build_synthetic_search_service


def mock_readiness() -> dict[str, bool]:
    """Return the fixed readiness map for the offline mock composition."""
    return {
        "openalex": True,
        "semantic_scholar": True,
    }


def create_mock_app() -> FastAPI:
    """Build the only app composition exposed by the mock-server entry point."""
    return create_app(
        build_synthetic_search_service(),
        readiness_probe=mock_readiness,
    )
