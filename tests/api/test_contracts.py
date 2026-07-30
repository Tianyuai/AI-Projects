from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_search.api.contracts import (
    LiveHealthResponse,
    ReadyHealthResponse,
    SearchRequest,
)
from paper_search.domain.models import DependencyStatus


def test_search_request_has_prd_defaults_and_strips_identity() -> None:
    request = SearchRequest(query_id=" q1 ", query=" graph retrieval ")

    assert request.model_dump() == {
        "query_id": "q1",
        "query": "graph retrieval",
        "budget_profile": "balanced",
        "include_trace": True,
        "mode": "replay",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"query_id": "", "query": "valid"},
        {"query_id": "q1", "query": ""},
        {"query_id": "q1", "query": "valid", "budget_profile": "large"},
        {"query_id": "q1", "query": "valid", "extra": True},
    ],
)
def test_search_request_rejects_invalid_payload(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(payload)


def test_health_contracts_are_strict() -> None:
    assert LiveHealthResponse().model_dump() == {"status": "ok"}
    response = ReadyHealthResponse(
        status="degraded",
        execution_mode="replay",
        snapshot_set_id="mock-snapshot-v1",
        dependencies=[
            DependencyStatus(dependency="llm", state="degraded", cache_hit=False, error_codes=[]),
            DependencyStatus(
                dependency="openalex", state="degraded", cache_hit=False, error_codes=[]
            ),
            DependencyStatus(
                dependency="semantic_scholar",
                state="degraded",
                cache_hit=False,
                error_codes=[],
            ),
        ],
        last_authorized_probe_at=None,
    )

    assert response.model_dump() == {
        "status": "degraded",
        "execution_mode": "replay",
        "snapshot_set_id": "mock-snapshot-v1",
        "dependencies": [
            {"dependency": "llm", "state": "degraded", "cache_hit": False, "error_codes": []},
            {
                "dependency": "openalex",
                "state": "degraded",
                "cache_hit": False,
                "error_codes": [],
            },
            {
                "dependency": "semantic_scholar",
                "state": "degraded",
                "cache_hit": False,
                "error_codes": [],
            },
        ],
        "last_authorized_probe_at": None,
    }
    with pytest.raises(ValidationError):
        ReadyHealthResponse.model_validate(
            {
                "status": "ready",
                "execution_mode": "invalid",
                "snapshot_set_id": None,
                "dependencies": [],
                "last_authorized_probe_at": None,
            }
        )
