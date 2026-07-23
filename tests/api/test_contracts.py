from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_search.api.contracts import (
    LiveHealthResponse,
    ReadyHealthResponse,
    SearchRequest,
)


def test_search_request_has_prd_defaults_and_strips_identity() -> None:
    request = SearchRequest(query_id=" q1 ", query=" graph retrieval ")

    assert request.model_dump() == {
        "query_id": "q1",
        "query": "graph retrieval",
        "budget_profile": "balanced",
        "include_trace": True,
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
        providers={
            "openalex": "ready",
            "semantic_scholar": "degraded",
        },
    )

    assert response.model_dump() == {
        "status": "degraded",
        "providers": {
            "openalex": "ready",
            "semantic_scholar": "degraded",
        },
    }
    with pytest.raises(ValidationError):
        ReadyHealthResponse.model_validate(
            {
                "status": "ready",
                "providers": {"openalex": "unknown"},
            }
        )
