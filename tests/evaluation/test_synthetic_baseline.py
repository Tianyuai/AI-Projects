from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from paper_search.api.contracts import SearchRequest
from paper_search.domain.models import (
    QueryAnalysisResult,
    QuerySpec,
    SearchPlan,
    StructuredSearchResponse,
    SubQuery,
    UsageActual,
)
from paper_search.evaluation.official_adapter import InternalPredictionRecord
from paper_search.evaluation.synthetic_baseline import (
    SYNTHETIC_QUERIES,
    run_synthetic_baseline,
    validate_synthetic_requests,
)


def _response(request: SearchRequest, paper_ids: list[str]) -> StructuredSearchResponse:
    spec = QuerySpec(
        original_query=request.query,
        research_goal="synthetic baseline",
    )
    return StructuredSearchResponse(
        query_id=request.query_id,
        query_analysis=QueryAnalysisResult(
            query_spec=spec,
            search_plan=SearchPlan(
                subqueries=[
                    SubQuery(
                        query_id=f"{request.query_id}-sq-1",
                        text=request.query,
                        query_type="exact",
                        target_constraints=[],
                        priority=1,
                        provider_hint="either",
                    )
                ],
                inherited_hard_filters={},
                rationale="synthetic",
            ),
        ),
        selected_paper_ids=paper_ids,
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        search_trace=[],
        usage=UsageActual(),
        stop_reason="completed",
        is_partial=False,
        warnings=[],
        config_hash="sha256:" + "a" * 64,
        git_sha="synthetic-task8c",
    )


class RecordingService:
    def __init__(self, failing_query_id: str | None = None) -> None:
        self.failing_query_id = failing_query_id
        self.calls: list[str] = []

    async def __call__(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResponse:
        self.calls.append(request.query_id)
        if request.query_id == self.failing_query_id:
            raise TimeoutError("synthetic failure must not be persisted")
        return _response(request, [f"openalex:W{len(self.calls)}"])


def _requests() -> tuple[SearchRequest, ...]:
    return (
        SearchRequest(query_id="synthetic-q1", query="synthetic one"),
        SearchRequest(query_id="synthetic-q2", query="synthetic two"),
        SearchRequest(query_id="synthetic-q3", query="synthetic three"),
    )


def test_catalog_is_fixed_strict_and_unique() -> None:
    assert SYNTHETIC_QUERIES
    assert all(request.include_trace is False for request in SYNTHETIC_QUERIES)
    assert all("synthetic" in request.query.casefold() for request in SYNTHETIC_QUERIES)
    assert len({request.query_id for request in SYNTHETIC_QUERIES}) == len(
        SYNTHETIC_QUERIES
    )
    assert validate_synthetic_requests(SYNTHETIC_QUERIES) is SYNTHETIC_QUERIES


def test_validate_synthetic_requests_rejects_empty_and_duplicate() -> None:
    with pytest.raises(ValueError, match=r"^synthetic query catalog must not be empty$"):
        validate_synthetic_requests(())

    duplicate = SearchRequest(query_id="synthetic-q1", query="synthetic duplicate")
    with pytest.raises(ValueError, match=r"^duplicate query_id: synthetic-q1$"):
        validate_synthetic_requests((_requests()[0], duplicate))


def test_batch_keeps_order_and_continues_after_query_exception(
    tmp_path: Path,
) -> None:
    service = RecordingService(failing_query_id="synthetic-q2")
    output = tmp_path / "predictions.jsonl"

    records = asyncio.run(
        run_synthetic_baseline(
            _requests(),
            search_service=service,
            output=output,
        )
    )

    assert service.calls == ["synthetic-q1", "synthetic-q2", "synthetic-q3"]
    assert records == [
        InternalPredictionRecord(
            query_id="synthetic-q1",
            selected_paper_ids=["openalex:W1"],
        ),
        InternalPredictionRecord(
            query_id="synthetic-q2",
            selected_paper_ids=[],
        ),
        InternalPredictionRecord(
            query_id="synthetic-q3",
            selected_paper_ids=["openalex:W3"],
        ),
    ]
    assert output.read_bytes() == (
        b'{"query_id":"synthetic-q1","selected_paper_ids":["openalex:W1"]}\n'
        b'{"query_id":"synthetic-q2","selected_paper_ids":[]}\n'
        b'{"query_id":"synthetic-q3","selected_paper_ids":["openalex:W3"]}\n'
    )


def test_preflight_failure_does_not_call_service_or_replace_output(
    tmp_path: Path,
) -> None:
    service = RecordingService()
    output = tmp_path / "predictions.jsonl"
    output.write_bytes(b"preserve-me\n")
    duplicate = SearchRequest(query_id="synthetic-q1", query="synthetic duplicate")

    with pytest.raises(ValueError, match=r"^duplicate query_id: synthetic-q1$"):
        asyncio.run(
            run_synthetic_baseline(
                (_requests()[0], duplicate),
                search_service=service,
                output=output,
            )
        )

    assert service.calls == []
    assert output.read_bytes() == b"preserve-me\n"
