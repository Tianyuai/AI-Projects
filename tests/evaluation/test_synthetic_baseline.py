from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import cast

import pytest

from paper_search.api.contracts import SearchRequest
from paper_search.domain.models import (
    DependencyStatus,
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
    _run_synthetic_batch,
    _validate_synthetic_requests,
    run_synthetic_baseline,
)


def _response(request: SearchRequest, paper_ids: list[str]) -> StructuredSearchResponse:
    spec = QuerySpec(
        original_query=request.query,
        research_goal="synthetic baseline",
    )
    return StructuredSearchResponse(
        run_id=f"synthetic-run-{request.query_id}",
        query_id=request.query_id,
        execution_mode="replay",
        snapshot_set_id="synthetic-snapshot-v1",
        snapshot_captured_at=None,
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
        planner_fallback=False,
        planner_status="primary",
        dependency_status=[
            DependencyStatus(dependency="llm", state="replayed", cache_hit=True, error_codes=[]),
            DependencyStatus(
                dependency="openalex", state="replayed", cache_hit=True, error_codes=[]
            ),
            DependencyStatus(
                dependency="semantic_scholar",
                state="replayed",
                cache_hit=True,
                error_codes=[],
            ),
        ],
        warnings=[],
        prompt_version="synthetic-baseline-v1",
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


class MismatchedResponseService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResponse:
        self.calls.append(request.query_id)
        response = _response(request, [f"openalex:W{len(self.calls)}"])
        if request.query_id == "synthetic-q2":
            return response.model_copy(update={"query_id": "synthetic-unexpected"})
        return response


def _requests() -> tuple[SearchRequest, ...]:
    return (
        SearchRequest(query_id="synthetic-q1", query="synthetic one"),
        SearchRequest(query_id="synthetic-q2", query="synthetic two"),
        SearchRequest(query_id="synthetic-q3", query="synthetic three"),
    )


def test_catalog_is_fixed_strict_and_unique() -> None:
    assert SYNTHETIC_QUERIES == (
        SearchRequest(
            query_id="synthetic-graph-retrieval",
            query="Synthetic graph retrieval research",
            budget_profile="low",
            include_trace=False,
        ),
        SearchRequest(
            query_id="synthetic-empty-result",
            query="Synthetic zero-result literature search",
            budget_profile="balanced",
            include_trace=False,
        ),
        SearchRequest(
            query_id="synthetic-budget-path",
            query="Synthetic budget-aware scholarly search",
            budget_profile="low",
            include_trace=False,
        ),
    )
    assert _validate_synthetic_requests(SYNTHETIC_QUERIES) == SYNTHETIC_QUERIES


def test_public_runner_only_accepts_an_output_path() -> None:
    signature = inspect.signature(run_synthetic_baseline)

    assert tuple(signature.parameters) == ("output",)
    assert signature.parameters["output"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        asyncio.run(run_synthetic_baseline(Path("predictions.jsonl")))  # type: ignore[call-arg]


def test_validate_synthetic_requests_rejects_empty_duplicate_and_invalid() -> None:
    with pytest.raises(ValueError, match=r"^synthetic query catalog must not be empty$"):
        _validate_synthetic_requests(())

    duplicate = SearchRequest(query_id="synthetic-q1", query="synthetic duplicate")
    with pytest.raises(ValueError, match=r"^duplicate query_id: synthetic-q1$"):
        _validate_synthetic_requests((_requests()[0], duplicate))

    invalid = SearchRequest.model_construct(
        query_id="synthetic-q2",
        query="synthetic invalid",
        budget_profile="not-a-profile",
        include_trace=False,
    )
    with pytest.raises(
        ValueError,
        match=r"^synthetic query catalog contains invalid request$",
    ):
        _validate_synthetic_requests((_requests()[0], invalid))

    with pytest.raises(
        ValueError,
        match=r"^synthetic query catalog contains invalid request$",
    ):
        _validate_synthetic_requests(
            cast(tuple[SearchRequest, ...], (_requests()[0], object()))
        )

    coercible = SearchRequest.model_construct(
        query_id="synthetic-q2",
        query="synthetic coercible",
        budget_profile="low",
        include_trace="false",
    )
    with pytest.raises(
        ValueError,
        match=r"^synthetic query catalog contains invalid request$",
    ):
        _validate_synthetic_requests((_requests()[0], coercible))


def test_batch_keeps_order_and_continues_after_query_exception(
    tmp_path: Path,
) -> None:
    service = RecordingService(failing_query_id="synthetic-q2")
    output = tmp_path / "predictions.jsonl"

    records = asyncio.run(
        _run_synthetic_batch(
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


def test_batch_isolates_mismatched_response_query_id_and_keeps_request_order(
    tmp_path: Path,
) -> None:
    service = MismatchedResponseService()
    output = tmp_path / "predictions.jsonl"

    records = asyncio.run(
        _run_synthetic_batch(
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
            _run_synthetic_batch(
                (_requests()[0], duplicate),
                search_service=service,
                output=output,
            )
        )

    assert service.calls == []
    assert output.read_bytes() == b"preserve-me\n"


@pytest.mark.parametrize(
    "invalid_response",
    [
        {
            "query_id": "synthetic-q2",
            "selected_paper_ids": ["openalex:FORGED"],
        },
        StructuredSearchResponse.model_construct(
            query_id="synthetic-q2",
            selected_paper_ids=["openalex:FORGED"],
        ),
    ],
)
def test_batch_isolates_invalid_response_and_continues(
    tmp_path: Path,
    invalid_response: object,
) -> None:
    class InvalidResponseService:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def __call__(self, request: SearchRequest) -> StructuredSearchResponse:
            self.calls.append(request.query_id)
            if request.query_id == "synthetic-q2":
                return cast(StructuredSearchResponse, invalid_response)
            return _response(request, [f"openalex:W{len(self.calls)}"])

    service = InvalidResponseService()
    records = asyncio.run(
        _run_synthetic_batch(
            _requests(),
            search_service=service,
            output=tmp_path / "predictions.jsonl",
        )
    )

    assert service.calls == ["synthetic-q1", "synthetic-q2", "synthetic-q3"]
    assert [record.selected_paper_ids for record in records] == [
        ["openalex:W1"],
        [],
        ["openalex:W3"],
    ]


def test_batch_isolates_coercible_constructed_response(
    tmp_path: Path,
) -> None:
    class CoercibleResponseService:
        async def __call__(
            self,
            request: SearchRequest,
        ) -> StructuredSearchResponse:
            valid = _response(request, ["openalex:W1"])
            return StructuredSearchResponse.model_construct(
                **{
                    **valid.model_dump(mode="python"),
                    "selected_paper_ids": ("openalex:FORGED",),
                    "is_partial": "false",
                }
            )

    records = asyncio.run(
        _run_synthetic_batch(
            (_requests()[0],),
            search_service=CoercibleResponseService(),
            output=tmp_path / "predictions.jsonl",
        )
    )

    assert records == [
        InternalPredictionRecord(
            query_id="synthetic-q1",
            selected_paper_ids=[],
        )
    ]


@pytest.mark.parametrize(
    ("is_partial", "stop_reason", "warnings"),
    [
        (False, "completed", []),
        (True, "budget_exhausted", ["synthetic soft limit"]),
        (True, "provider_failure", ["synthetic hard failure"]),
    ],
)
def test_batch_preserves_selected_ids_from_valid_response_states(
    tmp_path: Path,
    is_partial: bool,
    stop_reason: str,
    warnings: list[str],
) -> None:
    class ResponseStateService:
        async def __call__(
            self,
            request: SearchRequest,
        ) -> StructuredSearchResponse:
            return _response(request, ["openalex:W1"]).model_copy(
                update={
                    "is_partial": is_partial,
                    "stop_reason": stop_reason,
                    "warnings": warnings,
                }
            )

    records = asyncio.run(
        _run_synthetic_batch(
            (_requests()[0],),
            search_service=ResponseStateService(),
            output=tmp_path / "predictions.jsonl",
        )
    )

    assert records[0].selected_paper_ids == ["openalex:W1"]
