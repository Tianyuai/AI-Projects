from __future__ import annotations

from pathlib import Path

import pytest

from paper_search.domain.models import (
    DependencyStatus,
    FusedPaper,
    Paper,
    QueryAnalysisResult,
    QuerySpec,
    SearchPlan,
    StructuredSearchResponse,
    SubQuery,
    UsageActual,
)
from paper_search.evaluation.dataset import PredictionRecord, read_jsonl
from paper_search.evaluation.official_adapter import (
    InternalPredictionRecord,
    adapt_prediction_record,
)
from paper_search.evaluation.predictions import (
    prediction_from_response,
    write_prediction_records,
    write_response_predictions,
)


def _response(
    query_id: str,
    selected_paper_ids: list[str],
) -> StructuredSearchResponse:
    query_spec = QuerySpec(
        original_query=f"synthetic query {query_id}",
        research_goal="exercise prediction serialization",
    )
    return StructuredSearchResponse(
        run_id=f"prediction-run-{query_id}",
        query_id=query_id,
        execution_mode="replay",
        snapshot_set_id="prediction-snapshot-v1",
        snapshot_captured_at=None,
        query_analysis=QueryAnalysisResult(
            query_spec=query_spec,
            search_plan=SearchPlan(
                subqueries=[
                    SubQuery(
                        query_id=f"{query_id}-sq-1",
                        text=query_spec.original_query,
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
        selected_paper_ids=selected_paper_ids,
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
        prompt_version="prediction-v1",
        config_hash="sha256:" + "a" * 64,
        git_sha="abc1234",
    )


def test_write_prediction_records_preserves_order_and_bytes(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    records = [
        InternalPredictionRecord(
            query_id="synthetic-q2",
            selected_paper_ids=["s2:S2"],
        ),
        InternalPredictionRecord(
            query_id="synthetic-q1",
            selected_paper_ids=[],
        ),
    ]

    written = write_prediction_records(output, records)

    assert written == records
    assert written is not records
    assert output.read_bytes() == (
        b'{"query_id":"synthetic-q2","selected_paper_ids":["s2:S2"]}\n'
        b'{"query_id":"synthetic-q1","selected_paper_ids":[]}\n'
    )


def test_write_prediction_records_rejects_duplicate_before_write(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_bytes(b"preserve-me\n")

    with pytest.raises(
        ValueError,
        match=r"^duplicate query_id: synthetic-q1$",
    ):
        write_prediction_records(
            output,
            [
                InternalPredictionRecord(
                    query_id="synthetic-q1",
                    selected_paper_ids=[],
                ),
                InternalPredictionRecord(
                    query_id="synthetic-q1",
                    selected_paper_ids=["openalex:W1"],
                ),
            ],
        )

    assert output.read_bytes() == b"preserve-me\n"


def test_write_response_predictions_is_deterministic_and_adapter_compatible(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictions.jsonl"

    records = write_response_predictions(
        output,
        [
            _response("q1", ["openalex:W1", "s2:S1"]),
            _response("q2", []),
        ],
    )

    assert records == [
        InternalPredictionRecord(
            query_id="q1",
            selected_paper_ids=["openalex:W1", "s2:S1"],
        ),
        InternalPredictionRecord(query_id="q2", selected_paper_ids=[]),
    ]
    assert output.read_bytes() == (
        b'{"query_id":"q1","selected_paper_ids":["openalex:W1","s2:S1"]}\n'
        b'{"query_id":"q2","selected_paper_ids":[]}\n'
    )
    assert read_jsonl(output, InternalPredictionRecord) == records
    assert [adapt_prediction_record(record) for record in records] == [
        PredictionRecord(
            query_id="q1",
            predicted_paper_ids=["openalex:W1", "s2:S1"],
        ),
        PredictionRecord(query_id="q2", predicted_paper_ids=[]),
    ]


def test_write_response_predictions_rejects_duplicate_query_before_write(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_bytes(b"sentinel\n")

    with pytest.raises(ValueError, match=r"^duplicate query_id: q1$"):
        write_response_predictions(
            output,
            [_response("q1", ["openalex:W1"]), _response("q1", [])],
        )

    assert output.read_bytes() == b"sentinel\n"


def test_official_prediction_prefers_arxiv_identity_for_selected_openalex_paper() -> None:
    response = _response("q1", ["openalex:W1", "openalex:W2"]).model_copy(
        update={
            "fused_papers": [
                FusedPaper(
                    paper=Paper(
                        canonical_id="openalex:W1",
                        title="ArXiv-backed paper",
                        arxiv_id="2210.05663",
                        openalex_id="W1",
                    ),
                    score=1.0,
                    source_ranks={"openalex": 1},
                ),
                FusedPaper(
                    paper=Paper(
                        canonical_id="openalex:W2",
                        title="OpenAlex-only paper",
                        openalex_id="W2",
                    ),
                    score=0.5,
                    source_ranks={"openalex": 2},
                ),
            ]
        }
    )

    assert prediction_from_response(response) == InternalPredictionRecord(
        query_id="q1",
        selected_paper_ids=["arxiv:2210.05663", "openalex:W2"],
    )
