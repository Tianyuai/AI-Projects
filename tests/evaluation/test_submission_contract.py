from __future__ import annotations

import pytest

from paper_search.evaluation.official_adapter import (
    AstaPaperFindingQuery,
    InternalPredictionRecord,
)
from paper_search.evaluation.submission_contract import validate_submission_records


def test_submission_contract_accepts_ordered_query_only_predictions() -> None:
    summary = validate_submission_records(
        [
            AstaPaperFindingQuery(query_id="public-001", query="first query"),
            AstaPaperFindingQuery(query_id="hidden-002", query="second query"),
        ],
        [
            InternalPredictionRecord(
                query_id="public-001",
                selected_paper_ids=["arxiv:2501.10120", "openalex:W1"],
            ),
            InternalPredictionRecord(query_id="hidden-002"),
        ],
    )

    assert summary == {
        "query_count": 2,
        "prediction_count": 2,
        "selected_paper_id_count": 2,
        "query_order_matches": True,
    }


def test_submission_contract_rejects_reordered_predictions() -> None:
    queries = [
        AstaPaperFindingQuery(query_id="q1", query="one"),
        AstaPaperFindingQuery(query_id="q2", query="two"),
    ]
    predictions = [
        InternalPredictionRecord(query_id="q2"),
        InternalPredictionRecord(query_id="q1"),
    ]

    with pytest.raises(ValueError, match="same order"):
        validate_submission_records(queries, predictions)


def test_submission_contract_rejects_duplicate_paper_ids() -> None:
    with pytest.raises(ValueError, match="duplicate selected paper id"):
        validate_submission_records(
            [AstaPaperFindingQuery(query_id="q1", query="one")],
            [
                InternalPredictionRecord(
                    query_id="q1",
                    selected_paper_ids=["arxiv:1", "arxiv:1"],
                )
            ],
        )
