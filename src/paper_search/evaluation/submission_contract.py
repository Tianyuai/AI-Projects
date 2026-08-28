"""Minimal query-only JSONL exchange contract for public and hidden evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from paper_search.evaluation.official_adapter import (
    AstaPaperFindingQuery,
    InternalPredictionRecord,
)


def validate_submission_records(
    queries: Sequence[AstaPaperFindingQuery],
    predictions: Sequence[InternalPredictionRecord],
) -> dict[str, object]:
    """Require exactly one ordered prediction record for every evaluator query."""

    normalized_queries = [AstaPaperFindingQuery.model_validate(row) for row in queries]
    normalized_predictions = [
        InternalPredictionRecord.model_validate(row) for row in predictions
    ]
    query_ids = [row.query_id for row in normalized_queries]
    prediction_ids = [row.query_id for row in normalized_predictions]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("evaluator query ids must be unique")
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("submission prediction query ids must be unique")
    if query_ids != prediction_ids:
        raise ValueError(
            "predictions must cover evaluator query ids in the same order"
        )
    selected_count = 0
    for record in normalized_predictions:
        if len(record.selected_paper_ids) != len(set(record.selected_paper_ids)):
            raise ValueError(
                f"duplicate selected paper id for query: {record.query_id}"
            )
        selected_count += len(record.selected_paper_ids)
    return {
        "query_count": len(normalized_queries),
        "prediction_count": len(normalized_predictions),
        "selected_paper_id_count": selected_count,
        "query_order_matches": True,
    }


__all__ = ["validate_submission_records"]
