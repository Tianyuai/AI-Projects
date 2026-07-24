"""Synthetic structured-response prediction serialization."""

from collections.abc import Sequence
from pathlib import Path

from paper_search.domain.models import StructuredSearchResponse
from paper_search.evaluation.dataset import write_jsonl_atomic
from paper_search.evaluation.official_adapter import InternalPredictionRecord


def prediction_from_response(
    response: StructuredSearchResponse,
) -> InternalPredictionRecord:
    """Copy the fixed prediction fields without scores or label access."""
    return InternalPredictionRecord(
        query_id=response.query_id,
        selected_paper_ids=response.selected_paper_ids,
    )


def write_prediction_records(
    path: Path,
    records: Sequence[InternalPredictionRecord],
) -> list[InternalPredictionRecord]:
    """Validate and atomically write ordered deterministic prediction records."""
    ordered = list(records)
    seen: set[str] = set()
    for record in ordered:
        if record.query_id in seen:
            raise ValueError(f"duplicate query_id: {record.query_id}")
        seen.add(record.query_id)
    write_jsonl_atomic(path, ordered)
    return ordered


def write_response_predictions(
    path: Path,
    responses: Sequence[StructuredSearchResponse],
) -> list[InternalPredictionRecord]:
    """Convert structured responses and write deterministic predictions."""
    return write_prediction_records(
        path,
        [prediction_from_response(response) for response in responses],
    )
