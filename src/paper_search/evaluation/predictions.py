"""Synthetic structured-response prediction serialization."""

from collections.abc import Sequence
from pathlib import Path

from paper_search.domain.models import Paper, StructuredSearchResponse
from paper_search.evaluation.dataset import normalize_paper_id, write_jsonl_atomic
from paper_search.evaluation.official_adapter import InternalPredictionRecord


def paper_evaluation_id(paper: Paper) -> str:
    """Choose the strongest scorer-facing identity available for one paper."""

    if paper.arxiv_id:
        return normalize_paper_id(paper.arxiv_id, kind="arxiv")
    return paper.canonical_id


def prediction_from_response(
    response: StructuredSearchResponse,
) -> InternalPredictionRecord:
    """Emit scorer-compatible identities without changing internal canonical IDs."""
    papers_by_id = {
        item.paper.canonical_id: item.paper for item in response.fused_papers
    }
    selected_paper_ids: list[str] = []
    seen: set[str] = set()
    for canonical_id in response.selected_paper_ids:
        paper = papers_by_id.get(canonical_id)
        evaluation_id = (
            paper_evaluation_id(paper)
            if paper is not None
            else canonical_id
        )
        if evaluation_id in seen:
            continue
        seen.add(evaluation_id)
        selected_paper_ids.append(evaluation_id)
    return InternalPredictionRecord(
        query_id=response.query_id,
        selected_paper_ids=selected_paper_ids,
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


__all__ = [
    "paper_evaluation_id",
    "prediction_from_response",
    "write_prediction_records",
    "write_response_predictions",
]
