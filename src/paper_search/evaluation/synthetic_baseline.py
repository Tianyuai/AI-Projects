"""Deterministic offline synthetic baseline batch."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from paper_search.api.contracts import SearchRequest
from paper_search.domain.models import StructuredSearchResponse
from paper_search.evaluation.official_adapter import InternalPredictionRecord
from paper_search.evaluation.predictions import (
    prediction_from_response,
    write_prediction_records,
)
from paper_search.evaluation.synthetic_mocks import build_synthetic_search_service


SYNTHETIC_QUERIES = (
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


class SyntheticSearchService(Protocol):
    async def __call__(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResponse: ...


def validate_synthetic_requests(
    requests: Sequence[SearchRequest],
) -> tuple[SearchRequest, ...]:
    """Return an immutable validated catalog before any query executes."""
    if not requests:
        raise ValueError("synthetic query catalog must not be empty")
    ordered = requests if isinstance(requests, tuple) else tuple(requests)
    seen: set[str] = set()
    for request in ordered:
        if request.query_id in seen:
            raise ValueError(f"duplicate query_id: {request.query_id}")
        seen.add(request.query_id)
    return ordered


async def run_synthetic_baseline(
    requests: Sequence[SearchRequest],
    *,
    search_service: SyntheticSearchService,
    output: Path,
) -> list[InternalPredictionRecord]:
    """Run an ordered synthetic batch and isolate query-level failures."""
    ordered = validate_synthetic_requests(requests)
    records: list[InternalPredictionRecord] = []
    for request in ordered:
        try:
            response = await search_service(request)
            if response.query_id != request.query_id:
                raise ValueError("response query_id does not match request query_id")
            record = prediction_from_response(response)
        except Exception:
            record = InternalPredictionRecord(
                query_id=request.query_id,
                selected_paper_ids=[],
            )
        records.append(record)
    return write_prediction_records(output, records)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write deterministic Task 8C synthetic predictions"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        asyncio.run(
            run_synthetic_baseline(
                SYNTHETIC_QUERIES,
                search_service=build_synthetic_search_service(),
                output=args.output,
            )
        )
    except (OSError, ValueError):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
