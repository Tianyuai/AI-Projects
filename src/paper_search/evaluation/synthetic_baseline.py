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


class _SyntheticSearchService(Protocol):
    async def __call__(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResponse: ...


def _validate_synthetic_requests(
    requests: Sequence[object],
) -> tuple[SearchRequest, ...]:
    """Return an immutable validated catalog before any query executes."""
    if not requests:
        raise ValueError("synthetic query catalog must not be empty")
    ordered = requests if isinstance(requests, tuple) else tuple(requests)
    seen: set[str] = set()
    validated: list[SearchRequest] = []
    for candidate in ordered:
        if not isinstance(candidate, SearchRequest):
            raise ValueError("synthetic query catalog contains invalid request")
        try:
            original = candidate.model_dump(mode="python", warnings="error")
            request = SearchRequest.model_validate(original, strict=True)
            if request.model_dump(mode="python") != original:
                raise ValueError("request is not in canonical form")
        except Exception as error:
            raise ValueError(
                "synthetic query catalog contains invalid request"
            ) from error
        if request.query_id in seen:
            raise ValueError(f"duplicate query_id: {request.query_id}")
        seen.add(request.query_id)
        validated.append(request)
    return tuple(validated)


async def _run_synthetic_batch(
    requests: Sequence[object],
    *,
    search_service: _SyntheticSearchService,
    output: Path,
) -> list[InternalPredictionRecord]:
    """Run an ordered synthetic batch and isolate query-level failures."""
    ordered = _validate_synthetic_requests(requests)
    records: list[InternalPredictionRecord] = []
    for request in ordered:
        try:
            response = await search_service(request)
            if not isinstance(response, StructuredSearchResponse):
                raise ValueError("search service returned an invalid response")
            original = response.model_dump(mode="python", warnings="error")
            validated_response = StructuredSearchResponse.model_validate(
                original,
                strict=True,
            )
            if validated_response.model_dump(mode="python") != original:
                raise ValueError("search service response is not in canonical form")
            if validated_response.query_id != request.query_id:
                raise ValueError("response query_id does not match request query_id")
            record = prediction_from_response(validated_response)
        except Exception:
            record = InternalPredictionRecord(
                query_id=request.query_id,
                selected_paper_ids=[],
            )
        records.append(record)
    return write_prediction_records(output, records)


async def run_synthetic_baseline(
    *,
    output: Path,
) -> list[InternalPredictionRecord]:
    """Run the fixed offline synthetic catalog through the fixed mock stack."""
    return await _run_synthetic_batch(
        SYNTHETIC_QUERIES,
        search_service=build_synthetic_search_service(),
        output=output,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write deterministic Task 8C synthetic predictions",
        allow_abbrev=False,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        asyncio.run(
            run_synthetic_baseline(
                output=args.output,
            )
        )
    except (OSError, ValueError):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
