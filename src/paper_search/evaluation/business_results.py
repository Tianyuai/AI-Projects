"""Canonical, transport-independent business results for evaluation replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal

from paper_search.application.contracts import SearchErrorCode
from paper_search.domain.models import (
    DomainModel,
    NonEmptyStr,
    PlannerStatus,
    QueryAnalysisResult,
    RankedPaper,
    ResolvedCitationEdge,
    Sha256,
    StructuredSearchResponse,
)


class BusinessResultRecord(DomainModel):
    """Only search semantics that must remain equal across capture and replay."""

    schema_version: Literal["business-result-v1"] = "business-result-v1"
    query_id: NonEmptyStr
    query_analysis: QueryAnalysisResult | None
    selected_paper_ids: list[NonEmptyStr]
    high_relevance: list[RankedPaper]
    partial_relevance: list[RankedPaper]
    citation_edges: list[ResolvedCitationEdge]
    is_partial: bool
    planner_status: PlannerStatus | None
    planner_fallback: bool
    warnings: list[NonEmptyStr]
    stop_reason: NonEmptyStr
    hard_failure_code: SearchErrorCode | None


def business_result_from_response(
    response: StructuredSearchResponse,
) -> BusinessResultRecord:
    """Project a success response onto transport-independent business fields."""
    return BusinessResultRecord(
        query_id=response.query_id,
        query_analysis=response.query_analysis,
        selected_paper_ids=response.selected_paper_ids,
        high_relevance=response.high_relevance,
        partial_relevance=response.partial_relevance,
        citation_edges=response.citation_edges,
        is_partial=response.is_partial,
        planner_status=response.planner_status,
        planner_fallback=response.planner_fallback,
        warnings=response.warnings,
        stop_reason=response.stop_reason,
        hard_failure_code=None,
    )


def hard_failure_business_result(
    *,
    query_id: str,
    error_code: SearchErrorCode,
    stop_reason: str,
) -> BusinessResultRecord:
    """Build the stable empty evidence projection for a hard failure."""
    return BusinessResultRecord(
        query_id=query_id,
        query_analysis=None,
        selected_paper_ids=[],
        high_relevance=[],
        partial_relevance=[],
        citation_edges=[],
        is_partial=False,
        planner_status=None,
        planner_fallback=False,
        warnings=[],
        stop_reason=stop_reason,
        hard_failure_code=error_code,
    )


def canonical_business_result_bytes(record: BusinessResultRecord) -> bytes:
    """Encode one deterministic UTF-8 JSONL business record."""
    return (
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def business_result_sha256(record: BusinessResultRecord) -> Sha256:
    """Hash exactly the canonical business bytes."""
    return f"sha256:{hashlib.sha256(canonical_business_result_bytes(record)).hexdigest()}"


def compare_business_results(
    capture: Sequence[BusinessResultRecord],
    replay: Sequence[BusinessResultRecord],
) -> None:
    """Require identical unique query ordering and identical business semantics."""
    capture_records = list(capture)
    replay_records = list(replay)
    for label, records in (("capture", capture_records), ("replay", replay_records)):
        seen: set[str] = set()
        for record in records:
            if record.query_id in seen:
                raise ValueError(f"{label} duplicate query_id: {record.query_id}")
            seen.add(record.query_id)

    if len(capture_records) != len(replay_records):
        raise ValueError(
            "query cardinality mismatch: "
            f"capture={len(capture_records)}, replay={len(replay_records)}"
        )

    capture_ids = [record.query_id for record in capture_records]
    replay_ids = [record.query_id for record in replay_records]
    if capture_ids != replay_ids:
        raise ValueError(
            f"query order mismatch: capture={capture_ids!r}, replay={replay_ids!r}"
        )

    for captured, replayed in zip(capture_records, replay_records, strict=True):
        if canonical_business_result_bytes(captured) != canonical_business_result_bytes(
            replayed
        ):
            raise ValueError(f"business result mismatch for query_id: {captured.query_id}")
