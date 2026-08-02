from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from paper_search.domain.models import (
    CandidateEvidence,
    Paper,
    RankedPaper,
    StructuredSearchResponse,
)
from paper_search.evaluation.business_results import (
    BusinessResultRecord,
    business_result_from_response,
    business_result_sha256,
    canonical_business_result_bytes,
    compare_business_results,
)


def _record(query_id: str = "q1", **changes: object) -> BusinessResultRecord:
    values: dict[str, object] = {
        "schema_version": "business-result-v1",
        "query_id": query_id,
        "query_analysis": None,
        "selected_paper_ids": ["openalex:W1"],
        "high_relevance": [],
        "partial_relevance": [],
        "citation_edges": [],
        "is_partial": False,
        "planner_status": "primary",
        "planner_fallback": False,
        "warnings": [],
        "stop_reason": "completed",
        "hard_failure_code": None,
    }
    values.update(changes)
    return BusinessResultRecord.model_validate(values)


def test_canonical_business_result_bytes_are_compact_sorted_utf8_jsonl() -> None:
    encoded = canonical_business_result_bytes(_record(query_id="查询"))

    assert encoded.endswith(b"\n")
    assert b" " not in encoded
    assert "查询".encode() in encoded
    assert encoded.startswith(b'{"citation_edges":')


def test_business_result_record_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        BusinessResultRecord.model_validate(
            _record().model_dump(mode="json") | {"run_id": "transport-only"}
        )


def test_business_hash_changes_for_semantic_fields() -> None:
    original = _record()

    for changed in (
        _record(selected_paper_ids=[]),
        _record(warnings=["provider degraded"]),
        _record(stop_reason="soft_stop"),
    ):
        assert business_result_sha256(changed) != business_result_sha256(original)


def test_canonical_business_result_rejects_nonfinite_nested_evidence() -> None:
    evidence = CandidateEvidence(
        paper_id="openalex:W1",
        lexical_score=float("nan"),
        embedding_score=0.0,
        constraint_coverage=0.0,
        source_agreement=0.0,
        authority_score=0.0,
        recency_score=0.0,
        final_score=0.0,
        scoring_version="v1",
        relevance_level="high",
    )
    ranked = RankedPaper(
        paper=Paper(canonical_id="openalex:W1", title="One"),
        evidence=evidence,
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_business_result_bytes(_record(high_relevance=[ranked]))


def test_transport_fields_cannot_enter_business_record_or_hash() -> None:
    semantic = _record().model_dump()
    semantic.pop("schema_version")
    first = SimpleNamespace(**semantic, run_id="run-1", execution_mode="replay")
    second = SimpleNamespace(
        **semantic,
        run_id="run-2",
        request_id="request-2",
        execution_mode="live",
        snapshot_refs=["different"],
        cache_hit=False,
        usage={"search_api_calls": 99},
        diagnostics=["different"],
    )

    assert business_result_sha256(
        business_result_from_response(cast(StructuredSearchResponse, first))
    ) == business_result_sha256(
        business_result_from_response(cast(StructuredSearchResponse, second))
    )


def test_compare_business_results_accepts_equal_ordered_records() -> None:
    compare_business_results([_record("q1"), _record("q2")], [_record("q1"), _record("q2")])


def test_compare_business_results_rejects_missing_extra_duplicate_reordered_and_unequal() -> None:
    expected = [_record("q1"), _record("q2")]

    cases = (
        (expected, [_record("q1")], "query cardinality"),
        ([_record("q1")], expected, "query cardinality"),
        (expected, [_record("q1"), _record("q1")], "duplicate query_id"),
        (expected, list(reversed(expected)), "query order"),
        (expected, [_record("q1"), _record("q2", warnings=["changed"])], "business result mismatch"),
    )
    for capture, replay, message in cases:
        try:
            compare_business_results(capture, replay)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"expected {message}")
