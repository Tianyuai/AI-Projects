from __future__ import annotations

from scripts.freeze_fusion_activation_context import (
    _atomic_bytes,
    _merge_constraint_row,
    _merge_task_row,
    _project_context,
    _query_header,
    _replace_deterministic_constraint_signals,
)
from paper_search.learning.gated_feature_fusion_ranker import FusionQueryContext
from paper_search.learning.query_constraint_profile import QueryConstraintProfile


def test_query_header_reads_only_the_stable_shard_prefix() -> None:
    raw = (
        b'{"query_id":"q1","query":"papers using \\"quoted\\" methods",'
        b'"gold_paper_ids":[],"candidates":[{"large":"payload"}]}'
    )

    assert _query_header(raw) == ("q1", 'papers using "quoted" methods')


def test_local_effective_task_replaces_non_gated_frozen_task() -> None:
    row = {
        "tasks": [{"normalized_value": "old", "confidence": 0.5}],
        "ambiguous_fields": [],
        "task_label_status": "review_failed_fallback",
    }

    merged = _merge_task_row(row, tasks=["ranking"])

    assert merged["tasks"] == [
        {
            "normalized_value": "ranking",
            "confidence": 0.9,
            "evidence_span": "ranking",
            "strength": "must",
        }
    ]
    assert merged["task_label_status"] == "runtime_deterministic"


def test_local_effective_task_replaces_ambiguous_frozen_task() -> None:
    row = {
        "tasks": [{"normalized_value": "old", "confidence": 1.0}],
        "ambiguous_fields": ["tasks"],
        "task_label_status": "accepted",
    }

    merged = _merge_task_row(row, tasks=["ranking"])

    assert merged["tasks"][0]["normalized_value"] == "ranking"
    assert merged["ambiguous_fields"] == []


def test_local_effective_method_replaces_low_confidence_frozen_method() -> None:
    row = {
        "labels": ["method"],
        "methods": ["old method"],
        "datasets": [],
        "label_sources": {"method": "model"},
        "label_confidence": {"method": 0.5},
        "evidence": {"method": ["old method"]},
        "status": "review_required",
    }
    profile = QueryConstraintProfile(
        labels=["method"],
        methods=["new explicit method"],
        constraint_count=1,
        confidence=0.9,
    )

    merged = _merge_constraint_row(
        row,
        profile=profile,
        effective_signals={"method"},
    )

    assert merged["methods"] == ["new explicit method"]
    assert merged["label_sources"]["method"] == "local_deterministic"
    assert merged["label_confidence"]["method"] == 0.9
    assert merged["status"] == "accepted"


def test_effective_method_drops_unrelated_low_confidence_label() -> None:
    row = {
        "labels": ["dataset"],
        "methods": [],
        "datasets": ["uncertain dataset"],
        "label_sources": {"dataset": "model"},
        "label_confidence": {"dataset": 0.4},
        "evidence": {"dataset": ["uncertain dataset"]},
        "status": "review_required",
    }
    profile = QueryConstraintProfile(
        labels=["method"],
        methods=["new explicit method"],
        constraint_count=1,
        confidence=0.9,
    )

    merged = _merge_constraint_row(
        row,
        profile=profile,
        effective_signals={"method"},
    )

    assert merged["labels"] == ["method"]
    assert merged["datasets"] == []
    assert "dataset" not in merged["label_confidence"]


def test_signal_projection_removes_sibling_entity_signal() -> None:
    context = FusionQueryContext(
        constraint_profile=QueryConstraintProfile(
            labels=["method", "dataset"],
            methods=["m"],
            datasets=["d"],
            confidence=1.0,
            constraint_count=2,
        )
    )

    projected = _project_context(context, "method")

    assert projected.constraint_profile is not None
    assert projected.constraint_profile.labels == ["method"]
    assert projected.constraint_profile.methods == ["m"]
    assert projected.constraint_profile.datasets == []


def test_atomic_freeze_output_refuses_different_overwrite(tmp_path) -> None:
    path = tmp_path / "frozen.json"
    _atomic_bytes(path, b"one")
    _atomic_bytes(path, b"one")

    try:
        _atomic_bytes(path, b"two")
    except FileExistsError:
        pass
    else:
        raise AssertionError("different immutable bytes were overwritten")


def test_local_year_replaces_stale_single_year_bounds() -> None:
    row = {
        "labels": ["year"],
        "year_from": 2020,
        "year_to": 2020,
        "label_sources": {"year": "rule"},
        "label_confidence": {"year": 1.0},
        "evidence": {"year": ["2020"]},
        "status": "accepted",
    }
    profile = QueryConstraintProfile(
        labels=["year"],
        year_to=2020,
        confidence=1.0,
        constraint_count=1,
    )

    replaced = _replace_deterministic_constraint_signals(
        row, profile=profile, signals={"year"}
    )

    assert replaced["year_from"] is None
    assert replaced["year_to"] == 2020
    assert replaced["label_sources"]["year"] == "local_deterministic"


def test_local_parser_removes_entity_name_year_label() -> None:
    row = {
        "labels": ["year"],
        "year_from": 2023,
        "year_to": 2023,
        "label_sources": {"year": "rule"},
        "label_confidence": {"year": 1.0},
        "evidence": {"year": ["2023"]},
        "status": "accepted",
    }
    profile = QueryConstraintProfile(labels=[], confidence=1.0, constraint_count=0)

    replaced = _replace_deterministic_constraint_signals(
        row, profile=profile, signals={"year"}
    )

    assert "year" not in replaced["labels"]
    assert replaced["year_from"] is None
    assert replaced["year_to"] is None
