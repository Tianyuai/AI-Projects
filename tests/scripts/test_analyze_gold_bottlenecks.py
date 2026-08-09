from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

try:
    from scripts.analyze_gold_bottlenecks import (
        AvailabilityStatus,
        DiagnosticUsage,
        GoldIndex,
        OfflineContext,
        PipelineStage,
        ProbeBatch,
        ProbeCounters,
        assemble_report,
        assert_safe_report,
        load_offline_context,
        write_atomic_text,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts.analyze_gold_bottlenecks":
        raise

    AvailabilityStatus = Literal[
        "available",
        "exact_not_found",
        "unknown_transient",
        "invalid_identifier",
        "integrity_failure",
    ]
    PipelineStage = Literal[
        "selected_top50",
        "ranked_outside_top50",
        "filtered_out",
        "not_retrieved",
    ]

    @dataclass(frozen=True)
    class GoldIndex:
        query_count: int
        raw_gold_identifier_count: int
        normalized_query_work_count: int
        unique_work_count: int
        terminal_identifier_counts: dict[str, int]
        query_to_works: dict[str, frozenset[str]]
        work_to_identifier_kind: dict[str, str]

    @dataclass(frozen=True)
    class OfflineContext:
        gold_index: GoldIndex
        pipeline_stage_by_association: dict[tuple[str, str], PipelineStage]
        unique_stage_counts: dict[str, int]
        input_hashes: dict[str, str]
        source_run_id: str
        source_git_sha: str

    @dataclass(frozen=True)
    class ProbeCounters:
        http_attempts: int
        http_status_counts: dict[str, int]
        timeout_count: int

    @dataclass(frozen=True)
    class ProbeBatch:
        status_by_work: dict[str, AvailabilityStatus]
        counters: ProbeCounters

    @dataclass(frozen=True)
    class DiagnosticUsage:
        unique_requests_planned: int
        http_attempts: int
        retries: int
        http_200: int
        http_404: int
        http_429: int
        http_5xx: int
        timeouts: int
        ledger_checkpoint_before_sha256: str
        ledger_checkpoint_after_sha256: str

    def _missing_api(*args: object, **kwargs: object) -> Any:
        raise AssertionError("diagnostic module is not implemented")

    assemble_report = _missing_api
    assert_safe_report = _missing_api
    load_offline_context = _missing_api
    write_atomic_text = _missing_api


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = ROOT / "runs" / "dev-20260809T061903Z-9bd861e90299"
GOLD_PATH = ROOT / "data" / "dev" / "gold.jsonl"
ID_MAP_PATH = ROOT / "data" / "identifier-map.json"


def _synthetic_context() -> OfflineContext:
    index = GoldIndex(
        query_count=2,
        raw_gold_identifier_count=3,
        normalized_query_work_count=3,
        unique_work_count=3,
        terminal_identifier_counts={"doi": 2, "openalex": 1},
        query_to_works={
            "q1": frozenset({"doi:10.1000/a", "openalex:W1"}),
            "q2": frozenset({"doi:10.1000/b"}),
        },
        work_to_identifier_kind={
            "doi:10.1000/a": "doi",
            "doi:10.1000/b": "doi",
            "openalex:W1": "openalex",
        },
    )
    return OfflineContext(
        gold_index=index,
        pipeline_stage_by_association={
            ("q1", "doi:10.1000/a"): "selected_top50",
            ("q1", "openalex:W1"): "ranked_outside_top50",
            ("q2", "doi:10.1000/b"): "not_retrieved",
        },
        unique_stage_counts={
            "selected_top50": 1,
            "ranked_outside_top50": 1,
            "filtered_out": 0,
            "not_retrieved": 1,
        },
        input_hashes={
            "gold_sha256": "a" * 64,
            "identifier_map_sha256": "b" * 64,
            "executions_sha256": "c" * 64,
            "business_results_sha256": "d" * 64,
            "gates_sha256": "e" * 64,
            "run_sha256": "f" * 64,
        },
        source_run_id="dev-synthetic",
        source_git_sha="1" * 40,
    )


def _synthetic_probe() -> ProbeBatch:
    return ProbeBatch(
        status_by_work={
            "doi:10.1000/a": "available",
            "doi:10.1000/b": "exact_not_found",
            "openalex:W1": "available",
        },
        counters=ProbeCounters(
            http_attempts=3,
            http_status_counts={"200": 2, "404": 1, "429": 0, "5xx": 0},
            timeout_count=0,
        ),
    )


def _synthetic_usage() -> DiagnosticUsage:
    return DiagnosticUsage(
        unique_requests_planned=3,
        http_attempts=3,
        retries=0,
        http_200=2,
        http_404=1,
        http_429=0,
        http_5xx=0,
        timeouts=0,
        ledger_checkpoint_before_sha256="2" * 64,
        ledger_checkpoint_after_sha256="3" * 64,
    )


def test_offline_fixed_inputs_rebuild_expected_denominators_and_stages() -> None:
    context = load_offline_context(SOURCE_RUN, GOLD_PATH, ID_MAP_PATH)

    assert (
        context.gold_index.query_count,
        context.gold_index.raw_gold_identifier_count,
        context.gold_index.normalized_query_work_count,
        context.gold_index.unique_work_count,
    ) == (60, 143, 139, 134)
    assert context.gold_index.terminal_identifier_counts == {"doi": 128, "openalex": 6}

    association_counts = {
        stage: sum(
            value == stage
            for value in context.pipeline_stage_by_association.values()
        )
        for stage in (
            "selected_top50",
            "ranked_outside_top50",
            "filtered_out",
            "not_retrieved",
        )
    }
    assert association_counts == {
        "selected_top50": 8,
        "ranked_outside_top50": 6,
        "filtered_out": 0,
        "not_retrieved": 125,
    }
    assert context.unique_stage_counts == {
        "selected_top50": 8,
        "ranked_outside_top50": 6,
        "filtered_out": 0,
        "not_retrieved": 122,
    }


def test_assemble_report_conserves_denominators_and_reuses_work_status() -> None:
    payload = assemble_report(_synthetic_context(), _synthetic_probe(), _synthetic_usage())

    assert set(payload) == {
        "schema_version",
        "source_run_id",
        "source_git_sha",
        "input_hashes",
        "counts",
        "availability",
        "pipeline_stages",
        "cross_tab",
        "query_coverage",
        "usage",
        "diagnostic_complete",
        "recommended_direction",
        "reason_codes",
    }
    assert payload["schema_version"] == "gold-bottleneck-attribution-v1"
    assert sum(payload["availability"].values()) == 3
    assert sum(payload["pipeline_stages"].values()) == 3
    assert sum(
        count
        for row in payload["cross_tab"].values()
        for count in row.values()
    ) == 3
    assert payload["availability"] == {
        "available": 2,
        "exact_not_found": 1,
        "unknown_transient": 0,
        "invalid_identifier": 0,
        "integrity_failure": 0,
    }
    assert payload["recommended_direction"] is None


@pytest.mark.parametrize(
    ("availability", "pipeline", "expected_direction"),
    [
        ("exact_not_found", "not_retrieved", "new_data_source_probe"),
        ("available", "not_retrieved", "retrieval_query_evolution_probe"),
        ("available", "filtered_out", "hard_filter_diagnosis"),
        ("available", "ranked_outside_top50", "selector_rerank_offline"),
    ],
)
def test_direction_recommendation_requires_one_complete_strictly_largest_bucket(
    availability: AvailabilityStatus,
    pipeline: PipelineStage,
    expected_direction: str,
) -> None:
    context = _synthetic_context()
    context = OfflineContext(
        gold_index=context.gold_index,
        pipeline_stage_by_association={
            ("q1", "doi:10.1000/a"): pipeline,
            ("q1", "openalex:W1"): "selected_top50",
            ("q2", "doi:10.1000/b"): "selected_top50",
        },
        unique_stage_counts=context.unique_stage_counts,
        input_hashes=context.input_hashes,
        source_run_id=context.source_run_id,
        source_git_sha=context.source_git_sha,
    )
    probe = ProbeBatch(
        status_by_work={
            "doi:10.1000/a": availability,
            "doi:10.1000/b": "available",
            "openalex:W1": "available",
        },
        counters=_synthetic_probe().counters,
    )

    payload = assemble_report(context, probe, _synthetic_usage())

    assert payload["diagnostic_complete"] is True
    assert payload["recommended_direction"] == expected_direction


def test_tie_or_incomplete_diagnostic_has_no_direction() -> None:
    context = _synthetic_context()
    probe = ProbeBatch(
        status_by_work={
            "doi:10.1000/a": "unknown_transient",
            "doi:10.1000/b": "exact_not_found",
            "openalex:W1": "available",
        },
        counters=_synthetic_probe().counters,
    )

    payload = assemble_report(context, probe, _synthetic_usage())

    assert payload["diagnostic_complete"] is False
    assert payload["recommended_direction"] is None
    assert payload["reason_codes"] == [
        "unknown_transient_present",
    ]


def test_safe_report_rejects_extra_keys_forbidden_keys_and_forbidden_values() -> None:
    payload = assemble_report(_synthetic_context(), _synthetic_probe(), _synthetic_usage())

    with pytest.raises(ValueError, match="extra keys"):
        assert_safe_report({**payload, "unexpected": 1})

    with pytest.raises(ValueError, match="forbidden key"):
        assert_safe_report({**payload, "query_id": "not-allowed"})

    unsafe = json.loads(json.dumps(payload))
    unsafe["source_run_id"] = "https://example.invalid/secret"
    with pytest.raises(ValueError, match="forbidden string"):
        assert_safe_report(unsafe)


def test_atomic_write_preserves_existing_destination_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "report.json"
    destination.write_text("old", encoding="utf-8")

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        write_atomic_text(destination, "new")
    assert destination.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"gate_result": "failed"}, "Gate"),
        ({"status": "failed"}, "status"),
    ],
)
def test_offline_sealed_run_validation_fails_closed(
    tmp_path: Path,
    mutation: dict[str, Any],
    message: str,
) -> None:
    run_copy = tmp_path / SOURCE_RUN.name
    run_copy.mkdir()
    for name in (
        "run.json",
        "gates.json",
        "executions.jsonl",
        "business-results.jsonl",
    ):
        source = SOURCE_RUN / name
        target = run_copy / name
        target.write_bytes(source.read_bytes())
    run_record = json.loads((run_copy / "run.json").read_text(encoding="utf-8"))
    run_record.update(mutation)
    (run_copy / "run.json").write_text(
        json.dumps(run_record),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_offline_context(run_copy, GOLD_PATH, ID_MAP_PATH)
