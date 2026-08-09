from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import IdentifierMap
from scripts.analyze_title_retention import (
    analyze_run,
    assert_aggregate_only,
    reserve_title_slots,
    retains_baseline_golds,
    weighted_rrf_ids,
)


def _paper(identifier: str) -> Paper:
    return Paper(canonical_id=f"openalex:{identifier}", title=identifier)


def test_weighted_rrf_preserves_denominator_and_promotes_title_source() -> None:
    openalex = [_paper("W1"), _paper("W2")]
    titles = [_paper("W2"), _paper("W3")]
    eligible = {"openalex:W1", "openalex:W2", "openalex:W3"}

    baseline = weighted_rrf_ids(
        openalex,
        titles,
        eligible,
        title_weight=1.0,
        limit=3,
    )
    boosted = weighted_rrf_ids(
        openalex,
        titles,
        eligible,
        title_weight=3.0,
        limit=3,
    )

    assert baseline[0] == "openalex:W2"
    assert boosted.index("openalex:W3") < boosted.index("openalex:W1")


def test_weighted_rrf_is_deterministic_for_ties() -> None:
    eligible = {"openalex:W1", "openalex:W2"}

    result = weighted_rrf_ids(
        [_paper("W2")],
        [_paper("W1")],
        eligible,
        title_weight=1.0,
        limit=2,
    )

    assert result == ["openalex:W1", "openalex:W2"]


@pytest.mark.parametrize("weight", [0.0, -1.0, math.inf, math.nan])
def test_weighted_rrf_rejects_invalid_title_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="title_weight"):
        weighted_rrf_ids([], [], set(), title_weight=weight)


def test_reserve_title_slots_replaces_lowest_non_title_only() -> None:
    result = reserve_title_slots(
        ["openalex:W1", "openalex:W2", "openalex:W3"],
        ["openalex:W4", "openalex:W2"],
        {"openalex:W1", "openalex:W2", "openalex:W3", "openalex:W4"},
        minimum=2,
        limit=3,
    )

    assert result == ["openalex:W1", "openalex:W2", "openalex:W4"]


def test_reserve_title_slots_never_adds_filtered_candidate() -> None:
    result = reserve_title_slots(
        ["openalex:W1", "openalex:W2"],
        ["openalex:W3"],
        {"openalex:W1", "openalex:W2"},
        minimum=1,
        limit=2,
    )

    assert result == ["openalex:W1", "openalex:W2"]


def test_reserve_title_slots_deduplicates_inputs() -> None:
    result = reserve_title_slots(
        ["openalex:W1", "openalex:W1", "openalex:W2"],
        ["openalex:W3", "openalex:W3"],
        {"openalex:W1", "openalex:W2", "openalex:W3"},
        minimum=1,
        limit=3,
    )

    assert result == ["openalex:W1", "openalex:W2", "openalex:W3"]


@pytest.mark.parametrize(
    ("minimum", "limit"),
    [(-1, 50), (True, 50), (1, 0), (2, 1)],
)
def test_reserve_title_slots_rejects_invalid_bounds(
    minimum: int,
    limit: int,
) -> None:
    with pytest.raises(ValueError):
        reserve_title_slots([], [], set(), minimum=minimum, limit=limit)


def test_retains_baseline_golds_requires_each_query_hit_to_survive() -> None:
    gold = {
        "q1": {"openalex:W1", "openalex:W2"},
        "q2": {"openalex:W3"},
    }
    baseline = {
        "q1": ["openalex:W1"],
        "q2": ["openalex:W3"],
    }
    retained = {
        "q1": ["openalex:W1", "openalex:W2"],
        "q2": ["openalex:W3"],
    }
    regressed = {
        "q1": ["openalex:W2"],
        "q2": ["openalex:W3"],
    }

    assert retains_baseline_golds(
        gold,
        baseline,
        retained,
        IdentifierMap({}),
    )
    assert not retains_baseline_golds(
        gold,
        baseline,
        regressed,
        IdentifierMap({}),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"query_id": "private"},
        {"nested": {"paper_id": "private"}},
        {"items": [{"title": "private"}]},
        {"request_id": "private"},
        {"response": "private"},
    ],
)
def test_aggregate_payload_rejects_private_record_keys(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="aggregate-only"):
        assert_aggregate_only(payload)


def test_aggregate_payload_accepts_counts_metrics_and_parameters() -> None:
    assert_aggregate_only(
        {
            "schema_version": "title-retention-offline-v1",
            "query_count": 60,
            "metrics": {"macro_f1": 0.1},
            "parameters": {"title_weight": 1.5, "minimum_title_slots": 3},
            "reason_codes": ["macro_f1_not_improved"],
        }
    )


def _write_synthetic_run(root: Path, *, reverse_selected: bool = False) -> None:
    responses = root / "snapshots" / "responses"
    (responses / "llm").mkdir(parents=True)
    (responses / "openalex").mkdir(parents=True)
    title_llm = {
        "choices": [
            {"message": {"content": json.dumps({"titles": ["Synthetic"]})}}
        ]
    }
    baseline_oa = {
        "meta": {"next_cursor": None},
        "results": [
            {"id": "https://openalex.org/W1", "title": "One"},
            {"id": "https://openalex.org/W2", "title": "Two"},
        ],
    }
    partial_oa = {
        "meta": {"next_cursor": None},
        "results": [
            {"id": "https://openalex.org/W3", "title": "Three"},
            {"id": None, "title": None},
        ],
    }
    successful_title_oa = {
        "meta": {"next_cursor": None},
        "results": [
            {"id": "https://openalex.org/W4", "title": "Four"},
        ],
    }
    (responses / "llm" / "titles.bin").write_text(
        json.dumps(title_llm),
        encoding="utf-8",
    )
    (responses / "openalex" / "baseline.bin").write_text(
        json.dumps(baseline_oa),
        encoding="utf-8",
    )
    (responses / "openalex" / "partial.bin").write_text(
        json.dumps(partial_oa),
        encoding="utf-8",
    )
    (responses / "openalex" / "successful-title.bin").write_text(
        json.dumps(successful_title_oa),
        encoding="utf-8",
    )
    diagnostics = [
        {
            "dependency": "openalex",
            "errors": [],
            "snapshot_refs": [
                {"snapshot_path": "responses/openalex/baseline.bin"}
            ],
        },
        {
            "dependency": "llm",
            "errors": [],
            "snapshot_refs": [
                {"snapshot_path": "responses/llm/titles.bin"}
            ],
        },
        {
            "dependency": "openalex",
            "errors": [{"code": "provider_error"}],
            "snapshot_refs": [
                {"snapshot_path": "responses/openalex/partial.bin"}
            ],
        },
        {
            "dependency": "openalex",
            "errors": [],
            "snapshot_refs": [
                {"snapshot_path": "responses/openalex/successful-title.bin"}
            ],
        },
    ]
    execution = {
        "query_id": "q1",
        "diagnostics": diagnostics,
        "post_filter_paper_ids": ["openalex:W1", "openalex:W2"],
    }
    selected = ["openalex:W2", "openalex:W1"] if reverse_selected else [
        "openalex:W1",
        "openalex:W2",
    ]
    business = {
        "query_id": "q1",
        "selected_paper_ids": selected,
        "query_analysis": {
            "query_spec": {
                "original_query": "synthetic",
                "research_goal": "synthetic",
            }
        },
    }
    (root / "executions.jsonl").write_text(
        json.dumps(execution) + "\n",
        encoding="utf-8",
    )
    (root / "business-results.jsonl").write_text(
        json.dumps(business) + "\n",
        encoding="utf-8",
    )
    (root / "run.json").write_text(
        json.dumps({"run_id": "synthetic", "source_git_sha": "synthetic"}),
        encoding="utf-8",
    )


def test_analyze_run_reconstructs_then_recovers_partial_page(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_synthetic_run(run)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "synthetic",
                "relevant_paper_ids": ["openalex:W3"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    identifier_map = tmp_path / "identifier-map.json"
    identifier_map.write_text("{}", encoding="utf-8")

    payload = analyze_run(
        run,
        gold,
        identifier_map,
        expected_query_count=1,
        expected_total_selected=2,
        require_unchanged_filter=False,
    )

    assert payload["reconstruction"] == {
        "exact_query_sequences": 1,
        "query_count": 1,
        "total_selected": 2,
    }
    assert payload["partial_success"]["newly_eligible_papers"] == 1
    assert payload["partial_success"]["historical_pool_exact_gold"] == 0
    assert payload["partial_success"]["repaired_pool_exact_gold"] == 1
    assert payload["partial_success"]["newly_eligible_exact_gold"] == 1
    assert payload["partial_success"]["newly_eligible_gold_queries"] == 1
    variants = {item["name"]: item for item in payload["variants"]}
    assert variants["historical_rrf"]["exact_gold_count"] == 0
    assert variants["repaired_rrf"]["exact_gold_count"] == 1
    assert variants["repaired_rrf"]["changed_sequence_queries"] == 1
    assert variants["repaired_rrf"]["changed_set_queries"] == 1
    assert (
        variants["repaired_rrf"]["metrics"]["macro_f1"]
        > variants["historical_rrf"]["metrics"]["macro_f1"]
    )
    assert_aggregate_only(payload)


def test_analyze_run_fails_closed_on_reconstruction_mismatch(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_synthetic_run(run, reverse_selected=True)
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "synthetic",
                "relevant_paper_ids": ["openalex:W3"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    identifier_map = tmp_path / "identifier-map.json"
    identifier_map.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="historical Top-50 reconstruction mismatch",
    ):
        analyze_run(
            run,
            gold,
            identifier_map,
            expected_query_count=1,
            expected_total_selected=2,
            require_unchanged_filter=False,
        )
