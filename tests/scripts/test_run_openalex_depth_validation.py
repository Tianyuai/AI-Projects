from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.run_openalex_depth_validation import (
    build_gold_blind_request_row,
    comparison_candidate_counts,
    select_disjoint_depth_rows,
    verify_gold_blind_request_plan,
)


def _proposal(query_id: str, *, signal: str = "unconstrained") -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": f"query text {query_id}",
        "gold_paper_ids": ["doi:10.1/private"],
        "signal": signal,
        "source_action": {
            "action_id": "policy-1",
            "action_type": "text_search",
            "payload": {"query_text": "derived retrieval text", "search_mode": "lexical"},
            "strategy": "learned_action_ranker",
        },
        "cursor": "opaque-next-cursor",
        "source_snapshot_sha256": "sha256:" + "a" * 64,
        "source_retrieval_sha256": "sha256:" + "b" * 64,
        "source_hit_count": 50,
    }


def test_depth_request_plan_contains_no_gold_or_final_test_material() -> None:
    request = build_gold_blind_request_row(_proposal("AutoScholarQuery_train_1"))

    serialized = json.dumps(request, sort_keys=True).casefold()
    assert request["query_id"] == "AutoScholarQuery_train_1"
    assert request["cursor"] == "opaque-next-cursor"
    assert request["rank_offset"] == 50
    assert "gold" not in serialized
    assert "final_test" not in serialized
    verify_gold_blind_request_plan([request])


def test_depth_selection_is_deterministic_disjoint_and_requires_exact_page_one() -> None:
    rows = [
        _proposal("AutoScholarQuery_train_1"),
        _proposal("AutoScholarQuery_train_2"),
        _proposal("AutoScholarQuery_train_3"),
        {**_proposal("AutoScholarQuery_train_4"), "source_hit_count": 49},
    ]

    first = select_disjoint_depth_rows(
        rows,
        prior_query_ids={"AutoScholarQuery_train_2"},
        limit=2,
        seed="depth-v1",
    )
    second = select_disjoint_depth_rows(
        list(reversed(rows)),
        prior_query_ids={"AutoScholarQuery_train_2"},
        limit=2,
        seed="depth-v1",
    )

    assert [row["query_id"] for row in first] == [
        row["query_id"] for row in second
    ]
    assert {row["query_id"] for row in first} == {
        "AutoScholarQuery_train_1",
        "AutoScholarQuery_train_3",
    }


def test_depth_request_plan_rejects_hidden_gold_field() -> None:
    request = build_gold_blind_request_row(_proposal("AutoScholarQuery_train_1"))
    request["metadata"] = {"gold_id": "doi:10.1/private"}

    with pytest.raises(ValueError, match="Gold-blind"):
        verify_gold_blind_request_plan([request])


def test_depth_validation_cli_can_run_as_a_script_file() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_openalex_depth_validation.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_depth_comparison_exposes_the_aggregate_contract_counts() -> None:
    counts = comparison_candidate_counts(
        baseline_pre_cap=208,
        augmented_pre_cap=244,
        baseline_after_cap=200,
        augmented_after_cap=200,
    )

    assert counts == {
        "baseline_candidate_count": 200,
        "augmented_candidate_count": 200,
        "added_candidate_count": 0,
        "baseline_candidate_count_pre_cap": 208,
        "augmented_candidate_count_pre_cap": 244,
    }
