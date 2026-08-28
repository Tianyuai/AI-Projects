from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "finalize_query_adaptive_recall_pilot.py"
)
SPEC = importlib.util.spec_from_file_location("finalize_recall_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _gate() -> dict[str, object]:
    return {
        "thresholds": {
            "minimum_gold_hit_query_count": 25,
            "minimum_gold_hit_query_rate": 0.07,
            "minimum_gold_hit_count": 25,
            "minimum_macro_candidate_recall": 0.04,
            "minimum_hit_query_count_by_required_stratum": {
                "negation": 1,
                "task": 1,
                "unstructured": 1,
            },
            "maximum_missing_action_identity_count": 0,
            "maximum_final_failed_query_count": 0,
            "maximum_llm_call_count": 0,
        }
    }


def test_validation_gate_passes_only_when_every_threshold_passes() -> None:
    metrics = {
        "gold_hit_query_count": 25,
        "gold_hit_query_rate": 0.071,
        "gold_hit_count": 26,
        "macro_candidate_recall": 0.041,
    }
    result = MODULE._evaluate_validation_gate(
        gate=_gate(),
        metrics=metrics,
        strata={
            "negation": {"hit_query_count": 1},
            "task": {"hit_query_count": 2},
            "unstructured": {"hit_query_count": 1},
        },
        missing_action_count=0,
        final_failed_query_count=0,
        llm_call_count=0,
    )

    assert result["passed"] is True
    assert all(item["passed"] for item in result["checks"])


def test_validation_gate_reports_failed_required_stratum() -> None:
    metrics = {
        "gold_hit_query_count": 25,
        "gold_hit_query_rate": 0.071,
        "gold_hit_count": 26,
        "macro_candidate_recall": 0.041,
    }
    result = MODULE._evaluate_validation_gate(
        gate=_gate(),
        metrics=metrics,
        strata={
            "negation": {"hit_query_count": 0},
            "task": {"hit_query_count": 2},
            "unstructured": {"hit_query_count": 1},
        },
        missing_action_count=0,
        final_failed_query_count=0,
        llm_call_count=0,
    )

    assert result["passed"] is False
    failed = [item["name"] for item in result["checks"] if not item["passed"]]
    assert failed == ["minimum_hit_query_count_by_required_stratum.negation"]


def test_gold_hits_merge_candidates_across_split_retrieval_attempts() -> None:
    candidates = [
        {"canonical_id": "openalex:W1", "doi": "10.1000/irrelevant"},
        {"canonical_id": "openalex:W2", "doi": "10.48550/arxiv.2401.00001"},
    ]

    assert MODULE._gold_hit_ids(
        ("arxiv:2401.00001", "doi:10.1000/missing"), candidates
    ) == ["arxiv:2401.00001"]


def test_reconciled_authorization_calls_exclude_non_dispatched_receipts() -> None:
    assert MODULE._reconciled_authorization_calls(
        observed_receipt_calls=3758,
        reconciliation={
            "failed_action_result_count": 1293,
            "conservative_in_flight_call_count": 1,
        },
    ) == 2466


def test_canary_index_accepts_multiple_queries_in_one_batch(tmp_path: Path) -> None:
    report = tmp_path / "canary-report.json"
    report.write_text(
        json.dumps(
            {
                "result": {
                    "per_query": [
                        {"query_id": "q-1"},
                        {"query_id": "q-2"},
                    ]
                },
                "usage": {"llm_calls": 0},
            }
        ),
        encoding="utf-8",
    )

    assert hasattr(MODULE, "_index_canary_reports")
    indexed = MODULE._index_canary_reports([report])

    assert set(indexed) == {"q-1", "q-2"}
    assert indexed["q-1"][1] == report
    assert indexed["q-2"][1] == report
