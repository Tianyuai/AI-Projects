from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "freeze_expanded_recall_training_package.py"
)
SPEC = importlib.util.spec_from_file_location("freeze_expanded_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_expanded_handoff_appends_unique_roots_without_changing_ready_ceiling() -> None:
    base = {
        "schema_version": "openalex-ranking-training-handoff-v1",
        "cumulative_unique_ready_query_ids": ["q1", "q2"],
        "ordered_receipt_roots": ["C:/base"],
        "online_llm_requests": 0,
        "test_partition_touched": False,
    }

    expanded = MODULE._expanded_handoff(
        base,
        receipt_roots=(Path("C:/base"), Path("C:/discovery"), Path("C:/validation")),
        validation_result_sha256="sha256:" + "1" * 64,
    )

    assert expanded["cumulative_unique_ready_query_ids"] == ["q1", "q2"]
    assert expanded["ordered_receipt_roots"] == [
        str(Path("C:/base").resolve()),
        str(Path("C:/discovery").resolve()),
        str(Path("C:/validation").resolve()),
    ]
    assert expanded["high_recall_candidate_supplement"]["llm_request_count"] == 0


def test_expanded_handoff_preserves_prior_additive_roots_when_chained() -> None:
    base = {
        "schema_version": "openalex-ranking-training-handoff-v1",
        "cumulative_unique_ready_query_ids": ["q1"],
        "ordered_receipt_roots": ["C:/base", "C:/prior-supplement"],
        "high_recall_candidate_supplement": {
            "receipt_roots": ["C:/prior-supplement"],
            "llm_request_count": 0,
            "test_partition_touched": False,
        },
        "online_llm_requests": 0,
        "test_partition_touched": False,
    }

    expanded = MODULE._expanded_handoff(
        base,
        receipt_roots=(
            Path("C:/base"),
            Path("C:/prior-supplement"),
            Path("C:/new-supplement"),
        ),
        validation_result_sha256="sha256:" + "1" * 64,
    )

    assert expanded["high_recall_candidate_supplement"]["receipt_roots"] == [
        str(Path("C:/prior-supplement").resolve()),
        str(Path("C:/new-supplement").resolve()),
    ]


def test_script_can_start_directly() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_selected_batch_indexes_support_disjoint_checkpoint_workers() -> None:
    assert list(MODULE._selected_batch_indexes(10, start=3, end=7)) == [3, 4, 5, 6]
    assert list(MODULE._selected_batch_indexes(10, start=8, end=None)) == [8, 9]


def test_historical_baseline_supports_chained_expanded_coverage() -> None:
    original = {"query_count": 10}

    assert MODULE._historical_baseline({"coverage": original}) is original
    assert MODULE._historical_baseline({"historical_before": original}) is original


def test_supplemented_query_ids_only_include_paths_below_new_roots(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    supplement = tmp_path / "supplement"
    indexed = {
        "q1": (base / "q1.json", supplement / "q1.json"),
        "q2": (base / "q2.json",),
    }

    assert MODULE._supplemented_query_ids(indexed, (supplement,)) == {"q1"}
    assert MODULE._non_additive_paths(
        indexed["q1"], (supplement,)
    ) == (base / "q1.json",)
