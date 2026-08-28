from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from scripts.evaluate_large_scale_fusion_oof import (
    _build_incremental_oof_comparison,
    _checkpoint_identity,
    _incremental_oof_split,
)


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "evaluate_large_scale_fusion_oof.py"
)


def test_cli_exposes_exact_incremental_warm_start_configuration() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--incremental-new-only" in completed.stdout
    assert "--production-replay-shard-dir" in completed.stdout
    assert "--production-replay-every-batches" in completed.stdout
    assert "--reliability-pair-budget" in completed.stdout
    assert "--task-provenance-pair-budget" in completed.stdout
    assert "--entity-pair-budget" in completed.stdout
    assert "--hard-constraint-pair-budget" in completed.stdout
    assert "--anchor-family" in completed.stdout
    assert "--folds" in completed.stdout


def test_incremental_oof_holds_out_only_new_queries() -> None:
    base_ids = {"base-1", "base-2", "base-3"}
    new_ids = {"new-1", "new-2", "new-3", "new-4", "new-5", "new-6"}
    package_ids = sorted(base_ids | new_ids)
    fold = 1

    split = _incremental_oof_split(
        package_query_ids=package_ids,
        base_query_ids=base_ids,
        fold=fold,
    )

    expected_holdout = {
        query_id
        for query_id in new_ids
        if split.fold_by_query_id[query_id] == fold
    }
    assert split.incremental_query_ids == new_ids
    assert split.held_out_query_ids == expected_holdout
    assert split.training_query_ids == (base_ids | (new_ids - expected_holdout))
    assert base_ids <= split.training_query_ids
    assert not (base_ids & split.held_out_query_ids)


def test_incremental_oof_rejects_base_ids_outside_expanded_package() -> None:
    with pytest.raises(ValueError, match="base query IDs"):
        _incremental_oof_split(
            package_query_ids=["base-1", "new-1"],
            base_query_ids={"base-1", "missing-base"},
            fold=1,
        )


def test_checkpoint_identity_binds_holdout_and_anchor_semantics() -> None:
    first = _checkpoint_identity(
        "sha256:input",
        fold=1,
        epochs=1,
        held_out_query_ids={"new-1", "new-2"},
        anchored_families={"task_provenance"},
    )
    same = _checkpoint_identity(
        "sha256:input",
        fold=1,
        epochs=1,
        held_out_query_ids={"new-2", "new-1"},
        anchored_families={"task_provenance"},
    )
    changed_holdout = _checkpoint_identity(
        "sha256:input",
        fold=1,
        epochs=1,
        held_out_query_ids={"new-1"},
        anchored_families={"task_provenance"},
    )
    changed_anchor = _checkpoint_identity(
        "sha256:input",
        fold=1,
        epochs=1,
        held_out_query_ids={"new-1", "new-2"},
        anchored_families=set(),
    )

    assert first == same
    assert first != changed_holdout
    assert first != changed_anchor


def test_incremental_report_compares_candidate_to_unchanged_production_f5() -> None:
    rows = [
        {
            "query_id": f"q{fold}",
            "fold": fold,
            "gold_count": 1,
            "gold_ranks": {
                "B0": [6],
                "F4": [4],
                "F5": [2],
                "PRODUCTION_F4": [5],
                "PRODUCTION_F5": [3],
            },
        }
        for fold in (1, 2, 3)
    ]

    report = _build_incremental_oof_comparison(rows, incremental_query_count=3)

    assert report["metrics"]["PRODUCTION_F5"]["recall_at_5"] == 1.0
    comparison = report["candidate_vs_production"]["F5_vs_PRODUCTION_F5"]
    assert comparison["overall"]["mrr"]["delta"] > 0
    assert comparison["folds"]["1"]["mrr"]["delta"] > 0
