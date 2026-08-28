from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.audit_online_recall_ceiling import summarize_records


def test_direct_script_entrypoint_resolves_project_modules(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/audit_online_recall_ceiling.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_summary_separates_recall_misses_from_conditional_ranking_gaps() -> None:
    summary = summarize_records(
        [
            {
                "query_id": "q-method",
                "gold_count": 2,
                "gold_ranks": [3],
                "labels": ["method"],
            },
            {
                "query_id": "q-negation",
                "gold_count": 1,
                "gold_ranks": [],
                "labels": ["negation"],
            },
            {
                "query_id": "q-unconstrained",
                "gold_count": 1,
                "gold_ranks": [25],
                "labels": [],
            },
        ]
    )

    overall = summary["overall"]
    assert overall["query_count"] == 3
    assert overall["candidate_gold_hit_query_count"] == 2
    assert overall["candidate_gold_miss_query_count"] == 1
    assert overall["gold_in_top_5_query_count"] == 1
    assert overall["gold_in_top_20_query_count"] == 1
    assert overall["candidate_hit_but_below_top_20_query_count"] == 1
    assert overall["candidate_micro_recall"] == 0.5
    assert overall["top_20_query_rate_given_candidate_hit"] == 0.5
    assert overall["dominant_bottleneck_at_20"] == "balanced"

    assert summary["by_stratum"]["method"]["gold_in_top_20_query_count"] == 1
    assert (
        summary["by_stratum"]["negation"]["candidate_gold_miss_query_count"]
        == 1
    )
    assert (
        summary["by_stratum"]["negation"]["dominant_bottleneck_at_20"]
        == "recall"
    )
    assert (
        summary["by_stratum"]["unconstrained"][
            "candidate_hit_but_below_top_20_query_count"
        ]
        == 1
    )
    assert (
        summary["by_stratum"]["unconstrained"]["dominant_bottleneck_at_20"]
        == "ranking"
    )


def test_summary_rejects_duplicate_query_ids() -> None:
    row = {
        "query_id": "q1",
        "gold_count": 1,
        "gold_ranks": [],
        "labels": [],
    }

    with pytest.raises(ValueError, match="duplicate query id"):
        summarize_records([row, row])
