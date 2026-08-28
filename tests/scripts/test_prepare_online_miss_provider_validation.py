from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_online_miss_provider_validation import (
    audit_authorizes_recall,
    collect_prior_query_ids,
    primary_stratum,
    select_stratified,
)


def test_primary_stratum_uses_diagnostic_priority() -> None:
    assert primary_stratum(["method", "negation"]) == "negation"
    assert primary_stratum(["dataset", "method"]) == "method"
    assert primary_stratum([]) == "unconstrained"


def test_audit_authorizes_recall_reads_nested_summary() -> None:
    assert audit_authorizes_recall(
        {
            "decision": {
                "recommended_next_branch": "openalex-s2-supplemental-recall"
            },
            "summary": {"overall": {"dominant_bottleneck_at_20": "recall"}},
        }
    )


def test_select_stratified_enforces_exact_non_overlapping_quotas() -> None:
    rows = [
        *(
            {"query_id": f"method-{index}", "stratum": "method"}
            for index in range(4)
        ),
        *(
            {"query_id": f"negation-{index}", "stratum": "negation"}
            for index in range(4)
        ),
        *(
            {"query_id": f"unc-{index}", "stratum": "unconstrained"}
            for index in range(8)
        ),
    ]

    selected = select_stratified(
        rows,
        quotas={"method": 2, "negation": 3, "unconstrained": 4},
    )

    assert len(selected) == 9
    assert len({row["query_id"] for row in selected}) == 9
    assert sum(row["stratum"] == "method" for row in selected) == 2
    assert sum(row["stratum"] == "negation" for row in selected) == 3
    assert sum(row["stratum"] == "unconstrained" for row in selected) == 4


def test_collect_prior_query_ids_reads_frozen_and_executed_artifacts(
    tmp_path: Path,
) -> None:
    recall_root = tmp_path / "recall_policy"
    package = recall_root / "prior"
    package.mkdir(parents=True)
    (package / "partition.jsonl").write_text(
        json.dumps({"query_id": "q-partition"}) + "\n", encoding="utf-8"
    )
    (package / "query-selection.json").write_text(
        json.dumps({"query_ids": ["q-selection"]}), encoding="utf-8"
    )
    generation = package / "receipts" / "generation" / "attempt-01"
    generation.mkdir(parents=True)
    (generation / "AutoScholarQuery_train_42.json").write_text(
        "{}", encoding="utf-8"
    )

    assert collect_prior_query_ids(recall_root) == {
        "q-partition",
        "q-selection",
        "AutoScholarQuery_train_42",
    }
