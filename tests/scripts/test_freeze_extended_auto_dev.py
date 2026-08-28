from __future__ import annotations

import json
from pathlib import Path

from scripts.freeze_extended_auto_dev import freeze_extended_auto_dev


def test_freeze_extended_auto_dev_keeps_only_selected_unconsumed_queries(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "dev.jsonl"
    rows = [
        {
            "dataset": "pasa",
            "split": "auto_dev",
            "role": "development",
            "revision": "r1",
            "query_id": f"q{index}",
            "query": query,
            "gold_paper_ids": [f"openalex:W{index}"],
        }
        for index, query in enumerate(
            ("method after 2020", "dataset without images", "translation task", "other")
        )
    ]
    partition.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected_query_ids": ["q0", "q1", "q2"]}))
    consumed = tmp_path / "consumed.json"
    consumed.write_text(json.dumps({"sample": [{"query_id": "q0"}]}))
    output = tmp_path / "frozen.json"
    coverage = tmp_path / "coverage.json"

    report = freeze_extended_auto_dev(
        partition_path=partition,
        selection_manifest_path=selection,
        consumed_manifest_path=consumed,
        output_path=output,
        coverage_path=coverage,
        seed="test",
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert {row["query_id"] for row in manifest["sample"]} == {"q1", "q2"}
    assert report["query_count"] == 2
    assert report["constraint_category_counts"]["dataset_name"] == 1
    assert report["constraint_category_counts"]["negation"] == 1
    assert report["constraint_category_counts"]["task_name"] == 1
    assert report["test_partition_touched"] is False
