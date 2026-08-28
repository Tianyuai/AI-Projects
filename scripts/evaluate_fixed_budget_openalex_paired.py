from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper_search.learning.fixed_budget_comparison import (
    evaluate_paired_candidate_pools,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_run(
    root: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, int],
]:
    rows: dict[str, dict[str, Any]] = {}
    actions: dict[str, list[dict[str, Any]]] = {}
    for report_path in sorted(root.rglob("canary-report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for row in report["result"]["per_query"]:
            query_id = str(row["query_id"])
            if query_id in rows:
                raise ValueError(f"duplicate result for {query_id}")
            rows[query_id] = row
        for query_id, value in report["actions_by_query"].items():
            if query_id in actions:
                raise ValueError(f"duplicate action batch for {query_id}")
            actions[query_id] = value if isinstance(value, list) else [value]
    raw_hit_counts: dict[str, int] = {}
    for receipt_path in sorted(root.rglob("retrieval/attempt-01/*.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        query_id = str(receipt["query_id"])
        if query_id in raw_hit_counts:
            raise ValueError(f"duplicate retrieval receipt for {query_id}")
        raw_hit_counts[query_id] = sum(
            len(result["hits"]) for result in receipt["results"]
        )
    return rows, actions, raw_hit_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    selected = {item["query_id"]: int(item["fold"]) for item in manifest["sample"]}
    query_text_by_id = {
        row["query_id"]: row["query"]
        for line in args.partition.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))["query_id"] in selected
    }
    baseline_rows, baseline_actions, baseline_raw = _load_run(args.baseline_root)
    candidate_rows, candidate_actions, candidate_raw = _load_run(args.candidate_root)
    result = evaluate_paired_candidate_pools(
        fold_by_query=selected,
        query_text_by_id=query_text_by_id,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        baseline_actions=baseline_actions,
        candidate_actions=candidate_actions,
        baseline_raw_hit_counts=baseline_raw,
        candidate_raw_hit_counts=candidate_raw,
        promotion_gate=gate["promotion_gate"],
    )
    result["input_sha256"] = {
        "manifest": _sha256(args.manifest),
        "gate": _sha256(args.gate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
