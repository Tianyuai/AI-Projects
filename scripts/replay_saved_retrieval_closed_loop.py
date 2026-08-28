"""Replay saved action receipts through production-style RRF and truncation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.evaluation.saved_receipt_replay import (
    aggregate_saved_replays,
    replay_saved_query,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_index(root: Path) -> dict[str, Path]:
    paths = sorted(root.glob("openalex/batch-*/retrieval/attempt-01/*.json"))
    index = {path.stem: path for path in paths}
    if len(index) != len(paths):
        raise ValueError("duplicate saved retrieval receipt query ids")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sample = manifest["sample"]
    partition = {row["query_id"]: row for row in _read_jsonl(args.partition)}
    receipts = _receipt_index(args.receipt_root)
    replays = []
    for selected in sample:
        query_id = str(selected["query_id"])
        row = partition.get(query_id)
        receipt_path = receipts.get(query_id)
        if row is None or receipt_path is None:
            raise ValueError(f"missing frozen input for {query_id}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["attempt_status"] != "succeeded":
            raise ValueError(f"receipt did not succeed for {query_id}")
        action_results = [
            (
                str(result["action_id"]),
                [Paper.model_validate(hit) for hit in result["hits"]],
            )
            for result in receipt["results"]
        ]
        replays.append(
            replay_saved_query(
                query_id=query_id,
                query=str(row["query"]),
                gold_paper_ids=list(row["gold_paper_ids"]),
                action_results=action_results,
                fold=int(selected["fold"]),
            )
        )

    summary = aggregate_saved_replays(replays)
    payload = {
        "schema_version": "saved-retrieval-closed-loop-v1",
        "ranking": {
            "fusion": "action_level_rrf",
            "rrf_k": 60,
            "hard_filter": "rule_fallback_query_spec",
            "cutoffs": [5, 10, 20, 50],
            "optional_stages": [],
        },
        "input_sha256": {
            "manifest": _sha256(args.manifest),
            "partition": _sha256(args.partition),
        },
        "test_partition_touched": False,
        "summary": summary.model_dump(mode="json"),
    }
    content = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    write_frozen_bytes(args.output, content)
    print(json.dumps(payload["summary"]["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
