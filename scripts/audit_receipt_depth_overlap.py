"""Run a training-only depth/overlap ablation over frozen retrieval receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.receipt_depth_ablation import analyze_receipt_depths


_A_PRIME_ACTIONS = (
    "ceiling-candidate-anchor",
    "ceiling-candidate-text-1",
    "ceiling-candidate-text-2",
    "ceiling-candidate-text-3",
    "semantic-backfill-original",
    "ceiling-candidate-boolean-relaxed",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _successful_receipts(root: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    paths = sorted(root.rglob("retrieval/attempt-01/*.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["attempt_status"] == "succeeded":
            selected[str(payload["query_id"])] = payload
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--semantic-receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("split") != "auto_train":
        raise ValueError("depth ablation requires frozen auto_train queries")
    partition = {row["query_id"]: row for row in _read_jsonl(args.partition)}
    receipts = _successful_receipts(args.receipt_root)
    semantic_receipts = _successful_receipts(args.semantic_receipt_root)
    queries = []
    for selected in freeze["sample"]:
        query_id = str(selected["query_id"])
        receipt = receipts.get(query_id)
        semantic_receipt = semantic_receipts.get(query_id)
        row = partition.get(query_id)
        if receipt is None or semantic_receipt is None or row is None:
            raise ValueError(f"missing frozen training input for {query_id}")
        by_id = {str(result["action_id"]): result for result in receipt["results"]}
        semantic_by_id = {
            str(result["action_id"]): result for result in semantic_receipt["results"]
        }
        if "semantic-backfill-original" not in semantic_by_id:
            raise ValueError(f"missing semantic-original for {query_id}")
        actions = [
            (
                action_id,
                [
                    Paper.model_validate(hit)
                    for hit in (
                        semantic_by_id[action_id]["hits"]
                        if action_id == "semantic-backfill-original"
                        else by_id.get(action_id, {"hits": []})["hits"]
                    )
                ],
            )
            for action_id in _A_PRIME_ACTIONS
        ]
        if not actions or actions[0][0] != "ceiling-candidate-anchor":
            raise ValueError(f"missing lexical anchor for {query_id}")
        queries.append(
            {
                "query_id": query_id,
                "fold": int(selected["fold"]),
                "gold_paper_ids": row["gold_paper_ids"],
                "actions": actions,
            }
        )

    report = analyze_receipt_depths(queries=queries)
    report["composition"] = list(_A_PRIME_ACTIONS)
    report["input_sha256"] = {
        "freeze": _sha256(args.freeze),
        "partition": _sha256(args.partition),
    }
    content = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    write_frozen_bytes(args.output, content)
    print(json.dumps({"depths": report["depths"], "action_positions": report["action_positions"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
