"""Freeze a blinded PaSa training/development task-validity review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.task_validity_audit import (
    build_blind_review_packet,
    build_task_validity_cases,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _build_evidence_rows(
    rows: list[dict[str, Any]],
    *,
    hit_query_ids: set[str],
    role: str,
) -> list[dict[str, object]]:
    return [
        {
            "query_id": str(row["query_id"]),
            "role": role,
            "cohort": "hit" if str(row["query_id"]) in hit_query_ids else "miss",
            "fold": int(row["fold"]),
        }
        for row in rows
    ]


def _action_hit_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row["query_id"])
        for row in rows
        if int(row.get("gold_hit_count", 0)) > 0
    }


def _graph_hit_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row["query_id"])
        for row in rows
        if row.get("graph_gold_hit_ids") or row.get("pre_graph_gold_hit_ids")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--raw-dev", type=Path, required=True)
    parser.add_argument("--train-freeze", type=Path, required=True)
    parser.add_argument("--availability-cache", type=Path, required=True)
    parser.add_argument("--train-base-labels", type=Path, required=True)
    parser.add_argument("--train-semantic-labels", type=Path, required=True)
    parser.add_argument("--train-graph-labels", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--dev-graph-labels", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--key-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--seed", default="pasa-task-validity-blind90-v1")
    args = parser.parse_args()

    train_freeze = json.loads(args.train_freeze.read_text(encoding="utf-8"))
    dev_manifest = json.loads(args.dev_manifest.read_text(encoding="utf-8"))
    train_base = _read_jsonl(args.train_base_labels)
    train_semantic = _read_jsonl(args.train_semantic_labels)
    train_graph = _read_jsonl(args.train_graph_labels)
    dev_graph = _read_jsonl(args.dev_graph_labels)
    train_hits = (
        _action_hit_ids(train_base)
        | _action_hit_ids(train_semantic)
        | _graph_hit_ids(train_graph)
    )
    dev_hits = _graph_hit_ids(dev_graph)
    train_evidence = _build_evidence_rows(
        train_freeze["sample"], hit_query_ids=train_hits, role="training"
    )
    dev_evidence = _build_evidence_rows(
        dev_manifest["sample"], hit_query_ids=dev_hits, role="development"
    )
    raw_train = {row["qid"]: row for row in _read_jsonl(args.raw_train)}
    raw_dev = {row["qid"]: row for row in _read_jsonl(args.raw_dev)}
    availability_payload = json.loads(
        args.availability_cache.read_text(encoding="utf-8")
    )
    availability = {
        str(row["gold_id"]): str(row["status"])
        for row in availability_payload["records"]
    }
    cases = [
        *build_task_validity_cases(
            raw_by_id=raw_train,
            evidence_rows=train_evidence,
            availability_by_gold_id=availability,
        ),
        *build_task_validity_cases(
            raw_by_id=raw_dev,
            evidence_rows=dev_evidence,
            availability_by_gold_id=availability,
        ),
    ]
    packet = build_blind_review_packet(
        cases,
        targets={
            "training": {"miss": 30, "hit": 30},
            "development": {"miss": 15, "hit": 15},
        },
        seed=args.seed,
    )
    review = {
        key: value for key, value in packet.items() if key != "private_key"
    }
    private_key = {
        "schema_version": "pasa-task-validity-private-key-v1",
        "seed": args.seed,
        "cases": packet["private_key"],
        "test_partition_touched": False,
    }
    selected_by_id = {
        str(row["query_id"]): row
        for row in cases
        if str(row["query_id"])
        in {str(key["query_id"]) for key in packet["private_key"]}
    }
    summary = {
        "schema_version": "pasa-task-validity-audit-freeze-v1",
        "query_count": len(packet["review_cases"]),
        "selection_counts": packet["selection_counts"],
        "zero_title_overlap_count": sum(
            float(row["best_gold_title_token_recall"]) == 0
            for row in selected_by_id.values()
        ),
        "all_gold_availability_audited_count": sum(
            all(status != "not_audited" for status in row["gold_availability"])
            for row in selected_by_id.values()
        ),
        "review_outcome_hidden": True,
        "production_policy_changed": False,
        "test_partition_touched": False,
        "input_sha256": {
            name: _sha256(path)
            for name, path in {
                "raw_train": args.raw_train,
                "raw_dev": args.raw_dev,
                "train_freeze": args.train_freeze,
                "availability_cache": args.availability_cache,
                "train_base_labels": args.train_base_labels,
                "train_semantic_labels": args.train_semantic_labels,
                "train_graph_labels": args.train_graph_labels,
                "dev_manifest": args.dev_manifest,
                "dev_graph_labels": args.dev_graph_labels,
            }.items()
        },
    }
    write_frozen_bytes(args.review_output, _json_bytes(review))
    write_frozen_bytes(args.key_output, _json_bytes(private_key))
    write_frozen_bytes(args.summary_output, _json_bytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
