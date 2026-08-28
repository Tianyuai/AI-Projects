from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper_search.learning.unified_recall_context import load_frozen_recall_query_specs


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    query = "Find ImageNet-C classification using ViT without fine tuning"
    digest = _sha256(query.encode())
    partition = _jsonl(
        [
            {
                "dataset": "pasa",
                "split": "auto_train",
                "role": "training",
                "query_id": "q-1",
                "query": query,
                "gold_paper_ids": ["arxiv:2001.00001"],
            },
            {
                "dataset": "pasa",
                "split": "auto_train",
                "role": "training",
                "query_id": "q-not-strict-ready",
                "query": "Excluded from the strict-ready context",
                "gold_paper_ids": ["arxiv:2001.00002"],
            },
        ]
    )
    task = _jsonl(
        [
            {
                "query_id": "q-1",
                "query_sha256": digest,
                "role": "training",
                "split": "auto_train",
                "tasks": ["image classification"],
                "status": "accepted",
            }
        ]
    )
    constraint = _jsonl(
        [
            {
                "query_id": "q-1",
                "query_sha256": digest,
                "role": "training",
                "split": "auto_train",
                "methods": ["ViT"],
                "datasets": ["ImageNet-C"],
                "exclusions": ["fine tuning"],
                "year_from": None,
                "year_to": None,
                "status": "accepted",
            }
        ]
    )
    partition_path = tmp_path / "partition.jsonl"
    context = tmp_path / "unified-context"
    context.mkdir()
    partition_path.write_bytes(partition)
    (context / "task-labels.jsonl").write_bytes(task)
    (context / "constraint-labels.jsonl").write_bytes(constraint)
    (context / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "pasa-priority-unified-training-context-v1",
                "query_count": 1,
                "role": "training",
                "split": "auto_train",
                "test_partition_touched": False,
                "inputs": {"partition_sha256": _sha256(partition)},
                "outputs": {
                    "task_labels_sha256": _sha256(task),
                    "constraint_labels_sha256": _sha256(constraint),
                },
            }
        ),
        encoding="utf-8",
    )
    return partition_path, context / "manifest.json"


def test_load_frozen_recall_query_specs_binds_partition_and_context(tmp_path: Path) -> None:
    partition, manifest = _fixture(tmp_path)

    loaded = load_frozen_recall_query_specs(
        partition_path=partition,
        manifest_path=manifest,
    )

    spec = loaded["q-1"]
    assert spec.tasks == ["image classification"]
    assert spec.methods == ["ViT"]
    assert spec.datasets == ["ImageNet-C"]
    assert spec.exclusions == ["fine tuning"]


def test_load_frozen_recall_query_specs_fails_on_context_hash_change(
    tmp_path: Path,
) -> None:
    partition, manifest = _fixture(tmp_path)
    (manifest.parent / "task-labels.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="task label hash mismatch"):
        load_frozen_recall_query_specs(
            partition_path=partition,
            manifest_path=manifest,
        )
