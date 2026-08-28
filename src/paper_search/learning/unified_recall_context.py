"""Hash-bound loading of the unified local context for blind recall actions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from paper_search.domain.models import QuerySpec
from paper_search.learning.query_constraint_annotations import query_sha256
from paper_search.query.parser import rule_fallback


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"context row is not a mapping: {path}")
        rows.append(cast(dict[str, object], raw))
    return rows


def _by_query(
    rows: list[dict[str, object]], *, label: str
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{label} query identity is invalid")
        if query_id in result:
            raise ValueError(f"duplicate {label} query identity: {query_id}")
        result[query_id] = row
    return result


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _ordered_unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        folded = normalized.casefold()
        if not normalized or folded in seen:
            continue
        seen.add(folded)
        result.append(normalized)
    return result


def load_frozen_recall_query_specs(
    *,
    partition_path: Path,
    manifest_path: Path,
) -> dict[str, QuerySpec]:
    """Load one complete auto_train context without consulting Gold documents."""

    partition_bytes = partition_path.read_bytes()
    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_raw, dict):
        raise ValueError("unified context manifest is invalid")
    manifest = cast(dict[str, object], manifest_raw)
    if (
        manifest.get("role") != "training"
        or manifest.get("split") != "auto_train"
        or manifest.get("test_partition_touched") is not False
    ):
        raise ValueError("unified recall context is not isolated auto_train data")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("unified context hash bindings are missing")
    if inputs.get("partition_sha256") != _sha256(partition_bytes):
        raise ValueError("partition hash mismatch")

    context_root = manifest_path.parent
    task_path = context_root / "task-labels.jsonl"
    constraint_path = context_root / "constraint-labels.jsonl"
    task_bytes = task_path.read_bytes()
    constraint_bytes = constraint_path.read_bytes()
    if outputs.get("task_labels_sha256") != _sha256(task_bytes):
        raise ValueError("task label hash mismatch")
    if outputs.get("constraint_labels_sha256") != _sha256(constraint_bytes):
        raise ValueError("constraint label hash mismatch")

    partition_rows = _jsonl(partition_path)
    task_by_id = _by_query(_jsonl(task_path), label="task context")
    constraint_by_id = _by_query(
        _jsonl(constraint_path), label="constraint context"
    )
    partition_by_id = _by_query(partition_rows, label="partition")
    expected_count = manifest.get("query_count")
    if (
        type(expected_count) is not int
        or expected_count != len(task_by_id)
        or expected_count != len(constraint_by_id)
    ):
        raise ValueError("unified context query count mismatch")
    if set(task_by_id) != set(constraint_by_id):
        raise ValueError("constraint context coverage mismatch")
    if not set(task_by_id).issubset(partition_by_id):
        raise ValueError("unified context is not a subset of the training partition")

    specs: dict[str, QuerySpec] = {}
    for query_id in sorted(task_by_id):
        partition_row = partition_by_id[query_id]
        if (
            partition_row.get("role") != "training"
            or partition_row.get("split") != "auto_train"
        ):
            raise ValueError("partition contains non-training data")
        query = partition_row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"partition query is invalid: {query_id}")
        digest = query_sha256(query)
        task = task_by_id[query_id]
        constraint = constraint_by_id[query_id]
        if task.get("query_sha256") != digest:
            raise ValueError(f"task context query hash mismatch: {query_id}")
        if constraint.get("query_sha256") != digest:
            raise ValueError(f"constraint context query hash mismatch: {query_id}")
        base = rule_fallback(query)
        year_from = constraint.get("year_from")
        year_to = constraint.get("year_to")
        specs[query_id] = base.model_copy(
            update={
                "tasks": _ordered_unique(
                    [*_strings(task.get("tasks")), *_strings(constraint.get("tasks"))]
                ),
                "methods": _ordered_unique(_strings(constraint.get("methods"))),
                "datasets": _ordered_unique(_strings(constraint.get("datasets"))),
                "exclusions": _ordered_unique(
                    [
                        *_strings(constraint.get("exclusions")),
                        *base.exclusions,
                    ]
                ),
                "year_from": year_from if isinstance(year_from, int) else None,
                "year_to": year_to if isinstance(year_to, int) else None,
            }
        )
    return specs


__all__ = ["load_frozen_recall_query_specs"]
