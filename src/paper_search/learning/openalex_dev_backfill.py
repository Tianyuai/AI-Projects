"""Offline planning and receipt inventory for auto_dev A-prime backfills.

This module is deliberately separate from the production/training scheduler.  It
only inventories development receipts whose generation provenance identifies the
frozen ``core4-semantic-boolean-v1`` contract and emits immutable, hash-bound
work manifests for a later online collection window.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from paper_search.learning.openalex_daily_schedule import (
    ScheduledQueryActions,
    SearchActionIdentity,
    search_action_identity,
)


SCHEMA_VERSION = "openalex-auto-dev-backfill-v1"
CORE4_POLICY = "core4-semantic-boolean-v1"
CORE4_MISSING_ACTIONS_POLICY = "core4-semantic-boolean-missing-actions-v1"
_TERMINAL_CODES = {"invalid_work"}


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def load_auto_dev_rows(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or row.get("dataset") != "pasa"
            or row.get("split") != "auto_dev"
            or row.get("role") != "development"
        ):
            raise ValueError("backfill requires only pasa auto_dev development rows")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("auto_dev row has an invalid query_id")
        rows.append(row)
    if not rows or len({str(row["query_id"]) for row in rows}) != len(rows):
        raise ValueError("auto_dev partition must be non-empty and query IDs unique")
    return tuple(rows)


def load_manifest_query_ids(path: Path) -> frozenset[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample = payload.get("sample")
    if not isinstance(sample, list):
        raise ValueError("latest batch manifest has no sample")
    ids = {
        str(item["query_id"])
        for item in sample
        if isinstance(item, dict) and isinstance(item.get("query_id"), str)
    }
    if len(ids) != len(sample):
        raise ValueError("latest batch manifest contains invalid or duplicate IDs")
    return frozenset(ids)


def _identity_map(actions: object) -> tuple[dict[str, SearchActionIdentity], bool]:
    if not isinstance(actions, list):
        return {}, False
    result: dict[str, SearchActionIdentity] = {}
    valid = True
    for action in actions:
        if not isinstance(action, Mapping) or not isinstance(action.get("action_id"), str):
            valid = False
            continue
        try:
            identity = search_action_identity(action)
        except (TypeError, ValueError):
            valid = False
            continue
        if identity is None or (
            action["action_id"] in result and result[action["action_id"]] != identity
        ):
            valid = False
            continue
        result[action["action_id"]] = identity
    return result, valid


def _terminal_or_success(errors: object) -> bool:
    if not isinstance(errors, list):
        return False
    if not errors:
        return True
    return all(
        isinstance(item, Mapping)
        and item.get("code") in _TERMINAL_CODES
        and item.get("retryable") is False
        for item in errors
    )


def _is_exact_core4_generation(payload: Mapping[str, object]) -> bool:
    provenance = payload.get("generation_provenance")
    return isinstance(provenance, Mapping) and (
        provenance.get("candidate_policy") in {CORE4_POLICY, CORE4_MISSING_ACTIONS_POLICY}
        and provenance.get("collection_mode")
        in {"core4_semantic_boolean", "scheduled_missing_actions"}
    )


def _paired_retrieval(generation_path: Path) -> Path:
    parts = list(generation_path.parts)
    index = parts.index("generation")
    parts[index] = "retrieval"
    return Path(*parts)


def inventory_core4_receipts(
    *, receipt_roots: Sequence[Path], required: Mapping[str, Sequence[SearchActionIdentity]]
) -> dict[str, Any]:
    """Inventory complete, conflicting, and absent exact-config query receipts."""

    attempts: dict[str, list[dict[str, Any]]] = {}
    files_scanned = 0
    parse_errors = 0
    for root in receipt_roots:
        if not root.is_dir():
            raise ValueError(f"receipt root is unavailable: {root}")
        for generation_path in sorted(root.rglob("*.json")):
            if generation_path.parent.name != "attempt-01" or generation_path.parent.parent.name != "generation":
                continue
            try:
                generation = json.loads(generation_path.read_text(encoding="utf-8"))
                if not isinstance(generation, Mapping) or not _is_exact_core4_generation(
                    generation
                ):
                    continue
                query_id = generation.get("query_id")
                if not isinstance(query_id, str) or query_id not in required:
                    continue
                retrieval_path = _paired_retrieval(generation_path)
                retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
                files_scanned += 2
                if not isinstance(retrieval, Mapping):
                    raise ValueError("retrieval receipt is not an object")
                identities, valid_actions = _identity_map(generation.get("actions"))
                expected = set(required[query_id])
                observed = set(identities.values())
                results = retrieval.get("results")
                result_ids = {
                    item.get("action_id")
                    for item in results
                    if isinstance(item, Mapping)
                } if isinstance(results, list) else set()
                action_ids = set(identities)
                result_complete = isinstance(results, list) and result_ids == action_ids
                settled = result_complete and all(
                    isinstance(item, Mapping)
                    and item.get("infrastructure_failure") is not True
                    and _terminal_or_success(item.get("errors"))
                    for item in results
                )
                complete = (
                    generation.get("attempt_status") == "succeeded"
                    and retrieval.get("attempt_status") == "succeeded"
                    and valid_actions
                    and observed == expected
                    and settled
                )
                attempts.setdefault(query_id, []).append(
                    {
                        "path": str(generation_path),
                        "complete": complete,
                        "action_count": len(observed),
                        "expected_action_count": len(expected),
                        "result_complete": result_complete,
                        "settled": settled,
                        "retrieval_status": retrieval.get("attempt_status"),
                    }
                )
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError):
                parse_errors += 1

    complete_ids = {
        query_id
        for query_id, values in attempts.items()
        if any(bool(value.get("complete")) for value in values)
    }
    conflict_ids = {
        query_id
        for query_id, values in attempts.items()
        if query_id not in complete_ids and values
    }
    required_ids = set(required)
    return {
        "files_scanned": files_scanned,
        "parse_error_count": parse_errors,
        "attempted_query_count": len(attempts),
        "complete_query_ids": sorted(complete_ids),
        "conflict_query_ids": sorted(conflict_ids),
        "missing_query_ids": sorted(required_ids - complete_ids - conflict_ids),
        "attempts_by_query": attempts,
    }


def select_query_ids(
    *,
    required_query_ids: Sequence[str],
    complete_query_ids: Sequence[str],
    conflict_query_ids: Sequence[str],
    target_count: int,
) -> tuple[str, ...]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    complete = set(complete_query_ids)
    conflicts = set(conflict_query_ids)
    selected = sorted(complete)
    missing = [
        query_id
        for query_id in sorted(required_query_ids)
        if query_id not in complete and query_id not in conflicts
    ]
    needed = max(0, target_count - len(selected))
    if len(selected) + min(needed, len(missing)) < target_count:
        raise ValueError("not enough non-conflicting missing queries for target")
    return tuple(selected + missing[:needed])


def build_work(
    query_ids: Sequence[str], required: Mapping[str, Sequence[SearchActionIdentity]]
) -> tuple[ScheduledQueryActions, ...]:
    return tuple(
        ScheduledQueryActions(query_id=query_id, missing_actions=tuple(required[query_id]))
        for query_id in sorted(query_ids)
    )


def assign_work_to_keys(
    work: Sequence[ScheduledQueryActions], *, key_count: int, max_calls: int
) -> dict[int, tuple[ScheduledQueryActions, ...]]:
    if key_count <= 0 or max_calls <= 0:
        raise ValueError("key_count and max_calls must be positive")
    buckets: dict[int, list[ScheduledQueryActions]] = {
        key: [] for key in range(1, key_count + 1)
    }
    used = {key: 0 for key in buckets}
    for item in sorted(work, key=lambda value: value.query_id):
        candidates = [
            key
            for key in buckets
            if used[key] + len(item.missing_actions) <= max_calls
        ]
        if not candidates:
            raise ValueError("work exceeds the configured key capacity")
        key = min(candidates, key=lambda value: (used[value], value))
        buckets[key].append(item)
        used[key] += len(item.missing_actions)
    return {key: tuple(values) for key, values in buckets.items() if values}


def _constraint_categories(query: str) -> set[str]:
    lowered = query.casefold()
    categories: set[str] = set()
    if re.search(r"\b(no|not|without|non[- ]|neither|exclude|excluding|absence)\b", lowered):
        categories.add("negation")
    if re.search(r"\b(19|20)\d{2}\b", lowered):
        categories.add("year")
    if re.search(r"\b(method|algorithm|framework|model|approach|architecture)\b", lowered):
        categories.add("method_name")
    if re.search(r"\b(dataset|benchmark|corpus|imagenet|coco|mnist|pubmed)\b", lowered):
        categories.add("dataset_name")
    if re.search(r"\b(classification|segmentation|detection|retrieval|forecast|recognition|translation)\b", lowered):
        categories.add("task_name")
    if not categories:
        categories.add("other")
    if len(categories) >= 2:
        categories.add("multi_label")
    return categories


def select_train_constraint_queries(
    rows: Sequence[Mapping[str, object]], *, excluded: set[str], limit: int
) -> dict[str, list[str]]:
    if limit < 0:
        raise ValueError("limit must not be negative")
    selected: dict[str, list[str]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        query = str(row["query"])
        if query_id in excluded:
            continue
        for category in sorted(_constraint_categories(query)):
            selected.setdefault(category, []).append(query_id)
    output: dict[str, list[str]] = {}
    remaining = limit
    for category in sorted(selected):
        if remaining <= 0:
            break
        values = selected[category][:remaining]
        output[category] = values
        remaining -= len(values)
    return output


def manifest_digest(payload: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


__all__ = [
    "CORE4_POLICY",
    "SCHEMA_VERSION",
    "assign_work_to_keys",
    "build_work",
    "canonical_bytes",
    "inventory_core4_receipts",
    "load_auto_dev_rows",
    "load_manifest_query_ids",
    "manifest_digest",
    "select_query_ids",
    "select_train_constraint_queries",
    "sha256_file",
]
