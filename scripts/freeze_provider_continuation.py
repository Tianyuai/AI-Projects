"""Freeze only provider actions that have no proven successful retrieval receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.recall_experiments.contracts import (  # noqa: E402
    RecallActionBatch,
    assert_no_forbidden_identifier_keys_or_patterns,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _jsonl(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def successful_query_ids(run_root: Path) -> set[str]:
    """Return queries with at least one receipt-proven successful action."""

    succeeded: set[str] = set()
    for path in sorted(run_root.glob("**/retrieval/attempt-*/*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict) or raw.get("attempt_status") != "succeeded":
            continue
        query_id = raw.get("query_id")
        results = raw.get("results")
        if not isinstance(query_id, str) or not isinstance(results, list):
            continue
        if any(
            isinstance(result, dict)
            and result.get("infrastructure_failure") is not True
            and result.get("errors") == []
            for result in results
        ):
            succeeded.add(query_id)
    return succeeded


def remaining_query_ids(
    source_query_ids: list[str],
    *,
    succeeded: set[str],
    exhausted: set[str],
) -> list[str]:
    excluded = succeeded | exhausted
    return [query_id for query_id in source_query_ids if query_id not in excluded]


def _load_partition(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("source partition row is invalid")
        query_id = row.get("query_id")
        if (
            not isinstance(query_id, str)
            or query_id in seen
            or row.get("role") != "training"
            or row.get("split") != "auto_train"
        ):
            raise ValueError("source partition identity or isolation is invalid")
        seen.add(query_id)
        rows.append(row)
    if not rows:
        raise ValueError("source partition is empty")
    return rows


def _load_actions(path: Path, query_ids: set[str]) -> dict[str, dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != query_ids:
        raise ValueError("source action coverage does not match partition")
    actions: dict[str, dict[str, object]] = {}
    for query_id, value in raw.items():
        batch = RecallActionBatch.model_validate(value)
        dumped = batch.model_dump(mode="json")
        assert_no_forbidden_identifier_keys_or_patterns(dumped)
        actions[str(query_id)] = dumped
    return actions


def freeze_continuation(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.workspace_root).resolve()
    resolve = lambda value: (root / value).resolve()  # noqa: E731
    partition_path = resolve(args.partition)
    actions_path = resolve(args.actions)
    run_root = resolve(args.run_root)
    output = resolve(args.output)
    rows = _load_partition(partition_path)
    query_ids = {str(row["query_id"]) for row in rows}
    actions = _load_actions(actions_path, query_ids)
    succeeded = successful_query_ids(run_root)
    exhausted = set(args.exhausted_query_id)
    unexpected = (succeeded | exhausted).difference(query_ids)
    if unexpected:
        raise ValueError("provider continuation contains unexpected query identities")
    if succeeded.intersection(exhausted):
        raise ValueError("a successful provider query cannot be marked exhausted")
    source_ids = [str(row["query_id"]) for row in rows]
    remaining_ids = remaining_query_ids(
        source_ids,
        succeeded=succeeded,
        exhausted=exhausted,
    )
    remaining_id_set = set(remaining_ids)
    remaining_rows = [
        row for row in rows if str(row["query_id"]) in remaining_id_set
    ]
    if not remaining_rows:
        raise ValueError("provider continuation has no remaining queries")
    remaining_actions = {query_id: actions[query_id] for query_id in remaining_ids}
    partition_bytes = _jsonl(remaining_rows)
    actions_bytes = _canonical_bytes(remaining_actions)
    manifest = {
        "schema_version": "provider-recall-continuation-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "provider": args.provider,
        "source_query_count": len(rows),
        "already_succeeded_query_count": len(succeeded),
        "exhausted_infrastructure_query_ids": sorted(exhausted),
        "remaining_query_count": len(remaining_rows),
        "selection": "receipt-proven-success-and-exhausted-exclusion-v1",
        "recommended_chunk_size": 1,
        "source_hashes": {
            "partition_sha256": _sha256(partition_path.read_bytes()),
            "actions_sha256": _sha256(actions_path.read_bytes()),
        },
        "output_hashes": {
            "partition_sha256": _sha256(partition_bytes),
            "actions_sha256": _sha256(actions_bytes),
        },
        "safety": {
            "actions_regenerated": False,
            "completed_action_identities_repeated": False,
            "exhausted_action_repeated": False,
            "llm_requests_made": 0,
            "online_requests_made_during_freeze": 0,
            "final_test_touched": False,
        },
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _write_immutable(output / "partition.jsonl", partition_bytes)
    _write_immutable(output / "actions.json", actions_bytes)
    _write_immutable(output / "manifest.json", manifest_bytes)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="semantic_scholar")
    parser.add_argument("--exhausted-query-id", action="append", default=[])
    return parser


def main() -> None:
    manifest = freeze_continuation(build_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
