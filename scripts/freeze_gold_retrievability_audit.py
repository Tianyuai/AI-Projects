"""Freeze a deterministic, stratified PaSa auto_train audit sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_search.learning.gold_retrievability_audit import (
    build_frozen_audit_manifest,
    freeze_audit_manifest,
)


def _query_ids_from_value(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = {
            str(item)
            for key, item in value.items()
            if key == "query_id" and isinstance(item, str)
        }
        for item in value.values():
            found.update(_query_ids_from_value(item))
        return found
    if isinstance(value, list):
        list_found: set[str] = set()
        for item in value:
            list_found.update(_query_ids_from_value(item))
        return list_found
    return set()


def _query_ids_from_path(path: Path) -> set[str]:
    if path.suffix == ".jsonl":
        found: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                found.update(_query_ids_from_value(json.loads(line)))
        return found
    return _query_ids_from_value(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=385)
    parser.add_argument("--seed", default="pasa-gold-retrievability-v1")
    parser.add_argument("--exclude-query-id", action="append", default=[])
    parser.add_argument(
        "--exclude-query-ids-from",
        action="append",
        type=Path,
        default=[],
    )
    args = parser.parse_args()

    excluded_query_ids = set(args.exclude_query_id)
    for path in args.exclude_query_ids_from:
        excluded_query_ids.update(_query_ids_from_path(path))

    manifest = build_frozen_audit_manifest(
        args.partition,
        sample_size=args.sample_size,
        seed=args.seed,
        excluded_query_ids=frozenset(excluded_query_ids),
    )
    manifest_sha256 = freeze_audit_manifest(args.output, manifest)
    print(f"population_query_count={manifest.population_query_count}")
    print(f"sample_query_count={manifest.sample_query_count}")
    print(f"excluded_query_count={manifest.excluded_query_count}")
    print(f"fold_counts={manifest.fold_counts}")
    print(f"source_sha256={manifest.source_sha256}")
    print(f"manifest_sha256={manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
