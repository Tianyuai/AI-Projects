"""Freeze a larger receipt-backed auto_dev panel disjoint from consumed canaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.learning.gold_retrievability_audit import (  # noqa: E402
    build_frozen_audit_manifest,
    freeze_audit_manifest,
)
from paper_search.learning.openalex_dev_backfill import (  # noqa: E402
    _constraint_categories,
)


def _query_ids(value: object) -> set[str]:
    if isinstance(value, Mapping):
        found = {
            str(item)
            for key, item in value.items()
            if key == "query_id" and isinstance(item, str)
        }
        for item in value.values():
            found.update(_query_ids(item))
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found.update(_query_ids(item))
        return found
    return set()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def freeze_extended_auto_dev(
    *,
    partition_path: Path,
    selection_manifest_path: Path,
    consumed_manifest_path: Path,
    output_path: Path,
    coverage_path: Path,
    seed: str,
) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in partition_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["query_id"]): row for row in rows}
    if len(by_id) != len(rows) or any(
        row.get("split") != "auto_dev" or row.get("role") != "development"
        for row in rows
    ):
        raise ValueError("extended evaluation requires a unique auto_dev partition")
    selection_payload = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    selected = set(selection_payload.get("selected_query_ids", []))
    consumed = _query_ids(
        json.loads(consumed_manifest_path.read_text(encoding="utf-8"))
    )
    eligible = selected - consumed
    if not eligible or not selected.issubset(by_id):
        raise ValueError("extended evaluation selection is empty or outside auto_dev")
    excluded = frozenset(set(by_id) - eligible)
    manifest = build_frozen_audit_manifest(
        partition_path,
        sample_size=len(eligible),
        seed=seed,
        excluded_query_ids=excluded,
    )
    manifest_hash = freeze_audit_manifest(output_path, manifest)
    category_counts: Counter[str] = Counter()
    for query_id in eligible:
        category_counts.update(_constraint_categories(str(by_id[query_id]["query"])))
    report: dict[str, object] = {
        "schema_version": "extended-auto-dev-coverage-v1",
        "query_count": len(eligible),
        "selected_before_consumed_exclusion": len(selected),
        "consumed_overlap_count": len(selected & consumed),
        "constraint_category_counts": dict(sorted(category_counts.items())),
        "intent_family_counts": dict(
            sorted(Counter(item.intent_family for item in manifest.sample).items())
        ),
        "length_bucket_counts": dict(
            sorted(Counter(item.length_bucket for item in manifest.sample).items())
        ),
        "gold_count_bucket_counts": dict(
            sorted(Counter(item.gold_count_bucket for item in manifest.sample).items())
        ),
        "fold_counts": manifest.fold_counts,
        "manifest_path": str(output_path),
        "manifest_sha256": manifest_hash,
        "online_requests_made": 0,
        "test_partition_touched": False,
    }
    _write_json(coverage_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--consumed-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args(argv)
    report = freeze_extended_auto_dev(
        partition_path=args.partition,
        selection_manifest_path=args.selection_manifest,
        consumed_manifest_path=args.consumed_manifest,
        output_path=args.output,
        coverage_path=args.coverage,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
