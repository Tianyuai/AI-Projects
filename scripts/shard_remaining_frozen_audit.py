from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_search.learning.gold_retrievability_audit import (
    FrozenAuditManifest,
    freeze_audit_manifest,
    shard_frozen_audit_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--completed-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()

    manifest = FrozenAuditManifest.model_validate_json(
        args.manifest.read_text(encoding="utf-8")
    )
    completed: set[str] = set()
    for path in args.completed_run_root.rglob("canary-report.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        completed.update(
            str(row["query_id"]) for row in report["result"]["per_query"]
        )
    shards = shard_frozen_audit_manifest(
        manifest,
        skip_query_ids=completed,
        shard_count=args.shard_count,
    )
    for index, shard in enumerate(shards, start=1):
        path = args.output_dir / f"remaining-shard-{index:02d}.json"
        digest = freeze_audit_manifest(path, shard)
        print(
            f"shard={index} queries={shard.sample_query_count} sha256={digest}"
        )
    print(f"completed_query_count={len(completed)}")
    print(f"remaining_query_count={sum(shard.sample_query_count for shard in shards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
