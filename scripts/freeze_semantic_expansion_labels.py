from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.gold_retrievability_audit import FrozenAuditManifest
from paper_search.learning.provider_action_dataset import (
    freeze_provider_action_labels,
    load_provider_action_labels_from_canary_runs,
)
from paper_search.learning.provider_action_labels import ProviderActionLabel


def _manifest_ids(path: Path) -> set[str]:
    manifest = FrozenAuditManifest.model_validate_json(path.read_text(encoding="utf-8"))
    return {row.query_id for row in manifest.sample}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    label_sets = [
        load_provider_action_labels_from_canary_runs(
            partition_path=args.partition,
            provider_run_roots={"openalex": root / "openalex"},
        )
        for root in args.run_root
    ]
    labels = [label for label_set in label_sets for label in label_set]
    expected_ids = _manifest_ids(args.manifest)
    actual_ids = {label.query_id for label in labels}
    if len(labels) != len(expected_ids):
        raise ValueError(
            f"expected one semantic label per query: labels={len(labels)} "
            f"queries={len(expected_ids)}"
        )
    if actual_ids != expected_ids:
        raise ValueError(
            f"label/manifest mismatch: missing={len(expected_ids - actual_ids)} "
            f"unexpected={len(actual_ids - expected_ids)}"
        )
    excluded_ids = set().union(
        *(_manifest_ids(path) for path in args.exclude_manifest)
    ) if args.exclude_manifest else set()
    overlap = actual_ids.intersection(excluded_ids)
    if overlap:
        raise ValueError(f"semantic expansion overlaps excluded manifests: {len(overlap)}")
    for label in labels:
        if not isinstance(label, ProviderActionLabel):
            raise TypeError("invalid provider action label")
        if label.role != "training" or label.provider != "openalex":
            raise ValueError(f"unexpected label scope: {label.query_id}")
        if label.action.search_mode != "semantic":
            raise ValueError(f"non-semantic action in expansion: {label.query_id}")
        if label.retrieval_status != "available":
            raise ValueError(f"unavailable frozen receipt: {label.query_id}")

    label_sha256 = freeze_provider_action_labels(labels, args.output)
    total_gold = sum(label.gold_association_count or 0 for label in labels)
    total_hits = sum(label.gold_hit_count or 0 for label in labels)
    macro_recall = sum(label.action_recall or 0.0 for label in labels) / len(labels)
    positive_queries = sum((label.gold_hit_count or 0) > 0 for label in labels)
    report = {
        "schema_version": "semantic-expansion-label-freeze-v1",
        "dataset": "pasa_auto_train",
        "label_semantics": "direct_semantic_effectiveness_not_method_marginal_gain",
        "query_count": len(actual_ids),
        "label_count": len(labels),
        "positive_query_count": positive_queries,
        "positive_query_rate": positive_queries / len(labels),
        "gold_association_count": total_gold,
        "gold_hit_count": total_hits,
        "macro_candidate_recall": macro_recall,
        "micro_candidate_recall": total_hits / total_gold,
        "provider": "openalex",
        "search_mode": "semantic",
        "actions_per_query": 1,
        "run_root_count": len(args.run_root),
        "excluded_manifest_count": len(args.exclude_manifest),
        "excluded_overlap_count": 0,
        "label_sha256": label_sha256,
        "confirmation_consumed": False,
        "final_test_consumed": False,
        "router_training_ready": False,
        "router_training_blocker": (
            "paired lexical baseline receipts are required to calculate "
            "semantic marginal-gain labels"
        ),
    }
    report_bytes = _canonical_bytes(report) + b"\n"
    write_frozen_bytes(args.report, report_bytes)
    report_sha256 = "sha256:" + hashlib.sha256(report_bytes).hexdigest()
    print(json.dumps({**report, "report_sha256": report_sha256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
