from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.cpu_semantic_router_promotion import _load_labels
from paper_search.learning.provider_action_labels import ProviderActionLabel
from paper_search.learning.semantic_router_label_audit import (
    LabelCompressionCriteria,
    audit_binary_label_compression,
    baseline_gold_hit_counts,
)


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _load_provider_labels(path: Path) -> tuple[list[ProviderActionLabel], bytes]:
    content = path.read_bytes()
    rows = [
        ProviderActionLabel.model_validate_json(line)
        for line in content.decode("utf-8").splitlines()
        if line.strip()
    ]
    return rows, content


def _load_metadata(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    metadata: dict[str, dict[str, Any]] = {}
    hashes = []
    for path in paths:
        content = path.read_bytes()
        manifest = json.loads(content)
        hashes.append(_sha256(content))
        for raw in manifest["sample"]:
            query_id = str(raw["query_id"])
            if query_id in metadata:
                raise ValueError(f"duplicate manifest query: {query_id}")
            metadata[query_id] = raw
    return metadata, hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-labels", type=Path, required=True)
    parser.add_argument("--baseline-labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    method_rows, method_content = _load_labels(args.method_labels)
    if any(row.role != "training" for row in method_rows):
        raise ValueError("label compression audit accepts training labels only")
    baseline_rows, baseline_content = _load_provider_labels(args.baseline_labels)
    if any(row.role != "training" for row in baseline_rows):
        raise ValueError("baseline labels must be training-only")
    metadata, manifest_hashes = _load_metadata(args.manifest)
    baseline_counts = baseline_gold_hit_counts(baseline_rows)
    criteria = LabelCompressionCriteria(
        minimum_overall_examples_per_strength=30,
        minimum_examples_per_strength_per_stratum=10,
        minimum_qualifying_strata_per_family=2,
        minimum_qualifying_families=2,
    )
    audit = audit_binary_label_compression(
        method_rows,
        metadata=metadata,
        baseline_gold_hit_counts=baseline_counts,
        criteria=criteria,
    )
    report = {
        "schema_version": "semantic-router-label-compression-audit-v1",
        "scope": "training_only_frozen_strata_no_development_or_test",
        **audit,
        "interpretation": {
            "single_hit": "semantic adds exactly one previously missed Gold paper",
            "multi_hit": "semantic adds at least two previously missed Gold papers",
            "cost_observation": "all paired semantic examples use one API call, so per-query API cost is constant and cannot supervise a learned cost model",
        },
        "inputs": {
            "method_labels_sha256": _sha256(method_content),
            "baseline_labels_sha256": _sha256(baseline_content),
            "manifest_sha256": manifest_hashes,
        },
        "development_labels_read": False,
        "final_test_consumed": False,
    }
    content = _canonical_bytes(report)
    write_frozen_bytes(args.output, content)
    print(json.dumps({**report, "report_sha256": _sha256(content)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
