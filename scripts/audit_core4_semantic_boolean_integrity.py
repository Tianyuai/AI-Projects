from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.fixed_budget_integrity import (
    audit_core4_semantic_boolean_queries,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if gate["sample_manifest_sha256"] != _sha256(args.manifest):
        raise ValueError("A-prime gate does not match the frozen manifest")
    if gate["source_partition_sha256"] != _sha256(args.partition):
        raise ValueError("A-prime gate does not match the frozen partition")
    selected = {str(row["query_id"]) for row in manifest["sample"]}
    rows = [
        json.loads(line)
        for line in args.partition.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line)["query_id"] in selected
    ]
    result = audit_core4_semantic_boolean_queries(rows)
    result["fold_counts"] = manifest["fold_counts"]
    result["input_sha256"] = {
        "manifest": _sha256(args.manifest),
        "partition": _sha256(args.partition),
        "gate": _sha256(args.gate),
    }
    content = (
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    write_frozen_bytes(args.output, content)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
