from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_search.evaluation.dataset import sha256_file
from paper_search.learning.method_sequence_gate import (
    MethodSequenceGate,
    assess_method_sequence,
    build_method_sequence_evidence,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--base-labels", type=Path, required=True)
    parser.add_argument("--semantic-labels", type=Path, required=True)
    parser.add_argument("--graph-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate_payload = json.loads(args.gate.read_text(encoding="utf-8"))
    freeze_payload = json.loads(args.freeze.read_text(encoding="utf-8"))
    if sha256_file(args.freeze) != gate_payload["frozen_manifest_sha256"]:
        raise ValueError("frozen manifest hash does not match the predeclared gate")
    if len(freeze_payload["sample"]) != gate_payload["query_count"]:
        raise ValueError("frozen query count does not match the predeclared gate")
    gate = MethodSequenceGate.model_validate(gate_payload["gate"])
    evidence = build_method_sequence_evidence(
        frozen_rows=freeze_payload["sample"],
        base_labels=_read_jsonl(args.base_labels),
        semantic_labels=_read_jsonl(args.semantic_labels),
        graph_labels=_read_jsonl(args.graph_labels),
    )
    decision = assess_method_sequence(evidence, gate)
    promoted = [
        stage.method for stage in (decision.semantic, decision.graph) if stage.promote
    ]
    payload = {
        "schema_version": "method-sequence-development-result-v1",
        "gate_sha256": sha256_file(args.gate),
        "input_sha256": {
            "freeze": sha256_file(args.freeze),
            "base_labels": sha256_file(args.base_labels),
            "semantic_labels": sha256_file(args.semantic_labels),
            "graph_labels": sha256_file(args.graph_labels),
        },
        "decision": decision.model_dump(mode="json"),
        "training_expansion_methods": promoted,
        "production_policy_changed": False,
        "confirmation_sample_consumed": False,
        "test_partition_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "semantic_promote": decision.semantic.promote,
                "graph_promote": decision.graph.promote,
                "training_expansion_methods": promoted,
                "test_partition_touched": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
