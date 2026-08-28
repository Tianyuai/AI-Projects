from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_search.evaluation.dataset import sha256_file
from paper_search.learning.receipt_recombination import (
    analyze_receipt_recombination,
    analyze_structured_graph_marginals,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--lexical-labels", type=Path, required=True)
    parser.add_argument("--semantic-labels", type=Path, required=True)
    parser.add_argument("--graph-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("split") != "auto_train":
        raise ValueError("receipt recombination requires the frozen auto_train split")
    lexical_labels = _read_jsonl(args.lexical_labels)
    semantic_labels = _read_jsonl(args.semantic_labels)
    graph_labels = _read_jsonl(args.graph_labels)
    report = analyze_receipt_recombination(
        frozen_rows=freeze["sample"],
        lexical_labels=lexical_labels,
        semantic_labels=semantic_labels,
    )
    report["structured_graph_sequence"] = analyze_structured_graph_marginals(
        frozen_rows=freeze["sample"],
        lexical_labels=lexical_labels,
        semantic_labels=semantic_labels,
        graph_labels=graph_labels,
    )
    report["input_sha256"] = {
        "freeze": sha256_file(args.freeze),
        "lexical_labels": sha256_file(args.lexical_labels),
        "semantic_labels": sha256_file(args.semantic_labels),
        "graph_labels": sha256_file(args.graph_labels),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "query_count": report["query_count"],
                "maximum_lexical_actions_per_query": report[
                    "maximum_lexical_actions_per_query"
                ],
                "test_partition_touched": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
