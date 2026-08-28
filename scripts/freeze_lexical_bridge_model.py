from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper_search.learning.lexical_bridge import (
    LexicalBridgeExample,
    SupervisedLexicalBridge,
)
from paper_search.learning.lexical_bridge_deployment import (
    freeze_lexical_bridge_model,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--train-partition", type=Path, required=True)
    parser.add_argument("--training-oof", type=Path, required=True)
    parser.add_argument("--independent-dev", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    training_oof = json.loads(args.training_oof.read_text(encoding="utf-8"))
    if (
        training_oof.get("selected_variant") != "word_char_neighbor_idf_k6"
        or training_oof.get("selected_variant_passed") is not True
        or training_oof.get("test_partition_touched") is not False
    ):
        raise ValueError("training OOF does not authorize the promoted bridge")
    independent_dev = json.loads(args.independent_dev.read_text(encoding="utf-8"))
    if (
        independent_dev.get("bridge_configuration")
        != {
            "representation": "word_char",
            "learning_objective": "neighbor_idf",
            "max_expansion_terms": 6,
        }
        or independent_dev.get("pre_online_gate", {}).get("passed") is not True
        or independent_dev.get("test_partition_touched") is not False
    ):
        raise ValueError("independent development result does not authorize promotion")

    allowed_ids = {row["query_id"] for row in _jsonl(args.train_partition)}
    training = [
        row for row in _jsonl(args.raw_train) if row.get("qid") in allowed_ids
    ]
    if len(training) != len(allowed_ids) or {row["qid"] for row in training} != allowed_ids:
        raise ValueError("raw training rows do not match the frozen partition")
    bridge = SupervisedLexicalBridge.fit(
        [
            LexicalBridgeExample(
                query=str(row["question"]),
                gold_titles=tuple(str(title) for title in row["answer"]),
            )
            for row in training
        ],
        representation="word_char",
        learning_objective="neighbor_idf",
    )
    manifest = freeze_lexical_bridge_model(
        bridge,
        model_path=args.model_output,
        manifest_path=args.manifest_output,
        training_query_count=len(training),
        raw_train_sha256=_sha256(args.raw_train),
        train_partition_sha256=_sha256(args.train_partition),
        training_oof_sha256=_sha256(args.training_oof),
        independent_dev_sha256=_sha256(args.independent_dev),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
