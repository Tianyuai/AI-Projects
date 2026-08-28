from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from paper_search.learning.candidates import query_content_terms
from paper_search.learning.lexical_bridge import (
    LexicalBridgeExample,
    SupervisedLexicalBridge,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _best_title_recall(terms: set[str], titles: list[str]) -> float:
    recalls = []
    for title in titles:
        title_terms = set(query_content_terms(title))
        recalls.append(
            len(terms.intersection(title_terms)) / len(title_terms)
            if title_terms
            else 0.0
        )
    return max(recalls, default=0.0)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "proposal_count": sum(row["proposal_generated"] for row in rows),
        "improved_query_count": sum(row["potential_delta"] > 0 for row in rows),
        "baseline_mean_best_title_recall": mean(
            row["baseline_best_title_recall"] for row in rows
        ),
        "potential_mean_best_title_recall": mean(
            row["potential_best_title_recall"] for row in rows
        ),
        "mean_potential_delta": mean(row["potential_delta"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--train-partition", type=Path, required=True)
    parser.add_argument("--raw-dev", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=("word", "word_char"), default="word")
    parser.add_argument(
        "--learning-objective",
        choices=("neighbor_idf", "association", "support_normalized_idf"),
        default="neighbor_idf",
    )
    parser.add_argument("--max-expansion-terms", type=int, default=3)
    args = parser.parse_args()

    raw_train = _jsonl(args.raw_train)
    allowed_train_ids = {
        row["query_id"] for row in _jsonl(args.train_partition)
    }
    train = [row for row in raw_train if row["qid"] in allowed_train_ids]
    if {row["qid"] for row in train} != allowed_train_ids:
        raise ValueError("raw train does not cover the frozen training partition")
    dev = {row["qid"]: row for row in _jsonl(args.raw_dev)}
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    bridge = SupervisedLexicalBridge.fit(
        [
            LexicalBridgeExample(
                query=row["question"],
                gold_titles=tuple(row["answer"]),
            )
            for row in train
        ],
        representation=args.representation,
        learning_objective=args.learning_objective,
    )

    rows: list[dict[str, Any]] = []
    for frozen in freeze["sample"]:
        raw = dev[frozen["query_id"]]
        baseline_terms = set(query_content_terms(raw["question"]))
        baseline = _best_title_recall(baseline_terms, raw["answer"])
        proposal = bridge.propose(
            raw["question"], max_expansion_terms=args.max_expansion_terms
        )
        bridge_recall = (
            _best_title_recall(
                set(query_content_terms(proposal.query_text)),
                raw["answer"],
            )
            if proposal is not None
            else 0.0
        )
        potential = max(baseline, bridge_recall)
        rows.append(
            {
                **frozen,
                "proposal_generated": proposal is not None,
                "expansion_terms": (
                    list(proposal.expansion_terms) if proposal is not None else []
                ),
                "baseline_best_title_recall": baseline,
                "bridge_best_title_recall": bridge_recall,
                "potential_best_title_recall": potential,
                "potential_delta": potential - baseline,
            }
        )

    strata: dict[str, dict[str, Any]] = {}
    for field in ("fold", "intent_family", "length_bucket", "gold_count_bucket"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[field])].append(row)
        strata[field] = {
            name: _aggregate(group) for name, group in sorted(grouped.items())
        }
    overall = _aggregate(rows)
    positive_folds = sum(
        value["mean_potential_delta"] > 0 for value in strata["fold"].values()
    )
    positive_intents = sum(
        value["mean_potential_delta"] > 0
        for value in strata["intent_family"].values()
    )
    positive_lengths = sum(
        value["mean_potential_delta"] > 0
        for value in strata["length_bucket"].values()
    )
    payload = {
        "schema_version": "supervised-lexical-bridge-offline-audit-v1",
        "test_partition_touched": False,
        "training_query_count": len(train),
        "development_query_count": len(rows),
        "bridge_configuration": {
            "representation": args.representation,
            "learning_objective": args.learning_objective,
            "max_expansion_terms": args.max_expansion_terms,
        },
        "input_sha256": {
            "raw_train": "sha256:" + hashlib.sha256(args.raw_train.read_bytes()).hexdigest(),
            "train_partition": "sha256:"
            + hashlib.sha256(args.train_partition.read_bytes()).hexdigest(),
            "raw_dev": "sha256:" + hashlib.sha256(args.raw_dev.read_bytes()).hexdigest(),
            "freeze": "sha256:" + hashlib.sha256(args.freeze.read_bytes()).hexdigest(),
        },
        "overall": overall,
        "strata": strata,
        "pre_online_gate": {
            "passed": (
                overall["improved_query_count"] >= 20
                and overall["mean_potential_delta"] > 0
                and positive_folds == 3
                and positive_intents >= 2
                and positive_lengths >= 2
            ),
            "minimum_improved_queries": 20,
            "positive_fold_count": positive_folds,
            "positive_intent_family_count": positive_intents,
            "positive_length_bucket_count": positive_lengths,
            "meaning": "Candidate-family potential only; not a production promotion decision.",
        },
        "queries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "queries"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
