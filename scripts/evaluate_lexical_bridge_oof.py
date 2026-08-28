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
from paper_search.learning.lexical_bridge_validation import (
    BridgeFoldResult,
    bridge_training_gate,
    deterministic_bridge_fold,
)


VARIANTS = (
    ("word_neighbor_idf_k3", "word", "neighbor_idf", 3),
    ("word_char_neighbor_idf_k3", "word_char", "neighbor_idf", 3),
    ("word_char_association_k3", "word_char", "association", 3),
    ("word_char_neighbor_idf_k5", "word_char", "neighbor_idf", 5),
    ("word_char_association_k5", "word_char", "association", 5),
    ("word_char_neighbor_idf_k6", "word_char", "neighbor_idf", 6),
    (
        "word_char_support_normalized_idf_k6",
        "word_char",
        "support_normalized_idf",
        6,
    ),
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


def _length_bucket(query: str) -> str:
    length = len(query_content_terms(query))
    if length <= 8:
        return "short"
    if length <= 16:
        return "medium"
    return "long"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "proposal_count": sum(row["proposal_generated"] for row in rows),
        "improved_query_count": sum(row["potential_delta"] > 0 for row in rows),
        "improvement_rate": sum(row["potential_delta"] > 0 for row in rows)
        / len(rows),
        "baseline_mean_best_title_recall": mean(
            row["baseline_best_title_recall"] for row in rows
        ),
        "potential_mean_best_title_recall": mean(
            row["potential_best_title_recall"] for row in rows
        ),
        "mean_potential_delta": mean(row["potential_delta"] for row in rows),
    }


def _evaluate_variant(
    rows: list[dict[str, Any]],
    *,
    representation: str,
    learning_objective: str,
    max_expansion_terms: int,
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    fold_results: list[BridgeFoldResult] = []
    for fold in (1, 2, 3):
        training = [row for row in rows if row["fold"] != fold]
        validation = [row for row in rows if row["fold"] == fold]
        bridge = SupervisedLexicalBridge.fit(
            [
                LexicalBridgeExample(
                    query=row["question"], gold_titles=tuple(row["answer"])
                )
                for row in training
            ],
            representation=representation,  # type: ignore[arg-type]
            learning_objective=learning_objective,  # type: ignore[arg-type]
        )
        fold_rows: list[dict[str, Any]] = []
        for raw in validation:
            baseline_terms = set(query_content_terms(raw["question"]))
            baseline = _best_title_recall(baseline_terms, raw["answer"])
            proposal = bridge.propose(
                raw["question"], max_expansion_terms=max_expansion_terms
            )
            bridge_recall = (
                _best_title_recall(
                    set(query_content_terms(proposal.query_text)), raw["answer"]
                )
                if proposal is not None
                else 0.0
            )
            potential = max(baseline, bridge_recall)
            item = {
                "query_id": raw["qid"],
                "fold": fold,
                "length_bucket": _length_bucket(raw["question"]),
                "proposal_generated": proposal is not None,
                "expansion_terms": (
                    list(proposal.expansion_terms) if proposal is not None else []
                ),
                "baseline_best_title_recall": baseline,
                "bridge_best_title_recall": bridge_recall,
                "potential_best_title_recall": potential,
                "potential_delta": potential - baseline,
            }
            fold_rows.append(item)
            evaluated.append(item)
        fold_summary = _summary(fold_rows)
        by_length: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in fold_rows:
            by_length[item["length_bucket"]].append(item)
        negative_strata = sum(
            _summary(items)["mean_potential_delta"] < 0
            for items in by_length.values()
        )
        fold_results.append(
            BridgeFoldResult(
                fold=fold,
                query_count=fold_summary["query_count"],
                improved_query_count=fold_summary["improved_query_count"],
                mean_potential_delta=fold_summary["mean_potential_delta"],
                negative_stratum_count=negative_strata,
            )
        )
    gate = bridge_training_gate(fold_results)
    return {
        "overall": _summary(evaluated),
        "folds": [item.__dict__ for item in fold_results],
        "training_gate": gate.__dict__,
        "queries": evaluated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--train-partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-variant", choices=[item[0] for item in VARIANTS])
    args = parser.parse_args()

    allowed_ids = {row["query_id"] for row in _jsonl(args.train_partition)}
    rows = [row for row in _jsonl(args.raw_train) if row["qid"] in allowed_ids]
    if len(rows) != len(allowed_ids) or {row["qid"] for row in rows} != allowed_ids:
        raise ValueError("raw train does not exactly cover the frozen training partition")
    for row in rows:
        row["fold"] = deterministic_bridge_fold(row["qid"])

    variants = [
        item for item in VARIANTS if args.only_variant in {None, item[0]}
    ]
    results: dict[str, Any] = {}
    for name, representation, objective, max_expansion_terms in variants:
        results[name] = _evaluate_variant(
            rows,
            representation=representation,
            learning_objective=objective,
            max_expansion_terms=max_expansion_terms,
        )
        summary = results[name]["overall"]
        print(
            json.dumps(
                {
                    "variant": name,
                    "improved_query_count": summary["improved_query_count"],
                    "improvement_rate": summary["improvement_rate"],
                    "mean_potential_delta": summary["mean_potential_delta"],
                    "passed": results[name]["training_gate"]["passed"],
                }
            ),
            flush=True,
        )
    order = {name: index for index, (name, _, _, _) in enumerate(VARIANTS)}
    selected = max(
        results,
        key=lambda name: (
            results[name]["overall"]["improved_query_count"],
            results[name]["overall"]["mean_potential_delta"],
            -order[name],
        ),
    )
    payload = {
        "schema_version": "lexical-bridge-auto-train-oof-v1",
        "test_partition_touched": False,
        "auto_dev_touched": False,
        "training_query_count": len(rows),
        "fold_assignment": "sha256(query_id) % 3 + 1",
        "selection_rule": "improved_query_count, then mean_potential_delta, then frozen variant order",
        "input_sha256": {
            "raw_train": "sha256:"
            + hashlib.sha256(args.raw_train.read_bytes()).hexdigest(),
            "train_partition": "sha256:"
            + hashlib.sha256(args.train_partition.read_bytes()).hexdigest(),
        },
        "selected_variant": selected,
        "selected_variant_passed": results[selected]["training_gate"]["passed"],
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_variant": selected,
                "selected_variant_passed": payload["selected_variant_passed"],
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
