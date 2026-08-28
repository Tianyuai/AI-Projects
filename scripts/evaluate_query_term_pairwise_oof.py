from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from paper_search.learning.candidates import query_content_terms
from paper_search.learning.lexical_bridge import (
    LexicalBridgeExample,
    SupervisedLexicalBridge,
)
from paper_search.learning.lexical_bridge_validation import deterministic_bridge_fold
from paper_search.learning.query_term_pairwise_ranker import (
    QueryTermCandidate,
    QueryTermPairwiseRanker,
    build_query_term_candidates,
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


def _training_candidates(
    bridge: SupervisedLexicalBridge,
    rows: list[dict[str, Any]],
) -> list[QueryTermCandidate]:
    result = []
    for index, row in enumerate(rows):
        candidates = build_query_term_candidates(
            bridge,
            query_id=row["qid"],
            query=row["question"],
            gold_titles=tuple(row["answer"]),
            exclude_training_index=index,
        )
        positives = [item for item in candidates if item.relevant]
        negatives = sorted(
            (item for item in candidates if not item.relevant),
            key=lambda item: (
                -item.similarity_sum * item.title_idf,
                item.term,
            ),
        )[:12]
        if positives and negatives:
            result.extend([*positives, *negatives])
    return result


def _evaluate_selection(
    *,
    bridge: SupervisedLexicalBridge,
    row: dict[str, Any],
    selected_terms: tuple[str, ...],
    generic_df_cutoff: float,
) -> dict[str, float | int]:
    gold_terms = {
        term for title in row["answer"] for term in query_content_terms(title)
    }
    overlap = set(selected_terms).intersection(gold_terms)
    baseline_terms = set(query_content_terms(row["question"]))
    baseline_recall = _best_title_recall(baseline_terms, row["answer"])
    potential_recall = max(
        baseline_recall,
        _best_title_recall(baseline_terms.union(selected_terms), row["answer"]),
    )
    training_count = bridge._query_matrix.shape[0]
    generic_count = 0
    for term in selected_terms:
        document_frequency = (
            (training_count + 1) / math.exp(bridge._title_idf[term] - 1) - 1
        )
        generic_count += document_frequency >= generic_df_cutoff
    return {
        "selected_count": len(selected_terms),
        "gold_overlap_count": len(overlap),
        "gold_overlap_query": int(bool(overlap)),
        "weighted_gold_overlap": sum(bridge._title_idf[term] for term in overlap),
        "generic_count": generic_count,
        "potential_delta": potential_recall - baseline_recall,
    }


def _aggregate(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    selected_count = sum(int(row["selected_count"]) for row in rows)
    return {
        "query_count": len(rows),
        "gold_overlap_query_count": sum(
            int(row["gold_overlap_query"]) for row in rows
        ),
        "gold_overlap_count": sum(int(row["gold_overlap_count"]) for row in rows),
        "weighted_gold_overlap": sum(
            float(row["weighted_gold_overlap"]) for row in rows
        ),
        "generic_term_rate": (
            sum(int(row["generic_count"]) for row in rows) / selected_count
            if selected_count
            else 0.0
        ),
        "improved_query_count": sum(
            float(row["potential_delta"]) > 0 for row in rows
        ),
        "mean_potential_delta": mean(
            float(row["potential_delta"]) for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--train-partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    allowed = {row["query_id"] for row in _jsonl(args.train_partition)}
    rows = [row for row in _jsonl(args.raw_train) if row["qid"] in allowed]
    if len(rows) != len(allowed) or {row["qid"] for row in rows} != allowed:
        raise ValueError("raw training rows do not match the frozen partition")
    for row in rows:
        row["fold"] = deterministic_bridge_fold(row["qid"])

    fold_results = []
    for fold in (1, 2, 3):
        training = [row for row in rows if row["fold"] != fold]
        validation = [row for row in rows if row["fold"] == fold]
        bridge = SupervisedLexicalBridge.fit(
            [
                LexicalBridgeExample(row["question"], tuple(row["answer"]))
                for row in training
            ],
            representation="word_char",
            learning_objective="neighbor_idf",
        )
        training_candidates = _training_candidates(bridge, training)
        ranker = QueryTermPairwiseRanker(
            dimension=32768,
            epochs=4,
            seed=17,
        )
        pair_count = ranker.fit(training_candidates)
        training_count = bridge._query_matrix.shape[0]
        title_document_frequencies = np.asarray(
            [
                (training_count + 1) / math.exp(value - 1) - 1
                for value in bridge._title_idf.values()
            ]
        )
        generic_df_cutoff = float(
            np.quantile(title_document_frequencies, 0.95)
        )
        legacy_rows = []
        pairwise_rows = []
        for row in validation:
            candidates = build_query_term_candidates(
                bridge,
                query_id=row["qid"],
                query=row["question"],
                gold_titles=tuple(row["answer"]),
            )
            scores = ranker.score(candidates)
            selected = tuple(
                item.term
                for item, _ in sorted(
                    zip(candidates, scores, strict=True),
                    key=lambda pair: (-pair[1], pair[0].term),
                )[:6]
            )
            legacy = bridge.propose(
                row["question"],
                neighbors=12,
                max_expansion_terms=6,
                min_neighbor_support=2,
            )
            legacy_terms = legacy.expansion_terms if legacy is not None else ()
            legacy_rows.append(
                _evaluate_selection(
                    bridge=bridge,
                    row=row,
                    selected_terms=legacy_terms,
                    generic_df_cutoff=generic_df_cutoff,
                )
            )
            pairwise_rows.append(
                _evaluate_selection(
                    bridge=bridge,
                    row=row,
                    selected_terms=selected,
                    generic_df_cutoff=generic_df_cutoff,
                )
            )
        fold_result = {
            "fold": fold,
            "training_query_count": len(training),
            "validation_query_count": len(validation),
            "training_candidate_count": len(training_candidates),
            "preference_pair_count": pair_count,
            "legacy": _aggregate(legacy_rows),
            "pairwise": _aggregate(pairwise_rows),
        }
        fold_results.append(fold_result)
        print(json.dumps(fold_result, ensure_ascii=False), flush=True)

    weighted_non_decreasing = all(
        fold["pairwise"]["weighted_gold_overlap"]
        >= fold["legacy"]["weighted_gold_overlap"]
        for fold in fold_results
    )
    weighted_strict_folds = sum(
        fold["pairwise"]["weighted_gold_overlap"]
        > fold["legacy"]["weighted_gold_overlap"]
        for fold in fold_results
    )
    generic_non_increasing = all(
        fold["pairwise"]["generic_term_rate"]
        <= fold["legacy"]["generic_term_rate"]
        for fold in fold_results
    )
    generic_strict_folds = sum(
        fold["pairwise"]["generic_term_rate"]
        < fold["legacy"]["generic_term_rate"]
        for fold in fold_results
    )
    recall_non_decreasing = all(
        fold["pairwise"]["mean_potential_delta"]
        >= fold["legacy"]["mean_potential_delta"]
        for fold in fold_results
    )
    gate = {
        "passed": (
            weighted_non_decreasing
            and weighted_strict_folds >= 2
            and generic_non_increasing
            and generic_strict_folds >= 2
            and recall_non_decreasing
        ),
        "weighted_gold_overlap_non_decreasing_all_folds": weighted_non_decreasing,
        "weighted_gold_overlap_strict_fold_count": weighted_strict_folds,
        "generic_term_rate_non_increasing_all_folds": generic_non_increasing,
        "generic_term_rate_strict_fold_count": generic_strict_folds,
        "mean_potential_delta_non_decreasing_all_folds": recall_non_decreasing,
        "stopping_rule": "A failure ends lexical-bridge local optimization; no tuning loop.",
    }
    payload = {
        "schema_version": "query-term-pairwise-oof-v1",
        "test_partition_touched": False,
        "training_query_count": len(rows),
        "configuration": {
            "candidate_min_neighbor_support": 1,
            "neighbors": 12,
            "max_selected_terms": 6,
            "hard_negatives_per_query": 12,
            "dimension": 32768,
            "epochs": 4,
            "seed": 17,
        },
        "input_sha256": {
            "raw_train": "sha256:"
            + hashlib.sha256(args.raw_train.read_bytes()).hexdigest(),
            "train_partition": "sha256:"
            + hashlib.sha256(args.train_partition.read_bytes()).hexdigest(),
        },
        "folds": fold_results,
        "promotion_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"promotion_gate": gate, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
