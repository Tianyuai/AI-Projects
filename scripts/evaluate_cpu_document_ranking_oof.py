"""Evaluate conservative CPU document reranking on frozen A-prime receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.cpu_document_ranker import (
    CpuPairwiseDocumentRanker,
    DocumentRankingQuery,
    build_production_document_candidates,
)
from paper_search.learning.document_ranking_oof import evaluate_document_ranking_oof


_LEXICAL_ACTIONS = (
    "ceiling-candidate-anchor",
    "ceiling-candidate-text-1",
    "ceiling-candidate-text-2",
    "ceiling-candidate-text-3",
)
_SEMANTIC_ACTION = "semantic-backfill-original"
_BOOLEAN_ACTION = "ceiling-candidate-boolean-relaxed"
_LEARNED_WEIGHTS = (0.20, 0.35, 0.50)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _successful_receipts(root: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("retrieval/attempt-01/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["attempt_status"] == "succeeded":
            selected[str(payload["query_id"])] = payload
    return selected


def _folded_queries(
    *,
    freeze: dict[str, Any],
    partition: dict[str, dict[str, Any]],
    lexical_receipts: dict[str, dict[str, Any]],
    semantic_receipts: dict[str, dict[str, Any]],
) -> list[tuple[int, DocumentRankingQuery]]:
    output: list[tuple[int, DocumentRankingQuery]] = []
    for selected in freeze["sample"]:
        query_id = str(selected["query_id"])
        row = partition.get(query_id)
        lexical = lexical_receipts.get(query_id)
        semantic = semantic_receipts.get(query_id)
        if row is None or lexical is None or semantic is None:
            raise ValueError(f"missing frozen document-ranking input for {query_id}")
        lexical_by_id = {
            str(result["action_id"]): result for result in lexical["results"]
        }
        semantic_by_id = {
            str(result["action_id"]): result for result in semantic["results"]
        }
        if _SEMANTIC_ACTION not in semantic_by_id:
            raise ValueError(f"missing semantic-original for {query_id}")
        action_results: list[tuple[str, list[Paper]]] = []
        for action_id in _LEXICAL_ACTIONS:
            if action_id in lexical_by_id:
                action_results.append(
                    (
                        action_id,
                        [
                            Paper.model_validate(hit)
                            for hit in lexical_by_id[action_id]["hits"]
                        ],
                    )
                )
        action_results.append(
            (
                _SEMANTIC_ACTION,
                [
                    Paper.model_validate(hit)
                    for hit in semantic_by_id[_SEMANTIC_ACTION]["hits"]
                ],
            )
        )
        if _BOOLEAN_ACTION in lexical_by_id:
            action_results.append(
                (
                    _BOOLEAN_ACTION,
                    [
                        Paper.model_validate(hit)
                        for hit in lexical_by_id[_BOOLEAN_ACTION]["hits"]
                    ],
                )
            )
        filtered = build_production_document_candidates(
            str(row["query"]), action_results
        )
        output.append(
            (
                int(selected["fold"]),
                DocumentRankingQuery(
                    query_id=query_id,
                    query=str(row["query"]),
                    gold_paper_ids=list(row["gold_paper_ids"]),
                    candidates=filtered,
                ),
            )
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--lexical-receipt-root", type=Path, required=True)
    parser.add_argument("--semantic-receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("split") != "auto_train":
        raise ValueError("document ranking OOF requires frozen auto_train queries")
    partition = {row["query_id"]: row for row in _read_jsonl(args.partition)}
    folded = _folded_queries(
        freeze=freeze,
        partition=partition,
        lexical_receipts=_successful_receipts(args.lexical_receipt_root),
        semantic_receipts=_successful_receipts(args.semantic_receipt_root),
    )
    variants: dict[str, dict[str, object]] = {}
    for learned_weight in _LEARNED_WEIGHTS:
        variants[f"learned_weight_{learned_weight:.2f}"] = evaluate_document_ranking_oof(
            folded,
            ranker_factory=lambda seed, weight=learned_weight: CpuPairwiseDocumentRanker(
                learned_weight=weight,
                seed=seed,
            ),
        )
    passing = [
        (name, report)
        for name, report in variants.items()
        if report["promotion"]["promote"]
    ]
    selected_variant = max(
        passing,
        key=lambda item: (
            item[1]["candidate"]["macro_recall_at"][10],
            item[1]["candidate"]["macro_recall_at"][20],
            item[1]["candidate"]["macro_recall_at"][50],
            item[0],
        ),
        default=(None, None),
    )[0]
    payload = {
        "schema_version": "cpu-document-ranking-oof-selection-v1",
        "scope": "pasa_auto_train_three_fold_oof",
        "query_count": len(folded),
        "fixed_learned_weights": list(_LEARNED_WEIGHTS),
        "selected_variant": selected_variant,
        "variants": variants,
        "input_sha256": {
            "freeze": _sha256(args.freeze),
            "partition": _sha256(args.partition),
            "lexical_summary": _sha256(args.lexical_receipt_root / "summary.json"),
            "semantic_summary": _sha256(args.semantic_receipt_root / "summary.json"),
        },
        "development_labels_read": False,
        "test_partition_touched": False,
    }
    content = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    write_frozen_bytes(args.output, content)
    print(
        json.dumps(
            {
                "selected_variant": selected_variant,
                "variants": {
                    name: {
                        "baseline": report["baseline"]["macro_recall_at"],
                        "candidate": report["candidate"]["macro_recall_at"],
                        "promotion": report["promotion"],
                    }
                    for name, report in variants.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
