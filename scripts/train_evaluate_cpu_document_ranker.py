"""Train the selected CPU document ranker and replay independent A-prime receipts."""

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
from paper_search.learning.document_ranking_oof import evaluate_document_ranker


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


def _filtered_query(
    *,
    query_id: str,
    query: str,
    gold_paper_ids: list[str],
    action_results: list[tuple[str, list[Paper]]],
) -> DocumentRankingQuery:
    return DocumentRankingQuery(
        query_id=query_id,
        query=query,
        gold_paper_ids=gold_paper_ids,
        candidates=build_production_document_candidates(query, action_results),
    )


def _training_queries(
    *,
    freeze: dict[str, Any],
    partition: dict[str, dict[str, Any]],
    lexical_receipts: dict[str, dict[str, Any]],
    semantic_receipts: dict[str, dict[str, Any]],
) -> list[DocumentRankingQuery]:
    output: list[DocumentRankingQuery] = []
    lexical_actions = (
        "ceiling-candidate-anchor",
        "ceiling-candidate-text-1",
        "ceiling-candidate-text-2",
        "ceiling-candidate-text-3",
    )
    for selected in freeze["sample"]:
        query_id = str(selected["query_id"])
        row = partition[query_id]
        lexical_by_id = {
            str(result["action_id"]): result
            for result in lexical_receipts[query_id]["results"]
        }
        semantic_by_id = {
            str(result["action_id"]): result
            for result in semantic_receipts[query_id]["results"]
        }
        action_results = [
            (
                action_id,
                [Paper.model_validate(hit) for hit in lexical_by_id[action_id]["hits"]],
            )
            for action_id in lexical_actions
            if action_id in lexical_by_id
        ]
        action_results.append(
            (
                "semantic-backfill-original",
                [
                    Paper.model_validate(hit)
                    for hit in semantic_by_id["semantic-backfill-original"]["hits"]
                ],
            )
        )
        if "ceiling-candidate-boolean-relaxed" in lexical_by_id:
            action_results.append(
                (
                    "ceiling-candidate-boolean-relaxed",
                    [
                        Paper.model_validate(hit)
                        for hit in lexical_by_id[
                            "ceiling-candidate-boolean-relaxed"
                        ]["hits"]
                    ],
                )
            )
        output.append(
            _filtered_query(
                query_id=query_id,
                query=str(row["query"]),
                gold_paper_ids=list(row["gold_paper_ids"]),
                action_results=action_results,
            )
        )
    return output


def _evaluation_queries(
    *,
    manifest: dict[str, Any],
    partition: dict[str, dict[str, Any]],
    receipts: dict[str, dict[str, Any]],
) -> list[tuple[int, DocumentRankingQuery]]:
    output: list[tuple[int, DocumentRankingQuery]] = []
    for selected in manifest["sample"]:
        query_id = str(selected["query_id"])
        row = partition[query_id]
        receipt = receipts[query_id]
        action_results = [
            (
                str(result["action_id"]),
                [Paper.model_validate(hit) for hit in result["hits"]],
            )
            for result in receipt["results"]
        ]
        output.append(
            (
                int(selected["fold"]),
                _filtered_query(
                    query_id=query_id,
                    query=str(row["query"]),
                    gold_paper_ids=list(row["gold_paper_ids"]),
                    action_results=action_results,
                ),
            )
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-selection", type=Path, required=True)
    parser.add_argument("--train-freeze", type=Path, required=True)
    parser.add_argument("--train-partition", type=Path, required=True)
    parser.add_argument("--train-lexical-root", type=Path, required=True)
    parser.add_argument("--train-semantic-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-partition", type=Path, required=True)
    parser.add_argument("--evaluation-receipt-root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    args = parser.parse_args()

    oof = json.loads(args.oof_selection.read_text(encoding="utf-8"))
    if oof["selected_variant"] != "learned_weight_0.50":
        raise ValueError("independent evaluation requires the frozen 0.50 OOF variant")
    train_freeze = json.loads(args.train_freeze.read_text(encoding="utf-8"))
    evaluation_manifest = json.loads(
        args.evaluation_manifest.read_text(encoding="utf-8")
    )
    train = _training_queries(
        freeze=train_freeze,
        partition={
            row["query_id"]: row for row in _read_jsonl(args.train_partition)
        },
        lexical_receipts=_successful_receipts(args.train_lexical_root),
        semantic_receipts=_successful_receipts(args.train_semantic_root),
    )
    ranker = CpuPairwiseDocumentRanker(learned_weight=0.50, seed=20260818)
    pair_count = ranker.fit(train)
    ranker.save(args.model_output)
    model_sha256 = _sha256(args.model_output)
    evaluation = _evaluation_queries(
        manifest=evaluation_manifest,
        partition={
            row["query_id"]: row
            for row in _read_jsonl(args.evaluation_partition)
        },
        receipts=_successful_receipts(args.evaluation_receipt_root),
    )
    result = evaluate_document_ranker(evaluation, ranker=ranker)
    model_manifest = {
        "schema_version": "cpu-pairwise-document-ranker-manifest-v1",
        "model_id": ranker.model_id,
        "dimension": ranker.dimension,
        "epochs": ranker.epochs,
        "learning_rate": ranker.learning_rate,
        "l2": ranker.l2,
        "learned_weight": ranker.learned_weight,
        "hard_negative_limit": ranker.hard_negative_limit,
        "seed": ranker.seed,
        "training_query_count": len(train),
        "preference_pair_count": pair_count,
        "model_sha256": model_sha256,
        "oof_selection_sha256": _sha256(args.oof_selection),
        "development_labels_read": False,
        "test_partition_touched": False,
    }
    evaluation_payload = {
        "schema_version": "cpu-document-ranking-independent90-v1",
        "model_sha256": model_sha256,
        "evaluation_manifest_sha256": _sha256(args.evaluation_manifest),
        "evaluation_partition_sha256": _sha256(args.evaluation_partition),
        "result": result,
        "test_partition_touched": False,
    }
    write_frozen_bytes(
        args.manifest_output,
        (json.dumps(model_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    write_frozen_bytes(
        args.evaluation_output,
        (json.dumps(evaluation_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "model": model_manifest,
                "baseline": result["baseline"],
                "candidate": result["candidate"],
                "promotion": result["promotion"],
                "candidate_pool_identity_unchanged": result[
                    "candidate_pool_identity_unchanged"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
