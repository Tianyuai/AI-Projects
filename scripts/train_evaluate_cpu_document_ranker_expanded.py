"""Train the OOF-approved expanded CPU ranker and compare it with frozen v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.cpu_document_ranker import CpuPairwiseDocumentRanker
from paper_search.learning.document_ranking_oof import (
    evaluate_document_ranker_comparison,
)
from paper_search.learning.document_ranking_receipts import (
    load_a_prime_folded_document_ranking_queries,
    load_folded_document_ranking_evaluation_queries,
    load_supplemental_document_ranking_queries,
)
from paper_search.ranking.cpu_document import load_cpu_document_ranking_stage


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_expansion_oof(
    path: Path,
    *,
    target_query_count: int,
    supplemental_query_count: int,
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result", {})
    if payload.get("schema_version") != "cpu-document-ranking-expansion-oof-v1":
        raise ValueError("unsupported document ranking expansion OOF artifact")
    if (
        payload.get("target_query_count") != target_query_count
        or payload.get("supplemental_training_query_count")
        != supplemental_query_count
    ):
        raise ValueError("document ranking expansion OOF query counts do not match")
    if (
        payload.get("development_labels_read") is not False
        or payload.get("test_partition_touched") is not False
        or result.get("candidate_pool_identity_unchanged") is not True
        or result.get("promotion", {}).get("promote") is not True
    ):
        raise ValueError("document ranking expansion OOF did not authorize training")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-freeze", type=Path, required=True)
    parser.add_argument("--train-partition", type=Path, required=True)
    parser.add_argument("--target-lexical-root", type=Path, required=True)
    parser.add_argument("--target-semantic-root", type=Path, required=True)
    parser.add_argument(
        "--supplemental-lexical-root", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--supplemental-semantic-root", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--expected-supplemental-query-count", type=int, required=True
    )
    parser.add_argument(
        "--maximum-supplemental-lexical-actions", type=int, required=True
    )
    parser.add_argument("--expansion-oof", type=Path, required=True)
    parser.add_argument("--current-model-manifest", type=Path, required=True)
    parser.add_argument("--current-model-weights", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-partition", type=Path, required=True)
    parser.add_argument("--evaluation-receipt-root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    args = parser.parse_args(argv)

    target_folded = load_a_prime_folded_document_ranking_queries(
        freeze_path=args.target_freeze,
        partition_path=args.train_partition,
        lexical_receipt_root=args.target_lexical_root,
        semantic_receipt_root=args.target_semantic_root,
    )
    target = [query for _fold, query in target_folded]
    supplemental = load_supplemental_document_ranking_queries(
        partition_path=args.train_partition,
        lexical_receipt_roots=args.supplemental_lexical_root,
        semantic_receipt_roots=args.supplemental_semantic_root,
        excluded_query_ids={query.query_id for query in target},
        expected_query_count=args.expected_supplemental_query_count,
        maximum_lexical_action_count=args.maximum_supplemental_lexical_actions,
    )
    _validate_expansion_oof(
        args.expansion_oof,
        target_query_count=len(target),
        supplemental_query_count=len(supplemental),
    )
    current_stage = load_cpu_document_ranking_stage(
        args.current_model_manifest,
        args.current_model_weights,
    )
    evaluation = load_folded_document_ranking_evaluation_queries(
        manifest_path=args.evaluation_manifest,
        partition_path=args.evaluation_partition,
        receipt_root=args.evaluation_receipt_root,
    )

    ranker = CpuPairwiseDocumentRanker(learned_weight=0.50, seed=20260818)
    pair_count = ranker.fit([*target, *supplemental])
    comparison = evaluate_document_ranker_comparison(
        evaluation,
        baseline_ranker=current_stage.ranker,
        candidate_ranker=ranker,
    )
    ranker.save(args.model_output)
    model_hash = _sha256(args.model_output)
    replacement_authorized = bool(
        comparison["candidate_pool_identity_unchanged"]
        and comparison["promotion"]["promote"]
    )
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
        "training_query_count": len(target) + len(supplemental),
        "target_training_query_count": len(target),
        "supplemental_training_query_count": len(supplemental),
        "preference_pair_count": pair_count,
        "model_sha256": model_hash,
        "expansion_oof_sha256": _sha256(args.expansion_oof),
        "current_model_sha256": _sha256(args.current_model_weights),
        "replacement_authorized": replacement_authorized,
        "development_labels_used_for_training": False,
        "test_partition_touched": False,
    }
    evaluation_payload = {
        "schema_version": "cpu-document-ranking-expanded-stability-v1",
        "model_sha256": model_hash,
        "current_model_sha256": _sha256(args.current_model_weights),
        "evaluation_manifest_sha256": _sha256(args.evaluation_manifest),
        "evaluation_partition_sha256": _sha256(args.evaluation_partition),
        "comparison": comparison,
        "replacement_authorized": replacement_authorized,
        "development_labels_read_for_evaluation": True,
        "development_labels_used_for_training": False,
        "test_partition_touched": False,
    }
    write_frozen_bytes(
        args.manifest_output,
        (
            json.dumps(model_manifest, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8"),
    )
    write_frozen_bytes(
        args.evaluation_output,
        (
            json.dumps(
                evaluation_payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "model": model_manifest,
                "comparison": comparison,
                "replacement_authorized": replacement_authorized,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
