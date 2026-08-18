"""Compare frozen A-prime OOF ranking with disjoint supplemental CPU training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.cpu_document_ranker import CpuPairwiseDocumentRanker
from paper_search.learning.document_ranking_oof import (
    evaluate_document_ranking_oof_expansion,
)
from paper_search.learning.document_ranking_receipts import (
    load_a_prime_folded_document_ranking_queries,
    load_supplemental_document_ranking_queries,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_tree_sha256(roots: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for index, root in enumerate(roots):
        for path in sorted(root.rglob("retrieval/attempt-01/*.json")):
            digest.update(f"{index}:{path.relative_to(root).as_posix()}\0".encode())
            digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _validate_current_oof(path: Path, report: dict[str, object]) -> str:
    current = json.loads(path.read_text(encoding="utf-8"))
    selected_name = current.get("selected_variant")
    selected = current.get("variants", {}).get(selected_name)
    if selected_name != "learned_weight_0.50" or not isinstance(selected, dict):
        raise ValueError("current OOF artifact does not contain frozen v3 selection")
    normalized_baseline = json.loads(
        json.dumps(report["baseline"], sort_keys=True)
    )
    if selected.get("candidate") != normalized_baseline:
        raise ValueError("expanded OOF baseline does not reproduce current v3 OOF")
    return _sha256(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-freeze", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
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
    parser.add_argument("--current-oof", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    target = load_a_prime_folded_document_ranking_queries(
        freeze_path=args.target_freeze,
        partition_path=args.partition,
        lexical_receipt_root=args.target_lexical_root,
        semantic_receipt_root=args.target_semantic_root,
    )
    target_ids = {query.query_id for _fold, query in target}
    supplemental = load_supplemental_document_ranking_queries(
        partition_path=args.partition,
        lexical_receipt_roots=args.supplemental_lexical_root,
        semantic_receipt_roots=args.supplemental_semantic_root,
        excluded_query_ids=target_ids,
        expected_query_count=args.expected_supplemental_query_count,
        maximum_lexical_action_count=args.maximum_supplemental_lexical_actions,
    )
    report = evaluate_document_ranking_oof_expansion(
        target,
        supplemental_training_queries=supplemental,
        baseline_ranker_factory=lambda seed: CpuPairwiseDocumentRanker(
            learned_weight=0.50,
            seed=seed,
        ),
        candidate_ranker_factory=lambda seed: CpuPairwiseDocumentRanker(
            learned_weight=0.50,
            seed=seed,
        ),
    )
    current_oof_sha256 = (
        _validate_current_oof(args.current_oof, report)
        if args.current_oof is not None
        else None
    )
    payload = {
        "schema_version": "cpu-document-ranking-expansion-oof-v1",
        "scope": "pasa_auto_train_a_prime_target_with_paired2000_supplement",
        "target_query_count": len(target),
        "supplemental_training_query_count": len(supplemental),
        "training_protocol": {
            "baseline": "two_target_folds_only",
            "candidate": "two_target_folds_plus_disjoint_supplemental",
            "learned_weight": 0.50,
            "feature_and_hyperparameter_changes": False,
        },
        "result": report,
        "input_sha256": {
            "target_freeze": _sha256(args.target_freeze),
            "partition": _sha256(args.partition),
            "target_lexical_receipts": _receipt_tree_sha256(
                [args.target_lexical_root]
            ),
            "target_semantic_receipts": _receipt_tree_sha256(
                [args.target_semantic_root]
            ),
            "supplemental_lexical_receipts": _receipt_tree_sha256(
                args.supplemental_lexical_root
            ),
            "supplemental_semantic_receipts": _receipt_tree_sha256(
                args.supplemental_semantic_root
            ),
            "current_oof": current_oof_sha256,
        },
        "development_labels_read": False,
        "test_partition_touched": False,
    }
    write_frozen_bytes(
        args.output,
        (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "target_query_count": len(target),
                "supplemental_training_query_count": len(supplemental),
                "baseline": report["baseline"],
                "candidate": report["candidate"],
                "promotion": report["promotion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
