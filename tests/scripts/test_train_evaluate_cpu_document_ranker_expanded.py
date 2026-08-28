from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from paper_search.domain.models import Paper
from paper_search.learning.cpu_document_ranker import CpuPairwiseDocumentRanker
from scripts.train_evaluate_cpu_document_ranker_expanded import main


def _paper(identifier: str, title: str) -> dict[str, object]:
    return Paper(
        canonical_id=identifier,
        openalex_id=identifier,
        title=title,
        is_retracted=False,
    ).model_dump(mode="json")


def _receipt(root: Path, query_id: str, action_id: str, gold: str) -> None:
    target = root / "openalex" / "batch-0001" / "retrieval" / "attempt-01"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{query_id}.json").write_text(
        json.dumps(
            {
                "attempt_status": "succeeded",
                "query_id": query_id,
                "results": [
                    {
                        "action_id": action_id,
                        "hits": [
                            _paper(f"{gold}1", "image synthesis"),
                            _paper(gold, "graph retrieval"),
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_expanded_training_cli_writes_model_and_direct_comparison(
    tmp_path: Path,
) -> None:
    train_partition = tmp_path / "auto_train.jsonl"
    train_rows = [
        {
            "query_id": f"target-{fold}",
            "query": "graph retrieval",
            "gold_paper_ids": [f"openalex:W{fold}0"],
            "role": "training",
            "split": "auto_train",
        }
        for fold in (1, 2, 3)
    ]
    train_rows.append(
        {
            "query_id": "supplemental-1",
            "query": "graph retrieval",
            "gold_paper_ids": ["openalex:W90"],
            "role": "training",
            "split": "auto_train",
        }
    )
    train_partition.write_text(
        "\n".join(json.dumps(row) for row in train_rows) + "\n",
        encoding="utf-8",
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "split": "auto_train",
                "sample": [
                    {"query_id": f"target-{fold}", "fold": fold}
                    for fold in (1, 2, 3)
                ],
            }
        ),
        encoding="utf-8",
    )
    target_lexical = tmp_path / "target-lexical"
    target_semantic = tmp_path / "target-semantic"
    for fold in (1, 2, 3):
        query_id = f"target-{fold}"
        gold = f"openalex:W{fold}0"
        _receipt(target_lexical, query_id, "ceiling-candidate-anchor", gold)
        _receipt(target_semantic, query_id, "semantic-backfill-original", gold)
    supplemental_lexical = tmp_path / "supplemental-lexical"
    supplemental_semantic = tmp_path / "supplemental-semantic"
    _receipt(supplemental_lexical, "supplemental-1", "policy-1", "openalex:W90")
    _receipt(
        supplemental_semantic,
        "supplemental-1",
        "semantic-backfill-original",
        "openalex:W90",
    )
    expansion_oof = tmp_path / "expansion-oof.json"
    expansion_oof.write_text(
        json.dumps(
            {
                "schema_version": "cpu-document-ranking-expansion-oof-v1",
                "target_query_count": 3,
                "supplemental_training_query_count": 1,
                "result": {
                    "candidate_pool_identity_unchanged": True,
                    "promotion": {"promote": True},
                },
                "development_labels_read": False,
                "test_partition_touched": False,
            }
        ),
        encoding="utf-8",
    )
    current_weights = tmp_path / "current.f64"
    current_weights.write_bytes(np.zeros(64, dtype="<f8").tobytes())
    current_hash = hashlib.sha256(current_weights.read_bytes()).hexdigest()
    current_manifest = tmp_path / "current.json"
    current_manifest.write_text(
        json.dumps(
            {
                "schema_version": "cpu-pairwise-document-ranker-manifest-v1",
                "model_id": CpuPairwiseDocumentRanker.model_id,
                "model_sha256": f"sha256:{current_hash}",
                "dimension": 64,
                "epochs": 1,
                "learning_rate": 0.05,
                "l2": 0.000001,
                "learned_weight": 0.5,
                "hard_negative_limit": 10,
                "seed": 1,
            }
        ),
        encoding="utf-8",
    )
    dev_partition = tmp_path / "auto_dev.jsonl"
    dev_rows = [
        {
            "query_id": f"dev-{fold}",
            "query": "graph retrieval",
            "gold_paper_ids": [f"openalex:W{fold}5"],
            "role": "development",
            "split": "auto_dev",
        }
        for fold in (1, 2, 3)
    ]
    dev_partition.write_text(
        "\n".join(json.dumps(row) for row in dev_rows) + "\n",
        encoding="utf-8",
    )
    dev_manifest = tmp_path / "dev-manifest.json"
    dev_manifest.write_text(
        json.dumps(
            {
                "sample": [
                    {"query_id": f"dev-{fold}", "fold": fold}
                    for fold in (1, 2, 3)
                ]
            }
        ),
        encoding="utf-8",
    )
    dev_receipts = tmp_path / "dev-receipts"
    for fold in (1, 2, 3):
        _receipt(
            dev_receipts,
            f"dev-{fold}",
            "ceiling-candidate-anchor",
            f"openalex:W{fold}5",
        )
    model_output = tmp_path / "expanded.f64"
    manifest_output = tmp_path / "expanded.json"
    evaluation_output = tmp_path / "evaluation.json"

    exit_code = main(
        [
            "--target-freeze", str(freeze),
            "--train-partition", str(train_partition),
            "--target-lexical-root", str(target_lexical),
            "--target-semantic-root", str(target_semantic),
            "--supplemental-lexical-root", str(supplemental_lexical),
            "--supplemental-semantic-root", str(supplemental_semantic),
            "--expected-supplemental-query-count", "1",
            "--maximum-supplemental-lexical-actions", "1",
            "--expansion-oof", str(expansion_oof),
            "--current-model-manifest", str(current_manifest),
            "--current-model-weights", str(current_weights),
            "--evaluation-manifest", str(dev_manifest),
            "--evaluation-partition", str(dev_partition),
            "--evaluation-receipt-root", str(dev_receipts),
            "--model-output", str(model_output),
            "--manifest-output", str(manifest_output),
            "--evaluation-output", str(evaluation_output),
        ]
    )

    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert model_output.is_file()
    assert manifest["training_query_count"] == 4
    assert manifest["development_labels_used_for_training"] is False
    assert evaluation["comparison"]["schema_version"] == (
        "cpu-document-ranker-comparison-v1"
    )
    assert evaluation["development_labels_read_for_evaluation"] is True
    assert evaluation["test_partition_touched"] is False
