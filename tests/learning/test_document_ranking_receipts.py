from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_search.domain.models import Paper
from paper_search.learning.document_ranking_receipts import (
    load_a_prime_folded_document_ranking_queries,
    load_folded_document_ranking_evaluation_queries,
    load_supplemental_document_ranking_queries,
)


def _paper(identifier: str, title: str) -> dict[str, object]:
    return Paper(
        canonical_id=identifier,
        openalex_id=identifier,
        title=title,
        is_retracted=False,
    ).model_dump(mode="json")


def _write_receipt(
    root: Path,
    query_id: str,
    actions: list[tuple[str, list[dict[str, object]]]],
) -> None:
    target = root / "openalex" / "batch-0001" / "retrieval" / "attempt-01"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{query_id}.json").write_text(
        json.dumps(
            {
                "attempt_status": "succeeded",
                "query_id": query_id,
                "results": [
                    {"action_id": action_id, "hits": hits}
                    for action_id, hits in actions
                ],
            }
        ),
        encoding="utf-8",
    )


def test_supplemental_loader_pairs_training_receipts_and_excludes_target_ids(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "auto_train.jsonl"
    partition.write_text(
        "\n".join(
            json.dumps(
                {
                    "query_id": query_id,
                    "query": "graph retrieval",
                    "gold_paper_ids": [gold],
                    "role": "training",
                    "split": "auto_train",
                }
            )
            for query_id, gold in (
                ("q-1", "openalex:W1"),
                ("q-2", "openalex:W2"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    lexical_root = tmp_path / "lexical"
    semantic_root = tmp_path / "semantic"
    for query_id, gold in (("q-1", "openalex:W1"), ("q-2", "openalex:W2")):
        _write_receipt(
            lexical_root,
            query_id,
            [("policy-1", [_paper(gold, "graph retrieval")])],
        )
        _write_receipt(
            semantic_root,
            query_id,
            [
                (
                    "semantic-backfill-original",
                    [_paper(gold, "graph retrieval")],
                )
            ],
        )

    queries = load_supplemental_document_ranking_queries(
        partition_path=partition,
        lexical_receipt_roots=[lexical_root],
        semantic_receipt_roots=[semantic_root],
        excluded_query_ids={"q-2"},
        expected_query_count=1,
    )

    assert [query.query_id for query in queries] == ["q-1"]
    assert queries[0].candidates[0].source_ranks == {
        "policy-1": 1,
        "semantic-backfill-original": 1,
    }


def test_supplemental_loader_accepts_fewer_actions_than_frozen_maximum(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "auto_train.jsonl"
    partition.write_text(
        json.dumps(
            {
                "query_id": "q-short",
                "query": "functional logistic regression",
                "gold_paper_ids": ["openalex:W1"],
                "role": "training",
                "split": "auto_train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lexical_root = tmp_path / "lexical"
    semantic_root = tmp_path / "semantic"
    _write_receipt(
        lexical_root,
        "q-short",
        [
            ("policy-1", [_paper("openalex:W1", "functional regression")]),
            ("policy-2", [_paper("openalex:W2", "logistic models")]),
        ],
    )
    _write_receipt(
        semantic_root,
        "q-short",
        [
            (
                "semantic-backfill-original",
                [_paper("openalex:W1", "functional regression")],
            )
        ],
    )

    queries = load_supplemental_document_ranking_queries(
        partition_path=partition,
        lexical_receipt_roots=[lexical_root],
        semantic_receipt_roots=[semantic_root],
        excluded_query_ids=set(),
        expected_query_count=1,
        maximum_lexical_action_count=3,
    )

    assert len(queries) == 1


def test_supplemental_loader_rejects_non_training_partition(tmp_path: Path) -> None:
    partition = tmp_path / "auto_dev.jsonl"
    partition.write_text(
        json.dumps(
            {
                "query_id": "q-dev",
                "query": "graph retrieval",
                "gold_paper_ids": ["openalex:W1"],
                "role": "development",
                "split": "auto_dev",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="auto_train"):
        load_supplemental_document_ranking_queries(
            partition_path=partition,
            lexical_receipt_roots=[tmp_path / "lexical"],
            semantic_receipt_roots=[tmp_path / "semantic"],
            excluded_query_ids=set(),
            expected_query_count=1,
        )


def test_a_prime_loader_preserves_frozen_folds_and_action_identity(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "auto_train.jsonl"
    partition.write_text(
        json.dumps(
            {
                "query_id": "q-target",
                "query": "graph retrieval",
                "gold_paper_ids": ["openalex:W10"],
                "role": "training",
                "split": "auto_train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "split": "auto_train",
                "sample": [{"query_id": "q-target", "fold": 2}],
            }
        ),
        encoding="utf-8",
    )
    lexical_root = tmp_path / "target-lexical"
    semantic_root = tmp_path / "target-semantic"
    _write_receipt(
        lexical_root,
        "q-target",
        [
            (
                "ceiling-candidate-anchor",
                [_paper("openalex:W10", "graph retrieval")],
            ),
            (
                "ceiling-candidate-boolean-relaxed",
                [_paper("openalex:W11", "graph methods")],
            ),
        ],
    )
    _write_receipt(
        semantic_root,
        "q-target",
        [
            (
                "semantic-backfill-original",
                [_paper("openalex:W10", "graph retrieval")],
            )
        ],
    )

    folded = load_a_prime_folded_document_ranking_queries(
        freeze_path=freeze,
        partition_path=partition,
        lexical_receipt_root=lexical_root,
        semantic_receipt_root=semantic_root,
    )

    assert [fold for fold, _query in folded] == [2]
    assert folded[0][1].query_id == "q-target"
    assert folded[0][1].candidates[0].source_ranks == {
        "ceiling-candidate-anchor": 1,
        "semantic-backfill-original": 1,
    }


def test_evaluation_loader_requires_development_partition_and_frozen_folds(
    tmp_path: Path,
) -> None:
    partition = tmp_path / "auto_dev.jsonl"
    partition.write_text(
        json.dumps(
            {
                "query_id": "q-dev",
                "query": "graph retrieval",
                "gold_paper_ids": ["openalex:W20"],
                "role": "development",
                "split": "auto_dev",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"sample": [{"query_id": "q-dev", "fold": 3}]}
        ),
        encoding="utf-8",
    )
    receipts = tmp_path / "receipts"
    _write_receipt(
        receipts,
        "q-dev",
        [
            (
                "ceiling-candidate-anchor",
                [_paper("openalex:W20", "graph retrieval")],
            )
        ],
    )

    folded = load_folded_document_ranking_evaluation_queries(
        manifest_path=manifest,
        partition_path=partition,
        receipt_root=receipts,
    )

    assert [(fold, query.query_id) for fold, query in folded] == [(3, "q-dev")]
