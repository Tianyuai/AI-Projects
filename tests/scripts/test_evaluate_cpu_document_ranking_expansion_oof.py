from __future__ import annotations

import json
from pathlib import Path

from paper_search.domain.models import Paper
from scripts.evaluate_cpu_document_ranking_expansion_oof import (
    _validate_current_oof,
    main,
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


def test_expansion_oof_cli_writes_training_only_comparison(tmp_path: Path) -> None:
    partition = tmp_path / "auto_train.jsonl"
    rows = [
        {
            "query_id": f"target-{fold}",
            "query": "graph retrieval",
            "gold_paper_ids": [f"openalex:W{fold}0"],
            "role": "training",
            "split": "auto_train",
        }
        for fold in (1, 2, 3)
    ]
    rows.append(
        {
            "query_id": "supplemental-1",
            "query": "graph retrieval",
            "gold_paper_ids": ["openalex:W90"],
            "role": "training",
            "split": "auto_train",
        }
    )
    partition.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
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
        _write_receipt(
            target_lexical,
            query_id,
            [
                (
                    "ceiling-candidate-anchor",
                    [
                        _paper(f"openalex:W{fold}1", "image synthesis"),
                        _paper(gold, "graph retrieval"),
                    ],
                )
            ],
        )
        _write_receipt(
            target_semantic,
            query_id,
            [("semantic-backfill-original", [_paper(gold, "graph retrieval")])],
        )
    supplemental_lexical = tmp_path / "supplemental-lexical"
    supplemental_semantic = tmp_path / "supplemental-semantic"
    _write_receipt(
        supplemental_lexical,
        "supplemental-1",
        [
            (
                "policy-1",
                [
                    _paper("openalex:W91", "image synthesis"),
                    _paper("openalex:W90", "graph retrieval"),
                ],
            )
        ],
    )
    _write_receipt(
        supplemental_semantic,
        "supplemental-1",
        [
            (
                "semantic-backfill-original",
                [_paper("openalex:W90", "graph retrieval")],
            )
        ],
    )
    output = tmp_path / "result.json"

    exit_code = main(
        [
            "--target-freeze",
            str(freeze),
            "--partition",
            str(partition),
            "--target-lexical-root",
            str(target_lexical),
            "--target-semantic-root",
            str(target_semantic),
            "--supplemental-lexical-root",
            str(supplemental_lexical),
            "--supplemental-semantic-root",
            str(supplemental_semantic),
            "--expected-supplemental-query-count",
            "1",
            "--maximum-supplemental-lexical-actions",
            "1",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["schema_version"] == "cpu-document-ranking-expansion-oof-v1"
    assert payload["target_query_count"] == 3
    assert payload["supplemental_training_query_count"] == 1
    assert payload["development_labels_read"] is False
    assert payload["test_partition_touched"] is False


def test_current_oof_validation_normalizes_json_cutoff_keys(tmp_path: Path) -> None:
    metrics = {
        "query_count": 3,
        "macro_recall_at": {10: 0.1, 20: 0.2, 50: 0.3},
    }
    current = tmp_path / "current.json"
    current.write_text(
        json.dumps(
            {
                "selected_variant": "learned_weight_0.50",
                "variants": {
                    "learned_weight_0.50": {"candidate": metrics}
                },
            }
        ),
        encoding="utf-8",
    )

    digest = _validate_current_oof(current, {"baseline": metrics})

    assert digest.startswith("sha256:")
