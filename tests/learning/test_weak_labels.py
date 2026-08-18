from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_search.learning.weak_labels import (
    build_query_term_labels,
    freeze_query_term_labels,
)


def test_builder_labels_only_query_derived_terms_without_gold_payload() -> None:
    rows = build_query_term_labels(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id="q1",
        query="Which paper introduced graph diffusion networks?",
        gold_titles=["Graph Diffusion Networks for Learning"],
    )

    by_term = {row.action_text: row for row in rows}
    assert by_term["graph"].label == "positive"
    assert by_term["diffusion"].label == "positive"
    assert by_term["networks"].label == "positive"
    assert by_term["introduced"].label == "hard_negative"
    payload = json.dumps([row.model_dump(mode="json") for row in rows])
    assert "Graph Diffusion Networks for Learning" not in payload
    assert "gold" not in payload.casefold()


def test_builder_rejects_final_test_labels() -> None:
    with pytest.raises(ValueError, match="final_test"):
        build_query_term_labels(
            dataset="pasa",
            split="auto_test",
            role="final_test",
            query_id="q1",
            query="graph diffusion",
            gold_titles=["Graph Diffusion"],
        )


def test_freeze_joins_only_allowed_frozen_query_ids(tmp_path: Path) -> None:
    partition = tmp_path / "partition.jsonl"
    partition.write_text(
        json.dumps(
            {
                "dataset": "pasa",
                "split": "auto_train",
                "role": "training",
                "revision": "fixed",
                "query_id": "keep",
                "query": "graph diffusion method",
                "gold_paper_ids": ["arxiv:1"],
                "source_components": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid": "keep",
                        "question": " graph diffusion method ",
                        "answer": ["Graph Diffusion"],
                    }
                ),
                json.dumps(
                    {
                        "qid": "excluded",
                        "question": "private query",
                        "answer": ["Private Gold"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = freeze_query_term_labels(
        partition_path=partition,
        source_path=source,
        output_path=tmp_path / "labels.jsonl",
    )

    assert manifest.query_count == 1
    assert manifest.label_count == 3
    output = (tmp_path / "labels.jsonl").read_text(encoding="utf-8")
    assert "excluded" not in output
    assert "Private Gold" not in output
