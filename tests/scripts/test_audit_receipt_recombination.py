from __future__ import annotations

import json
import sys

from scripts.audit_receipt_recombination import main


def _label(query_id: str, action_id: str, mode: str) -> dict[str, object]:
    return {
        "dataset": "pasa",
        "split": "auto_train",
        "role": "training",
        "query_id": query_id,
        "query": query_id,
        "provider": "openalex",
        "action": {
            "action_id": action_id,
            "action_type": "text_search",
            "text": query_id,
            "origin": "deterministic_rule",
            "provider_hint": "openalex",
            "search_mode": mode,
        },
        "retrieval_status": "available",
        "gold_association_count": 1,
        "gold_hit_ids": [],
        "gold_hit_count": 0,
        "action_recall": 0.0,
        "novel_over_anchor_hit_count": 0,
        "error_codes": [],
    }


def test_cli_freezes_hash_bound_recombination_report(tmp_path, monkeypatch) -> None:
    freeze = tmp_path / "freeze.json"
    lexical = tmp_path / "lexical.jsonl"
    semantic = tmp_path / "semantic.jsonl"
    graph = tmp_path / "graph.jsonl"
    output = tmp_path / "result.json"
    query_ids = ("q1", "q2", "q3")
    freeze.write_text(
        json.dumps(
            {
                "split": "auto_train",
                "sample": [
                    {"query_id": query_id, "fold": fold}
                    for query_id, fold in zip(query_ids, (1, 2, 3), strict=True)
                ],
            }
        ),
        encoding="utf-8",
    )
    lexical.write_text(
        "".join(
            json.dumps(_label(query_id, "ceiling-candidate-anchor", "lexical"))
            + "\n"
            for query_id in query_ids
        ),
        encoding="utf-8",
    )
    semantic.write_text(
        "".join(
            json.dumps(_label(query_id, "semantic-backfill-original", "semantic"))
            + "\n"
            for query_id in query_ids
        ),
        encoding="utf-8",
    )
    graph.write_text(
        "".join(
            json.dumps(
                {
                    "dataset": "pasa", "split": "auto_train", "role": "training",
                    "query_id": query_id, "query": query_id,
                    "routing_label": "not_beneficial", "gold_association_count": 1,
                    "anchor_gold_hit_ids": [], "pre_graph_gold_hit_ids": [],
                    "graph_gold_hit_ids": [], "graph_marginal_gold_hit_ids": [],
                    "graph_marginal_recall": 0.0, "seed_count": 1,
                    "graph_action_count": 1, "search_api_calls": 2,
                }
            )
            + "\n"
            for query_id in query_ids
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_receipt_recombination.py",
            "--freeze",
            str(freeze),
            "--lexical-labels",
            str(lexical),
            "--semantic-labels",
            str(semantic),
            "--graph-labels",
            str(graph),
            "--output",
            str(output),
        ],
    )

    assert main() == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["query_count"] == 3
    assert payload["test_partition_touched"] is False
    assert payload["structured_graph_sequence"]["query_count"] == 3
    assert set(payload["input_sha256"]) == {
        "freeze", "lexical_labels", "semantic_labels", "graph_labels"
    }
