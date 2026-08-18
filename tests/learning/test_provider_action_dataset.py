from __future__ import annotations

import json

from paper_search.learning.provider_action_dataset import (
    freeze_provider_action_labels,
    load_provider_action_labels_from_canary_runs,
    merge_provider_action_label_sets,
)
from paper_search.learning.provider_action_labels import ProviderActionLabel


def _write_json(path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_loads_action_level_labels_from_saved_canary_receipts(tmp_path) -> None:
    partition = tmp_path / "pasa_auto_dev.jsonl"
    partition.write_text(
        json.dumps(
            {
                "dataset": "pasa",
                "split": "auto_dev",
                "role": "development",
                "query_id": "q-1",
                "query": "Find graph ℒ papers",
                "gold_paper_ids": ["arxiv:2001.00001"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    batch = tmp_path / "runs" / "openalex" / "batch-0001"
    _write_json(
        batch / "canary-report.json",
        {
            "actions_by_query": {
                "q-1": [
                    {
                        "action_id": "policy-1",
                        "action_type": "text_search",
                        "strategy": "learned_action_ranker",
                        "payload": {"query_text": "Find graph L papers"},
                    },
                    {
                        "action_id": "policy-2",
                        "action_type": "text_search",
                        "strategy": "learned_action_ranker",
                        "payload": {
                            "query_text": "graph diffusion",
                            "search_mode": "semantic",
                        },
                    },
                ]
            }
        },
    )
    _write_json(
        batch / "retrieval" / "attempt-01" / "q-1.json",
        {
            "attempt_id": "attempt-01",
            "attempt_status": "succeeded",
            "query_id": "q-1",
            "results": [
                {
                    "action_id": "policy-1",
                    "action_type": "text_search",
                    "hits": [],
                    "usage": {
                        "search_api_calls": 1,
                        "input_tokens": "[REDACTED]",
                        "output_tokens": "[REDACTED]",
                    },
                },
                {
                    "action_id": "policy-2",
                    "action_type": "text_search",
                    "hits": [
                        {
                            "canonical_id": "arxiv:2001.00001",
                            "arxiv_id": "2001.00001",
                            "title": "Graph diffusion",
                            "sources": ["openalex"],
                        }
                    ],
                    "usage": {"search_api_calls": 1},
                },
            ],
        },
    )

    labels = load_provider_action_labels_from_canary_runs(
        partition_path=partition,
        provider_run_roots={"openalex": batch.parent},
    )

    assert len(labels) == 2
    assert labels[0].action.origin == "original_query"
    assert labels[0].gold_hit_count == 0
    assert labels[1].action.origin == "deterministic_rule"
    assert labels[1].action.search_mode == "semantic"
    assert labels[1].gold_hit_count == 1
    assert labels[1].novel_over_anchor_hit_count == 1
    digest = freeze_provider_action_labels(
        labels,
        tmp_path / "labels" / "provider-actions.jsonl",
    )
    frozen_lines = (
        tmp_path / "labels" / "provider-actions.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert digest.startswith("sha256:")
    assert len(frozen_lines) == 2


def test_saved_infrastructure_failure_is_unavailable(tmp_path) -> None:
    partition = tmp_path / "pasa_auto_train.jsonl"
    partition.write_text(
        json.dumps(
            {
                "dataset": "pasa",
                "split": "auto_train",
                "role": "training",
                "query_id": "q-1",
                "query": "Find graph papers",
                "gold_paper_ids": ["arxiv:2001.00001"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    batch = tmp_path / "runs" / "semantic_scholar" / "batch-0001"
    action = {
        "action_id": "policy-1",
        "action_type": "text_search",
        "strategy": "learned_action_ranker",
        "payload": {"query_text": "Find graph papers"},
    }
    _write_json(batch / "canary-report.json", {"actions_by_query": {"q-1": [action]}})
    _write_json(
        batch / "retrieval" / "attempt-01" / "q-1.json",
        {
            "attempt_id": "attempt-01",
            "attempt_status": "failed",
            "query_id": "q-1",
            "results": [
                {
                    "action_id": "policy-1",
                    "action_type": "text_search",
                    "hits": [],
                    "errors": [
                        {
                            "code": "rate_limited",
                            "message": "limited",
                            "retryable": True,
                            "provider": "semantic_scholar",
                        }
                    ],
                    "infrastructure_failure": True,
                }
            ],
        },
    )

    [label] = load_provider_action_labels_from_canary_runs(
        partition_path=partition,
        provider_run_roots={"semantic_scholar": batch.parent},
    )

    assert label.retrieval_status == "unavailable"
    assert label.error_codes == ("rate_limited",)


def test_merge_overrides_corrected_receipt_and_recomputes_anchor_novelty() -> None:
    def label(
        action_id: str,
        *,
        origin: str,
        search_mode: str,
        hits: list[str],
    ) -> ProviderActionLabel:
        return ProviderActionLabel.model_validate(
            {
                "dataset": "pasa",
                "split": "auto_train",
                "role": "training",
                "query_id": "q-1",
                "query": "find graph papers",
                "provider": "openalex",
                "action": {
                    "action_id": action_id,
                    "action_type": "text_search",
                    "text": "find graph papers",
                    "origin": origin,
                    "provider_hint": "openalex",
                    "search_mode": search_mode,
                },
                "retrieval_status": "available",
                "gold_association_count": 2,
                "gold_hit_ids": hits,
                "gold_hit_count": len(hits),
                "action_recall": len(hits) / 2,
                "novel_over_anchor_hit_count": len(hits),
            }
        )

    anchor = label(
        "anchor", origin="original_query", search_mode="lexical", hits=["doi:a"]
    )
    stale = label(
        "semantic", origin="deterministic_rule", search_mode="semantic", hits=[]
    )
    corrected = label(
        "semantic",
        origin="deterministic_rule",
        search_mode="semantic",
        hits=["doi:a", "doi:b"],
    )

    merged = merge_provider_action_label_sets(
        [[anchor, stale], [corrected]], later_sets_override=True
    )

    assert [row.action.action_id for row in merged] == ["anchor", "semantic"]
    assert merged[0].novel_over_anchor_hit_count == 0
    assert merged[1].gold_hit_ids == ("doi:a", "doi:b")
    assert merged[1].novel_over_anchor_hit_count == 1
