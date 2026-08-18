from __future__ import annotations

import json

from paper_search.learning.fixed_budget_integrity import (
    audit_core4_semantic_boolean_queries,
    audit_fixed_budget_receipts,
)
from paper_search.learning.fixed_budget_comparison import (
    evaluate_paired_candidate_pools,
)


def _write_report(root, name: str, actions: list[dict[str, object]]) -> None:
    batch = root / name
    batch.mkdir(parents=True)
    (batch / "canary-report.json").write_text(
        json.dumps({"actions_by_query": {"q-1": actions}}),
        encoding="utf-8",
    )


def test_offline_integrity_rebuilds_scheme_a_from_existing_receipts(tmp_path) -> None:
    query = "Which paper proposed graph diffusion for retrieval?"
    structured = tmp_path / "structured"
    semantic = tmp_path / "semantic"
    _write_report(
        structured,
        "batch-0001",
        [
            {
                "action_id": "structured-anchor-original",
                "strategy": "structured:anchor-original",
                "action_type": "text_search",
                "payload": {"query_text": query, "search_mode": "lexical"},
            },
            {
                "action_id": "structured-anchor-semantic",
                "strategy": "structured:anchor-semantic",
                "action_type": "text_search",
                "payload": {"query_text": query, "search_mode": "semantic"},
            },
            {
                "action_id": "structured-title-target",
                "strategy": "structured:relation-target",
                "action_type": "title_search",
                "payload": {"title_text": "graph diffusion for retrieval"},
            },
            {
                "action_id": "structured-relation-target",
                "strategy": "structured:relation-target",
                "action_type": "text_search",
                "payload": {
                    "query_text": "proposed graph diffusion for retrieval",
                    "search_mode": "lexical",
                },
            },
            {
                "action_id": "structured-graph-1-citations",
                "strategy": "structured:openalex-citation-graph",
                "action_type": "citation_expand",
                "payload": {
                    "seed_canonical_id": "openalex:W1",
                    "direction": "citations",
                    "limit": 50,
                },
            },
        ],
    )
    _write_report(
        semantic,
        "batch-0001",
        [
            {
                "action_id": "semantic-backfill-original",
                "strategy": "openalex-semantic-backfill-v1",
                "action_type": "text_search",
                "payload": {"query_text": query, "search_mode": "semantic"},
            }
        ],
    )

    result = audit_fixed_budget_receipts(
        structured_root=structured,
        semantic_root=semantic,
    )

    assert result["query_count"] == 1
    assert result["maximum_action_count"] <= 6
    assert result["duplicate_action_count"] == 0
    assert result["anchor_failure_count"] == 0
    assert result["graph_action_count"] == 0
    assert result["receipt_compatible_query_count"] == 1
    assert result["passed"] is True


def test_a_prime_integrity_audits_frozen_queries_without_online_calls() -> None:
    result = audit_core4_semantic_boolean_queries(
        [
            {
                "query_id": "q-1",
                "query": (
                    "Which paper proposed graph diffusion networks for information "
                    "retrieval without supervised fine tuning?"
                ),
            },
            {"query_id": "q-2", "query": "Find graph retrieval"},
        ]
    )

    assert result["query_count"] == 2
    assert result["maximum_action_count"] <= 6
    assert result["duplicate_action_count"] == 0
    assert result["anchor_failure_count"] == 0
    assert result["semantic_original_failure_count"] == 0
    assert result["composition_failure_count"] == 0
    assert result["forbidden_action_count"] == 0
    assert result["unused_budget_query_count"] == 1
    assert result["passed"] is True


def test_paired_gate_requires_two_improved_folds_and_no_decline() -> None:
    manifest = {"q-1": 1, "q-2": 2, "q-3": 3}
    baseline = {
        "q-1": {"candidate_recall": 0.0, "gold_hit_count": 0, "candidate_count": 2},
        "q-2": {"candidate_recall": 0.0, "gold_hit_count": 0, "candidate_count": 2},
        "q-3": {"candidate_recall": 1.0, "gold_hit_count": 1, "candidate_count": 2},
    }
    candidate = {
        query_id: {
            "candidate_recall": 1.0,
            "gold_hit_count": 1,
            "candidate_count": 2,
        }
        for query_id in manifest
    }
    actions = {
        query_id: [
            {
                "action_type": "text_search",
                "payload": {"query_text": query_id, "search_mode": "lexical"},
            },
            {
                "action_type": "text_search",
                "payload": {"query_text": query_id, "search_mode": "semantic"},
            },
        ]
        for query_id in manifest
    }

    result = evaluate_paired_candidate_pools(
        fold_by_query=manifest,
        query_text_by_id={query_id: query_id for query_id in manifest},
        baseline_rows=baseline,
        candidate_rows=candidate,
        baseline_actions=actions,
        candidate_actions=actions,
        baseline_raw_hit_counts={query_id: 3 for query_id in manifest},
        candidate_raw_hit_counts={query_id: 3 for query_id in manifest},
    )

    assert result["decision"]["promote"] is True
    assert result["decision"]["improved_fold_count"] == 2
    assert result["decision"]["declined_fold_count"] == 0
    assert result["candidate"]["hit_query_count"] == 3
    assert result["candidate"]["gold_hits_per_action"] == 0.5
    assert result["candidate"]["duplicate_rate"] == 1 / 3


def test_a_prime_gate_enforces_safety_and_incremental_efficiency() -> None:
    folds = {"q-1": 1, "q-2": 2, "q-3": 3}
    baseline = {
        query_id: {
            "candidate_recall": 0.0,
            "gold_hit_count": 0,
            "candidate_count": 2,
        }
        for query_id in folds
    }
    candidate = {
        "q-1": {"candidate_recall": 1.0, "gold_hit_count": 1, "candidate_count": 3},
        "q-2": {"candidate_recall": 1.0, "gold_hit_count": 1, "candidate_count": 3},
        "q-3": {"candidate_recall": 0.0, "gold_hit_count": 0, "candidate_count": 3},
    }
    baseline_actions = {
        query_id: [
            {
                "action_type": "text_search",
                "payload": {"query_text": query_id, "search_mode": "lexical"},
            }
        ]
        for query_id in folds
    }
    candidate_actions = {
        query_id: [
            *baseline_actions[query_id],
            {
                "action_type": "text_search",
                "payload": {"query_text": query_id, "search_mode": "semantic"},
            },
        ]
        for query_id in folds
    }
    gate = {
        "maximum_openalex_actions_per_query": 6,
        "require_overall_candidate_oracle_strict_improvement": True,
        "minimum_improved_folds": 2,
        "maximum_declined_folds": 0,
        "require_hit_query_count_non_decrease": True,
        "require_gold_hit_count_non_decrease": True,
        "require_duplicate_rate_non_increase": True,
        "minimum_marginal_gold_hits_per_added_action": 0.03198,
    }

    result = evaluate_paired_candidate_pools(
        fold_by_query=folds,
        query_text_by_id={query_id: query_id for query_id in folds},
        baseline_rows=baseline,
        candidate_rows=candidate,
        baseline_actions=baseline_actions,
        candidate_actions=candidate_actions,
        baseline_raw_hit_counts={query_id: 2 for query_id in folds},
        candidate_raw_hit_counts={query_id: 3 for query_id in folds},
        promotion_gate=gate,
    )

    assert result["decision"]["promote"] is True
    assert result["marginal"]["gold_hit_count"] == 2
    assert result["marginal"]["added_action_count"] == 3
    assert result["marginal"]["gold_hits_per_added_action"] == 2 / 3


def test_a_prime_gate_rejects_metric_regression_even_when_macro_recall_improves() -> None:
    folds = {"q-1": 1, "q-2": 2, "q-3": 3}
    baseline = {
        "q-1": {"candidate_recall": 0.5, "gold_hit_count": 4, "candidate_count": 4},
        "q-2": {"candidate_recall": 0.0, "gold_hit_count": 0, "candidate_count": 4},
        "q-3": {"candidate_recall": 0.0, "gold_hit_count": 0, "candidate_count": 4},
    }
    candidate = {
        "q-1": {"candidate_recall": 0.25, "gold_hit_count": 1, "candidate_count": 3},
        "q-2": {"candidate_recall": 0.5, "gold_hit_count": 1, "candidate_count": 3},
        "q-3": {"candidate_recall": 0.5, "gold_hit_count": 1, "candidate_count": 3},
    }
    actions = {
        query_id: [
            {
                "action_type": "text_search",
                "payload": {"query_text": query_id, "search_mode": "lexical"},
            },
            {
                "action_type": "text_search",
                "payload": {"query_text": query_id, "search_mode": "semantic"},
            },
        ]
        for query_id in folds
    }

    result = evaluate_paired_candidate_pools(
        fold_by_query=folds,
        query_text_by_id={query_id: query_id for query_id in folds},
        baseline_rows=baseline,
        candidate_rows=candidate,
        baseline_actions=actions,
        candidate_actions=actions,
        baseline_raw_hit_counts={query_id: 4 for query_id in folds},
        candidate_raw_hit_counts={query_id: 4 for query_id in folds},
        promotion_gate={
            "require_overall_candidate_oracle_strict_improvement": True,
            "minimum_improved_folds": 2,
            "maximum_declined_folds": 1,
            "require_gold_hit_count_non_decrease": True,
        },
    )

    assert result["candidate_oracle_macro_recall_delta"] > 0
    assert result["decision"]["promote"] is False
    assert "gold_hit_count_non_decrease" in result["decision"]["failed_conditions"]
