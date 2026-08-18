from __future__ import annotations

import pytest

from paper_search.learning.receipt_recombination import (
    analyze_receipt_recombination,
    analyze_structured_graph_marginals,
)


def _label(
    query_id: str,
    action_id: str,
    hits: tuple[str, ...],
    *,
    gold_count: int = 3,
    search_mode: str = "lexical",
) -> dict[str, object]:
    return {
        "dataset": "pasa",
        "split": "auto_train",
        "role": "training",
        "query_id": query_id,
        "query": f"query {query_id}",
        "provider": "openalex",
        "action": {
            "action_id": action_id,
            "action_type": "text_search",
            "text": action_id,
            "origin": (
                "original_query" if action_id.endswith("anchor") else "deterministic_rule"
            ),
            "provider_hint": "openalex",
            "search_mode": search_mode,
        },
        "retrieval_status": "available",
        "gold_association_count": gold_count,
        "gold_hit_ids": list(hits),
        "gold_hit_count": len(hits),
        "action_recall": len(hits) / gold_count,
        "novel_over_anchor_hit_count": 0,
        "error_codes": [],
    }


def _fixture() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    frozen = [
        {"query_id": "q1", "fold": 1},
        {"query_id": "q2", "fold": 2},
        {"query_id": "q3", "fold": 3},
    ]
    lexical = [
        _label("q1", "ceiling-candidate-anchor", ("g1",)),
        _label("q1", "ceiling-candidate-text-1", ("g2",)),
        _label("q1", "ceiling-candidate-prf-1", ("g3",)),
        _label("q2", "ceiling-candidate-anchor", ("g1",)),
        _label("q2", "ceiling-candidate-boolean-relaxed", ("g2",)),
        _label("q2", "ceiling-candidate-text-1", ()),
        _label("q3", "ceiling-candidate-anchor", ()),
        _label("q3", "ceiling-candidate-phrase-proximity", ("g1",)),
        _label("q3", "ceiling-candidate-text-1", ("g2",)),
    ]
    semantic = [
        _label(
            "q1",
            "semantic-backfill-original",
            ("g3",),
            search_mode="semantic",
        ),
        _label(
            "q2",
            "semantic-backfill-original",
            (),
            search_mode="semantic",
        ),
        _label(
            "q3",
            "semantic-backfill-original",
            ("g3",),
            search_mode="semantic",
        ),
    ]
    return frozen, lexical, semantic


def test_recombination_reports_cumulative_top_k_and_semantic_compositions() -> None:
    frozen, lexical, semantic = _fixture()

    report = analyze_receipt_recombination(
        frozen_rows=frozen,
        lexical_labels=lexical,
        semantic_labels=semantic,
    )

    assert report["query_count"] == 3
    top_k = report["lexical_top_k"]
    assert [row["top_k"] for row in top_k] == [1, 2, 3]
    assert top_k[0]["overall"]["gold_hit_count"] == 2
    assert top_k[1]["overall"]["gold_hit_count"] == 5
    assert top_k[2]["overall"]["gold_hit_count"] == 7
    assert top_k[1]["marginal_new_gold_hit_count"] == 3
    assert top_k[2]["marginal_new_gold_hit_count"] == 2

    with_semantic = report["lexical_top_k_plus_semantic"]
    assert with_semantic[0]["overall"]["gold_hit_count"] == 4
    assert with_semantic[1]["overall"]["gold_hit_count"] == 7
    assert with_semantic[2]["overall"]["gold_hit_count"] == 8
    assert all(set(row["folds"]) == {"1", "2", "3"} for row in with_semantic)


def test_recombination_isolates_document_feedback_after_core_and_semantic() -> None:
    frozen, lexical, semantic = _fixture()

    report = analyze_receipt_recombination(
        frozen_rows=frozen,
        lexical_labels=lexical,
        semantic_labels=semantic,
    )

    sources = report["candidate_sources"]
    assert sources["semantic"]["new_gold_over_core_count"] == 2
    assert sources["boolean_phrase"]["new_gold_over_core_count"] == 2
    assert sources["prf"]["new_gold_over_core_count"] == 1
    assert sources["prf"]["new_gold_over_core_plus_semantic_count"] == 0
    assert sources["prf"]["action_count"] == 1
    assert sources["prf"]["positive_fold_count_over_core"] == 1
    assert report["test_partition_touched"] is False

    compositions = report["budget_six_compositions"]
    assert set(compositions) == {
        "receipt_prefix5_semantic",
        "core5_semantic",
        "core4_semantic",
        "core4_semantic_boolean",
        "core3_semantic_boolean_phrase",
        "core4_semantic_prf",
        "core3_semantic_boolean_prf",
    }
    assert all(row["maximum_action_count"] <= 6 for row in compositions.values())
    assert compositions["core5_semantic"]["overall"]["gold_hit_count"] == 6
    assert compositions["core4_semantic"]["overall"]["gold_hit_count"] == 6
    assert (
        compositions["core3_semantic_boolean_phrase"]["overall"]["gold_hit_count"]
        == 8
    )


def test_recombination_rejects_receipts_with_missing_frozen_queries() -> None:
    frozen, lexical, semantic = _fixture()

    with pytest.raises(ValueError, match="lexical labels do not match"):
        analyze_receipt_recombination(
            frozen_rows=frozen,
            lexical_labels=[row for row in lexical if row["query_id"] != "q3"],
            semantic_labels=semantic,
        )


def test_structured_graph_marginals_follow_base_semantic_structured_graph_order() -> None:
    frozen = [
        {"query_id": "q1", "fold": 1},
        {"query_id": "q2", "fold": 2},
        {"query_id": "q3", "fold": 3},
    ]
    lexical = [
        _label("q1", "ceiling-candidate-anchor", ("g1",), gold_count=4),
        _label("q2", "ceiling-candidate-anchor", ("g1",), gold_count=4),
        _label("q3", "ceiling-candidate-anchor", (), gold_count=4),
    ]
    semantic = [
        _label(
            "q1", "semantic-backfill-original", ("g2",), gold_count=4,
            search_mode="semantic",
        ),
        _label(
            "q2", "semantic-backfill-original", (), gold_count=4,
            search_mode="semantic",
        ),
        _label(
            "q3", "semantic-backfill-original", ("g1",), gold_count=4,
            search_mode="semantic",
        ),
    ]
    graph = [
        {
            "dataset": "pasa", "split": "auto_train", "role": "training",
            "query_id": "q1", "query": "q1", "routing_label": "beneficial",
            "gold_association_count": 4, "anchor_gold_hit_ids": ["g1"],
            "pre_graph_gold_hit_ids": ["g1", "g3"],
            "graph_gold_hit_ids": ["g4"], "graph_marginal_gold_hit_ids": ["g4"],
            "graph_marginal_recall": 0.25, "seed_count": 1,
            "graph_action_count": 2, "search_api_calls": 6,
        },
        {
            "dataset": "pasa", "split": "auto_train", "role": "training",
            "query_id": "q2", "query": "q2", "routing_label": "beneficial",
            "gold_association_count": 4, "anchor_gold_hit_ids": ["g1"],
            "pre_graph_gold_hit_ids": ["g1"], "graph_gold_hit_ids": ["g2"],
            "graph_marginal_gold_hit_ids": ["g2"],
            "graph_marginal_recall": 0.25, "seed_count": 1,
            "graph_action_count": 2, "search_api_calls": 6,
        },
        {
            "dataset": "pasa", "split": "auto_train", "role": "training",
            "query_id": "q3", "query": "q3", "routing_label": "not_beneficial",
            "gold_association_count": 4, "anchor_gold_hit_ids": [],
            "pre_graph_gold_hit_ids": ["g2"], "graph_gold_hit_ids": ["g2"],
            "graph_marginal_gold_hit_ids": [], "graph_marginal_recall": 0.0,
            "seed_count": 1, "graph_action_count": 2, "search_api_calls": 6,
        },
    ]

    result = analyze_structured_graph_marginals(
        frozen_rows=frozen,
        lexical_labels=lexical,
        semantic_labels=semantic,
        graph_labels=graph,
    )

    assert result["structured"]["new_gold_hit_count"] == 2
    assert result["structured"]["positive_fold_count"] == 2
    assert result["structured"]["incremental_action_count"] == 12
    assert result["graph"]["new_gold_hit_count"] == 2
    assert result["graph"]["positive_fold_count"] == 2
    assert result["graph"]["incremental_action_count"] == 6
