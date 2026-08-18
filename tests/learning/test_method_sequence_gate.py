from __future__ import annotations

from paper_search.learning.method_sequence_gate import (
    MethodSequenceGate,
    QueryMethodEvidence,
    assess_method_sequence,
    build_method_sequence_evidence,
)


def _row(index: int) -> QueryMethodEvidence:
    semantic_hits = ("g2",) if index < 4 else ()
    graph_hits = ("g3",) if index == 0 else ()
    return QueryMethodEvidence(
        query_id=f"q{index}",
        fold=index % 3 + 1,
        intent_family="request" if index % 2 else "which_what",
        length_bucket=f"q{index % 4 + 1}",
        gold_count_bucket="3_plus",
        gold_association_count=3,
        base_gold_hit_ids=("g1",),
        semantic_gold_hit_ids=semantic_hits,
        graph_gold_hit_ids=graph_hits,
        base_action_count=2,
        semantic_action_count=1,
        graph_action_count=5,
    )


def test_sequence_gate_promotes_stable_efficient_semantic_only() -> None:
    gate = MethodSequenceGate(
        minimum_beneficial_queries=3,
        minimum_positive_folds=3,
        minimum_positive_intent_families=2,
        minimum_positive_length_buckets=3,
        minimum_positive_gold_count_buckets=1,
        minimum_efficiency_ratio_to_baseline=1.0,
    )

    decision = assess_method_sequence([_row(index) for index in range(6)], gate)

    assert decision.semantic.promote is True
    assert decision.semantic.beneficial_query_count == 4
    assert decision.semantic.positive_fold_count == 3
    assert decision.semantic.marginal_efficiency_ratio_to_baseline > 1.0
    assert decision.graph.promote is False
    assert "minimum_beneficial_queries" in decision.graph.failed_conditions
    assert "minimum_efficiency_ratio_to_baseline" in decision.graph.failed_conditions


def test_sequence_gate_measures_graph_after_semantic_union() -> None:
    rows = [_row(index) for index in range(6)]
    gate = MethodSequenceGate(
        minimum_beneficial_queries=1,
        minimum_positive_folds=1,
        minimum_positive_intent_families=1,
        minimum_positive_length_buckets=1,
        minimum_positive_gold_count_buckets=1,
        minimum_efficiency_ratio_to_baseline=0.0,
    )

    decision = assess_method_sequence(rows, gate)

    assert decision.semantic.new_gold_hit_count == 4
    assert decision.graph.new_gold_hit_count == 1
    assert decision.graph.macro_recall_before == decision.semantic.macro_recall_after


def test_build_evidence_joins_frozen_strata_and_method_labels() -> None:
    def label(action_id: str, hits: list[str]) -> dict[str, object]:
        return {
            "dataset": "pasa",
            "split": "auto_train",
            "role": "training",
            "query_id": "q1",
            "query": "query one",
            "provider": "openalex",
            "action": {
                "action_id": action_id,
                "action_type": "text_search",
                "text": action_id,
                "origin": "deterministic_rule",
                "provider_hint": "openalex",
                "search_mode": (
                    "semantic" if "semantic" in action_id else "lexical"
                ),
            },
            "retrieval_status": "available",
            "gold_association_count": 3,
            "gold_hit_ids": hits,
            "gold_hit_count": len(hits),
            "action_recall": len(hits) / 3,
            "novel_over_anchor_hit_count": len(hits),
            "error_codes": [],
        }

    rows = build_method_sequence_evidence(
        frozen_rows=[
            {
                "query_id": "q1",
                "fold": 2,
                "intent_family": "request",
                "length_bucket": "q3",
                "gold_count_bucket": "3_plus",
            }
        ],
        base_labels=[label("base-1", ["g1"]), label("base-2", [])],
        semantic_labels=[label("semantic-backfill-original", ["g2"])],
        graph_labels=[
            {
                "dataset": "pasa",
                "split": "auto_train",
                "role": "training",
                "query_id": "q1",
                "query": "query one",
                "routing_label": "beneficial",
                "gold_association_count": 3,
                "anchor_gold_hit_ids": [],
                "pre_graph_gold_hit_ids": [],
                "graph_gold_hit_ids": ["g3"],
                "graph_marginal_gold_hit_ids": ["g3"],
                "graph_marginal_recall": 1 / 3,
                "seed_count": 2,
                "graph_action_count": 4,
                "search_api_calls": 6,
            }
        ],
    )

    assert rows == [
        QueryMethodEvidence(
            query_id="q1",
            fold=2,
            intent_family="request",
            length_bucket="q3",
            gold_count_bucket="3_plus",
            gold_association_count=3,
            base_gold_hit_ids=("g1",),
            semantic_gold_hit_ids=("g2",),
            graph_gold_hit_ids=("g3",),
            base_action_count=2,
            semantic_action_count=1,
            graph_action_count=4,
        )
    ]
