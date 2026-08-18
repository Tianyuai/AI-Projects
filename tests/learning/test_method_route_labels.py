from __future__ import annotations

from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.method_route_labels import (
    paired_semantic_method_labels,
    semantic_method_labels,
)
from paper_search.learning.method_route_labels import graph_method_route_labels
from paper_search.learning.graph_method_labels import GraphMethodLabel
from paper_search.learning.provider_action_labels import ProviderActionLabel


def _row(*, query_id: str, status: str, novel: int | None) -> ProviderActionLabel:
    available = status == "available"
    return ProviderActionLabel(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id=query_id,
        query=f"query {query_id}",
        provider="openalex",
        action=PolicyActionCandidate(
            action_id=f"semantic-{query_id}",
            action_type="text_search",
            text=f"query {query_id}",
            origin="deterministic_rule",
            provider_hint="openalex",
            search_mode="semantic",
        ),
        retrieval_status=status,
        gold_association_count=2 if available else None,
        gold_hit_ids=("doi:x",) if available and novel else (),
        gold_hit_count=1 if available and novel else (0 if available else None),
        action_recall=0.5 if available and novel else (0.0 if available else None),
        novel_over_anchor_hit_count=novel,
    )


def test_semantic_method_labels_use_marginal_gold_and_keep_unavailable() -> None:
    labels = semantic_method_labels(
        [
            _row(query_id="positive", status="available", novel=1),
            _row(query_id="negative", status="available", novel=0),
            _row(query_id="unavailable", status="unavailable", novel=None),
        ]
    )

    assert [row.routing_label for row in labels] == [
        "not_beneficial",
        "beneficial",
        "unavailable",
    ]
    assert labels[1].marginal_gold_hit_count == 1


def test_semantic_method_labels_ignore_lexical_and_other_provider() -> None:
    semantic = _row(query_id="kept", status="available", novel=1)
    lexical = semantic.model_copy(
        update={
            "action": semantic.action.model_copy(update={"search_mode": "lexical"})
        }
    )
    s2 = semantic.model_copy(update={"provider": "semantic_scholar"})

    assert semantic_method_labels([lexical, s2]) == []


def test_semantic_method_labels_aggregate_repeated_receipts_per_query() -> None:
    negative = _row(query_id="same", status="available", novel=0)
    positive = _row(query_id="same", status="available", novel=1).model_copy(
        update={"action": negative.action.model_copy(update={"action_id": "backfill"})}
    )

    labels = semantic_method_labels([negative, positive])

    assert len(labels) == 1
    assert labels[0].routing_label == "beneficial"
    assert labels[0].marginal_gold_hit_count == 1
    assert labels[0].method_action_count == 2
    assert labels[0].search_api_calls == 2


def test_semantic_method_labels_exclude_historical_invalid_request_receipt() -> None:
    rejected = _row(query_id="same", status="available", novel=0).model_copy(
        update={"error_codes": ("invalid_request",)}
    )
    valid = _row(query_id="same", status="available", novel=1).model_copy(
        update={"action": rejected.action.model_copy(update={"action_id": "backfill"})}
    )

    [label] = semantic_method_labels([rejected, valid])

    assert label.routing_label == "beneficial"
    assert label.method_action_count == 1
    assert label.search_api_calls == 1


def test_paired_semantic_labels_use_union_of_all_production_baseline_hits() -> None:
    semantic = _row(query_id="same", status="available", novel=2).model_copy(
        update={"gold_hit_ids": ("doi:a", "doi:c"), "gold_hit_count": 2, "action_recall": 1.0}
    )
    lexical_a = semantic.model_copy(
        update={
            "action": semantic.action.model_copy(
                update={"action_id": "lexical-a", "search_mode": "lexical"}
            ),
            "gold_hit_ids": ("doi:a",),
            "gold_hit_count": 1,
            "action_recall": 0.5,
            "novel_over_anchor_hit_count": 0,
        }
    )
    lexical_b = lexical_a.model_copy(
        update={
            "action": lexical_a.action.model_copy(update={"action_id": "lexical-b"}),
            "gold_hit_ids": ("doi:b",),
        }
    )

    [label] = paired_semantic_method_labels(
        baseline_rows=[lexical_a, lexical_b],
        semantic_rows=[semantic],
    )

    assert label.routing_label == "beneficial"
    assert label.marginal_gold_hit_count == 1
    assert label.marginal_recall == 0.5
    assert label.method_action_count == 1


def test_graph_method_labels_map_without_changing_reward() -> None:
    source = GraphMethodLabel(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id="q1",
        query="citation graph retrieval",
        routing_label="beneficial",
        gold_association_count=2,
        graph_marginal_gold_hit_ids=("doi:x",),
        graph_marginal_recall=0.5,
        seed_count=4,
        graph_action_count=8,
        search_api_calls=8,
    )

    [label] = graph_method_route_labels([source])

    assert label.method == "graph"
    assert label.marginal_gold_hit_count == 1
    assert label.seed_count == 4
    assert label.method_action_count == 8
