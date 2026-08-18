from __future__ import annotations

from paper_search.domain.models import Paper, UsageActual
from paper_search.learning.structured_graph_dataset import build_structured_method_labels
from paper_search.recall_experiments.contracts import (
    CitationExpandAction,
    CitationExpandPayload,
    RecallActionBatch,
    RetrievalActionResult,
    TextSearchAction,
    TextSearchPayload,
)


def _paper(identifier: str) -> Paper:
    return Paper(canonical_id=identifier, title=identifier)


def test_builds_semantic_and_graph_marginal_labels_from_same_receipt() -> None:
    actions = RecallActionBatch(
        actions=[
            TextSearchAction(
                action_id="anchor",
                strategy="structured:anchor-original",
                action_type="text_search",
                payload=TextSearchPayload(query_text="query", search_mode="lexical"),
            ),
            TextSearchAction(
                action_id="semantic",
                strategy="structured:anchor-semantic",
                action_type="text_search",
                payload=TextSearchPayload(query_text="query", search_mode="semantic"),
            ),
            CitationExpandAction(
                action_id="graph",
                strategy="structured:openalex-citation-graph",
                action_type="citation_expand",
                payload=CitationExpandPayload(
                    seed_canonical_id="openalex:W1",
                    direction="references",
                    limit=50,
                ),
            ),
        ]
    )
    results = [
        RetrievalActionResult(
            action_id="anchor",
            action_type="text_search",
            hits=[_paper("doi:10.1000/a")],
        ),
        RetrievalActionResult(
            action_id="semantic",
            action_type="text_search",
            hits=[_paper("doi:10.1000/b")],
        ),
        RetrievalActionResult(
            action_id="graph",
            action_type="citation_expand",
            hits=[_paper("doi:10.1000/c")],
            usage=UsageActual(search_api_calls=1),
        ),
    ]

    semantic, graph = build_structured_method_labels(
        dataset="pasa",
        split="auto_dev",
        role="development",
        query_id="q1",
        query="query",
        gold_paper_ids=["doi:10.1000/a", "doi:10.1000/b", "doi:10.1000/c"],
        actions=actions,
        results=results,
    )

    assert semantic.routing_label == "beneficial"
    assert semantic.marginal_gold_hit_count == 1
    assert graph.routing_label == "beneficial"
    assert graph.graph_marginal_gold_hit_ids == ("doi:10.1000/c",)
    assert graph.seed_count == 1
    assert graph.search_api_calls == 1
