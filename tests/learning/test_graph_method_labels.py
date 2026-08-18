from __future__ import annotations

from paper_search.domain.models import Paper
from paper_search.learning.graph_method_labels import build_graph_method_label


def _paper(identifier: str) -> Paper:
    return Paper(canonical_id=identifier, title=identifier)


def test_graph_method_label_uses_only_hits_novel_over_pre_graph_pool() -> None:
    label = build_graph_method_label(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id="q-1",
        query="find graph papers",
        gold_paper_ids=["doi:10.1000/a", "doi:10.1000/b"],
        anchor_hits=[_paper("doi:10.1000/a")],
        pre_graph_hits=[_paper("doi:10.1000/a")],
        graph_hits=[_paper("doi:10.1000/a"), _paper("doi:10.1000/b")],
        seed_count=2,
        graph_action_count=4,
        graph_infrastructure_failure=False,
        search_api_calls=6,
    )

    assert label.routing_label == "beneficial"
    assert label.pre_graph_gold_hit_ids == ("doi:10.1000/a",)
    assert label.graph_marginal_gold_hit_ids == ("doi:10.1000/b",)
    assert label.graph_marginal_recall == 0.5


def test_graph_method_label_marks_failed_graph_receipts_unavailable() -> None:
    label = build_graph_method_label(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id="q-1",
        query="find graph papers",
        gold_paper_ids=["doi:10.1000/a"],
        anchor_hits=[],
        pre_graph_hits=[],
        graph_hits=[],
        seed_count=1,
        graph_action_count=2,
        graph_infrastructure_failure=True,
        search_api_calls=0,
    )

    assert label.routing_label == "unavailable"
    assert label.graph_marginal_gold_hit_ids == ()
