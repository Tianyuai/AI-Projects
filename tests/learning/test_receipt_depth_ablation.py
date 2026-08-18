from __future__ import annotations

from paper_search.domain.models import Paper
from paper_search.learning.receipt_depth_ablation import analyze_receipt_depths


def _paper(identifier: str) -> Paper:
    return Paper(
        canonical_id=identifier,
        title=identifier,
        openalex_id=identifier,
        is_retracted=False,
    )


def test_depth_ablation_requires_per_query_gold_equivalence_for_zero_loss() -> None:
    report = analyze_receipt_depths(
        queries=[
            {
                "query_id": "q1",
                "fold": 1,
                "gold_paper_ids": ["openalex:W1"],
                "actions": [
                    ("a1", [_paper("openalex:W9"), _paper("openalex:W1")]),
                    ("a2", [_paper("openalex:W9")]),
                ],
            },
            {
                "query_id": "q2",
                "fold": 2,
                "gold_paper_ids": ["openalex:W2"],
                "actions": [("a1", [_paper("openalex:W2")])],
            },
        ],
        depths=(1, 2),
    )

    depth1, depth2 = report["depths"]
    assert depth1["zero_recall_loss"] is False
    assert depth2["zero_recall_loss"] is True
    assert depth2["duplicate_rate"] > 0
    assert report["action_positions"][1]["overlap_candidate_count"] == 1
