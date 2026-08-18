from __future__ import annotations

import pytest

from paper_search.evaluation.dev_closed_loop import (
    aggregate_development_closed_loop,
    score_development_query,
)


def test_development_closed_loop_separates_oracle_from_final_recall() -> None:
    result = score_development_query(
        query_id="dev-1",
        gold_paper_ids=["arxiv:1", "arxiv:2"],
        candidate_paper_ids=["arxiv:1", "arxiv:2", "openalex:W3"],
        final_paper_ids=["arxiv:1"],
    )

    assert result.candidate_oracle_recall == 1.0
    assert result.final_recall == 0.5
    assert result.oracle_final_gap == 0.5


def test_development_closed_loop_aggregates_macro_metrics() -> None:
    summary = aggregate_development_closed_loop(
        [
            score_development_query(
                query_id="dev-1",
                gold_paper_ids=["arxiv:1"],
                candidate_paper_ids=["arxiv:1"],
                final_paper_ids=["arxiv:1"],
            ),
            score_development_query(
                query_id="dev-2",
                gold_paper_ids=["arxiv:2"],
                candidate_paper_ids=[],
                final_paper_ids=[],
            ),
        ]
    )

    assert summary.query_count == 2
    assert summary.completed_query_count == 2
    assert summary.candidate_oracle_macro_recall == 0.5
    assert summary.final_macro_recall == 0.5
    assert summary.oracle_final_macro_gap == 0.0


def test_development_closed_loop_rejects_empty_gold() -> None:
    with pytest.raises(ValueError, match="gold"):
        score_development_query(
            query_id="dev-1",
            gold_paper_ids=[],
            candidate_paper_ids=[],
            final_paper_ids=[],
        )
