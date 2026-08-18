from __future__ import annotations

import pytest

from paper_search.domain.models import Paper
from paper_search.evaluation.saved_receipt_replay import (
    aggregate_saved_replays,
    replay_saved_query,
)


def _paper(identifier: str, *, arxiv_id: str | None = None) -> Paper:
    return Paper(
        canonical_id=identifier,
        title=identifier,
        openalex_id=identifier if identifier.startswith("openalex:") else None,
        arxiv_id=arxiv_id,
        is_retracted=False,
    )


def test_saved_replay_uses_action_level_rrf_and_keeps_oracle_pre_truncation() -> None:
    shared = _paper("openalex:W1", arxiv_id="1234.5678")
    lexical_only = _paper("openalex:W2")
    semantic_only = _paper("openalex:W3")
    filler = _paper("openalex:W4")

    replay = replay_saved_query(
        query_id="dev-1",
        query="papers about retrieval",
        gold_paper_ids=["arxiv:1234.5678", "openalex:W3"],
        action_results=[
            ("lexical", [filler, shared, lexical_only]),
            ("semantic", [semantic_only, shared]),
        ],
        cutoffs=(1, 2),
    )

    assert replay.ranked_paper_ids[0] == "arxiv:1234.5678"
    assert replay.candidate_oracle_recall == 1.0
    assert replay.recall_at == {1: 0.5, 2: 1.0}


def test_saved_replay_aggregates_macro_metrics_by_fold() -> None:
    hit = replay_saved_query(
        query_id="dev-1",
        query="retrieval",
        gold_paper_ids=["openalex:W1"],
        action_results=[("a", [_paper("openalex:W1")])],
        cutoffs=(1,),
        fold=1,
    )
    miss = replay_saved_query(
        query_id="dev-2",
        query="retrieval",
        gold_paper_ids=["openalex:W2"],
        action_results=[("a", [_paper("openalex:W3")])],
        cutoffs=(1,),
        fold=2,
    )

    result = aggregate_saved_replays([hit, miss], cutoffs=(1,))

    assert result.overall.candidate_oracle_macro_recall == pytest.approx(0.5)
    assert result.overall.macro_recall_at == {1: pytest.approx(0.5)}
    assert result.by_fold[1].macro_recall_at == {1: 1.0}
    assert result.by_fold[2].macro_recall_at == {1: 0.0}
