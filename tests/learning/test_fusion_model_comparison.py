from __future__ import annotations

from paper_search.domain.models import Paper
from paper_search.learning.cpu_document_ranker import (
    DocumentRankingQuery,
    build_document_candidates,
)
from paper_search.learning.fusion_model_comparison import (
    evaluate_fusion_model_set,
)


class _IdentityRanker:
    def rank(self, query: str, candidates: list[object]) -> list[object]:
        return list(candidates)


class _ReverseRanker:
    def rank(self, query: str, candidates: list[object]) -> list[object]:
        return list(reversed(candidates))


def _queries() -> list[tuple[int, DocumentRankingQuery]]:
    rows: list[tuple[int, DocumentRankingQuery]] = []
    for fold in (1, 2, 3):
        negative = Paper(
            canonical_id=f"openalex:W{fold}1",
            openalex_id=f"W{fold}1",
            title="negative",
            is_retracted=False,
        )
        positive = Paper(
            canonical_id=f"openalex:W{fold}2",
            openalex_id=f"W{fold}2",
            title="positive",
            is_retracted=False,
        )
        rows.append(
            (
                fold,
                DocumentRankingQuery(
                    query_id=f"q{fold}",
                    query="positive",
                    gold_paper_ids=[f"openalex:W{fold}2"],
                    candidates=build_document_candidates(
                        [("lexical", [negative, positive])]
                    ),
                ),
            )
        )
    return rows


def test_fusion_model_set_compares_models_and_checks_reloaded_parity() -> None:
    report = evaluate_fusion_model_set(
        _queries(),
        rankers={
            "B0": _IdentityRanker(),
            "F5-production": _IdentityRanker(),
            "F5-candidate": _ReverseRanker(),
        },
        replay_rankers={
            "B0": _IdentityRanker(),
            "F5-production": _IdentityRanker(),
            "F5-candidate": _ReverseRanker(),
        },
        gate_model="F5-candidate",
        production_model="F5-production",
        cutoffs=(1, 2, 10, 20, 50),
    )

    assert report["models"]["B0"]["macro_recall_at"][1] == 0.0
    assert report["models"]["F5-candidate"]["macro_recall_at"][1] == 1.0
    assert report["auto_dev_gate"]["passed"] is True
    assert report["auto_dev_gate"]["production_model"] == "F5-production"
    assert report["auto_dev_gate"]["deltas_vs_production"]["mrr"] > 0.0
    assert report["live_replay_gate"]["passed"] is True
    assert report["live_replay_gate"]["identical_query_count"] == 3
    assert report["candidate_pool_identity_unchanged"] is True
    assert report["test_partition_touched"] is False


def test_fusion_model_set_rejects_candidate_that_regresses_vs_production() -> None:
    report = evaluate_fusion_model_set(
        _queries(),
        rankers={
            "B0": _IdentityRanker(),
            "F5-production": _ReverseRanker(),
            "F5-candidate": _IdentityRanker(),
        },
        replay_rankers={
            "B0": _IdentityRanker(),
            "F5-production": _ReverseRanker(),
            "F5-candidate": _IdentityRanker(),
        },
        gate_model="F5-candidate",
        production_model="F5-production",
        cutoffs=(1, 2, 10, 20, 50),
    )

    assert report["auto_dev_gate"]["deltas_vs_B0"][1] == 0.0
    assert report["auto_dev_gate"]["deltas_vs_production"]["mrr"] < 0.0
    assert report["auto_dev_gate"]["passed"] is False
