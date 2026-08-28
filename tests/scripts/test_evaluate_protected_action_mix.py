from __future__ import annotations

from paper_search.domain.models import Paper, ProviderResult, UsageActual
from scripts.evaluate_protected_action_mix import (
    paper_identity_from_record,
    paper_identity_record,
    offline_confirmation_decision,
    round_robin_cap_action_results,
)


def _result(prefix: str, count: int) -> ProviderResult[list[Paper]]:
    return ProviderResult(
        data=[Paper(canonical_id=f"{prefix}:{index}", title=f"paper {index}") for index in range(count)],
        usage=UsageActual(),
        provenance={
            "provider": "fixture",
            "endpoint": "/fixture",
            "model_id": "fixture-v1",
            "requested_at": "2026-08-28T00:00:00Z",
            "response_hash": "sha256:" + "0" * 64,
        },
        cache_hit=True,
        latency_ms=0,
        errors=[],
    )


def test_round_robin_cap_preserves_each_selected_action_prefix() -> None:
    capped = round_robin_cap_action_results(
        {"source-b": _result("b", 4), "source-a": _result("a", 4)},
        5,
    )

    assert [paper.canonical_id for paper in capped["source-a"].data] == [
        "a:0",
        "a:1",
    ]
    assert [paper.canonical_id for paper in capped["source-b"].data] == [
        "b:0",
        "b:1",
        "b:2",
    ]


def test_offline_decision_requires_zero_regression_and_one_strict_gain() -> None:
    metrics = {
        "query_count": 24,
        "baseline_gold_pool_hit_query_count": 4,
        "hybrid_gold_pool_hit_query_count": 7,
        "hybrid_gold_pool_regressed_query_count": 0,
        "baseline_top5_hit_query_count": 1,
        "hybrid_top5_hit_query_count": 3,
        "baseline_top10_hit_query_count": 1,
        "hybrid_top10_hit_query_count": 3,
        "baseline_top20_hit_query_count": 2,
        "hybrid_top20_hit_query_count": 4,
        "stratum_regression_count": 0,
        "f5_query_count": 24,
        "within_action_budget_query_count": 24,
    }

    passed = offline_confirmation_decision(metrics)
    regressed = offline_confirmation_decision(
        {**metrics, "hybrid_gold_pool_regressed_query_count": 1}
    )

    assert passed["passed"] is True
    assert passed["decision"] == "eligible_for_new_disjoint_live_confirmation"
    assert regressed["passed"] is False
    assert "gold_pool_query_regression" in regressed["failed_gates"]


def test_ranked_pool_identity_round_trip_preserves_aliases() -> None:
    paper = Paper(
        canonical_id="doi:10.48550/arxiv.2401.12345",
        title="Alias-bearing paper",
        doi="10.48550/arxiv.2401.12345",
        arxiv_id="2401.12345",
        openalex_id="W1234567890",
    )

    restored = paper_identity_from_record(paper_identity_record(paper))

    assert restored.canonical_id == paper.canonical_id
    assert restored.doi == paper.doi
    assert restored.arxiv_id == paper.arxiv_id
    assert restored.openalex_id == paper.openalex_id
