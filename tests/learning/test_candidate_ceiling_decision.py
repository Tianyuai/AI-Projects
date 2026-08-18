from __future__ import annotations

from paper_search.learning.candidate_ceiling_decision import (
    CandidateCeilingBatchEvidence,
    ProviderCeilingBatchEvidence,
    assess_candidate_ceiling,
    summarize_candidate_ceiling_batch,
)
from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.policy import BoundedQueryPolicy, RuleActionScorer
from paper_search.learning.provider_action_labels import ProviderActionLabel


def _label(
    query_id: str,
    action_id: str,
    hits: tuple[str, ...],
    *,
    origin: str = "deterministic_rule",
) -> ProviderActionLabel:
    return ProviderActionLabel(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id=query_id,
        query="graph diffusion retrieval",
        provider="openalex",
        action=PolicyActionCandidate(
            action_id=action_id,
            action_type="text_search",
            text=(
                "graph diffusion retrieval"
                if origin == "original_query"
                else action_id.replace("-", " ")
            ),
            origin=origin,
            provider_hint="openalex",
        ),
        retrieval_status="available",
        gold_association_count=2,
        gold_hit_ids=hits,
        gold_hit_count=len(hits),
        action_recall=len(hits) / 2,
        novel_over_anchor_hit_count=(
            len(hits) if origin != "original_query" else 0
        ),
    )


def test_batch_summary_compares_current_top3_oracle3_and_full_pool() -> None:
    labels = []
    for query_id in ("q-1", "q-2"):
        labels.extend(
            [
                _label(query_id, "anchor", (f"{query_id}-g1",), origin="original_query"),
                _label(query_id, "bad-a", ()),
                _label(query_id, "bad-b", ()),
                _label(query_id, "zz-good", (f"{query_id}-g2",)),
            ]
        )
    policy = BoundedQueryPolicy(RuleActionScorer(), confidence_threshold=0.0)

    summary = summarize_candidate_ceiling_batch(
        labels,
        batch_id="fold-1",
        policy=policy,
    )

    [provider] = summary.providers
    assert provider.provider == "openalex"
    assert provider.query_count == 2
    assert provider.current_top3_macro_recall == 0.5
    assert provider.oracle_at_3_macro_recall == 1.0
    assert provider.all_candidate_macro_recall == 1.0
    assert provider.preference_query_count == 2


def _batch(batch_id: str, *, gap: float, preferences: int) -> CandidateCeilingBatchEvidence:
    providers = []
    for provider in ("openalex", "semantic_scholar"):
        providers.append(
            ProviderCeilingBatchEvidence(
                provider=provider,
                query_count=10,
                current_top3_macro_recall=0.10,
                oracle_at_3_macro_recall=0.10 + gap,
                all_candidate_macro_recall=0.10 + gap,
                preference_query_count=preferences,
                unavailable_action_count=0,
            )
        )
    return CandidateCeilingBatchEvidence(batch_id=batch_id, providers=providers)


def test_decision_requires_all_three_batches_before_recommending_change() -> None:
    decision = assess_candidate_ceiling(
        [_batch("fold-1", gap=0.10, preferences=10)],
    )

    assert decision.status == "insufficient_evidence"
    assert decision.ranking_optimization_supported_providers == ()


def test_decision_supports_cpu_ranking_only_after_consistent_prefixed_evidence() -> None:
    decision = assess_candidate_ceiling(
        [
            _batch("fold-1", gap=0.05, preferences=8),
            _batch("fold-2", gap=0.04, preferences=7),
            _batch("fold-3", gap=0.00, preferences=6),
        ],
    )

    assert decision.status == "complete"
    assert set(decision.ranking_optimization_supported_providers) == {
        "openalex",
        "semantic_scholar",
    }
    assert decision.candidate_generation_change_supported is False
    assert decision.minimum_preference_queries == 20
    assert decision.minimum_pooled_macro_gap == 0.03
