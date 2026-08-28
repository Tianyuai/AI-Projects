from __future__ import annotations

import asyncio

from paper_search.domain.models import Paper, QuerySpec
from paper_search.learning.adaptive_openalex_recall import (
    assess_openalex_recall_confidence,
    attribute_openalex_gold_miss,
    select_openalex_supplement_actions,
)
from paper_search.learning.cpu_document_ranker import (
    DocumentCandidateEvidence,
    DocumentRankingQuery,
)
from paper_search.recall_experiments.contracts import (
    RecallGenerationContext,
    TextSearchAction,
    TextSearchPayload,
)


def _candidate(
    paper_id: str,
    title: str,
    *,
    sources: tuple[str, ...],
    source_ranks: dict[str, int],
) -> DocumentCandidateEvidence:
    return DocumentCandidateEvidence(
        paper=Paper(
            canonical_id=paper_id,
            title=title,
            sources=list(sources),
        ),
        baseline_score=sum(1.0 / (60 + rank) for rank in source_ranks.values()),
        source_ranks=source_ranks,
    )


def test_pasa_only_gold_is_diagnostic_evidence_not_an_online_hit() -> None:
    query = DocumentRankingQuery(
        query_id="q-pasa-only",
        query="Find graph diffusion retrieval papers",
        gold_paper_ids=["arxiv:2301.00001"],
        candidates=[
            _candidate(
                "arxiv:2301.00001",
                "Graph Diffusion for Retrieval",
                sources=("pasa_paper_database",),
                source_ranks={"pasa-local-original": 1},
            ),
            _candidate(
                "openalex:W2",
                "A Different Retrieval Paper",
                sources=("openalex",),
                source_ranks={"production-topic": 1},
            ),
        ],
    )

    attribution = attribute_openalex_gold_miss(query)

    assert attribution.category == "pasa_only_gold"
    assert attribution.online_gold_hit_count == 0
    assert attribution.pasa_only_gold_count == 1
    assert attribution.pasa_used_for_action_generation is False


def test_same_title_with_different_identity_is_flagged_for_alias_audit() -> None:
    query = DocumentRankingQuery(
        query_id="q-alias",
        query="Find graph diffusion retrieval papers",
        gold_paper_ids=["arxiv:2301.00001"],
        candidates=[
            _candidate(
                "arxiv:2301.00001",
                "Graph Diffusion for Retrieval",
                sources=("pasa_paper_database",),
                source_ranks={"pasa-local-original": 1},
            ),
            _candidate(
                "openalex:W999",
                "Graph diffusion for retrieval",
                sources=("openalex",),
                source_ranks={"production-topic": 2},
            ),
        ],
    )

    attribution = attribute_openalex_gold_miss(query)

    assert attribution.category == "identity_mismatch_suspected"
    assert attribution.title_alias_candidate_count == 1


def test_online_gold_hit_is_not_reported_as_a_recall_miss() -> None:
    query = DocumentRankingQuery(
        query_id="q-online-hit",
        query="Find graph diffusion retrieval papers",
        gold_paper_ids=["openalex:W1"],
        candidates=[
            _candidate(
                "openalex:W1",
                "Graph Diffusion for Retrieval",
                sources=("openalex",),
                source_ranks={"production-topic": 1},
            )
        ],
    )

    attribution = attribute_openalex_gold_miss(query)

    assert attribution.category == "online_gold_hit"
    assert attribution.online_gold_hit_count == 1


def test_deep_supported_pool_with_facet_coverage_is_adequate() -> None:
    candidates = [
        _candidate(
            f"openalex:W{index}",
            (
                "Vision Transformer for Image Classification"
                if index == 0
                else f"Image Classification Study {index}"
            ),
            sources=("openalex",),
            source_ranks=(
                {"production-topic": 1, "production-context": 2}
                if index == 0
                else {"production-topic": index + 1}
            ),
        )
        for index in range(60)
    ]
    spec = QuerySpec(
        original_query="Find vision transformer papers",
        research_goal="Find vision transformer papers",
        methods=["vision transformer"],
        tasks=["image classification"],
    )

    decision = assess_openalex_recall_confidence(spec, candidates)

    assert decision.low_confidence is False
    assert decision.reason_codes == []
    assert decision.candidate_count == 60
    assert decision.cross_action_supported_candidate_count == 1
    assert decision.covered_facet_count == decision.required_facet_count == 2


def test_shallow_pool_triggers_low_yield_supplement() -> None:
    spec = QuerySpec(
        original_query="Find graph retrieval papers",
        research_goal="Find graph retrieval papers",
    )
    candidates = [
        _candidate(
            f"openalex:W{index}",
            f"Graph Retrieval Study {index}",
            sources=("openalex",),
            source_ranks={"production-topic": index + 1},
        )
        for index in range(10)
    ]

    decision = assess_openalex_recall_confidence(spec, candidates)

    assert decision.low_confidence is True
    assert "low_yield" in decision.reason_codes


def test_missing_facets_and_action_disagreement_trigger_supplement() -> None:
    spec = QuerySpec(
        original_query="Find ImageNet-C vision transformer papers",
        research_goal="Find ImageNet-C vision transformer papers",
        methods=["vision transformer"],
        datasets=["ImageNet-C"],
    )
    candidates = [
        _candidate(
            f"openalex:W{index}",
            f"Generic Neural Network Study {index}",
            sources=("openalex",),
            source_ranks={
                "production-topic" if index % 2 == 0 else "production-context":
                index + 1
            },
        )
        for index in range(60)
    ]

    decision = assess_openalex_recall_confidence(spec, candidates)

    assert decision.low_confidence is True
    assert "facet_gap" in decision.reason_codes
    assert "cross_action_disagreement" in decision.reason_codes


def test_unconstrained_deep_but_semantically_misaligned_pool_triggers_supplement() -> None:
    spec = QuerySpec(
        original_query="Find adaptive rounding per-neuron quantization papers",
        research_goal="Find adaptive rounding per-neuron quantization papers",
    )
    candidates = [
        _candidate(
            f"openalex:W{index}",
            f"Generic Neural Network Compression Study {index}",
            sources=("openalex",),
            source_ranks={
                "production-topic": index + 1,
                "production-context": index + 2,
            },
        )
        for index in range(60)
    ]

    decision = assess_openalex_recall_confidence(spec, candidates)

    assert decision.low_confidence is True
    assert "low_query_alignment" in decision.reason_codes
    assert decision.query_aligned_candidate_count == 0
    assert decision.minimum_query_aligned_candidate_count > 0


def test_missing_facet_alone_triggers_even_with_cross_action_support() -> None:
    spec = QuerySpec(
        original_query="Find ImageNet-C vision transformer papers",
        research_goal="Find ImageNet-C vision transformer papers",
        methods=["vision transformer"],
        datasets=["ImageNet-C"],
    )
    candidates = [
        _candidate(
            f"openalex:W{index}",
            f"Generic Robust Classification Study {index}",
            sources=("openalex",),
            source_ranks={
                "production-topic": index + 1,
                "production-context": index + 2,
            },
        )
        for index in range(60)
    ]

    decision = assess_openalex_recall_confidence(spec, candidates)

    assert decision.low_confidence is True
    assert "facet_gap" in decision.reason_codes
    assert "cross_action_disagreement" not in decision.reason_codes


def test_supplement_selector_is_gold_blind_deduplicated_and_budget_bounded() -> None:
    query = "Find ImageNet-C vision transformer papers"
    spec = QuerySpec(
        original_query=query,
        research_goal=query,
        methods=["vision transformer"],
        datasets=["ImageNet-C"],
    )
    context = RecallGenerationContext(
        query_id="q-supplement",
        original_query=query,
        query_spec=spec,
    )
    first_round = [
        TextSearchAction(
            action_id="production-original",
            strategy="production",
            action_type="text_search",
            payload=TextSearchPayload(query_text=query, search_mode="lexical"),
        )
    ]
    decision = assess_openalex_recall_confidence(spec, [])

    selected = asyncio.run(
        select_openalex_supplement_actions(
            context,
            frozen_query_specs={"q-supplement": spec},
            first_round_actions=first_round,
            decision=decision.model_copy(
                update={
                    "reason_codes": ["facet_gap", "cross_action_disagreement"]
                }
            ),
            max_total_actions=3,
        )
    )

    assert [action.action_id for action in selected.actions] == [
        "high-recall-entity-lexical",
        "high-recall-context-semantic",
    ]
    identities = {
        (action.payload.search_mode, action.payload.query_text.casefold())
        for action in [*first_round, *selected.actions]
    }
    assert len(identities) == 3
    assert len(first_round) + len(selected.actions) == 3


def test_adequate_pool_does_not_schedule_supplemental_actions() -> None:
    query = "Find graph retrieval papers"
    spec = QuerySpec(original_query=query, research_goal=query)
    context = RecallGenerationContext(
        query_id="q-adequate",
        original_query=query,
        query_spec=spec,
    )
    adequate = assess_openalex_recall_confidence(
        spec,
        [
            _candidate(
                f"openalex:W{index}",
                f"Graph Retrieval Study {index}",
                sources=("openalex",),
                source_ranks=(
                    {"production-topic": 1, "production-context": 2}
                    if index == 0
                    else {"production-topic": index + 1}
                ),
            )
            for index in range(60)
        ],
    )

    selected = asyncio.run(
        select_openalex_supplement_actions(
            context,
            frozen_query_specs={"q-adequate": spec},
            first_round_actions=[],
            decision=adequate,
        )
    )

    assert selected.actions == []
