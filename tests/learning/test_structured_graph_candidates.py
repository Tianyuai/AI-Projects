from __future__ import annotations

import asyncio

from paper_search.domain.models import Paper, QuerySpec
from paper_search.learning.structured_graph_candidates import (
    FixedBudgetOpenAlexQueryGenerator,
    OpenAlexGraphExpansionGenerator,
    StructuredCandidateGenerator,
    StructuredGraphCandidateGenerator,
)
from paper_search.recall_experiments.contracts import (
    RecallGenerationContext,
    RetrievalActionResult,
    SeedCandidate,
)
from paper_search.recall_experiments.validation import validate_action_batch


def _context(query: str) -> RecallGenerationContext:
    return RecallGenerationContext(
        query_id="query-1",
        original_query=query,
        query_spec=QuerySpec(original_query=query, research_goal=query),
    )


def test_structured_generation_preserves_anchor_and_decomposes_relation_target() -> None:
    generator = StructuredGraphCandidateGenerator(max_actions=12)

    result = asyncio.run(
        generator.generate(
            _context(
                "What is the work that introduced the concept of "
                "Zero-Shot Super-Resolution (ZSSR)?"
            )
        )
    )

    actions = result.action_batch.actions
    assert actions[0].action_id == "structured-anchor-original"
    assert actions[0].payload.query_text.startswith("What is the work")
    assert any(
        action.action_id == "structured-title-target"
        and action.payload.title_text
        == "Zero-Shot Super-Resolution ZSSR"
        for action in actions
    )
    assert any(
        action.action_id == "structured-relation-target"
        and action.payload.query_text
        == "introduced Zero-Shot Super-Resolution ZSSR"
        for action in actions
    )
    assert result.provenance["candidate_policy"] == "structured-graph-candidate-pool-v1"
    assert len(actions) <= 8


def test_structured_generation_splits_multi_constraint_query_into_facets() -> None:
    generator = StructuredGraphCandidateGenerator(max_actions=12)

    result = asyncio.run(
        generator.generate(
            _context(
                "Which works explored prompt learning for efficient and lightweight "
                "video understanding?"
            )
        )
    )

    texts = {
        action.payload.query_text
        for action in result.action_batch.actions
        if action.action_type == "text_search"
    }
    assert "prompt learning video understanding" in texts
    assert "prompt learning efficient lightweight" in texts


def test_refinement_selects_grounded_seed_and_adds_two_openalex_graph_directions() -> None:
    generator = StructuredGraphCandidateGenerator(max_actions=12, max_graph_seeds=1)
    context = _context(
        "Which works explored prompt learning for efficient lightweight video understanding?"
    )
    anchor = asyncio.run(generator.generate(context))
    relevant = Paper(
        canonical_id="openalex:W1000000001",
        openalex_id="W1000000001",
        title="Efficient Prompt Learning for Video Understanding",
        abstract="A lightweight method for video recognition.",
        sources=["openalex"],
    )
    noisy = Paper(
        canonical_id="openalex:W2000000002",
        openalex_id="W2000000002",
        title="Protein folding with diffusion models",
        sources=["openalex"],
    )
    refinement_context = context.model_copy(
        update={
            "seed_candidates": [
                SeedCandidate(paper=noisy),
                SeedCandidate(paper=relevant),
            ]
        }
    )

    result = asyncio.run(
        generator.refine(
            refinement_context,
            anchor,
            [
                RetrievalActionResult(
                    action_id="structured-anchor-original",
                    action_type="text_search",
                    hits=[noisy, relevant],
                )
            ],
        )
    )

    added = result.action_batch.actions[len(anchor.action_batch.actions) :]
    assert [(action.payload.seed_canonical_id, action.payload.direction) for action in added] == [
        (relevant.canonical_id, "references"),
        (relevant.canonical_id, "citations"),
    ]
    assert result.provenance["graph_seed_ids"] == relevant.canonical_id


def test_refinement_does_not_expand_ungrounded_or_non_openalex_seeds() -> None:
    generator = StructuredGraphCandidateGenerator(max_actions=12)
    context = _context("graph retrieval for scholarly documents")
    anchor = asyncio.run(generator.generate(context))
    refinement_context = context.model_copy(
        update={
            "seed_candidates": [
                SeedCandidate(
                    paper=Paper(
                        canonical_id="doi:10.1000/noisy",
                        title="Unrelated medical study",
                        sources=["openalex"],
                    )
                )
            ]
        }
    )

    result = asyncio.run(
        generator.refine(refinement_context, anchor, [])
    )

    assert result.action_batch == anchor.action_batch
    assert result.provenance["graph_status"] == "no_grounded_openalex_seed"


def test_generation_artifact_matches_validated_batch_for_compatibility_unicode() -> None:
    context = _context(
        "Which work used 𝔾3,0,0 geometric products to compute "
        "rotation-invariant features?"
    )

    result = asyncio.run(StructuredGraphCandidateGenerator().generate(context))

    validated = validate_action_batch(
        result.artifact_bytes.decode("utf-8"),
        context,
        allowed_actions=["text_search", "title_search", "citation_expand"],
        max_actions=12,
    )
    assert validated == result.action_batch


def test_structured_generator_emits_only_non_anchor_non_graph_actions() -> None:
    result = asyncio.run(
        StructuredCandidateGenerator(max_actions=4).generate(
            _context(
                "What is the work that introduced the concept of "
                "Zero-Shot Super-Resolution (ZSSR)?"
            )
        )
    )

    assert 1 <= len(result.action_batch.actions) <= 4
    assert all(action.action_type != "citation_expand" for action in result.action_batch.actions)
    assert all(
        getattr(action.payload, "query_text", None)
        != "What is the work that introduced the concept of Zero-Shot Super-Resolution (ZSSR)?"
        for action in result.action_batch.actions
    )
    assert all(
        getattr(action.payload, "search_mode", "lexical") == "lexical"
        for action in result.action_batch.actions
    )
    assert result.provenance["collection_mode"] == "structured_only"


def test_graph_expansion_is_independent_from_structured_generation() -> None:
    context = _context("efficient prompt learning for video understanding")
    anchor = asyncio.run(FixedBudgetOpenAlexQueryGenerator().generate(context))
    relevant = Paper(
        canonical_id="openalex:W1000000001",
        openalex_id="W1000000001",
        title="Efficient Prompt Learning for Video Understanding",
        abstract="A lightweight method for video recognition.",
        sources=["openalex"],
    )
    refinement_context = context.model_copy(
        update={"seed_candidates": [SeedCandidate(paper=relevant)]}
    )

    result = asyncio.run(
        OpenAlexGraphExpansionGenerator(max_graph_seeds=1).refine(
            refinement_context,
            anchor,
            [],
        )
    )

    added = result.action_batch.actions[len(anchor.action_batch.actions) :]
    assert [action.action_type for action in added] == [
        "citation_expand",
        "citation_expand",
    ]
    assert anchor.provenance["graph_status"] == "disabled"
    assert result.provenance["graph_status"] == "grounded_openalex_expansion"


def test_fixed_budget_generator_emits_two_anchors_plus_at_most_four_structured() -> None:
    query = (
        "Which works explored prompt learning for efficient and lightweight "
        "video understanding?"
    )

    result = asyncio.run(
        FixedBudgetOpenAlexQueryGenerator(max_openalex_actions=6).generate(
            _context(query)
        )
    )

    actions = result.action_batch.actions
    assert 2 <= len(actions) <= 6
    assert actions[0].payload.query_text == query
    assert actions[0].payload.search_mode == "lexical"
    assert actions[1].payload.query_text == query
    assert actions[1].payload.search_mode == "semantic"
    assert sum(
        getattr(action.payload, "search_mode", "lexical") == "semantic"
        for action in actions
    ) == 1
    assert all(action.action_type != "citation_expand" for action in actions)
    identities = []
    for action in actions:
        text = getattr(action.payload, "query_text", None) or getattr(
            action.payload, "title_text"
        )
        identities.append(
            (
                action.action_type,
                getattr(action.payload, "search_mode", "lexical"),
                " ".join(text.split()).casefold(),
            )
        )
    assert len(identities) == len(set(identities))
    assert result.provenance["graph_status"] == "disabled"
