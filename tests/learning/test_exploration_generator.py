from __future__ import annotations

import asyncio

from paper_search.learning.exploration import DeterministicExplorationQueryGenerator
from paper_search.query.parser import rule_fallback
from paper_search.recall_experiments.contracts import RecallGenerationContext


def _context(query_id: str) -> RecallGenerationContext:
    query = (
        "Which papers study graph diffusion neural retrieval on MS MARCO "
        "with contrastive learning?"
    )
    return RecallGenerationContext(
        query_id=query_id,
        original_query=query,
        query_spec=rule_fallback(query),
    )


def _texts(query_id: str) -> list[str]:
    result = asyncio.run(
        DeterministicExplorationQueryGenerator().generate(_context(query_id))
    )
    return [
        action.payload.query_text
        if action.action_type == "text_search"
        else action.payload.title_text
        for action in result.action_batch.actions
    ]


def test_exploration_keeps_anchor_and_content_compression() -> None:
    texts = _texts("q-1")

    assert texts[0] == _context("q-1").original_query
    assert texts[1] == (
        "graph diffusion neural retrieval marco contrastive learning"
    )
    assert len(texts) == 3


def test_exploration_is_reproducible_and_rotates_only_the_third_action() -> None:
    first = _texts("q-1")

    assert _texts("q-1") == first
    variants = {_texts(f"q-{index}")[2] for index in range(20)}
    assert len(variants) > 1
    assert all(_texts(f"q-{index}")[:2] == first[:2] for index in range(20))


def test_exploration_emits_no_llm_receipts_and_records_policy_version() -> None:
    generator = DeterministicExplorationQueryGenerator()
    result = asyncio.run(generator.generate(_context("q-1")))

    assert generator.generator_type == "fixed_actions"
    assert generator.source_sha256.startswith("sha256:")
    assert len(generator.source_sha256) == 71
    assert result.call_receipts == []
    assert result.provenance["exploration_policy"] == "anchor-compress-rotate-v1"
    assert result.provenance["gold_visibility"] == "blind"
    assert int(result.provenance["candidate_pool_size"]) >= 3
    selected_ids = result.provenance["selected_candidate_ids"].split(",")
    assert selected_ids[:2] == ["candidate-anchor", "candidate-text-1"]
    assert len(selected_ids) == 3
