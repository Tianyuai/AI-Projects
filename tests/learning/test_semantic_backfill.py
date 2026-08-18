from __future__ import annotations

import asyncio

from paper_search.domain.models import QuerySpec
from paper_search.learning.semantic_backfill import SemanticBackfillQueryGenerator
from paper_search.recall_experiments.contracts import RecallGenerationContext
from paper_search.recall_experiments.validation import validate_action_batch


def test_semantic_backfill_generates_exactly_one_gold_blind_openalex_action() -> None:
    context = RecallGenerationContext(
        query_id="q-1",
        original_query="Which work used 𝔾 geometric products?",
        query_spec=QuerySpec(
            original_query="Which work used 𝔾 geometric products?",
            research_goal="Which work used 𝔾 geometric products?",
        ),
    )

    result = asyncio.run(SemanticBackfillQueryGenerator().generate(context))

    assert len(result.action_batch.actions) == 1
    action = result.action_batch.actions[0]
    assert action.action_id == "semantic-backfill-original"
    assert action.action_type == "text_search"
    assert action.payload.search_mode == "semantic"
    assert action.payload.query_text == "Which work used G geometric products?"
    assert result.provenance["gold_visibility"] == "blind"
    assert (
        validate_action_batch(
            result.artifact_bytes.decode("utf-8"),
            context,
            allowed_actions=["text_search"],
            max_actions=1,
        )
        == result.action_batch
    )
