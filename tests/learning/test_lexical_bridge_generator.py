from __future__ import annotations

import asyncio
import json
from typing import cast

from paper_search.domain.models import QuerySpec
from paper_search.learning.lexical_bridge import (
    LexicalBridgeExample,
    SupervisedLexicalBridge,
)
from paper_search.learning.lexical_bridge_deployment import LoadedLexicalBridge
from paper_search.learning.lexical_bridge_generator import (
    LexicalBridgeCandidateGenerator,
)
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    RecallGenerationContext,
    TextSearchAction,
    TextSearchPayload,
)
from paper_search.recall_experiments.generation.base import (
    GenerationResult,
    QueryGenerator,
)
from paper_search.recall_experiments.validation import validate_action_batch


def _canonical(batch: RecallActionBatch) -> bytes:
    return json.dumps(
        batch.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class _BaseGenerator:
    def __init__(self, query_texts: tuple[str, ...]) -> None:
        self._query_texts = query_texts

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        batch = RecallActionBatch(
            actions=[
                TextSearchAction(
                    action_id=f"policy-{index}",
                    strategy="learned_action_ranker",
                    action_type="text_search",
                    payload=TextSearchPayload(query_text=text),
                )
                for index, text in enumerate(self._query_texts, start=1)
            ]
        )
        return GenerationResult(
            query_id=context.query_id,
            action_batch=batch,
            artifact_bytes=_canonical(batch),
            provenance={"generator": "production_lexical"},
        )


def _loaded_bridge() -> LoadedLexicalBridge:
    bridge = SupervisedLexicalBridge.fit(
        [
            LexicalBridgeExample(
                query="multimodal representation learning",
                gold_titles=("Cross modal retrieval with aligned embeddings",),
            ),
            LexicalBridgeExample(
                query="multi modal representation alignment",
                gold_titles=("Cross-modal retrieval using joint embeddings",),
            ),
            LexicalBridgeExample(
                query="protein structure prediction",
                gold_titles=("Protein folding with geometric networks",),
            ),
        ],
        representation="word_char",
        learning_objective="neighbor_idf",
    )
    return LoadedLexicalBridge(
        bridge=bridge,
        source_sha256="sha256:" + "1" * 64,
        max_expansion_terms=6,
        neighbors=12,
        min_neighbor_support=2,
        manifest={},
    )


def _context() -> RecallGenerationContext:
    query = "multimodality representations"
    return RecallGenerationContext(
        query_id="q-1",
        original_query=query,
        query_spec=QuerySpec(original_query=query, research_goal=query),
    )


def test_lexical_bridge_generator_appends_one_independent_lexical_action() -> None:
    generator = LexicalBridgeCandidateGenerator(
        cast(QueryGenerator, _BaseGenerator(("multimodality representations",))),
        bridge=_loaded_bridge(),
        max_actions=2,
    )

    result = asyncio.run(generator.generate(_context()))

    assert len(result.action_batch.actions) == 2
    action = result.action_batch.actions[-1]
    assert action.action_id == "lexical-bridge-1"
    assert action.strategy == "candidate-family:lexical-bridge"
    assert action.payload.search_mode == "lexical"
    assert "retrieval" in action.payload.query_text
    assert result.provenance["lexical_bridge_status"] == "appended"
    assert result.provenance["lexical_bridge_model_sha256"] == "sha256:" + "1" * 64
    assert (
        validate_action_batch(
            result.artifact_bytes.decode("utf-8"),
            _context(),
            allowed_actions=["text_search"],
            max_actions=2,
        )
        == result.action_batch
    )


def test_lexical_bridge_generator_deduplicates_against_base_actions() -> None:
    loaded = _loaded_bridge()
    proposal = loaded.bridge.propose(
        _context().original_query,
        max_expansion_terms=loaded.max_expansion_terms,
    )
    assert proposal is not None
    generator = LexicalBridgeCandidateGenerator(
        cast(QueryGenerator, _BaseGenerator((proposal.query_text,))),
        bridge=loaded,
        max_actions=2,
    )

    result = asyncio.run(generator.generate(_context()))

    assert len(result.action_batch.actions) == 1
    assert result.provenance["lexical_bridge_status"] == "duplicate"


def test_lexical_bridge_generator_falls_back_to_base_on_inference_error() -> None:
    class _FailingBridge:
        def propose(self, query: str, **kwargs):
            raise ValueError("broken inference")

    loaded = _loaded_bridge()
    failing = LoadedLexicalBridge(
        bridge=cast(SupervisedLexicalBridge, _FailingBridge()),
        source_sha256=loaded.source_sha256,
        max_expansion_terms=6,
        neighbors=12,
        min_neighbor_support=2,
        manifest={},
    )
    generator = LexicalBridgeCandidateGenerator(
        cast(QueryGenerator, _BaseGenerator(("baseline query",))),
        bridge=failing,
        max_actions=2,
    )

    result = asyncio.run(generator.generate(_context()))

    assert len(result.action_batch.actions) == 1
    assert result.provenance["lexical_bridge_status"] == "inference_failed"
