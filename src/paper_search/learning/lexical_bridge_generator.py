"""Independent candidate-family wrapper for the frozen lexical bridge."""

from __future__ import annotations

import json
import unicodedata

from paper_search.learning.lexical_bridge_deployment import LoadedLexicalBridge
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


def _normalized_search_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _artifact_bytes(batch: RecallActionBatch) -> bytes:
    return json.dumps(
        batch.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class LexicalBridgeCandidateGenerator:
    """Append at most one bridge action without changing the base generator."""

    generator_type = "local_cpu"
    model_id = "supervised-lexical-bridge-openalex-v2"

    def __init__(
        self,
        base: QueryGenerator,
        *,
        bridge: LoadedLexicalBridge,
        max_actions: int,
    ) -> None:
        if max_actions <= 0:
            raise ValueError("max actions must be positive")
        self._base = base
        self._bridge = bridge
        self._max_actions = max_actions
        self.source_sha256 = bridge.source_sha256

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        base_result = await self._base.generate(context)
        provenance = {
            **base_result.provenance,
            "lexical_bridge_model_sha256": self._bridge.source_sha256,
        }
        if len(base_result.action_batch.actions) >= self._max_actions:
            return base_result.model_copy(
                update={
                    "provenance": {
                        **provenance,
                        "lexical_bridge_status": "action_budget_exhausted",
                    }
                }
            )
        try:
            proposal = self._bridge.bridge.propose(
                context.original_query,
                neighbors=self._bridge.neighbors,
                max_expansion_terms=self._bridge.max_expansion_terms,
                min_neighbor_support=self._bridge.min_neighbor_support,
            )
        except Exception:
            return base_result.model_copy(
                update={
                    "provenance": {
                        **provenance,
                        "lexical_bridge_status": "inference_failed",
                    }
                }
            )
        if proposal is None:
            return base_result.model_copy(
                update={
                    "provenance": {
                        **provenance,
                        "lexical_bridge_status": "abstained",
                    }
                }
            )
        seen = {
            _normalized_search_text(action.payload.query_text)
            for action in base_result.action_batch.actions
            if isinstance(action, TextSearchAction)
        }
        if _normalized_search_text(proposal.query_text) in seen:
            return base_result.model_copy(
                update={
                    "provenance": {
                        **provenance,
                        "lexical_bridge_status": "duplicate",
                    }
                }
            )
        action = TextSearchAction(
            action_id="lexical-bridge-1",
            strategy="candidate-family:lexical-bridge",
            action_type="text_search",
            payload=TextSearchPayload(query_text=proposal.query_text),
        )
        batch = RecallActionBatch(
            actions=[*base_result.action_batch.actions, action]
        )
        return base_result.model_copy(
            update={
                "action_batch": batch,
                "artifact_bytes": _artifact_bytes(batch),
                "provenance": {
                    **provenance,
                    "lexical_bridge_status": "appended",
                    "lexical_bridge_terms": ",".join(proposal.expansion_terms),
                },
            }
        )


__all__ = ["LexicalBridgeCandidateGenerator"]
