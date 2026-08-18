"""Frozen Gold-blind OpenAlex semantic action backfill."""

from __future__ import annotations

import hashlib
import json
import unicodedata

from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    RecallGenerationContext,
    TextSearchAction,
    TextSearchPayload,
)
from paper_search.recall_experiments.generation.base import GenerationResult


_POLICY_VERSION = "openalex-semantic-backfill-v1"
_POLICY_BYTES = json.dumps(
    {
        "action_count": 1,
        "query": "nfkc-original-query",
        "search_mode": "semantic",
        "version": _POLICY_VERSION,
    },
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")


class SemanticBackfillQueryGenerator:
    """Emit exactly one deterministic semantic action for an isolated training query."""

    generator_type = "fixed_actions"
    backfill_policy = _POLICY_VERSION
    source_sha256 = "sha256:" + hashlib.sha256(_POLICY_BYTES).hexdigest()

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        query = " ".join(unicodedata.normalize("NFKC", context.original_query).split())
        batch = RecallActionBatch(
            actions=[
                TextSearchAction(
                    action_id="semantic-backfill-original",
                    strategy=_POLICY_VERSION,
                    action_type="text_search",
                    payload=TextSearchPayload(
                        query_text=query,
                        search_mode="semantic",
                    ),
                )
            ]
        )
        artifact_bytes = json.dumps(
            batch.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return GenerationResult(
            query_id=context.query_id,
            action_batch=batch,
            artifact_bytes=artifact_bytes,
            provenance={
                "backfill_policy": _POLICY_VERSION,
                "collection_mode": "semantic_backfill",
                "gold_visibility": "blind",
            },
        )


__all__ = ["SemanticBackfillQueryGenerator"]
