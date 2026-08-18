"""Gold-blind deterministic action exploration for training-data collection."""

from __future__ import annotations

import hashlib
import json

from paper_search.learning.candidates import DeterministicActionCandidateGenerator
from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.routing import RuleQueryRouter
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    RecallGenerationContext,
    TextSearchAction,
    TextSearchPayload,
    TitleSearchAction,
    TitleSearchPayload,
)
from paper_search.recall_experiments.generation.base import GenerationResult


_POLICY_VERSION = "anchor-compress-rotate-v1"
_POLICY_MATERIAL = json.dumps(
    {
        "candidate_generator": "deterministic-action-candidates-v1",
        "selection": ["anchor", "content_compression", "stable_rotating_variant"],
        "selection_hash": "blake2b-64-query-id",
        "version": _POLICY_VERSION,
    },
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")


def _stable_index(value: str, length: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % length


def _to_recall_action(
    candidate: PolicyActionCandidate,
    *,
    index: int,
) -> TextSearchAction | TitleSearchAction:
    common = {
        "action_id": f"explore-{index}",
        "strategy": _POLICY_VERSION,
    }
    if candidate.action_type == "title_search":
        return TitleSearchAction(
            **common,
            action_type="title_search",
            payload=TitleSearchPayload(title_text=candidate.text),
        )
    return TextSearchAction(
        **common,
        action_type="text_search",
        payload=TextSearchPayload(query_text=candidate.text),
    )


class DeterministicExplorationQueryGenerator:
    """Collect anchor/compression/rotating-variant receipts without Gold access."""

    generator_type = "fixed_actions"
    exploration_policy = _POLICY_VERSION
    source_sha256 = "sha256:" + hashlib.sha256(_POLICY_MATERIAL).hexdigest()

    def __init__(self, *, candidate_pool_size: int = 12) -> None:
        self._router = RuleQueryRouter()
        self._candidate_generator = DeterministicActionCandidateGenerator(
            max_candidates=candidate_pool_size,
            include_semantic_anchor=False,
        )

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        routed = self._router.route(context.original_query)
        candidates = self._candidate_generator.generate(
            routed.query_spec,
            query_kind=routed.query_kind,
        )
        anchor = candidates[0]
        non_anchor = candidates[1:]
        selected = [anchor]
        if non_anchor:
            selected.append(non_anchor[0])
        variants = non_anchor[1:]
        if variants:
            selected.append(variants[_stable_index(context.query_id, len(variants))])
        actions = [
            _to_recall_action(candidate, index=index)
            for index, candidate in enumerate(selected, start=1)
        ]
        batch = RecallActionBatch(actions=actions)
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
                "generator": "deterministic_exploration",
                "exploration_policy": _POLICY_VERSION,
                "gold_visibility": "blind",
                "query_kind": routed.query_kind,
                "candidate_pool_size": str(len(candidates)),
                "selected_candidate_ids": ",".join(
                    candidate.action_id for candidate in selected
                ),
            },
        )


__all__ = ["DeterministicExplorationQueryGenerator"]
