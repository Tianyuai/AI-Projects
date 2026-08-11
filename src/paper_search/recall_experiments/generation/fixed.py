"""Byte-preserving fixed action replay."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence

from paper_search.recall_experiments.contracts import RecallGenerationContext
from paper_search.recall_experiments.generation.base import GenerationResult
from paper_search.recall_experiments.validation import validate_action_batch


class FixedActionGenerator:
    """Replay hash-bound action payloads without changing their serialized form."""

    def __init__(
        self,
        actions_by_query: Mapping[str, bytes | str | Mapping[str, object]],
        *,
        expected_query_ids: Sequence[str],
        allowed_actions: Collection[str],
        max_actions: int,
    ) -> None:
        expected = tuple(expected_query_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("expected query IDs must be unique")
        if set(actions_by_query) != set(expected):
            raise ValueError("fixed action query coverage does not match expected query IDs")
        self._actions_by_query = dict(actions_by_query)
        self._expected_query_ids = frozenset(expected)
        self._allowed_actions = frozenset(allowed_actions)
        self._max_actions = max_actions

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        if context.query_id not in self._expected_query_ids:
            raise ValueError(f"unknown query ID for fixed generation: {context.query_id}")
        raw = self._actions_by_query[context.query_id]
        artifact_bytes = _as_bytes(raw)
        action_batch = validate_action_batch(
            artifact_bytes.decode("utf-8"),
            context,
            allowed_actions=self._allowed_actions,
            max_actions=self._max_actions,
        )
        return GenerationResult(
            query_id=context.query_id,
            action_batch=action_batch,
            artifact_bytes=artifact_bytes,
        )


def _as_bytes(raw: bytes | str | Mapping[str, object]) -> bytes:
    if isinstance(raw, bytes):
        raw.decode("utf-8")
        return raw
    if isinstance(raw, str):
        raw.encode("utf-8").decode("utf-8")
        return raw.encode("utf-8")
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


__all__ = ["FixedActionGenerator"]
