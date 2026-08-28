"""Frozen fixed-action replay."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence

from paper_search.recall_experiments.contracts import RecallGenerationContext
from paper_search.recall_experiments.contracts import RecallActionBatch
from paper_search.recall_experiments.generation.base import GenerationResult
from paper_search.recall_experiments.validation import validate_action_batch


class FixedActionGenerator:
    """Validate and freeze one immutable action payload for every expected query."""

    generator_type = "fixed_actions"

    def __init__(
        self,
        actions_by_query: Mapping[str, bytes | str | Mapping[str, object]],
        *,
        expected_query_ids: Sequence[str],
        allowed_actions: Collection[str],
        max_actions: int,
        source_sha256: str | None = None,
    ) -> None:
        expected = tuple(expected_query_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("expected query IDs must be unique")
        if set(actions_by_query) != set(expected):
            raise ValueError("fixed action query coverage does not match expected query IDs")
        self._actions_by_query = {
            query_id: _freeze_action_bytes(raw) for query_id, raw in actions_by_query.items()
        }
        self._expected_query_ids = frozenset(expected)
        self._allowed_actions = frozenset(allowed_actions)
        self._max_actions = max_actions
        self.source_sha256 = source_sha256

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        if context.query_id not in self._expected_query_ids:
            raise ValueError(f"unknown query ID for fixed generation: {context.query_id}")
        artifact_bytes = self._actions_by_query[context.query_id]
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


def _freeze_action_bytes(raw: bytes | str | Mapping[str, object]) -> bytes:
    try:
        if isinstance(raw, bytes):
            frozen = raw
            decoded = json.loads(frozen.decode("utf-8"))
        elif isinstance(raw, str):
            frozen = raw.encode("utf-8")
            decoded = json.loads(raw)
        else:
            frozen = json.dumps(
                raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            decoded = json.loads(frozen)
        RecallActionBatch.model_validate(decoded)
    except (TypeError, UnicodeDecodeError, UnicodeEncodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("fixed actions must contain a valid UTF-8 JSON action batch") from error
    return frozen


__all__ = ["FixedActionGenerator"]
