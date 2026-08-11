"""Offline manual-action generation from a user-prepared JSON artifact."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from paper_search.recall_experiments.generation.fixed import FixedActionGenerator


class ManualActionGenerator(FixedActionGenerator):
    """Load manually prepared action batches; this class never owns an LLM client."""

    def __init__(
        self,
        actions_path: str | Path,
        *,
        expected_query_ids: Sequence[str],
        allowed_actions: Collection[str],
        max_actions: int,
    ) -> None:
        path = Path(actions_path)
        try:
            decoded = json.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("manual actions must be a UTF-8 JSON artifact") from error
        if not isinstance(decoded, Mapping) or not all(isinstance(key, str) for key in decoded):
            raise ValueError("manual actions must map query IDs to action batches")
        super().__init__(
            decoded,
            expected_query_ids=expected_query_ids,
            allowed_actions=allowed_actions,
            max_actions=max_actions,
        )


__all__ = ["ManualActionGenerator"]
