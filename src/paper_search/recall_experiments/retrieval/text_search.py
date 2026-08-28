"""Execute validated text-search actions through an injected backend."""

from __future__ import annotations

from paper_search.recall_experiments.contracts import (
    RetrievalActionResult,
    RetrievalExecutionContext,
    RecallSearchAction,
    TextSearchAction,
)
from paper_search.recall_experiments.retrieval.backends import SearchBackend


class TextSearchHandler:
    """Adapt one normalized text action to the configured search backend."""

    def __init__(self, *, backend: SearchBackend) -> None:
        self._backend = backend

    async def execute(
        self,
        action: RecallSearchAction,
        context: RetrievalExecutionContext,
    ) -> RetrievalActionResult:
        if not isinstance(action, TextSearchAction):
            raise TypeError("text search handler requires a text-search action")
        filters = dict(context.provider_filters)
        limit = context.max_results_per_action
        if action.payload.search_mode == "semantic":
            filters["_search_mode"] = "semantic"
            limit = min(limit, 50)
        result = await self._backend.search(
            action_id=action.action_id,
            query=action.payload.query_text,
            filters=filters,
            limit=limit,
        )
        return RetrievalActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            hits=result.hits,
            usage=result.usage,
            provenance=result.provenance,
            errors=result.errors,
            infrastructure_failure=result.infrastructure_failure,
        )


__all__ = ["TextSearchHandler"]
