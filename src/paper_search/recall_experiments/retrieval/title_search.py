"""Execute validated title-search actions through an injected backend."""

from __future__ import annotations

from paper_search.recall_experiments.contracts import (
    RetrievalActionResult,
    RetrievalExecutionContext,
    RecallSearchAction,
    TitleSearchAction,
)
from paper_search.recall_experiments.retrieval.backends import SearchBackend


class TitleSearchHandler:
    """Search the complete title text that generation and validation approved."""

    def __init__(self, *, backend: SearchBackend) -> None:
        self._backend = backend

    async def execute(
        self,
        action: RecallSearchAction,
        context: RetrievalExecutionContext,
    ) -> RetrievalActionResult:
        if not isinstance(action, TitleSearchAction):
            raise TypeError("title search handler requires a title-search action")
        result = await self._backend.search(
            action_id=action.action_id,
            query=action.payload.title_text,
            filters=dict(context.provider_filters),
            limit=context.max_results_per_action,
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


__all__ = ["TitleSearchHandler"]
