"""Execute citation actions against one frozen seed candidate."""

from __future__ import annotations

from paper_search.domain.models import ErrorDetail
from paper_search.recall_experiments.contracts import (
    CitationExpandAction,
    RecallSearchAction,
    RetrievalActionResult,
    RetrievalExecutionContext,
)
from paper_search.recall_experiments.retrieval.backends import CitationBackend


class CitationExpandHandler:
    """Resolve the requested frozen seed and delegate one expansion action."""

    def __init__(self, *, backend: CitationBackend) -> None:
        self._backend = backend

    async def execute(
        self,
        action: RecallSearchAction,
        context: RetrievalExecutionContext,
    ) -> RetrievalActionResult:
        if not isinstance(action, CitationExpandAction):
            raise TypeError("citation handler requires a citation-expand action")
        seed = next(
            (
                candidate.paper
                for candidate in context.seed_candidates
                if candidate.paper.canonical_id == action.payload.seed_canonical_id
            ),
            None,
        )
        if seed is None:
            return RetrievalActionResult(
                action_id=action.action_id,
                action_type=action.action_type,
                errors=[
                    ErrorDetail(
                        code="seed_unavailable",
                        message="citation action seed is absent from the execution context",
                        retryable=False,
                        provider="semantic_scholar",
                    )
                ],
            )

        result = await self._backend.expand(
            action_id=action.action_id,
            seed=seed,
            direction=action.payload.direction,
            limit=min(action.payload.limit, context.max_results_per_action),
        )
        return RetrievalActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            hits=[
                seed,
                *[
                    paper
                    for paper in result.hits
                    if paper.canonical_id != seed.canonical_id
                ],
            ],
            usage=result.usage,
            provenance={
                **result.provenance,
                "seed": "frozen",
                "expanded": result.provenance.get("provider", "citation_backend"),
            },
            errors=result.errors,
            infrastructure_failure=result.infrastructure_failure,
        )


__all__ = ["CitationExpandHandler"]
