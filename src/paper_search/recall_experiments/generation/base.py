"""Generator boundary for immutable recall-action batches."""

from __future__ import annotations

from typing import Protocol

from paper_search.domain.models import DomainModel
from pydantic import Field
from paper_search.recall_experiments.contracts import RecallActionBatch, RecallGenerationContext


class GenerationResult(DomainModel):
    """One validated, serialized action batch for a single query."""

    query_id: str
    action_batch: RecallActionBatch
    artifact_bytes: bytes
    provenance: dict[str, str] = Field(default_factory=dict)


class QueryGenerator(Protocol):
    """Produce the one action batch that retrieval is allowed to execute."""

    async def generate(self, context: RecallGenerationContext) -> GenerationResult: ...


__all__ = ["GenerationResult", "QueryGenerator"]
