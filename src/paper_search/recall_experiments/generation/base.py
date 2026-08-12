"""Generator boundary for immutable recall-action batches."""

from __future__ import annotations

from typing import Literal, Protocol

from paper_search.domain.models import DomainModel, ErrorDetail, UsageActual
from pydantic import Field
from paper_search.recall_experiments.contracts import RecallActionBatch, RecallGenerationContext


class LLMCallReceipt(DomainModel):
    call_kind: Literal["initial", "repair"]
    usage: UsageActual
    provenance: dict[str, str] = Field(default_factory=dict)
    errors: list[ErrorDetail] = Field(default_factory=list)
    terminal_state: Literal["succeeded", "repairable_failure", "infrastructure_failure"]


class GenerationResult(DomainModel):
    """One validated, serialized action batch for a single query."""

    query_id: str
    action_batch: RecallActionBatch
    artifact_bytes: bytes
    provenance: dict[str, str] = Field(default_factory=dict)
    call_receipts: list[LLMCallReceipt] = Field(default_factory=list)
    repair_count: int = Field(default=0, ge=0, le=1)


class QueryGenerator(Protocol):
    """Produce the one action batch that retrieval is allowed to execute."""

    async def generate(self, context: RecallGenerationContext) -> GenerationResult: ...


__all__ = ["GenerationResult", "LLMCallReceipt", "QueryGenerator"]
