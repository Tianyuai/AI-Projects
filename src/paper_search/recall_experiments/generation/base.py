"""Generator boundary for immutable recall-action batches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from paper_search.domain.models import DomainModel, ErrorDetail, UsageActual
from pydantic import Field
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    RecallGenerationContext,
    RetrievalActionResult,
)


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
    repair_count: int = Field(default=0, ge=0, le=2)


class QueryGenerator(Protocol):
    """Produce the one action batch that retrieval is allowed to execute."""

    async def generate(self, context: RecallGenerationContext) -> GenerationResult: ...


@runtime_checkable
class EvidenceSteeredQueryGenerator(QueryGenerator, Protocol):
    """Optionally refine an anchor batch using only first-round retrieval evidence."""

    async def refine(
        self,
        context: RecallGenerationContext,
        anchor_generation: GenerationResult,
        first_round_results: Sequence[RetrievalActionResult],
    ) -> GenerationResult: ...


__all__ = [
    "EvidenceSteeredQueryGenerator",
    "GenerationResult",
    "LLMCallReceipt",
    "QueryGenerator",
]
