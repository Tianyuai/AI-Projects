"""Async, budgeted constraint reranking over capture/replay LLM adapters."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Protocol

from pydantic import Field

from paper_search.application.contracts import DependencyDiagnostic, SnapshotRef
from paper_search.control.budget import (
    HardBudgetController,
    ReservationError,
)
from paper_search.domain.models import (
    BudgetReservation,
    ErrorDetail,
    Paper,
    ProviderResult,
    UsageActual,
    UsageEstimate,
)
from paper_search.ranking.rerank import (
    ConstraintReranker,
    ConstraintRerankResult,
)


class AsyncConstraintRerankingStage(Protocol):
    async def rerank(
        self,
        papers: list[Paper],
        constraints: list[str],
        *,
        controller: HardBudgetController,
    ) -> ConstraintRerankResult: ...


class _Analyzer(Protocol):
    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]: ...


class LLMConstraintRerankResult(ConstraintRerankResult):
    snapshot_refs: list[SnapshotRef] = Field(default_factory=list)
    diagnostics: list[DependencyDiagnostic] = Field(default_factory=list)


def _degraded(
    papers: list[Paper],
    *,
    snapshot_refs: list[SnapshotRef] | None = None,
    diagnostics: list[DependencyDiagnostic] | None = None,
) -> LLMConstraintRerankResult:
    result = ConstraintReranker(lambda paper, constraints: None).rerank(
        papers,
        ["force-evaluator"],
    )
    return LLMConstraintRerankResult(
        **result.model_dump(mode="python"),
        snapshot_refs=snapshot_refs or [],
        diagnostics=diagnostics or [],
    )


def _settle_or_verify(
    controller: HardBudgetController,
    reservation: BudgetReservation,
    result: ProviderResult[Any],
) -> None:
    terminal = controller.terminal_outcome(reservation)
    if terminal is None:
        controller.settle(reservation, result.usage)
        return
    mode, recorded = terminal
    if mode != "settled" or recorded != result.usage:
        raise ReservationError("LLM settlement receipt does not match result")


def _snapshot_ref(result: ProviderResult[Any]) -> SnapshotRef | None:
    provenance = result.provenance
    names = (
        "snapshot_entry_id",
        "snapshot_cache_key",
        "snapshot_response_sha256",
        "snapshot_path",
    )
    if not any(name in provenance for name in names):
        return None
    if not all(name in provenance for name in names):
        raise ValueError("incomplete LLM snapshot provenance")
    return SnapshotRef(
        entry_id=provenance["snapshot_entry_id"],
        dependency="llm",
        cache_key=provenance["snapshot_cache_key"],
        response_sha256=provenance["snapshot_response_sha256"],
        captured_at=datetime.fromisoformat(provenance["requested_at"]),
        snapshot_path=provenance["snapshot_path"],
    )


def _diagnostic(
    result: ProviderResult[Any],
    ref: SnapshotRef | None,
) -> DependencyDiagnostic:
    return DependencyDiagnostic(
        dependency="llm",
        endpoint="constraint_rerank",
        model_id=None,
        usage=result.usage,
        latency_ms=result.latency_ms,
        cache_hit=result.cache_hit,
        snapshot_refs=[] if ref is None else [ref],
        errors=[
            ErrorDetail(
                code=error.code,
                message="Reranking dependency reported an error",
                retryable=error.retryable,
                provider="llm",
            )
            for error in result.errors
        ],
    )


class LLMConstraintRerankingStage:
    def __init__(
        self,
        *,
        analyzer: _Analyzer,
        call_estimate: UsageEstimate,
        batch_size: int = 15,
        max_batches: int = 2,
    ) -> None:
        if type(batch_size) is not int or not 1 <= batch_size <= 15:
            raise ValueError("batch_size must be between 1 and 15")
        if type(max_batches) is not int or not 1 <= max_batches <= 2:
            raise ValueError("max_batches must be between 1 and 2")
        if call_estimate.llm_calls < 1 or call_estimate.cost_cny is None:
            raise ValueError("rerank calls require a valued LLM estimate")
        self._analyzer = analyzer
        self._call_estimate = call_estimate
        self._batch_size = batch_size
        self._max_batches = max_batches

    async def rerank(
        self,
        papers: list[Paper],
        constraints: list[str],
        *,
        controller: HardBudgetController,
    ) -> ConstraintRerankResult:
        if not papers:
            return LLMConstraintRerankResult(
                ranked=[],
                status="applied",
                processed_count=0,
                truncated=False,
                batch_count=0,
                warnings=[],
                snapshot_refs=[],
                diagnostics=[],
            )
        process_limit = min(
            len(papers),
            controller.budget.max_rerank_candidates,
            self._batch_size * self._max_batches,
        )
        processed = papers[:process_limit]
        if not processed:
            return _degraded(papers)
        assessments: dict[str, object] = {}
        refs: list[SnapshotRef] = []
        diagnostics: list[DependencyDiagnostic] = []
        for offset in range(0, len(processed), self._batch_size):
            batch = processed[offset : offset + self._batch_size]
            reservation = controller.reserve(
                f"llm.constraint_rerank:{offset // self._batch_size + 1}",
                self._call_estimate,
            )
            try:
                result = await self._analyzer.generate_json(
                    prompt_name="constraint_rerank",
                    payload={
                        "papers": [
                            {
                                "canonical_id": paper.canonical_id,
                                "title": paper.title,
                                "abstract": paper.abstract,
                            }
                            for paper in batch
                        ],
                        "constraints": list(constraints),
                    },
                    reservation=reservation,
                )
                _settle_or_verify(controller, reservation, result)
            except asyncio.CancelledError:
                if controller.terminal_outcome(reservation) is None:
                    controller.fail_closed(reservation, UsageActual())
                raise
            except ReservationError:
                if controller.terminal_outcome(reservation) is None:
                    controller.fail_closed(reservation)
                raise
            except Exception:
                if controller.terminal_outcome(reservation) is None:
                    controller.release(reservation)
                raise
            ref = _snapshot_ref(result)
            if ref is not None:
                refs.append(ref)
            diagnostics.append(_diagnostic(result, ref))
            if result.errors:
                return _degraded(
                    papers,
                    snapshot_refs=refs,
                    diagnostics=diagnostics,
                )
            raw_assessments = result.data.get("assessments")
            if not isinstance(raw_assessments, list):
                return _degraded(
                    papers,
                    snapshot_refs=refs,
                    diagnostics=diagnostics,
                )
            for raw in raw_assessments:
                if not isinstance(raw, dict):
                    return _degraded(
                        papers,
                        snapshot_refs=refs,
                        diagnostics=diagnostics,
                    )
                paper_id = raw.get("paper_id")
                if not isinstance(paper_id, str) or paper_id in assessments:
                    return _degraded(
                        papers,
                        snapshot_refs=refs,
                        diagnostics=diagnostics,
                    )
                assessments[paper_id] = raw
        if set(assessments) != {paper.canonical_id for paper in processed}:
            return _degraded(
                papers,
                snapshot_refs=refs,
                diagnostics=diagnostics,
            )
        ranked = ConstraintReranker(
            lambda paper, normalized: assessments[paper.canonical_id],
            max_candidates=process_limit,
            batch_size=self._batch_size,
            max_batches=self._max_batches,
        ).rerank(processed, constraints)
        return LLMConstraintRerankResult(
            **ranked.model_dump(mode="python"),
            snapshot_refs=refs,
            diagnostics=diagnostics,
        )


__all__ = [
    "AsyncConstraintRerankingStage",
    "LLMConstraintRerankResult",
    "LLMConstraintRerankingStage",
]
