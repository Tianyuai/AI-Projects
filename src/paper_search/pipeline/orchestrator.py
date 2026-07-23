"""Minimal deterministic orchestration for mock-driven provider contracts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pydantic import Field

from paper_search.control.budget import BudgetExceededError, HardBudgetController, ReservationError
from paper_search.domain.models import (
    BudgetReservation,
    DomainModel,
    Paper,
    ProviderResult,
    QueryAnalysisResult,
    UsageActual,
    UsageEstimate,
)
from paper_search.processing.deduplicate import deduplicate_papers
from paper_search.processing.filter import apply_hard_filters
from paper_search.query.parser import QueryParser, rule_fallback
from paper_search.query.planner import QueryPlanner
from paper_search.ranking.fusion import fuse_provider_results
from paper_search.retrieval.base import SearchProvider


Analyzer = Callable[[str, BudgetReservation], Awaitable[ProviderResult[dict[str, Any]]]]


class MinimalSearchResult(DomainModel):
    query_analysis: QueryAnalysisResult
    papers: list[Paper]
    provider_results: dict[str, ProviderResult[list[Paper]]]
    trace: list[dict[str, Any]]
    usage: UsageActual
    stop_reason: str
    is_partial: bool
    warnings: list[str]
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prompt_version: str


class MockSearchOrchestrator:
    """Own reservations around injected mock analyzer and provider dependencies."""

    def __init__(
        self,
        *,
        controller: HardBudgetController,
        analyzer: Analyzer,
        providers: Mapping[str, SearchProvider],
        config_hash: str,
        prompt_version: str,
        analysis_estimate: UsageEstimate,
        provider_estimate: UsageEstimate,
    ) -> None:
        self._controller = controller
        self._analyzer = analyzer
        self._providers = dict(providers)
        self._config_hash = config_hash
        self._prompt_version = prompt_version
        self._analysis_estimate = analysis_estimate
        self._provider_estimate = provider_estimate
        self._parser = QueryParser(QueryPlanner())

    def _fallback(self, query: str) -> QueryAnalysisResult:
        spec = rule_fallback(query)
        return QueryAnalysisResult(query_spec=spec, search_plan=QueryPlanner().finalize(spec, None))

    @staticmethod
    def _combine_provider_results(
        results: list[ProviderResult[list[Paper]]],
    ) -> ProviderResult[list[Paper]]:
        last = results[-1]
        costs = [item.usage.cost_cny for item in results]
        return last.model_copy(
            update={
                "data": [paper for item in results for paper in item.data],
                "usage": UsageActual(
                    search_api_calls=sum(item.usage.search_api_calls for item in results),
                    llm_calls=sum(item.usage.llm_calls for item in results),
                    input_tokens=sum(item.usage.input_tokens for item in results),
                    output_tokens=sum(item.usage.output_tokens for item in results),
                    cost_cny=sum(cost for cost in costs if cost is not None)
                    if all(cost is not None for cost in costs)
                    else None,
                    elapsed_ms=sum(item.usage.elapsed_ms for item in results),
                ),
                "cache_hit": all(item.cache_hit for item in results),
                "latency_ms": sum(item.latency_ms for item in results),
                "errors": [error for item in results for error in item.errors],
            }
        )

    async def run(self, query: str, *, max_provider_results: int) -> MinimalSearchResult:
        warnings: list[str] = []
        trace: list[dict[str, Any]] = []
        provider_results: dict[str, ProviderResult[list[Paper]]] = {}
        try:
            analysis_reservation = self._controller.reserve("query.analyze", self._analysis_estimate)
        except BudgetExceededError:
            analysis = self._fallback(query)
            return self._result(
                analysis,
                [],
                provider_results,
                trace,
                "hard_stop",
                True,
                ["analysis: budget unavailable"],
            )
        try:
            analysis_result = await self._analyzer(query, analysis_reservation)
        except Exception:
            self._controller.fail_closed(analysis_reservation)
            return self._result(
                self._fallback(query),
                [],
                provider_results,
                trace,
                "hard_stop",
                True,
                ["analysis: dependency failure"],
            )
        try:
            self._controller.settle(analysis_reservation, analysis_result.usage)
        except ReservationError:
            self._controller.fail_closed(analysis_reservation)
            raise
        if analysis_result.errors:
            warnings.append("analysis: analyzer returned errors")
        analysis = await self._parser.parse(query, analysis_result)
        trace.append({"step": "analyze", "prompt_version": self._prompt_version})

        collected: dict[str, list[ProviderResult[list[Paper]]]] = {}
        stopped = False
        for subquery in analysis.search_plan.subqueries:
            names = [
                name
                for name in sorted(self._providers)
                if subquery.provider_hint == "either" or subquery.provider_hint == name
            ]
            for name in names:
                status = self._controller.stop_status()
                if status != "continue":
                    warnings.append(f"{name}: budget unavailable")
                    stopped = True
                    break
                try:
                    reservation = self._controller.reserve(
                        f"{name}.search:{subquery.query_id}", self._provider_estimate
                    )
                except BudgetExceededError:
                    warnings.append(f"{name}: budget unavailable")
                    stopped = True
                    break
                try:
                    result = await self._providers[name].search(
                        subquery.text,
                        analysis.search_plan.inherited_hard_filters,
                        max_provider_results,
                        reservation,
                    )
                    self._controller.settle(reservation, result.usage)
                except ReservationError:
                    self._controller.fail_closed(reservation)
                    raise
                collected.setdefault(name, []).append(result)
                trace.append(
                    {"step": "retrieve", "provider": name, "subquery_id": subquery.query_id}
                )
                if result.errors:
                    warnings.append(f"{name}: provider returned errors")
            if stopped:
                break
        provider_results = {
            name: self._combine_provider_results(results) for name, results in collected.items()
        }

        merged = deduplicate_papers([paper for result in provider_results.values() for paper in result.data])
        trace.append({"step": "deduplicate", "count": len(merged.papers)})
        filtered = apply_hard_filters(merged.papers, analysis.query_spec)
        trace.append({"step": "filter", "accepted": len(filtered.accepted)})
        accepted_ids = {item.paper.canonical_id for item in filtered.accepted}
        fused = fuse_provider_results(provider_results, method="rrf")
        papers = [item.paper for item in fused if item.paper.canonical_id in accepted_ids]
        trace.append({"step": "fuse", "count": len(papers)})
        status = self._controller.stop_status()
        stop_reason = status if status != "continue" else "completed"
        partial = bool(warnings) or stop_reason != "completed"
        return self._result(analysis, papers, provider_results, trace, stop_reason, partial, warnings)

    def _result(
        self,
        analysis: QueryAnalysisResult,
        papers: list[Paper],
        provider_results: dict[str, ProviderResult[list[Paper]]],
        trace: list[dict[str, Any]],
        stop_reason: str,
        is_partial: bool,
        warnings: list[str],
    ) -> MinimalSearchResult:
        return MinimalSearchResult(
            query_analysis=analysis,
            papers=papers,
            provider_results=provider_results,
            trace=trace,
            usage=self._controller.committed_usage,
            stop_reason=stop_reason,
            is_partial=is_partial,
            warnings=warnings,
            config_hash=self._config_hash,
            prompt_version=self._prompt_version,
        )
