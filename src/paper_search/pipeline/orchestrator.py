"""Minimal deterministic orchestration for mock-driven provider contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import Field, model_validator

from paper_search.application.contracts import DependencyDiagnostic, SnapshotRef
from paper_search.control.budget import BudgetExceededError, HardBudgetController, ReservationError
from paper_search.control.pricing import ActualCostPricer
from paper_search.domain.models import (
    BudgetReservation,
    DependencyName,
    DomainModel,
    ErrorDetail,
    NonEmptyStr,
    Paper,
    PlannerStatus,
    ProviderResult,
    QueryAnalysisResult,
    RankedPaper,
    ResolvedCitationEdge,
    UsageActual,
    UsageEstimate,
    SearchMode,
)
from paper_search.processing.deduplicate import deduplicate_papers
from paper_search.processing.filter import apply_hard_filters
from paper_search.query.parser import PlannerDependencyError, QueryParser, rule_fallback
from paper_search.query.planner import QueryPlanner
from paper_search.ranking.fusion import FusedPaper, fuse_provider_results
from paper_search.retrieval.base import SearchProvider
from paper_search.retrieval.routing import RetrievalPolicy, route_baseline_subqueries

if TYPE_CHECKING:
    from paper_search.ranking.cpu_document import DocumentRankingStage


Analyzer = Callable[[str, BudgetReservation], Awaitable[ProviderResult[dict[str, Any]]]]
RepairAnalyzer = Callable[
    [str, str, BudgetReservation], Awaitable[ProviderResult[dict[str, Any]]]
]


class OrchestratorResult(DomainModel):
    query_analysis: QueryAnalysisResult
    fused_papers: list[FusedPaper]
    high_relevance: list[RankedPaper]
    partial_relevance: list[RankedPaper]
    citation_edges: list[ResolvedCitationEdge]
    provider_results: dict[DependencyName, ProviderResult[list[Paper]]]
    retrieved_paper_ids: list[NonEmptyStr] = Field(default_factory=list)
    post_filter_paper_ids: list[NonEmptyStr] = Field(default_factory=list)
    pre_truncation_candidates: list[Paper] = Field(default_factory=list)
    diagnostics: list[DependencyDiagnostic]
    planner_status: PlannerStatus
    trace: list[dict[str, object]]
    usage: UsageActual
    stop_reason: str
    is_partial: bool
    warnings: list[str]
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prompt_version: str
    snapshot_set_id: str | None = None
    snapshot_captured_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_minimal_result(cls, value: object) -> object:
        """Accept the temporary Phase 1 result constructor without fake evidence."""

        if not isinstance(value, Mapping) or "papers" not in value:
            return value
        if "fused_papers" in value:
            raise ValueError("legacy papers and fused_papers cannot both be provided")
        data = dict(value)
        papers = data.pop("papers")
        if not isinstance(papers, list):
            raise ValueError("legacy papers must be a list")
        data.update(
            {
                "fused_papers": [
                    FusedPaper(paper=Paper.model_validate(paper), score=0.0, source_ranks={})
                    for paper in papers
                ],
                "high_relevance": [],
                "partial_relevance": [],
                "citation_edges": [],
                "diagnostics": [],
                "planner_status": "primary",
            }
        )
        return data

    @property
    def papers(self) -> list[Paper]:
        """Compatibility view while callers migrate to evidence-bearing fields."""

        ranked = [*self.high_relevance, *self.partial_relevance]
        if ranked:
            return [item.paper for item in ranked]
        return [item.paper for item in self.fused_papers]


MinimalSearchResult = OrchestratorResult


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
        provider_estimates: Mapping[str, UsageEstimate] | None = None,
        routing_limits: tuple[int, int, int] | None = None,
        retrieval_policy: RetrievalPolicy | None = None,
        execution_mode: SearchMode = "live",
        document_ranker: DocumentRankingStage | None = None,
        max_output_papers: int | None = None,
        pricer: ActualCostPricer | None = None,
        provider_adapter_names: Mapping[DependencyName, str] | None = None,
    ) -> None:
        self._controller = controller
        self._analyzer = analyzer
        unsupported = set(providers).difference({"openalex", "semantic_scholar"})
        if unsupported:
            raise ValueError("providers must be openalex or semantic_scholar")
        self._providers: dict[DependencyName, SearchProvider] = {
            cast(DependencyName, name): provider
            for name, provider in providers.items()
        }
        self._config_hash = config_hash
        self._prompt_version = prompt_version
        self._analysis_estimate = analysis_estimate
        self._provider_estimate = provider_estimate
        self._provider_estimates = dict(provider_estimates or {})
        self._routing_limits = routing_limits
        self._retrieval_policy = retrieval_policy
        self._execution_mode = execution_mode
        self._document_ranker = document_ranker
        if (
            max_output_papers is not None
            and (
                isinstance(max_output_papers, bool)
                or not isinstance(max_output_papers, int)
                or max_output_papers < 1
            )
        ):
            raise ValueError("max_output_papers must be a positive integer")
        self._max_output_papers = max_output_papers
        self._pricer = pricer
        self._provider_adapter_names = dict(provider_adapter_names or {})
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
        hashes = [item.provenance["response_hash"] for item in results]
        snapshot_refs: list[dict[str, object]] = []
        for item in results:
            raw_refs = item.provenance.get("snapshot_refs")
            if raw_refs is None:
                continue
            decoded = json.loads(raw_refs)
            if not isinstance(decoded, list) or any(
                not isinstance(raw, dict) for raw in decoded
            ):
                raise ValueError("invalid provider snapshot provenance")
            snapshot_refs.extend(decoded)
        provenance = dict(last.provenance)
        provenance["requested_at"] = results[0].provenance["requested_at"]
        provenance["response_hash"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    hashes,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        if snapshot_refs:
            provenance["snapshot_refs"] = json.dumps(
                snapshot_refs,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        combined_data: list[Paper] = []
        seen_ids: set[str] = set()
        for item in results:
            for paper in item.data:
                if paper.canonical_id in seen_ids:
                    continue
                seen_ids.add(paper.canonical_id)
                combined_data.append(paper)
        return last.model_copy(
            update={
                "data": combined_data,
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
                "provenance": provenance,
            }
        )

    @staticmethod
    def _diagnostic(
        dependency: DependencyName,
        result: ProviderResult[Any],
    ) -> DependencyDiagnostic:
        raw_refs = result.provenance.get("snapshot_refs", "[]")
        decoded_refs = json.loads(raw_refs)
        if not isinstance(decoded_refs, list):
            raise ValueError("invalid dependency snapshot provenance")
        refs = [SnapshotRef.model_validate(raw) for raw in decoded_refs]
        return DependencyDiagnostic(
            dependency=dependency,
            endpoint=result.provenance["endpoint"],
            model_id=result.provenance.get("model_id"),
            usage=result.usage,
            latency_ms=result.latency_ms,
            cache_hit=result.cache_hit,
            snapshot_refs=refs,
            errors=result.errors,
        )

    @staticmethod
    def _failure_diagnostic(
        dependency: DependencyName,
        *,
        code: str = "provider_error",
    ) -> DependencyDiagnostic:
        return DependencyDiagnostic(
            dependency=dependency,
            endpoint="dependency",
            model_id=None,
            usage=UsageActual(),
            latency_ms=0,
            cache_hit=False,
            snapshot_refs=[],
            errors=[
                ErrorDetail(
                    code=code,
                    message="Dependency execution failed",
                    retryable=False,
                    provider=dependency,
                )
            ],
        )

    def _settle_or_verify(
        self,
        reservation: BudgetReservation,
        actual: UsageActual,
    ) -> None:
        terminal = self._controller.terminal_outcome(reservation)
        if terminal is None:
            self._controller.settle(reservation, actual)
            return
        mode, recorded = terminal
        if mode != "settled" or recorded != actual:
            raise ReservationError("dependency settlement receipt does not match result")

    def _fail_closed_if_active(self, reservation: BudgetReservation) -> None:
        if self._controller.terminal_outcome(reservation) is None:
            self._controller.fail_closed(reservation)

    def _repair_estimate(self) -> UsageEstimate:
        """Reserve the remaining bounded LLM budget for one parser repair call."""
        committed = self._controller.committed_usage
        budget = self._controller.budget
        remaining_tokens = max(
            0,
            budget.max_total_tokens
            - committed.input_tokens
            - committed.output_tokens,
        )
        estimated_tokens = (
            self._analysis_estimate.input_tokens
            + self._analysis_estimate.output_tokens
        )
        repair_output_tokens = (
            remaining_tokens * self._analysis_estimate.output_tokens
            // estimated_tokens
            if estimated_tokens > 0
            else 0
        )
        repair_input_tokens = remaining_tokens - repair_output_tokens
        remaining_cost = max(
            Decimal("0"),
            Decimal(str(budget.max_cost_cny))
            - self._controller.known_committed_cost_cny,
        )
        remaining_elapsed = max(
            0,
            budget.max_elapsed_seconds * 1_000 - committed.elapsed_ms,
        )
        return UsageEstimate(
            llm_calls=max(1, self._analysis_estimate.llm_calls),
            input_tokens=repair_input_tokens,
            output_tokens=repair_output_tokens,
            cost_cny=remaining_cost,
            elapsed_ms=remaining_elapsed,
        )

    def _analysis_repair(
        self,
        query: str,
        diagnostics: list[DependencyDiagnostic],
    ) -> Callable[[str], Awaitable[ProviderResult[dict[str, Any]]]] | None:
        repair_method = cast(
            RepairAnalyzer | None,
            getattr(self._analyzer, "repair", None),
        )
        if not callable(repair_method):
            return None
        estimate = self._repair_estimate()
        if not self._controller.can_reserve(estimate):
            return None

        async def repair(invalid_analysis: str) -> ProviderResult[dict[str, Any]]:
            reservation = self._controller.reserve("query.repair", estimate)
            try:
                repaired = await repair_method(
                    query,
                    invalid_analysis,
                    reservation,
                )
                self._settle_or_verify(reservation, repaired.usage)
            except Exception:
                self._fail_closed_if_active(reservation)
                raise
            diagnostics.append(self._diagnostic("llm", repaired))
            return repaired

        return repair

    async def run(self, query: str, *, max_provider_results: int) -> OrchestratorResult:
        warnings: list[str] = []
        trace: list[dict[str, object]] = []
        provider_results: dict[DependencyName, ProviderResult[list[Paper]]] = {}
        diagnostics: list[DependencyDiagnostic] = []
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
                diagnostics=diagnostics,
                planner_status="rules_fallback",
            )
        try:
            analysis_result = await self._analyzer(query, analysis_reservation)
        except Exception:
            if self._execution_mode == "replay":
                if self._controller.terminal_outcome(analysis_reservation) is None:
                    self._controller.release(analysis_reservation)
                diagnostics.append(
                    self._failure_diagnostic("llm", code="integrity_failure")
                )
            else:
                self._fail_closed_if_active(analysis_reservation)
                diagnostics.append(self._failure_diagnostic("llm"))
            return self._result(
                self._fallback(query),
                [],
                provider_results,
                trace,
                "hard_stop",
                True,
                ["analysis: dependency failure"],
                diagnostics=diagnostics,
                planner_status="rules_fallback",
            )
        try:
            self._settle_or_verify(analysis_reservation, analysis_result.usage)
        except ReservationError:
            self._fail_closed_if_active(analysis_reservation)
            raise
        if analysis_result.errors:
            warnings.append("analysis: analyzer returned errors")
        diagnostics.append(self._diagnostic("llm", analysis_result))
        try:
            analysis = await self._parser.parse(
                query,
                analysis_result,
                repair=self._analysis_repair(query, diagnostics),
            )
        except PlannerDependencyError:
            codes = {error.code for error in analysis_result.errors}
            stop_reason = (
                "snapshot_unavailable"
                if "snapshot_unavailable" in codes
                else "dependency_failure"
            )
            return self._result(
                self._fallback(query),
                [],
                provider_results,
                trace,
                stop_reason,
                True,
                ["analysis: dependency failure"],
                diagnostics=diagnostics,
                planner_status="rules_fallback",
            )
        if analysis.planner_status == "rules_fallback":
            warnings.append("planner_rules_fallback")
        trace.append({"step": "analyze", "prompt_version": self._prompt_version})

        collected: dict[DependencyName, list[ProviderResult[list[Paper]]]] = {}
        action_results: dict[str, ProviderResult[list[Paper]]] = {}
        stopped = False
        routes: list[
            tuple[
                str,
                str,
                Literal["text_search", "title_search"],
                Literal["lexical", "semantic"],
                list[DependencyName],
                str | None,
                str | None,
            ]
        ]
        if self._retrieval_policy is not None:
            routes = [
                (
                    item.route_id,
                    item.text,
                    item.action_type,
                    item.search_mode,
                    cast(list[DependencyName], list(item.providers)),
                    item.routing_reason,
                    item.method,
                )
                for item in self._retrieval_policy.route(
                    analysis.search_plan,
                    original_query=analysis.query_spec.original_query,
                )
            ]
        elif self._routing_limits is None:
            routes = [
                (
                    subquery.query_id,
                    subquery.text,
                    subquery.action_type,
                    subquery.search_mode,
                    [
                        name
                        for name in sorted(self._providers)
                        if subquery.provider_hint == "either"
                        or subquery.provider_hint == name
                    ],
                    None,
                    None,
                )
                for subquery in analysis.search_plan.subqueries
            ]
        else:
            minimum, openalex_maximum, semantic_maximum = self._routing_limits
            routes = [
                (
                    item.subquery_id,
                    item.text,
                    next(
                        subquery.action_type
                        for subquery in analysis.search_plan.subqueries
                        if subquery.query_id == item.subquery_id
                    ),
                    next(
                        subquery.search_mode
                        for subquery in analysis.search_plan.subqueries
                        if subquery.query_id == item.subquery_id
                    ),
                    cast(list[DependencyName], list(item.providers)),
                    item.routing_reason,
                    None,
                )
                for item in route_baseline_subqueries(
                    analysis.search_plan,
                    min_openalex_calls=minimum,
                    max_openalex_calls=openalex_maximum,
                    max_semantic_scholar_calls=semantic_maximum,
                )
            ]
        for (
            subquery_id,
            subquery_text,
            action_type,
            search_mode,
            names,
            routing_reason,
            method,
        ) in routes:
            primary_result: ProviderResult[list[Paper]] | None = None
            for name in names:
                if (
                    name == "semantic_scholar"
                    and routing_reason not in {"high_priority_supplement"}
                    and primary_result is not None
                    and primary_result.data
                    and not primary_result.errors
                ):
                    trace.append(
                        {
                            "step": "skip_optional_provider",
                            "provider": name,
                            "subquery_id": subquery_id,
                            "reason": "openalex_sufficient",
                        }
                    )
                    continue
                status = self._controller.stop_status()
                if status != "continue":
                    warnings.append(f"{name}: budget unavailable")
                    stopped = True
                    break
                try:
                    reservation = self._controller.reserve(
                        f"{name}.search:{subquery_id}",
                        self._provider_estimates.get(name, self._provider_estimate),
                    )
                except BudgetExceededError:
                    warnings.append(f"{name}: budget unavailable")
                    stopped = True
                    break
                try:
                    provider_filters = dict(
                        analysis.search_plan.inherited_hard_filters
                    )
                    if name == "openalex" and search_mode == "semantic":
                        provider_filters["_search_mode"] = search_mode
                    result = await self._providers[name].search(
                        subquery_text,
                        provider_filters,
                        max_provider_results,
                        reservation,
                    )
                    self._settle_or_verify(reservation, result.usage)
                except ReservationError:
                    self._fail_closed_if_active(reservation)
                    raise
                except Exception:  # noqa: BLE001
                    if self._execution_mode == "replay":
                        if self._controller.terminal_outcome(reservation) is None:
                            self._controller.release(reservation)
                        diagnostics.append(
                            self._failure_diagnostic(
                                name,
                                code="integrity_failure",
                            )
                        )
                    elif self._controller.terminal_outcome(reservation) is None:
                        if (
                            self._controller.formal_live
                            and self._pricer is not None
                            and name in self._provider_adapter_names
                        ):
                            valued = self._pricer.value_actual(
                                dependency=name,
                                model_or_adapter=(
                                    self._provider_adapter_names[name]
                                ),
                                usage=UsageActual(search_api_calls=1),
                            )
                            self._controller.fail_closed(reservation, valued)
                        else:
                            try:
                                if self._controller.formal_live:
                                    self._controller.fail_closed(
                                        reservation,
                                        UsageActual(search_api_calls=1),
                                    )
                                else:
                                    self._controller.settle(
                                        reservation,
                                        UsageActual(search_api_calls=1),
                                    )
                            except ReservationError:
                                self._fail_closed_if_active(reservation)
                                raise
                    warnings.append(f"{name}: provider exception")
                    if (
                        self._execution_mode != "replay"
                        and name in {"openalex", "semantic_scholar"}
                    ):
                        diagnostics.append(self._failure_diagnostic(name))
                    continue
                collected.setdefault(name, []).append(result)
                action_results[f"{name}:{subquery_id}:{search_mode}"] = result
                if name == "openalex":
                    primary_result = result
                trace.append(
                    {
                        "step": "retrieve",
                        "provider": name,
                        "subquery_id": subquery_id,
                        "action_type": action_type,
                        "search_mode": search_mode,
                        "method": method,
                    }
                )
                if result.errors:
                    warnings.append(f"{name}: provider returned errors")
            if stopped:
                break
        provider_results = {
            name: self._combine_provider_results(results) for name, results in collected.items()
        }
        diagnostics.extend(
            self._diagnostic(name, result)
            for name, result in sorted(provider_results.items())
        )

        fusion_input = dict(action_results)
        merged = deduplicate_papers(
            [
                paper
                for result in fusion_input.values()
                for paper in result.data
            ]
        )
        trace.append({"step": "deduplicate", "count": len(merged.papers)})
        filtered = apply_hard_filters(merged.papers, analysis.query_spec)
        trace.append({"step": "filter", "accepted": len(filtered.accepted)})
        accepted_ids = {item.paper.canonical_id for item in filtered.accepted}
        fused = fuse_provider_results(
            cast(Mapping[str, ProviderResult[list[Paper]]], fusion_input),
            method="rrf",
        )
        papers = [item.paper for item in fused if item.paper.canonical_id in accepted_ids]
        fused_by_id = {item.paper.canonical_id: item for item in fused}
        selected_fused = [fused_by_id[paper.canonical_id] for paper in papers]
        trace.append({"step": "fuse", "count": len(papers)})
        if self._document_ranker is not None and selected_fused:
            prior_ids = [item.paper.canonical_id for item in selected_fused]
            ranked_fused = self._document_ranker.rank(
                analysis.query_spec.original_query,
                selected_fused,
            )
            ranked_ids = [item.paper.canonical_id for item in ranked_fused]
            if len(ranked_ids) != len(prior_ids) or set(ranked_ids) != set(prior_ids):
                raise ValueError("document ranker changed candidate identity")
            selected_fused = ranked_fused
            papers = [item.paper for item in selected_fused]
            trace.append(
                {
                    "step": "document_rank",
                    "status": "applied",
                    "model_id": self._document_ranker.model_id,
                    "count": len(papers),
                }
            )
        high_relevance: list[RankedPaper] = []
        partial_relevance: list[RankedPaper] = []
        citation_edges: list[ResolvedCitationEdge] = []
        pre_truncation_candidates = list(papers)
        if self._max_output_papers is not None and len(papers) > self._max_output_papers:
            papers = papers[: self._max_output_papers]
            selected_fused = [
                fused_by_id[paper.canonical_id] for paper in papers
            ]
            trace.append(
                {
                    "step": "truncate",
                    "count": len(papers),
                    "max_output_papers": self._max_output_papers,
                }
            )
        status = self._controller.stop_status()
        stop_reason = status if status != "continue" else "completed"
        partial = bool(warnings) or stop_reason != "completed"
        return self._result(
            analysis,
            papers,
            provider_results,
            trace,
            stop_reason,
            partial,
            warnings,
            fused_papers=selected_fused,
            high_relevance=high_relevance,
            partial_relevance=partial_relevance,
            citation_edges=citation_edges,
            diagnostics=diagnostics,
            planner_status=analysis.planner_status,
            retrieved_paper_ids=[paper.canonical_id for paper in merged.papers],
            post_filter_paper_ids=[
                item.paper.canonical_id for item in filtered.accepted
            ],
            pre_truncation_candidates=pre_truncation_candidates,
        )

    def _result(
        self,
        analysis: QueryAnalysisResult,
        papers: list[Paper],
        provider_results: dict[DependencyName, ProviderResult[list[Paper]]],
        trace: list[dict[str, object]],
        stop_reason: str,
        is_partial: bool,
        warnings: list[str],
        *,
        fused_papers: list[FusedPaper] | None = None,
        high_relevance: list[RankedPaper] | None = None,
        partial_relevance: list[RankedPaper] | None = None,
        citation_edges: list[ResolvedCitationEdge] | None = None,
        diagnostics: list[DependencyDiagnostic] | None = None,
        planner_status: PlannerStatus = "primary",
        retrieved_paper_ids: list[str] | None = None,
        post_filter_paper_ids: list[str] | None = None,
        pre_truncation_candidates: list[Paper] | None = None,
    ) -> OrchestratorResult:
        del papers
        return OrchestratorResult(
            query_analysis=analysis,
            fused_papers=fused_papers or [],
            high_relevance=high_relevance or [],
            partial_relevance=partial_relevance or [],
            citation_edges=citation_edges or [],
            provider_results=provider_results,
            retrieved_paper_ids=retrieved_paper_ids or [],
            post_filter_paper_ids=post_filter_paper_ids or [],
            pre_truncation_candidates=pre_truncation_candidates or [],
            diagnostics=diagnostics or [],
            planner_status=planner_status,
            trace=trace,
            usage=self._controller.committed_usage,
            stop_reason=stop_reason,
            is_partial=is_partial,
            warnings=warnings,
            config_hash=self._config_hash,
            prompt_version=self._prompt_version,
        )
