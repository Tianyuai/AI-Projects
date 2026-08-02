"""Minimal deterministic orchestration for mock-driven provider contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol, cast

from pydantic import Field, model_validator

from paper_search.application.contracts import DependencyDiagnostic, SnapshotRef
from paper_search.control.budget import BudgetExceededError, HardBudgetController, ReservationError
from paper_search.domain.models import (
    BudgetReservation,
    CandidateEvidence,
    DependencyName,
    DomainModel,
    ErrorDetail,
    Paper,
    PlannerStatus,
    ProviderResult,
    QueryAnalysisResult,
    RankedPaper,
    ResolvedCitationEdge,
    SubQuery,
    UsageActual,
    UsageEstimate,
)
from paper_search.processing.deduplicate import deduplicate_papers
from paper_search.processing.filter import apply_hard_filters
from paper_search.query.parser import PlannerDependencyError, QueryParser, rule_fallback
from paper_search.query.planner import QueryPlanner
from paper_search.ranking.embedding import (
    EmbeddingRankingStage,
    sanitize_embedding_model_id,
    sanitize_embedding_warnings,
)
from paper_search.graph.citation_expand import CitationExpansionResult
from paper_search.ranking.fusion import FusedPaper, fuse_provider_results
from paper_search.ranking.rerank import ConstraintRerankResult, ConstraintScoredPaper
from paper_search.retrieval.base import SearchProvider


Analyzer = Callable[[str, BudgetReservation], Awaitable[ProviderResult[dict[str, Any]]]]


class CitationExpansionStage(Protocol):
    def expand(self, seeds: list[Paper]) -> CitationExpansionResult: ...


class ConstraintRerankingStage(Protocol):
    def rerank(self, papers: list[Paper], constraints: list[str]) -> ConstraintRerankResult: ...


_SAFE_CITATION_WARNINGS = frozenset({"unresolved_citation_edge"})
_SAFE_RERANK_WARNINGS = frozenset({"rerank_unavailable"})


class OrchestratorResult(DomainModel):
    query_analysis: QueryAnalysisResult
    fused_papers: list[FusedPaper]
    high_relevance: list[RankedPaper]
    partial_relevance: list[RankedPaper]
    citation_edges: list[ResolvedCitationEdge]
    provider_results: dict[DependencyName, ProviderResult[list[Paper]]]
    diagnostics: list[DependencyDiagnostic]
    planner_status: PlannerStatus
    trace: list[dict[str, object]]
    usage: UsageActual
    stop_reason: str
    is_partial: bool
    warnings: list[str]
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prompt_version: str

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
        embedding_ranker: EmbeddingRankingStage | None = None,
        citation_expander: CitationExpansionStage | None = None,
        constraint_reranker: ConstraintRerankingStage | None = None,
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
        self._embedding_ranker = embedding_ranker
        self._citation_expander = citation_expander
        self._constraint_reranker = constraint_reranker
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
    def _failure_diagnostic(dependency: DependencyName) -> DependencyDiagnostic:
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
                    code="provider_error",
                    message="Dependency execution failed",
                    retryable=False,
                    provider=dependency,
                )
            ],
        )

    @staticmethod
    def _ranked_paper(
        item: ConstraintScoredPaper,
        subqueries: list[SubQuery],
    ) -> RankedPaper:
        relevance_level = "high" if item.score >= 0.8 else "partial"
        return RankedPaper(
            paper=item.paper,
            evidence=CandidateEvidence(
                paper_id=item.paper.canonical_id,
                matched_subqueries=[subquery.query_id for subquery in subqueries],
                matched_constraints=[],
                unmatched_constraints=[],
                filter_reasons=[],
                lexical_score=0.0,
                embedding_score=0.0,
                rerank_score=item.score,
                constraint_coverage=item.assessment.constraint_coverage,
                source_agreement=min(1.0, len(item.paper.sources) / 2),
                authority_score=0.0,
                recency_score=0.0,
                final_score=item.score,
                scoring_version="constraint-rerank-v1",
                relevance_level=relevance_level,
            ),
        )

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
            self._controller.fail_closed(analysis_reservation)
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
            self._controller.settle(analysis_reservation, analysis_result.usage)
        except ReservationError:
            self._controller.fail_closed(analysis_reservation)
            raise
        if analysis_result.errors:
            warnings.append("analysis: analyzer returned errors")
        diagnostics.append(self._diagnostic("llm", analysis_result))
        try:
            analysis = await self._parser.parse(query, analysis_result)
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
                except Exception:  # noqa: BLE001
                    try:
                        self._controller.settle(
                            reservation,
                            UsageActual(search_api_calls=1),
                        )
                    except ReservationError:
                        self._controller.fail_closed(reservation)
                        raise
                    warnings.append(f"{name}: provider exception")
                    if name in {"openalex", "semantic_scholar"}:
                        diagnostics.append(self._failure_diagnostic(name))
                    continue
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
        diagnostics.extend(
            self._diagnostic(name, result)
            for name, result in sorted(provider_results.items())
        )

        merged = deduplicate_papers([paper for result in provider_results.values() for paper in result.data])
        trace.append({"step": "deduplicate", "count": len(merged.papers)})
        filtered = apply_hard_filters(merged.papers, analysis.query_spec)
        trace.append({"step": "filter", "accepted": len(filtered.accepted)})
        accepted_ids = {item.paper.canonical_id for item in filtered.accepted}
        fused = fuse_provider_results(
            cast(Mapping[str, ProviderResult[list[Paper]]], provider_results),
            method="rrf",
        )
        papers = [item.paper for item in fused if item.paper.canonical_id in accepted_ids]
        fused_by_id = {item.paper.canonical_id: item for item in fused}
        selected_fused = [fused_by_id[paper.canonical_id] for paper in papers]
        trace.append({"step": "fuse", "count": len(papers)})
        if self._embedding_ranker is not None and papers:
            embedding = self._embedding_ranker.rank(
                analysis.query_spec.original_query,
                papers,
            )
            if embedding.status == "applied":
                papers = [item.paper for item in embedding.ranked]
                selected_fused = [
                    fused_by_id[paper.canonical_id] for paper in papers
                ]
            safe_model_id = sanitize_embedding_model_id(embedding.model_id)
            safe_warnings = sanitize_embedding_warnings(embedding.warnings)
            trace.append(
                {
                    "step": "embedding",
                    "status": embedding.status,
                    "model_id": safe_model_id,
                    "device": embedding.device,
                    "fallback_used": embedding.fallback_used,
                    "count": len(papers),
                }
            )
            warnings.extend(f"embedding: {warning}" for warning in safe_warnings)
        citation_edges: list[ResolvedCitationEdge] = []
        if self._citation_expander is not None and papers:
            prior_ids = {paper.canonical_id for paper in papers}
            try:
                citation = self._citation_expander.expand([papers[0]])
            except Exception:  # noqa: BLE001
                warnings.append("citation: expansion_unavailable")
                trace.append(
                    {"step": "citation", "status": "degraded", "count": len(papers)}
                )
            else:
                citation_edges.extend(citation.edges)
                additions = [
                    paper for paper in citation.papers if paper.canonical_id not in prior_ids
                ]
                papers = [*papers, *additions]
                safe_citation_warnings = [
                    warning
                    for warning in citation.warnings
                    if warning in _SAFE_CITATION_WARNINGS
                ]
                warnings.extend(f"citation: {warning}" for warning in safe_citation_warnings)
                trace.append(
                    {
                        "step": "citation",
                        "status": "applied",
                        "count": len(papers),
                        "expanded_count": len(additions),
                        "edge_count": len(citation.edges),
                        "skipped_edge_count": citation.skipped_edge_count,
                        "truncated": citation.truncated,
                    }
                )
        high_relevance: list[RankedPaper] = []
        partial_relevance: list[RankedPaper] = []
        if self._constraint_reranker is not None and papers:
            constraints = [
                *analysis.query_spec.must_have,
                *analysis.query_spec.should_have,
                *analysis.query_spec.exclusions,
            ]
            try:
                rerank = self._constraint_reranker.rerank(papers, constraints)
            except Exception:  # noqa: BLE001
                warnings.append("rerank: rerank_unavailable")
                trace.append(
                    {"step": "rerank", "status": "degraded", "count": len(papers)}
                )
            else:
                if rerank.status == "applied":
                    papers = [item.paper for item in rerank.ranked]
                    ranked_evidence = [
                        self._ranked_paper(item, analysis.search_plan.subqueries)
                        for item in rerank.ranked
                    ]
                    high_relevance = [
                        item
                        for item in ranked_evidence
                        if item.evidence.relevance_level == "high"
                    ]
                    partial_relevance = [
                        item
                        for item in ranked_evidence
                        if item.evidence.relevance_level == "partial"
                    ]
                safe_rerank_warnings = [
                    warning for warning in rerank.warnings if warning in _SAFE_RERANK_WARNINGS
                ]
                warnings.extend(f"rerank: {warning}" for warning in safe_rerank_warnings)
                trace.append(
                    {
                        "step": "rerank",
                        "status": rerank.status,
                        "count": len(papers),
                        "processed_count": rerank.processed_count,
                        "batch_count": rerank.batch_count,
                        "truncated": rerank.truncated,
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
    ) -> OrchestratorResult:
        del papers
        return OrchestratorResult(
            query_analysis=analysis,
            fused_papers=fused_papers or [],
            high_relevance=high_relevance or [],
            partial_relevance=partial_relevance or [],
            citation_edges=citation_edges or [],
            provider_results=provider_results,
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
