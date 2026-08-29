"""Minimal deterministic orchestration for mock-driven provider contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

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
    QuerySpec,
    RankedPaper,
    ResolvedCitationEdge,
    UsageActual,
    UsageEstimate,
    SearchMode,
)
from paper_search.evaluation.dataset import IdentifierMap
from paper_search.learning.adaptive_openalex_recall import (
    assess_openalex_recall_confidence,
)
from paper_search.learning.cpu_document_ranker import DocumentCandidateEvidence
from paper_search.processing.deduplicate import deduplicate_papers
from paper_search.processing.filter import apply_hard_filters
from paper_search.query.parser import (
    ClassifiedQueryAnalysis,
    PlannerDependencyError,
    QueryParser,
    rule_fallback,
)
from paper_search.query.low_confidence_supplement import (
    LowConfidenceLLMActionSelection,
    select_low_confidence_llm_action,
)
from paper_search.query.planner import QueryPlanner
from paper_search.query.semantic_actions import PROTECTED_ACTION_PROMPT_VERSION
from paper_search.ranking.fusion import FusedPaper, fuse_provider_results
from paper_search.recall_experiments.contracts import RecallActionBatch
from paper_search.retrieval.base import SearchProvider
from paper_search.retrieval.routing import RetrievalPolicy, route_baseline_subqueries

if TYPE_CHECKING:
    from paper_search.ranking.cpu_document import DocumentRankingStage


Analyzer = Callable[[str, BudgetReservation], Awaitable[ProviderResult[dict[str, Any]]]]
RepairAnalyzer = Callable[
    [str, str, BudgetReservation], Awaitable[ProviderResult[dict[str, Any]]]
]
OpenAlexSupplementSelector = Callable[
    [QuerySpec, Sequence[DocumentCandidateEvidence]], RecallActionBatch
]


class QueryPlanEnricher(Protocol):
    """Deterministically enrich a parsed plan without another remote call."""

    def enrich(
        self,
        analysis: ClassifiedQueryAnalysis,
    ) -> tuple[ClassifiedQueryAnalysis, dict[str, object]]: ...


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
        openalex_supplement_selector: OpenAlexSupplementSelector | None = None,
        max_total_openalex_actions: int | None = None,
        query_plan_enricher: QueryPlanEnricher | None = None,
        identifier_map: IdentifierMap | None = None,
        identifier_alias_count: int = 0,
        max_raw_candidates: int | None = None,
        max_deduplicated_candidates: int | None = None,
        max_additional_raw_candidates: int | None = None,
        max_total_raw_candidates: int | None = None,
        low_confidence_analyzer: Analyzer | None = None,
        low_confidence_prompt_version: str | None = None,
        low_confidence_analysis_estimate: UsageEstimate | None = None,
        max_low_confidence_raw_candidates: int | None = None,
        max_low_confidence_deduplicated_candidates: int | None = None,
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
        if openalex_supplement_selector is None:
            if max_total_openalex_actions is not None:
                raise ValueError(
                    "max_total_openalex_actions requires an OpenAlex supplement selector"
                )
        elif (
            type(max_total_openalex_actions) is not int
            or not 1 <= max_total_openalex_actions <= 7
        ):
            raise ValueError(
                "OpenAlex supplement selection requires a total action bound from one to seven"
            )
        self._openalex_supplement_selector = openalex_supplement_selector
        self._max_total_openalex_actions = max_total_openalex_actions
        self._query_plan_enricher = query_plan_enricher
        if (
            isinstance(identifier_alias_count, bool)
            or not isinstance(identifier_alias_count, int)
            or identifier_alias_count < 0
        ):
            raise ValueError("identifier_alias_count must be a non-negative integer")
        self._identifier_map = identifier_map
        self._identifier_alias_count = identifier_alias_count
        for name, value in (
            ("max_raw_candidates", max_raw_candidates),
            ("max_deduplicated_candidates", max_deduplicated_candidates),
            ("max_additional_raw_candidates", max_additional_raw_candidates),
            ("max_total_raw_candidates", max_total_raw_candidates),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            max_raw_candidates is not None
            and max_total_raw_candidates is not None
            and max_total_raw_candidates < max_raw_candidates
        ):
            raise ValueError("max_total_raw_candidates cannot be below max_raw_candidates")
        self._max_raw_candidates = max_raw_candidates
        self._max_deduplicated_candidates = max_deduplicated_candidates
        self._max_additional_raw_candidates = max_additional_raw_candidates
        self._max_total_raw_candidates = max_total_raw_candidates
        evidence_method = getattr(query_plan_enricher, "soft_concept_terms", None)
        soft_concept_evidence = cast(
            Callable[[str], Iterable[str]] | None,
            evidence_method if callable(evidence_method) else None,
        )
        low_confidence_values = (
            low_confidence_analyzer,
            low_confidence_prompt_version,
            low_confidence_analysis_estimate,
            max_low_confidence_raw_candidates,
            max_low_confidence_deduplicated_candidates,
        )
        if any(value is not None for value in low_confidence_values) and not all(
            value is not None for value in low_confidence_values
        ):
            raise ValueError(
                "low-confidence analyzer, prompt, estimate, and quotas must be configured together"
            )
        if (
            low_confidence_prompt_version is not None
            and low_confidence_prompt_version != PROTECTED_ACTION_PROMPT_VERSION
        ):
            raise ValueError("low-confidence supplement requires the protected v3 prompt")
        for name, value in (
            ("max_low_confidence_raw_candidates", max_low_confidence_raw_candidates),
            (
                "max_low_confidence_deduplicated_candidates",
                max_low_confidence_deduplicated_candidates,
            ),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        self._low_confidence_analyzer = low_confidence_analyzer
        self._low_confidence_prompt_version = low_confidence_prompt_version
        self._low_confidence_analysis_estimate = low_confidence_analysis_estimate
        self._max_low_confidence_raw_candidates = max_low_confidence_raw_candidates
        self._max_low_confidence_deduplicated_candidates = (
            max_low_confidence_deduplicated_candidates
        )
        self._parser = QueryParser(
            QueryPlanner(
                prompt_version=prompt_version,
                soft_concept_evidence=soft_concept_evidence,
            )
        )
        self._low_confidence_parser = (
            QueryParser(
                QueryPlanner(
                    prompt_version=cast(str, low_confidence_prompt_version),
                    soft_concept_evidence=soft_concept_evidence,
                )
            )
            if low_confidence_analyzer is not None
            else None
        )

    @staticmethod
    def _round_robin_cap_action_results(
        results: Mapping[str, ProviderResult[list[Paper]]],
        limit: int | None,
    ) -> dict[str, ProviderResult[list[Paper]]]:
        """Retain equal-rank evidence across actions before taking deeper ranks."""

        if limit is None or sum(len(item.data) for item in results.values()) <= limit:
            return dict(results)
        source_ids = list(results)
        retained: dict[str, list[Paper]] = {source_id: [] for source_id in source_ids}
        retained_count = 0
        max_depth = max((len(results[source_id].data) for source_id in source_ids), default=0)
        for rank in range(max_depth):
            for source_id in source_ids:
                data = results[source_id].data
                if rank >= len(data):
                    continue
                retained[source_id].append(data[rank])
                retained_count += 1
                if retained_count == limit:
                    return {
                        item_id: results[item_id].model_copy(
                            update={"data": retained[item_id]}
                        )
                        for item_id in source_ids
                    }
        return dict(results)

    def _openalex_candidate_evidence(
        self,
        action_results: Mapping[str, ProviderResult[list[Paper]]],
        query_spec: QuerySpec,
    ) -> list[DocumentCandidateEvidence]:
        openalex_results = {
            action_id: result
            for action_id, result in action_results.items()
            if action_id.startswith("openalex:")
        }
        if not openalex_results:
            return []
        merged = deduplicate_papers(
            [paper for result in openalex_results.values() for paper in result.data],
            id_map=self._identifier_map,
        )
        accepted_ids = {
            item.paper.canonical_id
            for item in apply_hard_filters(merged.papers, query_spec).accepted
        }
        return [
            DocumentCandidateEvidence(
                paper=item.paper,
                baseline_score=item.score,
                source_ranks=item.source_ranks,
            )
            for item in fuse_provider_results(
                openalex_results,
                method="rrf",
                id_map=self._identifier_map,
            )
            if item.paper.canonical_id in accepted_ids
        ]

    def _fair_merge_nonreinforcing_fused(
        self,
        baseline: Sequence[FusedPaper],
        supplemental: Sequence[FusedPaper],
    ) -> list[FusedPaper]:
        """Admit new supplemental members without changing baseline evidence."""

        baseline_papers = [item.paper for item in baseline]
        merged = list(baseline)
        for candidate in supplemental:
            if len(
                deduplicate_papers(
                    [*baseline_papers, candidate.paper],
                    id_map=self._identifier_map,
                ).papers
            ) == len(baseline_papers):
                continue
            merged.append(candidate)
        return sorted(merged, key=lambda item: item.score, reverse=True)

    def _fallback(self, query: str) -> QueryAnalysisResult:
        spec = rule_fallback(query)
        return QueryAnalysisResult(query_spec=spec, search_plan=QueryPlanner().finalize(spec, None))

    @staticmethod
    def _provider_search_filters(
        inherited_hard_filters: Mapping[str, object],
    ) -> dict[str, object]:
        """Return only constraints supported by both remote search adapters.

        Venue and exclusion constraints remain part of ``QuerySpec`` and are
        enforced after normalization. Passing them to either provider would
        make an otherwise valid request fail before dispatch.
        """

        return {
            name: inherited_hard_filters[name]
            for name in ("year_from", "year_to")
            if name in inherited_hard_filters
        }

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

    async def _select_low_confidence_llm_supplement(
        self,
        query: str,
        analysis: ClassifiedQueryAnalysis,
        action_results: Mapping[str, ProviderResult[list[Paper]]],
        *,
        diagnostics: list[DependencyDiagnostic],
        trace: list[dict[str, object]],
        warnings: list[str],
    ) -> LowConfidenceLLMActionSelection | None:
        if (
            self._low_confidence_analyzer is None
            or self._low_confidence_parser is None
            or self._low_confidence_analysis_estimate is None
            or self._low_confidence_prompt_version is None
        ):
            return None
        candidates = self._openalex_candidate_evidence(
            action_results,
            analysis.query_spec,
        )
        decision = assess_openalex_recall_confidence(
            analysis.query_spec,
            candidates,
        )
        trace.append(
            {
                "step": "assess_low_confidence_recall",
                **decision.model_dump(mode="json"),
            }
        )
        if analysis.query_spec.exclusions:
            trace.append(
                {
                    "step": "skip_llm_supplement",
                    "reason": "strict_negation_abstention",
                }
            )
            return None
        if not decision.low_confidence:
            trace.append(
                {
                    "step": "skip_llm_supplement",
                    "reason": "production_pool_adequate",
                }
            )
            return None
        try:
            reservation = self._controller.reserve(
                "query.low_confidence_analyze",
                self._low_confidence_analysis_estimate,
            )
        except BudgetExceededError:
            trace.append(
                {
                    "step": "llm_supplement_fallback",
                    "reason": "budget_unavailable",
                }
            )
            warnings.append("low-confidence supplement: budget unavailable")
            return None
        try:
            result = await self._low_confidence_analyzer(query, reservation)
            self._settle_or_verify(reservation, result.usage)
        except Exception:  # noqa: BLE001
            if self._execution_mode == "replay":
                if self._controller.terminal_outcome(reservation) is None:
                    self._controller.release(reservation)
                diagnostics.append(
                    self._failure_diagnostic("llm", code="integrity_failure")
                )
            else:
                self._fail_closed_if_active(reservation)
                diagnostics.append(self._failure_diagnostic("llm"))
            trace.append(
                {
                    "step": "llm_supplement_fallback",
                    "reason": "analyzer_exception",
                }
            )
            warnings.append("low-confidence supplement: analyzer exception")
            return None
        diagnostics.append(self._diagnostic("llm", result))
        try:
            candidate_analysis = await self._low_confidence_parser.parse(
                query,
                result,
                repair=None,
            )
        except PlannerDependencyError:
            trace.append(
                {
                    "step": "llm_supplement_fallback",
                    "reason": "analysis_unavailable",
                }
            )
            warnings.append("low-confidence supplement: analysis unavailable")
            return None
        analyze_trace: dict[str, object] = {
            "step": "analyze_low_confidence_supplement",
            "prompt_version": self._low_confidence_prompt_version,
            "planner_status": candidate_analysis.planner_status,
            "model_id": result.provenance.get("model_id", "unavailable"),
            "cache_hit": result.cache_hit,
            "usage": result.usage.model_dump(mode="json"),
            "production_query_spec_retained": True,
            "production_search_plan_retained": True,
        }
        raw_pricing_receipt = result.provenance.get("pricing_receipt")
        if raw_pricing_receipt is not None:
            try:
                parsed_pricing_receipt = json.loads(raw_pricing_receipt)
            except json.JSONDecodeError:
                parsed_pricing_receipt = None
            if isinstance(parsed_pricing_receipt, dict):
                analyze_trace["pricing_receipt"] = parsed_pricing_receipt
        trace.append(analyze_trace)
        selection = select_low_confidence_llm_action(
            analysis.query_spec,
            analysis.search_plan,
            candidate_analysis.search_plan,
            decision,
        )
        if selection is None:
            trace.append(
                {
                    "step": "skip_llm_supplement",
                    "reason": "no_safe_novel_action",
                }
            )
            return None
        trace.append(
            {
                "step": "select_low_confidence_supplement",
                "policy": "low-confidence-llm-lexical-supplement-v1",
                "source_query_id": selection.source_query_id,
                "query_text": selection.action.text,
                "provider_hint": selection.action.provider_hint,
                "search_mode": selection.action.search_mode,
                "novel_phrase_count": selection.novel_phrase_count,
                "novel_term_count": selection.novel_term_count,
                "gold_features_used": False,
            }
        )
        return selection

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
        if analysis.planner_status == "repaired":
            warnings = [
                warning
                for warning in warnings
                if warning != "analysis: analyzer returned errors"
            ]
        if analysis.planner_status == "rules_fallback":
            warnings.append("planner_rules_fallback")
        analyze_trace: dict[str, object] = {
            "step": "analyze",
            "prompt_version": self._prompt_version,
            "planner_status": analysis.planner_status,
            "model_id": analysis_result.provenance.get("model_id", "unavailable"),
            "cache_hit": analysis_result.cache_hit,
            "usage": analysis_result.usage.model_dump(mode="json"),
        }
        raw_pricing_receipt = analysis_result.provenance.get("pricing_receipt")
        if raw_pricing_receipt is not None:
            try:
                parsed_pricing_receipt = json.loads(raw_pricing_receipt)
            except json.JSONDecodeError:
                parsed_pricing_receipt = None
            if isinstance(parsed_pricing_receipt, dict):
                analyze_trace["pricing_receipt"] = parsed_pricing_receipt
        trace.append(analyze_trace)
        if self._query_plan_enricher is not None:
            analysis, enrichment_receipt = self._query_plan_enricher.enrich(analysis)
            trace.append(enrichment_receipt)

        collected: dict[DependencyName, list[ProviderResult[list[Paper]]]] = {}
        action_results: dict[str, ProviderResult[list[Paper]]] = {}
        supplement_source_ids: set[str] = set()
        llm_supplement_source_ids: set[str] = set()
        openalex_action_attempt_count = 0
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
                    provider_filters = self._provider_search_filters(
                        analysis.search_plan.inherited_hard_filters
                    )
                    if name == "openalex" and search_mode == "semantic":
                        provider_filters["_search_mode"] = search_mode
                    if name == "openalex":
                        openalex_action_attempt_count += 1
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
                        "query_text": subquery_text,
                        "action_type": action_type,
                        "search_mode": search_mode,
                        "routing_reason": routing_reason,
                        "method": method,
                        "result_count": len(result.data),
                        "error_count": len(result.errors),
                        "cache_hit": result.cache_hit,
                        "response_hash": result.provenance.get(
                            "response_hash", "unavailable"
                        ),
                    }
                )
                if result.errors:
                    warnings.append(f"{name}: provider returned errors")
            if stopped:
                break
        if (
            self._openalex_supplement_selector is not None
            and self._max_total_openalex_actions is not None
            and openalex_action_attempt_count < self._max_total_openalex_actions
            and "openalex" in self._providers
        ):
            candidates = self._openalex_candidate_evidence(
                action_results,
                analysis.query_spec,
            )
            supplement = self._openalex_supplement_selector(
                analysis.query_spec,
                candidates,
            )
            remaining = self._max_total_openalex_actions - openalex_action_attempt_count
            if len(supplement.actions) > min(1, remaining):
                raise ValueError("OpenAlex supplement exceeded its bounded action slot")
            existing_identities = {
                (search_mode, " ".join(text.split()).casefold())
                for _route_id, text, _action_type, search_mode, names, _reason, _method in routes
                if "openalex" in names
            }
            for supplement_action in supplement.actions:
                if supplement_action.action_type != "text_search":
                    raise ValueError("OpenAlex supplement must be a text search action")
                supplement_identity = (
                    supplement_action.payload.search_mode,
                    " ".join(supplement_action.payload.query_text.split()).casefold(),
                )
                if supplement_identity in existing_identities:
                    trace.append(
                        {
                            "step": "skip_supplement",
                            "provider": "openalex",
                            "subquery_id": supplement_action.action_id,
                            "reason": "duplicate_action_identity",
                        }
                    )
                    continue
                existing_identities.add(supplement_identity)
                if self._controller.stop_status() != "continue":
                    warnings.append("openalex supplement: budget unavailable")
                    break
                try:
                    reservation = self._controller.reserve(
                        f"openalex.search:{supplement_action.action_id}",
                        self._provider_estimates.get(
                            "openalex",
                            self._provider_estimate,
                        ),
                    )
                except BudgetExceededError:
                    warnings.append("openalex supplement: budget unavailable")
                    break
                try:
                    provider_filters = self._provider_search_filters(
                        analysis.search_plan.inherited_hard_filters
                    )
                    if supplement_action.payload.search_mode == "semantic":
                        provider_filters["_search_mode"] = "semantic"
                    openalex_action_attempt_count += 1
                    result = await self._providers["openalex"].search(
                        supplement_action.payload.query_text,
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
                                "openalex",
                                code="integrity_failure",
                            )
                        )
                    elif self._controller.terminal_outcome(reservation) is None:
                        if (
                            self._controller.formal_live
                            and self._pricer is not None
                            and "openalex" in self._provider_adapter_names
                        ):
                            valued = self._pricer.value_actual(
                                dependency="openalex",
                                model_or_adapter=self._provider_adapter_names[
                                    "openalex"
                                ],
                                usage=UsageActual(search_api_calls=1),
                            )
                            self._controller.fail_closed(reservation, valued)
                        else:
                            self._controller.settle(
                                reservation,
                                UsageActual(search_api_calls=1),
                            )
                    trace.append(
                        {
                            "step": "supplement_fallback",
                            "provider": "openalex",
                            "subquery_id": supplement_action.action_id,
                            "reason": "provider_exception",
                        }
                    )
                    warnings.append("openalex supplement: provider exception")
                    continue
                collected.setdefault("openalex", []).append(result)
                source_id = (
                    "openalex:"
                    f"{supplement_action.action_id}:"
                    f"{supplement_action.payload.search_mode}"
                )
                action_results[source_id] = result
                supplement_source_ids.add(source_id)
                trace.append(
                    {
                        "step": "retrieve_supplement",
                        "provider": "openalex",
                        "subquery_id": supplement_action.action_id,
                        "query_text": supplement_action.payload.query_text,
                        "action_type": supplement_action.action_type,
                        "search_mode": supplement_action.payload.search_mode,
                        "strategy": supplement_action.strategy,
                        "result_count": len(result.data),
                        "error_count": len(result.errors),
                        "cache_hit": result.cache_hit,
                        "response_hash": result.provenance.get(
                            "response_hash", "unavailable"
                        ),
                    }
                )
                if result.errors:
                    warnings.append("openalex supplement: provider returned errors")
        llm_selection = await self._select_low_confidence_llm_supplement(
            query,
            analysis,
            action_results,
            diagnostics=diagnostics,
            trace=trace,
            warnings=warnings,
        )
        if llm_selection is not None:
            llm_action = llm_selection.action
            provider_names: list[DependencyName] = ["openalex"]
            if (
                llm_action.provider_hint == "semantic_scholar"
                and "semantic_scholar" in self._providers
            ):
                provider_names.append("semantic_scholar")
            existing_requests = {
                (
                    str(item.get("provider")),
                    str(item.get("search_mode")),
                    " ".join(str(item.get("query_text", "")).split()).casefold(),
                )
                for item in trace
                if item.get("step") in {"retrieve", "retrieve_supplement"}
            }
            for name in provider_names:
                request_identity = (
                    name,
                    llm_action.search_mode,
                    " ".join(llm_action.text.split()).casefold(),
                )
                if request_identity in existing_requests:
                    trace.append(
                        {
                            "step": "skip_llm_supplement",
                            "provider": name,
                            "reason": "duplicate_provider_action_identity",
                        }
                    )
                    continue
                existing_requests.add(request_identity)
                if self._controller.stop_status() != "continue":
                    warnings.append("low-confidence supplement: budget unavailable")
                    trace.append(
                        {
                            "step": "llm_supplement_fallback",
                            "provider": name,
                            "reason": "budget_unavailable",
                        }
                    )
                    break
                try:
                    reservation = self._controller.reserve(
                        f"{name}.search:llm-low-confidence-v3",
                        self._provider_estimates.get(name, self._provider_estimate),
                    )
                except BudgetExceededError:
                    warnings.append("low-confidence supplement: budget unavailable")
                    trace.append(
                        {
                            "step": "llm_supplement_fallback",
                            "provider": name,
                            "reason": "budget_unavailable",
                        }
                    )
                    break
                try:
                    provider_filters = self._provider_search_filters(
                        analysis.search_plan.inherited_hard_filters
                    )
                    if name == "openalex" and llm_action.search_mode == "semantic":
                        provider_filters["_search_mode"] = "semantic"
                    result = await self._providers[name].search(
                        llm_action.text,
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
                            self._failure_diagnostic(name, code="integrity_failure")
                        )
                    elif self._controller.terminal_outcome(reservation) is None:
                        if (
                            self._controller.formal_live
                            and self._pricer is not None
                            and name in self._provider_adapter_names
                        ):
                            valued = self._pricer.value_actual(
                                dependency=name,
                                model_or_adapter=self._provider_adapter_names[name],
                                usage=UsageActual(search_api_calls=1),
                            )
                            self._controller.fail_closed(reservation, valued)
                        else:
                            self._controller.settle(
                                reservation,
                                UsageActual(search_api_calls=1),
                            )
                    warnings.append(f"low-confidence supplement: {name} exception")
                    trace.append(
                        {
                            "step": "llm_supplement_fallback",
                            "provider": name,
                            "reason": "provider_exception",
                        }
                    )
                    continue
                collected.setdefault(name, []).append(result)
                source_id = (
                    f"{name}:llm-low-confidence-v3:{llm_action.search_mode}"
                )
                action_results[source_id] = result
                llm_supplement_source_ids.add(source_id)
                trace.append(
                    {
                        "step": "retrieve_llm_supplement",
                        "provider": name,
                        "subquery_id": llm_selection.source_query_id,
                        "query_text": llm_action.text,
                        "action_type": llm_action.action_type,
                        "search_mode": llm_action.search_mode,
                        "result_count": len(result.data),
                        "error_count": len(result.errors),
                        "cache_hit": result.cache_hit,
                        "response_hash": result.provenance.get(
                            "response_hash", "unavailable"
                        ),
                    }
                )
                if result.errors:
                    warnings.append(
                        f"low-confidence supplement: {name} returned errors"
                    )
        provider_results = {
            name: self._combine_provider_results(results) for name, results in collected.items()
        }
        diagnostics.extend(
            self._diagnostic(name, result)
            for name, result in sorted(provider_results.items())
        )

        baseline_fusion_input = {
            source_id: result
            for source_id, result in action_results.items()
            if source_id not in supplement_source_ids
            and source_id not in llm_supplement_source_ids
        }
        supplemental_fusion_input = {
            source_id: action_results[source_id]
            for source_id in action_results
            if source_id in supplement_source_ids
        }
        llm_supplemental_fusion_input = {
            source_id: action_results[source_id]
            for source_id in action_results
            if source_id in llm_supplement_source_ids
        }
        baseline_raw_before = sum(
            len(result.data) for result in baseline_fusion_input.values()
        )
        supplemental_raw_before = sum(
            len(result.data) for result in supplemental_fusion_input.values()
        )
        llm_supplemental_raw_before = sum(
            len(result.data) for result in llm_supplemental_fusion_input.values()
        )
        baseline_fusion_input = self._round_robin_cap_action_results(
            baseline_fusion_input,
            self._max_raw_candidates,
        )
        supplemental_fusion_input = self._round_robin_cap_action_results(
            supplemental_fusion_input,
            self._max_additional_raw_candidates,
        )
        llm_supplemental_fusion_input = self._round_robin_cap_action_results(
            llm_supplemental_fusion_input,
            self._max_low_confidence_raw_candidates,
        )
        fusion_input = {**baseline_fusion_input, **supplemental_fusion_input}
        if self._max_total_raw_candidates is not None:
            fusion_input = self._round_robin_cap_action_results(
                fusion_input,
                self._max_total_raw_candidates,
            )
            baseline_fusion_input = {
                source_id: result
                for source_id, result in fusion_input.items()
                if source_id not in supplement_source_ids
            }
            supplemental_fusion_input = {
                source_id: result
                for source_id, result in fusion_input.items()
                if source_id in supplement_source_ids
            }
        baseline_raw_after = sum(
            len(result.data) for result in baseline_fusion_input.values()
        )
        supplemental_raw_after = sum(
            len(result.data) for result in supplemental_fusion_input.values()
        )
        llm_supplemental_raw_after = sum(
            len(result.data) for result in llm_supplemental_fusion_input.values()
        )
        if supplement_source_ids:
            baseline_fused = fuse_provider_results(
                cast(
                    Mapping[str, ProviderResult[list[Paper]]],
                    baseline_fusion_input,
                ),
                method="rrf",
                id_map=self._identifier_map,
            )
            supplemental_fused = fuse_provider_results(
                cast(
                    Mapping[str, ProviderResult[list[Paper]]],
                    supplemental_fusion_input,
                ),
                method="rrf",
                id_map=self._identifier_map,
            )
            fused = self._fair_merge_nonreinforcing_fused(
                baseline_fused,
                supplemental_fused,
            )
            trace.append(
                {
                    "step": "merge_supplement",
                    "policy": "nonreinforcing-fair-initial-order-v1",
                    "baseline_candidate_count": len(baseline_fused),
                    "supplemental_candidate_count": len(supplemental_fused),
                    "admitted_supplemental_candidate_count": (
                        len(fused) - len(baseline_fused)
                    ),
                }
            )
        else:
            fused = fuse_provider_results(
                cast(Mapping[str, ProviderResult[list[Paper]]], fusion_input),
                method="rrf",
                id_map=self._identifier_map,
            )
        if (
            self._max_deduplicated_candidates is not None
            and len(fused) > self._max_deduplicated_candidates
        ):
            fused = fused[: self._max_deduplicated_candidates]
        production_pool_count = len(fused)
        if llm_supplement_source_ids:
            llm_supplemental_fused = fuse_provider_results(
                cast(
                    Mapping[str, ProviderResult[list[Paper]]],
                    llm_supplemental_fusion_input,
                ),
                method="rrf",
                id_map=self._identifier_map,
            )
            fused = self._fair_merge_nonreinforcing_fused(
                fused,
                llm_supplemental_fused,
            )
            trace.append(
                {
                    "step": "merge_llm_supplement",
                    "policy": "independent-nonreinforcing-quota-v1",
                    "baseline_candidate_count": production_pool_count,
                    "supplemental_candidate_count": len(llm_supplemental_fused),
                    "admitted_supplemental_candidate_count": (
                        len(fused) - production_pool_count
                    ),
                    "baseline_evidence_immutable": True,
                }
            )
        deduplicated_before = len(fused)
        final_deduplicated_limit = self._max_deduplicated_candidates
        if (
            final_deduplicated_limit is not None
            and self._max_low_confidence_deduplicated_candidates is not None
        ):
            final_deduplicated_limit += (
                self._max_low_confidence_deduplicated_candidates
            )
        if (
            final_deduplicated_limit is not None
            and len(fused) > final_deduplicated_limit
        ):
            fused = fused[:final_deduplicated_limit]
        deduplication_input = [item.paper for item in fused]
        merged = deduplicate_papers(
            deduplication_input,
            id_map=self._identifier_map,
        )
        trace.append(
            {
                "step": "candidate_cap",
                "baseline_raw_before": baseline_raw_before,
                "baseline_raw_after": baseline_raw_after,
                "supplemental_raw_before": supplemental_raw_before,
                "supplemental_raw_after": supplemental_raw_after,
                "llm_supplemental_raw_before": llm_supplemental_raw_before,
                "llm_supplemental_raw_after": llm_supplemental_raw_after,
                "total_raw_after": (
                    baseline_raw_after
                    + supplemental_raw_after
                    + llm_supplemental_raw_after
                ),
                "deduplicated_before": deduplicated_before,
                "deduplicated_after": len(merged.papers),
                "max_raw_candidates": self._max_raw_candidates,
                "max_total_raw_candidates": self._max_total_raw_candidates,
                "max_deduplicated_candidates": self._max_deduplicated_candidates,
                "max_low_confidence_raw_candidates": (
                    self._max_low_confidence_raw_candidates
                ),
                "max_low_confidence_deduplicated_candidates": (
                    self._max_low_confidence_deduplicated_candidates
                ),
                "max_total_deduplicated_candidates": final_deduplicated_limit,
            }
        )
        trace.append(
            {
                "step": "deduplicate",
                "count": len(merged.papers),
                "merge_count": len(merged.decisions),
                "identifier_aliases_enabled": self._identifier_map is not None,
                "identifier_alias_count": self._identifier_alias_count,
            }
        )
        filtered = apply_hard_filters(merged.papers, analysis.query_spec)
        trace.append({"step": "filter", "accepted": len(filtered.accepted)})
        accepted_by_id = {
            item.paper.canonical_id: item for item in filtered.accepted
        }
        selected_fused = [
            item.model_copy(
                update={
                    "score": item.score
                    * accepted_by_id[item.paper.canonical_id].score_multiplier
                }
            )
            for item in fused
            if item.paper.canonical_id in accepted_by_id
        ]
        selected_fused.sort(key=lambda item: (-item.score, item.paper.canonical_id))
        reason_counts: dict[str, int] = {}
        penalized_count = 0
        for item in selected_fused:
            accepted = accepted_by_id[item.paper.canonical_id]
            if accepted.score_multiplier < 1.0:
                penalized_count += 1
            for reason in accepted.uncertainty_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        trace.append(
            {
                "step": "uncertainty_adjustment",
                "candidate_count": len(selected_fused),
                "penalized_count": penalized_count,
                "reason_counts": dict(sorted(reason_counts.items())),
            }
        )
        papers = [item.paper for item in selected_fused]
        trace.append({"step": "fuse", "count": len(papers)})
        if self._document_ranker is not None and selected_fused:
            prior_ids = [item.paper.canonical_id for item in selected_fused]
            contextual_rank = getattr(self._document_ranker, "rank_with_context", None)
            ranked_fused = (
                contextual_rank(
                    analysis.query_spec.original_query,
                    selected_fused,
                    query_spec=analysis.query_spec,
                )
                if callable(contextual_rank)
                else self._document_ranker.rank(
                    analysis.query_spec.original_query,
                    selected_fused,
                )
            )
            ranked_ids = [item.paper.canonical_id for item in ranked_fused]
            if len(ranked_ids) != len(prior_ids) or set(ranked_ids) != set(prior_ids):
                raise ValueError("document ranker changed candidate identity")
            selected_fused = ranked_fused
            papers = [item.paper for item in selected_fused]
            rank_trace: dict[str, object] = {
                "step": "document_rank",
                "status": "applied",
                "model_id": self._document_ranker.model_id,
                "count": len(papers),
            }
            context_receipt = getattr(self._document_ranker, "context_receipt", None)
            if callable(context_receipt):
                receipt = context_receipt(
                    analysis.query_spec.original_query,
                    query_spec=analysis.query_spec,
                )
                if receipt is not None:
                    rank_trace["query_context"] = receipt
            deployment_role = getattr(self._document_ranker, "deployment_role", None)
            if deployment_role is not None:
                rank_trace["deployment_role"] = deployment_role
            failover_receipt = getattr(self._document_ranker, "failover_receipt", None)
            if failover_receipt:
                rank_trace["failover_receipt"] = failover_receipt
            trace.append(rank_trace)
        high_relevance: list[RankedPaper] = []
        partial_relevance: list[RankedPaper] = []
        citation_edges: list[ResolvedCitationEdge] = []
        pre_truncation_candidates = list(papers)
        if self._max_output_papers is not None and len(papers) > self._max_output_papers:
            selected_fused = selected_fused[: self._max_output_papers]
            papers = [item.paper for item in selected_fused]
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
