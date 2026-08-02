"""Canonical request-scoped search application boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Protocol
from uuid import uuid4

from paper_search.application.contracts import (
    DependencyDiagnostic,
    SearchErrorCode,
    SearchErrorResponse,
    SearchExecutionResult,
    SearchFailure,
    SearchRequest,
    SearchSuccess,
)
from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import (
    SearchBudget,
    SearchMode,
    StructuredSearchResponse,
    UsageActual,
)
from paper_search.pipeline.orchestrator import OrchestratorResult
from paper_search.pipeline.response import to_structured_response


class SearchOrchestrator(Protocol):
    async def run(
        self,
        query: str,
        *,
        max_provider_results: int,
    ) -> OrchestratorResult: ...


OrchestratorFactory = Callable[[HardBudgetController, str], SearchOrchestrator]
RunIdFactory = Callable[[], str]

_SAFE_WARNING_EXACT = frozenset(
    {
        "analysis: analyzer returned errors",
        "analysis: dependency failure",
        "analysis: budget unavailable",
        "planner_rules_fallback",
        "citation: expansion_unavailable",
        "citation: unresolved_citation_edge",
        "rerank: rerank_unavailable",
        "embedding: encoder_unavailable",
        "embedding: cuda_oom_cpu_fallback",
        "embedding: unsanitized_warning",
    }
)
_SAFE_PROVIDER_WARNING_SUFFIXES = frozenset(
    {
        "budget unavailable",
        "provider exception",
        "provider returned errors",
    }
)
_MALFORMED_LLM_CODES = frozenset(
    {"invalid_json", "invalid_response", "empty_response"}
)
_SAFE_ERROR_DETAILS: dict[SearchErrorCode, str] = {
    "invalid_request": "The search request is invalid",
    "live_not_authorized": "Live search is not authorized",
    "config_mismatch": "The requested mode does not match the application binding",
    "snapshot_unavailable": "Required replay data is unavailable",
    "budget_exhausted": "The search budget was exhausted",
    "dependency_failure": "A required search dependency failed",
    "integrity_failure": "Search integrity validation failed",
    "validation_attempt_conflict": "The validation attempt conflicts with prior state",
    "internal_error": "The search could not be completed",
}


class SearchApplicationError(RuntimeError):
    """Typed application failure carrying only safe public state."""

    def __init__(
        self,
        error: SearchErrorResponse,
        usage: UsageActual,
        diagnostics: list[DependencyDiagnostic],
    ) -> None:
        super().__init__(error.detail)
        self.error = error
        self.usage = usage
        self.diagnostics = diagnostics


class SearchApplicationService:
    def __init__(
        self,
        *,
        orchestrator_factory: OrchestratorFactory,
        budgets: Mapping[str, SearchBudget],
        mode: SearchMode,
        snapshot_set_id: str,
        snapshot_captured_at: datetime | None,
        git_sha: str,
        max_provider_results: int,
        run_id_factory: RunIdFactory | None = None,
    ) -> None:
        self._orchestrator_factory = orchestrator_factory
        self._budgets = dict(budgets)
        self._mode = mode
        self._snapshot_set_id = snapshot_set_id
        self._snapshot_captured_at = snapshot_captured_at
        self._git_sha = git_sha
        self._max_provider_results = max_provider_results
        self._run_id_factory = run_id_factory or (lambda: str(uuid4()))

    @staticmethod
    def _safe_warnings(warnings: list[str]) -> list[str]:
        safe: list[str] = []
        for warning in warnings:
            if warning in _SAFE_WARNING_EXACT:
                safe.append(warning)
                continue
            dependency, separator, suffix = warning.partition(": ")
            if (
                separator
                and dependency in {"openalex", "semantic_scholar"}
                and suffix in _SAFE_PROVIDER_WARNING_SUFFIXES
            ):
                safe.append(warning)
        return safe

    @staticmethod
    def _failure_code(result: OrchestratorResult) -> SearchErrorCode | None:
        llm_errors = {
            error.code
            for diagnostic in result.diagnostics
            if diagnostic.dependency == "llm"
            for error in diagnostic.errors
        }
        if "snapshot_unavailable" in llm_errors:
            return "snapshot_unavailable"
        if "integrity_failure" in llm_errors:
            return "integrity_failure"
        if llm_errors and not (
            result.planner_status == "rules_fallback"
            and llm_errors.issubset(_MALFORMED_LLM_CODES)
        ):
            return "dependency_failure"

        provider_error_codes = {
            error.code
            for diagnostic in result.diagnostics
            if diagnostic.dependency in {"openalex", "semantic_scholar"}
            for error in diagnostic.errors
        }
        if "integrity_failure" in provider_error_codes:
            return "integrity_failure"
        if "snapshot_unavailable" in provider_error_codes:
            return "snapshot_unavailable"

        provider_failures = {
            diagnostic.dependency
            for diagnostic in result.diagnostics
            if diagnostic.dependency in {"openalex", "semantic_scholar"}
            and diagnostic.errors
        }
        if not result.papers and provider_failures == {
            "openalex",
            "semantic_scholar",
        }:
            return "dependency_failure"
        if not result.papers and result.stop_reason in {"hard_stop", "soft_stop"}:
            return "budget_exhausted"
        if result.stop_reason == "snapshot_unavailable":
            return "snapshot_unavailable"
        if result.stop_reason == "dependency_failure":
            return "dependency_failure"
        return None

    @staticmethod
    def _business_payload(
        *,
        request: SearchRequest,
        result: OrchestratorResult | None,
        failure_code: SearchErrorCode | None,
        warnings: list[str],
    ) -> dict[str, object]:
        if result is None or failure_code is not None:
            return {
                "schema_version": "business-result-v1",
                "query_id": request.query_id,
                "query_analysis": None,
                "selected_paper_ids": [],
                "high_relevance": [],
                "partial_relevance": [],
                "citation_edges": [],
                "is_partial": False,
                "planner_status": None,
                "planner_fallback": False,
                "warnings": [],
                "stop_reason": result.stop_reason if result is not None else "internal_error",
                "hard_failure_code": failure_code or "internal_error",
            }
        planner_fallback = result.planner_status == "rules_fallback"
        return {
            "schema_version": "business-result-v1",
            "query_id": request.query_id,
            "query_analysis": result.query_analysis.model_dump(mode="json"),
            "selected_paper_ids": [paper.canonical_id for paper in result.papers],
            "fused_papers": [
                item.model_dump(mode="json") for item in result.fused_papers
            ],
            "high_relevance": [
                item.model_dump(mode="json") for item in result.high_relevance
            ],
            "partial_relevance": [
                item.model_dump(mode="json") for item in result.partial_relevance
            ],
            "citation_edges": [
                item.model_dump(mode="json") for item in result.citation_edges
            ],
            "is_partial": result.is_partial or planner_fallback,
            "planner_status": result.planner_status,
            "planner_fallback": planner_fallback,
            "warnings": warnings,
            "stop_reason": result.stop_reason,
            "hard_failure_code": None,
        }

    @classmethod
    def _business_hash(
        cls,
        *,
        request: SearchRequest,
        result: OrchestratorResult | None,
        failure_code: SearchErrorCode | None,
        warnings: list[str],
    ) -> str:
        payload = cls._business_payload(
            request=request,
            result=result,
            failure_code=failure_code,
            warnings=warnings,
        )
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    @staticmethod
    def _failure(
        *,
        request: SearchRequest,
        run_id: str,
        code: SearchErrorCode,
        usage: UsageActual,
        stop_reason: str,
        diagnostics: list[DependencyDiagnostic],
        result: OrchestratorResult | None,
    ) -> SearchExecutionResult:
        error = SearchErrorResponse(
            code=code,
            detail=_SAFE_ERROR_DETAILS[code],
            retryable=code in {"snapshot_unavailable", "dependency_failure"},
            run_id=run_id,
        )
        return SearchExecutionResult(
            outcome=SearchFailure(
                query_id=request.query_id,
                run_id=run_id,
                error=error,
                usage=usage,
                stop_reason=stop_reason,
            ),
            diagnostics=diagnostics,
            business_result_sha256=SearchApplicationService._business_hash(
                request=request,
                result=result,
                failure_code=code,
                warnings=[],
            ),
        )

    async def execute(self, request: SearchRequest) -> SearchExecutionResult:
        run_id = self._run_id_factory()
        if request.mode != self._mode:
            code: SearchErrorCode = (
                "live_not_authorized" if request.mode == "live" else "config_mismatch"
            )
            return self._failure(
                request=request,
                run_id=run_id,
                code=code,
                usage=UsageActual(),
                stop_reason=code,
                diagnostics=[],
                result=None,
            )
        budget = self._budgets.get(request.budget_profile)
        if budget is None:
            return self._failure(
                request=request,
                run_id=run_id,
                code="invalid_request",
                usage=UsageActual(),
                stop_reason="invalid_request",
                diagnostics=[],
                result=None,
            )
        controller = HardBudgetController(
            budget,
            formal_live=self._mode == "live",
        )
        try:
            orchestrator = self._orchestrator_factory(controller, run_id)
            result = await orchestrator.run(
                request.query,
                max_provider_results=self._max_provider_results,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            return self._failure(
                request=request,
                run_id=run_id,
                code="internal_error",
                usage=controller.committed_usage,
                stop_reason="internal_error",
                diagnostics=[],
                result=None,
            )

        warnings = self._safe_warnings(result.warnings)
        if result.planner_status == "rules_fallback" and "planner_rules_fallback" not in warnings:
            warnings.append("planner_rules_fallback")
        result = result.model_copy(
            update={
                "warnings": warnings,
                "is_partial": result.is_partial
                or result.planner_status == "rules_fallback",
            }
        )
        failure_code = self._failure_code(result)
        if failure_code is not None:
            return self._failure(
                request=request,
                run_id=run_id,
                code=failure_code,
                usage=result.usage,
                stop_reason=result.stop_reason,
                diagnostics=result.diagnostics,
                result=result,
            )
        response = to_structured_response(
            result,
            query_id=request.query_id,
            git_sha=self._git_sha,
            run_id=run_id,
            execution_mode=self._mode,
            snapshot_set_id=result.snapshot_set_id or self._snapshot_set_id,
            snapshot_captured_at=(
                result.snapshot_captured_at or self._snapshot_captured_at
            ),
            include_trace=request.include_trace,
        )
        return SearchExecutionResult(
            outcome=SearchSuccess(response=response),
            diagnostics=result.diagnostics,
            business_result_sha256=self._business_hash(
                request=request,
                result=result,
                failure_code=None,
                warnings=warnings,
            ),
        )

    async def __call__(self, request: SearchRequest) -> StructuredSearchResponse:
        execution = await self.execute(request)
        if isinstance(execution.outcome, SearchFailure):
            raise SearchApplicationError(
                execution.outcome.error,
                execution.outcome.usage,
                execution.diagnostics,
            )
        return execution.outcome.response
