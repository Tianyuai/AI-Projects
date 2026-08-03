from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from paper_search.control.budget import (
    BudgetExceededError,
    HardBudgetController,
    ReservationError,
)
from paper_search.application.contracts import DependencyDiagnostic, SearchRequest, SnapshotRef
from paper_search.application.experiments import OptionalStageUnavailableError
from paper_search.application.service import SearchApplicationService
from paper_search.domain.models import (
    BudgetReservation,
    CitationEdge,
    CitationExpansion,
    ErrorDetail,
    Paper,
    ProviderPaperId,
    ProviderResult,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.graph.citation_expand import CitationExpansionResult
from paper_search.evaluation.execution_adapter import adapt_execution
from paper_search.graph.provider_stage import (
    CitationExpansionUnavailableError,
    ProviderCitationExpansionStage,
)
from paper_search.llm.snapshot_adapters import LLMAdapterError
from paper_search.pipeline.orchestrator import MockSearchOrchestrator
from paper_search.ranking.embedding import EmbeddingRankingResult, EmbeddingScore
from paper_search.ranking.rerank import (
    ConstraintRerankResult,
    ConstraintScoredPaper,
)
from paper_search.ranking.llm_stage import LLMConstraintRerankingStage
from paper_search.retrieval.snapshot_adapters import ProviderAdapterError


def _budget(**updates: object) -> SearchBudget:
    values = {
        "max_search_api_calls": 6,
        "target_search_api_calls": 1,
        "max_llm_calls": 2,
        "target_llm_calls": 1,
        "max_total_tokens": 100,
        "max_cost_cny": 1.0,
        "max_elapsed_seconds": 2,
        "soft_deadline_seconds": 1,
    }
    values.update(updates)
    return SearchBudget.model_validate(values)


def _result(provider: str, data: Any, usage: UsageActual, *, failed: bool = False) -> ProviderResult[Any]:
    return ProviderResult[Any](
        data=data,
        usage=usage,
        provenance={
            "provider": provider,
            "endpoint": "/synthetic",
            "model_id": "fixture",
            "requested_at": datetime(2026, 7, 23, tzinfo=UTC).isoformat(),
            "response_hash": f"sha256:{provider}",
        },
        cache_hit=False,
        latency_ms=0,
        errors=(
            [ErrorDetail(code="timeout", message="synthetic", retryable=True, provider=provider)]
            if failed
            else []
        ),
    )


class FakeAnalyzer:
    def __init__(self, events: list[str], *, elapsed_ms: int = 0) -> None:
        self.events = events
        self.elapsed_ms = elapsed_ms

    async def __call__(self, query: str, _: object) -> ProviderResult[dict[str, object]]:
        self.events.append("analyze")
        return _result(
            "llm",
            {
                "query_spec": {"original_query": query, "research_goal": "find papers"},
                "search_plan": {
                "subqueries": [
                    {
                        "query_id": "model-1",
                        "text": f"{query} openalex",
                        "query_type": "exact",
                        "target_constraints": ["papers"],
                        "priority": 1,
                        "provider_hint": "openalex",
                    },
                    {
                        "query_id": "model-2",
                        "text": f"{query} semantic",
                        "query_type": "decomposed",
                        "target_constraints": ["papers"],
                        "priority": 2,
                        "provider_hint": "semantic_scholar",
                    },
                    {
                        "query_id": "model-3",
                        "text": query,
                        "query_type": "expanded",
                        "target_constraints": ["papers"],
                        "priority": 3,
                        "provider_hint": "either",
                    },
                ],
                    "inherited_hard_filters": {},
                    "rationale": "fixture",
                },
            },
            UsageActual(llm_calls=1, cost_cny=0.1, elapsed_ms=self.elapsed_ms),
        )


class FailedAnalyzer:
    def __init__(self, events: list[str], *, raises: bool = False) -> None:
        self.events = events
        self.raises = raises

    async def __call__(
        self, query: str, _: object
    ) -> ProviderResult[dict[str, object]]:
        self.events.append("analyze")
        if self.raises:
            raise TimeoutError("synthetic analyzer timeout")
        return _result(
            "llm",
            {},
            UsageActual(llm_calls=1, cost_cny=0.1),
            failed=True,
        )


class FakeProvider:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        failed: bool = False,
        empty: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.failed = failed
        self.empty = empty

    async def search(self, query: str, filters: dict[str, object], limit: int, reservation: object) -> ProviderResult[list[Paper]]:
        assert query
        assert filters == {}
        assert limit == 5
        assert reservation is not None
        self.events.append(self.name)
        paper = Paper(
            canonical_id="openalex:W1" if self.name == "openalex" else "s2:S1",
            title=f"{self.name} paper",
            openalex_id="W1" if self.name == "openalex" else None,
            semantic_scholar_id="S1" if self.name != "openalex" else None,
            sources=[self.name],
        )
        return _result(
            self.name,
            [] if self.failed or self.empty else [paper],
            UsageActual(search_api_calls=1),
            failed=self.failed,
        )


class SettlingAnalyzer(FakeAnalyzer):
    def __init__(
        self,
        events: list[str],
        controller: HardBudgetController,
    ) -> None:
        super().__init__(events)
        self.controller = controller

    async def __call__(
        self,
        query: str,
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, object]]:
        result = await super().__call__(query, reservation)
        self.controller.settle(reservation, result.usage)
        return result


class SettlingProvider(FakeProvider):
    def __init__(
        self,
        name: str,
        events: list[str],
        controller: HardBudgetController,
    ) -> None:
        super().__init__(name, events)
        self.controller = controller

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: object,
    ) -> ProviderResult[list[Paper]]:
        assert isinstance(reservation, BudgetReservation)
        result = await super().search(query, filters, limit, reservation)
        self.controller.settle(reservation, result.usage)
        return result


class IntegrityProvider(FakeProvider):
    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: object,
    ) -> ProviderResult[list[Paper]]:
        del query, filters, limit, reservation
        raise ValueError("snapshot response hash mismatch")


class SnapshotProvider(FakeProvider):
    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: object,
    ) -> ProviderResult[list[Paper]]:
        result = await super().search(query, filters, limit, reservation)
        index = len(self.events)
        ref = SnapshotRef(
            entry_id=f"entry-{index}",
            dependency=self.name,
            cache_key="sha256:" + f"{index:x}" * 64,
            response_sha256="sha256:" + f"{index:x}" * 64,
            captured_at=datetime(2026, 7, 23, tzinfo=UTC),
            snapshot_path=f"responses/{self.name}/{index}.bin",
        )
        return result.model_copy(
            update={
                "cache_hit": True,
                "provenance": {
                    **result.provenance,
                    "snapshot_refs": json.dumps(
                        [ref.model_dump(mode="json")],
                        separators=(",", ":"),
                    ),
                },
            }
        )


class RaisingProvider:
    def __init__(self, name: str, events: list[str], error: Exception) -> None:
        self.name = name
        self.events = events
        self.error = error

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: object,
    ) -> ProviderResult[list[Paper]]:
        assert query
        assert filters == {}
        assert limit == 5
        assert reservation is not None
        self.events.append(self.name)
        raise self.error


class FakeEmbeddingRanker:
    def __init__(
        self,
        *,
        degraded: bool = False,
        reverse_on_degraded: bool = False,
    ) -> None:
        self.degraded = degraded
        self.reverse_on_degraded = reverse_on_degraded
        self.calls: list[tuple[str, list[str]]] = []

    def rank(
        self,
        query: str,
        papers: Sequence[Paper],
    ) -> EmbeddingRankingResult:
        self.calls.append((query, [paper.canonical_id for paper in papers]))
        if self.degraded and not self.reverse_on_degraded:
            ordered = list(papers)
        else:
            ordered = list(reversed(papers))
        return EmbeddingRankingResult(
            ranked=[
                EmbeddingScore(paper=paper, similarity=0.0 if self.degraded else 0.8)
                for paper in ordered
            ],
            status="degraded" if self.degraded else "applied",
            model_id="fixture-embedding-v1",
            device="cpu",
            fallback_used=False,
            warnings=["encoder_unavailable"] if self.degraded else [],
        )


class MaliciousEmbeddingRanker:
    def rank(
        self,
        query: str,
        papers: Sequence[Paper],
    ) -> EmbeddingRankingResult:
        private_warning = (
            f"query={query}; ids={','.join(paper.canonical_id for paper in papers)}; "
            r"path=D:\private-cache\secret-model"
        )
        private_code = "query_graph_retrieval_ids_openalex_w1_s2_s1_private_cache"
        return EmbeddingRankingResult(
            ranked=[EmbeddingScore(paper=paper, similarity=0.0) for paper in papers],
            status="degraded",
            model_id=r"D:\private-cache\secret-model",
            device="cpu",
            fallback_used=True,
            warnings=["cuda_oom_cpu_fallback", private_warning, private_code],
        )


class FakeCitationExpander:
    def __init__(self, extra: Paper) -> None:
        self.extra = extra
        self.calls: list[list[str]] = []

    async def expand(
        self,
        seeds: list[Paper],
        *,
        controller: HardBudgetController,
    ) -> CitationExpansionResult:
        assert controller is not None
        self.calls.append([paper.canonical_id for paper in seeds])
        return CitationExpansionResult(
            papers=[*seeds, self.extra],
            edges=[],
            skipped_edge_count=0,
            truncated=False,
            warnings=[],
        )


class FakeConstraintReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    async def rerank(
        self,
        papers: list[Paper],
        constraints: list[str],
        *,
        controller: HardBudgetController,
    ) -> ConstraintRerankResult:
        assert controller is not None
        self.calls.append(
            ([paper.canonical_id for paper in papers], list(constraints))
        )
        ranked = [
            ConstraintScoredPaper(
                paper=paper,
                score=0.5,
                assessment={
                    "matched_constraint_count": 0,
                    "unmatched_constraint_count": 0,
                    "relevance_score": 0.5,
                    "constraint_coverage": 0.0,
                },
            )
            for paper in reversed(papers)
        ]
        return ConstraintRerankResult(
            ranked=ranked,
            status="applied",
            processed_count=len(ranked),
            truncated=False,
            batch_count=1 if ranked else 0,
            warnings=[],
        )


class CitationProvider:
    def __init__(self, *, failed: bool = False, error_code: str = "timeout") -> None:
        self.failed = failed
        self.error_code = error_code
        self.calls: list[tuple[str, str, BudgetReservation]] = []

    def _expansion_result(
        self,
        direction: str,
        paper_id: ProviderPaperId,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        self.calls.append((direction, paper_id.value, reservation))
        index = len(self.calls)
        expanded_id = "S2" if direction == "references" else "S3"
        expanded = Paper(
            canonical_id=f"s2:{expanded_id}",
            title=f"Expanded {expanded_id}",
            semantic_scholar_id=expanded_id,
        )
        if direction == "references":
            citing, cited = paper_id, ProviderPaperId(
                provider="semantic_scholar",
                value=expanded_id,
            )
        else:
            citing, cited = (
                ProviderPaperId(provider="semantic_scholar", value=expanded_id),
                paper_id,
            )
        ref = SnapshotRef(
            entry_id=f"citation-{index}",
            dependency="semantic_scholar",
            cache_key="sha256:" + f"{index:x}" * 64,
            response_sha256="sha256:" + f"{index:x}" * 64,
            captured_at=datetime(2026, 7, 23, tzinfo=UTC),
            snapshot_path=f"responses/semantic_scholar/{index}.bin",
        )
        result = _result(
            "semantic_scholar",
            CitationExpansion(
                papers=[expanded],
                raw_edges=[
                    CitationEdge(
                        provider="semantic_scholar",
                        citing_provider_id=citing,
                        cited_provider_id=cited,
                    )
                ],
            ),
            UsageActual(search_api_calls=1),
            failed=self.failed,
        )
        if self.failed and self.error_code != "timeout":
            result = result.model_copy(
                update={
                    "errors": [
                        ErrorDetail(
                            code=self.error_code,
                            message="synthetic",
                            retryable=False,
                            provider="semantic_scholar",
                        )
                    ]
                }
            )
        return result.model_copy(
            update={
                "cache_hit": direction == "citations",
                "provenance": {
                    **result.provenance,
                    "snapshot_refs": json.dumps(
                        [ref.model_dump(mode="json")],
                        separators=(",", ":"),
                    ),
                },
            }
        )

    async def references(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        assert limit == 2
        return self._expansion_result("references", paper_id, reservation)

    async def citations(
        self,
        paper_id: ProviderPaperId,
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[CitationExpansion]:
        assert limit == 2
        return self._expansion_result("citations", paper_id, reservation)


class RerankAnalyzer:
    def __init__(self, *, failed: bool = False, error_code: str = "timeout") -> None:
        self.failed = failed
        self.error_code = error_code
        self.calls: list[tuple[dict[str, object], BudgetReservation]] = []

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]:
        assert prompt_name == "constraint_rerank"
        self.calls.append((payload, reservation))
        papers = payload["papers"]
        assert isinstance(papers, list)
        constraints = payload["constraints"]
        assert isinstance(constraints, list)
        data = {
            "assessments": [
                {
                    "paper_id": paper["canonical_id"],
                    "matched_constraint_count": len(constraints),
                    "unmatched_constraint_count": 0,
                    "relevance_score": 1.0 if index == 1 else 0.5,
                }
                for index, paper in enumerate(papers)
            ]
        }
        result = _result(
            "llm",
            data,
            UsageActual(llm_calls=1, input_tokens=10, output_tokens=10, cost_cny=0.1),
            failed=False,
        )
        if self.failed:
            result = result.model_copy(
                update={
                    "errors": [
                        ErrorDetail(
                            code=self.error_code,
                            message="synthetic",
                            retryable=self.error_code == "timeout",
                            provider="llm",
                        )
                    ]
                }
            )
        ref = SnapshotRef(
            entry_id="rerank-1",
            dependency="llm",
            cache_key="sha256:" + "a" * 64,
            response_sha256="sha256:" + "b" * 64,
            captured_at=datetime(2026, 7, 23, tzinfo=UTC),
            snapshot_path="responses/llm/rerank.bin",
        )
        return result.model_copy(
            update={
                "cache_hit": True,
                "provenance": {
                    **result.provenance,
                    "snapshot_entry_id": ref.entry_id,
                    "snapshot_cache_key": ref.cache_key,
                    "snapshot_response_sha256": ref.response_sha256,
                    "snapshot_path": ref.snapshot_path,
                },
            }
        )


class OverrunRerankAnalyzer(RerankAnalyzer):
    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]:
        result = await super().generate_json(
            prompt_name=prompt_name,
            payload=payload,
            reservation=reservation,
        )
        return result.model_copy(
            update={
                "usage": result.usage.model_copy(update={"llm_calls": 2}),
            }
        )


def test_provider_citation_stage_awaits_budgeted_calls_and_retains_snapshot_refs() -> None:
    provider = CitationProvider()
    controller = HardBudgetController(_budget(max_citation_seeds=1))
    stage = ProviderCitationExpansionStage(
        provider=provider,
        call_estimate=UsageEstimate(search_api_calls=1),
        per_direction_limit=2,
        max_expanded=2,
    )
    seed = Paper(
        canonical_id="s2:S1",
        title="Seed",
        semantic_scholar_id="S1",
    )

    result = asyncio.run(stage.expand([seed], controller=controller))

    assert [(direction, paper_id) for direction, paper_id, _ in provider.calls] == [
        ("references", "S1"),
        ("citations", "S1"),
    ]
    assert [paper.canonical_id for paper in result.papers] == [
        "s2:S1",
        "s2:S2",
        "s2:S3",
    ]
    assert [ref.entry_id for ref in result.snapshot_refs] == [
        "citation-1",
        "citation-2",
    ]
    assert controller.committed_usage.search_api_calls == 2


def test_structured_provider_citation_failure_degrades_in_orchestrator() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget(max_citation_seeds=1))
    stage = ProviderCitationExpansionStage(
        provider=CitationProvider(failed=True),
        call_estimate=UsageEstimate(search_api_calls=1),
        per_direction_limit=2,
        max_expanded=2,
    )
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FakeAnalyzer(events),
        providers={
            "semantic_scholar": FakeProvider("semantic_scholar", events),
        },
        config_hash="sha256:" + "c" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        citation_expander=stage,
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert [paper.canonical_id for paper in result.papers] == ["s2:S1"]
    assert result.warnings[-1] == "citation: expansion_unavailable"
    assert result.trace[-1] == {
        "step": "citation",
        "status": "degraded",
        "count": 1,
    }
    assert result.stop_reason == "completed"
    assert controller.committed_usage.search_api_calls == 3


def test_llm_rerank_stage_awaits_same_controller_without_nested_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer = RerankAnalyzer()
    controller = HardBudgetController(_budget(max_rerank_candidates=2))
    stage = LLMConstraintRerankingStage(
        analyzer=analyzer,
        call_estimate=UsageEstimate(
            llm_calls=1,
            input_tokens=10,
            output_tokens=10,
            cost_cny=0.1,
        ),
    )
    papers = [
        Paper(canonical_id="paper:1", title="One"),
        Paper(canonical_id="paper:2", title="Two"),
    ]

    async def scenario() -> ConstraintRerankResult:
        def forbidden(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("production code must not call asyncio.run")

        monkeypatch.setattr(asyncio, "run", forbidden)
        return await stage.rerank(
            papers,
            ["graph retrieval"],
            controller=controller,
        )

    result = asyncio.run(scenario())

    assert [item.paper.canonical_id for item in result.ranked] == [
        "paper:2",
        "paper:1",
    ]
    assert [ref.entry_id for ref in result.snapshot_refs] == ["rerank-1"]
    assert controller.committed_usage.llm_calls == 1


def test_llm_rerank_stage_degrades_on_structured_dependency_failure() -> None:
    analyzer = RerankAnalyzer(failed=True)
    controller = HardBudgetController(_budget(max_rerank_candidates=2))
    stage = LLMConstraintRerankingStage(
        analyzer=analyzer,
        call_estimate=UsageEstimate(
            llm_calls=1,
            input_tokens=10,
            output_tokens=10,
            cost_cny=0.1,
        ),
    )
    papers = [Paper(canonical_id="paper:1", title="One")]

    result = asyncio.run(
        stage.rerank(papers, ["constraint"], controller=controller)
    )

    assert result.status == "degraded"
    assert [item.paper for item in result.ranked] == papers
    assert result.warnings == ["rerank_unavailable"]
    assert controller.committed_usage.llm_calls == 1


def test_expected_llm_rerank_timeout_degrades_without_failing_orchestrator() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget(max_llm_calls=3))
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "5" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        constraint_reranker=LLMConstraintRerankingStage(
            analyzer=RerankAnalyzer(failed=True),
            call_estimate=UsageEstimate(
                llm_calls=1,
                input_tokens=10,
                output_tokens=10,
                cost_cny=0.1,
            ),
        ),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert result.stop_reason == "completed"
    assert result.warnings[-1] == "rerank: rerank_unavailable"
    assert result.diagnostics[-1].errors[0].code == "timeout"


def test_optional_rerank_authentication_diagnostic_is_preserved() -> None:
    events: list[str] = []

    class AuthUnavailableRerank:
        async def rerank(
            self,
            papers: list[Paper],
            constraints: list[str],
            *,
            controller: HardBudgetController,
        ) -> ConstraintRerankResult:
            del papers, constraints, controller
            diagnostic = DependencyDiagnostic(
                dependency="llm",
                endpoint="constraint_rerank",
                model_id=None,
                usage=UsageActual(),
                latency_ms=0,
                cache_hit=False,
                snapshot_refs=[],
                errors=[
                    ErrorDetail(
                        code="authentication_error",
                        message="synthetic auth failure",
                        retryable=False,
                        provider="llm",
                    )
                ],
            )
            error = OptionalStageUnavailableError("rerank unavailable")
            error.diagnostic = diagnostic
            raise error

    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "6" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        constraint_reranker=AuthUnavailableRerank(),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert result.stop_reason == "dependency_failure"
    assert result.diagnostics[-1].errors[0].code == "authentication_error"


@pytest.mark.parametrize(
    "error",
    [
        ReservationError("reservation mismatch"),
        BudgetExceededError("budget exhausted"),
        ProviderAdapterError("authentication failed"),
        ValueError("malformed provider snapshot provenance"),
    ],
)
def test_optional_citation_propagates_protected_failures(error: Exception) -> None:
    events: list[str] = []

    class FailingCitation:
        async def expand(
            self,
            seeds: list[Paper],
            *,
            controller: HardBudgetController,
        ) -> CitationExpansionResult:
            del seeds, controller
            raise error

    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "4" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        citation_expander=FailingCitation(),
    )

    with pytest.raises(type(error), match=str(error)):
        asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))


@pytest.mark.parametrize(
    "error",
    [
        ReservationError("reservation mismatch"),
        BudgetExceededError("budget exhausted"),
        LLMAdapterError("authentication failed"),
        ValueError("incomplete LLM snapshot provenance"),
    ],
)
def test_optional_rerank_propagates_protected_failures(error: Exception) -> None:
    events: list[str] = []

    class FailingRerank:
        async def rerank(
            self,
            papers: list[Paper],
            constraints: list[str],
            *,
            controller: HardBudgetController,
        ) -> ConstraintRerankResult:
            del papers, constraints, controller
            raise error

    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "3" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        constraint_reranker=FailingRerank(),
    )

    with pytest.raises(type(error), match=str(error)):
        asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))


def test_citation_cancellation_terminally_fails_active_reservation() -> None:
    class CancellingProvider(CitationProvider):
        reservation: BudgetReservation | None = None

        async def references(
            self,
            paper_id: ProviderPaperId,
            limit: int,
            reservation: BudgetReservation,
        ) -> ProviderResult[CitationExpansion]:
            del paper_id, limit
            self.reservation = reservation
            raise asyncio.CancelledError

    provider = CancellingProvider()
    controller = HardBudgetController(_budget(max_citation_seeds=1))
    stage = ProviderCitationExpansionStage(
        provider=provider,
        call_estimate=UsageEstimate(search_api_calls=1),
    )
    seed = Paper(
        canonical_id="s2:S1",
        title="Seed",
        semantic_scholar_id="S1",
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(stage.expand([seed], controller=controller))

    assert provider.reservation is not None
    assert controller.terminal_outcome(provider.reservation) == (
        "failed",
        UsageActual(),
    )
    assert controller.reserved_usage.search_api_calls == 0


def test_llm_rerank_cancellation_terminally_fails_active_reservation() -> None:
    class CancellingAnalyzer:
        reservation: BudgetReservation | None = None

        async def generate_json(
            self,
            *,
            prompt_name: str,
            payload: dict[str, object],
            reservation: BudgetReservation,
        ) -> ProviderResult[dict[str, Any]]:
            del prompt_name, payload
            self.reservation = reservation
            raise asyncio.CancelledError

    analyzer = CancellingAnalyzer()
    controller = HardBudgetController(_budget(max_rerank_candidates=1))
    stage = LLMConstraintRerankingStage(
        analyzer=analyzer,
        call_estimate=UsageEstimate(
            llm_calls=1,
            input_tokens=10,
            output_tokens=10,
            cost_cny=0.1,
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            stage.rerank(
                [Paper(canonical_id="paper:1", title="One")],
                ["constraint"],
                controller=controller,
            )
        )

    assert analyzer.reservation is not None
    assert controller.terminal_outcome(analyzer.reservation) == (
        "failed",
        UsageActual(),
    )
    assert controller.reserved_usage.llm_calls == 0


def test_citation_stage_propagates_adapter_authentication_failure() -> None:
    class AuthenticationFailureProvider(CitationProvider):
        async def references(
            self,
            paper_id: ProviderPaperId,
            limit: int,
            reservation: BudgetReservation,
        ) -> ProviderResult[CitationExpansion]:
            del paper_id, limit, reservation
            raise ProviderAdapterError("authentication failed")

    controller = HardBudgetController(_budget(max_citation_seeds=1))
    stage = ProviderCitationExpansionStage(
        provider=AuthenticationFailureProvider(),
        call_estimate=UsageEstimate(search_api_calls=1),
    )

    with pytest.raises(ProviderAdapterError, match="authentication failed"):
        asyncio.run(
            stage.expand(
                [
                    Paper(
                        canonical_id="s2:S1",
                        title="Seed",
                        semantic_scholar_id="S1",
                    )
                ],
                controller=controller,
            )
        )


def test_llm_rerank_stage_propagates_adapter_authentication_failure() -> None:
    class AuthenticationFailureAnalyzer:
        async def generate_json(
            self,
            *,
            prompt_name: str,
            payload: dict[str, object],
            reservation: BudgetReservation,
        ) -> ProviderResult[dict[str, Any]]:
            del prompt_name, payload, reservation
            raise LLMAdapterError("authentication failed")

    controller = HardBudgetController(_budget(max_rerank_candidates=1))
    stage = LLMConstraintRerankingStage(
        analyzer=AuthenticationFailureAnalyzer(),
        call_estimate=UsageEstimate(
            llm_calls=1,
            input_tokens=10,
            output_tokens=10,
            cost_cny=0.1,
        ),
    )

    with pytest.raises(LLMAdapterError, match="authentication failed"):
        asyncio.run(
            stage.rerank(
                [Paper(canonical_id="paper:1", title="One")],
                ["constraint"],
                controller=controller,
            )
        )


def test_citation_stage_propagates_malformed_snapshot_provenance() -> None:
    class MalformedSnapshotProvider(CitationProvider):
        def _expansion_result(
            self,
            direction: str,
            paper_id: ProviderPaperId,
            reservation: BudgetReservation,
        ) -> ProviderResult[CitationExpansion]:
            result = super()._expansion_result(direction, paper_id, reservation)
            return result.model_copy(
                update={
                    "provenance": {
                        **result.provenance,
                        "snapshot_refs": "{malformed",
                    }
                }
            )

    controller = HardBudgetController(_budget(max_citation_seeds=1))
    stage = ProviderCitationExpansionStage(
        provider=MalformedSnapshotProvider(),
        call_estimate=UsageEstimate(search_api_calls=1),
    )

    with pytest.raises(ValueError, match="invalid citation snapshot provenance"):
        asyncio.run(
            stage.expand(
                [
                    Paper(
                        canonical_id="s2:S1",
                        title="Seed",
                        semantic_scholar_id="S1",
                    )
                ],
                controller=controller,
            )
        )


def test_llm_stage_propagates_incomplete_flattened_snapshot_provenance() -> None:
    class IncompleteSnapshotAnalyzer(RerankAnalyzer):
        async def generate_json(
            self,
            *,
            prompt_name: str,
            payload: dict[str, object],
            reservation: BudgetReservation,
        ) -> ProviderResult[dict[str, Any]]:
            result = await super().generate_json(
                prompt_name=prompt_name,
                payload=payload,
                reservation=reservation,
            )
            provenance = dict(result.provenance)
            provenance.pop("snapshot_cache_key")
            return result.model_copy(update={"provenance": provenance})

    controller = HardBudgetController(_budget(max_rerank_candidates=1))
    stage = LLMConstraintRerankingStage(
        analyzer=IncompleteSnapshotAnalyzer(),
        call_estimate=UsageEstimate(
            llm_calls=1,
            input_tokens=10,
            output_tokens=10,
            cost_cny=0.1,
        ),
    )

    with pytest.raises(ValueError, match="incomplete LLM snapshot provenance"):
        asyncio.run(
            stage.rerank(
                [Paper(canonical_id="paper:1", title="One")],
                ["constraint"],
                controller=controller,
            )
        )


def test_replay_llm_rerank_snapshot_miss_is_a_typed_service_failure() -> None:
    events: list[str] = []

    def factory(
        controller: HardBudgetController,
        run_id: str,
    ) -> MockSearchOrchestrator:
        del run_id
        return MockSearchOrchestrator(
            controller=controller,
            analyzer=FakeAnalyzer(events),
            providers={"openalex": FakeProvider("openalex", events)},
            config_hash="sha256:" + "2" * 64,
            prompt_version="query-analyze-v1",
            analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
            provider_estimate=UsageEstimate(search_api_calls=1),
            execution_mode="replay",
            constraint_reranker=LLMConstraintRerankingStage(
                analyzer=RerankAnalyzer(
                    failed=True,
                    error_code="snapshot_unavailable",
                ),
                call_estimate=UsageEstimate(
                    llm_calls=1,
                    input_tokens=10,
                    output_tokens=10,
                    cost_cny=0.1,
                ),
            ),
        )

    service = SearchApplicationService(
        orchestrator_factory=factory,
        budgets={"balanced": _budget(max_llm_calls=3)},
        mode="replay",
        snapshot_set_id="fixture-snapshot-set",
        snapshot_captured_at=datetime(2026, 7, 23, tzinfo=UTC),
        git_sha="a" * 40,
        max_provider_results=5,
        run_id_factory=lambda: "rerank-snapshot-miss",
    )

    execution = asyncio.run(
        service.execute(
            SearchRequest(
                query_id="q-rerank-miss",
                query="graph retrieval",
                mode="replay",
            )
        )
    )

    assert execution.outcome.kind == "failure"
    assert execution.outcome.error.code == "snapshot_unavailable"
    assert any(
        error.code == "snapshot_unavailable"
        for diagnostic in execution.diagnostics
        if diagnostic.dependency == "llm"
        for error in diagnostic.errors
    )


def test_replay_citation_snapshot_miss_is_a_typed_service_failure() -> None:
    events: list[str] = []

    def factory(
        controller: HardBudgetController,
        run_id: str,
    ) -> MockSearchOrchestrator:
        del run_id
        return MockSearchOrchestrator(
            controller=controller,
            analyzer=FakeAnalyzer(events),
            providers={
                "semantic_scholar": FakeProvider("semantic_scholar", events),
            },
            config_hash="sha256:" + "0" * 64,
            prompt_version="query-analyze-v1",
            analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
            provider_estimate=UsageEstimate(search_api_calls=1),
            execution_mode="replay",
            citation_expander=ProviderCitationExpansionStage(
                provider=CitationProvider(
                    failed=True,
                    error_code="snapshot_unavailable",
                ),
                call_estimate=UsageEstimate(search_api_calls=1),
            ),
        )

    service = SearchApplicationService(
        orchestrator_factory=factory,
        budgets={"balanced": _budget(max_search_api_calls=8)},
        mode="replay",
        snapshot_set_id="fixture-snapshot-set",
        snapshot_captured_at=datetime(2026, 7, 23, tzinfo=UTC),
        git_sha="c" * 40,
        max_provider_results=5,
        run_id_factory=lambda: "citation-snapshot-miss",
    )

    execution = asyncio.run(
        service.execute(
            SearchRequest(
                query_id="q-citation-miss",
                query="graph retrieval",
                mode="replay",
            )
        )
    )

    assert execution.outcome.kind == "failure"
    assert execution.outcome.error.code == "snapshot_unavailable"
    assert any(
        error.code == "snapshot_unavailable"
        for diagnostic in execution.diagnostics
        if diagnostic.dependency == "semantic_scholar"
        for error in diagnostic.errors
    )


def test_optional_snapshot_refs_survive_service_and_evaluation_adaptation() -> None:
    events: list[str] = []

    def factory(
        controller: HardBudgetController,
        run_id: str,
    ) -> MockSearchOrchestrator:
        del run_id
        return MockSearchOrchestrator(
            controller=controller,
            analyzer=FakeAnalyzer(events),
            providers={
                "semantic_scholar": FakeProvider("semantic_scholar", events),
            },
            config_hash="sha256:" + "1" * 64,
            prompt_version="query-analyze-v1",
            analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
            provider_estimate=UsageEstimate(search_api_calls=1),
            execution_mode="replay",
            citation_expander=ProviderCitationExpansionStage(
                provider=CitationProvider(),
                call_estimate=UsageEstimate(search_api_calls=1),
                per_direction_limit=2,
                max_expanded=2,
            ),
        )

    service = SearchApplicationService(
        orchestrator_factory=factory,
        budgets={"balanced": _budget(max_search_api_calls=8)},
        mode="replay",
        snapshot_set_id="fixture-snapshot-set",
        snapshot_captured_at=datetime(2026, 7, 23, tzinfo=UTC),
        git_sha="b" * 40,
        max_provider_results=5,
        run_id_factory=lambda: "citation-evidence-run",
    )
    execution = asyncio.run(
        service.execute(
            SearchRequest(
                query_id="q-citation-evidence",
                query="graph retrieval",
                mode="replay",
            )
        )
    )

    assert execution.outcome.kind == "success"
    service_refs = [
        ref.entry_id
        for diagnostic in execution.diagnostics
        for ref in diagnostic.snapshot_refs
        if diagnostic.endpoint == "citation_expansion"
    ]
    assert service_refs == ["citation-1", "citation-2"]

    adapted = adapt_execution(
        expected_query_id="q-citation-evidence",
        result=execution,
    )
    evaluation_refs = [
        ref.entry_id
        for diagnostic in adapted.execution.diagnostics
        for ref in diagnostic.snapshot_refs
    ]
    assert evaluation_refs == ["citation-1", "citation-2"]


def test_optional_stage_reservation_mismatch_hard_stops_instead_of_degrading() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget(max_llm_calls=3))
    stage = LLMConstraintRerankingStage(
        analyzer=OverrunRerankAnalyzer(),
        call_estimate=UsageEstimate(
            llm_calls=1,
            input_tokens=10,
            output_tokens=10,
            cost_cny=0.1,
        ),
    )
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "d" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        constraint_reranker=stage,
    )

    with pytest.raises(ReservationError, match="exceeds its reservation"):
        asyncio.run(
            orchestrator.run("graph retrieval", max_provider_results=5)
        )

    assert controller.stop_status() == "hard_stop"


def test_orchestrator_runs_optional_citation_then_rerank_stages() -> None:
    events: list[str] = []
    extra = Paper(canonical_id="fixture:extra", title="Expanded fixture")
    citation = FakeCitationExpander(extra)
    reranker = FakeConstraintReranker()
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "9" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        citation_expander=citation,
        constraint_reranker=reranker,
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert citation.calls == [["openalex:W1"]]
    assert reranker.calls == [(["openalex:W1", "fixture:extra"], [])]
    assert [paper.canonical_id for paper in result.papers] == [
        "fixture:extra",
        "openalex:W1",
    ]
    assert [item["step"] for item in result.trace[-2:]] == ["citation", "rerank"]


def test_orchestrator_accepts_dependency_owned_terminal_settlement() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget())
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=SettlingAnalyzer(events, controller),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "8" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert result.stop_reason == "completed"
    assert controller.committed_usage.llm_calls == 1


def test_orchestrator_accepts_provider_owned_terminal_settlement() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget())
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FakeAnalyzer(events),
        providers={"openalex": SettlingProvider("openalex", events, controller)},
        config_hash="sha256:" + "7" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert result.stop_reason == "completed"
    assert controller.committed_usage.search_api_calls == 2


def test_locked_baseline_router_prevents_unconditional_either_fanout() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget(max_search_api_calls=12)),
        analyzer=FakeAnalyzer(events),
        providers={
            "openalex": FakeProvider("openalex", events),
            "semantic_scholar": FakeProvider("semantic_scholar", events),
        },
        config_hash="sha256:" + "6" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        routing_limits=(3, 6, 2),
    )

    asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert events.count("openalex") == 3
    assert events.count("semantic_scholar") <= 2


def test_replay_integrity_failure_records_zero_external_spend() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget())
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FakeAnalyzer(events),
        providers={"openalex": IntegrityProvider("openalex", events)},
        config_hash="sha256:" + "5" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        execution_mode="replay",
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert controller.committed_usage.search_api_calls == 0
    assert result.diagnostics[-1].errors[0].code == "integrity_failure"


def test_formal_live_provider_exception_fails_closed_without_integrity_abort() -> None:
    from pathlib import Path

    from paper_search.control.pricing import (
        ActualCostPricer,
        parse_pricing_policy_bytes,
    )

    policy = parse_pricing_policy_bytes(
        Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml").read_bytes()
    )
    events: list[str] = []
    controller = HardBudgetController(_budget(), formal_live=True)
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FakeAnalyzer(events),
        providers={"openalex": IntegrityProvider("openalex", events)},
        config_hash="sha256:" + "4" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        execution_mode="live",
        pricer=ActualCostPricer(policy),
        provider_adapter_names={"openalex": "openalex-works-v1"},
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert controller.stop_status() == "hard_stop"
    assert controller.committed_usage.search_api_calls == 1
    assert controller.committed_usage.cost_cny is not None
    assert result.diagnostics[-1].errors[0].code == "provider_error"
    assert "openalex: provider exception" in result.warnings


def test_orchestrator_keeps_order_when_optional_stage_degrades() -> None:
    events: list[str] = []

    class BrokenCitation:
        async def expand(
            self,
            seeds: list[Paper],
            *,
            controller: HardBudgetController,
        ) -> CitationExpansionResult:
            del seeds, controller
            raise CitationExpansionUnavailableError("private fixture failure")

    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "a" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        citation_expander=BrokenCitation(),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert [paper.canonical_id for paper in result.papers] == ["openalex:W1"]
    assert result.warnings[-1] == "citation: expansion_unavailable"


def test_orchestrator_orders_budgeted_mock_pipeline_and_records_trace() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events), "semantic_scholar": FakeProvider("semantic_scholar", events)},
        config_hash="sha256:" + "b" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert events == ["analyze", "openalex", "semantic_scholar", "openalex", "semantic_scholar"]
    assert [paper.canonical_id for paper in result.papers] == ["openalex:W1", "s2:S1"]
    assert [item["step"] for item in result.trace] == [
        "analyze",
        "retrieve",
        "retrieve",
        "retrieve",
        "retrieve",
        "deduplicate",
        "filter",
        "fuse",
    ]
    assert set(result.provider_results) == {"openalex", "semantic_scholar"}
    assert result.fused_papers[0].paper.canonical_id == "openalex:W1"
    assert result.fused_papers[0].score > 0
    assert result.fused_papers[0].source_ranks == {"openalex": 1}
    assert result.config_hash == "sha256:" + "b" * 64
    assert result.prompt_version == "query-analyze-v1"
    assert result.stop_reason == "completed"


def test_orchestrator_aggregates_snapshot_refs_from_every_subquery() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={"openalex": SnapshotProvider("openalex", events)},
        config_hash="sha256:" + "b" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    diagnostic = next(
        item for item in result.diagnostics if item.dependency == "openalex"
    )
    assert len(diagnostic.snapshot_refs) == 2
    assert [ref.entry_id for ref in diagnostic.snapshot_refs] == [
        "entry-2",
        "entry-3",
    ]


def test_orchestrator_rejects_structured_planner_transport_error() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FailedAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "e" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert result.query_analysis.query_spec.ambiguities == ["rules_only_fallback"]
    assert result.warnings == ["analysis: dependency failure"]
    assert result.stop_reason == "dependency_failure"
    assert result.is_partial is True
    assert events == ["analyze"]


def test_orchestrator_fails_closed_on_analyzer_exception_without_calling_provider() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget())
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FailedAnalyzer(events, raises=True),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "f" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert events == ["analyze"]
    assert result.query_analysis.query_spec.ambiguities == ["rules_only_fallback"]
    assert result.stop_reason == "hard_stop"
    assert result.is_partial is True
    assert result.warnings == ["analysis: dependency failure"]
    assert controller.stop_status() == "hard_stop"


def test_orchestrator_treats_all_empty_provider_results_as_completed() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events, empty=True)},
        config_hash="sha256:" + "1" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert result.papers == []
    assert result.stop_reason == "completed"
    assert result.is_partial is False
    assert result.warnings == []


def test_orchestrator_soft_stop_prevents_provider_calls() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events, elapsed_ms=1_000),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "2" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(
            llm_calls=1,
            cost_cny=0.1,
            elapsed_ms=1_000,
        ),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert events == ["analyze"]
    assert result.stop_reason == "soft_stop"
    assert result.is_partial is True
    assert result.warnings == ["openalex: budget unavailable"]


def test_orchestrator_retains_valid_sibling_result_when_one_provider_fails() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={
            "openalex": FakeProvider("openalex", events, failed=True),
            "semantic_scholar": FakeProvider("semantic_scholar", events),
        },
        config_hash="sha256:" + "3" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert "openalex" in events
    assert "semantic_scholar" in events
    assert [paper.canonical_id for paper in result.papers] == ["s2:S1"]
    assert result.stop_reason == "completed"
    assert result.is_partial is True
    assert "openalex: provider returned errors" in result.warnings


def test_orchestrator_applies_injected_embedding_after_fusion() -> None:
    events: list[str] = []
    embedding = FakeEmbeddingRanker()
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={
            "openalex": FakeProvider("openalex", events),
            "semantic_scholar": FakeProvider("semantic_scholar", events),
        },
        config_hash="sha256:" + "4" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        embedding_ranker=embedding,
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert embedding.calls == [("graph retrieval", ["openalex:W1", "s2:S1"])]
    assert [paper.canonical_id for paper in result.papers] == [
        "s2:S1",
        "openalex:W1",
    ]
    assert result.trace[-1] == {
        "step": "embedding",
        "status": "applied",
        "model_id": "fixture-embedding-v1",
        "device": "cpu",
        "fallback_used": False,
        "count": 2,
    }


def test_orchestrator_embedding_degradation_keeps_fused_order() -> None:
    events: list[str] = []
    embedding = FakeEmbeddingRanker(degraded=True)
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={
            "openalex": FakeProvider("openalex", events),
            "semantic_scholar": FakeProvider("semantic_scholar", events),
        },
        config_hash="sha256:" + "5" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        embedding_ranker=embedding,
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert [paper.canonical_id for paper in result.papers] == [
        "openalex:W1",
        "s2:S1",
    ]
    assert result.is_partial is True
    assert result.warnings[-1] == "embedding: encoder_unavailable"


def test_orchestrator_embedding_degradation_ignores_reversed_ranked_order() -> None:
    events: list[str] = []
    embedding = FakeEmbeddingRanker(degraded=True, reverse_on_degraded=True)
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={
            "openalex": FakeProvider("openalex", events),
            "semantic_scholar": FakeProvider("semantic_scholar", events),
        },
        config_hash="sha256:" + "7" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        embedding_ranker=embedding,
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert embedding.calls == [("graph retrieval", ["openalex:W1", "s2:S1"])]
    assert [paper.canonical_id for paper in result.papers] == [
        "openalex:W1",
        "s2:S1",
    ]
    assert result.trace[-1] == {
        "step": "embedding",
        "status": "degraded",
        "model_id": "fixture-embedding-v1",
        "device": "cpu",
        "fallback_used": False,
        "count": 2,
    }
    assert result.warnings[-1] == "embedding: encoder_unavailable"


def test_orchestrator_sanitizes_injected_embedding_trace_metadata() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={
            "openalex": FakeProvider("openalex", events),
            "semantic_scholar": FakeProvider("semantic_scholar", events),
        },
        config_hash="sha256:" + "8" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        embedding_ranker=MaliciousEmbeddingRanker(),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert result.trace[-1]["model_id"] == "local_model"
    assert result.warnings[-3:] == [
        "embedding: cuda_oom_cpu_fallback",
        "embedding: unsanitized_warning",
        "embedding: unsanitized_warning",
    ]
    public_metadata = result.model_dump_json(include={"trace", "warnings"})
    assert "graph retrieval" not in public_metadata
    assert "openalex:W1" not in public_metadata
    assert "s2:S1" not in public_metadata
    assert "private-cache" not in public_metadata
    assert "query_graph_retrieval" not in public_metadata
    assert "openalex_w1" not in public_metadata
    assert "s2_s1" not in public_metadata
    assert "private_cache" not in public_metadata


def test_orchestrator_default_path_does_not_invoke_or_trace_embedding() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "6" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert "embedding" not in [item["step"] for item in result.trace]


def test_orchestrator_records_provider_failure_and_skips_calls_on_budget_stop() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget(max_search_api_calls=1, target_search_api_calls=1))
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FakeAnalyzer(events),
        providers={"openalex": FakeProvider("openalex", events, failed=True), "semantic_scholar": FakeProvider("semantic_scholar", events)},
        config_hash="sha256:" + "c" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert events == ["analyze", "openalex"]
    assert result.papers == []
    assert result.is_partial is True
    assert result.stop_reason == "hard_stop"
    assert result.warnings == [
        "openalex: provider returned errors",
        "semantic_scholar: budget unavailable",
    ]


def test_orchestrator_switches_provider_after_direct_timeout() -> None:
    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={
            "openalex": RaisingProvider(
                "openalex", events, TimeoutError("fixture timeout")
            ),
            "semantic_scholar": FakeProvider("semantic_scholar", events),
        },
        config_hash="sha256:" + "d" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert events[:3] == ["analyze", "openalex", "semantic_scholar"]
    assert [paper.canonical_id for paper in result.papers] == ["s2:S1"]
    assert result.is_partial is True
    assert "openalex: provider exception" in result.warnings


class OverrunProvider(FakeProvider):
    async def search(self, query: str, filters: dict[str, object], limit: int, reservation: object) -> ProviderResult[list[Paper]]:
        result = await super().search(query, filters, limit, reservation)
        return result.model_copy(update={"usage": UsageActual(search_api_calls=2)})


def test_orchestrator_fails_closed_when_a_provider_exceeds_its_reservation() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget())
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FakeAnalyzer(events),
        providers={"openalex": OverrunProvider("openalex", events)},
        config_hash="sha256:" + "d" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    try:
        asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))
    except RuntimeError:
        pass
    else:
        raise AssertionError("over-reservation usage must fail the orchestration")

    assert controller.stop_status() == "hard_stop"
