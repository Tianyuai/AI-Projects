from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from paper_search.control.budget import HardBudgetController
from paper_search.application.contracts import SnapshotRef
from paper_search.domain.models import (
    BudgetReservation,
    ErrorDetail,
    Paper,
    ProviderResult,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.graph.citation_expand import CitationExpansionResult
from paper_search.pipeline.orchestrator import MockSearchOrchestrator
from paper_search.ranking.embedding import EmbeddingRankingResult, EmbeddingScore
from paper_search.ranking.rerank import (
    ConstraintRerankResult,
    ConstraintScoredPaper,
)


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

    def expand(self, seeds: Sequence[Paper]) -> CitationExpansionResult:
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

    def rerank(
        self,
        papers: Sequence[Paper],
        constraints: Sequence[str],
    ) -> ConstraintRerankResult:
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


def test_orchestrator_keeps_order_when_optional_stage_degrades() -> None:
    events: list[str] = []

    class BrokenCitation:
        def expand(self, seeds: Sequence[Paper]) -> CitationExpansionResult:
            raise RuntimeError("private fixture failure")

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
