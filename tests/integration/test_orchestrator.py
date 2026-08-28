from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import pytest

from paper_search.control.budget import (
    HardBudgetController,
)
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
from paper_search.pipeline.orchestrator import MockSearchOrchestrator
from paper_search.learning.cross_vocabulary_bridge import (
    select_production_cross_vocabulary_supplement,
)
from paper_search.evaluation.dataset import IdentifierMap
from paper_search.ranking.fusion import FusedPaper
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    TextSearchAction,
    TextSearchPayload,
)
from paper_search.retrieval.routing import FixedBudgetOpenAlexPolicy


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


class RepairableAnalyzer:
    def __init__(
        self,
        events: list[str],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.events = events
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    async def __call__(
        self, query: str, _: object
    ) -> ProviderResult[dict[str, object]]:
        self.events.append("analyze")
        return _result(
            "llm",
            {},
            UsageActual(
                llm_calls=1,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cost_cny=0.1,
            ),
        ).model_copy(
            update={
                "errors": [
                    ErrorDetail(
                        code="invalid_json",
                        message="synthetic malformed JSON",
                        retryable=False,
                        provider="llm",
                    )
                ]
            }
        )

    async def repair(
        self,
        query: str,
        invalid_analysis: str,
        _: object,
    ) -> ProviderResult[dict[str, object]]:
        assert query == "graph retrieval"
        assert invalid_analysis == "{}"
        self.events.append("repair")
        return _result(
            "llm",
            {
                "query_spec": {
                    "original_query": query,
                    "research_goal": "find repaired papers",
                },
                "search_plan": {
                    "subqueries": [
                        {
                            "query_id": "repaired-1",
                            "text": query,
                            "query_type": "exact",
                            "target_constraints": [],
                            "priority": 1,
                            "provider_hint": "openalex",
                        }
                    ],
                    "inherited_hard_filters": {},
                    "rationale": "synthetic repair",
                },
            },
            UsageActual(
                llm_calls=1,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cost_cny=0.1,
            ),
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


class FakeDocumentRanker:
    model_id = "fixture-document-ranker-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rank(
        self,
        query: str,
        candidates: Sequence[FusedPaper],
    ) -> list[FusedPaper]:
        self.calls.append(
            (query, [item.paper.canonical_id for item in candidates])
        )
        return list(reversed(candidates))


class ContextAwareFakeDocumentRanker(FakeDocumentRanker):
    def rank_with_context(self, query, candidates, *, query_spec):
        self.calls.append(
            (query, [item.paper.canonical_id for item in candidates])
        )
        self.query_spec = query_spec
        return list(reversed(candidates))

    def context_receipt(self, query, *, query_spec):
        return {
            "schema_version": "fusion-query-context-receipt-v1",
            "query_sha256": "sha256:fixture",
            "context_sha256": "sha256:context",
        }


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


def test_semantic_action_v2_abstains_from_model_actions_for_negation() -> None:
    query = (
        "graph attention networks for molecular property prediction "
        "without 3D conformers"
    )
    valid_bridge = (
        "graph attention networks attention based message passing "
        "molecular property prediction"
    )

    class SemanticAnalyzer:
        async def __call__(
            self,
            received_query: str,
            reservation: object,
        ) -> ProviderResult[dict[str, object]]:
            assert received_query == query
            assert reservation is not None
            return _result(
                "llm",
                {
                    "query_spec": {
                        "original_query": query,
                        "research_goal": "Find faithful molecular prediction papers",
                        "topics": ["molecular property prediction"],
                        "methods": ["graph attention networks"],
                        "must_have": [
                            "graph attention networks",
                            "molecular property prediction",
                        ],
                        "exclusions": ["3D conformers"],
                    },
                    "search_plan": {
                        "subqueries": [
                            {
                                "query_id": "valid-bridge",
                                "text": valid_bridge,
                                "query_type": "expanded",
                                "target_constraints": ["graph attention networks"],
                                "priority": 1,
                                "provider_hint": "openalex",
                                "search_mode": "semantic",
                            },
                            {
                                "query_id": "hallucinated",
                                "text": "quantum turbulence spectral reconstruction",
                                "query_type": "expanded",
                                "target_constraints": ["quantum turbulence"],
                                "priority": 2,
                                "provider_hint": "openalex",
                            },
                            {
                                "query_id": "excluded-positive",
                                "text": "graph attention networks with 3D conformers",
                                "query_type": "expanded",
                                "target_constraints": ["graph attention networks"],
                                "priority": 3,
                                "provider_hint": "openalex",
                            },
                        ],
                        "inherited_hard_filters": {},
                        "rationale": "Exercise strict negation abstention",
                    },
                },
                UsageActual(llm_calls=1, cost_cny=0.1),
            )

    class RecordingProvider:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def search(
            self,
            received_query: str,
            filters: dict[str, object],
            limit: int,
            reservation: object,
        ) -> ProviderResult[list[Paper]]:
            assert filters in ({}, {"_search_mode": "semantic"})
            assert limit == 5
            assert reservation is not None
            self.queries.append(received_query)
            return _result(
                "openalex",
                [],
                UsageActual(search_api_calls=1),
            )

    provider = RecordingProvider()
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=SemanticAnalyzer(),
        providers={"openalex": provider},
        config_hash="sha256:" + "a" * 64,
        prompt_version="query-analyze-semantic-actions-v2",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(orchestrator.run(query, max_provider_results=5))

    assert provider.queries[0] == query
    assert valid_bridge not in provider.queries
    assert "quantum turbulence spectral reconstruction" not in provider.queries
    assert "graph attention networks with 3D conformers" not in provider.queries
    assert result.prompt_version == "query-analyze-semantic-actions-v2"



def test_max_output_papers_truncates_final_papers_but_not_pool() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget())

    class MultiProvider:
        async def search(
            self,
            query: str,
            filters: dict[str, object],
            limit: int,
            reservation: object,
        ) -> ProviderResult[list[Paper]]:
            del query, filters, limit, reservation
            return _result(
                "openalex",
                [
                    Paper(
                        canonical_id=f"openalex:W{i}",
                        title=f"Paper {i}",
                        openalex_id=f"W{i}",
                        sources=["openalex"],
                    )
                    for i in range(3)
                ],
                UsageActual(search_api_calls=1),
            )

    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=FakeAnalyzer(events),
        providers={"openalex": MultiProvider()},
        config_hash="sha256:" + "c" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        max_output_papers=1,
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert [paper.canonical_id for paper in result.papers] == ["openalex:W0"]
    assert len(result.retrieved_paper_ids) == 3
    assert len(result.post_filter_paper_ids) == 3
    assert len(result.pre_truncation_candidates) == 3


def test_fixed_budget_policy_executes_original_semantic_once_and_preserves_identity() -> None:
    events: list[tuple[str, str, str]] = []

    class RecordingProvider(FakeProvider):
        async def search(
            self,
            query: str,
            filters: dict[str, object],
            limit: int,
            reservation: object,
        ) -> ProviderResult[list[Paper]]:
            mode = str(filters.pop("_search_mode", "lexical"))
            events.append((self.name, mode, query))
            return await super().search(query, filters, limit, reservation)

    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget(max_search_api_calls=7)),
        analyzer=FakeAnalyzer([]),
        providers={
            "openalex": RecordingProvider("openalex", []),
            "semantic_scholar": RecordingProvider("semantic_scholar", []),
        },
        config_hash="sha256:" + "6" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        retrieval_policy=FixedBudgetOpenAlexPolicy(max_openalex_calls=6),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    openalex = [event for event in events if event[0] == "openalex"]
    assert 2 <= len(openalex) <= 6
    assert openalex[0] == ("openalex", "lexical", "graph retrieval")
    assert openalex[1] == ("openalex", "semantic", "graph retrieval")
    assert sum(mode == "semantic" for _, mode, _ in openalex) == 1
    assert all(provider != "semantic_scholar" for provider, _, _ in events)
    retrieval_trace = [
        item for item in result.trace if item.get("step") == "retrieve"
    ]
    assert all(
        item["action_type"] in {"text_search", "title_search"}
        for item in retrieval_trace
    )
    assert all(
        item["method"] in {"lexical_original", "semantic_original", "structured"}
        for item in retrieval_trace
    )


def test_orchestrator_executes_and_traces_local_supervised_plan_enrichment() -> None:
    class Enricher:
        def enrich(self, analysis: Any) -> tuple[Any, dict[str, object]]:
            action = analysis.search_plan.subqueries[0].model_copy(
                update={
                    "query_id": "sq-supervised-lexical-bridge",
                    "text": "learned vocabulary bridge",
                    "priority": len(analysis.search_plan.subqueries) + 1,
                    "provider_hint": "openalex",
                }
            )
            return analysis.model_copy(
                update={
                    "search_plan": analysis.search_plan.model_copy(
                        update={
                            "subqueries": [*analysis.search_plan.subqueries, action]
                        }
                    )
                }
            ), {
                "step": "supervised_query_expansion",
                "status": "appended",
                "model_sha256": "sha256:" + "a" * 64,
            }

    provider = FakeProvider("openalex", [])
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget(max_search_api_calls=6)),
        analyzer=FakeAnalyzer([]),
        providers={"openalex": provider},
        config_hash="sha256:" + "8" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        routing_limits=(3, 6, 0),
        query_plan_enricher=Enricher(),
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert any(
        item.get("step") == "retrieve"
        and item.get("subquery_id") == "sq-supervised-lexical-bridge"
        for item in result.trace
    )
    assert result.trace[1] == {
        "step": "supervised_query_expansion",
        "status": "appended",
        "model_sha256": "sha256:" + "a" * 64,
    }


def test_orchestrator_uses_frozen_soft_terms_for_semantic_action_gate() -> None:
    query = "compact representations for long scientific document retrieval"
    candidate = (
        "compact vector embeddings for long documents semantic retrieval literature"
    )

    class SemanticAnalyzer:
        async def __call__(
            self, raw_query: str, _: object
        ) -> ProviderResult[dict[str, object]]:
            return _result(
                "llm",
                {
                    "query_spec": {
                        "original_query": raw_query,
                        "research_goal": raw_query,
                    },
                    "search_plan": {
                        "subqueries": [
                            {
                                "query_id": "model-soft-rewrite",
                                "text": candidate,
                                "query_type": "expanded",
                                "target_constraints": [
                                    "compact vector embeddings",
                                    "long documents",
                                ],
                                "priority": 1,
                                "provider_hint": "openalex",
                                "search_mode": "lexical",
                            }
                        ],
                        "inherited_hard_filters": {},
                        "rationale": "fixture",
                    },
                },
                UsageActual(llm_calls=1, cost_cny=0.1),
            )

    class EvidenceEnricher:
        def soft_concept_terms(self, _: str) -> tuple[str, ...]:
            return ("embeddings",)

        def enrich(self, analysis: Any) -> tuple[Any, dict[str, object]]:
            return analysis, {
                "step": "supervised_query_expansion",
                "status": "abstained",
            }

    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget(max_search_api_calls=6)),
        analyzer=SemanticAnalyzer(),
        providers={"openalex": FakeProvider("openalex", [])},
        config_hash="sha256:" + "9" * 64,
        prompt_version="query-analyze-semantic-actions-v2",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        routing_limits=(3, 6, 0),
        query_plan_enricher=EvidenceEnricher(),
    )

    result = asyncio.run(orchestrator.run(query, max_provider_results=5))

    assert candidate in [
        item.text for item in result.query_analysis.search_plan.subqueries
    ], result.query_analysis.search_plan.subqueries
    assert any(
        item.get("step") == "retrieve" and item.get("query_text") == candidate
        for item in result.trace
    ), result.trace


def test_orchestrator_applies_frozen_aliases_during_production_deduplication() -> None:
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget(max_search_api_calls=8)),
        analyzer=FakeAnalyzer([]),
        providers={
            "openalex": FakeProvider("openalex", []),
            "semantic_scholar": FakeProvider("semantic_scholar", []),
        },
        config_hash="sha256:" + "8" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        routing_limits=(3, 6, 2),
        identifier_map=IdentifierMap.from_bytes(
            b'{"openalex:W1":"s2:S1"}', source="frozen PASA aliases"
        ),
        identifier_alias_count=1,
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert len(result.fused_papers) == 1
    dedup_trace = next(item for item in result.trace if item["step"] == "deduplicate")
    assert dedup_trace["identifier_alias_count"] == 1


def test_orchestrator_enforces_raw_deduplicated_and_output_candidate_caps() -> None:
    class ManyPapersProvider:
        def __init__(self) -> None:
            self.call_count = 0

        async def search(
            self,
            query: str,
            filters: dict[str, object],
            limit: int,
            reservation: object,
        ) -> ProviderResult[list[Paper]]:
            del query, filters, limit, reservation
            self.call_count += 1
            papers = [
                Paper(
                    canonical_id=f"openalex:W{self.call_count}{index}",
                    openalex_id=f"W{self.call_count}{index}",
                    title=f"Paper {self.call_count}-{index}",
                    is_retracted=False,
                    sources=["openalex"],
                )
                for index in range(1, 4)
            ]
            return _result("openalex", papers, UsageActual(search_api_calls=1))

    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget(max_search_api_calls=8)),
        analyzer=FakeAnalyzer([]),
        providers={"openalex": ManyPapersProvider()},
        config_hash="sha256:" + "8" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        routing_limits=(3, 6, 0),
        max_raw_candidates=3,
        max_deduplicated_candidates=2,
        max_output_papers=1,
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    cap_trace = next(item for item in result.trace if item["step"] == "candidate_cap")
    assert cap_trace["baseline_raw_before"] > 3
    assert cap_trace["baseline_raw_after"] == 3
    assert cap_trace["deduplicated_after"] == 2
    assert len(result.pre_truncation_candidates) == 2
    assert len(result.fused_papers) == 1


def test_uncertainty_multiplier_changes_the_pre_rank_candidate_order() -> None:
    class ConstraintAnalyzer:
        async def __call__(
            self, query: str, reservation: object
        ) -> ProviderResult[dict[str, object]]:
            del reservation
            return _result(
                "llm",
                {
                    "query_spec": {
                        "original_query": query,
                        "research_goal": query,
                        "exclusions": ["survey"],
                    },
                    "search_plan": {
                        "subqueries": [
                            {
                                "query_id": "sq-1",
                                "text": query,
                                "query_type": "exact",
                                "target_constraints": [],
                                "priority": 1,
                                "provider_hint": "openalex",
                            }
                        ],
                        "inherited_hard_filters": {},
                        "rationale": "fixture",
                    },
                },
                UsageActual(llm_calls=1),
            )

    class ConstraintProvider:
        async def search(
            self,
            query: str,
            filters: dict[str, object],
            limit: int,
            reservation: object,
        ) -> ProviderResult[list[Paper]]:
            del query, filters, limit, reservation
            return _result(
                "openalex",
                [
                    Paper(
                        canonical_id="openalex:W1",
                        openalex_id="W1",
                        title="Uncertain paper",
                        abstract=None,
                        is_retracted=None,
                        sources=["openalex"],
                    ),
                    Paper(
                        canonical_id="openalex:W2",
                        openalex_id="W2",
                        title="Certain paper",
                        abstract="A primary empirical study.",
                        is_retracted=False,
                        sources=["openalex"],
                    ),
                ],
                UsageActual(search_api_calls=1),
            )

    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget(max_search_api_calls=3)),
        analyzer=ConstraintAnalyzer(),
        providers={"openalex": ConstraintProvider()},
        config_hash="sha256:" + "8" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        routing_limits=(1, 1, 0),
    )

    result = asyncio.run(orchestrator.run("methods excluding survey", max_provider_results=5))

    assert [item.paper.canonical_id for item in result.fused_papers] == [
        "openalex:W2",
        "openalex:W1",
    ]
    uncertainty_trace = next(
        item for item in result.trace if item["step"] == "uncertainty_adjustment"
    )
    assert uncertainty_trace["penalized_count"] == 1
    assert uncertainty_trace["reason_counts"] == {
        "missing_abstract_for_exclusion": 1,
        "unknown_retraction_status": 1,
    }


def test_bounded_unconstrained_supplement_preserves_six_actions_and_appends_one() -> None:
    class SixActionAnalyzer:
        async def __call__(
            self,
            query: str,
            _reservation: object,
        ) -> ProviderResult[dict[str, object]]:
            return _result(
                "llm",
                {
                    "query_spec": {
                        "original_query": query,
                        "research_goal": "find papers",
                    },
                    "search_plan": {
                        "subqueries": [
                            {
                                "query_id": f"structured-{index}",
                                "text": f"{query} variant {index}",
                                "query_type": "expanded",
                                "target_constraints": [],
                                "priority": index,
                                "provider_hint": "openalex",
                            }
                            for index in range(1, 6)
                        ],
                        "inherited_hard_filters": {},
                        "rationale": "six baseline OpenAlex actions",
                    },
                },
                UsageActual(llm_calls=1, cost_cny=0.1),
            )

    class BridgeEvidenceProvider:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def search(
            self,
            query: str,
            filters: dict[str, object],
            limit: int,
            reservation: object,
        ) -> ProviderResult[list[Paper]]:
            del filters, limit, reservation
            self.queries.append(query)
            if query == "graph neural retrieval alignment":
                papers = [
                    Paper(
                        canonical_id="openalex:W99",
                        title="Supplemental alignment paper",
                        openalex_id="W99",
                        sources=["openalex"],
                    )
                ]
            else:
                titles = (
                    "Graph neural retrieval alignment modeling",
                    "Graph neural search alignment modeling",
                    "Neural retrieval alignment modeling",
                    "Graph retrieval benchmark alignment",
                    "Neural benchmark evaluation modeling",
                    "Graph benchmark systems modeling",
                    "Graph neural retrieval evaluation",
                )
                selected = range(4) if len(self.queries) == 1 else range(7)
                papers = [
                    Paper(
                        canonical_id=f"openalex:W{index + 1}",
                        title=titles[index],
                        openalex_id=f"W{index + 1}",
                        sources=["openalex"],
                    )
                    for index in selected
                ]
            return _result(
                "openalex",
                papers,
                UsageActual(search_api_calls=1),
            )

    provider = BridgeEvidenceProvider()
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget(max_search_api_calls=8)),
        analyzer=SixActionAnalyzer(),
        providers={"openalex": provider},
        config_hash="sha256:" + "d" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        retrieval_policy=FixedBudgetOpenAlexPolicy(max_openalex_calls=6),
        openalex_supplement_selector=(
            select_production_cross_vocabulary_supplement
        ),
        max_total_openalex_actions=7,
    )

    result = asyncio.run(
        orchestrator.run(
            "graph neural retrieval benchmark",
            max_provider_results=50,
        )
    )

    assert len(provider.queries) == 7
    assert provider.queries[-1] == "graph neural retrieval alignment"
    assert {f"openalex:W{index}" for index in range(1, 8)}.issubset(
        set(result.retrieved_paper_ids)
    )
    assert "openalex:W99" in result.retrieved_paper_ids
    assert any(item.get("step") == "retrieve_supplement" for item in result.trace)


def test_live_and_replay_share_nonreinforcing_fair_supplement_merge() -> None:
    class TwoActionAnalyzer:
        async def __call__(
            self,
            query: str,
            _reservation: object,
        ) -> ProviderResult[dict[str, object]]:
            return _result(
                "llm",
                {
                    "query_spec": {
                        "original_query": query,
                        "research_goal": "find papers",
                    },
                    "search_plan": {
                        "subqueries": [
                            {
                                "query_id": "base-one",
                                "text": "baseline one",
                                "query_type": "exact",
                                "target_constraints": [],
                                "priority": 1,
                                "provider_hint": "openalex",
                            },
                            {
                                "query_id": "base-two",
                                "text": "baseline two",
                                "query_type": "expanded",
                                "target_constraints": [],
                                "priority": 2,
                                "provider_hint": "openalex",
                            },
                        ],
                        "inherited_hard_filters": {},
                        "rationale": "two baseline actions",
                    },
                },
                UsageActual(llm_calls=1, cost_cny=0.1),
            )

    class FairMergeProvider:
        async def search(
            self,
            query: str,
            filters: dict[str, object],
            limit: int,
            reservation: object,
        ) -> ProviderResult[list[Paper]]:
            del filters, limit, reservation
            ids = {
                "baseline one": ("openalex:W1", "openalex:W2"),
                "baseline two": ("openalex:W1",),
                "supplemental terms": ("openalex:W3", "openalex:W1"),
            }[query]
            return _result(
                "openalex",
                [
                    Paper(
                        canonical_id=identifier,
                        title=f"Paper {identifier}",
                        openalex_id=identifier.removeprefix("openalex:"),
                        sources=["openalex"],
                    )
                    for identifier in ids
                ],
                UsageActual(search_api_calls=1),
            )

    class EvidenceRanker:
        model_id = "fixture-evidence-ranker-v1"

        def __init__(self) -> None:
            self.rows: list[tuple[str, float, dict[str, int]]] = []

        def rank(
            self,
            _query: str,
            candidates: Sequence[FusedPaper],
        ) -> list[FusedPaper]:
            self.rows = [
                (item.paper.canonical_id, item.score, dict(item.source_ranks))
                for item in candidates
            ]
            return list(candidates)

    def fixed_selector(_query_spec, _candidates) -> RecallActionBatch:
        return RecallActionBatch(
            actions=[
                TextSearchAction(
                    action_id="fair-supplement-v1",
                    strategy="fixture:fair-supplement",
                    action_type="text_search",
                    payload=TextSearchPayload(
                        query_text="supplemental terms",
                        search_mode="lexical",
                    ),
                )
            ]
        )

    evidence_by_mode: dict[str, list[tuple[str, float, dict[str, int]]]] = {}
    merge_trace_by_mode: dict[str, dict[str, object]] = {}
    for mode in ("live", "replay"):
        ranker = EvidenceRanker()
        orchestrator = MockSearchOrchestrator(
            controller=HardBudgetController(_budget(max_search_api_calls=7)),
            analyzer=TwoActionAnalyzer(),
            providers={"openalex": FairMergeProvider()},
            config_hash="sha256:" + "f" * 64,
            prompt_version="query-analyze-v1",
            analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
            provider_estimate=UsageEstimate(search_api_calls=1),
            openalex_supplement_selector=fixed_selector,
            max_total_openalex_actions=7,
            document_ranker=ranker,
            execution_mode=mode,
        )

        result = asyncio.run(
            orchestrator.run("fair candidate merge", max_provider_results=5)
        )
        evidence_by_mode[mode] = ranker.rows
        merge_rows = [
            row for row in result.trace if row.get("step") == "merge_supplement"
        ]
        assert len(merge_rows) == 1
        merge_trace_by_mode[mode] = merge_rows[0]

    assert evidence_by_mode["live"] == evidence_by_mode["replay"]
    rows = evidence_by_mode["live"]
    assert [identifier for identifier, _score, _sources in rows] == [
        "openalex:W1",
        "openalex:W3",
        "openalex:W2",
    ]
    by_id = {identifier: sources for identifier, _score, sources in rows}
    assert set(by_id["openalex:W1"]) == {
        "openalex:sq-1:lexical",
        "openalex:sq-2:lexical",
    }
    assert set(by_id["openalex:W3"]) == {
        "openalex:fair-supplement-v1:lexical"
    }
    assert merge_trace_by_mode["live"] == merge_trace_by_mode["replay"]


def test_supplement_provider_failure_keeps_the_complete_baseline_pool() -> None:
    class FailingSupplementProvider:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def search(
            self,
            query: str,
            filters: dict[str, object],
            limit: int,
            reservation: object,
        ) -> ProviderResult[list[Paper]]:
            del filters, limit, reservation
            self.queries.append(query)
            if query == "bounded supplement":
                raise TimeoutError("synthetic supplement timeout")
            index = len(self.queries)
            return _result(
                "openalex",
                [
                    Paper(
                        canonical_id=f"openalex:W{index}",
                        title=f"Baseline paper {index}",
                        openalex_id=f"W{index}",
                        sources=["openalex"],
                    )
                ],
                UsageActual(search_api_calls=1),
            )

    def fixed_selector(_query_spec, _candidates) -> RecallActionBatch:
        return RecallActionBatch(
            actions=[
                TextSearchAction(
                    action_id="bounded-supplement-v1",
                    strategy="fixture:bounded-supplement",
                    action_type="text_search",
                    payload=TextSearchPayload(
                        query_text="bounded supplement",
                        search_mode="lexical",
                    ),
                )
            ]
        )

    provider = FailingSupplementProvider()
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget(max_search_api_calls=7)),
        analyzer=FakeAnalyzer([]),
        providers={"openalex": provider},
        config_hash="sha256:" + "e" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        openalex_supplement_selector=fixed_selector,
        max_total_openalex_actions=7,
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    baseline_ids = {
        f"openalex:W{index}" for index in range(1, len(provider.queries))
    }
    assert baseline_ids == set(result.retrieved_paper_ids)
    assert "openalex supplement: provider exception" in result.warnings
    assert any(
        item.get("step") == "supplement_fallback"
        and item.get("reason") == "provider_exception"
        for item in result.trace
    )


@pytest.mark.parametrize("execution_mode", ["live", "replay"])
def test_low_confidence_llm_action_is_supplemental_and_keeps_primary_context(
    execution_mode: Literal["live", "replay"],
) -> None:
    query = "rare mechanism evidence"

    class PrimaryAnalyzer:
        async def __call__(self, value, _reservation):
            return _result(
                "llm",
                {
                    "query_spec": {
                        "original_query": value,
                        "research_goal": "retain the production interpretation",
                    },
                    "search_plan": {
                        "subqueries": [
                            {
                                "query_id": "production-1",
                                "text": "rare mechanism baseline",
                                "query_type": "expanded",
                                "target_constraints": ["rare mechanism"],
                                "priority": 1,
                                "provider_hint": "openalex",
                                "search_mode": "lexical",
                            }
                        ],
                        "inherited_hard_filters": {},
                        "rationale": "production plan",
                    },
                },
                UsageActual(llm_calls=1),
            )

    class SupplementalAnalyzer:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, value, _reservation):
            self.calls += 1
            return _result(
                "llm",
                {
                    "query_spec": {
                        "original_query": value,
                        "research_goal": "candidate interpretation must not replace production",
                    },
                    "search_plan": {
                        "subqueries": [
                            {
                                "query_id": "candidate-s2",
                                "text": "orthogonal rare mechanism terminology",
                                "query_type": "expanded",
                                "action_type": "text_search",
                                "target_constraints": ["rare mechanism"],
                                "priority": 1,
                                "provider_hint": "semantic_scholar",
                                "search_mode": "lexical",
                            }
                        ],
                        "inherited_hard_filters": {},
                        "rationale": "one independent action",
                    },
                },
                UsageActual(llm_calls=1),
            )

    class Provider:
        def __init__(self, name: str) -> None:
            self.name = name
            self.queries: list[str] = []

        async def search(self, value, filters, limit, reservation):
            del filters, limit, reservation
            self.queries.append(value)
            if value == "orthogonal rare mechanism terminology" and self.name == "openalex":
                ids = ["9"]
            elif value == "orthogonal rare mechanism terminology":
                ids = ["S1"]
            else:
                ids = ["1", "2", "3"]
            return _result(
                self.name,
                [
                    Paper(
                        canonical_id=(
                            f"openalex:W{identifier}"
                            if self.name == "openalex"
                            else f"s2:{identifier}"
                        ),
                        openalex_id=(
                            f"W{identifier}" if self.name == "openalex" else None
                        ),
                        semantic_scholar_id=(
                            identifier
                            if self.name == "semantic_scholar"
                            else None
                        ),
                        title=f"Rare mechanism evidence {identifier}",
                        sources=[self.name],
                    )
                    for identifier in ids
                ],
                UsageActual(search_api_calls=1),
            )

    secondary = SupplementalAnalyzer()
    openalex = Provider("openalex")
    s2 = Provider("semantic_scholar")
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(
            _budget(
                max_llm_calls=3,
                max_search_api_calls=4,
                max_total_tokens=200,
                max_elapsed_seconds=20,
                soft_deadline_seconds=19,
            )
        ),
        analyzer=PrimaryAnalyzer(),
        low_confidence_analyzer=secondary,
        low_confidence_prompt_version="query-analyze-protected-actions-v3",
        low_confidence_analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        providers={"openalex": openalex, "semantic_scholar": s2},
        config_hash="sha256:" + "a" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        routing_limits=(1, 1, 0),
        execution_mode=execution_mode,
        max_raw_candidates=3,
        max_deduplicated_candidates=3,
        max_low_confidence_raw_candidates=2,
        max_low_confidence_deduplicated_candidates=2,
    )

    result = asyncio.run(orchestrator.run(query, max_provider_results=50))

    assert secondary.calls == 1
    assert result.query_analysis.query_spec.research_goal == (
        "retain the production interpretation"
    )
    assert "candidate-s2" not in {
        item.query_id for item in result.query_analysis.search_plan.subqueries
    }
    assert "orthogonal rare mechanism terminology" not in {
        item.text for item in result.query_analysis.search_plan.subqueries
    }
    assert result.pre_truncation_candidates, (result.warnings, result.trace)
    assert {"openalex:W1", "openalex:W2", "openalex:W3"}.issubset(
        {paper.canonical_id for paper in result.pre_truncation_candidates}
    )
    assert {"openalex:W9", "s2:S1"}.issubset(
        {paper.canonical_id for paper in result.pre_truncation_candidates}
    )
    assert any(
        row.get("step") == "analyze_low_confidence_supplement"
        and row.get("prompt_version") == "query-analyze-protected-actions-v3"
        for row in result.trace
    )
    assert sum(
        row.get("step") == "retrieve_llm_supplement" for row in result.trace
    ) == 2
    assert any(row.get("step") == "merge_llm_supplement" for row in result.trace)


def test_low_confidence_llm_failure_returns_the_complete_primary_pool() -> None:
    class FailingSupplementAnalyzer:
        async def __call__(self, _query, _reservation):
            raise TimeoutError("synthetic low-confidence analyzer failure")

    events: list[str] = []
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(
            _budget(max_llm_calls=3, max_search_api_calls=3)
        ),
        analyzer=FakeAnalyzer(events),
        low_confidence_analyzer=FailingSupplementAnalyzer(),
        low_confidence_prompt_version="query-analyze-protected-actions-v3",
        low_confidence_analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "b" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        routing_limits=(1, 1, 0),
        max_low_confidence_raw_candidates=2,
        max_low_confidence_deduplicated_candidates=2,
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert result.pre_truncation_candidates
    assert "low-confidence supplement: analyzer exception" in result.warnings
    assert any(
        row.get("step") == "llm_supplement_fallback"
        and row.get("reason") == "analyzer_exception"
        for row in result.trace
    )


def test_negation_strictly_abstains_before_the_second_llm_call() -> None:
    class NegationAnalyzer:
        async def __call__(self, query, _reservation):
            return _result(
                "llm",
                {
                    "query_spec": {
                        "original_query": query,
                        "research_goal": "find graph retrieval papers without transformers",
                        "exclusions": ["transformers"],
                    },
                    "search_plan": {
                        "subqueries": [
                            {
                                "query_id": "production-negation",
                                "text": "graph retrieval",
                                "query_type": "expanded",
                                "target_constraints": ["graph retrieval"],
                                "priority": 1,
                                "provider_hint": "openalex",
                                "search_mode": "lexical",
                            }
                        ],
                        "inherited_hard_filters": {},
                        "rationale": "strict negation fixture",
                    },
                },
                UsageActual(llm_calls=1),
            )

    class CountingSupplementAnalyzer:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, _query, _reservation):
            self.calls += 1
            raise AssertionError("negation must abstain before the second LLM call")

    supplement = CountingSupplementAnalyzer()
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(
            _budget(max_llm_calls=3, max_search_api_calls=2)
        ),
        analyzer=NegationAnalyzer(),
        low_confidence_analyzer=supplement,
        low_confidence_prompt_version="query-analyze-protected-actions-v3",
        low_confidence_analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        providers={"openalex": FakeProvider("openalex", [])},
        config_hash="sha256:" + "c" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        routing_limits=(1, 1, 0),
        max_low_confidence_raw_candidates=2,
        max_low_confidence_deduplicated_candidates=2,
    )

    result = asyncio.run(
        orchestrator.run(
            "graph retrieval without transformers",
            max_provider_results=5,
        )
    )

    assert supplement.calls == 0
    assert any(
        row.get("step") == "skip_llm_supplement"
        and row.get("reason") == "strict_negation_abstention"
        for row in result.trace
    )


def test_orchestrator_applies_injected_document_ranker_after_fusion() -> None:
    events: list[str] = []
    document_ranker = FakeDocumentRanker()
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={
            "openalex": FakeProvider("openalex", events),
            "semantic_scholar": FakeProvider("semantic_scholar", events),
        },
        config_hash="sha256:" + "3" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        document_ranker=document_ranker,
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert document_ranker.calls == [
        ("graph retrieval", ["openalex:W1", "s2:S1"])
    ]
    assert [paper.canonical_id for paper in result.papers] == [
        "s2:S1",
        "openalex:W1",
    ]
    assert result.trace[-1] == {
        "step": "document_rank",
        "status": "applied",
        "model_id": "fixture-document-ranker-v1",
        "count": 2,
    }


def test_orchestrator_passes_analysis_context_and_traces_its_identity() -> None:
    events: list[str] = []
    document_ranker = ContextAwareFakeDocumentRanker()
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={
            "openalex": FakeProvider("openalex", events),
            "semantic_scholar": FakeProvider("semantic_scholar", events),
        },
        config_hash="sha256:" + "3" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        document_ranker=document_ranker,
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    assert document_ranker.query_spec.original_query == "graph retrieval"
    assert result.trace[-1]["query_context"] == {
        "schema_version": "fusion-query-context-receipt-v1",
        "query_sha256": "sha256:fixture",
        "context_sha256": "sha256:context",
    }


def test_orchestrator_traces_deployment_ranker_role_and_failover_receipt() -> None:
    events: list[str] = []
    document_ranker = FakeDocumentRanker()
    document_ranker.deployment_role = "F4-reliability"
    document_ranker.failover_receipt = [
        {"role": "F5-gated-fusion", "reason": "hash_mismatch"}
    ]
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=FakeAnalyzer(events),
        providers={
            "openalex": FakeProvider("openalex", events),
            "semantic_scholar": FakeProvider("semantic_scholar", events),
        },
        config_hash="sha256:" + "3" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
        document_ranker=document_ranker,
    )

    result = asyncio.run(orchestrator.run("graph retrieval", max_provider_results=5))

    rank_trace = result.trace[-1]
    assert rank_trace["deployment_role"] == "F4-reliability"
    assert rank_trace["failover_receipt"] == [
        {"role": "F5-gated-fusion", "reason": "hash_mismatch"}
    ]


def test_provider_filters_only_receive_years_while_venue_is_filtered_locally() -> None:
    class VenueAnalyzer:
        async def __call__(
            self,
            query: str,
            _reservation: object,
        ) -> ProviderResult[dict[str, object]]:
            return _result(
                "llm",
                {
                    "query_spec": {
                        "original_query": query,
                        "research_goal": "find venue papers",
                        "year_from": 2020,
                        "venues": ["NeurIPS"],
                    },
                    "search_plan": {
                        "subqueries": [
                            {
                                "query_id": "venue-openalex",
                                "text": query,
                                "query_type": "exact",
                                "target_constraints": ["NeurIPS"],
                                "priority": 1,
                                "provider_hint": "openalex",
                            },
                            {
                                "query_id": "venue-s2",
                                "text": f"{query} methods",
                                "query_type": "decomposed",
                                "target_constraints": ["NeurIPS"],
                                "priority": 2,
                                "provider_hint": "semantic_scholar",
                            },
                            {
                                "query_id": "venue-either",
                                "text": f"{query} results",
                                "query_type": "expanded",
                                "target_constraints": ["NeurIPS"],
                                "priority": 3,
                                "provider_hint": "either",
                            },
                        ],
                        "inherited_hard_filters": {},
                        "rationale": "venue fixture",
                    },
                },
                UsageActual(llm_calls=1, cost_cny=0.1),
            )

    class VenueProvider:
        def __init__(self, name: str) -> None:
            self.name = name
            self.filters: list[dict[str, object]] = []

        async def search(
            self,
            query: str,
            filters: dict[str, object],
            limit: int,
            reservation: object,
        ) -> ProviderResult[list[Paper]]:
            del query, limit, reservation
            self.filters.append(dict(filters))
            return _result(
                self.name,
                [
                    Paper(
                        canonical_id="openalex:W100",
                        title="Accepted venue paper",
                        publication_year=2021,
                        venue="NeurIPS",
                        openalex_id="W100",
                        sources=[self.name],
                    ),
                    Paper(
                        canonical_id="openalex:W200",
                        title="Rejected venue paper",
                        publication_year=2021,
                        venue="ICML",
                        openalex_id="W200",
                        sources=[self.name],
                    ),
                ],
                UsageActual(search_api_calls=1),
            )

    openalex = VenueProvider("openalex")
    semantic_scholar = VenueProvider("semantic_scholar")
    orchestrator = MockSearchOrchestrator(
        controller=HardBudgetController(_budget()),
        analyzer=VenueAnalyzer(),
        providers={
            "openalex": openalex,
            "semantic_scholar": semantic_scholar,
        },
        config_hash="sha256:" + "7" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run(
            "NeurIPS graph retrieval papers since 2020",
            max_provider_results=5,
        )
    )

    assert openalex.filters
    assert semantic_scholar.filters
    assert all(filters == {"year_from": 2020} for filters in openalex.filters)
    assert all(
        filters == {"year_from": 2020} for filters in semantic_scholar.filters
    )
    assert [paper.canonical_id for paper in result.papers] == [
        "openalex:W100"
    ]



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

    assert events == ["analyze", "openalex", "semantic_scholar", "openalex"]
    assert any(item.get("step") == "skip_optional_provider" for item in result.trace)
    assert [paper.canonical_id for paper in result.papers] == ["openalex:W1", "s2:S1"]
    assert [item["step"] for item in result.trace] == [
        "analyze",
        "retrieve",
        "retrieve",
        "retrieve",
        "skip_optional_provider",
        "candidate_cap",
        "deduplicate",
        "filter",
        "uncertainty_adjustment",
        "fuse",
    ]
    assert set(result.provider_results) == {"openalex", "semantic_scholar"}
    assert result.fused_papers[0].paper.canonical_id == "openalex:W1"
    assert result.fused_papers[0].score > 0
    assert result.fused_papers[0].source_ranks == {
        "openalex:sq-1:lexical": 1,
        "openalex:sq-3:lexical": 1,
    }
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


def test_orchestrator_repairs_malformed_analysis_once_before_fallback() -> None:
    events: list[str] = []
    controller = HardBudgetController(_budget(max_llm_calls=3))
    analyzer = RepairableAnalyzer(events)
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=analyzer,
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "a" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert events[:2] == ["analyze", "repair"]
    assert result.query_analysis.planner_status == "repaired"
    assert result.query_analysis.query_spec.research_goal == "find repaired papers"
    assert controller.committed_usage.llm_calls == 2
    assert result.stop_reason == "completed"


def test_orchestrator_can_repair_when_cost_reservation_reaches_decimal_cap() -> None:
    events: list[str] = []
    controller = HardBudgetController(
        _budget(max_llm_calls=5, max_cost_cny=0.30)
    )
    analyzer = RepairableAnalyzer(
        events,
        input_tokens=10,
        output_tokens=10,
    )
    orchestrator = MockSearchOrchestrator(
        controller=controller,
        analyzer=analyzer,
        providers={"openalex": FakeProvider("openalex", events)},
        config_hash="sha256:" + "a" * 64,
        prompt_version="query-analyze-v1",
        analysis_estimate=UsageEstimate(
            llm_calls=3,
            input_tokens=80,
            output_tokens=20,
            cost_cny=0.1,
        ),
        provider_estimate=UsageEstimate(search_api_calls=1),
    )

    result = asyncio.run(
        orchestrator.run("graph retrieval", max_provider_results=5)
    )

    assert events[:2] == ["analyze", "repair"]
    assert result.query_analysis.planner_status == "repaired"


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
