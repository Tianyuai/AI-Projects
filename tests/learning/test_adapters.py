from __future__ import annotations

import asyncio
from decimal import Decimal
from datetime import UTC, datetime

from paper_search.domain.models import (
    BudgetReservation,
    ProviderResult,
    QueryAnalysisResult,
    UsageActual,
    UsageEstimate,
)
from paper_search.learning.adapters import (
    QueryPolicyAnalyzerAdapter,
    RecallQueryPolicyGenerator,
)
from paper_search.learning.contracts import PolicyActionCandidate, QueryPolicyInput
from paper_search.learning.policy import BoundedQueryPolicy
from paper_search.query.parser import rule_fallback
from paper_search.recall_experiments.contracts import RecallGenerationContext
from paper_search.recall_experiments.generation.base import GenerationResult


class MappingScorer:
    model_id = "fake-action-ranker-v1"

    def __init__(self, default_score: float) -> None:
        self.default_score = default_score

    def score(
        self,
        request: QueryPolicyInput,
        candidates: list[PolicyActionCandidate],
    ) -> list[float]:
        del request
        return [self.default_score for _ in candidates]


class RecordingScorer(MappingScorer):
    def __init__(self, default_score: float) -> None:
        super().__init__(default_score)
        self.candidate_counts: list[int] = []

    def score(
        self,
        request: QueryPolicyInput,
        candidates: list[PolicyActionCandidate],
    ) -> list[float]:
        self.candidate_counts.append(len(candidates))
        return super().score(request, candidates)


def _context() -> RecallGenerationContext:
    query = "graph retrieval papers"
    return RecallGenerationContext(
        query_id="q1",
        original_query=query,
        query_spec=rule_fallback(query),
        seed_queries=["neural graph retrieval"],
    )


def _reservation() -> BudgetReservation:
    return BudgetReservation(
        reservation_id="r1",
        action="query.analyze",
        reserved=UsageEstimate(llm_calls=1, input_tokens=100, output_tokens=100),
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


def test_recall_adapter_emits_the_shared_action_batch_without_llm_receipts() -> None:
    generator = RecallQueryPolicyGenerator(
        BoundedQueryPolicy(MappingScorer(0.9), confidence_threshold=0.5),
        max_actions=3,
    )

    result = asyncio.run(generator.generate(_context()))

    assert [action.payload.query_text for action in result.action_batch.actions] == [
        "graph retrieval papers",
        "graph retrieval papers",
        "graph retrieval",
    ]
    assert result.action_batch.actions[0].payload.search_mode == "lexical"
    assert result.action_batch.actions[1].payload.search_mode == "semantic"
    assert result.call_receipts == []
    assert result.provenance["model_id"] == "fake-action-ranker-v1"
    assert result.provenance["fallback_required"] == "false"


def test_recall_adapter_generates_local_candidates_when_no_seed_queries_exist() -> None:
    query = "Which paper proposed graph diffusion networks for retrieval?"
    context = RecallGenerationContext(
        query_id="q-local",
        original_query=query,
        query_spec=rule_fallback(query),
        seed_queries=[],
    )
    generator = RecallQueryPolicyGenerator(
        BoundedQueryPolicy(MappingScorer(0.9), confidence_threshold=0.5),
        max_actions=5,
    )

    result = asyncio.run(generator.generate(context))

    assert len(result.action_batch.actions) >= 2
    assert any(
        action.action_type == "title_search"
        for action in result.action_batch.actions
    )


def test_recall_adapter_ranks_a_larger_pool_than_the_action_budget() -> None:
    scorer = RecordingScorer(0.9)
    generator = RecallQueryPolicyGenerator(
        BoundedQueryPolicy(scorer, confidence_threshold=0.5),
        max_actions=3,
        candidate_pool_size=12,
    )
    query = "Which paper proposed graph diffusion networks for MS MARCO retrieval?"
    context = RecallGenerationContext(
        query_id="q-pool",
        original_query=query,
        query_spec=rule_fallback(query),
        seed_queries=[],
    )

    result = asyncio.run(generator.generate(context))

    assert len(result.action_batch.actions) == 3
    assert scorer.candidate_counts[0] > 3


def test_recall_adapter_can_emit_three_unique_lexical_production_actions() -> None:
    generator = RecallQueryPolicyGenerator(
        BoundedQueryPolicy(MappingScorer(0.9), confidence_threshold=0.5),
        max_actions=3,
        force_lexical_unique=True,
    )

    result = asyncio.run(generator.generate(_context()))

    texts = [action.payload.query_text for action in result.action_batch.actions]
    assert len(texts) == 3
    assert len({" ".join(text.casefold().split()) for text in texts}) == 3
    assert all(
        action.payload.search_mode == "lexical"
        for action in result.action_batch.actions
    )


def test_recall_adapter_calls_existing_generator_only_below_threshold() -> None:
    class FallbackGenerator:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, context: RecallGenerationContext) -> GenerationResult:
            self.calls += 1
            return GenerationResult(
                query_id=context.query_id,
                action_batch={
                    "actions": [
                        {
                            "action_id": "llm-1",
                            "action_type": "text_search",
                            "strategy": "llm_fallback",
                            "payload": {"query_text": "fallback query"},
                        }
                    ]
                },
                artifact_bytes=b'{"actions":[]}',
                provenance={"generator": "llm"},
            )

    fallback = FallbackGenerator()
    generator = RecallQueryPolicyGenerator(
        BoundedQueryPolicy(MappingScorer(0.1), confidence_threshold=0.5),
        max_actions=3,
        fallback=fallback,
    )

    result = asyncio.run(generator.generate(_context()))

    assert fallback.calls == 1
    assert result.action_batch.actions[0].payload.query_text == "fallback query"
    assert result.provenance["query_policy_gate"] == "confidence_below_threshold"


def test_analyzer_adapter_produces_a_strict_zero_llm_query_analysis() -> None:
    adapter = QueryPolicyAnalyzerAdapter(
        BoundedQueryPolicy(MappingScorer(0.9), confidence_threshold=0.5),
        max_actions=5,
    )

    result = asyncio.run(adapter("graph retrieval papers", _reservation()))

    analysis = QueryAnalysisResult.model_validate(result.data)
    assert analysis.query_spec.original_query == "graph retrieval papers"
    assert analysis.search_plan.subqueries[0].text == "graph retrieval papers"
    assert len(analysis.search_plan.subqueries) >= 3
    assert result.usage == UsageActual(cost_cny=Decimal("0"))
    assert result.provenance["provider"] == "local_query_policy"
    assert result.provenance["model_id"] == "fake-action-ranker-v1"


def test_analyzer_adapter_calls_existing_llm_analyzer_below_threshold() -> None:
    calls = 0

    async def fallback(
        query: str,
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, object]]:
        nonlocal calls
        calls += 1
        assert query == "graph retrieval papers"
        assert reservation.reservation_id == "r1"
        return ProviderResult(
            data={"fallback": True},
            usage=UsageActual(llm_calls=1),
            provenance={
                "provider": "llm",
                "endpoint": "query_analyze",
                "model_id": "fallback-model",
                "requested_at": "2026-01-01T00:00:00Z",
                "response_hash": "sha256:" + "0" * 64,
                "snapshot_refs": "[]",
            },
            cache_hit=False,
            latency_ms=1,
            errors=[],
        )

    adapter = QueryPolicyAnalyzerAdapter(
        BoundedQueryPolicy(MappingScorer(0.1), confidence_threshold=0.5),
        max_actions=5,
        fallback=fallback,
    )

    result = asyncio.run(adapter("graph retrieval papers", _reservation()))

    assert calls == 1
    assert result.data == {"fallback": True}
