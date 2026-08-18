"""Adapters that expose one query policy to experiments and production orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

from paper_search.domain.models import (
    BudgetReservation,
    ProviderResult,
    QueryAnalysisResult,
    SearchPlan,
    SubQuery,
    UsageActual,
)
from paper_search.learning.contracts import PolicyActionCandidate, QueryPolicyInput
from paper_search.learning.candidates import DeterministicActionCandidateGenerator
from paper_search.learning.policy import BoundedQueryPolicy
from paper_search.learning.routing import RuleQueryRouter
from paper_search.query.planner import QueryPlanner
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    RecallGenerationContext,
    TextSearchAction,
    TextSearchPayload,
    TitleSearchAction,
    TitleSearchPayload,
)
from paper_search.recall_experiments.generation.base import GenerationResult, QueryGenerator


AnalyzerFallback = Callable[
    [str, BudgetReservation],
    Awaitable[ProviderResult[dict[str, Any]]],
]


def _seed_actions(seed_queries: list[str]) -> list[PolicyActionCandidate]:
    return [
        PolicyActionCandidate(
            action_id=f"seed-{index}",
            action_type="text_search",
            text=query,
            origin="seed_query",
            provider_hint="either",
        )
        for index, query in enumerate(seed_queries, start=1)
    ]


class RecallQueryPolicyGenerator:
    """Use the local action policy behind Scheme B's QueryGenerator boundary."""

    def __init__(
        self,
        policy: BoundedQueryPolicy,
        *,
        max_actions: int,
        candidate_pool_size: int = 12,
        fallback: QueryGenerator | None = None,
        source_sha256: str | None = None,
        force_lexical_unique: bool = False,
    ) -> None:
        self._policy = policy
        self._max_actions = max_actions
        self._fallback = fallback
        self._force_lexical_unique = force_lexical_unique
        self._candidate_pool_size = candidate_pool_size
        self.source_sha256 = source_sha256
        self.generator_type = (
            "local_cpu_fallback" if fallback is not None else "local_cpu"
        )
        self.model_id = policy.model_id
        self._router = RuleQueryRouter()
        self._candidate_generator = DeterministicActionCandidateGenerator(
            max_candidates=candidate_pool_size
        )

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        routed = self._router.route(context.original_query)
        local_candidates = self._candidate_generator.generate(
            routed.query_spec,
            query_kind=routed.query_kind,
        )
        request = QueryPolicyInput(
            query_id=context.query_id,
            original_query=context.original_query,
            query_kind=routed.query_kind,
            query_spec=context.query_spec,
            seed_actions=[
                *local_candidates,
                *_seed_actions(context.seed_queries),
            ],
            allowed_action_types=["text_search", "title_search"],
            max_actions=(
                min(self._candidate_pool_size, 10)
                if self._force_lexical_unique
                else self._max_actions
            ),
        )
        output = self._policy.plan(request)
        if output.fallback_required and self._fallback is not None:
            fallback_result = await self._fallback.generate(context)
            return fallback_result.model_copy(
                update={
                    "provenance": {
                        **fallback_result.provenance,
                        "query_policy_gate": "confidence_below_threshold",
                        "query_policy_model_id": output.model_id,
                    }
                }
            )

        actions: list[TextSearchAction | TitleSearchAction] = []
        seen_search_texts: set[str] = set()
        for ranked in output.ranked_actions:
            candidate = ranked.action
            if (
                self._force_lexical_unique
                and candidate.action_type == "text_search"
                and candidate.search_mode != "lexical"
            ):
                continue
            normalized_text = " ".join(candidate.text.casefold().split())
            if self._force_lexical_unique and normalized_text in seen_search_texts:
                continue
            seen_search_texts.add(normalized_text)
            index = len(actions) + 1
            common = {
                "action_id": f"policy-{index}",
                "strategy": "learned_action_ranker",
            }
            if candidate.action_type == "title_search":
                actions.append(
                    TitleSearchAction(
                        **common,
                        action_type="title_search",
                        payload=TitleSearchPayload(title_text=candidate.text),
                    )
                )
            else:
                actions.append(
                    TextSearchAction(
                        **common,
                        action_type="text_search",
                        payload=TextSearchPayload(
                            query_text=candidate.text,
                            search_mode=candidate.search_mode,
                        ),
                    )
                )
            if len(actions) >= self._max_actions:
                break
        batch = RecallActionBatch(actions=actions)
        artifact_bytes = json.dumps(
            batch.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return GenerationResult(
            query_id=context.query_id,
            action_batch=batch,
            artifact_bytes=artifact_bytes,
            provenance={
                "generator": "local_query_policy",
                "model_id": output.model_id,
                "query_kind": output.query_kind,
                "confidence": f"{output.confidence:.6f}",
                "fallback_required": str(output.fallback_required).lower(),
            },
        )


class QueryPolicyAnalyzerAdapter:
    """Use the same local policy behind the production Analyzer callable."""

    def __init__(
        self,
        policy: BoundedQueryPolicy,
        *,
        max_actions: int,
        candidate_pool_size: int = 12,
        fallback: AnalyzerFallback | None = None,
    ) -> None:
        self._policy = policy
        self._max_actions = max_actions
        self._fallback = fallback
        self._router = RuleQueryRouter()
        self._planner = QueryPlanner()
        self._candidate_generator = DeterministicActionCandidateGenerator(
            max_candidates=candidate_pool_size
        )

    async def __call__(
        self,
        query: str,
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]:
        routed = self._router.route(query)
        output = self._policy.plan(
            QueryPolicyInput(
                query_id="production-query",
                original_query=routed.query_spec.original_query,
                query_kind=routed.query_kind,
                query_spec=routed.query_spec,
                seed_actions=self._candidate_generator.generate(
                    routed.query_spec,
                    query_kind=routed.query_kind,
                ),
                allowed_action_types=["text_search", "title_search"],
                max_actions=self._max_actions,
            )
        )
        if output.fallback_required and self._fallback is not None:
            return await self._fallback(query, reservation)

        constraints = (
            routed.query_spec.must_have
            + routed.query_spec.topics
            + routed.query_spec.methods
            + routed.query_spec.tasks
            + routed.query_spec.datasets
            + routed.query_spec.venues
        )
        subqueries = [
            SubQuery(
                query_id=f"policy-{index}",
                text=ranked.action.text,
                query_type=(
                    "exact"
                    if ranked.action.origin == "original_query"
                    or ranked.action.action_type == "title_search"
                    else "expanded"
                ),
                action_type=ranked.action.action_type,
                target_constraints=constraints,
                priority=index,
                provider_hint=ranked.action.provider_hint,
                search_mode=ranked.action.search_mode,
            )
            for index, ranked in enumerate(output.ranked_actions, start=1)
        ]
        plan = self._planner.finalize(
            routed.query_spec,
            SearchPlan(
                subqueries=subqueries,
                inherited_hard_filters={},
                rationale="Local bounded action policy",
            ),
        )
        analysis = QueryAnalysisResult(query_spec=routed.query_spec, search_plan=plan)
        data = analysis.model_dump(mode="json")
        encoded = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return ProviderResult(
            data=data,
            usage=UsageActual(cost_cny=Decimal("0")),
            provenance={
                "provider": "local_query_policy",
                "endpoint": "query_analyze",
                "model_id": output.model_id,
                "requested_at": "1970-01-01T00:00:00Z",
                "response_hash": "sha256:" + hashlib.sha256(encoded).hexdigest(),
                "snapshot_refs": "[]",
                "query_kind": output.query_kind,
                "confidence": f"{output.confidence:.6f}",
                "fallback_required": str(output.fallback_required).lower(),
            },
            cache_hit=False,
            latency_ms=0,
            errors=[],
        )


__all__ = ["QueryPolicyAnalyzerAdapter", "RecallQueryPolicyGenerator"]
