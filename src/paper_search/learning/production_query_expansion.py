"""Hash-traced supervised query expansion for the production search plan."""

from __future__ import annotations

from paper_search.domain.models import SubQuery
from paper_search.learning.lexical_bridge_deployment import LoadedLexicalBridge
from paper_search.query.parser import ClassifiedQueryAnalysis


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


class SupervisedLexicalBridgePlanEnricher:
    """Append at most one local learned OpenAlex action to a parsed plan."""

    def __init__(
        self,
        bridge: LoadedLexicalBridge,
        *,
        max_total_subqueries: int = 6,
    ) -> None:
        if type(max_total_subqueries) is not int or max_total_subqueries < 1:
            raise ValueError("max_total_subqueries must be a positive integer")
        self._bridge = bridge
        self._max_total_subqueries = max_total_subqueries

    def soft_concept_terms(self, query: str) -> tuple[str, ...]:
        """Expose only terms supported by the exact frozen production bridge."""

        try:
            proposal = self._bridge.bridge.propose(
                query,
                neighbors=self._bridge.neighbors,
                max_expansion_terms=self._bridge.max_expansion_terms,
                min_neighbor_support=self._bridge.min_neighbor_support,
            )
        except Exception:  # fail-safe local inference boundary
            return ()
        return proposal.expansion_terms if proposal is not None else ()

    def enrich(
        self, analysis: ClassifiedQueryAnalysis
    ) -> tuple[ClassifiedQueryAnalysis, dict[str, object]]:
        receipt: dict[str, object] = {
            "step": "supervised_query_expansion",
            "model_id": "supervised-lexical-bridge-openalex-v2",
            "model_sha256": self._bridge.source_sha256,
            "training_query_count": self._bridge.manifest.get(
                "training_query_count"
            ),
            "configured_action_budget": self._max_total_subqueries,
            "action_count_before": len(analysis.search_plan.subqueries),
            "action_count_after": len(analysis.search_plan.subqueries),
            "budget_policy": "llm-replaces-rule-fallback-before-local-bridge",
        }
        plan = analysis.search_plan
        if len(plan.subqueries) >= self._max_total_subqueries:
            return analysis, {**receipt, "status": "action_budget_exhausted"}
        try:
            proposal = self._bridge.bridge.propose(
                analysis.query_spec.original_query,
                neighbors=self._bridge.neighbors,
                max_expansion_terms=self._bridge.max_expansion_terms,
                min_neighbor_support=self._bridge.min_neighbor_support,
            )
        except Exception:  # fail-safe local inference boundary
            return analysis, {**receipt, "status": "inference_failed"}
        if proposal is None:
            return analysis, {**receipt, "status": "abstained"}
        existing = {
            (item.action_type, item.search_mode, _normalized(item.text))
            for item in plan.subqueries
        }
        identity = ("text_search", "lexical", _normalized(proposal.query_text))
        if identity in existing:
            return analysis, {**receipt, "status": "duplicate"}
        target_constraints: list[str] = []
        seen_constraints: set[str] = set()
        for item in plan.subqueries:
            for value in item.target_constraints:
                key = _normalized(value)
                if key and key not in seen_constraints:
                    seen_constraints.add(key)
                    target_constraints.append(value)
        action = SubQuery(
            query_id="sq-supervised-lexical-bridge",
            text=proposal.query_text,
            query_type="expanded",
            action_type="text_search",
            target_constraints=target_constraints,
            priority=len(plan.subqueries) + 1,
            provider_hint="openalex",
            search_mode="lexical",
        )
        enriched = analysis.model_copy(
            update={
                "search_plan": plan.model_copy(
                    update={"subqueries": [*plan.subqueries, action]}
                )
            }
        )
        return enriched, {
            **receipt,
            "status": "appended",
            "action_count_after": len(enriched.search_plan.subqueries),
            "subquery_id": action.query_id,
            "query_text": action.text,
            "expansion_terms": list(proposal.expansion_terms),
            "neighbor_support": dict(proposal.neighbor_support),
            "maximum_similarity": proposal.maximum_similarity,
        }


__all__ = ["SupervisedLexicalBridgePlanEnricher"]
