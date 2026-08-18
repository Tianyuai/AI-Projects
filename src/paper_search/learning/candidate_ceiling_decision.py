"""Pre-registered evidence gates for candidate-ceiling experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import Field, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, UnitFloat
from paper_search.learning.action_diagnostics import diagnose_action_selection
from paper_search.learning.contracts import QueryPolicyInput
from paper_search.learning.policy import BoundedQueryPolicy
from paper_search.learning.provider_action_labels import Provider, ProviderActionLabel
from paper_search.learning.routing import RuleQueryRouter


class ProviderCeilingBatchEvidence(DomainModel):
    provider: Provider
    query_count: int = Field(strict=True, ge=0)
    current_top3_macro_recall: UnitFloat
    oracle_at_3_macro_recall: UnitFloat
    all_candidate_macro_recall: UnitFloat
    preference_query_count: int = Field(strict=True, ge=0)
    unavailable_action_count: int = Field(strict=True, ge=0)


class CandidateCeilingBatchEvidence(DomainModel):
    batch_id: NonEmptyStr
    providers: list[ProviderCeilingBatchEvidence]

    @model_validator(mode="after")
    def validate_unique_providers(self) -> CandidateCeilingBatchEvidence:
        providers = [item.provider for item in self.providers]
        if not providers or len(providers) != len(set(providers)):
            raise ValueError("batch evidence requires unique providers")
        return self


class CandidateCeilingDecision(DomainModel):
    status: Literal["insufficient_evidence", "complete"]
    observed_batch_count: int = Field(strict=True, ge=0)
    required_batch_count: int = Field(strict=True, gt=0)
    required_consistent_batches: int = Field(strict=True, gt=0)
    minimum_preference_queries: int = Field(strict=True, gt=0)
    minimum_pooled_macro_gap: float = Field(strict=True, ge=0)
    ranking_optimization_supported_providers: tuple[Provider, ...]
    candidate_generation_change_supported: bool = False


def _union_recall(rows: list[ProviderActionLabel]) -> float:
    gold_count = rows[0].gold_association_count
    assert gold_count is not None
    hits = set().union(*(set(row.gold_hit_ids) for row in rows))
    return len(hits) / gold_count


def _has_preference(rows: list[ProviderActionLabel]) -> bool:
    rewards = {
        (
            row.action_recall,
            row.novel_over_anchor_hit_count,
        )
        for row in rows
    }
    return len(rewards) > 1


def _action_key(
    action_type: str,
    text: str,
    search_mode: str,
) -> tuple[str, str, str]:
    return action_type, search_mode, " ".join(text.casefold().split())


def summarize_candidate_ceiling_batch(
    labels: list[ProviderActionLabel],
    *,
    batch_id: str,
    policy: BoundedQueryPolicy,
) -> CandidateCeilingBatchEvidence:
    validated = [ProviderActionLabel.model_validate(label) for label in labels]
    if not validated:
        raise ValueError("candidate ceiling labels are empty")
    grouped: dict[tuple[Provider, str], list[ProviderActionLabel]] = defaultdict(list)
    for label in validated:
        grouped[(label.provider, label.query_id)].append(label)
    router = RuleQueryRouter()
    diagnostics: dict[Provider, list[tuple[float, float, float, bool]]] = defaultdict(list)
    unavailable_counts: dict[Provider, int] = defaultdict(int)
    for (provider, _query_id), rows in grouped.items():
        unavailable_counts[provider] += sum(
            row.retrieval_status == "unavailable" for row in rows
        )
        if any(row.retrieval_status == "unavailable" for row in rows):
            continue
        routed = router.route(rows[0].query)
        output = policy.plan(
            QueryPolicyInput(
                query_id=rows[0].query_id,
                original_query=rows[0].query,
                query_kind=routed.query_kind,
                query_spec=routed.query_spec,
                seed_actions=[row.action for row in rows],
                allowed_action_types=["text_search", "title_search"],
                max_actions=3,
            )
        )
        receipt_id_by_action = {
            _action_key(
                row.action.action_type,
                row.action.text,
                row.action.search_mode,
            ): row.action.action_id
            for row in rows
        }
        selected: list[tuple[str, str]] = [
            (
                receipt_id_by_action[
                    _action_key(
                        item.action.action_type,
                        item.action.text,
                        item.action.search_mode,
                    )
                ],
                provider,
            )
            for item in output.ranked_actions
        ]
        diagnostic = diagnose_action_selection(
            rows,
            selected_action_provider_pairs=selected,
            max_actions=3,
        )
        diagnostics[provider].append(
            (
                diagnostic.selected_recall,
                diagnostic.oracle_recall,
                _union_recall(rows),
                _has_preference(rows),
            )
        )
    providers = []
    for provider in sorted({label.provider for label in validated}):
        values = diagnostics[provider]
        if not values:
            providers.append(
                ProviderCeilingBatchEvidence(
                    provider=provider,
                    query_count=0,
                    current_top3_macro_recall=0.0,
                    oracle_at_3_macro_recall=0.0,
                    all_candidate_macro_recall=0.0,
                    preference_query_count=0,
                    unavailable_action_count=unavailable_counts[provider],
                )
            )
            continue
        providers.append(
            ProviderCeilingBatchEvidence(
                provider=provider,
                query_count=len(values),
                current_top3_macro_recall=sum(item[0] for item in values) / len(values),
                oracle_at_3_macro_recall=sum(item[1] for item in values) / len(values),
                all_candidate_macro_recall=sum(item[2] for item in values) / len(values),
                preference_query_count=sum(item[3] for item in values),
                unavailable_action_count=unavailable_counts[provider],
            )
        )
    return CandidateCeilingBatchEvidence(batch_id=batch_id, providers=providers)


def assess_candidate_ceiling(
    batches: list[CandidateCeilingBatchEvidence],
    *,
    required_batch_count: int = 3,
    required_consistent_batches: int = 2,
    minimum_preference_queries: int = 20,
    minimum_pooled_macro_gap: float = 0.03,
) -> CandidateCeilingDecision:
    validated = [CandidateCeilingBatchEvidence.model_validate(batch) for batch in batches]
    unique_batch_ids = {batch.batch_id for batch in validated}
    if len(unique_batch_ids) != required_batch_count:
        return CandidateCeilingDecision(
            status="insufficient_evidence",
            observed_batch_count=len(unique_batch_ids),
            required_batch_count=required_batch_count,
            required_consistent_batches=required_consistent_batches,
            minimum_preference_queries=minimum_preference_queries,
            minimum_pooled_macro_gap=minimum_pooled_macro_gap,
            ranking_optimization_supported_providers=(),
        )
    by_provider: dict[Provider, list[ProviderCeilingBatchEvidence]] = defaultdict(list)
    for batch in validated:
        for provider_evidence in batch.providers:
            by_provider[provider_evidence.provider].append(provider_evidence)
    supported: list[Provider] = []
    for provider_name, rows in by_provider.items():
        if len(rows) != required_batch_count:
            continue
        total_queries = sum(row.query_count for row in rows)
        if total_queries == 0:
            continue
        pooled_current = sum(
            row.current_top3_macro_recall * row.query_count for row in rows
        ) / total_queries
        pooled_oracle = sum(
            row.oracle_at_3_macro_recall * row.query_count for row in rows
        ) / total_queries
        consistent_batches = sum(
            row.oracle_at_3_macro_recall > row.current_top3_macro_recall
            for row in rows
        )
        if (
            consistent_batches >= required_consistent_batches
            and pooled_oracle - pooled_current >= minimum_pooled_macro_gap
            and sum(row.preference_query_count for row in rows)
            >= minimum_preference_queries
        ):
            supported.append(provider_name)
    return CandidateCeilingDecision(
        status="complete",
        observed_batch_count=len(unique_batch_ids),
        required_batch_count=required_batch_count,
        required_consistent_batches=required_consistent_batches,
        minimum_preference_queries=minimum_preference_queries,
        minimum_pooled_macro_gap=minimum_pooled_macro_gap,
        ranking_optimization_supported_providers=tuple(sorted(supported)),
    )


__all__ = [
    "CandidateCeilingBatchEvidence",
    "CandidateCeilingDecision",
    "ProviderCeilingBatchEvidence",
    "assess_candidate_ceiling",
    "summarize_candidate_ceiling_batch",
]
