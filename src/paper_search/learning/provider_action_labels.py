"""Provider-observed supervision for query-action ranking."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from paper_search.domain.models import (
    DomainModel,
    ErrorDetail,
    NonEmptyStr,
    Paper,
    UnitFloat,
    UsageActual,
)
from paper_search.evaluation.dataset import normalize_paper_id
from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.data_isolation import DatasetRole
from paper_search.recall_experiments.paper_identity import arxiv_datacite_anchor


Provider = Literal["openalex", "semantic_scholar"]
UNAVAILABLE_PROVIDER_ERROR_CODES = frozenset({"invalid_request"})


class ProviderActionObservation(DomainModel):
    provider: Provider
    action: PolicyActionCandidate
    hits: list[Paper] = Field(default_factory=list)
    usage: UsageActual = Field(default_factory=UsageActual)
    errors: list[ErrorDetail] = Field(default_factory=list)
    infrastructure_failure: bool = False


class ProviderActionLabel(DomainModel):
    dataset: NonEmptyStr
    split: NonEmptyStr
    role: Literal["training", "development"]
    query_id: NonEmptyStr
    query: NonEmptyStr
    provider: Provider
    action: PolicyActionCandidate
    retrieval_status: Literal["available", "unavailable"]
    gold_association_count: int | None = Field(default=None, strict=True, gt=0)
    gold_hit_ids: tuple[NonEmptyStr, ...] = ()
    gold_hit_count: int | None = Field(default=None, strict=True, ge=0)
    action_recall: UnitFloat | None = None
    novel_over_anchor_hit_count: int | None = Field(default=None, strict=True, ge=0)
    error_codes: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def validate_reward_shape(self) -> ProviderActionLabel:
        rewards = (
            self.gold_association_count,
            self.gold_hit_count,
            self.action_recall,
            self.novel_over_anchor_hit_count,
        )
        if self.retrieval_status == "available" and any(
            value is None for value in rewards
        ):
            raise ValueError("available retrieval labels require rewards")
        if self.retrieval_status == "unavailable" and any(
            value is not None for value in rewards
        ):
            raise ValueError("unavailable retrieval labels forbid rewards")
        if self.retrieval_status == "unavailable" and self.gold_hit_ids:
            raise ValueError("unavailable retrieval labels forbid Gold hits")
        if self.gold_hit_count is not None and self.gold_hit_count != len(
            self.gold_hit_ids
        ):
            raise ValueError("Gold hit count must match Gold hit IDs")
        return self


def _identity(value: str) -> str:
    normalized = normalize_paper_id(value)
    if normalized.startswith("arxiv:"):
        return arxiv_datacite_anchor(normalized)
    return normalized


def _paper_identities(paper: Paper) -> set[str]:
    values = [paper.canonical_id]
    if paper.doi is not None:
        values.append(f"doi:{paper.doi}")
    if paper.arxiv_id is not None:
        values.append(f"arxiv:{paper.arxiv_id}")
    if paper.openalex_id is not None:
        values.append(f"openalex:{paper.openalex_id}")
    if paper.semantic_scholar_id is not None:
        values.append(f"s2:{paper.semantic_scholar_id}")
    return {_identity(value) for value in values}


def build_provider_action_labels(
    *,
    dataset: str,
    split: str,
    role: DatasetRole,
    query_id: str,
    query: str,
    gold_paper_ids: list[str],
    observations: list[ProviderActionObservation],
) -> list[ProviderActionLabel]:
    if role == "final_test":
        raise ValueError("final_test cannot produce provider action labels")
    if not gold_paper_ids:
        raise ValueError("provider action labels require Gold associations")
    output_role: Literal["training", "development"] = role
    gold = {_identity(value) for value in gold_paper_ids}
    hit_sets: list[set[str] | None] = []
    for observation in observations:
        if observation.infrastructure_failure or any(
            error.code in UNAVAILABLE_PROVIDER_ERROR_CODES
            for error in observation.errors
        ):
            hit_sets.append(None)
            continue
        identities = set().union(
            *(_paper_identities(paper) for paper in observation.hits)
        ) if observation.hits else set()
        hit_sets.append(gold.intersection(identities))
    anchor_by_provider: dict[Provider, set[str]] = {}
    for observation, hits in zip(observations, hit_sets, strict=True):
        if observation.action.origin == "original_query" and hits is not None:
            anchor_by_provider[observation.provider] = hits

    labels: list[ProviderActionLabel] = []
    for observation, hits in zip(observations, hit_sets, strict=True):
        error_codes = tuple(sorted({error.code for error in observation.errors}))
        if hits is None:
            labels.append(
                ProviderActionLabel(
                    dataset=dataset,
                    split=split,
                    role=output_role,
                    query_id=query_id,
                    query=query,
                    provider=observation.provider,
                    action=observation.action,
                    retrieval_status="unavailable",
                    error_codes=error_codes,
                )
            )
            continue
        anchor_hits = anchor_by_provider.get(observation.provider, set())
        labels.append(
            ProviderActionLabel(
                dataset=dataset,
                split=split,
                role=output_role,
                query_id=query_id,
                query=query,
                provider=observation.provider,
                action=observation.action,
                retrieval_status="available",
                gold_association_count=len(gold),
                gold_hit_ids=tuple(sorted(hits)),
                gold_hit_count=len(hits),
                action_recall=len(hits) / len(gold),
                novel_over_anchor_hit_count=len(hits.difference(anchor_hits)),
                error_codes=error_codes,
            )
        )
    return labels


__all__ = [
    "ProviderActionLabel",
    "ProviderActionObservation",
    "UNAVAILABLE_PROVIDER_ERROR_CODES",
    "build_provider_action_labels",
]
