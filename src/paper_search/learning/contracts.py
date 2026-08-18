"""Stable contracts shared by query-policy training and inference."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, QuerySpec, UnitFloat


PolicyActionType = Literal["text_search", "title_search"]
PolicyActionOrigin = Literal[
    "original_query",
    "deterministic_rule",
    "seed_query",
    "curated_vocabulary",
]


class PolicyActionCandidate(DomainModel):
    action_id: NonEmptyStr
    action_type: PolicyActionType
    text: NonEmptyStr
    origin: PolicyActionOrigin
    provider_hint: Literal["openalex", "semantic_scholar", "either"]
    search_mode: Literal["lexical", "semantic"] = "lexical"


QueryKind = Literal["navigational", "metadata", "semantic"]


class QueryPolicyInput(DomainModel):
    query_id: NonEmptyStr
    original_query: NonEmptyStr
    query_kind: QueryKind
    query_spec: QuerySpec
    seed_actions: list[PolicyActionCandidate] = Field(default_factory=list)
    allowed_action_types: list[PolicyActionType]
    max_actions: Annotated[int, Field(strict=True, gt=0, le=10)]

    @model_validator(mode="after")
    def validate_query_and_allowed_actions(self) -> QueryPolicyInput:
        def normalize(value: str) -> str:
            return " ".join(value.split()).casefold()

        if normalize(self.original_query) != normalize(self.query_spec.original_query):
            raise ValueError("policy input query must match query_spec original_query")
        if not self.allowed_action_types:
            raise ValueError("at least one action type must be allowed")
        if len(self.allowed_action_types) != len(set(self.allowed_action_types)):
            raise ValueError("allowed action types must be unique")
        return self


class RankedPolicyAction(DomainModel):
    action: PolicyActionCandidate
    score: UnitFloat


class QueryPolicyOutput(DomainModel):
    query_kind: QueryKind
    ranked_actions: list[RankedPolicyAction] = Field(min_length=1)
    confidence: UnitFloat
    fallback_required: bool
    fallback_reason: Literal["confidence_below_threshold"] | None = None
    model_id: NonEmptyStr

    @model_validator(mode="after")
    def validate_anchor_and_fallback(self) -> QueryPolicyOutput:
        if not any(
            item.action.origin == "original_query" for item in self.ranked_actions
        ):
            raise ValueError("ranked actions must retain the original query anchor")
        if self.fallback_required != (self.fallback_reason is not None):
            raise ValueError("fallback reason must match fallback requirement")
        return self


__all__ = [
    "PolicyActionCandidate",
    "PolicyActionOrigin",
    "PolicyActionType",
    "QueryKind",
    "QueryPolicyInput",
    "QueryPolicyOutput",
    "RankedPolicyAction",
]
