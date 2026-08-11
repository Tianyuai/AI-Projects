"""Strict, identifier-safe contracts for candidate-recall experiments."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator

from paper_search.domain.models import (
    DomainModel,
    ErrorDetail,
    NonEmptyStr,
    Paper,
    QuerySpec,
    UsageActual,
)


GoldVisibility = Literal["oracle", "blind", "historical"]
ObservableSearchState = Literal[
    "zero_results",
    "low_yield",
    "broad_noisy",
    "facet_gap",
    "duplicate_saturation",
    "entity_ambiguity",
    "provider_failure",
    "adequate",
]
ActionType = Literal["text_search", "title_search", "citation_expand"]
CitationDirection = Literal["references", "citations", "both"]

_PROVIDER_SOURCES = frozenset({"openalex", "semantic_scholar"})
_FORBIDDEN_IDENTIFIER_KEY = re.compile(
    r"(?:^|_)(?:doi|canonical_id|openalex_id|semantic_scholar_id|s2_id|"
    r"provider_request_id|request_id|url)(?:$|_)",
    flags=re.IGNORECASE,
)
_FORBIDDEN_IDENTIFIER_VALUE = re.compile(
    r"(?:\b10\.\d{4,9}/\S+|https?://\S+|\bW\d{6,}\b|\bS2:\S+)",
    flags=re.IGNORECASE,
)


class GoldDocument(DomainModel):
    title: NonEmptyStr
    abstract: str | None = None
    authors: list[NonEmptyStr] = Field(default_factory=list)
    publication_year: int | None = None
    metadata_coverage: dict[NonEmptyStr, bool]


class SeedCandidate(DomainModel):
    paper: Paper

    @model_validator(mode="after")
    def validate_provider_normalized_paper(self) -> SeedCandidate:
        if not set(self.paper.sources).intersection(_PROVIDER_SOURCES):
            raise ValueError("seed candidate must be a provider-normalized paper")
        return self


class RecallGenerationContext(DomainModel):
    query_id: NonEmptyStr
    original_query: NonEmptyStr
    query_spec: QuerySpec
    seed_queries: list[NonEmptyStr] = Field(default_factory=list)
    seed_candidates: list[SeedCandidate] = Field(default_factory=list)
    observable_state: ObservableSearchState | None = None
    gold_documents: list[GoldDocument] = Field(default_factory=list)


class ActionBase(DomainModel):
    action_id: NonEmptyStr
    strategy: NonEmptyStr


class TextSearchPayload(DomainModel):
    query_text: NonEmptyStr


class TitleSearchPayload(DomainModel):
    title_text: NonEmptyStr


class CitationExpandPayload(DomainModel):
    seed_canonical_id: NonEmptyStr
    direction: CitationDirection
    limit: Annotated[int, Field(strict=True, gt=0)]


class TextSearchAction(ActionBase):
    action_type: Literal["text_search"]
    payload: TextSearchPayload


class TitleSearchAction(ActionBase):
    action_type: Literal["title_search"]
    payload: TitleSearchPayload


class CitationExpandAction(ActionBase):
    action_type: Literal["citation_expand"]
    payload: CitationExpandPayload


RecallSearchAction = Annotated[
    TextSearchAction | TitleSearchAction | CitationExpandAction,
    Field(discriminator="action_type"),
]


class RecallActionBatch(DomainModel):
    actions: list[RecallSearchAction]


class RetrievalActionResult(DomainModel):
    action_id: NonEmptyStr
    action_type: ActionType
    hits: list[Paper] = Field(default_factory=list)
    usage: UsageActual = Field(default_factory=UsageActual)
    provenance: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    errors: list[ErrorDetail] = Field(default_factory=list)
    infrastructure_failure: bool = False


class RetrievalExecutionContext(DomainModel):
    query_id: NonEmptyStr
    provider_filters: dict[str, object] = Field(default_factory=dict)
    max_results_per_action: Annotated[int, Field(strict=True, gt=0)]
    seed_candidates: list[SeedCandidate] = Field(default_factory=list)


class RetrievalActionHandler(Protocol):
    async def execute(
        self, action: RecallSearchAction, context: RetrievalExecutionContext
    ) -> RetrievalActionResult: ...


class CandidateSourceEvidence(DomainModel):
    action_id: NonEmptyStr
    action_type: ActionType
    provenance: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)


class CandidatePoolEntry(DomainModel):
    paper: Paper
    source_evidence: list[CandidateSourceEvidence] = Field(default_factory=list)


class CandidatePool(DomainModel):
    query_id: NonEmptyStr
    policy_version: NonEmptyStr
    entries: list[CandidatePoolEntry] = Field(default_factory=list)


def generation_payload(
    context: RecallGenerationContext, *, visibility: GoldVisibility
) -> dict[str, object]:
    """Return the context visible to a generator for the requested Gold mode."""
    payload = context.model_dump(mode="json")
    if visibility == "blind":
        payload.pop("gold_documents", None)
    return payload


def assert_no_forbidden_identifier_keys_or_patterns(payload: object) -> None:
    """Fail closed if Gold-document payload material contains actionable identifiers.

    Seed canonical IDs are intentionally retained under ``seed_candidates`` so the
    fixed citation action contract can reference a provider-normalized seed.
    """
    _scan_identifier_payload(payload, path=())


def _scan_identifier_payload(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError("forbidden non-string identifier key")
            nested_path = (*path, key)
            if _is_seed_candidate_path(nested_path):
                continue
            if _FORBIDDEN_IDENTIFIER_KEY.search(key):
                raise ValueError(f"forbidden identifier key: {'.'.join(nested_path)}")
            _scan_identifier_payload(nested, path=nested_path)
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _scan_identifier_payload(nested, path=(*path, str(index)))
        return
    if isinstance(value, str) and _FORBIDDEN_IDENTIFIER_VALUE.search(value):
        raise ValueError(f"forbidden identifier pattern: {'.'.join(path)}")


def _is_seed_candidate_path(path: tuple[str, ...]) -> bool:
    return len(path) >= 2 and path[0] == "seed_candidates"
