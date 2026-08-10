from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from paper_search.application.contracts import DependencyDiagnostic, SnapshotRef
from paper_search.domain.models import (
    BudgetReservation,
    DomainModel,
    ErrorDetail,
    NonEmptyStr,
    QuerySpec,
    SearchPlan,
    UsageActual,
)
from paper_search.retrieval.openalex import canonicalize_openalex_search_query

EvolutionStrategy = Literal[
    "synonym", "entity_alias", "facet_combination", "task_decomposition"
]
NoOpReason = Literal["insufficient_grounded_facets", "no_novel_query"]
QueryEvolutionStatus = Literal[
    "generated",
    "no_op",
    "dependency_failure",
    "integrity_failure",
]
MAX_QUERY_CHARS = 300
_RESPONSE_SCHEMA = "query-evolution-proposal-v1"
_YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
_FIXED_INSTRUCTIONS = [
    "Use only facts and facets present in the payload.",
    "Do not infer gold papers, relevance labels, new venues, new years, or unrelated entities.",
    "Return zero to two complementary OpenAlex queries as strict JSON.",
    "Always include no_op_reason; use null when subqueries contains items.",
    "Copy every source_facets value exactly from the payload.",
]


class EvolutionSeedSubquery(DomainModel):
    text: NonEmptyStr
    target_constraints: list[NonEmptyStr]


class QueryEvolutionContext(DomainModel):
    original_query: NonEmptyStr
    query_spec: QuerySpec
    seed_subqueries: list[EvolutionSeedSubquery] = Field(min_length=3, max_length=5)
    candidate_count: int = Field(strict=True, ge=0)
    top_titles: list[NonEmptyStr] = Field(max_length=10)
    facets: list[NonEmptyStr] = Field(min_length=1)
    instructions: list[NonEmptyStr]
    response_schema: Literal["query-evolution-proposal-v1"]


class EvolutionSubquery(DomainModel):
    text: NonEmptyStr
    source_facets: list[NonEmptyStr] = Field(min_length=1)
    strategy: EvolutionStrategy


class QueryEvolutionProposal(DomainModel):
    subqueries: list[EvolutionSubquery] = Field(max_length=2)
    no_op_reason: NoOpReason | None

    @model_validator(mode="after")
    def validate_no_op(self) -> QueryEvolutionProposal:
        if bool(self.subqueries) == (self.no_op_reason is not None):
            raise ValueError("no_op_reason must exist exactly for an empty proposal")
        return self


class QueryEvolutionResult(DomainModel):
    status: QueryEvolutionStatus
    proposal: QueryEvolutionProposal | None = None
    snapshot_refs: list[SnapshotRef] = Field(default_factory=list)
    diagnostics: list[DependencyDiagnostic] = Field(default_factory=list)
    usage: UsageActual = Field(default_factory=UsageActual)


class _Analyzer(Protocol):
    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
    ) -> Any: ...


def _nfkc_and_collapse(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _dedupe_preserving_first(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _nfkc_and_collapse(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _payload_snapshot_refs(provenance: Mapping[str, object]) -> list[SnapshotRef]:
    raw_refs = provenance.get("snapshot_refs")
    if isinstance(raw_refs, str):
        decoded = json.loads(raw_refs)
        if isinstance(decoded, list):
            return [SnapshotRef.model_validate(item) for item in decoded]
    names = (
        "snapshot_entry_id",
        "snapshot_cache_key",
        "snapshot_response_sha256",
        "snapshot_path",
        "requested_at",
    )
    if all(name in provenance for name in names):
        return [
            SnapshotRef(
                entry_id=str(provenance["snapshot_entry_id"]),
                dependency="llm",
                cache_key=str(provenance["snapshot_cache_key"]),
                response_sha256=str(provenance["snapshot_response_sha256"]),
                captured_at=str(provenance["requested_at"]),
                snapshot_path=str(provenance["snapshot_path"]),
            )
        ]
    return []


def _diagnostic(
    *,
    usage: UsageActual,
    latency_ms: int,
    cache_hit: bool,
    model_id: str | None,
    snapshot_refs: list[SnapshotRef],
    errors: list[ErrorDetail],
) -> DependencyDiagnostic:
    return DependencyDiagnostic(
        dependency="llm",
        endpoint="query_evolve",
        model_id=model_id,
        usage=usage,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        snapshot_refs=snapshot_refs,
        errors=errors,
    )


def _canonical_query(value: str) -> str:
    normalized = _nfkc_and_collapse(value)
    if any(unicodedata.category(char) == "Cc" for char in normalized):
        raise ValueError("subquery text contains control characters")
    canonical = canonicalize_openalex_search_query(normalized)
    if not canonical:
        raise ValueError("subquery text must not be empty after normalization")
    if len(canonical) > MAX_QUERY_CHARS:
        raise ValueError("subquery text must be 300 characters or fewer")
    return canonical


def _validate_year_conflicts(text: str, spec: QuerySpec) -> None:
    for match in _YEAR_PATTERN.finditer(text):
        year = int(match.group(1))
        if spec.year_from is not None and year < spec.year_from:
            raise ValueError("subquery year conflicts with query year constraints")
        if spec.year_to is not None and year > spec.year_to:
            raise ValueError("subquery year conflicts with query year constraints")


def build_query_evolution_context(
    spec: QuerySpec,
    plan: SearchPlan,
    candidate_count: int,
    top_titles: list[str],
) -> QueryEvolutionContext:
    seed_subqueries = [
        EvolutionSeedSubquery(
            text=_nfkc_and_collapse(subquery.text),
            target_constraints=_dedupe_preserving_first(list(subquery.target_constraints)),
        )
        for subquery in plan.subqueries
    ]
    ordered_facets = [
        _nfkc_and_collapse(spec.original_query),
        _nfkc_and_collapse(spec.research_goal),
        *spec.topics,
        *spec.methods,
        *spec.tasks,
        *spec.datasets,
        *spec.domains,
        *spec.venues,
        *spec.must_have,
        *spec.should_have,
    ]
    for subquery in seed_subqueries:
        ordered_facets.extend(subquery.target_constraints)
    return QueryEvolutionContext(
        original_query=_nfkc_and_collapse(spec.original_query),
        query_spec=spec,
        seed_subqueries=seed_subqueries,
        candidate_count=candidate_count,
        top_titles=_dedupe_preserving_first(top_titles)[:10],
        facets=_dedupe_preserving_first(ordered_facets),
        instructions=list(_FIXED_INSTRUCTIONS),
        response_schema=_RESPONSE_SCHEMA,
    )


def validate_query_evolution_proposal(
    raw: object,
    context: QueryEvolutionContext,
) -> QueryEvolutionProposal:
    if not isinstance(raw, Mapping):
        raise ValueError("query evolution proposal must be a JSON object")
    proposal = QueryEvolutionProposal.model_validate(raw)
    if not proposal.subqueries:
        return proposal

    allowed_facets = set(context.facets)
    seen = {
        _canonical_query(context.original_query).casefold(),
        *(_canonical_query(subquery.text).casefold() for subquery in context.seed_subqueries),
    }
    validated_subqueries: list[EvolutionSubquery] = []
    for subquery in proposal.subqueries:
        if any(unicodedata.category(char) == "Cc" for char in unicodedata.normalize("NFKC", subquery.text)):
            raise ValueError("subquery text contains control characters")
        canonical_text = _canonical_query(subquery.text)
        _validate_year_conflicts(canonical_text, context.query_spec)
        key = canonical_text.casefold()
        if key in seen:
            raise ValueError("duplicate subquery text after canonicalization")
        seen.add(key)
        if any(facet not in allowed_facets for facet in subquery.source_facets):
            raise ValueError("source_facets must come from context facets")
        validated_subqueries.append(
            EvolutionSubquery(
                text=canonical_text,
                source_facets=list(subquery.source_facets),
                strategy=subquery.strategy,
            )
        )
    return QueryEvolutionProposal(
        subqueries=validated_subqueries,
        no_op_reason=proposal.no_op_reason,
    )


class QueryEvolutionGenerator:
    def __init__(self, *, analyzer: _Analyzer) -> None:
        self._analyzer = analyzer

    async def generate(
        self,
        context: QueryEvolutionContext,
        reservation: BudgetReservation,
    ) -> QueryEvolutionResult:
        llm_result = await self._analyzer.generate_json(
            prompt_name="query_evolve",
            payload=context.model_dump(mode="json"),
            reservation=reservation,
        )
        snapshot_refs = _payload_snapshot_refs(llm_result.provenance)
        errors = list(llm_result.errors)
        if llm_result.errors:
            return QueryEvolutionResult(
                status="dependency_failure",
                proposal=None,
                snapshot_refs=snapshot_refs,
                diagnostics=[
                    _diagnostic(
                        usage=llm_result.usage,
                        latency_ms=llm_result.latency_ms,
                        cache_hit=llm_result.cache_hit,
                        model_id=llm_result.provenance.get("model_id"),
                        snapshot_refs=snapshot_refs,
                        errors=errors,
                    )
                ],
                usage=llm_result.usage,
            )
        try:
            proposal = validate_query_evolution_proposal(llm_result.data, context)
        except ValueError as error:
            integrity = ErrorDetail(
                code="integrity_failure",
                message=str(error),
                retryable=False,
                provider="llm",
            )
            return QueryEvolutionResult(
                status="integrity_failure",
                proposal=None,
                snapshot_refs=snapshot_refs,
                diagnostics=[
                    _diagnostic(
                        usage=llm_result.usage,
                        latency_ms=llm_result.latency_ms,
                        cache_hit=llm_result.cache_hit,
                        model_id=llm_result.provenance.get("model_id"),
                        snapshot_refs=snapshot_refs,
                        errors=[integrity],
                    )
                ],
                usage=llm_result.usage,
            )
        status: QueryEvolutionStatus = "generated" if proposal.subqueries else "no_op"
        return QueryEvolutionResult(
            status=status,
            proposal=proposal,
            snapshot_refs=snapshot_refs,
            diagnostics=[
                _diagnostic(
                    usage=llm_result.usage,
                    latency_ms=llm_result.latency_ms,
                    cache_hit=llm_result.cache_hit,
                    model_id=llm_result.provenance.get("model_id"),
                    snapshot_refs=snapshot_refs,
                    errors=[],
                )
            ],
            usage=llm_result.usage,
        )
