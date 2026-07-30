from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    AfterValidator,
    model_validator,
)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictNonNegativeFloat = Annotated[
    float,
    Field(strict=True, ge=0, allow_inf_nan=False),
]
UnitFloat = Annotated[float, Field(ge=0, le=1)]
ProviderName = Literal["openalex", "semantic_scholar"]
SearchMode = Literal["replay", "live"]
DependencyName = Literal["llm", "openalex", "semantic_scholar"]
DependencyState = Literal["ready", "replayed", "degraded", "failed"]
PlannerStatus = Literal["primary", "repaired", "rules_fallback"]
DependencyErrorCode = Literal[
    "timeout",
    "network_error",
    "rate_limited",
    "server_error",
    "authentication_error",
    "invalid_request",
    "invalid_response",
    "invalid_record",
    "missing_record",
    "empty_response",
    "invalid_json",
    "budget_exhausted",
    "provider_error",
]


def validate_safe_relative_path(value: str) -> str:
    """Normalize a portable relative path while rejecting traversal and roots."""
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or str(path) == "."
        or any(part == ".." or ":" in part for part in path.parts)
    ):
        raise ValueError("snapshot path must be a safe relative path")
    return path.as_posix()


SafeRelativePath = Annotated[str, AfterValidator(validate_safe_relative_path)]
MoneyCny = Annotated[Decimal, Field(ge=Decimal("0"), decimal_places=6)]


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_years(year_from: int | None, year_to: int | None) -> None:
    maximum_year = date.today().year + 1
    for name, value in (("year_from", year_from), ("year_to", year_to)):
        if value is not None and not 1900 <= value <= maximum_year:
            raise ValueError(f"{name} must be between 1900 and {maximum_year}")
    if year_from is not None and year_to is not None and year_from > year_to:
        raise ValueError("year_from must not exceed year_to")


class QuerySpec(DomainModel):
    original_query: NonEmptyStr
    research_goal: NonEmptyStr
    topics: list[NonEmptyStr] = Field(default_factory=list)
    methods: list[NonEmptyStr] = Field(default_factory=list)
    tasks: list[NonEmptyStr] = Field(default_factory=list)
    datasets: list[NonEmptyStr] = Field(default_factory=list)
    domains: list[NonEmptyStr] = Field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    venues: list[NonEmptyStr] = Field(default_factory=list)
    must_have: list[NonEmptyStr] = Field(default_factory=list)
    should_have: list[NonEmptyStr] = Field(default_factory=list)
    exclusions: list[NonEmptyStr] = Field(default_factory=list)
    ambiguities: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_year_range(self) -> QuerySpec:
        _validate_years(self.year_from, self.year_to)
        return self


class SubQuery(DomainModel):
    query_id: NonEmptyStr
    text: NonEmptyStr
    query_type: Literal["exact", "expanded", "decomposed"]
    target_constraints: list[NonEmptyStr] = Field(default_factory=list)
    priority: Annotated[int, Field(strict=True, gt=0)]
    provider_hint: Literal["openalex", "semantic_scholar", "either"]


class SearchPlan(DomainModel):
    subqueries: list[SubQuery] = Field(min_length=1)
    inherited_hard_filters: dict[str, object]
    rationale: NonEmptyStr


class ProviderPaperId(DomainModel):
    provider: ProviderName
    value: NonEmptyStr


class Paper(DomainModel):
    canonical_id: NonEmptyStr
    title: NonEmptyStr
    abstract: str | None = None
    authors: list[NonEmptyStr] = Field(default_factory=list)
    publication_year: int | None = None
    venue: str | None = None
    doi: str | None = None
    openalex_id: str | None = None
    semantic_scholar_id: str | None = None
    url: str | None = None
    citation_count: NonNegativeInt | None = None
    reference_ids: list[ProviderPaperId] = Field(default_factory=list)
    cited_by_ids: list[ProviderPaperId] = Field(default_factory=list)
    is_retracted: bool | None = None
    sources: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_publication_year(self) -> Paper:
        _validate_years(self.publication_year, self.publication_year)
        return self


class CitationEdge(DomainModel):
    provider: ProviderName
    citing_provider_id: ProviderPaperId
    cited_provider_id: ProviderPaperId
    citing_canonical_id: str | None = None
    cited_canonical_id: str | None = None

    @model_validator(mode="after")
    def validate_provider_consistency(self) -> CitationEdge:
        endpoint_providers = {
            self.citing_provider_id.provider,
            self.cited_provider_id.provider,
        }
        if endpoint_providers != {self.provider}:
            raise ValueError("citation edge provider must match both endpoint providers")
        return self


class ResolvedCitationEdge(DomainModel):
    provider: ProviderName
    citing_canonical_id: NonEmptyStr
    cited_canonical_id: NonEmptyStr
    source_edge_hash: NonEmptyStr


class CandidateEvidence(DomainModel):
    paper_id: NonEmptyStr
    matched_subqueries: list[NonEmptyStr] = Field(default_factory=list)
    matched_constraints: list[NonEmptyStr] = Field(default_factory=list)
    unmatched_constraints: list[NonEmptyStr] = Field(default_factory=list)
    filter_reasons: list[NonEmptyStr] = Field(default_factory=list)
    lexical_score: float
    embedding_score: float
    rerank_score: float | None = None
    constraint_coverage: UnitFloat
    source_agreement: UnitFloat
    authority_score: UnitFloat
    recency_score: UnitFloat
    final_score: UnitFloat
    scoring_version: NonEmptyStr
    relevance_level: Literal["high", "partial", "irrelevant"]


class RankedPaper(DomainModel):
    paper: Paper
    evidence: CandidateEvidence


class SearchBudget(DomainModel):
    max_search_api_calls: StrictNonNegativeInt = 12
    target_search_api_calls: StrictNonNegativeInt = 8
    max_llm_calls: StrictNonNegativeInt = 5
    target_llm_calls: StrictNonNegativeInt = 3
    max_iterations: StrictNonNegativeInt = 2
    max_subqueries: StrictNonNegativeInt = 6
    max_rerank_candidates: StrictNonNegativeInt = 30
    max_output_papers: StrictNonNegativeInt = 50
    max_citation_seeds: StrictNonNegativeInt = 2
    target_citation_seeds: StrictNonNegativeInt = 1
    max_elapsed_seconds: StrictNonNegativeInt = 90
    soft_deadline_seconds: StrictNonNegativeInt = 80
    max_total_tokens: StrictNonNegativeInt
    max_cost_cny: StrictNonNegativeFloat

    @model_validator(mode="after")
    def validate_targets(self) -> SearchBudget:
        pairs = (
            ("target_search_api_calls", "max_search_api_calls"),
            ("target_llm_calls", "max_llm_calls"),
            ("target_citation_seeds", "max_citation_seeds"),
        )
        for target_name, maximum_name in pairs:
            if getattr(self, target_name) > getattr(self, maximum_name):
                raise ValueError(f"{target_name} must not exceed {maximum_name}")
        if self.soft_deadline_seconds >= self.max_elapsed_seconds:
            raise ValueError("soft_deadline_seconds must be less than max_elapsed_seconds")
        return self


class UsageEstimate(DomainModel):
    search_api_calls: StrictNonNegativeInt = 0
    llm_calls: StrictNonNegativeInt = 0
    input_tokens: StrictNonNegativeInt = 0
    output_tokens: StrictNonNegativeInt = 0
    cost_cny: MoneyCny | None = None
    elapsed_ms: StrictNonNegativeInt = 0


class UsageActual(UsageEstimate):
    pass


class BudgetReservation(DomainModel):
    reservation_id: NonEmptyStr
    action: NonEmptyStr
    reserved: UsageEstimate
    expires_at: datetime


class ErrorDetail(DomainModel):
    code: NonEmptyStr
    message: NonEmptyStr
    retryable: bool
    provider: NonEmptyStr
    request_id: str | None = None


class DependencyStatus(DomainModel):
    dependency: DependencyName
    state: DependencyState
    cache_hit: bool
    error_codes: list[DependencyErrorCode]


_DEPENDENCY_ORDER = {"llm": 0, "openalex": 1, "semantic_scholar": 2}


def validate_dependency_status_order(statuses: list[DependencyStatus]) -> None:
    dependencies = [status.dependency for status in statuses]
    if len(set(dependencies)) != len(dependencies) or dependencies != sorted(
        dependencies, key=_DEPENDENCY_ORDER.__getitem__
    ):
        raise ValueError(
            "dependency status order must be llm, openalex, semantic_scholar"
        )


class CitationExpansion(DomainModel):
    papers: list[Paper]
    raw_edges: list[CitationEdge]


T = TypeVar("T")


class ProviderResult(DomainModel, Generic[T]):
    data: T
    usage: UsageActual
    provenance: dict[str, NonEmptyStr]
    cache_hit: bool
    latency_ms: NonNegativeInt
    errors: list[ErrorDetail]

    @model_validator(mode="after")
    def validate_provenance(self) -> ProviderResult[T]:
        required = {"provider", "endpoint", "model_id", "requested_at", "response_hash"}
        missing = required.difference(self.provenance)
        if missing:
            raise ValueError(f"provenance missing required fields: {sorted(missing)}")
        return self


class QueryAnalysisResult(DomainModel):
    query_spec: QuerySpec
    search_plan: SearchPlan


class StructuredSearchResponse(DomainModel):
    run_id: NonEmptyStr = "legacy-run"
    query_id: NonEmptyStr
    execution_mode: SearchMode = "replay"
    snapshot_set_id: NonEmptyStr = "legacy-snapshot"
    snapshot_captured_at: datetime | None = None
    query_analysis: QueryAnalysisResult
    selected_paper_ids: list[NonEmptyStr]
    high_relevance: list[RankedPaper]
    partial_relevance: list[RankedPaper]
    citation_edges: list[ResolvedCitationEdge]
    search_trace: list[dict[str, object]]
    usage: UsageActual
    stop_reason: NonEmptyStr
    is_partial: bool
    planner_fallback: bool = False
    planner_status: PlannerStatus = "primary"
    dependency_status: list[DependencyStatus] = Field(default_factory=list)
    warnings: list[NonEmptyStr]
    prompt_version: NonEmptyStr = "legacy-prompt"
    config_hash: Sha256
    git_sha: NonEmptyStr

    @model_validator(mode="after")
    def validate_execution_invariants(self) -> StructuredSearchResponse:
        validate_dependency_status_order(self.dependency_status)
        if self.planner_status == "rules_fallback":
            if not self.planner_fallback or not self.is_partial:
                raise ValueError(
                    "rules_fallback requires planner_fallback and is_partial"
                )
            if "planner_rules_fallback" not in self.warnings:
                raise ValueError(
                    "rules_fallback requires planner_rules_fallback warning"
                )
        elif self.planner_fallback:
            raise ValueError("only rules_fallback may set planner_fallback")
        return self


for _model in (
    CitationExpansion,
    ProviderResult,
    QueryAnalysisResult,
    StructuredSearchResponse,
):
    _model.model_rebuild()
