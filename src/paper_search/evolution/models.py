from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, SubQuery

ConstraintKind = Literal[
    "must_have", "topics", "methods", "tasks", "datasets", "domains", "venues"
]
CoverageStatus = Literal["covered", "low_coverage", "uncovered"]
EvolutionStrategy = Literal["fixed_one_round", "fixed_two_round", "adaptive"]
StopReason = Literal[
    "round_failed",
    "coverage_complete",
    "max_rounds_reached",
    "marginal_gain_below_threshold",
    "budget_insufficient",
    "continue_evolution",
]


class ConstraintRef(DomainModel):
    kind: ConstraintKind
    value: NonEmptyStr
    normalized_value: NonEmptyStr


class CandidateConstraintObservation(DomainModel):
    paper_id: NonEmptyStr
    constraint: ConstraintRef
    matched: bool


class ConstraintCoverage(DomainModel):
    constraint: ConstraintRef
    matched_candidate_ids: list[NonEmptyStr]
    hit_count: int = Field(strict=True, ge=0)
    status: CoverageStatus


class CoverageReport(DomainModel):
    constraints: list[ConstraintCoverage]
    covered_count: int = Field(strict=True, ge=0)
    low_coverage_count: int = Field(strict=True, ge=0)
    uncovered_count: int = Field(strict=True, ge=0)

    @property
    def is_complete(self) -> bool:
        return self.low_coverage_count == 0 and self.uncovered_count == 0


class RoundPlan(DomainModel):
    round_number: int = Field(strict=True, gt=0)
    subqueries: list[SubQuery] = Field(min_length=1)


class MarginalGain(DomainModel):
    new_candidate_count: int = Field(strict=True, ge=0)
    new_high_relevance_count: int = Field(strict=True, ge=0)
    score: float = Field(ge=0, allow_inf_nan=False)
    f1_delta: float | None = Field(default=None, allow_inf_nan=False)
    recall_delta: float | None = Field(default=None, allow_inf_nan=False)


class StopDecision(DomainModel):
    should_continue: bool
    reason_code: StopReason
    strategy: EvolutionStrategy
    completed_rounds: int = Field(strict=True, ge=0)
    max_rounds: int = Field(strict=True, gt=0)
    marginal_gain_threshold: float = Field(ge=0, allow_inf_nan=False)
    checks: dict[str, bool]
    failed_stage: str | None = None
