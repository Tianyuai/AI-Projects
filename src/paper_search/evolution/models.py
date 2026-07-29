from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, SubQuery

ConstraintKind = Literal[
    "must_have", "topics", "methods", "tasks", "datasets", "domains", "venues"
]
CoverageStatus = Literal["covered", "low_coverage", "uncovered"]


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
