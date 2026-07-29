from collections.abc import Sequence
from dataclasses import dataclass

from paper_search.domain.models import QuerySpec

from .models import (
    CandidateConstraintObservation,
    ConstraintCoverage,
    ConstraintKind,
    ConstraintRef,
    CoverageReport,
)

_FIELDS: tuple[ConstraintKind, ...] = (
    "must_have", "topics", "methods", "tasks", "datasets", "domains", "venues"
)


def normalize_constraint_value(value: str) -> str:
    return " ".join(value.split()).casefold()


def extract_strong_constraints(spec: QuerySpec) -> tuple[ConstraintRef, ...]:
    result: list[ConstraintRef] = []
    for kind in _FIELDS:
        seen: set[str] = set()
        for raw in getattr(spec, kind):
            normalized = normalize_constraint_value(raw)
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(
                ConstraintRef(kind=kind, value=raw, normalized_value=normalized)
            )
    return tuple(result)


def _constraint_key(constraint: ConstraintRef) -> tuple[str, str]:
    return constraint.kind, constraint.normalized_value


@dataclass(frozen=True, slots=True)
class CoverageAnalyzer:
    covered_min_hits: int

    def __post_init__(self) -> None:
        if isinstance(self.covered_min_hits, bool) or not isinstance(
            self.covered_min_hits, int
        ):
            raise ValueError("covered_min_hits must be a positive integer")
        if self.covered_min_hits < 1:
            raise ValueError("covered_min_hits must be a positive integer")

    def analyze(
        self,
        spec: QuerySpec,
        candidate_ids: Sequence[str],
        observations: Sequence[CandidateConstraintObservation],
    ) -> CoverageReport:
        constraints = extract_strong_constraints(spec)
        unique_candidate_ids = sorted(set(candidate_ids))
        expected_keys = {
            (paper_id, _constraint_key(constraint))
            for paper_id in unique_candidate_ids
            for constraint in constraints
        }
        observed: dict[tuple[str, tuple[str, str]], bool] = {}
        for observation in observations:
            if (
                normalize_constraint_value(observation.constraint.value)
                != observation.constraint.normalized_value
            ):
                raise ValueError("constraint raw value does not match normalized value")
            cell_key = (observation.paper_id, _constraint_key(observation.constraint))
            if cell_key not in expected_keys:
                raise ValueError("observation is outside the expected coverage matrix")
            if cell_key in observed:
                raise ValueError("duplicate coverage matrix cell")
            observed[cell_key] = observation.matched
        if set(observed) != expected_keys:
            raise ValueError("coverage matrix is missing one or more cells")

        coverage: list[ConstraintCoverage] = []
        for constraint in constraints:
            constraint_key = _constraint_key(constraint)
            matched_ids = sorted(
                paper_id
                for paper_id in unique_candidate_ids
                if observed[(paper_id, constraint_key)]
            )
            hit_count = len(matched_ids)
            status = (
                "covered"
                if hit_count >= self.covered_min_hits
                else "low_coverage"
                if hit_count > 0
                else "uncovered"
            )
            coverage.append(
                ConstraintCoverage(
                    constraint=constraint,
                    matched_candidate_ids=matched_ids,
                    hit_count=hit_count,
                    status=status,
                )
            )

        covered_count = sum(item.status == "covered" for item in coverage)
        low_coverage_count = sum(item.status == "low_coverage" for item in coverage)
        uncovered_count = sum(item.status == "uncovered" for item in coverage)
        return CoverageReport(
            constraints=coverage,
            covered_count=covered_count,
            low_coverage_count=low_coverage_count,
            uncovered_count=uncovered_count,
        )
