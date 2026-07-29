from __future__ import annotations

import pytest

from paper_search.domain.models import QuerySpec
from paper_search.evolution import (
    CandidateConstraintObservation,
    ConstraintRef,
    CoverageAnalyzer,
    extract_strong_constraints,
)


def test_extracts_typed_constraints_in_field_order_and_deduplicates_per_kind() -> None:
    spec = QuerySpec(
        original_query="q",
        research_goal="g",
        must_have=[" Graph   RAG ", "graph rag"],
        topics=["Graph RAG"],
        year_from=2020,
        exclusions=["survey"],
    )
    refs = extract_strong_constraints(spec)
    assert [(item.kind, item.value, item.normalized_value) for item in refs] == [
        ("must_have", "Graph   RAG", "graph rag"),
        ("topics", "Graph RAG", "graph rag"),
    ]


def test_classifies_covered_low_and_uncovered_constraints() -> None:
    spec = QuerySpec(
        original_query="q",
        research_goal="g",
        methods=["m1", "m2", "m3"],
    )
    constraints = extract_strong_constraints(spec)
    observations = [
        CandidateConstraintObservation(paper_id="p1", constraint=constraints[0], matched=True),
        CandidateConstraintObservation(paper_id="p1", constraint=constraints[1], matched=True),
        CandidateConstraintObservation(paper_id="p1", constraint=constraints[2], matched=False),
        CandidateConstraintObservation(paper_id="p2", constraint=constraints[0], matched=True),
        CandidateConstraintObservation(paper_id="p2", constraint=constraints[1], matched=False),
        CandidateConstraintObservation(paper_id="p2", constraint=constraints[2], matched=False),
    ]
    report = CoverageAnalyzer(covered_min_hits=2).analyze(
        spec, ["p2", "p1"], observations
    )
    assert [item.status for item in report.constraints] == [
        "covered",
        "low_coverage",
        "uncovered",
    ]
    assert report.covered_count == 1
    assert report.low_coverage_count == 1
    assert report.uncovered_count == 1


def test_no_positive_constraints_yields_complete_empty_report() -> None:
    spec = QuerySpec(original_query="q", research_goal="g")

    report = CoverageAnalyzer(covered_min_hits=1).analyze(spec, ["p2", "p1"], [])

    assert report.constraints == []
    assert report.covered_count == 0
    assert report.low_coverage_count == 0
    assert report.uncovered_count == 0
    assert report.is_complete is True


def test_candidate_and_matched_ids_are_sorted_and_deduplicated() -> None:
    spec = QuerySpec(original_query="q", research_goal="g", topics=["Topic"])
    constraint = extract_strong_constraints(spec)[0]
    observations = [
        CandidateConstraintObservation(paper_id="p2", constraint=constraint, matched=True),
        CandidateConstraintObservation(paper_id="p1", constraint=constraint, matched=True),
    ]

    report = CoverageAnalyzer(covered_min_hits=1).analyze(
        spec, ["p2", "p1", "p2"], observations
    )

    assert report.constraints[0].matched_candidate_ids == ["p1", "p2"]
    assert report.constraints[0].hit_count == 2


@pytest.mark.parametrize("threshold", [True, False, 1.0, 0, -1])
def test_covered_min_hits_requires_a_positive_integer(threshold: object) -> None:
    with pytest.raises(ValueError):
        CoverageAnalyzer(threshold)  # type: ignore[arg-type]


def _matrix_fixture() -> tuple[QuerySpec, list[str], list[CandidateConstraintObservation]]:
    spec = QuerySpec(original_query="q", research_goal="g", topics=["topic"])
    constraint = extract_strong_constraints(spec)[0]
    candidate_ids = ["p1"]
    observations = [
        CandidateConstraintObservation(paper_id="p1", constraint=constraint, matched=True)
    ]
    return spec, candidate_ids, observations


@pytest.mark.parametrize("case", ["missing", "unknown_candidate", "unknown_constraint", "duplicate"])
def test_analyze_rejects_invalid_observation_matrix(case: str) -> None:
    spec, candidate_ids, observations = _matrix_fixture()
    constraint = extract_strong_constraints(spec)[0]
    if case == "missing":
        observations = []
    elif case == "unknown_candidate":
        observations = [
            CandidateConstraintObservation(paper_id="p2", constraint=constraint, matched=True)
        ]
    elif case == "unknown_constraint":
        unknown = ConstraintRef(kind="topics", value="other", normalized_value="other")
        observations = [
            CandidateConstraintObservation(paper_id="p1", constraint=unknown, matched=True)
        ]
    else:
        observations = observations * 2

    with pytest.raises(ValueError):
        CoverageAnalyzer(covered_min_hits=1).analyze(spec, candidate_ids, observations)
