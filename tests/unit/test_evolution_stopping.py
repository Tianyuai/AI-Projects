from __future__ import annotations

import math

import pytest

from paper_search.evolution import (
    ConstraintCoverage,
    ConstraintRef,
    CoverageReport,
    MarginalGain,
    StopDecision,
    decide_stop,
)


def complete_coverage() -> CoverageReport:
    return CoverageReport(constraints=[], covered_count=0, low_coverage_count=0, uncovered_count=0)


def incomplete_coverage() -> CoverageReport:
    constraint = ConstraintRef(kind="topics", value="topic", normalized_value="topic")
    return CoverageReport(
        constraints=[
            ConstraintCoverage(
                constraint=constraint,
                matched_candidate_ids=[],
                hit_count=0,
                status="uncovered",
            )
        ],
        covered_count=0,
        low_coverage_count=0,
        uncovered_count=1,
    )


def gain(score: float = 1.0) -> MarginalGain:
    return MarginalGain(
        new_candidate_count=1,
        new_high_relevance_count=1,
        score=score,
    )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"failed_stage": "ranking"}, "round_failed"),
        ({"coverage": complete_coverage()}, "coverage_complete"),
        ({"completed_rounds": 1, "strategy": "fixed_one_round"}, "max_rounds_reached"),
        ({"gain": gain(0.1), "marginal_gain_threshold": 0.5}, "marginal_gain_below_threshold"),
        ({"budget_available": False}, "budget_insufficient"),
        ({}, "continue_evolution"),
    ],
)
def test_decide_stop_returns_each_reason_code(kwargs: dict[str, object], reason: str) -> None:
    defaults: dict[str, object] = {
        "strategy": "adaptive",
        "completed_rounds": 0,
        "coverage": incomplete_coverage(),
        "gain": gain(),
        "budget_available": True,
        "max_rounds": 2,
        "marginal_gain_threshold": 0.5,
        "failed_stage": None,
    }
    defaults.update(kwargs)

    decision = decide_stop(**defaults)  # type: ignore[arg-type]

    assert decision.reason_code == reason
    assert decision.should_continue is (reason == "continue_evolution")
    assert set(decision.checks) == {
        "round_failed",
        "coverage_complete",
        "max_rounds_reached",
        "marginal_gain_below_threshold",
        "budget_insufficient",
    }


def test_adaptive_reason_precedence_is_deterministic() -> None:
    decision = decide_stop(
        strategy="adaptive",
        completed_rounds=2,
        coverage=complete_coverage(),
        gain=MarginalGain(
            new_candidate_count=0,
            new_high_relevance_count=0,
            score=0.0,
        ),
        budget_available=False,
        max_rounds=2,
        marginal_gain_threshold=0.5,
        failed_stage=None,
    )

    assert decision.reason_code == "coverage_complete"
    assert decision.should_continue is False
    assert decision.checks["coverage_complete"] is True
    assert decision.checks["max_rounds_reached"] is True
    assert decision.checks["budget_insufficient"] is True


@pytest.mark.parametrize("strategy", ["fixed_one_round", "fixed_two_round"])
def test_fixed_strategies_ignore_coverage_and_gain(strategy: str) -> None:
    decision = decide_stop(
        strategy=strategy,  # type: ignore[arg-type]
        completed_rounds=0,
        coverage=complete_coverage(),
        gain=gain(0.0),
        budget_available=True,
        max_rounds=1,
        marginal_gain_threshold=0.5,
        failed_stage=None,
    )

    assert decision.reason_code == "continue_evolution"
    assert decision.checks["coverage_complete"] is False
    assert decision.checks["marginal_gain_below_threshold"] is False


def test_failure_takes_precedence_over_every_other_stop_reason() -> None:
    decision = decide_stop(
        strategy="adaptive",
        completed_rounds=2,
        coverage=complete_coverage(),
        gain=gain(0.0),
        budget_available=False,
        max_rounds=2,
        marginal_gain_threshold=0.5,
        failed_stage="retrieval",
    )

    assert decision.reason_code == "round_failed"


@pytest.mark.parametrize("max_rounds", [0, -1])
def test_adaptive_requires_positive_max_rounds(max_rounds: int) -> None:
    with pytest.raises(ValueError):
        decide_stop(
            strategy="adaptive",
            completed_rounds=0,
            coverage=None,
            gain=None,
            budget_available=True,
            max_rounds=max_rounds,
            marginal_gain_threshold=0.5,
            failed_stage=None,
        )


@pytest.mark.parametrize("threshold", [-1.0, math.inf, math.nan])
def test_marginal_gain_threshold_must_be_finite_and_nonnegative(threshold: float) -> None:
    with pytest.raises(ValueError):
        decide_stop(
            strategy="adaptive",
            completed_rounds=0,
            coverage=None,
            gain=None,
            budget_available=True,
            max_rounds=2,
            marginal_gain_threshold=threshold,
            failed_stage=None,
        )


def test_stop_decision_is_a_domain_model() -> None:
    assert issubclass(StopDecision, object)
