from typing import ClassVar

from pydantic import Field

from paper_search.domain.models import DomainModel

from .models import CoverageReport, EvolutionStrategy, MarginalGain, StopDecision, StopReason


class _PolicyControls(DomainModel):
    strategy: EvolutionStrategy
    max_rounds: int = Field(strict=True, gt=0)
    marginal_gain_threshold: float = Field(strict=True, ge=0, allow_inf_nan=False)


class _RunControls(_PolicyControls):
    max_subqueries: int = Field(strict=True, gt=0)


class _StopInputs(_PolicyControls):
    completed_rounds: int = Field(strict=True, ge=0)
    coverage: CoverageReport | None
    gain: MarginalGain | None
    budget_available: bool = Field(strict=True)
    failed_stage: str | None


class _StopPolicy:
    precedence: ClassVar[tuple[tuple[str, StopReason], ...]] = (
        ("round_failed", "round_failed"),
        ("coverage_complete", "coverage_complete"),
        ("max_rounds_reached", "max_rounds_reached"),
        ("marginal_gain_below_threshold", "marginal_gain_below_threshold"),
        ("budget_insufficient", "budget_insufficient"),
    )


def validate_run_controls(
    *,
    strategy: EvolutionStrategy,
    max_rounds: int,
    max_subqueries: int,
    marginal_gain_threshold: float,
) -> _RunControls:
    return _RunControls.model_validate(
        {
            "strategy": strategy,
            "max_rounds": max_rounds,
            "max_subqueries": max_subqueries,
            "marginal_gain_threshold": marginal_gain_threshold,
        }
    )


def decide_stop(
    *,
    strategy: EvolutionStrategy,
    completed_rounds: int,
    coverage: CoverageReport | None,
    gain: MarginalGain | None,
    budget_available: bool,
    max_rounds: int,
    marginal_gain_threshold: float,
    failed_stage: str | None = None,
) -> StopDecision:
    inputs = _StopInputs.model_validate(
        {
            "strategy": strategy,
            "completed_rounds": completed_rounds,
            "coverage": coverage,
            "gain": gain,
            "budget_available": budget_available,
            "max_rounds": max_rounds,
            "marginal_gain_threshold": marginal_gain_threshold,
            "failed_stage": failed_stage,
        }
    )

    strategy = inputs.strategy
    completed_rounds = inputs.completed_rounds
    coverage = inputs.coverage
    gain = inputs.gain
    budget_available = inputs.budget_available
    max_rounds = inputs.max_rounds
    marginal_gain_threshold = inputs.marginal_gain_threshold
    failed_stage = inputs.failed_stage

    round_failed = failed_stage is not None
    coverage_complete = (
        strategy == "adaptive" and coverage is not None and coverage.is_complete
    )
    if strategy == "fixed_one_round":
        round_limit = completed_rounds >= 1
    elif strategy == "fixed_two_round":
        round_limit = completed_rounds >= 2
    else:
        round_limit = completed_rounds >= max_rounds
    low_gain = (
        strategy == "adaptive"
        and gain is not None
        and gain.score < marginal_gain_threshold
    )
    budget_insufficient = not budget_available
    checks = {
        "round_failed": round_failed,
        "coverage_complete": coverage_complete,
        "max_rounds_reached": round_limit,
        "marginal_gain_below_threshold": low_gain,
        "budget_insufficient": budget_insufficient,
    }
    reason: StopReason = "continue_evolution"
    for check_name, reason_code in _StopPolicy.precedence:
        if checks[check_name]:
            reason = reason_code
            break
    return StopDecision(
        should_continue=reason == "continue_evolution",
        reason_code=reason,
        strategy=strategy,
        completed_rounds=completed_rounds,
        max_rounds=max_rounds,
        marginal_gain_threshold=marginal_gain_threshold,
        checks=checks,
        failed_stage=failed_stage,
    )
