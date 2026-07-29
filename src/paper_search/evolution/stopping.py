from .models import EvolutionStrategy, MarginalGain, StopDecision, StopReason
from .models import CoverageReport


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
    round_failed = failed_stage is not None
    coverage_complete = (
        strategy == "adaptive" and coverage is not None and coverage.is_complete
    )
    round_limit = (
        completed_rounds >= (1 if strategy == "fixed_one_round" else 2 if strategy == "fixed_two_round" else max_rounds)
    )
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
    for check_name, reason_code in (
        ("round_failed", "round_failed"),
        ("coverage_complete", "coverage_complete"),
        ("max_rounds_reached", "max_rounds_reached"),
        ("marginal_gain_below_threshold", "marginal_gain_below_threshold"),
        ("budget_insufficient", "budget_insufficient"),
    ):
        if checks[check_name]:
            reason = reason_code  # type: ignore[assignment]
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
