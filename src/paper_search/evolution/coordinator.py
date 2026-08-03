from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from paper_search.domain.models import DomainModel, Paper, QuerySpec, UsageEstimate
from paper_search.errors import ProtectedExecutionError

from .coverage import CoverageAnalyzer, normalize_constraint_value
from .costing import RoundCostEstimator
from .generation import NextRoundGenerator, NoTargetedQueriesError
from .models import (
    CandidateConstraintObservation,
    CoverageReport,
    EvolutionResult,
    EvolutionStrategy,
    MarginalGain,
    RoundExecution,
    RoundPlan,
    StopDecision,
)
from .stopping import decide_stop, validate_run_controls

ModelT = TypeVar("ModelT", bound=DomainModel)


class RoundExecutor(Protocol):
    async def execute(self, spec: QuerySpec, plan: RoundPlan) -> RoundExecution: ...


class GainEvaluator(Protocol):
    def evaluate(
        self,
        previous_ids: frozenset[str],
        current_ids: frozenset[str],
        execution: RoundExecution,
    ) -> MarginalGain: ...


class BudgetPreflight(Protocol):
    def can_reserve(self, estimate: UsageEstimate) -> bool: ...


def _snapshot(model: ModelT) -> ModelT:
    return model.model_copy(deep=True)


def _snapshot_sequence(models: Sequence[ModelT]) -> list[ModelT]:
    return [_snapshot(model) for model in models]


def _merge_candidates(
    existing: Sequence[Paper],
    incoming: Sequence[Paper],
) -> list[Paper]:
    result = list(existing)
    seen = {paper.canonical_id for paper in result}
    for paper in incoming:
        if paper.canonical_id not in seen:
            seen.add(paper.canonical_id)
            result.append(paper)
    return result


def _merge_observations(
    existing: Sequence[CandidateConstraintObservation],
    incoming: Sequence[CandidateConstraintObservation],
) -> list[CandidateConstraintObservation]:
    result = list(existing)
    seen = {
        (item.paper_id, item.constraint.kind, item.constraint.normalized_value)
        for item in result
    }
    for item in incoming:
        key = (
            item.paper_id,
            item.constraint.kind,
            item.constraint.normalized_value,
        )
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _validate_incoming_observations(
    observations: Sequence[CandidateConstraintObservation],
) -> None:
    observed_cells: set[tuple[str, str, str]] = set()
    for observation in observations:
        if (
            normalize_constraint_value(observation.constraint.value)
            != observation.constraint.normalized_value
        ):
            raise ValueError("constraint raw value does not match normalized value")
        cell = (
            observation.paper_id,
            observation.constraint.kind,
            observation.constraint.normalized_value,
        )
        if cell in observed_cells:
            raise ValueError("duplicate incoming coverage matrix cell")
        observed_cells.add(cell)


def _finish(
    *,
    strategy: EvolutionStrategy,
    rounds: list[RoundExecution],
    candidates: list[Paper],
    decisions: list[StopDecision],
    warnings: list[str],
    failed_round: int | None = None,
) -> EvolutionResult:
    return EvolutionResult(
        strategy=strategy,
        rounds=rounds,
        candidates=candidates,
        decisions=decisions,
        stop_reason=decisions[-1].reason_code,
        warnings=warnings,
        failed_round=failed_round,
    )


def _failure(
    *,
    stage: str,
    failed_round: int,
    strategy: EvolutionStrategy,
    max_rounds: int,
    marginal_gain_threshold: float,
    rounds: list[RoundExecution],
    candidates: list[Paper],
    decisions: list[StopDecision],
    coverage: CoverageReport | None,
    gain: MarginalGain | None,
) -> EvolutionResult:
    decision = decide_stop(
        strategy=strategy,
        completed_rounds=len(rounds),
        coverage=coverage,
        gain=gain,
        budget_available=True,
        max_rounds=max_rounds,
        marginal_gain_threshold=marginal_gain_threshold,
        failed_stage=stage,
    )
    return _finish(
        strategy=strategy,
        rounds=rounds,
        candidates=candidates,
        decisions=[*decisions, decision],
        warnings=[f"{stage}: dependency failure"],
        failed_round=failed_round,
    )


def _fixed_two_round_fallback(
    initial_plan: RoundPlan,
    round_number: int,
) -> RoundPlan:
    return RoundPlan(
        round_number=round_number,
        subqueries=_snapshot_sequence(initial_plan.subqueries),
    )


class EvolutionCoordinator:
    def __init__(
        self,
        *,
        executor: RoundExecutor,
        coverage_analyzer: CoverageAnalyzer,
        generator: NextRoundGenerator,
        estimator: RoundCostEstimator,
        gain_evaluator: GainEvaluator,
        budget: BudgetPreflight,
    ) -> None:
        self._executor = executor
        self._coverage_analyzer = coverage_analyzer
        self._generator = generator
        self._estimator = estimator
        self._gain_evaluator = gain_evaluator
        self._budget = budget

    async def run(
        self,
        *,
        spec: QuerySpec,
        initial_plan: RoundPlan,
        strategy: EvolutionStrategy,
        max_rounds: int,
        max_subqueries: int,
        marginal_gain_threshold: float,
    ) -> EvolutionResult:
        controls = validate_run_controls(
            strategy=strategy,
            max_rounds=max_rounds,
            max_subqueries=max_subqueries,
            marginal_gain_threshold=marginal_gain_threshold,
        )
        strategy = controls.strategy
        max_rounds = controls.max_rounds
        max_subqueries = controls.max_subqueries
        marginal_gain_threshold = controls.marginal_gain_threshold
        rounds: list[RoundExecution] = []
        candidates: list[Paper] = []
        observations: list[CandidateConstraintObservation] = []
        decisions: list[StopDecision] = []
        private_spec = _snapshot(spec)
        plan = _snapshot(initial_plan)
        plans = [plan]
        coverage: CoverageReport | None = None
        gain: MarginalGain | None = None

        try:
            estimate = _snapshot(
                self._estimator.estimate(_snapshot(plan), len(rounds))
            )
        except ProtectedExecutionError:
            raise
        except Exception:
            return _failure(
                stage="estimate",
                failed_round=plan.round_number,
                strategy=strategy,
                max_rounds=max_rounds,
                marginal_gain_threshold=marginal_gain_threshold,
                rounds=rounds,
                candidates=candidates,
                decisions=decisions,
                coverage=coverage,
                gain=gain,
            )
        try:
            budget_available = self._budget.can_reserve(_snapshot(estimate))
        except ProtectedExecutionError:
            raise
        except Exception:
            return _failure(
                stage="preflight",
                failed_round=plan.round_number,
                strategy=strategy,
                max_rounds=max_rounds,
                marginal_gain_threshold=marginal_gain_threshold,
                rounds=rounds,
                candidates=candidates,
                decisions=decisions,
                coverage=coverage,
                gain=gain,
            )
        if not budget_available:
            decision = decide_stop(
                strategy=strategy,
                completed_rounds=0,
                coverage=None,
                gain=None,
                budget_available=False,
                max_rounds=max_rounds,
                marginal_gain_threshold=marginal_gain_threshold,
            )
            return _finish(
                strategy=strategy,
                rounds=rounds,
                candidates=candidates,
                decisions=[decision],
                warnings=[],
            )

        while True:
            previous_ids = frozenset(paper.canonical_id for paper in candidates)
            try:
                execution = await self._executor.execute(
                    _snapshot(private_spec),
                    _snapshot(plan),
                )
                if execution.round_number != plan.round_number:
                    raise ValueError("execution round number does not match its plan")
                execution = _snapshot(execution)
                _validate_incoming_observations(execution.observations)
            except (ProtectedExecutionError, ValueError):
                raise
            except Exception:
                return _failure(
                    stage="execution",
                    failed_round=plan.round_number,
                    strategy=strategy,
                    max_rounds=max_rounds,
                    marginal_gain_threshold=marginal_gain_threshold,
                    rounds=rounds,
                    candidates=candidates,
                    decisions=decisions,
                    coverage=coverage,
                    gain=gain,
                )

            rounds.append(execution)
            candidates = _merge_candidates(candidates, execution.candidates)
            observations = _merge_observations(observations, execution.observations)
            current_ids = frozenset(paper.canonical_id for paper in candidates)

            try:
                coverage = _snapshot(
                    self._coverage_analyzer.analyze(
                        _snapshot(private_spec),
                        [paper.canonical_id for paper in candidates],
                        _snapshot_sequence(observations),
                    )
                )
            except Exception:
                return _failure(
                    stage="coverage",
                    failed_round=plan.round_number,
                    strategy=strategy,
                    max_rounds=max_rounds,
                    marginal_gain_threshold=marginal_gain_threshold,
                    rounds=rounds,
                    candidates=candidates,
                    decisions=decisions,
                    coverage=None,
                    gain=None,
                )

            try:
                gain = _snapshot(
                    self._gain_evaluator.evaluate(
                        previous_ids,
                        current_ids,
                        _snapshot(execution),
                    )
                )
            except Exception:
                return _failure(
                    stage="gain",
                    failed_round=plan.round_number,
                    strategy=strategy,
                    max_rounds=max_rounds,
                    marginal_gain_threshold=marginal_gain_threshold,
                    rounds=rounds,
                    candidates=candidates,
                    decisions=decisions,
                    coverage=coverage,
                    gain=None,
                )

            decision = decide_stop(
                strategy=strategy,
                completed_rounds=len(rounds),
                coverage=coverage,
                gain=gain,
                budget_available=True,
                max_rounds=max_rounds,
                marginal_gain_threshold=marginal_gain_threshold,
            )
            if not decision.should_continue:
                return _finish(
                    strategy=strategy,
                    rounds=rounds,
                    candidates=candidates,
                    decisions=[*decisions, decision],
                    warnings=[],
                )

            next_round_number = plan.round_number + 1
            try:
                generated_plan = await self._generator.generate(
                    spec=_snapshot(private_spec),
                    coverage=_snapshot(coverage),
                    prior_plans=_snapshot_sequence(plans),
                    round_number=next_round_number,
                    max_subqueries=max_subqueries,
                )
                if generated_plan.round_number != next_round_number:
                    raise ValueError("generated round number does not match request")
                next_plan = _snapshot(generated_plan)
            except NoTargetedQueriesError:
                if strategy == "fixed_two_round" and coverage.is_complete:
                    next_plan = _fixed_two_round_fallback(plans[0], next_round_number)
                else:
                    return _failure(
                        stage="generation",
                        failed_round=next_round_number,
                        strategy=strategy,
                        max_rounds=max_rounds,
                        marginal_gain_threshold=marginal_gain_threshold,
                        rounds=rounds,
                        candidates=candidates,
                        decisions=decisions,
                        coverage=coverage,
                        gain=gain,
                    )
            except Exception:
                return _failure(
                    stage="generation",
                    failed_round=next_round_number,
                    strategy=strategy,
                    max_rounds=max_rounds,
                    marginal_gain_threshold=marginal_gain_threshold,
                    rounds=rounds,
                    candidates=candidates,
                    decisions=decisions,
                    coverage=coverage,
                    gain=gain,
                )

            try:
                estimate = _snapshot(
                    self._estimator.estimate(_snapshot(next_plan), len(rounds))
                )
            except ProtectedExecutionError:
                raise
            except Exception:
                return _failure(
                    stage="estimate",
                    failed_round=next_plan.round_number,
                    strategy=strategy,
                    max_rounds=max_rounds,
                    marginal_gain_threshold=marginal_gain_threshold,
                    rounds=rounds,
                    candidates=candidates,
                    decisions=decisions,
                    coverage=coverage,
                    gain=gain,
                )
            try:
                budget_available = self._budget.can_reserve(_snapshot(estimate))
            except ProtectedExecutionError:
                raise
            except Exception:
                return _failure(
                    stage="preflight",
                    failed_round=next_plan.round_number,
                    strategy=strategy,
                    max_rounds=max_rounds,
                    marginal_gain_threshold=marginal_gain_threshold,
                    rounds=rounds,
                    candidates=candidates,
                    decisions=decisions,
                    coverage=coverage,
                    gain=gain,
                )

            decision = decide_stop(
                strategy=strategy,
                completed_rounds=len(rounds),
                coverage=coverage,
                gain=gain,
                budget_available=budget_available,
                max_rounds=max_rounds,
                marginal_gain_threshold=marginal_gain_threshold,
            )
            decisions.append(decision)
            if not decision.should_continue:
                return _finish(
                    strategy=strategy,
                    rounds=rounds,
                    candidates=candidates,
                    decisions=decisions,
                    warnings=[],
                )

            plans.append(next_plan)
            plan = next_plan
