from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from paper_search.control.budget import BudgetExceededError, ReservationError
from paper_search.domain.models import Paper, QuerySpec, UsageActual, UsageEstimate
from paper_search.evolution import (
    CandidateConstraintObservation,
    ConstraintCoverage,
    ConstraintRef,
    CoverageReport,
    EvolutionCoordinator,
    MarginalGain,
    RoundExecution,
    RoundPlan,
)


def query_spec() -> QuerySpec:
    return QuerySpec(
        original_query="query",
        research_goal="Find papers",
        topics=["topic"],
    )


def round_plan(round_number: int) -> RoundPlan:
    return RoundPlan(
        round_number=round_number,
        subqueries=[
            {
                "query_id": f"q-{round_number}",
                "text": f"query {round_number}",
                "query_type": "decomposed",
                "target_constraints": ["topic"],
                "priority": 1,
                "provider_hint": "either",
            }
        ],
    )


def paper(canonical_id: str, title: str) -> Paper:
    return Paper(canonical_id=canonical_id, title=title)


def observation(paper_id: str, *, matched: bool) -> CandidateConstraintObservation:
    return CandidateConstraintObservation(
        paper_id=paper_id,
        constraint=ConstraintRef(
            kind="topics",
            value="topic",
            normalized_value="topic",
        ),
        matched=matched,
    )


def execution(
    round_number: int,
    *,
    candidates: list[Paper] | None = None,
    observations: list[CandidateConstraintObservation] | None = None,
) -> RoundExecution:
    round_candidates = candidates or [paper(f"p{round_number}", f"Paper {round_number}")]
    round_observations = observations or [
        observation(item.canonical_id, matched=False) for item in round_candidates
    ]
    return RoundExecution(
        round_number=round_number,
        candidates=round_candidates,
        observations=round_observations,
        usage=UsageActual(search_api_calls=1, elapsed_ms=10),
        trace=[{"round": round_number}],
    )


def incomplete_coverage() -> CoverageReport:
    constraint = ConstraintRef(
        kind="topics",
        value="topic",
        normalized_value="topic",
    )
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


def complete_coverage() -> CoverageReport:
    return CoverageReport(
        constraints=[],
        covered_count=0,
        low_coverage_count=0,
        uncovered_count=0,
    )


class FakeExecutor:
    def __init__(
        self,
        executions: Sequence[RoundExecution],
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self._executions = list(executions)
        self._fail_on_call = fail_on_call
        self.calls: list[tuple[QuerySpec, RoundPlan]] = []

    async def execute(self, spec: QuerySpec, plan: RoundPlan) -> RoundExecution:
        self.calls.append((spec, plan))
        if len(self.calls) == self._fail_on_call:
            raise RuntimeError("secret executor payload")
        return self._executions[len(self.calls) - 1]


class RejectingExecutor(FakeExecutor):
    def __init__(
        self,
        executions: Sequence[RoundExecution],
        *,
        rejection: Exception,
        reject_on_call: int,
    ) -> None:
        super().__init__(executions)
        self._rejection = rejection
        self._reject_on_call = reject_on_call

    async def execute(self, spec: QuerySpec, plan: RoundPlan) -> RoundExecution:
        self.calls.append((spec, plan))
        if len(self.calls) == self._reject_on_call:
            raise self._rejection
        return self._executions[len(self.calls) - 1]


class FakeCoverageAnalyzer:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self._fail_on_call = fail_on_call
        self.calls: list[
            tuple[QuerySpec, list[str], list[CandidateConstraintObservation]]
        ] = []

    def analyze(
        self,
        spec: QuerySpec,
        candidate_ids: Sequence[str],
        observations: Sequence[CandidateConstraintObservation],
    ) -> CoverageReport:
        self.calls.append((spec, list(candidate_ids), list(observations)))
        if len(self.calls) == self._fail_on_call:
            raise RuntimeError("secret coverage matrix")
        return incomplete_coverage()


class RetainingCoverageAnalyzer(FakeCoverageAnalyzer):
    def __init__(self, reports: Sequence[CoverageReport]) -> None:
        super().__init__()
        self._reports = list(reports)

    def analyze(
        self,
        spec: QuerySpec,
        candidate_ids: Sequence[str],
        observations: Sequence[CandidateConstraintObservation],
    ) -> CoverageReport:
        self.calls.append((spec, list(candidate_ids), list(observations)))
        return self._reports[len(self.calls) - 1]


class FakeGenerator:
    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
        round_number_offset: int = 0,
    ) -> None:
        self._fail_on_call = fail_on_call
        self._round_number_offset = round_number_offset
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        spec: QuerySpec,
        coverage: CoverageReport,
        prior_plans: Sequence[RoundPlan],
        round_number: int,
        max_subqueries: int,
    ) -> RoundPlan:
        self.calls.append(
            {
                "spec": spec,
                "coverage": coverage,
                "prior_plans": list(prior_plans),
                "round_number": round_number,
                "max_subqueries": max_subqueries,
            }
        )
        if len(self.calls) == self._fail_on_call:
            raise RuntimeError("secret generation prompt")
        return round_plan(round_number + self._round_number_offset)


class FakeEstimator:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self._fail_on_call = fail_on_call
        self.calls: list[tuple[RoundPlan, int]] = []

    def estimate(self, plan: RoundPlan, completed_round_count: int) -> UsageEstimate:
        self.calls.append((plan, completed_round_count))
        if len(self.calls) == self._fail_on_call:
            raise RuntimeError("secret estimate details")
        return UsageEstimate(search_api_calls=1, elapsed_ms=10)


class FakeGainEvaluator:
    def __init__(self, *, fail_on_call: int | None = None, score: float = 1.0) -> None:
        self._fail_on_call = fail_on_call
        self._score = score
        self.calls: list[
            tuple[frozenset[str], frozenset[str], RoundExecution]
        ] = []

    def evaluate(
        self,
        previous_ids: frozenset[str],
        current_ids: frozenset[str],
        execution: RoundExecution,
    ) -> MarginalGain:
        self.calls.append((previous_ids, current_ids, execution))
        if len(self.calls) == self._fail_on_call:
            raise RuntimeError("secret gain labels")
        new_count = len(current_ids - previous_ids)
        return MarginalGain(
            new_candidate_count=new_count,
            new_high_relevance_count=new_count,
            score=self._score,
        )


class RetainingGainEvaluator(FakeGainEvaluator):
    def __init__(self, retained: MarginalGain) -> None:
        super().__init__()
        self._retained = retained

    def evaluate(
        self,
        previous_ids: frozenset[str],
        current_ids: frozenset[str],
        execution: RoundExecution,
    ) -> MarginalGain:
        self.calls.append((previous_ids, current_ids, execution))
        return self._retained


class FakeBudget:
    def __init__(
        self,
        availability: Sequence[bool] = (True, True, True),
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self._availability = list(availability)
        self._fail_on_call = fail_on_call
        self.calls: list[UsageEstimate] = []

    def can_reserve(self, estimate: UsageEstimate) -> bool:
        self.calls.append(estimate)
        if len(self.calls) == self._fail_on_call:
            raise RuntimeError("secret budget state")
        return self._availability[len(self.calls) - 1]


class MutatingEstimator(FakeEstimator):
    def estimate(self, plan: RoundPlan, completed_round_count: int) -> UsageEstimate:
        del completed_round_count
        plan.subqueries[0].target_constraints.append("corrupted")
        plan.subqueries.clear()
        raise RuntimeError("secret estimate mutation")


class MutatingExecutor(FakeExecutor):
    async def execute(self, spec: QuerySpec, plan: RoundPlan) -> RoundExecution:
        spec.topics.append("corrupted")
        plan.subqueries[0].target_constraints.append("corrupted")
        plan.subqueries.clear()
        raise RuntimeError("secret execution mutation")


class RetainingMutatingExecutor(FakeExecutor):
    def __init__(self, first_execution: RoundExecution) -> None:
        super().__init__([first_execution])
        self.retained = first_execution
        self.pre_second_call_snapshot: dict[str, Any] | None = None

    async def execute(self, spec: QuerySpec, plan: RoundPlan) -> RoundExecution:
        self.calls.append((spec, plan))
        if len(self.calls) == 1:
            return self.retained
        self.pre_second_call_snapshot = self.retained.model_dump(mode="json")
        self.retained.candidates[0].authors.append("Corrupt Author")
        self.retained.candidates.append(
            Paper(canonical_id="corrupted", title="Corrupted Paper")
        )
        self.retained.observations.clear()
        self.retained.trace[0]["round"] = "corrupted"
        raise RuntimeError("secret retained execution mutation")


class MutatingCoverageAnalyzer(FakeCoverageAnalyzer):
    def analyze(
        self,
        spec: QuerySpec,
        candidate_ids: Sequence[str],
        observations: Sequence[CandidateConstraintObservation],
    ) -> CoverageReport:
        del candidate_ids
        spec.topics.append("corrupted")
        assert isinstance(observations, list)
        observations.clear()
        raise RuntimeError("secret coverage mutation")


class MutatingGainEvaluator(FakeGainEvaluator):
    def evaluate(
        self,
        previous_ids: frozenset[str],
        current_ids: frozenset[str],
        execution: RoundExecution,
    ) -> MarginalGain:
        del previous_ids, current_ids
        execution.candidates[0].authors.append("Corrupt Author")
        execution.observations.clear()
        execution.trace[0]["round"] = "corrupted"
        raise RuntimeError("secret gain mutation")


class MutatingGenerator(FakeGenerator):
    async def generate(
        self,
        *,
        spec: QuerySpec,
        coverage: CoverageReport,
        prior_plans: Sequence[RoundPlan],
        round_number: int,
        max_subqueries: int,
    ) -> RoundPlan:
        del round_number, max_subqueries
        spec.topics.append("corrupted")
        coverage.constraints[0].matched_candidate_ids.append("corrupted")
        assert isinstance(prior_plans, list)
        prior_plans[0].subqueries[0].target_constraints.append("corrupted")
        prior_plans.clear()
        raise RuntimeError("secret generation mutation")


class MutatingRetainedCoverageGenerator(FakeGenerator):
    def __init__(self, retained_coverage: CoverageReport) -> None:
        super().__init__()
        self._retained_coverage = retained_coverage

    async def generate(
        self,
        *,
        spec: QuerySpec,
        coverage: CoverageReport,
        prior_plans: Sequence[RoundPlan],
        round_number: int,
        max_subqueries: int,
    ) -> RoundPlan:
        generated = await super().generate(
            spec=spec,
            coverage=coverage,
            prior_plans=prior_plans,
            round_number=round_number,
            max_subqueries=max_subqueries,
        )
        self._retained_coverage.__dict__["uncovered_count"] = 0
        return generated


class MutatingRetainedGainGenerator(FakeGenerator):
    def __init__(self, retained_gain: MarginalGain) -> None:
        super().__init__()
        self._retained_gain = retained_gain

    async def generate(
        self,
        *,
        spec: QuerySpec,
        coverage: CoverageReport,
        prior_plans: Sequence[RoundPlan],
        round_number: int,
        max_subqueries: int,
    ) -> RoundPlan:
        generated = await super().generate(
            spec=spec,
            coverage=coverage,
            prior_plans=prior_plans,
            round_number=round_number,
            max_subqueries=max_subqueries,
        )
        self._retained_gain.__dict__["score"] = 0.0
        return generated


class SharedEstimateEstimator(FakeEstimator):
    def __init__(self, shared_estimate: UsageEstimate) -> None:
        super().__init__()
        self.shared_estimate = shared_estimate

    def estimate(self, plan: RoundPlan, completed_round_count: int) -> UsageEstimate:
        self.calls.append((plan, completed_round_count))
        return self.shared_estimate


class MutatingBudget(FakeBudget):
    def can_reserve(self, estimate: UsageEstimate) -> bool:
        estimate.__dict__["search_api_calls"] = 999
        raise RuntimeError("secret budget mutation")


def coordinator(
    *,
    executor: FakeExecutor | None = None,
    coverage_analyzer: FakeCoverageAnalyzer | None = None,
    generator: FakeGenerator | None = None,
    estimator: FakeEstimator | None = None,
    gain_evaluator: FakeGainEvaluator | None = None,
    budget: FakeBudget | None = None,
) -> EvolutionCoordinator:
    return EvolutionCoordinator(
        executor=executor or FakeExecutor([execution(1), execution(2)]),
        coverage_analyzer=coverage_analyzer or FakeCoverageAnalyzer(),
        generator=generator or FakeGenerator(),
        estimator=estimator or FakeEstimator(),
        gain_evaluator=gain_evaluator or FakeGainEvaluator(),
        budget=budget or FakeBudget(),
    )


def run(
    instance: EvolutionCoordinator,
    *,
    strategy: str = "fixed_one_round",
    max_rounds: int = 2,
    max_subqueries: int = 2,
    marginal_gain_threshold: float = 0.5,
    spec: QuerySpec | None = None,
    initial_plan: RoundPlan | None = None,
) -> Any:
    run_spec = spec if spec is not None else query_spec()
    run_plan = initial_plan if initial_plan is not None else round_plan(1)
    return asyncio.run(
        instance.run(
            spec=run_spec,
            initial_plan=run_plan,
            strategy=strategy,
            max_rounds=max_rounds,
            max_subqueries=max_subqueries,
            marginal_gain_threshold=marginal_gain_threshold,
        )
    )


@pytest.mark.parametrize(
    "controls",
    [
        {"strategy": "not-a-strategy"},
        {"max_rounds": 0},
        {"max_subqueries": 0},
        {"marginal_gain_threshold": -0.1},
    ],
)
def test_invalid_run_controls_fail_before_any_dependency_call(
    controls: dict[str, object],
) -> None:
    executor = FakeExecutor([execution(1)])
    coverage_analyzer = FakeCoverageAnalyzer()
    generator = FakeGenerator()
    estimator = FakeEstimator()
    gain_evaluator = FakeGainEvaluator()
    budget = FakeBudget()
    instance = coordinator(
        executor=executor,
        coverage_analyzer=coverage_analyzer,
        generator=generator,
        estimator=estimator,
        gain_evaluator=gain_evaluator,
        budget=budget,
    )

    with pytest.raises(ValueError):
        run(instance, **controls)  # type: ignore[arg-type]

    assert executor.calls == []
    assert coverage_analyzer.calls == []
    assert generator.calls == []
    assert estimator.calls == []
    assert gain_evaluator.calls == []
    assert budget.calls == []


@pytest.mark.parametrize(
    ("stage", "instance"),
    [
        (
            "estimate",
            coordinator(estimator=FakeEstimator(fail_on_call=1)),
        ),
        (
            "preflight",
            coordinator(budget=FakeBudget(fail_on_call=1)),
        ),
        (
            "execution",
            coordinator(executor=FakeExecutor([execution(1)], fail_on_call=1)),
        ),
    ],
)
def test_precommit_dependency_failure_returns_empty_sanitized_result(
    stage: str,
    instance: EvolutionCoordinator,
) -> None:
    result = run(instance)

    assert result.rounds == []
    assert result.candidates == []
    assert result.stop_reason == "round_failed"
    assert result.failed_round == 1
    assert result.warnings == [f"{stage}: dependency failure"]
    assert result.decisions[-1].failed_stage == stage
    assert "secret" not in " ".join(result.warnings)


def test_preflight_rejection_stops_before_executor_runs() -> None:
    fake_executor = FakeExecutor([execution(1)])
    result = run(
        coordinator(
            executor=fake_executor,
            budget=FakeBudget(availability=[False]),
        )
    )

    assert result.rounds == []
    assert result.candidates == []
    assert result.stop_reason == "budget_insufficient"
    assert result.failed_round is None
    assert result.warnings == []
    assert fake_executor.calls == []


def test_execution_round_mismatch_does_not_commit_any_state() -> None:
    result = run(coordinator(executor=FakeExecutor([execution(2)])))

    assert result.rounds == []
    assert result.candidates == []
    assert result.stop_reason == "round_failed"
    assert result.failed_round == 1
    assert result.warnings == ["execution: dependency failure"]
    assert result.decisions[-1].failed_stage == "execution"


def test_later_executor_failure_preserves_committed_first_seen_state() -> None:
    first_paper = paper("p1", "First title")
    first_execution = execution(
        1,
        candidates=[first_paper],
        observations=[observation("p1", matched=False)],
    )
    execution_before = first_execution.model_dump(mode="json")
    result = run(
        coordinator(
            executor=FakeExecutor(
                [first_execution, execution(2)],
                fail_on_call=2,
            )
        ),
        strategy="fixed_two_round",
    )

    assert result.rounds[0].model_dump(mode="json") == execution_before
    assert result.candidates[0].model_dump(mode="json") == execution_before["candidates"][0]
    assert result.candidates[0] is result.rounds[0].candidates[0]
    assert result.candidates[0] is not first_paper
    assert result.stop_reason == "round_failed"
    assert result.failed_round == 2
    assert result.warnings == ["execution: dependency failure"]
    assert result.decisions[-1].failed_stage == "execution"


@pytest.mark.parametrize(
    ("rejection", "reason", "failed_round", "warnings", "failed_stage"),
    [
        (BudgetExceededError("reservation lost"), "budget_insufficient", None, [], None),
        (
            ReservationError("reservation invalid"),
            "round_failed",
            2,
            ["execution: dependency failure"],
            "execution",
        ),
    ],
)
def test_execution_rejection_preserves_prior_state_with_its_typed_outcome(
    rejection: Exception,
    reason: str,
    failed_round: int | None,
    warnings: list[str],
    failed_stage: str | None,
) -> None:
    first_execution = execution(1)
    first_before = first_execution.model_dump(mode="json")
    executor = RejectingExecutor(
        [first_execution, execution(2)],
        rejection=rejection,
        reject_on_call=2,
    )

    result = run(
        coordinator(executor=executor),
        strategy="fixed_two_round",
    )

    assert [item.model_dump(mode="json") for item in result.rounds] == [first_before]
    assert [item.model_dump(mode="json") for item in result.candidates] == first_before[
        "candidates"
    ]
    assert result.stop_reason == reason
    assert result.failed_round == failed_round
    assert result.warnings == warnings
    assert result.decisions[-1].failed_stage == failed_stage
    assert result.decisions[-1].checks["budget_insufficient"] is (
        reason == "budget_insufficient"
    )


def test_executor_cannot_mutate_retained_committed_output_on_later_failure() -> None:
    first_paper = Paper(
        canonical_id="p1",
        title="First title",
        authors=["Original Author"],
    )
    first_execution = execution(
        1,
        candidates=[first_paper],
        observations=[observation("p1", matched=False)],
    )
    executor = RetainingMutatingExecutor(first_execution)

    result = run(
        coordinator(executor=executor),
        strategy="fixed_two_round",
    )

    expected_round = executor.pre_second_call_snapshot
    assert expected_round is not None
    assert result.rounds[0].model_dump(mode="json") == expected_round
    assert [item.model_dump(mode="json") for item in result.candidates] == expected_round[
        "candidates"
    ]
    assert result.candidates[0] is result.rounds[0].candidates[0]
    assert result.rounds[0] is not first_execution
    assert result.candidates[0] is not first_paper
    assert result.stop_reason == "round_failed"
    assert result.failed_round == 2
    assert result.warnings == ["execution: dependency failure"]
    assert result.decisions[-1].failed_stage == "execution"


@pytest.mark.parametrize(
    ("stage", "updates"),
    [
        ("coverage", {"coverage_analyzer": FakeCoverageAnalyzer(fail_on_call=1)}),
        ("gain", {"gain_evaluator": FakeGainEvaluator(fail_on_call=1)}),
        ("generation", {"generator": FakeGenerator(fail_on_call=1)}),
        ("estimate", {"estimator": FakeEstimator(fail_on_call=2)}),
        ("preflight", {"budget": FakeBudget(fail_on_call=2)}),
    ],
)
def test_postcommit_dependency_failure_preserves_first_seen_state(
    stage: str,
    updates: dict[str, object],
) -> None:
    first_paper = paper("p1", "First title")
    first_execution = execution(
        1,
        candidates=[first_paper],
        observations=[observation("p1", matched=False)],
    )
    execution_before = first_execution.model_dump(mode="json")
    fake_executor = FakeExecutor([first_execution, execution(2)])
    instance = coordinator(executor=fake_executor, **updates)  # type: ignore[arg-type]

    result = run(instance, strategy="fixed_two_round")

    assert result.rounds[0].model_dump(mode="json") == execution_before
    assert result.candidates[0].model_dump(mode="json") == execution_before["candidates"][0]
    assert result.candidates[0] is result.rounds[0].candidates[0]
    assert result.candidates[0] is not first_paper
    assert result.stop_reason == "round_failed"
    assert result.failed_round == (2 if stage in {"generation", "estimate", "preflight"} else 1)
    assert result.warnings == [f"{stage}: dependency failure"]
    assert result.decisions[-1].failed_stage == stage
    assert "secret" not in " ".join(result.warnings)


def test_next_round_budget_rejection_preserves_committed_round() -> None:
    first_execution = execution(1)
    fake_executor = FakeExecutor([first_execution, execution(2)])

    result = run(
        coordinator(
            executor=fake_executor,
            budget=FakeBudget(availability=[True, False]),
        ),
        strategy="fixed_two_round",
    )

    assert result.rounds == [first_execution]
    assert result.candidates == first_execution.candidates
    assert result.stop_reason == "budget_insufficient"
    assert result.failed_round is None
    assert result.warnings == []
    assert len(fake_executor.calls) == 1
    assert [decision.reason_code for decision in result.decisions] == [
        "budget_insufficient"
    ]


def test_retained_coverage_return_cannot_change_decision_after_async_generation() -> None:
    retained_coverage = incomplete_coverage()
    result = run(
        coordinator(
            executor=FakeExecutor([execution(1), execution(2)]),
            coverage_analyzer=RetainingCoverageAnalyzer(
                [retained_coverage, complete_coverage()]
            ),
            generator=MutatingRetainedCoverageGenerator(retained_coverage),
        ),
        strategy="adaptive",
    )

    assert [item.round_number for item in result.rounds] == [1, 2]
    assert [item.reason_code for item in result.decisions] == [
        "continue_evolution",
        "coverage_complete",
    ]
    assert retained_coverage.is_complete is True


def test_retained_gain_return_cannot_change_decision_after_async_generation() -> None:
    retained_gain = MarginalGain(
        new_candidate_count=1,
        new_high_relevance_count=1,
        score=1.0,
    )
    result = run(
        coordinator(
            executor=FakeExecutor([execution(1), execution(2)]),
            coverage_analyzer=RetainingCoverageAnalyzer(
                [incomplete_coverage(), complete_coverage()]
            ),
            generator=MutatingRetainedGainGenerator(retained_gain),
            gain_evaluator=RetainingGainEvaluator(retained_gain),
        ),
        strategy="adaptive",
    )

    assert [item.round_number for item in result.rounds] == [1, 2]
    assert [item.reason_code for item in result.decisions] == [
        "continue_evolution",
        "coverage_complete",
    ]
    assert retained_gain.score == 0.0


@pytest.mark.parametrize("round_number_offset", [-1, 1])
def test_invalid_generated_round_sequence_fails_before_estimation_or_execution(
    round_number_offset: int,
) -> None:
    first_paper = paper("p1", "First title")
    first_execution = execution(
        1,
        candidates=[first_paper],
        observations=[observation("p1", matched=False)],
    )
    fake_executor = FakeExecutor([first_execution, execution(2)])
    estimator = FakeEstimator()

    result = run(
        coordinator(
            executor=fake_executor,
            generator=FakeGenerator(round_number_offset=round_number_offset),
            estimator=estimator,
        ),
        strategy="fixed_two_round",
    )

    assert result.rounds == [first_execution]
    assert result.candidates == [first_paper]
    assert result.stop_reason == "round_failed"
    assert result.failed_round == 2
    assert result.warnings == ["generation: dependency failure"]
    assert result.decisions[-1].failed_stage == "generation"
    assert len(estimator.calls) == 1
    assert len(fake_executor.calls) == 1


def test_estimator_mutation_then_failure_cannot_change_authoritative_plan() -> None:
    spec = query_spec()
    initial_plan = round_plan(1)
    spec_before = spec.model_dump(mode="json")
    plan_before = initial_plan.model_dump(mode="json")

    result = run(
        coordinator(estimator=MutatingEstimator()),
        spec=spec,
        initial_plan=initial_plan,
    )

    assert spec.model_dump(mode="json") == spec_before
    assert initial_plan.model_dump(mode="json") == plan_before
    assert result.rounds == []
    assert result.candidates == []
    assert result.warnings == ["estimate: dependency failure"]


def test_executor_mutation_then_failure_cannot_change_authoritative_inputs() -> None:
    spec = query_spec()
    initial_plan = round_plan(1)
    spec_before = spec.model_dump(mode="json")
    plan_before = initial_plan.model_dump(mode="json")

    result = run(
        coordinator(executor=MutatingExecutor([execution(1)])),
        spec=spec,
        initial_plan=initial_plan,
    )

    assert spec.model_dump(mode="json") == spec_before
    assert initial_plan.model_dump(mode="json") == plan_before
    assert result.rounds == []
    assert result.candidates == []
    assert result.warnings == ["execution: dependency failure"]


def test_coverage_mutation_then_failure_cannot_change_committed_state() -> None:
    spec = query_spec()
    first_execution = execution(1)
    spec_before = spec.model_dump(mode="json")
    execution_before = first_execution.model_dump(mode="json")

    result = run(
        coordinator(
            executor=FakeExecutor([first_execution]),
            coverage_analyzer=MutatingCoverageAnalyzer(),
        ),
        spec=spec,
    )

    assert spec.model_dump(mode="json") == spec_before
    assert result.rounds[0].model_dump(mode="json") == execution_before
    assert result.candidates[0].model_dump(mode="json") == execution_before["candidates"][0]
    assert result.rounds[0].observations == first_execution.observations
    assert result.warnings == ["coverage: dependency failure"]


def test_gain_mutation_then_failure_cannot_change_committed_state() -> None:
    first_paper = Paper(
        canonical_id="p1",
        title="First title",
        authors=["Original Author"],
    )
    first_execution = execution(
        1,
        candidates=[first_paper],
        observations=[observation("p1", matched=False)],
    )
    execution_before = first_execution.model_dump(mode="json")
    candidates_before = [item.model_dump(mode="json") for item in first_execution.candidates]

    result = run(
        coordinator(
            executor=FakeExecutor([first_execution]),
            gain_evaluator=MutatingGainEvaluator(),
        )
    )

    assert result.rounds[0].model_dump(mode="json") == execution_before
    assert [item.model_dump(mode="json") for item in result.candidates] == candidates_before
    assert result.rounds[0].observations == first_execution.observations
    assert result.warnings == ["gain: dependency failure"]


def test_generator_mutation_then_failure_cannot_change_prior_state() -> None:
    spec = query_spec()
    initial_plan = round_plan(1)
    first_execution = execution(1)
    spec_before = spec.model_dump(mode="json")
    plan_before = initial_plan.model_dump(mode="json")
    execution_before = first_execution.model_dump(mode="json")

    result = run(
        coordinator(
            executor=FakeExecutor([first_execution]),
            generator=MutatingGenerator(),
        ),
        strategy="fixed_two_round",
        spec=spec,
        initial_plan=initial_plan,
    )

    assert spec.model_dump(mode="json") == spec_before
    assert initial_plan.model_dump(mode="json") == plan_before
    assert result.rounds[0].model_dump(mode="json") == execution_before
    assert result.candidates[0].model_dump(mode="json") == execution_before["candidates"][0]
    assert result.warnings == ["generation: dependency failure"]


def test_budget_mutation_then_failure_cannot_change_estimator_output() -> None:
    shared_estimate = UsageEstimate(search_api_calls=1, elapsed_ms=10)
    estimate_before = shared_estimate.model_dump(mode="json")

    result = run(
        coordinator(
            estimator=SharedEstimateEstimator(shared_estimate),
            budget=MutatingBudget(),
        )
    )

    assert shared_estimate.model_dump(mode="json") == estimate_before
    assert result.rounds == []
    assert result.candidates == []
    assert result.warnings == ["preflight: dependency failure"]


def test_duplicate_candidates_and_observations_keep_first_seen_objects() -> None:
    first_paper = paper("duplicate", "First title")
    later_paper = paper("duplicate", "Later title")
    first_observation = observation("duplicate", matched=False)
    conflicting_observation = observation("duplicate", matched=True)
    observation_before = first_observation.model_dump(mode="json")
    analyzer = FakeCoverageAnalyzer()
    gain = FakeGainEvaluator()

    result = run(
        coordinator(
            executor=FakeExecutor(
                [
                    execution(
                        1,
                        candidates=[first_paper],
                        observations=[first_observation],
                    ),
                    execution(
                        2,
                        candidates=[later_paper],
                        observations=[conflicting_observation],
                    ),
                ]
            ),
            coverage_analyzer=analyzer,
            gain_evaluator=gain,
        ),
        strategy="fixed_two_round",
    )

    assert result.stop_reason == "max_rounds_reached"
    assert result.candidates == [first_paper]
    assert result.candidates[0] is result.rounds[0].candidates[0]
    assert result.candidates[0] is not first_paper
    assert analyzer.calls[-1][1] == ["duplicate"]
    assert analyzer.calls[-1][2] == [first_observation]
    assert analyzer.calls[-1][2][0].model_dump(mode="json") == observation_before
    assert gain.calls[-1][0] == frozenset({"duplicate"})
    assert gain.calls[-1][1] == frozenset({"duplicate"})


def test_duplicate_cells_in_an_incoming_execution_fail_before_commit() -> None:
    first_execution = execution(
        1,
        candidates=[paper("p1", "First paper")],
        observations=[observation("p1", matched=False)],
    )
    first_before = first_execution.model_dump(mode="json")
    second_execution = execution(
        2,
        candidates=[paper("p2", "Second paper")],
        observations=[
            observation("p2", matched=False),
            observation("p2", matched=True),
        ],
    )

    result = run(
        coordinator(executor=FakeExecutor([first_execution, second_execution])),
        strategy="fixed_two_round",
    )

    assert [item.model_dump(mode="json") for item in result.rounds] == [first_before]
    assert [item.model_dump(mode="json") for item in result.candidates] == first_before[
        "candidates"
    ]
    assert result.stop_reason == "round_failed"
    assert result.failed_round == 2
    assert result.warnings == ["execution: dependency failure"]
    assert result.decisions[-1].failed_stage == "execution"


def test_inconsistent_incoming_constraint_cannot_hide_behind_prior_normalized_cell() -> None:
    first_execution = execution(
        1,
        candidates=[paper("p1", "First paper")],
        observations=[observation("p1", matched=False)],
    )
    first_before = first_execution.model_dump(mode="json")
    forged_observation = CandidateConstraintObservation(
        paper_id="p1",
        constraint=ConstraintRef(
            kind="topics",
            value="unrelated",
            normalized_value="topic",
        ),
        matched=True,
    )
    second_execution = execution(
        2,
        candidates=[paper("p1", "Later paper")],
        observations=[forged_observation],
    )

    result = run(
        coordinator(executor=FakeExecutor([first_execution, second_execution])),
        strategy="fixed_two_round",
    )

    assert [item.model_dump(mode="json") for item in result.rounds] == [first_before]
    assert [item.model_dump(mode="json") for item in result.candidates] == first_before[
        "candidates"
    ]
    assert result.stop_reason == "round_failed"
    assert result.failed_round == 2
    assert result.warnings == ["execution: dependency failure"]
    assert result.decisions[-1].failed_stage == "execution"
