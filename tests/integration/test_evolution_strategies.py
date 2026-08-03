from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from paper_search.application.experiments import (
    ExperimentDefinition,
    ExperimentFlags,
    build_experiment_components,
)
from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import Paper, QuerySpec, SearchBudget, UsageActual
from paper_search.evolution import (
    CandidateConstraintObservation,
    CoverageAnalyzer,
    DeterministicRoundCostEstimator,
    EvolutionCoordinator,
    EvolutionStrategy,
    MarginalGain,
    RoundExecution,
    RoundPlan,
    RuleBasedNextRoundGenerator,
    extract_strong_constraints,
)


def query_spec() -> QuerySpec:
    return QuerySpec(
        original_query="graph retrieval",
        research_goal="Find graph retrieval papers",
        topics=["graph retrieval"],
    )


def round_plan(round_number: int) -> RoundPlan:
    return RoundPlan(
        round_number=round_number,
        subqueries=[
            {
                "query_id": f"round-{round_number}",
                "text": f"graph retrieval round {round_number}",
                "query_type": "decomposed",
                "target_constraints": ["graph retrieval"],
                "priority": 1,
                "provider_hint": "either",
            }
        ],
    )


class FakeExecutor:
    def __init__(self, *, coverage_complete_after: int | None = None) -> None:
        self.coverage_complete_after = coverage_complete_after
        self.plans: list[RoundPlan] = []

    async def execute(self, spec: QuerySpec, plan: RoundPlan) -> RoundExecution:
        self.plans.append(plan)
        constraint = extract_strong_constraints(spec)[0]
        matched = (
            self.coverage_complete_after is not None
            and plan.round_number >= self.coverage_complete_after
        )
        paper = Paper(
            canonical_id=f"paper-{plan.round_number}",
            title=f"Paper {plan.round_number}",
        )
        return RoundExecution(
            round_number=plan.round_number,
            candidates=[paper],
            observations=[
                CandidateConstraintObservation(
                    paper_id=paper.canonical_id,
                    constraint=constraint,
                    matched=matched,
                )
            ],
            usage=UsageActual(search_api_calls=1, elapsed_ms=10),
            trace=[{"round": plan.round_number}],
        )


class FakeGenerator:
    async def generate(
        self,
        *,
        spec: QuerySpec,
        coverage: object,
        prior_plans: Sequence[RoundPlan],
        round_number: int,
        max_subqueries: int,
    ) -> RoundPlan:
        del spec, coverage, prior_plans, max_subqueries
        return round_plan(round_number)


class FakeGainEvaluator:
    def evaluate(
        self,
        previous_ids: frozenset[str],
        current_ids: frozenset[str],
        execution: RoundExecution,
    ) -> MarginalGain:
        del execution
        new_count = len(current_ids - previous_ids)
        return MarginalGain(
            new_candidate_count=new_count,
            new_high_relevance_count=new_count,
            score=float(new_count),
        )


def build_fake_coordinator(
    *,
    coverage_complete_after: int | None = None,
    executor: FakeExecutor | None = None,
    generator: FakeGenerator | RuleBasedNextRoundGenerator | None = None,
) -> EvolutionCoordinator:
    budget = HardBudgetController(
        SearchBudget(
            max_search_api_calls=10,
            target_search_api_calls=10,
            max_llm_calls=1,
            target_llm_calls=0,
            max_iterations=4,
            max_subqueries=2,
            max_elapsed_seconds=10,
            soft_deadline_seconds=9,
            max_total_tokens=1,
            max_cost_cny=1.0,
        )
    )
    return EvolutionCoordinator(
        executor=executor or FakeExecutor(coverage_complete_after=coverage_complete_after),
        coverage_analyzer=CoverageAnalyzer(covered_min_hits=1),
        generator=generator or FakeGenerator(),
        estimator=DeterministicRoundCostEstimator(
            search_calls_per_subquery=1,
            llm_calls_per_round=0,
            input_tokens_per_subquery=0,
            output_tokens_per_subquery=0,
            cost_cny_per_subquery=0.0,
            elapsed_ms_per_subquery=10,
        ),
        gain_evaluator=FakeGainEvaluator(),
        budget=budget,
    )


@pytest.mark.parametrize(
    ("strategy", "expected_rounds"),
    [("fixed_one_round", 1), ("fixed_two_round", 2)],
)
def test_fixed_strategies_have_deterministic_round_counts(
    strategy: EvolutionStrategy,
    expected_rounds: int,
) -> None:
    coordinator = build_fake_coordinator()

    result = asyncio.run(
        coordinator.run(
            spec=query_spec(),
            initial_plan=round_plan(1),
            strategy=strategy,
            max_rounds=4,
            max_subqueries=2,
            marginal_gain_threshold=0.5,
        )
    )

    assert len(result.rounds) == expected_rounds
    assert result.stop_reason == "max_rounds_reached"
    assert len(result.decisions) == expected_rounds
    assert [decision.completed_rounds for decision in result.decisions] == list(
        range(1, expected_rounds + 1)
    )


def test_adaptive_stops_when_coverage_becomes_complete() -> None:
    coordinator = build_fake_coordinator(coverage_complete_after=2)

    result = asyncio.run(
        coordinator.run(
            spec=query_spec(),
            initial_plan=round_plan(1),
            strategy="adaptive",
            max_rounds=4,
            max_subqueries=2,
            marginal_gain_threshold=0.1,
        )
    )

    assert len(result.rounds) == 2
    assert result.stop_reason == "coverage_complete"
    assert [decision.reason_code for decision in result.decisions] == [
        "continue_evolution",
        "coverage_complete",
    ]


def test_fixed_two_round_reuses_initial_plan_when_complete_coverage_has_no_target() -> None:
    executor = FakeExecutor(coverage_complete_after=1)
    initial_plan = round_plan(1)
    coordinator = build_fake_coordinator(
        executor=executor,
        generator=RuleBasedNextRoundGenerator(),
    )

    result = asyncio.run(
        coordinator.run(
            spec=query_spec(),
            initial_plan=initial_plan,
            strategy="fixed_two_round",
            max_rounds=4,
            max_subqueries=2,
            marginal_gain_threshold=0.5,
        )
    )

    assert [round_execution.round_number for round_execution in result.rounds] == [1, 2]
    assert result.stop_reason == "max_rounds_reached"
    assert result.warnings == []
    assert executor.plans[1].round_number == 2
    assert executor.plans[1].subqueries == initial_plan.subqueries


@pytest.mark.parametrize(
    ("definition", "expected_rounds"),
    [
        (
            ExperimentDefinition(
                name="fixed-two-round",
                flags=ExperimentFlags(fixed_two_round=True),
                strategy="fixed-two-round",
            ),
            2,
        ),
        (
            ExperimentDefinition(
                name="adaptive-evolution",
                flags=ExperimentFlags(adaptive_evolution=True),
                strategy="adaptive-evolution",
            ),
            2,
        ),
    ],
)
def test_multi_round_experiments_wrap_the_same_single_round_executor(
    definition: ExperimentDefinition,
    expected_rounds: int,
) -> None:
    class NoOptionalDependencies:
        def build_embedding_ranker(self) -> object:
            raise AssertionError("not enabled")

        def build_citation_expander(self) -> object:
            raise AssertionError("not enabled")

        def build_constraint_reranker(self) -> object:
            raise AssertionError("not enabled")

    executor = FakeExecutor()
    coordinator = build_fake_coordinator(executor=executor)
    components = build_experiment_components(
        definition,
        dependencies=NoOptionalDependencies(),
    )

    result = asyncio.run(
        coordinator.run(
            spec=query_spec(),
            initial_plan=round_plan(1),
            strategy=components.evolution_strategy,
            max_rounds=expected_rounds,
            max_subqueries=2,
            marginal_gain_threshold=0.0,
        )
    )

    assert len(result.rounds) == expected_rounds
    assert executor.plans == [
        round_plan(index) for index in range(1, expected_rounds + 1)
    ]
