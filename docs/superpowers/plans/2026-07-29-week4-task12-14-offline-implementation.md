# Week 4 Task 12/14 Offline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic, budget-aware adaptive-query-evolution infrastructure and non-metric Task 14 delivery documents without invoking external Providers, LLMs, or frozen evaluation datasets.

**Architecture:** Add an isolated `paper_search.evolution` package that coordinates injected single-round execution while reusing existing domain and budget contracts. Keep adaptive behavior disconnected from API/runtime composition, expose only an offline ablation mapping, and preserve prior successful candidates when a later stage fails.

**Tech Stack:** Python 3.11, Pydantic 2 frozen domain models, `typing.Protocol`, pytest, Ruff, mypy, Markdown.

## Global Constraints

- Work only in `D:\AI Projects\.worktrees\week4` on branch `codex/week4-task12-14-offline`.
- The implementation base is commit `33f0cf4dba11c31be998af67b0f5457372269bb9`; do not merge Week 3 identifier-map or R3 work into this task.
- Follow strict RED-to-GREEN TDD for every implementation task.
- Every Python/test command must use `D:\Dev\uv\uv.exe run --no-sync --no-env-file`.
- Never read `.env`, use Provider credentials, make network requests, call a real LLM, or run dev/validation/test dataset experiments.
- Do not enable adaptive evolution in runtime configuration or API composition.
- Do not freeze formal thresholds or publish formal F1, Recall, cost, or ablation conclusions.
- Preserve existing budget `reserve`/`settle`/`release` accounting; preflight is advisory only.
- Preserve all candidates and committed rounds from successful earlier rounds when a later stage fails.
- Do not modify or delete the protected Week 3 untracked files listed in the handoff.
- Each task receives an independent specification-compliance review and code-quality review before proceeding.

## File Structure

Create:

- `src/paper_search/evolution/__init__.py` — public offline evolution contracts.
- `src/paper_search/evolution/models.py` — immutable coverage, round, gain, decision, and result models.
- `src/paper_search/evolution/coverage.py` — strong-constraint extraction and deterministic coverage analysis.
- `src/paper_search/evolution/generation.py` — injected next-round generator and deterministic rules implementation.
- `src/paper_search/evolution/costing.py` — injected round estimator and deterministic arithmetic implementation.
- `src/paper_search/evolution/stopping.py` — pure fixed/adaptive stopping policy.
- `src/paper_search/evolution/coordinator.py` — injected multi-round state machine and failure isolation.
- `tests/unit/test_evolution_coverage.py`
- `tests/unit/test_evolution_generation.py`
- `tests/unit/test_evolution_costing.py`
- `tests/unit/test_evolution_stopping.py`
- `tests/unit/test_evolution_coordinator.py`
- `tests/integration/test_evolution_strategies.py`
- `docs/architecture/current-system.md`
- `docs/demo/demo-runbook.md`
- `docs/deployment/new-environment-checklist.md`
- `docs/limitations-and-risks.md`
- `docs/defense/defense-outline.md`

Modify:

- `src/paper_search/control/budget.py` — add non-mutating `can_reserve`.
- `src/paper_search/evaluation/ablations.py` — map public flags to offline strategies only.
- `tests/unit/test_budget.py` — prove preflight/reservation semantic parity.
- `tests/evaluation/test_ablations.py` — prove deterministic strategy mapping and conflict rejection.

Do not modify:

- API composition or UI search-service composition.
- runtime YAML/configuration defaults;
- live Provider implementations;
- frozen data, evaluation outputs, or formal experiment records.

---

### Task 1: Typed Coverage Models and CoverageAnalyzer

**Files:**

- Create: `src/paper_search/evolution/__init__.py`
- Create: `src/paper_search/evolution/models.py`
- Create: `src/paper_search/evolution/coverage.py`
- Create: `tests/unit/test_evolution_coverage.py`

**Interfaces:**

- Consumes: `QuerySpec`, `DomainModel`, `NonEmptyStr` from `paper_search.domain.models`.
- Produces:
  - `ConstraintKind = Literal["must_have", "topics", "methods", "tasks", "datasets", "domains", "venues"]`
  - `CoverageStatus = Literal["covered", "low_coverage", "uncovered"]`
  - `ConstraintRef(kind, value, normalized_value)`
  - `CandidateConstraintObservation(paper_id, constraint, matched)`
  - `ConstraintCoverage(constraint, matched_candidate_ids, status)`
  - `CoverageReport(constraints, covered_count, low_coverage_count, uncovered_count)`
  - `extract_strong_constraints(spec) -> tuple[ConstraintRef, ...]`
  - `CoverageAnalyzer(covered_min_hits).analyze(spec, candidate_ids, observations) -> CoverageReport`

- [ ] **Step 1: Write failing extraction and status tests**

```python
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
```

Also add separate tests asserting:

- no positive strong constraints yields an empty, coverage-complete report;
- candidate IDs and matched IDs are stable, deduplicated, and sorted;
- `covered_min_hits` rejects booleans, non-integers, and values below one;
- missing matrix cells, unknown candidates, unknown constraints, and duplicate
  candidate/constraint cells raise `ValueError`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_evolution_coverage.py -v
```

Expected: collection fails because `paper_search.evolution` does not exist.

- [ ] **Step 3: Implement immutable coverage models**

Add the following definitions to `models.py`:

```python
from typing import Literal
from pydantic import Field
from paper_search.domain.models import DomainModel, NonEmptyStr

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
```

- [ ] **Step 4: Implement strict extraction and complete-matrix analysis**

Implement `coverage.py` with this public shape:

```python
_FIELDS: tuple[ConstraintKind, ...] = (
    "must_have", "topics", "methods", "tasks", "datasets", "domains", "venues"
)


def _normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def extract_strong_constraints(spec: QuerySpec) -> tuple[ConstraintRef, ...]:
    result: list[ConstraintRef] = []
    for kind in _FIELDS:
        seen: set[str] = set()
        for raw in getattr(spec, kind):
            normalized = _normalize(raw)
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(
                ConstraintRef(kind=kind, value=raw, normalized_value=normalized)
            )
    return tuple(result)


class CoverageAnalyzer:
    def __init__(self, covered_min_hits: int) -> None: ...

    def analyze(
        self,
        spec: QuerySpec,
        candidate_ids: Sequence[str],
        observations: Sequence[CandidateConstraintObservation],
    ) -> CoverageReport: ...
```

Inside `analyze`, build the exact expected Cartesian set of sorted unique
candidate IDs and extracted constraints. Reject any observation key outside that
set, any duplicate key, or any missing key. Calculate matched IDs and counts only
after validation, preserving extracted constraint order.

- [ ] **Step 5: Export the coverage API and verify GREEN**

Export the six public coverage symbols from `evolution/__init__.py`, then run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_evolution_coverage.py -v
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/evolution tests/unit/test_evolution_coverage.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evolution
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit**

```powershell
git add src/paper_search/evolution tests/unit/test_evolution_coverage.py
git commit -m "feat: add auditable constraint coverage analysis"
```

---

### Task 2: Deterministic Next-Round Query Generation

**Files:**

- Modify: `src/paper_search/evolution/models.py`
- Create: `src/paper_search/evolution/generation.py`
- Modify: `src/paper_search/evolution/__init__.py`
- Create: `tests/unit/test_evolution_generation.py`

**Interfaces:**

- Consumes: `CoverageReport`, `ConstraintRef`, `QuerySpec`, and existing `SubQuery`.
- Produces:
  - `RoundPlan(round_number, subqueries)`
  - `NoTargetedQueriesError`
  - async `NextRoundGenerator.generate(...) -> RoundPlan`
  - `RuleBasedNextRoundGenerator`

- [ ] **Step 1: Write failing protocol and deterministic-output tests**

```python
def test_targets_only_low_and_uncovered_constraints_in_stable_order() -> None:
    report = coverage_report(statuses=["covered", "low_coverage", "uncovered"])
    result = asyncio.run(
        RuleBasedNextRoundGenerator().generate(
            spec=query_spec(),
            coverage=report,
            prior_plans=[round_plan(1, ["already used"])],
            round_number=2,
            max_subqueries=2,
        )
    )
    assert result.round_number == 2
    assert [item.target_constraints for item in result.subqueries] == [["m2"], ["m3"]]
    assert [item.query_id for item in result.subqueries] == ["evolution-r2-q1", "evolution-r2-q2"]
```

Add tests for whitespace/case duplicate query exclusion, positive integer
validation, clipping to `max_subqueries`, and `NoTargetedQueriesError` when every
possible targeted query was already used.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_evolution_generation.py -v
```

Expected: import fails for `RoundPlan` or `RuleBasedNextRoundGenerator`.

- [ ] **Step 3: Add the round plan and generator protocol**

```python
class RoundPlan(DomainModel):
    round_number: int = Field(strict=True, gt=0)
    subqueries: list[SubQuery] = Field(min_length=1)
```

```python
class NextRoundGenerator(Protocol):
    async def generate(
        self,
        *,
        spec: QuerySpec,
        coverage: CoverageReport,
        prior_plans: Sequence[RoundPlan],
        round_number: int,
        max_subqueries: int,
    ) -> RoundPlan: ...
```

- [ ] **Step 4: Implement the rules generator**

For each low/uncovered constraint, build one `SubQuery`:

```python
SubQuery(
    query_id=f"evolution-r{round_number}-q{len(selected) + 1}",
    text=f"{spec.research_goal} {coverage_item.constraint.value}",
    query_type="decomposed",
    target_constraints=[coverage_item.constraint.value],
    priority=len(selected) + 1,
    provider_hint="either",
)
```

Normalize generated and previous query text with `" ".join(text.split()).casefold()`.
Skip previous text, stop at the explicit limit, and raise
`NoTargetedQueriesError("no unique targeted queries remain")` when no item
survives.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_evolution_generation.py -v
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/evolution tests/unit/test_evolution_generation.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evolution
git add src/paper_search/evolution tests/unit/test_evolution_generation.py
git commit -m "feat: add deterministic evolution query generation"
```

Expected: tests and static checks exit 0 before the commit.

---

### Task 3: Cost Estimation and Budget Preflight

**Files:**

- Create: `src/paper_search/evolution/costing.py`
- Modify: `src/paper_search/evolution/__init__.py`
- Modify: `src/paper_search/control/budget.py`
- Create: `tests/unit/test_evolution_costing.py`
- Modify: `tests/unit/test_budget.py`

**Interfaces:**

- Consumes: `RoundPlan`, `UsageEstimate`, and `HardBudgetController`.
- Produces:
  - `RoundCostEstimator.estimate(plan, completed_round_count) -> UsageEstimate`
  - `DeterministicRoundCostEstimator`
  - `HardBudgetController.can_reserve(estimate) -> bool`

- [ ] **Step 1: Write failing estimator and preflight parity tests**

```python
def test_estimate_scales_all_usage_dimensions_by_subquery_count() -> None:
    estimator = DeterministicRoundCostEstimator(
        search_calls_per_subquery=2,
        llm_calls_per_round=1,
        input_tokens_per_subquery=100,
        output_tokens_per_subquery=20,
        cost_cny_per_subquery=0.01,
        elapsed_ms_per_subquery=500,
    )
    estimate = estimator.estimate(round_plan_with_three_queries(), 0)
    assert estimate == UsageEstimate(
        search_api_calls=6,
        llm_calls=1,
        input_tokens=300,
        output_tokens=60,
        cost_cny=0.03,
        elapsed_ms=1500,
    )


def test_can_reserve_matches_reserve_without_mutating_state() -> None:
    controller = HardBudgetController(search_budget(max_search_api_calls=1))
    estimate = UsageEstimate(search_api_calls=1)
    assert controller.can_reserve(estimate) is True
    assert controller.reserved_usage == UsageEstimate()
    reservation = controller.reserve("test", estimate)
    assert reservation.reserved == estimate
    assert controller.can_reserve(UsageEstimate(search_api_calls=1)) is False
```

Also test invalid numeric assumptions, exact-limit behavior, fail-closed behavior,
active reservations, committed usage, elapsed/token/cost limits, and propagation
of `ReservationError` for an LLM estimate with unknown cost.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_evolution_costing.py tests/unit/test_budget.py -v
```

Expected: failures report missing estimator and `can_reserve`.

- [ ] **Step 3: Implement deterministic estimate arithmetic**

```python
class RoundCostEstimator(Protocol):
    def estimate(
        self,
        plan: RoundPlan,
        completed_round_count: int,
    ) -> UsageEstimate: ...


class DeterministicRoundCostEstimator:
    def __init__(
        self,
        *,
        search_calls_per_subquery: int,
        llm_calls_per_round: int,
        input_tokens_per_subquery: int,
        output_tokens_per_subquery: int,
        cost_cny_per_subquery: float,
        elapsed_ms_per_subquery: int,
    ) -> None: ...

    def estimate(
        self,
        plan: RoundPlan,
        completed_round_count: int,
    ) -> UsageEstimate:
        if (
            isinstance(completed_round_count, bool)
            or not isinstance(completed_round_count, int)
            or completed_round_count < 0
        ):
            raise ValueError("completed_round_count must be a nonnegative integer")
        count = len(plan.subqueries)
        return UsageEstimate(
            search_api_calls=self._search_calls_per_subquery * count,
            llm_calls=self._llm_calls_per_round,
            input_tokens=self._input_tokens_per_subquery * count,
            output_tokens=self._output_tokens_per_subquery * count,
            cost_cny=self._cost_cny_per_subquery * count,
            elapsed_ms=self._elapsed_ms_per_subquery * count,
        )
```

Validate integers as strict nonnegative integers and cost as a finite
nonnegative `int | float` excluding booleans.

- [ ] **Step 4: Add non-mutating budget preflight**

Add this method under the same `RLock` used by `reserve`:

```python
def can_reserve(self, estimate: UsageEstimate) -> bool:
    with self._lock:
        self._expire_locked()
        if self.stop_status() == "hard_stop":
            return False
        if estimate.llm_calls > 0 and estimate.cost_cny is None:
            raise ReservationError("LLM reservations require a known cost estimate")
        candidate = [
            *self._committed,
            *(item.reserved for item in self._reservations.values()),
            estimate,
        ]
        try:
            self._check_hard_limits(candidate)
        except BudgetExceededError:
            return False
        return True
```

Do not create, settle, or release a reservation in this method.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_evolution_costing.py tests/unit/test_budget.py -v
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/control/budget.py src/paper_search/evolution tests/unit/test_budget.py tests/unit/test_evolution_costing.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/control/budget.py src/paper_search/evolution
git add src/paper_search/control/budget.py src/paper_search/evolution tests/unit/test_budget.py tests/unit/test_evolution_costing.py
git commit -m "feat: add evolution cost and budget preflight"
```

Expected: tests and static checks exit 0 before the commit.

---

### Task 4: Auditable Stopping Policy

**Files:**

- Modify: `src/paper_search/evolution/models.py`
- Create: `src/paper_search/evolution/stopping.py`
- Modify: `src/paper_search/evolution/__init__.py`
- Create: `tests/unit/test_evolution_stopping.py`

**Interfaces:**

- Consumes: `CoverageReport`.
- Produces:
  - `EvolutionStrategy = Literal["fixed_one_round", "fixed_two_round", "adaptive"]`
  - stable `StopReason`
  - `MarginalGain`
  - `StopDecision`
  - `decide_stop(...) -> StopDecision`

- [ ] **Step 1: Write one failing test per reason and precedence**

```python
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
```

Add parameterized tests for all six reason codes, fixed-one and fixed-two ignoring
coverage/gain, failure taking precedence, adaptive maximum-round validation, and
finite nonnegative marginal thresholds.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_evolution_stopping.py -v
```

Expected: import failures for stop models and policy.

- [ ] **Step 3: Add gain and decision models**

```python
EvolutionStrategy = Literal["fixed_one_round", "fixed_two_round", "adaptive"]
StopReason = Literal[
    "round_failed",
    "coverage_complete",
    "max_rounds_reached",
    "marginal_gain_below_threshold",
    "budget_insufficient",
    "continue_evolution",
]


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
```

- [ ] **Step 4: Implement the pure policy**

Use these exact booleans:

```python
round_failed = failed_stage is not None
coverage_complete = (
    strategy == "adaptive" and coverage is not None and coverage.is_complete
)
round_limit = (
    completed_rounds >= (1 if strategy == "fixed_one_round"
                         else 2 if strategy == "fixed_two_round"
                         else max_rounds)
)
low_gain = (
    strategy == "adaptive"
    and gain is not None
    and gain.score < marginal_gain_threshold
)
budget_insufficient = not budget_available
```

Choose the first true reason in this order:
`round_failed`, `coverage_complete`, `max_rounds_reached`,
`marginal_gain_below_threshold`, `budget_insufficient`; otherwise return
`continue_evolution`. Always include all five booleans in `checks`.

- [ ] **Step 5: Verify GREEN and commit**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_evolution_stopping.py -v
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/evolution tests/unit/test_evolution_stopping.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evolution
git add src/paper_search/evolution tests/unit/test_evolution_stopping.py
git commit -m "feat: add auditable evolution stopping policy"
```

Expected: tests and static checks exit 0 before the commit.

---

### Task 5: Failure-Isolated EvolutionCoordinator

**Files:**

- Modify: `src/paper_search/evolution/models.py`
- Create: `src/paper_search/evolution/coordinator.py`
- Modify: `src/paper_search/evolution/__init__.py`
- Create: `tests/unit/test_evolution_coordinator.py`
- Create: `tests/integration/test_evolution_strategies.py`

**Interfaces:**

- Consumes: all previous evolution contracts plus `Paper`, `UsageActual`, and
  injected `HardBudgetController.can_reserve`.
- Produces:
  - `RoundExecution`
  - `EvolutionResult`
  - async `RoundExecutor.execute(spec, plan) -> RoundExecution`
  - `GainEvaluator.evaluate(previous_ids, current_ids, execution) -> MarginalGain`
  - `BudgetPreflight.can_reserve(estimate) -> bool`
  - `EvolutionCoordinator.run(...) -> EvolutionResult`

- [ ] **Step 1: Write failing fixed-strategy and adaptive integration tests**

```python
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
```

Add isolated failure tests for estimator, preflight rejection, executor,
coverage analyzer, gain evaluator, and generator. Each failure assertion must
verify that previously committed rounds and first-seen candidate objects remain
unchanged. Add a duplicate-paper test proving later metadata does not replace
the prior object and later conflicting observations do not overwrite prior
observations.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_evolution_coordinator.py tests/integration/test_evolution_strategies.py -v
```

Expected: imports fail for coordinator models and protocols.

- [ ] **Step 3: Add round/result models and injected protocols**

```python
class RoundExecution(DomainModel):
    round_number: int = Field(strict=True, gt=0)
    candidates: list[Paper]
    observations: list[CandidateConstraintObservation]
    usage: UsageActual
    trace: list[dict[str, Any]]


class EvolutionResult(DomainModel):
    strategy: EvolutionStrategy
    rounds: list[RoundExecution]
    candidates: list[Paper]
    decisions: list[StopDecision]
    stop_reason: StopReason
    warnings: list[str]
    failed_round: int | None = Field(default=None, strict=True, gt=0)
```

```python
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
```

- [ ] **Step 4: Implement append-only state helpers**

```python
def _merge_candidates(existing: Sequence[Paper], incoming: Sequence[Paper]) -> list[Paper]:
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
        key = (item.paper_id, item.constraint.kind, item.constraint.normalized_value)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
```

- [ ] **Step 5: Implement the coordinator state machine**

Constructor:

```python
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
    ) -> None: ...
```

Run signature:

```python
async def run(
    self,
    *,
    spec: QuerySpec,
    initial_plan: RoundPlan,
    strategy: EvolutionStrategy,
    max_rounds: int,
    max_subqueries: int,
    marginal_gain_threshold: float,
) -> EvolutionResult: ...
```

Implementation order:

1. Estimate the current plan; on exception return `round_failed` with
   `failed_stage="estimate"`.
2. Call `budget.can_reserve`; if false emit `budget_insufficient` without
   executing.
3. Execute; validate `execution.round_number == plan.round_number`; only then
   append it and merge candidates/observations.
4. Analyze cumulative coverage using all retained candidate IDs and first-seen
   observations.
5. Evaluate gain against the candidate ID set before this round.
6. Call `decide_stop(..., budget_available=True)`. Return immediately for a
   terminal reason.
7. Generate the next plan. On failure retain committed state and return
   `round_failed` with `failed_stage="generation"`.
8. Estimate the next plan and preflight it. Re-run `decide_stop` with the actual
   budget boolean, append exactly one post-round decision, and execute the next
   round only when it returns `continue_evolution`.

Catch dependency exceptions at these boundaries only. Convert them to stable
warnings such as `"generation: dependency failure"` without including exception
text, credentials, payloads, or query contents.

- [ ] **Step 6: Verify focused GREEN**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_evolution_coordinator.py tests/integration/test_evolution_strategies.py -v
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/evolution tests/unit/test_evolution_coordinator.py tests/integration/test_evolution_strategies.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evolution
```

Expected: all commands exit 0.

- [ ] **Step 7: Run the existing pipeline regression slice and commit**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/integration/test_orchestrator.py tests/unit/test_budget.py -v
git add src/paper_search/evolution tests/unit/test_evolution_coordinator.py tests/integration/test_evolution_strategies.py
git commit -m "feat: coordinate offline adaptive query rounds"
```

Expected: regression tests pass before the commit.

---

### Task 6: Offline Ablation Strategy Injection

**Files:**

- Modify: `src/paper_search/evaluation/ablations.py`
- Modify: `tests/evaluation/test_ablations.py`

**Interfaces:**

- Consumes: existing public module flags and `EvolutionStrategy`.
- Produces:
  - `evolution_strategy_for_modules(modules: Mapping[str, bool]) -> EvolutionStrategy`

- [ ] **Step 1: Write failing mapping tests**

```python
@pytest.mark.parametrize(
    ("fixed_two", "adaptive", "expected"),
    [
        (False, False, "fixed_one_round"),
        (True, False, "fixed_two_round"),
        (False, True, "adaptive"),
    ],
)
def test_maps_public_flags_to_offline_strategy(
    fixed_two: bool,
    adaptive: bool,
    expected: EvolutionStrategy,
) -> None:
    modules = all_public_modules_disabled()
    modules["fixed_two_round"] = fixed_two
    modules["adaptive_evolution"] = adaptive
    assert evolution_strategy_for_modules(modules) == expected


def test_rejects_conflicting_evolution_flags() -> None:
    modules = all_public_modules_disabled()
    modules["fixed_two_round"] = True
    modules["adaptive_evolution"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        evolution_strategy_for_modules(modules)
```

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_ablations.py -v
```

Expected: import failure for `evolution_strategy_for_modules`.

- [ ] **Step 3: Implement the mapping without executing an experiment**

```python
def evolution_strategy_for_modules(
    modules: Mapping[str, bool],
) -> EvolutionStrategy:
    fixed_two = modules.get("fixed_two_round")
    adaptive = modules.get("adaptive_evolution")
    if not isinstance(fixed_two, bool) or not isinstance(adaptive, bool):
        raise ValueError("evolution flags must be booleans")
    if fixed_two and adaptive:
        raise ValueError("evolution flags are mutually exclusive")
    if adaptive:
        return "adaptive"
    if fixed_two:
        return "fixed_two_round"
    return "fixed_one_round"
```

Do not modify `run_ablations`, `ExperimentAggregate`, split policy, runtime
configuration, or any aggregate values.

- [ ] **Step 4: Verify GREEN and commit**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_ablations.py -v
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check src/paper_search/evaluation/ablations.py tests/evaluation/test_ablations.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/ablations.py
git add src/paper_search/evaluation/ablations.py tests/evaluation/test_ablations.py
git commit -m "feat: map ablations to offline evolution strategies"
```

Expected: tests and static checks exit 0 before the commit.

---

### Task 7: Architecture and Demonstration Documentation

**Files:**

- Create: `docs/architecture/current-system.md`
- Create: `docs/demo/demo-runbook.md`

**Interfaces:**

- Consumes: implemented package boundaries and current CLI/API behavior.
- Produces: factual architecture description and executable demo skeleton with
  measured outputs explicitly deferred.

- [ ] **Step 1: Draft the current-system architecture document**

Use these exact sections:

```markdown
# Current System Architecture

## Scope and Revision
## QuerySpec and QueryPlanner
## Provider Boundary
## Cache and Budget Accounting
## Deduplication, Filtering, Fusion, and Ranking
## Evaluation, Snapshots, and Reproducibility
## Single-Run Pipeline
## Offline Adaptive Evolution Boundary
## Components Not Enabled in Main Configuration
```

State that the adaptive coordinator wraps an injected single-round executor,
does not replace `MockSearchOrchestrator`, and is not enabled in API composition.
Describe R2 only as retrieval diagnostic evidence and do not include relevance
metrics.

- [ ] **Step 2: Draft the demonstration runbook**

Use these exact sections:

```markdown
# Demonstration Runbook

## Preconditions
## Start the Mock API
## Health Check
## Run a Demonstration Query
## Inspect Trace and Usage
## Demonstrate Provider Degradation
## Stop the Service
## Outputs Deferred Until R3
```

Commands must use the mock server and `--no-env-file` where applicable. The
degraded scenario must use an injected fake or documented mock mode, never a
real Provider outage. State that formal metrics, costs, screenshots, and timing
measurements are intentionally absent.

- [ ] **Step 3: Audit the documents against source**

Run:

```powershell
rg -n "QuerySpec|QueryPlanner|HardBudgetController|MockSearchOrchestrator|adaptive|R2|R3" docs/architecture/current-system.md docs/demo/demo-runbook.md
rg -n "final F1|final Recall|最终指标|最终成本|正式结论" docs/architecture/current-system.md docs/demo/demo-runbook.md
```

Expected: the first command finds every required concept; the second command
finds no fabricated result statement.

- [ ] **Step 4: Commit**

```powershell
git add docs/architecture/current-system.md docs/demo/demo-runbook.md
git commit -m "docs: add architecture and demo runbook"
```

---

### Task 8: Deployment, Risk, and Defense Documentation

**Files:**

- Create: `docs/deployment/new-environment-checklist.md`
- Create: `docs/limitations-and-risks.md`
- Create: `docs/defense/defense-outline.md`

**Interfaces:**

- Consumes: project Python/uv constraints, handoff-confirmed R2/R3 facts, and
  Task 12's offline status.
- Produces: operational acceptance checklist, explicit limitations, and a
  defense outline with formal outputs deferred.

- [ ] **Step 1: Write the deployment and acceptance checklist**

Use checkboxes under these sections:

```markdown
# New Environment Deployment and Acceptance Checklist

## Python 3.11 and uv
## Dependency Installation
## Environment Variable Names
## Offline Test Gate
## Mock Server Gate
## API Readiness
## Fresh-Cache Run Gate
## Artifact Verification
## Secret-Handling Rules
```

Require checking variable names without printing values. Separate the offline
gate, which can run now, from fresh-cache/API-readiness gates that require later
operator authorization and credentials.

- [ ] **Step 2: Write limitations and risks**

Include each fact as an explicit statement:

```markdown
- Identifier-map wiring and the R3 baseline are external dependencies and were
  not present at this task's starting revision.
- R2 is retrieval diagnostic evidence only.
- R2's zero relevance metrics are caused by an identifier namespace mismatch
  and are not a retrieval-performance conclusion.
- Seven `invalid_work` records require later quality analysis.
- Adaptive evolution has no real fixed-strategy comparison yet.
- Relationship visualization has not passed its stage gate.
```

Also document mitigations: wait for R3, preserve frozen inputs, use identical
budget/data comparisons later, and keep adaptive behavior disabled until gates
pass.

- [ ] **Step 3: Write the defense outline**

Use these sections:

```markdown
# Defense Outline

## Problem Definition
## Architecture
## Baseline
## Innovation Module
## Experimental Method
## Failure Analysis
## Cost and Reproducibility
## Limitations
## Formal Results to Add After R3
```

The final section lists the required future artifact names—formal metric table,
cost table, and ablation table—without empty numeric cells or invented values.

- [ ] **Step 4: Run a prohibited-content audit**

```powershell
rg -n "(API_KEY|SECRET|TOKEN)=|sk-[A-Za-z0-9]|最终 F1|最终 Recall|最终成本为|优于 baseline" docs/deployment/new-environment-checklist.md docs/limitations-and-risks.md docs/defense/defense-outline.md
```

Expected: no secret assignment or unsupported final claim. Mentions that merely
name a variable must not include `=value`.

- [ ] **Step 5: Commit**

```powershell
git add docs/deployment/new-environment-checklist.md docs/limitations-and-risks.md docs/defense/defense-outline.md
git commit -m "docs: add deployment risk and defense materials"
```

---

### Task 9: Full Offline Verification and Final Review

**Files:**

- Modify only files required to fix failures attributable to Tasks 1–8.
- Do not change scope, thresholds, runtime configuration, datasets, or formal
  experiment outputs during this task.

**Interfaces:**

- Consumes: all Task 12 code/tests and Task 14 documents.
- Produces: final verification evidence and a clean branch ready for integration.

- [ ] **Step 1: Run all focused Task 12 tests**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_evolution_coverage.py tests/unit/test_evolution_generation.py tests/unit/test_evolution_costing.py tests/unit/test_evolution_stopping.py tests/unit/test_evolution_coordinator.py tests/integration/test_evolution_strategies.py tests/unit/test_budget.py tests/evaluation/test_ablations.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run Ruff**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check .
```

Expected: exit 0 with no diagnostics.

- [ ] **Step 3: Run strict mypy**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
```

Expected: exit 0 with no errors.

- [ ] **Step 4: Run the complete offline suite**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
```

Expected: all offline tests pass; the credential-gated OpenAlex live test remains
skipped.

- [ ] **Step 5: Audit scope and prohibited claims**

```powershell
git diff 33f0cf4dba11c31be998af67b0f5457372269bb9 --name-only
git diff --check 33f0cf4dba11c31be998af67b0f5457372269bb9
rg -n "OPENALEX_API_KEY|SEMANTIC_SCHOLAR_API_KEY|LLM_API_KEY" src/paper_search/evolution tests/unit/test_evolution_* tests/integration/test_evolution_strategies.py
rg -n "最终指标|最终成本|稳定提升|正式消融结论" docs/architecture docs/demo docs/deployment docs/limitations-and-risks.md docs/defense
```

Expected:

- changed files stay within this plan;
- `git diff --check` is clean;
- evolution code/tests contain no credential dependency;
- documents contain no unsupported final claim.

- [ ] **Step 6: Request independent specification and quality reviews**

Dispatch one reviewer against the approved spec and one reviewer against the
implementation diff. Require both reviewers to inspect:

- strict offline boundaries;
- coverage-matrix validation;
- budget semantic parity and no duplicate reservation;
- reason-code precedence;
- append-only failure isolation;
- absence of runtime enablement and fabricated Task 14 outputs.

Resolve every valid finding with a new RED-to-GREEN test when behavior changes,
then rerun Steps 1–5.

- [ ] **Step 7: Record final status**

```powershell
git status --short --branch
git log --oneline 33f0cf4dba11c31be998af67b0f5457372269bb9..HEAD
```

Expected: clean worktree and a reviewable sequence of focused commits. Do not
create a Git tag, run a formal baseline, or publish final metrics.
