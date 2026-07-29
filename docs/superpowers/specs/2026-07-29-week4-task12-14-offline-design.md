# Week 4 Task 12/14 Offline Infrastructure Design

Date: 2026-07-29

## Goal

Build the purely offline infrastructure needed to evaluate budget-aware adaptive
query evolution, and prepare Task 14 Markdown delivery material that does not
depend on the Week 3 R3 baseline, formal metrics, final configuration selection,
or external services.

The implementation starts from commit
`33f0cf4dba11c31be998af67b0f5457372269bb9` and remains isolated on branch
`codex/week4-task12-14-offline`.

## Scope

Task 12 includes:

- deterministic strong-constraint coverage analysis;
- an injected interface for generating a targeted next-round query plan;
- pre-execution usage and cost estimation;
- auditable stopping decisions;
- offline coordination of fixed-one-round, fixed-two-round, and adaptive
  strategies;
- fake-driven unit and integration tests;
- an injection boundary compatible with the existing ablation registry.

Task 14 includes:

- current-system architecture documentation;
- a demonstration runbook skeleton;
- a new-environment deployment and acceptance checklist;
- limitations and risks;
- a defense/presentation outline.

The following remain deferred until the Week 3 R3 baseline and later stage gates:

- real OpenAlex, Semantic Scholar, or LLM adaptive experiments;
- formal fixed-one/fixed-two/adaptive comparisons;
- F1, Recall, cost, or ablation conclusions;
- final threshold selection or main-configuration enablement;
- validation/test-set tuning;
- relationship visualization, final screenshots, final slides, tags, or release
  freezing.

## Existing Components and Ownership

The existing `paper_search.pipeline.orchestrator.MockSearchOrchestrator` remains
the owner of a single search run: query analysis, provider calls, reservations,
deduplication, filtering, fusion, optional ranking stages, and degraded behavior.
Task 12 does not duplicate that pipeline.

The new adaptive layer coordinates rounds through injected protocols. A future
production adapter may wrap the existing single-run pipeline, but this change
only provides the offline protocol and fake-backed tests.

Existing components reused directly are:

- `QuerySpec`, `SubQuery`, `UsageEstimate`, `UsageActual`, and the frozen
  `DomainModel` convention from `paper_search.domain.models`;
- `HardBudgetController` reservation and settlement rules;
- query normalization and deterministic ordering conventions from
  `paper_search.query.planner`;
- injected evaluator conventions and the existing `fixed_two_round` and
  `adaptive_evolution` flags in `paper_search.evaluation.ablations`;
- immutable result accumulation and degraded behavior patterns from the current
  pipeline and evaluation runner.

## Package Boundary

Create an isolated experimental package:

```text
src/paper_search/evolution/
├── __init__.py
├── models.py
├── coverage.py
├── generation.py
├── costing.py
├── stopping.py
└── coordinator.py
```

The package is not imported by the main API composition or enabled by runtime
configuration in this phase.

### `models.py`

This module owns immutable, JSON-serializable audit models:

- `ConstraintRef`: constraint kind, original value, and deterministic normalized
  value;
- `CandidateConstraintObservation`: paper ID, constraint reference, and whether
  the candidate matches;
- `ConstraintCoverage`: one constraint's candidate IDs, hit count, and coverage
  status;
- `CoverageReport`: ordered per-constraint results and status totals;
- `RoundPlan`: round number and ordered `SubQuery` values;
- `RoundExecution`: candidates, per-candidate observations, usage, and safe trace
  entries returned by an injected executor;
- `MarginalGain`: raw new-candidate and new-high-relevance counts, an injected
  deterministic gain score, and optional F1/Recall deltas reserved for a future
  formal evaluator;
- `StopDecision`: continue flag, stable reason code, strategy, round number,
  thresholds, and individual check outcomes;
- `EvolutionResult`: successfully committed rounds, retained candidates,
  decisions, stop reason, warnings, and an optional failed-round number.

Formal metric delta fields remain `None` in this offline phase. Fake tests may
populate deterministic values only to verify transport and stopping behavior;
they are never published as experiment results.

### `coverage.py`

`CoverageAnalyzer` is deterministic and has no Provider, LLM, environment, or
network dependency.

Positive strong constraints are drawn, in stable field order, from:

1. `must_have`;
2. `topics`;
3. `methods`;
4. `tasks`;
5. `datasets`;
6. `domains`;
7. `venues`.

`year_from`, `year_to`, and `exclusions` remain hard filtering conditions. They
are not positive query targets and therefore do not contribute to positive
candidate coverage.

Constraint values are whitespace-normalized and compared case-insensitively.
Duplicates are removed only within the same constraint kind. Equal text in two
different kinds remains two independently auditable constraints.

The analyzer receives the candidate ID set and explicit
`CandidateConstraintObservation` records instead of inferring typed constraints
from the existing string-only `CandidateEvidence.matched_constraints`. For every
candidate and extracted constraint, callers must supply exactly one observation.
Missing matrix cells, unknown candidate IDs, unknown constraints, and duplicate
cells are rejected instead of silently becoming non-matches.

For a caller-supplied `covered_min_hits` greater than zero:

- zero matching candidates means `uncovered`;
- a positive count below `covered_min_hits` means `low_coverage`;
- a count at or above `covered_min_hits` means `covered`.

Candidate IDs are deduplicated and sorted before counts are calculated. A
`QuerySpec` with no positive strong constraints produces an empty report that is
considered coverage-complete, so simple queries do not trigger an unnecessary
second round.

### `generation.py`

Define an asynchronous `NextRoundGenerator` protocol. It receives the original
`QuerySpec`, current `CoverageReport`, prior plans, the next round number, and an
explicit subquery limit. It returns a `RoundPlan`.

The offline implementation is a deterministic rule generator:

- it targets only low-coverage and uncovered constraints;
- it reuses existing `SubQuery` and normalization conventions;
- it preserves stable constraint and query ordering;
- it removes query text already used by earlier rounds;
- it never reads a prompt, environment variable, Provider, or LLM client.

Tests may inject a fake generator. No experimental generator is added to main
configuration.

### `costing.py`

Define a synchronous `RoundCostEstimator` protocol that maps a proposed
`RoundPlan` and current state to `UsageEstimate`.

The deterministic estimator uses explicit per-subquery search-call, token, cost,
and elapsed-time assumptions supplied at construction. It performs arithmetic
only; it does not reserve budget or execute work.

Add a read-only `HardBudgetController.can_reserve(estimate)` preflight using the
same locked hard-limit validation as `reserve()`. Invalid LLM estimates with
unknown cost remain errors. A `False` result means the estimate cannot currently
fit.

Preflight is advisory because concurrent usage may change after the check.
Actual work must continue to use the existing per-operation
`reserve`/`settle`/`release` mechanism. A later reservation rejection is
authoritative and becomes an audited `budget_insufficient` stop. No second
budget ledger and no aggregate round reservation are introduced.

### `stopping.py`

Stopping is a pure deterministic policy that emits `StopDecision` for every
decision point. Reason codes are:

- `round_failed`;
- `coverage_complete`;
- `max_rounds_reached`;
- `marginal_gain_below_threshold`;
- `budget_insufficient`;
- `continue_evolution`.

For the adaptive strategy, checks use the priority above. The decision records
all evaluated checks even though it exposes one primary reason code.

Fixed strategies do not use coverage or marginal-gain checks:

- fixed one round stops after one successfully committed round;
- fixed two rounds stops after two successfully committed rounds;
- both may stop earlier after a round failure or when the next round cannot fit
  the budget.

Adaptive thresholds and maximum rounds are explicit constructor or call
arguments. This phase supplies test values only and does not write them into
runtime configuration.

### `coordinator.py`

Define injected asynchronous protocols:

- `RoundExecutor`, responsible for executing one `RoundPlan` and returning
  `RoundExecution`;
- `GainEvaluator`, responsible for comparing the committed candidate set before
  and after a successful round and returning `MarginalGain`.

`EvolutionCoordinator` owns only the round state machine:

1. accept the `QuerySpec`, caller-supplied initial plan, strategy, and explicit
   non-final thresholds;
2. estimate and preflight every plan, including the first plan, before execution;
3. execute the plan and commit a round only after `RoundExecutor` returns a valid
   complete result;
4. deterministically merge candidates by canonical ID without replacing prior
   successful state;
5. analyze coverage and evaluate realized gain;
6. evaluate failure, coverage, round-limit, and marginal-gain stop conditions;
7. only when those checks permit another round, generate its plan, estimate it,
   and apply the budget preflight;
8. emit a `StopDecision` and repeat only when it says to continue.

For `fixed_two_round` only, if the targeted generator has no low-coverage or
uncovered target, the coordinator replays a deep snapshot of the initial plan.
The replay preserves query IDs because they identify the same query semantics;
the new `RoundPlan.round_number` distinguishes the separate execution instance.
This fallback is not used for other strategies or generator failures.

If generation, estimation, execution, coverage analysis, or gain evaluation
fails, the coordinator records a bounded safe warning and `round_failed`.
Previously committed rounds and candidates remain unchanged. Provider usage
already settled inside a failing executor remains owned by the shared budget
controller; the coordinator never rewrites accounting state.

## Ablation Integration

`paper_search.evaluation.ablations` already exposes public boolean flags named
`fixed_two_round` and `adaptive_evolution`. This phase may add a small mapping or
factory protocol that translates those flags into the corresponding offline
strategy. It must not:

- execute a frozen dataset;
- attach aggregate results;
- tune on validation data;
- claim a preferred strategy;
- enable either strategy in the main runtime configuration.

## Task 14 Documents

Create the following Markdown files:

### `docs/architecture/current-system.md`

Describe QuerySpec and QueryPlanner, Providers, cache and budget accounting,
deduplication/filtering/ranking, evaluation snapshots, the current single-run
pipeline, and the reserved adaptive coordinator boundary. Distinguish current
behavior from deferred work.

### `docs/demo/demo-runbook.md`

Provide startup, health-check, demonstration query, trace/usage inspection, and
degraded Provider scenarios. Provide commands and state explicitly where measured
outputs remain deferred; do not include final metrics or fabricated screenshots.

### `docs/deployment/new-environment-checklist.md`

Cover Python 3.11, uv, dependency installation, environment-variable name
checks, offline tests, mock server, API readiness, later fresh-cache execution,
and artifact validation. Explicitly prohibit recording secret values.

### `docs/limitations-and-risks.md`

State that identifier-map/R3 work is external and incomplete at this starting
revision, R2 is retrieval diagnostic evidence only, zero R2 relevance metrics
are not performance conclusions, seven `invalid_work` records remain for later
analysis, adaptive evolution has no real comparison yet, and relationship
visualization has not passed its stage gate.

### `docs/defense/defense-outline.md`

Provide sections for problem definition, architecture, baseline, innovation,
experimental method, failure analysis, cost/reproducibility, and limitations.
Mark formal R3 metrics and final conclusions as deferred inputs rather than
inventing values.

## Testing

Development follows RED-to-GREEN TDD.

Focused unit tests cover:

- typed constraint identity, normalization, ordering, and duplicate handling;
- covered, low-coverage, uncovered, and no-constraint reports;
- deterministic targeted query generation and prior-query exclusion;
- estimate arithmetic and validation;
- budget preflight equivalence with actual hard-limit rules;
- every stop reason and reason precedence;
- fixed-one, fixed-two, and adaptive state transitions;
- preservation of committed candidates after each injected failure point.

Integration tests use only fake round executors, fake candidate observations, and
fake gain evaluators. They assert that the three strategies produce deterministic
round counts and audit trails without network access.

Verification uses the existing project environment and always passes
`--no-env-file`:

```powershell
D:\Dev\uv\uv.exe run --no-sync --no-env-file pytest <focused tests>
D:\Dev\uv\uv.exe run --no-sync --no-env-file ruff check .
D:\Dev\uv\uv.exe run --no-sync --no-env-file mypy src
D:\Dev\uv\uv.exe run --no-sync --no-env-file pytest -q
```

No test may require a Provider credential, real HTTP request, real LLM call, or
dev/validation/test dataset experiment.

## Acceptance Criteria

- All new Task 12 behavior is deterministic, immutable, auditable, and injectable.
- The main runtime configuration and API composition do not enable adaptive
  evolution.
- Budget preflight shares hard-limit semantics with the existing controller;
  execution still uses real reservations and settlements.
- A failed new round cannot remove or replace candidates from prior successful
  rounds.
- Existing ablation names can select an offline strategy without producing
  experiment claims.
- All five Task 14 Markdown deliverables contain no fabricated metrics, costs,
  screenshots, or conclusions.
- Focused tests, Ruff, mypy, and the complete offline pytest suite pass with
  `--no-env-file`.
