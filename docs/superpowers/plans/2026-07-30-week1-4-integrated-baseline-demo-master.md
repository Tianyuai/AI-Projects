# Week 1–4 Integrated Baseline and Demo Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the separately completed Week 1–4 components into one reproducible formal baseline and one interactive demo that share the same application service.

**Architecture:** Execute four dependency-ordered plans: shared contracts/Gate 0, Replay/Live runtime spine, formal evaluation/freeze, then API/UI/experiments. Gates 0–5 are baseline delivery; Gate 6 is separate optional-module evidence. No later phase may bypass an earlier contract by constructing another pipeline.

**Tech Stack:** Python 3.11+, Pydantic v2, asyncio/httpx, SQLite, FastAPI/Uvicorn, browser-native JavaScript, pytest, Ruff, mypy, Git.

## Global Constraints

- Approved design authority: `docs/superpowers/specs/2026-07-30-week1-4-integrated-baseline-demo-design.md`.
- Detailed plans are normative for implementation mechanics:
  - `docs/superpowers/plans/2026-07-30-week1-4-phase1-contracts-gate0.md`
  - `docs/superpowers/plans/2026-07-30-week1-4-phase2-runtime-spine.md`
  - `docs/superpowers/plans/2026-07-30-week1-4-phase3-formal-evaluation.md`
  - `docs/superpowers/plans/2026-07-30-week1-4-phase4-api-ui-experiments.md`
- Execution order is Phase 1 → Phase 2 → Phase 3 → Phase 4. Within a phase, follow task order unless the detailed plan explicitly identifies independent tests.
- The main agent integrates one task at a time. With subagent-driven execution, each implementation task receives a fresh worker, then a spec-compliance review, then a code-quality review before the task commit is accepted.
- Test-only fake prices/transports/snapshots must be unmistakably fixture-scoped. Real prices, credentials, readiness, and gated data are operator inputs and are never fabricated.
- Network access is never inferred from plan approval. Every real live smoke, dev capture, validation capture, or live browser demonstration requires explicit user authorization at that checkpoint.
- Validation live authorization is also bounded by the one-attempt claim. Plan or code approval does not authorize consuming it.
- Preserve unrelated worktree changes and untracked files. Stage only files named by the active task.
- Full test/Ruff/mypy evidence is required at every phase exit and before final integration claims.

---

## Delivery Map

| Phase | Detailed plan | Primary output | Blocks |
| --- | --- | --- | --- |
| 1 | `2026-07-30-week1-4-phase1-contracts-gate0.md` | Canonical contracts, exact locks, pricing/Gate policies, V2 freeze migration, Gate 0 verifier | All real live/formal claims |
| 2 | `2026-07-30-week1-4-phase2-runtime-spine.md` | Dependency snapshot V2, replay/live adapters, ledgers, application service, composition, smoke CLI | Formal evaluation and unified server |
| 3 | `2026-07-30-week1-4-phase3-formal-evaluation.md` | Ordered failure-safe evaluation, atomic artifacts, validator, dev/validation capture/replay evidence | Delivery status and optional ablations |
| 4 | `2026-07-30-week1-4-phase4-api-ui-experiments.md` | Typed API, browser UI, serve lifecycle, default-off experiment wiring, Gate 5/6 evidence | Final Week 1–4 delivery |

## Approved Design Coverage

| Approved design area | Single owning plan/task |
| --- | --- |
| Normative request/response/readiness/error/execution contracts | Phase 1 Task 1 |
| Candidate, validation, and replay lock semantics | Phase 1 Task 2 |
| Fixed baseline identities, timeouts, retries, routing, and request budget | Phase 1 Tasks 2–3 |
| Pricing policy and quality/PRD threshold reconciliation | Phase 1 Task 3 |
| V1/V2 freeze migration and public status safety | Phase 1 Tasks 4–6 |
| LLM/OpenAlex/Semantic Scholar exact-byte capture and offline replay | Phase 2 Tasks 1–3 |
| Request/run/project budgets and CNY 160/200 project stops | Phase 2 Task 4 |
| Evidence-preserving orchestrator and one application service | Phase 2 Task 5 |
| Replay/live composition and smoke commands | Phase 2 Tasks 6–7 |
| Full-query prediction/failure/diagnostic preservation | Phase 3 Task 1 |
| Formal validity, baseline quality, reporting, and promotion Gates | Phase 3 Task 2 |
| Atomic run publication, safe reports, and access-controlled artifacts | Phase 3 Task 3 |
| Irrevocable one-attempt validation claim | Phase 3 Task 4 |
| Shared-service evaluation and formal command outcomes | Phase 3 Tasks 5–6 |
| Authoritative dev and validation chronology | Phase 3 Tasks 7–8 |
| Typed HTTP behavior and mode-aware readiness | Phase 4 Task 1 |
| Browser UI through canonical API | Phase 4 Task 2 |
| Replay-bound server and per-request live capture | Phase 4 Task 3 |
| Optional modules, corrected ablations, and default-off protection | Phase 4 Task 4 |
| Process E2E, real-browser Gate 5, docs, and Gate 6 | Phase 4 Tasks 5–8 |

## Shared Production Boundary

```python
bundle = CompositionRoot.compose(...)
result = await bundle.service.execute(request)
```

The following are the only allowed production consumers:

```text
paper-search smoke
paper-search evaluate
FastAPI /v1/search
browser UI through /v1/search
```

The browser UI, evaluation runner, and CLI must not import concrete LLM/Provider live or replay adapters.

---

### Task 1: Establish execution baseline and task protocol

**Files:**

- Read: all five approved design/plan documents listed above.
- No code change.

**Interfaces:**

Each detailed task follows this state transition:

```text
pending -> implementation complete -> spec review passed
        -> quality review passed -> focused verification passed -> committed
```

No task advances on failed review or verification.

**Steps:**

- [ ] Confirm the design commit and plan commit are ancestors of the execution branch.

```powershell
git log --oneline --decorate -5
git merge-base --is-ancestor 760945d982ab896570aff4388d1c03318c5899e3 HEAD
```

Expected result: the second command exits 0.

- [ ] Capture baseline worktree status and do not stage unrelated paths.

```powershell
git status --short
git diff --check
```

Expected result: no whitespace error; all pre-existing unrelated paths are recorded in the execution handoff.

- [ ] Run baseline verification before the first code task.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
```

Expected result: pytest reports the accepted baseline or better, and both static checks exit 0.

- [ ] For each detailed-plan task, dispatch one implementation worker with the exact task text and write scope.
- [ ] After worker completion, inspect the diff and run the task's focused verification.
- [ ] Dispatch a spec-compliance review; correct any missing/extra behavior.
- [ ] Dispatch a code-quality review; correct any maintainability, safety, or test weakness.
- [ ] Re-run focused verification, `git diff --check`, then make the task commit.

---

### Task 2: Execute Phase 1 and resolve Gate 0 truthfully

**Files:**

- Implement exactly: `docs/superpowers/plans/2026-07-30-week1-4-phase1-contracts-gate0.md`.

**Interfaces:**

Phase 1 exit is `(contracts stable, lock schemas stable, engineering checks pass, Gate0Report passed|blocked)`. Only `passed` authorizes real formal/live evidence claims.

**Steps:**

- [ ] Complete Phase 1 Tasks 1–5 with focused red/green/static evidence and task commits.
- [ ] Run the Gate 0 verifier against current repository evidence.
- [ ] If Gate 0 is blocked, preserve the blocking report and continue only fixture-based engineering allowed by later plans.
- [ ] When operator V2 freeze, identifier map, pricing, thresholds, and readiness evidence are supplied, re-run Gate 0 against those exact access-controlled bytes.
- [ ] Execute Phase 1 Task 6 only after `"passed": true`.
- [ ] Run Phase 1 exit verification.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/application tests/api tests/unit/test_config.py tests/unit/test_pricing.py tests/unit/test_budget.py tests/evaluation/test_freeze_schema.py tests/evaluation/test_freeze.py tests/evaluation/test_gate0.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
```

Expected result: all engineering checks exit 0. Gate 0 is reported separately as passed or blocked; a blocked Gate is never relabeled success.

**Review checkpoint:**

- Canonical contracts and exact lock schemas are stable.
- No production price was invented.
- V1 evidence cannot authorize formal execution.
- Public status changes, if any, are backed by a passing Gate 0 report.

---

### Task 3: Execute Phase 2 and pass engineering Gates 1–2

**Files:**

- Implement exactly: `docs/superpowers/plans/2026-07-30-week1-4-phase2-runtime-spine.md`.

**Interfaces:**

Phase 2 consumes Phase 1 models and emits one `ApplicationBundle` boundary plus verified replay/capture/ledger primitives. No concrete adapter is exported to consumers.

**Steps:**

- [ ] Complete Phase 2 Tasks 1–7 with task-level worker/review cycles.
- [ ] Run Gate 1 replay smoke twice with the network tripwire.
- [ ] Run fake-live Gate 2 contract tests for capture, pricing, ledgers, replay-lock emission, and replay equivalence.
- [ ] Stop before a real live Gate 2 command and request explicit network/cost authorization from the user.
- [ ] After authorization and a passing Gate 0, run the bounded real live smoke and immediate replay proof.
- [ ] Run Phase 2 exit verification.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_dependency_snapshot.py tests/unit/test_llm_snapshot_adapters.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_budget_ledger.py tests/unit/test_application_service.py tests/integration/test_application_composition.py tests/integration/test_smoke_cli.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
```

Expected result: all commands exit 0.

**Review checkpoint:**

- Replay objects cannot reach network code.
- Live capture seals exact successful bytes and excludes unsafe data.
- Actual cost is valued before settlement.
- `SearchApplicationService` is the only production search boundary.
- Optional modules are not constructed.

---

### Task 4: Execute Phase 3 and obtain Gates 3–4 evidence

**Files:**

- Implement exactly: `docs/superpowers/plans/2026-07-30-week1-4-phase3-formal-evaluation.md`.

**Interfaces:**

Phase 3 consumes `ApplicationBundle` and emits machine-verifiable run directories accepted by `verify-run` and capture/replay pairs accepted by `compare-replay`.

**Steps:**

- [ ] Complete Phase 3 Tasks 1–6 with task-level worker/review cycles.
- [ ] Verify synthetic formal capture/replay fixtures with both machine predicates.

```powershell
paper-search verify-run tests/fixtures/formal_run/capture
paper-search verify-run tests/fixtures/formal_run/replay
paper-search compare-replay tests/fixtures/formal_run/capture tests/fixtures/formal_run/replay
```

Expected result: all commands exit 0.

- [ ] Stop before Phase 3 Task 7 and request explicit authorization for the real dev capture and its maximum frozen run cap.
- [ ] Execute the one authoritative dev capture, immediate replay, and promotion review exactly as Task 7 specifies.
- [ ] Stop before Phase 3 Task 8 and request explicit authorization to consume the validation lock's one live attempt and maximum frozen run cap.
- [ ] Execute the validation capture and one offline replay proof exactly as Task 8 specifies.
- [ ] Run Phase 3 exit verification.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
```

Expected result: all commands exit 0. A valid run with failed quality remains complete evidence and blocks promotion where required.

**Review checkpoint:**

- Every frozen query has one ordered prediction/business record.
- Every hard failure has exactly one supplemental failure.
- Capture and replay business projections are byte-identical.
- Complete status and Gate result are independent.
- Exactly one live validation attempt exists for the promoted lock hash.

---

### Task 5: Execute Phase 4 and pass Gate 5

**Files:**

- Implement exactly: `docs/superpowers/plans/2026-07-30-week1-4-phase4-api-ui-experiments.md`.

**Interfaces:**

Phase 4 consumes the canonical service/result contract and exposes only `GET /health/live`, `GET /health/ready`, `POST /v1/search`, browser static assets, and the stable `paper-search serve` command.

**Steps:**

- [ ] Complete Phase 4 Tasks 1–5 with task-level worker/review cycles.
- [ ] Execute Phase 4 Task 6 in a real browser against a verified replay capture.
- [ ] Stop before the live browser portion and request explicit network/cost authorization.
- [ ] Complete documentation reconciliation only for Gates whose evidence exists.
- [ ] Run final baseline delivery verification.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file paper-search --help
```

Expected result: all commands exit 0.

**Review checkpoint:**

- API error/status mapping is exact.
- Browser UI uses `/v1/search` and the canonical response only.
- Replay is default and bound to one immutable snapshot set.
- Live API requests require lock permission, server permission, and request selection.
- Live HTTP success follows capture seal/validation.
- Main baseline remains fixed-one-round and optional modules remain off.

---

### Task 6: Keep Gate 6 separate from baseline delivery

**Files:**

- Execute only Phase 4 Task 8 access-controlled experiment outputs.
- Modify `configs/base.yaml` only after a separate promotion approval.

**Interfaces:**

Gate 6 consumes three same-configuration dev comparisons plus one selection-only validation comparison and emits `PromotionEvidence`; it never mutates baseline configuration by itself.

**Steps:**

- [ ] Confirm Gates 0–5 are complete before running optional-module comparisons.
- [ ] Run each selected optional experiment three times under identical frozen inputs, snapshots, budgets, and measurement rules.
- [ ] Evaluate the exact promotion policy: median dev macro-F1 delta `>= +0.01`, 1,000-sample bootstrap 95% lower bound `>= -0.005`, and validation macro-F1 drop `<= 0.01`.
- [ ] Present evidence and request a separate promotion decision.
- [ ] Keep every unapproved module default-off.

---

## Gate and Authorization Matrix

| Gate | Engineering work may proceed offline | Real network/cost approval | Human evidence/promotion approval |
| --- | --- | --- | --- |
| 0 — data/policy reconciliation | Yes | No | Required for real V2/pricing/readiness acceptance |
| 1 — replay smoke | Yes | No | No |
| 2 — live smoke/capture | Fake-live only | Required | Review captured evidence |
| 3 — complete dev baseline | Tests/replay only | Required | Required to promote validation lock |
| 4 — frozen validation | Tests/replay only | Required once per lock hash | Validation lock already approved; no tuning |
| 5 — API/UI delivery | Replay yes | Required for live browser demo | Delivery review |
| 6 — optional ablations | Replay where snapshot supports it | Required for any new live capture | Separate promotion decision |

## Final Acceptance Checklist

- [ ] Replay smoke succeeds with one command and no network attempt.
- [ ] Repeated replay has byte-identical canonical business projection.
- [ ] Authorized live smoke captures Qwen, OpenAlex, and Semantic Scholar safely within hard budgets.
- [ ] Dev and validation run directories contain every required query/artifact and pass `verify-run`.
- [ ] Capture/replay pairs pass `compare-replay`.
- [ ] Every authoritative Gate row reports numerator, denominator, value, applicability, and result.
- [ ] Every complete run records config, prompt, source, data, snapshot, usage, and safe environment identities.
- [ ] API readiness is valid for the bound replay mode.
- [ ] Browser UI displays results, provenance, usage, and degradation from the API response.
- [ ] Documentation and public manifest agree with verified evidence.
- [ ] Full pytest, Ruff, and mypy checks pass.
- [ ] Optional modules remain default-off unless separately promoted.

## Execution Choice

The previously selected execution mode is **Subagent-Driven (recommended)**: remain in this task, execute one detailed task at a time with fresh implementation and review agents, and pause only at the explicit real-evidence/network/validation-attempt approvals above.

Alternative: run `superpowers:executing-plans` in a separate execution task with the same phase and approval checkpoints.
