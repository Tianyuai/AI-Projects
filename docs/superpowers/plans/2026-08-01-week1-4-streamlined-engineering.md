# Week 1–4 Streamlined Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan one delivery
> package at a time. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the complete offline Replay baseline, formal fixture flow,
API, browser UI, and default-off experiment wiring with twelve package-level
implementation/review cycles instead of the remaining twenty-three
original-task cycles.

**Architecture:** The approved 2026-07-30 phase plans remain the exact
requirement and interface catalog. This plan changes only execution grouping:
adjacent original tasks are executed in order inside one delivery package,
receive focused tests per substep, and receive one combined specification and
code-quality review at package completion. Real network, cost-bearing,
one-attempt, and optional-promotion evidence is owned by the separate deferred
evidence plan.

**Tech Stack:** Python 3.11+, Pydantic v2, asyncio/httpx, SQLite, FastAPI,
Uvicorn, browser-native HTML/CSS/JavaScript, pytest, Ruff, mypy, Git.

## Global Constraints

- Approved product design:
  `docs/superpowers/specs/2026-07-30-week1-4-integrated-baseline-demo-design.md`.
- Approved execution simplification:
  `docs/superpowers/specs/2026-08-01-week1-4-streamlined-execution-design.md`.
- Exact original requirements and interfaces remain in:
  - `docs/superpowers/plans/2026-07-30-week1-4-phase1-contracts-gate0.md`
  - `docs/superpowers/plans/2026-07-30-week1-4-phase2-runtime-spine.md`
  - `docs/superpowers/plans/2026-07-30-week1-4-phase3-formal-evaluation.md`
  - `docs/superpowers/plans/2026-07-30-week1-4-phase4-api-ui-experiments.md`
- When dispatching a package, concatenate the complete original task sections
  named by that package into its brief. The worker does not read an entire
  phase plan and no original requirement is replaced by a summary.
- Execute original task substeps in their original order. Grouping changes the
  review boundary, not dependency order.
- Every new behavior and defect fix follows RED → GREEN → REFACTOR.
- One implementation agent writes one package. Do not run concurrent writers
  in the shared worktree.
- One independent reviewer issues both `Spec compliance` and `Code quality`
  verdicts for the complete net package diff.
- Consolidate all Critical and Important findings into one fix wave, then
  perform one complete package re-review. Open Critical or Important findings
  still block acceptance.
- Run focused tests for each original-task substep, package checks at package
  completion, and full pytest/Ruff/mypy only at phase exits, Task 5, and final
  delivery.
- Replay is structurally offline. Tests use fixture transports and prices.
  Never infer production prices, readiness, credentials, or Gate evidence.
- Preserve unrelated tracked and untracked files. The known malformed Codex
  checkpoint ref is not repaired or deleted by this plan.
- Stop only at a genuine implementation blocker. Real evidence and network
  authorizations belong to the deferred plan and do not block offline package
  execution.

## Package Protocol

Every package follows:

```text
brief created from exact original task sections
→ implementation agent executes ordered TDD substeps
→ package tests/static checks/diff check
→ one net diff package
→ independent spec + quality review
→ one consolidated fix wave when required
→ complete net re-review
→ package ledger entry
```

Implementation reports are local under `.superpowers/sdd/` and remain
untracked. Review packages are generated from the recorded package base to the
current package head, never from `HEAD~1`.

---

### Package P1-A: Close Gate 0 and Phase 1 Engineering

**Original requirements:** Phase 1 Task 5 and Phase 1 Exit Evidence. Phase 1
Task 6 is not included; it is deferred package E0.

**Files:**

- Modify: `src/paper_search/evaluation/gate0.py`
- Modify: `src/paper_search/evaluation/freeze_schema.py`
- Modify: `src/paper_search/control/pricing.py`
- Modify: `tests/evaluation/test_gate0.py`
- Modify: `tests/evaluation/test_freeze_schema.py`
- Modify: `tests/unit/test_pricing.py`
- Preserve: `.gitignore`
- Preserve unmodified: `tests/evaluation/test_dataset.py`
- Local only: `.superpowers/sdd/week1-4-phase1-task5-report.md`

**Interfaces:**

- Keep `Gate0ReasonCode`, `Gate0ArtifactEvidence`, `Gate0Report`,
  `verify_gate0(...)`, and `write_gate0_report(...)` exactly as approved.
- `IdentifierMap` coverage applies to every `relevant_paper_id` in every
  partition query record. `query_id` is nonempty and unique but is not a paper
  identifier.
- The exact verification boundary and CLI/report safety requirements are
  Section 6 of the streamlined design.

- [ ] **Step 1: Rebuild a clean local Task 5 history**

Verify that the branch is local/unpushed, the tracked worktree is clean, HEAD
contains the approved streamlined design/plan documents, and the base remains
`202b6f4915241f5cd19ab3377fdb551857f40e4c`.

```powershell
git branch -vv
git status --short
git diff --check
git merge-base --is-ancestor 202b6f4915241f5cd19ab3377fdb551857f40e4c HEAD
git diff --stat 202b6f4915241f5cd19ab3377fdb551857f40e4c..HEAD
```

Expected: the ancestry command exits 0; only the recorded local artifacts and
pre-existing unrelated files are untracked. If an upstream already contains
any Task 5 commit, stop instead of rewriting published history.

After explicit sandbox approval for the already user-approved local history
rewrite, use a mixed reset so no working-tree content is deleted:

```powershell
git reset --mixed 202b6f4915241f5cd19ab3377fdb551857f40e4c
git status --short
```

Expected: all net Task 5 code/tests and approved streamlined documents remain
in the working tree; unrelated untracked files remain unchanged.

Commit the approved documents first, then the exact seven-file Task 5 net
implementation:

```powershell
git add docs/superpowers/specs/2026-08-01-week1-4-streamlined-execution-design.md docs/superpowers/plans/2026-08-01-week1-4-streamlined-engineering.md docs/superpowers/plans/2026-08-01-week1-4-deferred-evidence.md docs/superpowers/plans/2026-07-30-week1-4-integrated-baseline-demo-master.md
git commit -m "docs: streamline Week 1-4 execution"
git add .gitignore src/paper_search/control/pricing.py src/paper_search/evaluation/freeze_schema.py src/paper_search/evaluation/gate0.py tests/unit/test_pricing.py tests/evaluation/test_freeze_schema.py tests/evaluation/test_gate0.py
git commit -m "feat: add fail-closed Gate 0 verification"
```

Verify that the report is not reachable from the rewritten branch:

```powershell
git log codex/week1-4-integrated-design --oneline -- .superpowers/sdd/week1-4-phase1-task5-report.md
git ls-files -- .superpowers/sdd/week1-4-phase1-task5-report.md
Test-Path -LiteralPath '.superpowers/sdd/week1-4-phase1-task5-report.md'
```

Expected: both Git queries produce no tracked report; `Test-Path` returns
`True`. Do not run garbage collection and do not modify the malformed
checkpoint ref.

- [ ] **Step 2: Add failing tests for the remaining review findings**

Add real filesystem tests for lexical parent symlink/reparse rejection,
POSIX same-inode content mutation detected by final descriptor re-hash,
descriptor closure when clock/report construction fails, sanitized invalid
CLI arguments and unexpected process-boundary exceptions, and fixed
per-artifact reason attribution.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_gate0.py tests/evaluation/test_freeze_schema.py tests/unit/test_pricing.py -k "symlink or reparse or same_inode or descriptor or cli or reason" -q
```

Expected: the new tests fail for the missing safeguards, not for fixture or
import errors.

- [ ] **Step 3: Implement the Task 5 convergence boundary**

Use the existing bound-artifact implementation. Add final same-descriptor hash
comparison, lexical ancestor checks, deterministic `try/finally` cleanup,
sanitized CLI parsing/error handling, and the approved `relevant_paper_ids`
coverage rule. Do not add a second freeze validator or weaken the existing
Windows/POSIX identity checks.

- [ ] **Step 4: Run P1-A verification**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
$env:PYTHONPATH='src'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_gate0.py tests/evaluation/test_freeze_schema.py tests/evaluation/test_freeze.py tests/evaluation/test_dataset.py tests/unit/test_pricing.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/evaluation/gate0.py src/paper_search/evaluation/freeze_schema.py src/paper_search/control/pricing.py tests/evaluation/test_gate0.py tests/evaluation/test_freeze_schema.py tests/unit/test_pricing.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/gate0.py src/paper_search/evaluation/freeze_schema.py src/paper_search/control/pricing.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
git diff --check
```

Expected: all engineering checks exit 0. The current repository Gate 0 CLI
returns nonzero with only deterministic sanitized blocking reasons; no public
status file changes.

- [ ] **Step 5: Commit and review P1-A**

Commit only the Task 5 code/test fixes. Generate one complete diff from the
post-rewrite document commit through P1-A HEAD, dispatch one security-focused
reviewer, batch all Critical/Important findings, and re-review the complete
range once after fixes.

---

### Package P2-A: Snapshot Store and LLM Capture/Replay

**Original requirements:** Phase 2 Tasks 1–2, concatenated verbatim in the
package brief.

**Files:**

- Create: `src/paper_search/storage/dependency_snapshot.py`
- Create: `src/paper_search/llm/snapshot_adapters.py`
- Create: `tests/unit/test_dependency_snapshot.py`
- Create: `tests/unit/test_llm_snapshot_adapters.py`
- Create: `tests/fixtures/dependency_snapshot_v2/`
- Modify: `src/paper_search/storage/__init__.py`
- Modify: `src/paper_search/llm/client.py`
- Modify: `src/paper_search/llm/__init__.py`
- Modify: `src/paper_search/query/parser.py`
- Modify: related cache, LLM client, and parser tests named by Tasks 1–2.

**Interfaces:** Produce the exact `DependencySnapshotManifestV2`, capture
writer/read-only index, priced live-capture analyzer, and structurally offline
replay analyzer specified by Phase 2 Tasks 1–2.

- [ ] Execute Task 1 RED/GREEN substeps and focused tests exactly as written.
- [ ] Execute Task 2 RED/GREEN substeps and focused tests exactly as written.
- [ ] Run the package verification:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_dependency_snapshot.py tests/unit/test_cache.py tests/unit/test_llm_client.py tests/unit/test_llm_snapshot_adapters.py tests/unit/test_query_parser.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/storage src/paper_search/llm src/paper_search/query/parser.py tests/unit/test_dependency_snapshot.py tests/unit/test_llm_snapshot_adapters.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/storage src/paper_search/llm src/paper_search/query/parser.py
git diff --check
```

- [ ] Commit substeps at coherent checkpoints, then perform one complete P2-A
  net review and one consolidated fix/re-review wave when required.

---

### Package P2-B: Provider Capture/Replay, Routing, and Ledgers

**Original requirements:** Phase 2 Tasks 3–4, concatenated verbatim.

**Files:** All create/modify/test paths named by Phase 2 Tasks 3–4, including
`retrieval/snapshot_adapters.py`, `retrieval/routing.py`, `control/ledger.py`,
Provider modules, budget modules, and their focused tests.

**Interfaces:** Produce priced OpenAlex/Semantic Scholar capture/replay,
deterministic bounded routing, and persistent request/run/project reservation
and settlement semantics.

- [ ] Execute Task 3 RED/GREEN substeps and focused tests exactly as written.
- [ ] Execute Task 4 RED/GREEN substeps and focused tests exactly as written.
- [ ] Run the package verification:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_openalex.py tests/unit/test_semantic_scholar.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_routing.py tests/unit/test_budget.py tests/unit/test_budget_ledger.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/retrieval src/paper_search/control tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_routing.py tests/unit/test_budget_ledger.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/retrieval src/paper_search/control
git diff --check
```

- [ ] Perform one complete P2-B review and one consolidated fix/re-review wave
  when required.

---

### Package P2-C: Application Service and Composition

**Original requirements:** Phase 2 Tasks 5–6, concatenated verbatim.

**Files:** All paths named by Phase 2 Tasks 5–6, centered on
`application/service.py`, `application/modes.py`,
`application/composition.py`, `pipeline/orchestrator.py`, `config.py`, and
their unit/integration tests.

**Interfaces:** Produce the sole `SearchApplicationService.execute()`
production boundary and one lock/mode-bound `CompositionRoot` returning an
`ApplicationBundle` with no concrete adapter exported to consumers.

- [ ] Execute Task 5 RED/GREEN substeps and focused tests exactly as written.
- [ ] Execute Task 6 RED/GREEN substeps and focused tests exactly as written.
- [ ] Run the package verification:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_application_service.py tests/integration/test_orchestrator.py tests/unit/test_response.py tests/integration/test_application_composition.py tests/unit/test_config.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/application src/paper_search/pipeline src/paper_search/config.py tests/unit/test_application_service.py tests/integration/test_application_composition.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/application src/paper_search/pipeline src/paper_search/config.py
git diff --check
```

- [ ] Perform one complete P2-C review and one consolidated fix/re-review wave
  when required.

---

### Package P2-D: Smoke CLI and Phase 2 Exit

**Original requirements:** Phase 2 Task 7 and Phase 2 Exit Evidence. The real
network smoke is deferred package E2.

**Files:** All paths named by Phase 2 Task 7.

**Interfaces:** Produce `CaptureSession`, `ArtifactFactory`, the stable
`paper-search` root and Replay-default `smoke` command. Run Replay Gate 1 and
fake-live Gate 2 only.

- [ ] Execute Task 7 RED/GREEN substeps with fixture transports and prices.
- [ ] Run replay smoke twice with the network tripwire and compare canonical
  business bytes.
- [ ] Run fake-live capture, seal, ledger, replay-lock, and immediate Replay
  equivalence tests.
- [ ] Run Phase 2 exit verification:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/integration/test_smoke_cli.py tests/test_packaging.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file paper-search --help
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_dependency_snapshot.py tests/unit/test_llm_snapshot_adapters.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_budget_ledger.py tests/unit/test_application_service.py tests/integration/test_application_composition.py tests/integration/test_smoke_cli.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
git diff --check
```

- [ ] Perform one complete P2-D review and one consolidated fix/re-review wave.

---

### Package P3-A: Execution Records and Gate Evaluation

**Original requirements:** Phase 3 Tasks 1–2, concatenated verbatim.

**Files:** All paths named by Phase 3 Tasks 1–2, centered on
`evaluation/execution_adapter.py`, `evaluation/business_results.py`,
`evaluation/gates.py`, `evaluation/predictions.py`, and
`evaluation/metrics.py`.

**Interfaces:** Every frozen query produces one ordered prediction/business
record; hard failures produce one linked failure. Produce authoritative formal,
quality, reporting, and promotion Gate evaluation.

- [ ] Execute Task 1 RED/GREEN substeps and focused tests exactly as written.
- [ ] Execute Task 2 RED/GREEN substeps and focused tests exactly as written.
- [ ] Run the package verification:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_execution_adapter.py tests/evaluation/test_business_results.py tests/evaluation/test_predictions.py tests/unit/test_application_service.py tests/evaluation/test_metrics.py tests/evaluation/test_gates.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/evaluation/execution_adapter.py src/paper_search/evaluation/business_results.py src/paper_search/evaluation/predictions.py src/paper_search/evaluation/metrics.py src/paper_search/evaluation/gates.py tests/evaluation/test_execution_adapter.py tests/evaluation/test_business_results.py tests/evaluation/test_metrics.py tests/evaluation/test_gates.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/execution_adapter.py src/paper_search/evaluation/business_results.py src/paper_search/evaluation/predictions.py src/paper_search/evaluation/metrics.py src/paper_search/evaluation/gates.py
git diff --check
```

- [ ] Perform one complete P3-A review and one consolidated fix/re-review wave.

---

### Package P3-B: Atomic Artifacts and Validation Claims

**Original requirements:** Phase 3 Tasks 3–4, concatenated verbatim.

**Files:** All paths named by Phase 3 Tasks 3–4, centered on
`application/artifacts.py`, `evaluation/attempts.py`,
`tests/evaluation/test_artifacts.py`, and `tests/evaluation/test_attempts.py`.

**Interfaces:** Produce failure-safe atomic formal run workspaces and an
irrevocable claim store whose terminal state cannot be removed or reset.

- [ ] Execute Task 3 RED/GREEN substeps and focused tests exactly as written.
- [ ] Execute Task 4 RED/GREEN substeps and focused tests exactly as written.
- [ ] Run the package verification:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_artifacts.py tests/evaluation/test_attempts.py tests/unit/test_budget_ledger.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/application/artifacts.py src/paper_search/evaluation/attempts.py tests/evaluation/test_artifacts.py tests/evaluation/test_attempts.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/application/artifacts.py src/paper_search/evaluation/attempts.py
git diff --check
```

- [ ] Perform one complete P3-B review and one consolidated fix/re-review wave.

---

### Package P3-C: Formal Runner, Validator, and Phase 3 Exit

**Original requirements:** Phase 3 Tasks 5–6 and Phase 3 Exit Evidence. Real
dev and validation captures are deferred packages E3 and E4.

**Files:** All paths named by Phase 3 Tasks 5–6 and the synthetic formal-run
fixtures.

**Interfaces:** `paper-search evaluate`, `verify-run`, and `compare-replay`
share the canonical service/runner/validator and stable exit codes.

- [ ] Execute Task 5 RED/GREEN substeps and focused tests exactly as written.
- [ ] Execute Task 6 RED/GREEN substeps and focused tests exactly as written.
- [ ] Build synthetic capture/replay fixtures and run:

```powershell
paper-search verify-run tests/fixtures/formal_run/capture
paper-search verify-run tests/fixtures/formal_run/replay
paper-search compare-replay tests/fixtures/formal_run/capture tests/fixtures/formal_run/replay
```

Expected: all commands exit 0 without network access.

- [ ] Run Phase 3 exit verification:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_runner.py tests/evaluation/test_validator.py tests/evaluation/test_cli.py tests/evaluation/test_formal_commands.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file paper-search --help
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
git diff --check
```

- [ ] Perform one complete P3-C review and one consolidated fix/re-review wave.

---

### Package P4-A: Typed API and Browser UI

**Original requirements:** Phase 4 Tasks 1–2, concatenated verbatim.

**Files:** All paths named by Phase 4 Tasks 1–2, centered on API routing,
static browser assets, UI application files, and API/UI/packaging tests.

**Interfaces:** Expose typed `GET /health/live`, `GET /health/ready`, and
`POST /v1/search`; serve browser assets whose JavaScript submits only the
canonical request and renders only canonical public response/error fields.

- [ ] Execute Task 1 RED/GREEN substeps and focused tests exactly as written.
- [ ] Execute Task 2 RED/GREEN substeps and focused tests exactly as written.
- [ ] Run the package verification:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/api/test_app.py tests/integration/test_api.py tests/ui/test_app.py tests/test_packaging.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/api src/paper_search/ui tests/api tests/integration/test_api.py tests/ui/test_app.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/api src/paper_search/ui
git diff --check
```

- [ ] Perform one complete P4-A review and one consolidated fix/re-review wave.

---

### Package P4-B: Serve Lifecycle

**Original requirements:** Phase 4 Task 3 verbatim.

**Files:** All paths named by Phase 4 Task 3, including the CLI, composition,
artifact lifecycle, serve tests, process tests, and packaging tests.

**Interfaces:** Produce the stable `paper-search serve` process lifecycle.
Replay is default; live requires all three authorization keys; shutdown closes
clients, ledgers, and temporary capture state.

- [ ] Execute Task 3 RED/GREEN substeps and focused tests exactly as written.
- [ ] Run the package verification:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/cli/test_serve.py tests/integration/test_serve_process.py tests/test_packaging.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file paper-search serve --help
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/cli.py src/paper_search/application/composition.py src/paper_search/application/artifacts.py tests/cli/test_serve.py tests/integration/test_serve_process.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/cli.py src/paper_search/application/composition.py src/paper_search/application/artifacts.py
git diff --check
```

- [ ] Perform one complete P4-B review and one consolidated fix/re-review wave.

---

### Package P4-C: Default-Off Experiment Wiring

**Original requirements:** Phase 4 Task 4 verbatim. Evidence and promotion are
deferred package E6.

**Files:** All paths named by Phase 4 Task 4.

**Interfaces:** Produce explicit experiment identities and async budgeted
provider/rerank stages. Baseline construction remains fixed-one-round and
constructs no optional stage.

- [ ] Execute Task 4 RED/GREEN substeps and focused tests exactly as written.
- [ ] Run the package verification:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/application/test_experiments.py tests/evaluation/test_ablations.py tests/integration/test_orchestrator.py tests/integration/test_evolution_strategies.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/application/experiments.py src/paper_search/graph/provider_stage.py src/paper_search/ranking/llm_stage.py src/paper_search/pipeline/orchestrator.py src/paper_search/evaluation/ablations.py tests/application/test_experiments.py tests/evaluation/test_ablations.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/application/experiments.py src/paper_search/graph/provider_stage.py src/paper_search/ranking/llm_stage.py src/paper_search/pipeline/orchestrator.py
git diff --check
```

- [ ] Perform one complete P4-C review and one consolidated fix/re-review wave.

---

### Package P4-D: Replay Browser Acceptance and Offline Delivery

**Original requirements:** Phase 4 Task 5, Replay-only portion of Task 6, the
evidence-neutral documentation requirements of Task 7, and Phase 4 Exit
Evidence. Live browser evidence and final Gate claims are deferred package E5.

**Files:**

- Modify only when needed for truthful offline documentation:
  `README.md`, `docs/architecture/current-system.md`,
  `docs/demo/demo-runbook.md`,
  `docs/deployment/new-environment-checklist.md`, and
  `docs/limitations-and-risks.md`.
- Do not modify `data/manifest.json` or `data/README.md` without E0.
- No source change unless Replay browser acceptance finds a reproducible defect;
  any defect follows TDD and remains inside this package review.

**Interfaces:** Produce a safe browser acceptance record for one verified
synthetic Replay fixture. Documentation describes implemented commands and
clearly labels E0/E2/E3/E4/E5/E6 evidence as pending.

- [ ] Execute Task 5 RED/GREEN substeps using Replay and deterministic
  fake-live transports only.
- [ ] Run the process E2E suite before browser acceptance:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/e2e/test_dual_mode_serve.py tests/integration/test_serve_process.py tests/integration/test_mock_server_process.py -q
```

- [ ] Start the server on loopback with the verified Phase 3 Replay fixture:

```powershell
paper-search serve --lock tests/fixtures/formal_run/replay/replay.lock.yaml --mode replay --snapshot-manifest tests/fixtures/formal_run/replay/snapshot-manifest.json --capture-output-root runs --host 127.0.0.1 --port 8000
```

- [ ] In a real browser, verify health/readiness, submit one fixture-supported
  Replay request, confirm results/provenance/usage/degradation/config/run IDs,
  repeat it, and confirm byte-equivalent visible business content. Confirm one
  `/v1/search` POST, no direct dependency requests, and no console errors.
- [ ] Stop the server gracefully and verify no client or ledger lock remains.
- [ ] Update documentation without claiming deferred real evidence.
- [ ] Run the contradiction/secret scan:

```powershell
rg -n -i "mock-only|waiting_for_human_label_freeze|authorization:|bearer |api[_-]?key\s*[:=]|private-gate0|^[A-Z]:\\" README.md docs data/manifest.json data/README.md
```

Expected: no credential value, private root, absolute local path, or false Gate
claim. Environment variable names without values may remain in deployment
instructions.

- [ ] Run Phase 4 and final offline verification:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file paper-search --help
git diff --check
```

- [ ] Perform one complete P4-D review and one consolidated fix/re-review wave.

---

## Final Whole-Branch Review

After P4-D passes:

- [ ] Generate one complete review package from the integration branch merge
  base through HEAD.
- [ ] Dispatch the most capable available reviewer using the
  `requesting-code-review` final-review template.
- [ ] Give one fix agent the complete final finding list, run covering tests,
  and re-review the complete branch.
- [ ] Run final pytest, Ruff, mypy, CLI help, Replay fixture validation, and
  browser acceptance evidence again before any completion claim.
- [ ] Use `finishing-a-development-branch` to present integration choices.

Completion of this plan means the offline engineering baseline is complete.
It does not mark deferred evidence packages E0, E2, E3, E4, E5, or E6 as
passed.
