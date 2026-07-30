# Week 1–4 Phase 3 Formal Evaluation and Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make dev and validation evaluation consume `SearchApplicationService`, preserve every query and failure, publish immutable formal run directories, evaluate every authoritative Gate, and prove capture/replay business equivalence.

**Architecture:** An execution adapter converts each `SearchExecutionResult` into one ordered prediction, one access-controlled canonical business result, and an optional linked failure. The formal runner preflights the entire frozen split, reserves the hierarchical ledger, executes one request at a time through the shared service, and stages all artifacts in one run workspace. A validator checks hashes/cardinality/provenance/sanitization/budgets before an atomic publish. Validation live execution additionally consumes a one-attempt claim immediately before first network dispatch.

**Tech Stack:** Python 3.11+, asyncio, Pydantic v2, SQLite, canonical JSON/JSONL, pytest, Ruff, mypy, existing evaluation metrics and identifier-map code.

## Global Constraints

- Phase 1 contracts, V2 freeze authority, lock/pricing/Gate policies and Phase 2 application service, composition, snapshots, and ledgers are prerequisites.
- The formal runner calls only `SearchApplicationService.execute()` from `ApplicationBundle`. It never constructs OpenAlex, Semantic Scholar, an LLM client, or a second orchestrator.
- Every frozen query remains in the ordered denominator. A hard failure creates an empty prediction and exactly one linked failure record.
- Partial success, hard failure, and successful empty retrieval are three distinct states.
- `status="complete"` means artifact-valid only. `gate_result="failed"` is allowed and preserved.
- Live formal runs require committed tracked source/config state, a Gate-0-approved V2 split, known/reconciled actual cost, and current authorized readiness.
- The dev capture is the authoritative scored dev run; immediate replay is reproducibility evidence, not a second experiment.
- The validation capture is the one authoritative scored validation run for its lock hash. A claim is never deleted or reset.
- Copy exact input lock bytes to `config.lock.yaml`; do not reserialize or mutate the validation lock.
- Run and claim atomic operations use sibling paths on the same filesystem.
- Raw snapshots, gated queries, gold labels, predictions, failures, and `business-results.jsonl` are access-controlled. Safe public reporting is aggregate only.
- `business-results.jsonl` is an explicit implementation detail required by the approved byte-equality definition; it stores the canonical projection whose hash appears in the execution envelope.
- Follow red-green-refactor and commit after each focused task.

---

## File Structure and Ownership

### Create

- `src/paper_search/evaluation/execution_adapter.py` — success/failure to ordered evaluation records.
- `src/paper_search/evaluation/business_results.py` — canonical transport-free projection and comparison.
- `src/paper_search/evaluation/gates.py` — formal-validity, baseline-quality, reporting, and promotion evaluation.
- `src/paper_search/evaluation/attempts.py` — irrevocable validation-attempt claim store.
- `src/paper_search/evaluation/validator.py` — formal run-directory validator.
- `tests/evaluation/test_execution_adapter.py`
- `tests/evaluation/test_business_results.py`
- `tests/evaluation/test_gates.py`
- `tests/evaluation/test_attempts.py`
- `tests/evaluation/test_artifacts.py`
- `tests/evaluation/test_validator.py`
- `tests/evaluation/test_formal_commands.py`
- `tests/fixtures/formal_run/` — synthetic access-safe valid and invalid capture/replay trees.

### Modify

- `src/paper_search/evaluation/dataset.py`
- `src/paper_search/evaluation/predictions.py`
- `src/paper_search/evaluation/metrics.py`
- `src/paper_search/evaluation/runner.py`
- `src/paper_search/application/artifacts.py`
- `src/paper_search/cli.py`
- `tests/evaluation/test_predictions.py`
- `tests/evaluation/test_metrics.py`
- `tests/evaluation/test_runner.py`
- `tests/evaluation/test_cli.py`
- `tests/unit/test_budget_ledger.py`

### Formal Run Tree

```text
runs/<run_id>/
├── run.json
├── config.lock.yaml
├── replay.lock.yaml
├── snapshot-manifest.json
├── snapshots/                 # capture mode only; manifest-declared exact bytes
├── predictions.jsonl
├── executions.jsonl
├── business-results.jsonl
├── metrics.json
├── usage.json
└── failures.jsonl
```

Replay runs retain the same metadata tree but do not duplicate `snapshots/`; their copied manifest contains artifact-root-relative paths to the verified capture bytes. `replay.lock.yaml` and `snapshot-manifest.json` are exact verified copies/bindings for replay and emitted artifacts for capture. Failed/interrupted workspaces publish beneath `runs/_failed/<run_id>/` and can never pass `verify-run`.

---

### Task 1: Define ordered execution, failure, and business-result records

**Files:**

- Create: `src/paper_search/evaluation/execution_adapter.py`
- Create: `src/paper_search/evaluation/business_results.py`
- Create: `tests/evaluation/test_execution_adapter.py`
- Create: `tests/evaluation/test_business_results.py`
- Modify: `src/paper_search/evaluation/dataset.py`
- Modify: `src/paper_search/evaluation/predictions.py`
- Modify: `tests/evaluation/test_predictions.py`
- Modify: `src/paper_search/application/service.py`
- Modify: `tests/unit/test_application_service.py`

**Interfaces:**

```python
class EvaluationFailureRecord(DomainModel):
    schema_version: Literal["evaluation-failure-v1"]
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    error_code: SearchErrorCode
    retryable: bool
    stop_reason: NonEmptyStr
    usage: UsageActual
    dependency_error_codes: list[DependencyErrorCode]
    diagnostics: list[DependencyDiagnostic]
    diagnostics_sha256: Sha256


class EvaluationExecutionRecord(DomainModel):
    schema_version: Literal["evaluation-execution-v1"]
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    outcome_kind: Literal["success", "failure"]
    business_result_sha256: Sha256
    usage: UsageActual
    diagnostics: list[DependencyDiagnostic]
    is_partial: bool
    planner_status: PlannerStatus | None
    planner_fallback: bool
    stop_reason: NonEmptyStr


class BusinessResultRecord(DomainModel):
    schema_version: Literal["business-result-v1"]
    query_id: NonEmptyStr
    query_analysis: QueryAnalysisResult | None
    selected_paper_ids: list[NonEmptyStr]
    high_relevance: list[RankedPaper]
    partial_relevance: list[RankedPaper]
    citation_edges: list[ResolvedCitationEdge]
    is_partial: bool
    planner_status: PlannerStatus | None
    planner_fallback: bool
    warnings: list[NonEmptyStr]
    stop_reason: NonEmptyStr
    hard_failure_code: SearchErrorCode | None


class AdaptedExecution(DomainModel):
    prediction: InternalPredictionRecord
    execution: EvaluationExecutionRecord
    business_result: BusinessResultRecord
    failure: EvaluationFailureRecord | None


def adapt_execution(
    *,
    expected_query_id: str,
    result: SearchExecutionResult,
) -> AdaptedExecution: ...


def canonical_business_result_bytes(record: BusinessResultRecord) -> bytes: ...
def business_result_sha256(record: BusinessResultRecord) -> Sha256: ...
def compare_business_results(
    capture: Sequence[BusinessResultRecord],
    replay: Sequence[BusinessResultRecord],
) -> None: ...
```

Canonical JSON uses UTF-8, `ensure_ascii=False`, sorted object keys, no insignificant whitespace, and a final newline per JSONL record. List order is semantic and preserved. Run IDs, timestamps, request IDs, execution mode, snapshot refs, cache/live usage differences, and transport diagnostics are excluded.

For hard failure:

- prediction uses the expected query ID and `selected_paper_ids=[]`;
- execution preserves the safe per-dependency diagnostics, cache/snapshot references, usage, latency, run identity, and business hash;
- business result uses `query_analysis=None`, empty evidence collections, `is_partial=false`, the safe stop reason, and the stable hard-failure code;
- exactly one `EvaluationFailureRecord` is emitted.

**Steps:**

- [ ] Add failing tests for success, partial success, successful empty retrieval, hard failure, mismatched query ID, failure diagnostic hashing, and exact field rejection.
- [ ] Add byte-level tests proving excluded transport fields do not alter business bytes while any selected ID/evidence/warning/stop-reason change does.
- [ ] Add comparison tests for missing, duplicate, extra, reordered, and unequal query records.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_execution_adapter.py tests/evaluation/test_business_results.py tests/evaluation/test_predictions.py tests/unit/test_application_service.py -q
```

Expected initial result: FAIL because record and canonical projection types are absent.

- [ ] Implement strict record models and adaptation.
- [ ] Move the Phase 2 private business serializer into `business_results.py`; update `SearchApplicationService` to hash this exact projection.
- [ ] Preserve existing official `PredictionRecord` adaptation without adding extra fields to it.
- [ ] Ensure diagnostics hashes use sanitized canonical diagnostics only.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_execution_adapter.py tests/evaluation/test_business_results.py tests/evaluation/test_predictions.py tests/unit/test_application_service.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/evaluation/execution_adapter.py src/paper_search/evaluation/business_results.py src/paper_search/evaluation/predictions.py tests/evaluation/test_execution_adapter.py tests/evaluation/test_business_results.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/execution_adapter.py src/paper_search/evaluation/business_results.py src/paper_search/evaluation/predictions.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/evaluation/execution_adapter.py src/paper_search/evaluation/business_results.py src/paper_search/evaluation/dataset.py src/paper_search/evaluation/predictions.py src/paper_search/application/service.py tests/evaluation/test_execution_adapter.py tests/evaluation/test_business_results.py tests/evaluation/test_predictions.py tests/unit/test_application_service.py
git commit -m "feat: add canonical evaluation execution records"
```

---

### Task 2: Implement authoritative metric and Gate evaluation

**Files:**

- Create: `src/paper_search/evaluation/gates.py`
- Create: `tests/evaluation/test_gates.py`
- Modify: `src/paper_search/evaluation/metrics.py`
- Modify: `tests/evaluation/test_metrics.py`

**Interfaces:**

```python
class MeasureValue(DomainModel):
    numerator: Decimal
    denominator: Decimal
    value: Decimal | None


class GateCheck(DomainModel):
    rule_id: NonEmptyStr
    classification: Literal[
        "formal_validity", "baseline_quality", "reporting_only", "promotion"
    ]
    applies: bool
    measure: MeasureValue
    operator: Literal["eq", "gt", "gte", "lte"]
    threshold: Decimal | int
    passed: bool | None


class GateEvaluation(DomainModel):
    split: Literal["dev", "validation"]
    formal_valid: bool
    quality_passed: bool
    gate_result: Literal["passed", "failed"]
    checks: list[GateCheck]


def evaluate_gates(
    *,
    frozen_queries: Sequence[EvaluationQuery],
    predictions: Sequence[PredictionRecord],
    failures: Sequence[EvaluationFailureRecord],
    metrics: EvaluationResult,
    audit_measures: Mapping[str, MeasureValue],
    ledger_report: LedgerReport,
    policy: QualityGatePolicy,
) -> GateEvaluation: ...
```

Every check reports numerator, denominator, value, configured rule, applicability, and result. `denominator=0` is invalid for fuzzy merge and any rule whose policy requires a nonempty audit. Missing predictions are no longer silently treated as adequate formal input even though score calculation can represent them as empty.

**Steps:**

- [ ] Add failing tests for every formal-validity and baseline-quality row from `configs/quality_gates_v1.yaml`.
- [ ] Add boundary tests at 0.99/0.90/0.95/0.98/0.02 and strict `> 0`.
- [ ] Add tests proving reporting-only F1 does not fail a Gate, while missing prediction/failure cardinality does.
- [ ] Add tests proving a complete run may have `gate_result="failed"`.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_metrics.py tests/evaluation/test_gates.py -q
```

Expected initial result: FAIL because `gates.py` and strict cardinality checks are absent.

- [ ] Extend metric output with explicit numerators/denominators needed by Gate evaluation without changing existing relevance formulas.
- [ ] Implement deterministic rule ordering from the frozen policy.
- [ ] Separate `formal_valid`, `quality_passed`, and reporting-only calculations.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_metrics.py tests/evaluation/test_gates.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/evaluation/metrics.py src/paper_search/evaluation/gates.py tests/evaluation/test_metrics.py tests/evaluation/test_gates.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/metrics.py src/paper_search/evaluation/gates.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/evaluation/metrics.py src/paper_search/evaluation/gates.py tests/evaluation/test_metrics.py tests/evaluation/test_gates.py
git commit -m "feat: evaluate formal and quality gates"
```

---

### Task 3: Expand the atomic artifact workspace and run schema

**Files:**

- Modify: `src/paper_search/application/artifacts.py`
- Create: `tests/evaluation/test_artifacts.py`

**Interfaces:**

```python
class RunManifest(DomainModel):
    schema_version: Literal["formal-run-v1"]
    run_id: NonEmptyStr
    status: Literal["incomplete", "failed", "interrupted", "complete"]
    gate_result: Literal["passed", "failed", "not_applicable"]
    execution_mode: SearchMode
    split: Literal["smoke", "dev", "validation"]
    frozen_manifest_sha256: Sha256
    partition_sha256: Sha256
    identifier_map_sha256: Sha256
    source_git_sha: NonEmptyStr
    tracked_source_dirty: bool
    config_hash: Sha256
    input_lock_sha256: Sha256
    prompt_version: NonEmptyStr
    snapshot_set_id: NonEmptyStr
    snapshot_manifest_sha256: Sha256
    experiment_name: NonEmptyStr
    optional_modules: dict[NonEmptyStr, bool]
    started_at: datetime
    ended_at: datetime | None
    readiness_summary: list[DependencyStatus]
    failure_count: NonNegativeInt


class FormalRunWorkspace:
    def write_prediction(self, record: InternalPredictionRecord) -> None: ...
    def write_execution(self, record: EvaluationExecutionRecord) -> None: ...
    def write_business_result(self, record: BusinessResultRecord) -> None: ...
    def write_failure(self, record: EvaluationFailureRecord) -> None: ...
    def write_metrics(self, metrics: EvaluationResult) -> None: ...
    def write_usage(self, report: LedgerReport) -> None: ...
    def finalize(
        self,
        *,
        gate_evaluation: GateEvaluation,
        replay_lock: ReplayLock,
        snapshot_manifest: DependencySnapshotManifestV2,
    ) -> Path: ...
    def fail(self, reason: SearchErrorCode) -> Path: ...
    def interrupt(self) -> Path: ...
```

Publication rules:

- working directory: `runs/.incomplete-<run_id>-<nonce>/`;
- complete destination: `runs/<run_id>/`;
- failed/interrupted destination: `runs/_failed/<run_id>/`;
- destination must not exist;
- all files are written and closed, validator passes, then the directory is renamed once;
- exact input lock bytes are copied before execution;
- capture snapshot sealing precedes `status="complete"`;
- replay copies and validates its bound manifest/lock; it does not emit a new snapshot identity.

**Steps:**

- [ ] Add failing tests for exact tree, exact lock bytes, append ordering, duplicate query rejection, finalize-before-seal, destination collision, cross-device root rejection, interrupted state, failed state, and complete-vs-Gate-failed independence.
- [ ] Add failure-injection tests at each file write and before/after validator call; no partial directory may appear as `runs/<run_id>/`.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_artifacts.py -q
```

Expected initial result: FAIL because the Phase 2 smoke artifact factory lacks formal run support.

- [ ] Implement formal workspace state transitions and atomic JSON/JSONL writes.
- [ ] Fsync file and parent-directory handles where supported; retain deterministic Windows-compatible replacement behavior.
- [ ] Keep access-controlled artifacts out of safe stdout and public report helpers.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_artifacts.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/application/artifacts.py tests/evaluation/test_artifacts.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/application/artifacts.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/application/artifacts.py tests/evaluation/test_artifacts.py
git commit -m "feat: add atomic formal run workspace"
```

---

### Task 4: Add irrevocable validation-attempt claims

**Files:**

- Create: `src/paper_search/evaluation/attempts.py`
- Create: `tests/evaluation/test_attempts.py`
- Modify: `tests/unit/test_budget_ledger.py`

**Interfaces:**

```python
class ValidationAttemptClaim(DomainModel):
    schema_version: Literal["validation-attempt-v1"]
    validation_lock_sha256: Sha256
    run_id: NonEmptyStr
    claimed_at: datetime
    state: Literal["claimed", "complete", "failed", "interrupted"]
    completed_at: datetime | None
    incident_ref: NonEmptyStr | None


class ValidationAttemptStore:
    def claim(
        self,
        *,
        validation_lock_sha256: Sha256,
        run_id: str,
        claimed_at: datetime,
    ) -> ValidationAttemptClaim: ...

    def transition(
        self,
        *,
        validation_lock_sha256: Sha256,
        target: Literal["complete", "failed", "interrupted"],
        completed_at: datetime,
        incident_ref: str | None = None,
    ) -> ValidationAttemptClaim: ...
```

Claim path is exactly:

```text
validation-attempts/<64-lowercase-lock-digest>.claim
```

Creation uses exclusive create-if-absent. Transitions write a sibling replacement atomically. There is no delete/reset API. A superseding human-approved lock hash is the only retry mechanism.

**Steps:**

- [ ] Add failing tests for exclusive concurrent claim, exact initial state, every legal terminal transition, illegal second transition, restart read, malformed claim, replay bypass, and no delete/reset public method.
- [ ] Add an orchestration-order test proving offline preflight and run-budget reserve occur before claim, and claim occurs immediately before first live dependency dispatch.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_attempts.py tests/unit/test_budget_ledger.py -q
```

Expected initial result: FAIL because attempt claims do not exist.

- [ ] Implement same-filesystem exclusive claim creation and terminal transitions.
- [ ] Map duplicate claims to `validation_attempt_conflict`.
- [ ] Require a different human-approved validation lock hash plus a nonempty incident reference before any replacement live validation attempt.
- [ ] Ensure process cancellation/`KeyboardInterrupt` can transition a created claim to `interrupted` from the runner's outer boundary.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_attempts.py tests/unit/test_budget_ledger.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/evaluation/attempts.py tests/evaluation/test_attempts.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/attempts.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/evaluation/attempts.py tests/evaluation/test_attempts.py tests/unit/test_budget_ledger.py
git commit -m "feat: add irrevocable validation attempt claims"
```

---

### Task 5: Rewrite formal evaluation around `SearchApplicationService`

**Files:**

- Modify: `src/paper_search/evaluation/runner.py`
- Modify: `tests/evaluation/test_runner.py`

**Interfaces:**

```python
class EvaluationRunRequest(DomainModel):
    split: Literal["dev", "validation"]
    mode: SearchMode
    lock_path: Path
    output_root: Path
    snapshot_manifest_path: Path | None
    network_authorized: bool


class EvaluationRunResult(DomainModel):
    run_id: NonEmptyStr
    run_path: Path
    status: Literal["complete", "failed", "interrupted"]
    gate_result: Literal["passed", "failed", "not_applicable"]


async def run_evaluation(
    request: EvaluationRunRequest,
    *,
    composition_root: type[CompositionRoot] = CompositionRoot,
    attempt_store_factory: Callable[[Path], ValidationAttemptStore],
    clock: Callable[[], datetime],
) -> EvaluationRunResult: ...
```

Execution order:

1. Read exact lock bytes and validate lock kind/source SHA/clean tracked state.
2. Validate V2 manifest, split, ordered query IDs/count/hash, identifier map, pricing/Gate policy, and replay manifest if applicable.
3. Validate current authorized readiness for live; no network dispatch during replay preflight.
4. Create run workspace and copy exact lock bytes.
5. Reserve run/project budget.
6. For validation live only, create the attempt claim.
7. For each ordered query: reserve query ledger, call `service.execute()` once, settle actual usage, adapt result, append one prediction/business result and optional failure, continue after query-scoped hard failure.
8. Stop on integrity/sanitization/unaccounted-usage failure.
9. Compute metrics/Gates, seal capture if live, validate, and atomically publish.
10. Transition a validation claim to the matching terminal state.

`asyncio.CancelledError`, `KeyboardInterrupt`, and process interruption are never converted into query failures.

**Steps:**

- [ ] Replace existing tests that directly inject one OpenAlex Provider with bundle/service fakes.
- [ ] Add failing tests for full preflight, ordered success, query-scoped exception continuation, hard failure empty prediction, integrity-stop behavior, exact settlement, cancellation/interruption, capture sealing, replay non-network behavior, and validation claim order.
- [ ] Retain tests for normalization, identifier map, predictions, and metrics by routing them through the new adapter boundaries.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_runner.py -q
```

Expected initial result: FAIL because the current runner directly constructs an OpenAlex-only path and aborts the batch on exceptions.

- [ ] Rewrite orchestration around `CompositionRoot.compose()` and `SearchApplicationService.execute()`.
- [ ] Remove authoritative reliance on `SQLiteResponseCache` and `SearchProvider` from the runner signature.
- [ ] Preserve `process_candidates()` only if another legacy test imports it; mark it non-authoritative in its docstring.
- [ ] Ensure failed queries remain in prediction order and every query ledger settles exactly once.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_runner.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/evaluation/runner.py tests/evaluation/test_runner.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/runner.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/evaluation/runner.py tests/evaluation/test_runner.py
git commit -m "refactor: run formal evaluation through application service"
```

---

### Task 6: Add formal run validation and command contracts

**Files:**

- Create: `src/paper_search/evaluation/validator.py`
- Create: `tests/evaluation/test_validator.py`
- Create: `tests/evaluation/test_formal_commands.py`
- Create: `tests/fixtures/formal_run/`
- Modify: `src/paper_search/cli.py`
- Modify: `tests/evaluation/test_cli.py`

**Interfaces:**

```python
class ValidationIssue(DomainModel):
    code: NonEmptyStr
    artifact: SafeRelativePath
    detail: NonEmptyStr


class RunValidationResult(DomainModel):
    valid: bool
    run_id: NonEmptyStr | None
    issues: list[ValidationIssue]


def validate_run_directory(path: Path) -> RunValidationResult: ...
def verify_run_command(path: Path) -> int: ...
def compare_replay_command(capture_path: Path, replay_path: Path) -> int: ...
```

Validator checks:

- exact required file set and no path escape;
- `run.json.status=="complete"`;
- exact lock bytes/hash and matching source/config/data/prompt identities;
- snapshot manifest bytes/hash/set identity and every response hash/path;
- capture mode owns the manifest-declared `snapshots/` bytes; replay mode verifies the artifact-root-relative capture bytes without copying or mutating them;
- exactly one ordered prediction and business record for each frozen query;
- exactly one ordered execution record for each frozen query, preserving safe diagnostics and snapshot/cache references;
- exactly one failure for each hard-failed query and none otherwise;
- business-record per-line hashes match envelope evidence;
- metrics and Gate policy identities;
- request/run/project ledger closure and caps;
- zero credential/authorization/raw-error/private-path sanitization findings;
- capture/replay mode-specific invariants.

CLI additions:

```text
paper-search evaluate ...
paper-search verify-run RUN_DIRECTORY
paper-search compare-replay CAPTURE_RUN REPLAY_RUN
```

Exit codes are stable:

- `0` success/valid/equivalent;
- `2` invalid command input or config mismatch;
- `3` artifact invalid;
- `4` replay mismatch;
- `5` Gate executed but quality failed (run remains complete);
- `130` interrupted.

**Steps:**

- [ ] Add a synthetic valid capture/replay pair and failing fixture for every validator category.
- [ ] Add failing command tests for help, bad input, complete Gate-failed run, invalid run, mismatch, and safe output.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_validator.py tests/evaluation/test_cli.py tests/evaluation/test_formal_commands.py -q
```

Expected initial result: FAIL because validator and commands are absent.

- [ ] Implement validation using bytes read once per file where identity is checked.
- [ ] Add CLI subcommands to the existing root parser without duplicating runner logic.
- [ ] Compare ordered `business-results.jsonl` canonical bytes, not transport metadata or only selected IDs.
- [ ] Keep command output to run ID/path, validity, Gate result, and safe issue codes.
- [ ] Re-run focused tests and full engineering verification.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_validator.py tests/evaluation/test_cli.py tests/evaluation/test_formal_commands.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file paper-search --help
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/evaluation/validator.py src/paper_search/cli.py tests/evaluation/test_validator.py tests/evaluation/test_cli.py tests/evaluation/test_formal_commands.py tests/fixtures/formal_run
git commit -m "feat: validate and compare formal runs"
```

---

### Task 7: Execute Gate 3 dev capture and promote the validation lock

**Files:**

- Access-controlled create: `runs/<dev-capture-run>/`
- Access-controlled create: `runs/<dev-replay-run>/`
- Access-controlled create: `validation.lock.yaml`
- No source/config edits during the authoritative run.

**Interfaces:**

Gate 3 produces exactly one reviewed tuple:

```text
(verified dev capture path,
 verified dev replay path,
 compare-replay exit 0,
 GateEvaluation with formal_valid=true and quality_passed=true,
 content-addressed validation lock)
```

No validation lock is emitted when any tuple member is absent or false.

**Preconditions:**

- Gate 0 report passes.
- Gate 1 replay smoke and Gate 2 authorized live smoke/replay pass.
- Main baseline candidate lock binds committed clean source, V2 dev split, approved pricing/Gate policy, budget, capture policy, and all optional modules off.
- Authorized readiness for Qwen/OpenAlex/Semantic Scholar is less than 15 minutes old.
- The user explicitly authorizes the live dev run and expected maximum CNY cap.

**Steps:**

- [ ] Verify clean tracked source/config state and capture the exact SHA.

```powershell
git status --short
git rev-parse HEAD
git diff --check
```

Expected result: no tracked source/config modifications and all commands exit 0.

- [ ] Run one live dev capture.

```powershell
paper-search evaluate --lock candidate.lock.yaml --split dev --mode live --output-root runs --allow-network
```

Expected result: exit `0` for a passing complete run or `5` for a complete quality-failed run. Any other exit blocks promotion.

- [ ] Verify the capture.

```powershell
paper-search verify-run runs/dev-capture
```

Expected result required for promotion: exit 0.

- [ ] Run immediate replay using the emitted lock and exact manifest.

```powershell
paper-search evaluate --lock runs/dev-capture/replay.lock.yaml --split dev --mode replay --output-root runs --snapshot-manifest runs/dev-capture/snapshot-manifest.json
paper-search verify-run runs/dev-replay
paper-search compare-replay runs/dev-capture runs/dev-replay
```

Expected result required for promotion: all commands exit 0.

- [ ] Review every Gate check, failure, cost, latency, cache, and identifier-map coverage result. Do not use validation data.
- [ ] If formal validity or any baseline-quality row fails, preserve both runs, create no validation lock, and return to an approved candidate-lock change process.
- [ ] If all applicable checks pass, emit a Git-external content-addressed `validation.lock.yaml` binding the current committed source SHA and validation split. Record the dev run ID/hash and human approval reference.
- [ ] Validate the lock offline and confirm no validation-attempt claim exists for its hash.

---

### Task 8: Execute Gate 4 one-attempt validation and replay proof

**Files:**

- Access-controlled create: `validation-attempts/<validation-lock-sha256>.claim`
- Access-controlled create: `runs/<validation-capture-run>/`
- Access-controlled create: `runs/<validation-replay-run>/`
- No source/config edits between validation lock promotion and live validation.

**Interfaces:**

Gate 4 consumes exactly one promoted validation lock hash and produces:

```text
(terminal validation-attempt claim,
 authoritative validation capture result,
 optional complete capture path,
 one verified replay path when capture is complete,
 compare-replay exit 0 when capture is complete)
```

The claim terminal state is authoritative even when no complete capture exists.

**Preconditions:**

- Task 7 produced an approved validation lock.
- Source SHA and every bound input identity still match.
- No prior claim exists for the exact validation lock hash.
- Authorized readiness is less than 15 minutes old.
- Run budget is reserved before claim creation.
- The user explicitly authorizes the one live validation attempt and expected maximum CNY cap.

**Steps:**

- [ ] Run the single live validation capture. The command creates the claim immediately before first network dispatch.

```powershell
paper-search evaluate --lock validation.lock.yaml --split validation --mode live --output-root runs --allow-network
```

Expected result: exit `0` for a complete Gate-passing run or `5` for a complete quality-failed run. The claim becomes terminal even when execution fails or is interrupted.

- [ ] Verify the capture whenever it reached `complete`.

```powershell
paper-search verify-run runs/validation-capture
```

Expected result for formal validity: exit 0.

- [ ] Run exactly one offline replay proof.

```powershell
paper-search evaluate --lock runs/validation-capture/replay.lock.yaml --split validation --mode replay --output-root runs --snapshot-manifest runs/validation-capture/snapshot-manifest.json
paper-search verify-run runs/validation-replay
paper-search compare-replay runs/validation-capture runs/validation-replay
```

Expected result required for reproducibility evidence: all commands exit 0.

- [ ] Preserve the validation result regardless of quality outcome. Do not change parameters or issue a second live run for the same lock hash.
- [ ] Produce only the sanitized `run.json`, aggregate metrics, aggregate usage, failure counts, and Gate evaluation for review; keep labels, per-query predictions/failures, snapshots, and business records access-controlled.

---

## Phase 3 Exit Evidence

- `paper-search evaluate`, `verify-run`, and `compare-replay` share one runner/validator implementation and stable exit codes.
- Every frozen query has exactly one ordered prediction and business record; each hard failure has exactly one supplemental failure.
- Every complete run is atomically published and can remain complete with a failed quality Gate.
- Dev and validation capture/replay pairs pass byte-level business projection comparison.
- The validation attempt claim proves exactly one live attempt for the promoted lock hash.
