# Week 1–4 Phase 1 Contracts, Locks, and Gate 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the shared application, lock, pricing, data-freeze, and Gate 0 contracts so later Replay/Live, evaluation, API, and UI work has one fail-closed foundation.

**Architecture:** Extend the existing strict frozen Pydantic domain models rather than creating parallel product/evaluation types. Add a small `application` contract and lock layer, a deterministic Decimal-based pricing policy, explicit V1/V2 freeze dispatch, and a read-only Gate 0 reconciler. Gate 0 may publish sanitized status only after every referenced private artifact and approval hash verifies.

**Tech Stack:** Python 3.11+, Pydantic v2, PyYAML, pytest, Ruff, mypy, atomic filesystem primitives already used by `evaluation.freeze`.

## Global Constraints

- Approved design: `docs/superpowers/specs/2026-07-30-week1-4-integrated-baseline-demo-design.md`.
- Preserve `DomainModel(extra="forbid", frozen=True)` for all normative models.
- Reuse `QueryAnalysisResult`, `RankedPaper`, `ResolvedCitationEdge`, `UsageActual`, and `ErrorDetail`; do not define API- or evaluation-specific copies.
- Use `Decimal` for monetary arithmetic and canonical JSON strings for monetary serialization. Never hash YAML/Python floating-point values.
- A production pricing policy must come from operator-verified source evidence. Unit tests use clearly named fixture rates; implementation must not infer production prices.
- V1 freeze evidence remains readable but never authorizes an integrated formal run.
- Serialized artifact paths are POSIX-relative and confined beneath a separately bound root. Never serialize Windows absolute paths into a lock or manifest.
- Gate 0 is read-only until every check passes. On failure, `data/manifest.json`, `README.md`, `data/README.md`, and `PRD.md` remain byte-for-byte unchanged.
- Keep public reports free of queries, labels, credentials, authorization headers, raw Provider/LLM payloads, and private absolute paths.
- Follow red-green-refactor. Run the focused failing test before implementation, then the same command after implementation.
- Commit only the files named by the current task. Preserve unrelated untracked files.

---

## File Structure and Ownership

### Create

- `src/paper_search/application/__init__.py` — exports only Phase 1 contracts and locks.
- `src/paper_search/application/contracts.py` — execution-envelope, diagnostics, stable error, snapshot, and validated primitive types.
- `src/paper_search/application/locks.py` — exact candidate, validation, and replay lock models and loaders.
- `src/paper_search/control/pricing.py` — versioned policy schema, canonical identity, and actual-cost valuation.
- `src/paper_search/evaluation/freeze_schema.py` — exact V1/V2 schema dispatch and explicit migration.
- `src/paper_search/evaluation/gate0.py` — sanitized, deterministic Gate 0 verifier and report.
- `configs/quality_gates_v1.yaml` — executable classification and formulas from the approved design.
- `tests/application/test_contracts.py`
- `tests/application/test_locks.py`
- `tests/unit/test_pricing.py`
- `tests/evaluation/test_freeze_schema.py`
- `tests/evaluation/test_gate0.py`
- `tests/fixtures/application/candidate.lock.yaml`
- `tests/fixtures/application/validation.lock.yaml`
- `tests/fixtures/application/replay.lock.yaml`
- `tests/fixtures/pricing/pricing-policy-test-v1.yaml`
- `tests/fixtures/evaluation/freeze_v2/` — synthetic non-gated V2 manifest, approval, partitions, identifier map, and readiness report.

### Modify

- `src/paper_search/domain/models.py` — shared money/hash/path/status types and expanded canonical response.
- `src/paper_search/api/contracts.py` — canonical request and readiness contracts.
- `src/paper_search/api/service.py` — temporary mock compatibility only.
- `src/paper_search/api/app.py` — temporary schema compatibility only; final HTTP mapping is Phase 4.
- `src/paper_search/pipeline/response.py` — populate required metadata without fabricating evidence.
- `src/paper_search/config.py` — typed runtime settings and policy identities.
- `configs/base.yaml` — `runtime.allow_live: false` and reproducible baseline settings.
- `src/paper_search/evaluation/freeze.py` — dispatch to V1/V2 parsing while preserving hardened publication behavior.
- `data/manifest.example.json` — sanitized V2 shape replacing the current unversioned example.
- `tests/api/test_contracts.py`
- `tests/api/test_service.py`
- `tests/api/test_app.py`
- `tests/unit/test_response.py`
- `tests/unit/test_config.py`
- `tests/test_config.py`
- `tests/unit/test_budget.py`
- `tests/evaluation/test_freeze.py`
- `tests/evaluation/test_dataset.py`
- `.gitignore` — ignore `/runs/`, `/validation-attempts/`, and private local Gate 0 input roots.

### Explicitly Deferred

- `application/service.py`, `application/composition.py`, snapshot adapters, run/project ledgers, and smoke execution belong to Phase 2.
- Evaluation run publication, validation-attempt claims, and formal metric Gates belong to Phase 3.
- Final HTTP error routing, browser UI, server lifecycle, and experiments belong to Phase 4.

---

### Task 1: Add canonical primitives and application contracts

**Files:**

- Create: `src/paper_search/application/__init__.py`
- Create: `src/paper_search/application/contracts.py`
- Create: `tests/application/test_contracts.py`
- Modify: `src/paper_search/domain/models.py`
- Modify: `src/paper_search/api/contracts.py`
- Modify: `src/paper_search/pipeline/response.py`
- Modify: `src/paper_search/api/service.py`
- Modify: `src/paper_search/api/app.py`
- Modify: `tests/api/test_contracts.py`
- Modify: `tests/api/test_service.py`
- Modify: `tests/api/test_app.py`
- Modify: `tests/unit/test_response.py`
- Modify: `tests/unit/test_models.py`

**Interfaces:**

```python
Sha256 = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
SafeRelativePath = Annotated[str, AfterValidator(validate_safe_relative_path)]
MoneyCny = Annotated[Decimal, Field(ge=Decimal("0"), decimal_places=6)]

SearchMode = Literal["replay", "live"]
DependencyName = Literal["llm", "openalex", "semantic_scholar"]
DependencyState = Literal["ready", "replayed", "degraded", "failed"]
PlannerStatus = Literal["primary", "repaired", "rules_fallback"]
DependencyErrorCode = Literal[
    "timeout",
    "network_error",
    "rate_limited",
    "server_error",
    "authentication_error",
    "invalid_request",
    "invalid_response",
    "invalid_record",
    "missing_record",
    "empty_response",
    "invalid_json",
    "budget_exhausted",
    "provider_error",
]
SearchErrorCode = Literal[
    "invalid_request",
    "live_not_authorized",
    "config_mismatch",
    "snapshot_unavailable",
    "budget_exhausted",
    "dependency_failure",
    "integrity_failure",
    "validation_attempt_conflict",
    "internal_error",
]
```

Implement these exact public models:

```python
class SearchRequest(DomainModel):
    query_id: NonEmptyStr
    query: NonEmptyStr
    budget_profile: Literal["low", "balanced"] = "balanced"
    include_trace: bool = True
    mode: SearchMode = "replay"


class DependencyStatus(DomainModel):
    dependency: DependencyName
    state: DependencyState
    cache_hit: bool
    error_codes: list[DependencyErrorCode]


class StructuredSearchResponse(DomainModel):
    run_id: NonEmptyStr
    query_id: NonEmptyStr
    execution_mode: SearchMode
    snapshot_set_id: NonEmptyStr
    snapshot_captured_at: datetime | None
    query_analysis: QueryAnalysisResult
    selected_paper_ids: list[NonEmptyStr]
    high_relevance: list[RankedPaper]
    partial_relevance: list[RankedPaper]
    citation_edges: list[ResolvedCitationEdge]
    search_trace: list[dict[str, object]]
    usage: UsageActual
    stop_reason: NonEmptyStr
    is_partial: bool
    planner_fallback: bool
    planner_status: PlannerStatus
    dependency_status: list[DependencyStatus]
    warnings: list[NonEmptyStr]
    prompt_version: NonEmptyStr
    config_hash: Sha256
    git_sha: NonEmptyStr
```

Implement these exact internal/public boundary models:

```python
class SnapshotRef(DomainModel):
    entry_id: NonEmptyStr
    dependency: DependencyName
    cache_key: Sha256
    response_sha256: Sha256
    captured_at: datetime
    snapshot_path: SafeRelativePath


class DependencyDiagnostic(DomainModel):
    dependency: DependencyName
    endpoint: NonEmptyStr
    model_id: NonEmptyStr | None
    usage: UsageActual
    latency_ms: NonNegativeInt
    cache_hit: bool
    snapshot_refs: list[SnapshotRef]
    errors: list[ErrorDetail]


class SearchErrorResponse(DomainModel):
    code: SearchErrorCode
    detail: NonEmptyStr
    retryable: bool
    run_id: NonEmptyStr | None


class SearchSuccess(DomainModel):
    kind: Literal["success"] = "success"
    response: StructuredSearchResponse


class SearchFailure(DomainModel):
    kind: Literal["failure"] = "failure"
    query_id: NonEmptyStr
    run_id: NonEmptyStr
    error: SearchErrorResponse
    usage: UsageActual
    stop_reason: NonEmptyStr


SearchOutcome = Annotated[
    SearchSuccess | SearchFailure,
    Field(discriminator="kind"),
]


class SearchExecutionResult(DomainModel):
    outcome: SearchOutcome
    diagnostics: list[DependencyDiagnostic]
    business_result_sha256: Sha256 | None


class ReadyHealthResponse(DomainModel):
    status: Literal["ready", "degraded"]
    execution_mode: SearchMode
    snapshot_set_id: NonEmptyStr | None
    dependencies: list[DependencyStatus]
    last_authorized_probe_at: datetime | None
```

Extend `SearchRequest`, `DependencyStatus`, `StructuredSearchResponse`, and `ReadyHealthResponse` with the exact normative fields in the design. Enforce:

- dependency status order is `llm`, `openalex`, `semantic_scholar`;
- `planner_status="rules_fallback"` implies `planner_fallback=true`, `is_partial=true`, and the fixed warning `planner_rules_fallback`;
- other planner states imply `planner_fallback=false`;
- `include_trace=false` removes only `search_trace`;
- the compatibility converter leaves evidence collections empty when the old mock pipeline did not produce evidence and never fabricates scores or edges.

**Steps:**

- [ ] Add failing model tests for valid/invalid `Sha256`, path normalization, discriminated outcomes, dependency order, planner invariants, forbidden extras, and Decimal JSON serialization.
- [ ] Run the focused tests and confirm collection or assertion failure because the new types and fields do not exist.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/application/test_contracts.py tests/api/test_contracts.py tests/unit/test_models.py -q
```

Expected initial result: FAIL with missing imports/fields or failed invariants.

- [ ] Add the shared validated primitives to `domain/models.py`; migrate `UsageEstimate.cost_cny` and `UsageActual.cost_cny` to `MoneyCny | None`.
- [ ] Add the exact application contracts and re-export stable public types from `application/__init__.py`.
- [ ] Extend request, response, and readiness models without defining a second `StructuredSearchResponse`.
- [ ] Update mock constructors and converter fixtures in one atomic migration. Use explicit synthetic values such as `run_id="mock-run-1"`, `execution_mode="replay"`, and `snapshot_set_id="mock-snapshot-v1"`.
- [ ] Add tests proving the old converter does not invent ranking evidence.
- [ ] Re-run the focused tests.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/application/test_contracts.py tests/api/test_contracts.py tests/api/test_service.py tests/api/test_app.py tests/unit/test_response.py tests/unit/test_models.py -q
```

Expected result: PASS.

- [ ] Run static checks for the migrated contract surface.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/application src/paper_search/domain/models.py src/paper_search/api tests/application tests/api tests/unit/test_models.py tests/unit/test_response.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/application src/paper_search/domain/models.py src/paper_search/api
```

Expected result: both commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/application/__init__.py src/paper_search/application/contracts.py src/paper_search/domain/models.py src/paper_search/api/contracts.py src/paper_search/api/service.py src/paper_search/api/app.py src/paper_search/pipeline/response.py tests/application/test_contracts.py tests/api/test_contracts.py tests/api/test_service.py tests/api/test_app.py tests/unit/test_response.py tests/unit/test_models.py
git commit -m "feat: freeze integrated application contracts"
```

---

### Task 2: Add reproducible runtime and exact lock schemas

**Files:**

- Create: `src/paper_search/application/locks.py`
- Create: `tests/application/test_locks.py`
- Create: `tests/fixtures/application/candidate.lock.yaml`
- Create: `tests/fixtures/application/validation.lock.yaml`
- Create: `tests/fixtures/application/replay.lock.yaml`
- Modify: `src/paper_search/application/__init__.py`
- Modify: `src/paper_search/config.py`
- Modify: `configs/base.yaml`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/test_config.py`

**Interfaces:**

```python
class ArtifactBinding(DomainModel):
    path: SafeRelativePath
    sha256: Sha256


class FrozenDataBinding(DomainModel):
    manifest: ArtifactBinding
    identifier_map: ArtifactBinding
    split: Literal["smoke", "dev", "validation"]
    query_count: PositiveInt
    partition_sha256: Sha256


class CapturePolicyBinding(DomainModel):
    snapshot_schema: Literal["dependency-snapshot-v2"]
    capture_policy_sha256: Sha256


class TimeoutBinding(DomainModel):
    connect_seconds: Literal[5]
    read_seconds: Literal[20]
    write_seconds: Literal[20]
    pool_seconds: Literal[5]


class RetryBinding(DomainModel):
    max_attempts: Literal[3]
    retryable_statuses: tuple[Literal[429], Literal["5xx"]]
    retry_timeouts: Literal[True]
    backoff_rule: Literal["min(8,2^retry_index)+jitter[0,1)"]


class PlannerBinding(DomainModel):
    prompt_config: ArtifactBinding
    normal_subqueries_min: Literal[3]
    normal_subqueries_max: Literal[5]
    configured_subqueries_max: Literal[6]
    repair_attempts: Literal[1]
    rules_fallback_enabled: Literal[True]


class RetrievalBinding(DomainModel):
    openalex_endpoint: Literal["/works"]
    semantic_scholar_endpoint: Literal["/graph/v1/paper/search"]
    openalex_calls_min: Literal[3]
    openalex_calls_max: Literal[6]
    semantic_scholar_calls_max: Literal[2]
    max_results_per_subquery: Literal[50]
    max_raw_candidates: Literal[300]
    max_deduplicated_candidates: Literal[200]
    max_output_papers: Literal[50]


class BaselineOptionalModules(DomainModel):
    embedding: Literal[False]
    citation_expansion: Literal[False]
    constraint_reranking: Literal[False]
    fixed_two_round: Literal[False]
    adaptive_evolution: Literal[False]


class BaselineBinding(DomainModel):
    primary_model: Literal["qwen3.7-plus"]
    fallback_model: Literal["qwen3.6-flash"]
    prompt_version: Literal["query-analyze-v1"]
    strategy: Literal["fixed-one-round"]
    planner: PlannerBinding
    retrieval: RetrievalBinding
    timeout: TimeoutBinding
    retry: RetryBinding
    optional_modules: BaselineOptionalModules


class CandidateLock(DomainModel):
    schema_version: Literal["integrated-lock-v1"]
    lock_kind: Literal["candidate"]
    created_at: datetime
    source_git_sha: NonEmptyStr
    runtime_allow_live: Literal[True]
    frozen_data: FrozenDataBinding
    baseline: BaselineBinding
    budget_config: ArtifactBinding
    pricing_policy: ArtifactBinding
    quality_gates: ArtifactBinding
    capture_policy: CapturePolicyBinding
    approval_ref: NonEmptyStr


class ValidationLock(CandidateLock):
    lock_kind: Literal["validation"]
    frozen_data: FrozenDataBinding  # split must be validation
    promoted_from_dev_run_id: NonEmptyStr
    promoted_from_dev_run_sha256: Sha256


class ReplayLock(DomainModel):
    schema_version: Literal["integrated-lock-v1"]
    lock_kind: Literal["replay"]
    created_at: datetime
    source_capture_run_id: NonEmptyStr
    source_git_sha: NonEmptyStr
    runtime_allow_live: bool
    frozen_data: FrozenDataBinding
    baseline: BaselineBinding
    budget_config: ArtifactBinding
    pricing_policy: ArtifactBinding
    quality_gates: ArtifactBinding
    capture_policy: CapturePolicyBinding
    snapshot_set_id: NonEmptyStr
    snapshot_manifest_sha256: Sha256


InputLock = Annotated[
    CandidateLock | ValidationLock | ReplayLock,
    Field(discriminator="lock_kind"),
]


def load_input_lock(path: Path, *, artifact_root: Path) -> InputLock: ...
def canonical_lock_bytes(lock: InputLock) -> bytes: ...
def lock_sha256(lock: InputLock) -> Sha256: ...
```

`RuntimeConfig` must expose reproducible settings separately from secret capabilities:

```python
class RuntimeSettings(BaseModel):
    allow_live: bool = False
    artifact_root: Path
    connect_timeout_seconds: Literal[5] = 5
    read_timeout_seconds: Literal[20] = 20
    write_timeout_seconds: Literal[20] = 20
    pool_timeout_seconds: Literal[5] = 5
    max_attempts: Literal[3] = 3
```

The lock loader validates relative paths against `artifact_root`, reads each referenced file once, hashes the same bytes it validates, and rejects environment overrides that would alter a frozen identity. A replay lock may preserve `runtime_allow_live=true` so `paper-search serve --allow-live` can use its immutable replay binding plus request-scoped live capture; replay execution itself remains offline regardless of this capability bit.

**Steps:**

- [ ] Add failing tests for exact fields, lock-kind discrimination, canonical hash stability, forbidden snapshot fields in candidate/validation locks, required manifest fields in replay locks, replay live-capability preservation, split restrictions, path escape, symlink escape where supported, and frozen-setting/environment mismatch.
- [ ] Run the focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/application/test_locks.py tests/unit/test_config.py tests/test_config.py -q
```

Expected initial result: FAIL because the lock loader and nested runtime schema are absent.

- [ ] Implement the exact lock models and canonical YAML/JSON identity calculation.
- [ ] Extend `RuntimeConfig` with `runtime`, policy bindings, capture policy, routing limits, and retry configuration while retaining secret loading by environment name.
- [ ] Set `runtime.allow_live: false` in `configs/base.yaml`; retain all optional modules off.
- [ ] Add a lock/config regression proving `configs/budget_balanced.yaml` still fixes 12 search calls, 5 LLM calls, 2 iterations, 6 configured subqueries, 90 elapsed seconds, 80 soft-deadline seconds, 24,000 total tokens, 50 output papers, and CNY 0.30.
- [ ] Ensure `RuntimeConfig.config_hash()` excludes secret values but includes every reproducible runtime field and policy identity.
- [ ] Add a test that mutating any lock field changes the lock hash.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/application/test_locks.py tests/unit/test_config.py tests/test_config.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/application/locks.py src/paper_search/config.py tests/application/test_locks.py tests/unit/test_config.py tests/test_config.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/application/locks.py src/paper_search/config.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/application/__init__.py src/paper_search/application/locks.py src/paper_search/config.py configs/base.yaml tests/application/test_locks.py tests/fixtures/application/candidate.lock.yaml tests/fixtures/application/validation.lock.yaml tests/fixtures/application/replay.lock.yaml tests/unit/test_config.py tests/test_config.py
git commit -m "feat: add integrated input lock schemas"
```

---

### Task 3: Add versioned pricing and quality-Gate policies

**Files:**

- Create: `src/paper_search/control/pricing.py`
- Create: `configs/quality_gates_v1.yaml`
- Create: `tests/unit/test_pricing.py`
- Create: `tests/fixtures/pricing/pricing-policy-test-v1.yaml`
- Modify: `src/paper_search/control/__init__.py`
- Modify: `src/paper_search/control/budget.py`
- Modify: `tests/unit/test_budget.py`
- Modify: `tests/application/test_locks.py`

**Interfaces:**

```python
class PricingRate(DomainModel):
    dependency: DependencyName
    model_or_adapter: NonEmptyStr
    unit: Literal["input_token", "output_token", "request"]
    price_cny_per_unit: MoneyCny


class PricingPolicy(DomainModel):
    schema_version: Literal["pricing-policy-v1"]
    currency: Literal["CNY"]
    effective_at: datetime
    source_identity: NonEmptyStr
    rounding_quantum_cny: Literal[Decimal("0.000001")]
    rates: list[PricingRate]


class ActualCostPricer:
    @property
    def policy_sha256(self) -> Sha256: ...

    def value_actual(
        self,
        *,
        dependency: DependencyName,
        model_or_adapter: str,
        usage: UsageActual,
    ) -> UsageActual: ...


class QualityGateRule(DomainModel):
    rule_id: NonEmptyStr
    classification: Literal[
        "formal_validity", "baseline_quality", "reporting_only", "promotion"
    ]
    measure: NonEmptyStr
    operator: Literal["eq", "gt", "gte", "lte"]
    threshold: Decimal | int
    applies_to: list[Literal["dev", "validation", "frozen_audit", "optional"]]
    source_refs: list[NonEmptyStr]
    resolution: NonEmptyStr
```

`configs/quality_gates_v1.yaml` must encode all rows in the approved design:

- exact prediction and failure cardinality;
- zero integrity/provenance/sanitization/unaccounted-usage failures;
- all budget ledgers within cap;
- model-produced analysis rate `>= 0.99`;
- strong-constraint recall `>= 0.90`;
- retrieval response rate `>= 0.95`;
- fuzzy-merge accuracy `>= 0.98` with nonempty denominator;
- hard-filter Recall loss `<= 0.02`;
- macro and micro Recall `> 0`;
- reporting-only relevance, failure, partial, fallback, latency, calls, tokens, cost, and cache metrics;
- optional promotion median delta `>= 0.01`, bootstrap lower bound `>= -0.005`, validation drop `<= 0.01`, and `bootstrap_samples=1000`.

For every numeric PRD/Week 1/R3 row, `source_refs` names the originating section and `resolution` states whether the approved integrated design classifies it as enforced, reporting-only, superseded by a stricter named rule, or optional-promotion-only. The loader rejects a policy when a discovered authoritative row has no explicit resolution. In particular, historical macro-F1/delta, structured-output, hard-failure, and partial-result rows cannot silently disappear; their relationship to the approved table must be explicit.

**Steps:**

- [ ] Add failing pricing tests using fixture-only rates and exact Decimal expectations.
- [ ] Add failing tests that unknown model/unit, duplicate rate, ineffective policy date, missing actual usage, or authoritative billed-amount mismatch raises a fail-closed pricing error.
- [ ] Add a policy test that parses all approved quality rows and rejects an unclassified PRD threshold.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_pricing.py tests/unit/test_budget.py tests/application/test_locks.py -q
```

Expected initial result: FAIL because pricing policy and policy-identity validation are absent.

- [ ] Implement canonical policy loading and Decimal valuation. Quantize once at the policy boundary using `ROUND_HALF_EVEN`.
- [ ] Tighten `HardBudgetController.settle()` so formal live settlement cannot commit `cost_cny=None`; retain an explicit non-formal compatibility path only where an existing unit test requires it.
- [ ] Write `quality_gates_v1.yaml` with the exact classifications above and add its hash to lock fixtures.
- [ ] Do not create a production `configs/pricing_v1.yaml` from guessed values. Document the exact expected schema through the validated fixture and Gate 0 error `pricing_policy_missing`.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_pricing.py tests/unit/test_budget.py tests/application/test_locks.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/control tests/unit/test_pricing.py tests/unit/test_budget.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/control
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/control/__init__.py src/paper_search/control/pricing.py src/paper_search/control/budget.py configs/quality_gates_v1.yaml tests/unit/test_pricing.py tests/fixtures/pricing/pricing-policy-test-v1.yaml tests/unit/test_budget.py tests/application/test_locks.py tests/fixtures/application
git commit -m "feat: add deterministic pricing and gate policies"
```

---

### Task 4: Add exact V2 freeze schema and explicit V1 migration

**Files:**

- Create: `src/paper_search/evaluation/freeze_schema.py`
- Create: `tests/evaluation/test_freeze_schema.py`
- Create: `tests/fixtures/evaluation/freeze_v2/`
- Modify: `src/paper_search/evaluation/freeze.py`
- Modify: `tests/evaluation/test_freeze.py`
- Modify: `data/manifest.example.json`

**Interfaces:**

```python
class FrozenPartitionV2(DomainModel):
    name: Literal["dev", "validation"]
    path: SafeRelativePath
    query_count: PositiveInt
    sha256: Sha256
    zero_answer_policy: Literal["allow", "forbid"]


class IdentifierMapBindingV2(DomainModel):
    path: SafeRelativePath
    sha256: Sha256
    entry_count: PositiveInt


class FreezeApprovalReportV2(DomainModel):
    schema_version: Literal["freeze-approval-v2"]
    approval_requested: Literal[True]
    approved_at: datetime
    approver_ref: NonEmptyStr
    audit_sha256: Sha256
    partition_hashes: dict[Literal["dev", "validation"], Sha256]
    identifier_map_sha256: Sha256


class FreezeApprovalBindingV2(DomainModel):
    report_path: SafeRelativePath
    report_sha256: Sha256
    approved_at: datetime
    approver_ref: NonEmptyStr


class FreezeManifestV2(DomainModel):
    schema_version: Literal["paper-search-freeze-v2"]
    dataset_revision: NonEmptyStr
    created_at: datetime
    annotation_status: Literal["frozen"]
    freeze_status: Literal["approved"]
    partitions: list[FrozenPartitionV2]
    gold_sha256: Sha256
    identifier_map: IdentifierMapBindingV2
    partition_immutability: Literal["content_addressed"]
    approval: FreezeApprovalBindingV2


FreezeManifest = Annotated[
    FreezeManifestV1 | FreezeManifestV2,
    Field(discriminator="schema_version"),
]


def load_freeze_manifest(path: Path, *, data_root: Path) -> FreezeManifest: ...
def migrate_v1_to_v2(
    v1: FreezeManifestV1,
    *,
    approval: FreezeApprovalReportV2,
    identifier_map: IdentifierMapBindingV2,
    dataset_revision: str,
) -> FreezeManifestV2: ...
```

The migration requires an already approved V1 freeze, verifies every old hash again, adds the V2-only identifier-map evidence, writes a new V2 approval report, and delegates guarded publication to the existing atomic/no-overwrite machinery.

**Steps:**

- [ ] Add failing tests for exact V2 fields, V1 readability, rejection of unapproved V1 input, revalidation after read, path confinement, approval-report matching, id-map identity, no-overwrite, idempotent exact match, and concurrent replacement.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_freeze_schema.py tests/evaluation/test_freeze.py -q
```

Expected initial result: FAIL because version dispatch and V2 models are absent.

- [ ] Extract only genuinely shared public hash/path primitives; do not import private `_confined_path` or `_json_bytes` helpers from `freeze.py`.
- [ ] Implement exact V1/V2 dispatch before the existing exact-field validator.
- [ ] Implement explicit migration and preserve the existing lock, guarded replace, evidence revalidation, and idempotence guarantees.
- [ ] Replace `data/manifest.example.json` with a sanitized V2 example that contains no real query or private path.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_freeze_schema.py tests/evaluation/test_freeze.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/evaluation/freeze.py src/paper_search/evaluation/freeze_schema.py tests/evaluation/test_freeze.py tests/evaluation/test_freeze_schema.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/freeze.py src/paper_search/evaluation/freeze_schema.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/evaluation/freeze.py src/paper_search/evaluation/freeze_schema.py tests/evaluation/test_freeze.py tests/evaluation/test_freeze_schema.py tests/fixtures/evaluation/freeze_v2 data/manifest.example.json
git commit -m "feat: add versioned V2 freeze schema"
```

---

### Task 5: Implement the deterministic Gate 0 reconciler

**Files:**

- Create: `src/paper_search/evaluation/gate0.py`
- Create: `tests/evaluation/test_gate0.py`
- Modify: `tests/evaluation/test_dataset.py`
- Modify: `.gitignore`

**Interfaces:**

```python
Gate0ReasonCode = Literal[
    "manifest_missing",
    "manifest_invalid",
    "approval_invalid",
    "partition_hash_mismatch",
    "partition_count_mismatch",
    "identifier_map_missing",
    "identifier_map_hash_mismatch",
    "identifier_map_coverage_failed",
    "pricing_policy_missing",
    "pricing_policy_invalid",
    "quality_policy_invalid",
    "readiness_evidence_invalid",
]


class Gate0ArtifactEvidence(DomainModel):
    identity: NonEmptyStr
    sha256: Sha256
    count: NonNegativeInt | None


class Gate0Report(DomainModel):
    schema_version: Literal["gate0-report-v1"]
    generated_at: datetime
    passed: bool
    blocking_reasons: list[Gate0ReasonCode]
    manifest: Gate0ArtifactEvidence | None
    partitions: list[Gate0ArtifactEvidence]
    identifier_map: Gate0ArtifactEvidence | None
    pricing_policy_sha256: Sha256 | None
    quality_gates_sha256: Sha256 | None
    readiness_report_sha256: Sha256 | None


def verify_gate0(
    *,
    data_root: Path,
    manifest_path: Path,
    pricing_policy_path: Path,
    quality_gates_path: Path,
    readiness_report_path: Path,
    clock: Callable[[], datetime],
) -> Gate0Report: ...


def write_gate0_report(path: Path, report: Gate0Report) -> None: ...
```

The report sorts partition evidence and blocking reasons deterministically. Readiness evidence records capability names and timestamps only; it is evidence for Gate 0 reconciliation, not current live authorization.

**Steps:**

- [ ] Add a complete synthetic passing fixture and one failing case for every reason code.
- [ ] Add tests that all partition query IDs resolve through the exact bound identifier map and that counts/hashes are calculated from bytes read once.
- [ ] Add a sanitization test containing fake authorization headers, credential-shaped strings, gated query text, and absolute paths; assert none appear in serialized output.
- [ ] Add a mutation test proving `data/manifest.json` and status docs are unchanged when Gate 0 fails.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_gate0.py tests/evaluation/test_dataset.py -k "gate0 or identifier_map or provider" -q
```

Expected initial result: FAIL because `gate0.py` is absent.

- [ ] Implement read-once hash/validation for manifest, approval, partitions, identifier map, pricing policy, quality policy, and readiness evidence.
- [ ] Reuse `IdentifierMap.from_bytes()` for coverage validation.
- [ ] Write reports atomically and never mutate source evidence.
- [ ] Add `/runs/`, `/validation-attempts/`, and `/private-gate0/` to `.gitignore`.
- [ ] Run the current repository evidence through the verifier without requesting live access. Record the expected blocked result in the task handoff; do not edit public status files.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file python -m paper_search.evaluation.gate0 --data-root data --manifest data/manifest.json --pricing-policy configs/pricing_v1.yaml --quality-gates configs/quality_gates_v1.yaml --readiness data/provider_readiness.json --report .gate0-report.json
```

Expected current result: nonzero exit with deterministic blocking reasons because the repository does not yet contain an approved production pricing policy or verified V2 public manifest.

- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/evaluation/test_gate0.py tests/evaluation/test_dataset.py -k "gate0 or identifier_map or provider" -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/evaluation/gate0.py tests/evaluation/test_gate0.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/evaluation/gate0.py
```

Expected result: test and static-check commands exit 0; the real-evidence Gate 0 command remains nonzero until operator inputs are available and verified.

- [ ] Commit.

```powershell
git add src/paper_search/evaluation/gate0.py tests/evaluation/test_gate0.py tests/evaluation/test_dataset.py .gitignore
git commit -m "feat: add fail-closed Gate 0 verification"
```

---

### Task 6: Reconcile public status only after real Gate 0 passes

**Files:**

- Conditional modify: `data/manifest.json`
- Conditional create: `data/gate0_evidence.json`
- Conditional modify: `README.md`
- Conditional modify: `data/README.md`
- Conditional modify: `PRD.md`

**Interfaces:**

The operator supplies an access-controlled root containing:

```text
private-gate0/
├── manifest.json
├── freeze-approval.json
├── dev.jsonl
├── validation.jsonl
├── identifier-map.json
├── pricing-policy-v1.yaml
└── provider-readiness.json
```

No private file is staged. The public `data/gate0_evidence.json` is exactly a serialized `Gate0Report` with aggregate counts and hashes.

**Steps:**

- [ ] Confirm the operator-supplied root and all paths are inside the approved access-controlled root.
- [ ] Run Gate 0 against exact operator paths.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file python -m paper_search.evaluation.gate0 --data-root 'private-gate0' --manifest 'private-gate0/manifest.json' --pricing-policy 'private-gate0/pricing-policy-v1.yaml' --quality-gates 'configs/quality_gates_v1.yaml' --readiness 'private-gate0/provider-readiness.json' --report 'private-gate0/gate0-report.json'
```

Expected result required before continuing: exit 0 and `"passed": true`.

- [ ] If the command is nonzero, stop this task, retain `private-gate0/gate0-report.json` outside tracked public paths, and leave all five listed public files unchanged.
- [ ] If the command exits 0, copy only the verified sanitized V2 public manifest projection and `private-gate0/gate0-report.json` into `data/gate0_evidence.json`.
- [ ] Update README/data README/PRD statements to cite the verified schema version, counts, and hashes without publishing labels, raw queries, credentials, or private paths.
- [ ] Verify the public projection contains no unsafe content.

```powershell
rg -n -i "authorization|bearer|api[_-]?key|secret|private-gate0|^[A-Z]:\\\\" data/manifest.json data/gate0_evidence.json README.md data/README.md PRD.md
```

Expected result: no matches.

- [ ] Run Phase 1 regression and full engineering checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/application tests/api tests/unit/test_config.py tests/unit/test_pricing.py tests/unit/test_budget.py tests/evaluation/test_freeze_schema.py tests/evaluation/test_freeze.py tests/evaluation/test_gate0.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
```

Expected result: all commands exit 0.

- [ ] Commit only after a real passing report exists.

```powershell
git add data/manifest.json data/gate0_evidence.json README.md data/README.md PRD.md
git commit -m "docs: reconcile verified V2 freeze status"
```

---

## Phase 1 Exit Evidence

- The canonical contracts import from one location and all direct constructors are migrated.
- Candidate and validation locks contain no snapshot manifest identity; replay locks require both manifest hash and snapshot-set ID.
- Pricing arithmetic is deterministic and unknown live cost fails closed.
- V1 evidence remains readable; only V2 approval can authorize an integrated formal run.
- Gate 0 emits a deterministic sanitized report and cannot mutate public status on failure.
- A real Gate 0 pass is required before Phase 2 live smoke, Phase 3 formal evaluation, or public “V2 frozen” claims. Replay fixture engineering may continue while Gate 0 is blocked.
