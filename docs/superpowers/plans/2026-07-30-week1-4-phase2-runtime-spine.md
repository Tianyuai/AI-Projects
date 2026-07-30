# Week 1–4 Phase 2 Replay/Live Runtime Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one structurally offline replay path and one explicitly authorized live-capture path behind `SearchApplicationService`, then prove the shared path with the `paper-search smoke` command.

**Architecture:** A sealed `dependency-snapshot-v2` store sits beneath LLM/OpenAlex/Semantic Scholar adapters. Live adapters price and settle actual usage, stage exact successful response bytes, and expose snapshot references; replay adapters receive no network transport and fail closed on misses. A persistent hierarchical ledger wraps request-local `HardBudgetController`. `CompositionRoot` validates one lock and one immutable mode binding, then returns the same application service used by later evaluation and API/UI consumers.

**Tech Stack:** Python 3.11+, asyncio, httpx, SQLite, Pydantic v2, FastAPI-compatible application service, pytest/pytest-asyncio, Ruff, mypy.

## Global Constraints

- Phase 1 contracts, lock loader, V2 freeze authority, pricing policy, and Gate policy are prerequisites.
- Approved design: `docs/superpowers/specs/2026-07-30-week1-4-integrated-baseline-demo-design.md`.
- Replay is structurally offline: replay objects have no `httpx.AsyncClient`, DNS helper, socket helper, live cache fallback, or callable live adapter.
- A replay lock plus operator-supplied manifest path is required. The path must resolve beneath the lock-bound artifact root; bytes, `snapshot_manifest_sha256`, and `snapshot_set_id` must all match before composition.
- Live requires `runtime.allow_live=true`, a live-capable candidate, validation, or manifest-bound replay lock, and command-level `--allow-network` (or Phase 4 server `--allow-live`). No single key is sufficient.
- Capture exact bytes only for successful dependency responses. Never capture authorization headers, secret fields, raw error bodies, or unallowlisted request fields.
- Actual usage is priced before settlement. Unknown cost is a hard failure for formal live work.
- Retried/failed attempts consume request/run/project usage even when their response bodies are not captured.
- Keep the main baseline fixed one round. Optional modules remain unconstructed.
- Preserve existing `generate_json(...)` and `SearchProvider` call signatures at adapter boundaries.
- Do not modify evaluation runner, final API error mapping, UI, or named experiments in this phase.
- Follow red-green-refactor and make one focused commit per task.

---

## File Structure and Ownership

### Create

- `src/paper_search/storage/dependency_snapshot.py` — V2 manifest, capture writer, and verified read-only index.
- `src/paper_search/llm/snapshot_adapters.py` — live capture and replay analyzer adapters.
- `src/paper_search/retrieval/snapshot_adapters.py` — live capture and replay Provider wrappers.
- `src/paper_search/retrieval/routing.py` — deterministic OpenAlex/Semantic Scholar baseline routing.
- `src/paper_search/control/ledger.py` — persistent run/project reservations and settlements.
- `src/paper_search/application/service.py` — canonical execution envelope and public success adapter.
- `src/paper_search/application/modes.py` — mode authorization and immutable mode binding.
- `src/paper_search/application/composition.py` — production composition root.
- `src/paper_search/application/artifacts.py` — capture session and atomic smoke-run publication primitives.
- `src/paper_search/cli.py` — stable root parser with `smoke`; later phases add subcommands.
- `tests/unit/test_dependency_snapshot.py`
- `tests/unit/test_llm_snapshot_adapters.py`
- `tests/unit/test_retrieval_snapshot_adapters.py`
- `tests/unit/test_budget_ledger.py`
- `tests/unit/test_application_service.py`
- `tests/integration/test_application_composition.py`
- `tests/integration/test_smoke_cli.py`
- `tests/fixtures/dependency_snapshot_v2/`

### Modify

- `src/paper_search/storage/__init__.py`
- `src/paper_search/llm/client.py`
- `src/paper_search/llm/__init__.py`
- `src/paper_search/retrieval/openalex.py`
- `src/paper_search/retrieval/semantic_scholar.py`
- `src/paper_search/retrieval/__init__.py`
- `src/paper_search/control/__init__.py`
- `src/paper_search/pipeline/orchestrator.py`
- `src/paper_search/pipeline/response.py`
- `src/paper_search/query/parser.py`
- `src/paper_search/config.py`
- `src/paper_search/application/__init__.py`
- `pyproject.toml`
- existing focused unit/integration tests for all modified modules.

### Preserve as Compatibility Layers

- `src/paper_search/storage/cache.py` remains the mutable V1 response cache, not replay authority.
- `src/paper_search/api/service.py:MockApiSearchService` remains available for legacy mock tests.
- Existing module CLIs remain available until Phase 3 marks the formal runner command as superseded.

---

### Task 1: Implement the sealed dependency snapshot V2 store

**Files:**

- Create: `src/paper_search/storage/dependency_snapshot.py`
- Create: `tests/unit/test_dependency_snapshot.py`
- Create: `tests/fixtures/dependency_snapshot_v2/`
- Modify: `src/paper_search/storage/__init__.py`
- Modify: `tests/unit/test_cache.py`

**Interfaces:**

```python
class DependencyRequestIdentity(DomainModel):
    schema_version: Literal["dependency-request-v1"]
    dependency: DependencyName
    operation: NonEmptyStr
    method: Literal["GET", "POST"]
    endpoint: NonEmptyStr
    model_or_adapter: NonEmptyStr
    canonical_request_sha256: Sha256


class SnapshotEntryV2(DomainModel):
    entry_id: NonEmptyStr
    request: DependencyRequestIdentity
    cache_key: Sha256
    response_sha256: Sha256
    captured_at: datetime
    response_path: SafeRelativePath
    safe_headers: dict[NonEmptyStr, NonEmptyStr]


class DependencySnapshotManifestV2(DomainModel):
    schema_version: Literal["dependency-snapshot-v2"]
    snapshot_set_id: Sha256
    sealed_at: datetime
    entries: list[SnapshotEntryV2]


class SnapshotRead(DomainModel):
    ref: SnapshotRef
    response_bytes: bytes


class DependencyCaptureStore:
    def stage_success(
        self,
        identity: DependencyRequestIdentity,
        *,
        response_bytes: bytes,
        safe_headers: Mapping[str, str],
        captured_at: datetime,
    ) -> SnapshotRef: ...

    def seal(self) -> DependencySnapshotManifestV2: ...


class DependencySnapshotReader:
    @property
    def snapshot_set_id(self) -> Sha256: ...

    def read(self, identity: DependencyRequestIdentity) -> SnapshotRead: ...
```

Canonicalization rules:

- `canonical_request_sha256` is calculated from an allowlisted canonical JSON object, not raw request headers/body.
- `cache_key` is the SHA-256 of the complete `DependencyRequestIdentity`.
- entries are sorted by `(dependency, cache_key, entry_id)`;
- duplicate cache keys are rejected;
- `snapshot_set_id` is the SHA-256 of canonical ordered entry metadata excluding `snapshot_set_id` and `sealed_at`;
- every `read()` re-reads the bound response path and verifies `response_sha256` before returning bytes;
- the manifest is read once, validated against its lock, and retained as immutable parsed state.

**Steps:**

- [ ] Add failing tests for all three dependency kinds, deterministic keys, exact-byte round trip, duplicate key, unsealed store, missing key, manifest tamper, payload tamper, absolute/escaping/symlink path, and secret-shaped safe header rejection.
- [ ] Add a replay-offline construction test proving the reader accepts no live client.
- [ ] Add a V1 migration test that accepts only Provider entries whose dependency, endpoint, request identity, and response hash can be established; ambiguous V1 entries are rejected.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_dependency_snapshot.py tests/unit/test_cache.py -q
```

Expected initial result: FAIL because the V2 store is absent.

- [ ] Implement atomic response staging to a store-local temporary file followed by same-filesystem replacement.
- [ ] Implement deterministic manifest sealing and no writes after `seal()`.
- [ ] Implement read-only lookup with hash verification at every read.
- [ ] Export only the public store types from `storage/__init__.py`.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_dependency_snapshot.py tests/unit/test_cache.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/storage tests/unit/test_dependency_snapshot.py tests/unit/test_cache.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/storage
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/storage/dependency_snapshot.py src/paper_search/storage/__init__.py tests/unit/test_dependency_snapshot.py tests/unit/test_cache.py tests/fixtures/dependency_snapshot_v2
git commit -m "feat: add sealed dependency snapshot store"
```

---

### Task 2: Add priced LLM live-capture and replay adapters

**Files:**

- Create: `src/paper_search/llm/snapshot_adapters.py`
- Create: `tests/unit/test_llm_snapshot_adapters.py`
- Modify: `src/paper_search/llm/client.py`
- Modify: `src/paper_search/llm/__init__.py`
- Modify: `tests/unit/test_llm_client.py`
- Modify: `tests/unit/test_query_parser.py`

**Interfaces:**

```python
class LLMResponseDecoder:
    def decode(
        self,
        response_bytes: bytes,
        *,
        model_id: str,
        captured_at: datetime,
        cache_hit: bool,
        snapshot_ref: SnapshotRef | None,
    ) -> ProviderResult[dict[str, Any]]: ...


class LiveCaptureLLMAnalyzer:
    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]: ...


class ReplayLLMAnalyzer:
    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: BudgetReservation,
    ) -> ProviderResult[dict[str, Any]]: ...
```

Both adapters use the same pure decoder. The live adapter:

1. builds the safe identity before dispatch;
2. performs at most three total attempts for timeout/429/5xx;
3. values each attempt's actual usage and accumulates all attempts into one settlement total;
4. stages exact successful `response.content` before returning parsed data;
5. never stages failed/error bodies.

The original request reservation is settled exactly once with the accumulated attempt usage. A terminal exception path calls the request controller's fail-closed settlement with the same accumulated measured usage; it does not settle one reservation multiple times.

The replay adapter:

1. builds the identical identity;
2. reads bytes from the bound `DependencySnapshotReader`;
3. decodes with `cache_hit=true`;
4. returns zero external-call/token cost for replay while preserving snapshot provenance;
5. maps a miss to `snapshot_unavailable` with no network fallback.

Planner classification must distinguish malformed content from transport/authentication failure. Only malformed content receives one repair attempt and then the frozen rules fallback.

**Steps:**

- [ ] Add failing tests for safe identity, exact-byte capture before parse return, identical replay parse, primary/repaired/rules-fallback status, snapshot miss, authentication failure, timeout retry, 429 retry, nonretryable 400, failed-body exclusion, and priced settlement.
- [ ] Add a test proving transport/authentication failure cannot become a rules fallback.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_llm_client.py tests/unit/test_llm_snapshot_adapters.py tests/unit/test_query_parser.py -q
```

Expected initial result: FAIL because snapshot adapters and planner classification are absent.

- [ ] Refactor response decoding into one pure decoder while retaining `OpenAICompatibleLLMClient.generate_json(...)`.
- [ ] Inject `ActualCostPricer` and capture/reader dependencies into wrappers, not the pure decoder.
- [ ] Sanitize endpoint/model/error information through fixed code mapping; never expose exception text.
- [ ] Implement the bounded repair/fallback classification.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_llm_client.py tests/unit/test_llm_snapshot_adapters.py tests/unit/test_query_parser.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/llm src/paper_search/query/parser.py tests/unit/test_llm_client.py tests/unit/test_llm_snapshot_adapters.py tests/unit/test_query_parser.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/llm src/paper_search/query/parser.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/llm/client.py src/paper_search/llm/snapshot_adapters.py src/paper_search/llm/__init__.py src/paper_search/query/parser.py tests/unit/test_llm_client.py tests/unit/test_llm_snapshot_adapters.py tests/unit/test_query_parser.py
git commit -m "feat: add LLM capture and replay adapters"
```

---

### Task 3: Add priced Provider capture/replay adapters and bounded routing

**Files:**

- Create: `src/paper_search/retrieval/snapshot_adapters.py`
- Create: `src/paper_search/retrieval/routing.py`
- Create: `tests/unit/test_retrieval_snapshot_adapters.py`
- Create: `tests/unit/test_routing.py`
- Modify: `src/paper_search/retrieval/openalex.py`
- Modify: `src/paper_search/retrieval/semantic_scholar.py`
- Modify: `src/paper_search/retrieval/__init__.py`
- Modify: `tests/unit/test_openalex.py`
- Modify: `tests/unit/test_semantic_scholar.py`

**Interfaces:**

```python
class RoutedSubquery(DomainModel):
    subquery_id: NonEmptyStr
    text: NonEmptyStr
    providers: tuple[Literal["openalex"], ...] | tuple[
        Literal["openalex"], Literal["semantic_scholar"]
    ]
    routing_reason: Literal[
        "primary",
        "high_priority_supplement",
        "uncovered_constraint_supplement",
    ]


def route_baseline_subqueries(
    plan: SearchPlan,
    *,
    min_openalex_calls: int = 3,
    max_openalex_calls: int = 6,
    max_semantic_scholar_calls: int = 2,
) -> list[RoutedSubquery]: ...


class LiveCaptureSearchProvider(SearchProvider): ...
class ReplaySearchProvider(SearchProvider): ...
```

Provider decoders are pure functions used by both modes. Identities include method, endpoint, canonical query/body, adapter version, page/batch operation, and result limit. OpenAlex pagination emits ordered per-page refs. Semantic Scholar GET and POST bodies produce different keys.

Retry policy is exact:

```python
retryable = timeout or status_code == 429 or 500 <= status_code <= 599
delay_seconds = min(8, 2**retry_index) + jitter()
max_attempts = 3
```

Every attempt remains under the original request reservation and hard deadline.

**Steps:**

- [ ] Add failing tests for identical live/replay normalized papers, OpenAlex page ordering, Semantic Scholar GET/POST identities, safe request canonicalization, successful-byte capture, failed-body exclusion, replay miss, replay network tripwire, and retry accounting.
- [ ] Add routing tests proving OpenAlex receives 3–6 bounded calls, Semantic Scholar receives only the top 1–2 qualifying subqueries, and `"either"` never causes unconditional dual fan-out.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_openalex.py tests/unit/test_semantic_scholar.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_routing.py -q
```

Expected initial result: FAIL because Provider snapshot wrappers and routing do not exist.

- [ ] Extract shared pure response decoders from both Provider clients.
- [ ] Add capture/replay wrappers without changing `SearchProvider` signatures.
- [ ] Include all snapshot refs in returned safe provenance; replace OpenAlex's JSON-encoded `cache_keys` asymmetry.
- [ ] Implement deterministic bounded routing.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_openalex.py tests/unit/test_semantic_scholar.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_routing.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/retrieval tests/unit/test_openalex.py tests/unit/test_semantic_scholar.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_routing.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/retrieval
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/retrieval/openalex.py src/paper_search/retrieval/semantic_scholar.py src/paper_search/retrieval/snapshot_adapters.py src/paper_search/retrieval/routing.py src/paper_search/retrieval/__init__.py tests/unit/test_openalex.py tests/unit/test_semantic_scholar.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_routing.py
git commit -m "feat: add provider capture replay and routing"
```

---

### Task 4: Add persistent run and project budget ledgers

**Files:**

- Create: `src/paper_search/control/ledger.py`
- Create: `tests/unit/test_budget_ledger.py`
- Modify: `src/paper_search/control/__init__.py`
- Modify: `tests/unit/test_budget.py`

**Interfaces:**

```python
class LedgerReservation(DomainModel):
    reservation_id: NonEmptyStr
    run_id: NonEmptyStr
    query_id: NonEmptyStr
    estimate: UsageEstimate
    state: Literal["reserved", "settled", "failed"]


class LedgerReport(DomainModel):
    run_id: NonEmptyStr
    reserved: UsageEstimate
    actual: UsageActual
    run_cap_cny: MoneyCny
    project_actual_cny: MoneyCny
    project_soft_stop_cny: MoneyCny
    project_hard_cap_cny: MoneyCny
    within_caps: bool


class SQLiteBudgetLedger:
    def reserve(
        self,
        *,
        run_id: str,
        query_id: str,
        estimate: UsageEstimate,
        run_cap_cny: MoneyCny,
    ) -> LedgerReservation: ...

    def settle(
        self,
        reservation: LedgerReservation,
        actual: UsageActual,
    ) -> None: ...

    def fail(
        self,
        reservation: LedgerReservation,
        actual: UsageActual,
    ) -> None: ...

    def report(self, run_id: str) -> LedgerReport: ...
```

SQLite tables use primary-keyed reservation IDs, `BEGIN IMMEDIATE` for reserve/settle transitions, and canonical integer micro-CNY storage. Settlement is exactly once. Recovery marks expired reservations failed with their already committed actual usage; it never silently releases known spend.

Caps:

- request: bound `configs/budget_balanced.yaml`, CNY 0.30;
- dev run: approved query count × CNY 0.30, expected 60 → CNY 18.00;
- validation run: approved query count × CNY 0.30, expected 30 → CNY 9.00;
- project soft stop: CNY 160.00;
- project hard cap: CNY 200.00.

Replay produces a run-local zero-external-spend report and never writes the live project ledger.

**Steps:**

- [ ] Add failing tests for atomic concurrent reserve, exact-once settlement, unknown cost, over-request, over-run, project soft stop, project hard cap, restart recovery, stale reservation, and replay non-mutation.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_budget.py tests/unit/test_budget_ledger.py -q
```

Expected initial result: FAIL because the hierarchical ledger is absent.

- [ ] Implement the SQLite schema and transactional state transitions.
- [ ] Keep `HardBudgetController` as the request-level authority; the new ledger reserves once before a query and settles after its request controller is final.
- [ ] Store CNY as integer micro-units and expose `Decimal` at model boundaries.
- [ ] Add crash/restart recovery tests using a real temporary SQLite file.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_budget.py tests/unit/test_budget_ledger.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/control tests/unit/test_budget.py tests/unit/test_budget_ledger.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/control
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/control/__init__.py src/paper_search/control/ledger.py tests/unit/test_budget.py tests/unit/test_budget_ledger.py
git commit -m "feat: add hierarchical budget ledger"
```

---

### Task 5: Preserve pipeline evidence and implement `SearchApplicationService`

**Files:**

- Create: `src/paper_search/application/service.py`
- Create: `tests/unit/test_application_service.py`
- Modify: `src/paper_search/application/__init__.py`
- Modify: `src/paper_search/pipeline/orchestrator.py`
- Modify: `src/paper_search/pipeline/response.py`
- Modify: `tests/integration/test_orchestrator.py`
- Modify: `tests/unit/test_response.py`

**Interfaces:**

```python
class OrchestratorResult(DomainModel):
    query_analysis: QueryAnalysisResult
    fused_papers: list[FusedPaper]
    high_relevance: list[RankedPaper]
    partial_relevance: list[RankedPaper]
    citation_edges: list[ResolvedCitationEdge]
    provider_results: dict[DependencyName, ProviderResult[list[Paper]]]
    diagnostics: list[DependencyDiagnostic]
    planner_status: PlannerStatus
    trace: list[dict[str, object]]
    usage: UsageActual
    stop_reason: NonEmptyStr
    is_partial: bool
    warnings: list[NonEmptyStr]
    config_hash: Sha256
    prompt_version: NonEmptyStr


class SearchApplicationService:
    async def execute(self, request: SearchRequest) -> SearchExecutionResult: ...

    async def __call__(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResponse: ...
```

`execute()` always returns exactly one typed outcome. `__call__()` returns the response for success and raises `SearchApplicationError(error, usage, diagnostics)` for hard failure. Each request gets a fresh `HardBudgetController` and run ID. Business-result hashing uses the Phase 3 canonical projection helper once introduced; until then, Phase 2 owns a private canonical serializer whose tests are moved unchanged in Phase 3.

Hard-failure/partial rules:

- snapshot miss, planner transport/auth failure, both Providers failing before candidates, or pre-useful-work exhaustion → `SearchFailure`;
- one Provider failure, rules fallback, or exhaustion after committed useful work → `SearchSuccess` with `is_partial=true`;
- all diagnostics, actual usage, snapshot refs, fused scores, source ranks, relevance groups, and citation edges survive conversion.

**Steps:**

- [ ] Add failing service tests for fresh request scope, success/failure discrimination, one-Provider degradation, both-Provider failure, rules fallback, trace suppression, snapshot refs across multiple subqueries/pages, safe warnings, and deterministic business hash.
- [ ] Add failing orchestrator/response tests showing current fusion scores and earlier provenance are lost.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_application_service.py tests/integration/test_orchestrator.py tests/unit/test_response.py -q
```

Expected initial result: FAIL because the application service and evidence-preserving result are absent.

- [ ] Replace `MinimalSearchResult` with `OrchestratorResult`, retaining a temporary import alias only if legacy tests require it.
- [ ] Aggregate every dependency result's provenance rather than copying only the last result.
- [ ] Preserve truthful `FusedPaper` and optional evidence through response conversion.
- [ ] Implement request-scoped service execution and typed hard-failure mapping.
- [ ] Ensure public warning/error strings come only from fixed safe templates.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_application_service.py tests/integration/test_orchestrator.py tests/unit/test_response.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/application/service.py src/paper_search/pipeline tests/unit/test_application_service.py tests/integration/test_orchestrator.py tests/unit/test_response.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/application/service.py src/paper_search/pipeline
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/application/__init__.py src/paper_search/application/service.py src/paper_search/pipeline/orchestrator.py src/paper_search/pipeline/response.py tests/unit/test_application_service.py tests/integration/test_orchestrator.py tests/unit/test_response.py
git commit -m "feat: add canonical search application service"
```

---

### Task 6: Implement mode binding and `CompositionRoot`

**Files:**

- Create: `src/paper_search/application/modes.py`
- Create: `src/paper_search/application/composition.py`
- Create: `tests/integration/test_application_composition.py`
- Modify: `src/paper_search/application/__init__.py`
- Modify: `src/paper_search/config.py`
- Modify: `tests/unit/test_config.py`

**Interfaces:**

```python
class ModeBinding(DomainModel):
    mode: SearchMode
    network_authorized: bool
    snapshot_set_id: NonEmptyStr | None
    snapshot_manifest_sha256: Sha256 | None


@dataclass(frozen=True)
class ApplicationBundle:
    service: SearchApplicationService
    readiness_probe: Callable[[], ReadyHealthResponse]
    config_hash: Sha256
    artifact_factory: ArtifactFactory
    experiment_id: Literal["main-baseline"]
    source_git_sha: str
    prompt_version: Literal["query-analyze-v1"]
    mode_binding: ModeBinding


class CompositionRoot:
    @classmethod
    def compose(
        cls,
        *,
        lock_path: Path,
        mode: SearchMode,
        artifact_root: Path,
        output_root: Path,
        snapshot_manifest_path: Path | None = None,
        network_authorized: bool = False,
        environ: Mapping[str, str] | None = None,
    ) -> ApplicationBundle: ...
```

Readiness is computed from the immutable mode binding and last explicitly authorized probe evidence; calling the readiness function performs no dependency call.

**Steps:**

- [ ] Add failing composition tests for replay success, replay missing/mismatched/out-of-root manifest, replay network tripwire, live missing each authorization key, secret absence from bundle/repr, request-local controller creation, immutable snapshot binding, and optional-module absence.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/integration/test_application_composition.py tests/unit/test_config.py -q
```

Expected initial result: FAIL because modes and composition do not exist.

- [ ] Implement mode validation before constructing dependencies.
- [ ] In replay mode, construct only readers/replay adapters.
- [ ] In live mode, resolve secrets at composition time, construct live clients/capture store, and omit secret values from all returned objects.
- [ ] Bind exact baseline identities, timeouts `5/20/20/5`, at most three attempts, routing limits, and optional modules off.
- [ ] Implement non-billable mode-aware readiness.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/integration/test_application_composition.py tests/unit/test_config.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/application src/paper_search/config.py tests/integration/test_application_composition.py tests/unit/test_config.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/application src/paper_search/config.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/application/__init__.py src/paper_search/application/modes.py src/paper_search/application/composition.py src/paper_search/config.py tests/integration/test_application_composition.py tests/unit/test_config.py
git commit -m "feat: add replay and live composition root"
```

---

### Task 7: Add atomic smoke capture and the stable CLI root

**Files:**

- Create: `src/paper_search/application/artifacts.py`
- Create: `src/paper_search/cli.py`
- Create: `tests/integration/test_smoke_cli.py`
- Modify: `src/paper_search/application/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_packaging.py`

**Interfaces:**

```python
class CaptureSession:
    @property
    def work_dir(self) -> Path: ...

    @property
    def snapshot_store(self) -> DependencyCaptureStore: ...

    def record_execution(self, result: SearchExecutionResult) -> None: ...
    def seal(self) -> tuple[DependencySnapshotManifestV2, ReplayLock]: ...
    def publish(self) -> Path: ...
    def fail(self, code: SearchErrorCode) -> Path: ...


class ArtifactFactory:
    def start_capture(
        self,
        *,
        run_id: str,
        input_lock_bytes: bytes,
    ) -> CaptureSession: ...


def build_parser() -> argparse.ArgumentParser: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

The emitted replay lock copies the input lock's `runtime_allow_live` capability while adding `snapshot_set_id` and `snapshot_manifest_sha256`. That bit never changes replay behavior; it only allows a later Phase 4 server to construct a separately authorized request-scoped live service.

Add:

```toml
[project.scripts]
paper-search = "paper_search.cli:main"
```

`smoke` contract:

```text
paper-search smoke
  --lock LOCK
  --output-root ROOT
  [--mode replay|live]
  [--snapshot-manifest MANIFEST]
  [--allow-network]
```

Replay is default. Live success order is execute → record → seal response bytes → write manifest → emit replay lock → validate smoke artifact → publish final directory. A capture that cannot seal/publish never returns command success.

**Steps:**

- [ ] Add failing CLI tests for help text, replay default, required replay manifest, replay tripwire, live two-key authorization, safe stdout/stderr, incomplete/failed directory behavior, manifest-bound replay lock, and byte-identical repeated replay projection.
- [ ] Add packaging test for the console entry point.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/integration/test_smoke_cli.py tests/test_packaging.py -q
```

Expected initial result: FAIL because CLI/artifact factory are absent.

- [ ] Implement the parser without reading environment or opening clients at module import.
- [ ] Implement sibling temporary work directories and same-filesystem final rename.
- [ ] Copy exact input lock bytes to `config.lock.yaml`; emit replay lock only after successful snapshot sealing.
- [ ] Keep smoke artifacts intentionally minimal; Phase 3 expands the same artifact module for formal evaluation.
- [ ] Re-run focused tests, CLI help, and Phase 2 regression.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/integration/test_smoke_cli.py tests/test_packaging.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file paper-search --help
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/unit/test_dependency_snapshot.py tests/unit/test_llm_snapshot_adapters.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_budget_ledger.py tests/unit/test_application_service.py tests/integration/test_application_composition.py tests/integration/test_smoke_cli.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/application/__init__.py src/paper_search/application/artifacts.py src/paper_search/cli.py pyproject.toml tests/integration/test_smoke_cli.py tests/test_packaging.py
git commit -m "feat: add integrated smoke command"
```

---

## Phase 2 Exit Evidence

- Gate 1: a replay fixture runs end to end twice with byte-identical business projections and a socket/DNS tripwire records zero attempts.
- Gate 2 engineering path: fake-live capture proves exact-byte sealing, pricing, hard budgets, replay-lock emission, and replay equivalence.
- Real Gate 2 remains operator-authorized and cannot run until Gate 0 passes with real pricing/readiness evidence and the user explicitly authorizes network use.
- `SearchApplicationService` is the only production search boundary; evaluation, API, and UI must consume `ApplicationBundle` rather than concrete adapters.
