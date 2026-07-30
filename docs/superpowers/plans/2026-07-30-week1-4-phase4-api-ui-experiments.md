# Week 1–4 Phase 4 API, UI, and Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the unified Replay/Live service through FastAPI and a browser UI, then wire all Week 3–4 advanced modules behind explicit default-off experiment identities.

**Architecture:** FastAPI consumes the Phase 2 `SearchApplicationService`/`ApplicationBundle` and Phase 3 atomic capture facilities; it does not construct a pipeline. A service router keeps the server bound to one replay snapshot set for its lifetime and may create request-scoped live services only when all three live authorization keys pass. Browser JavaScript posts the canonical `SearchRequest` to `/v1/search` and renders the canonical response. Optional stages are built only by an experiment registry and share the same one-round executor, budgets, snapshots, and evaluation path.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, browser-native HTML/CSS/JavaScript, httpx process tests, existing ranking/graph/evolution modules, pytest, Ruff, mypy.

## Global Constraints

- Phases 1–3 are prerequisites. The API/UI consumes their contracts, service, artifacts, locks, validator, and replay/capture adapters.
- `/v1/search`, `paper-search smoke`, and `paper-search evaluate` use the same `SearchApplicationService`; there is no UI-only or evaluation-only production pipeline.
- Invalid FastAPI request bodies map to stable `invalid_request` HTTP 400, not default 422.
- Broad exception handling must not collapse typed authorization, snapshot, budget, config, attempt, or integrity errors into generic 503.
- Replay is the server default and remains structurally offline.
- A live HTTP request requires all three keys: lock `runtime_allow_live=true`, server `--allow-live`, and request `"mode":"live"`.
- The server returns live HTTP 200 only after the request-scoped capture seals and validates. Publication failure is not a successful search response.
- Readiness is cached safe state; endpoints do not make recurring billable probes or expose credential values.
- Browser code receives only the canonical public response/error contract; it cannot submit a filesystem path or switch the bound snapshot set.
- Main baseline constructs no optional stage and uses fixed one round.
- Optional implementation does not imply promotion. Promotion occurs only after Gate 6 evidence and separate approval.
- Correct the factually wrong `citation-expansion.citation_expansion` and `llm-rerank.llm_rerank` flags together with their tests.
- Provider-backed citation expansion and LLM reranking are async and budgeted. Never call `asyncio.run()` inside the running service.
- Keep the mock server available as a legacy test tool until unified process tests pass; do not install its irreversible process-global audit hook in the unified server.
- Documentation status changes remain conditional on Gate 0 and the matching delivery Gate.

---

## File Structure and Ownership

### Create

- `src/paper_search/api/routing.py` — typed application outcome to HTTP mapping and replay/live service router.
- `src/paper_search/ui/static/index.html`
- `src/paper_search/ui/static/app.js`
- `src/paper_search/ui/static/styles.css`
- `src/paper_search/application/experiments.py`
- `src/paper_search/graph/provider_stage.py` — async budgeted citation stage.
- `src/paper_search/ranking/llm_stage.py` — async budgeted rerank stage.
- `tests/cli/test_serve.py`
- `tests/integration/test_serve_process.py`
- `tests/application/test_experiments.py`
- `tests/e2e/test_dual_mode_serve.py`

### Modify

- `src/paper_search/api/app.py`
- `src/paper_search/api/contracts.py` only if Phase 1 compatibility cleanup remains.
- `src/paper_search/api/service.py` only for mock compatibility.
- `src/paper_search/ui/app.py`
- `src/paper_search/ui/__init__.py`
- `src/paper_search/cli.py`
- `src/paper_search/application/composition.py`
- `src/paper_search/application/artifacts.py`
- `src/paper_search/pipeline/orchestrator.py`
- `src/paper_search/config.py`
- `configs/base.yaml`
- `configs/ablations.yaml`
- `src/paper_search/evaluation/ablations.py`
- `pyproject.toml`
- existing API/UI/experiment/evolution/packaging tests.

### Documentation Modified Only After Matching Evidence

- `README.md`
- `docs/architecture/current-system.md`
- `docs/demo/demo-runbook.md`
- `docs/deployment/new-environment-checklist.md`
- `docs/limitations-and-risks.md`
- `data/manifest.json` and data docs only after Gate 0.

---

### Task 1: Implement typed FastAPI errors and mode-aware readiness

**Files:**

- Create: `src/paper_search/api/routing.py`
- Modify: `src/paper_search/api/app.py`
- Modify: `src/paper_search/api/service.py`
- Modify: `tests/api/test_app.py`
- Modify: `tests/integration/test_api.py`

**Interfaces:**

```python
HTTP_STATUS_BY_SEARCH_ERROR: Final[dict[SearchErrorCode, int]] = {
    "invalid_request": 400,
    "live_not_authorized": 403,
    "config_mismatch": 409,
    "validation_attempt_conflict": 409,
    "budget_exhausted": 429,
    "snapshot_unavailable": 503,
    "dependency_failure": 503,
    "integrity_failure": 500,
    "internal_error": 500,
}


class SearchServiceRouter:
    async def execute(self, request: SearchRequest) -> SearchExecutionResult: ...
    def readiness(self) -> ReadyHealthResponse: ...


def create_app(
    service_router: SearchServiceRouter | None = None,
) -> FastAPI: ...
```

`SearchServiceRouter` delegates replay requests to the process-bound replay service. It delegates live requests to the authorized request-scoped live factory and waits for atomic capture publication before returning success. It never accepts snapshot paths from request data.

**Steps:**

- [ ] Add failing API tests for every stable error code/status mapping, Pydantic validation error → 400, unexpected exception → `internal_error` 500, partial success → 200, and exact error body.
- [ ] Add readiness tests for mode, snapshot identity, ordered dependency state, cached probe timestamp, and zero external calls.
- [ ] Add router tests for replay/live service selection and all three live keys.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/api/test_app.py tests/integration/test_api.py -q
```

Expected initial result: FAIL because the app still maps every service exception to one generic 503 and FastAPI returns 422.

- [ ] Implement a `RequestValidationError` handler returning the fixed safe `invalid_request` body.
- [ ] Map typed outcomes before the unexpected-exception boundary.
- [ ] Keep fixed safe details; never return exception text.
- [ ] Implement cached readiness without dependency calls.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/api/test_app.py tests/integration/test_api.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/api tests/api tests/integration/test_api.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/api
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/api/routing.py src/paper_search/api/app.py src/paper_search/api/service.py tests/api/test_app.py tests/integration/test_api.py
git commit -m "feat: add typed API routing"
```

---

### Task 2: Replace the separate UI search composition with browser API calls

**Files:**

- Create: `src/paper_search/ui/static/index.html`
- Create: `src/paper_search/ui/static/app.js`
- Create: `src/paper_search/ui/static/styles.css`
- Modify: `src/paper_search/ui/app.py`
- Modify: `src/paper_search/ui/__init__.py`
- Modify: `src/paper_search/api/app.py`
- Modify: `pyproject.toml`
- Modify: `tests/ui/test_app.py`
- Modify: `tests/test_packaging.py`

**Interfaces:**

```python
def install_ui(app: FastAPI) -> None: ...
```

Browser request construction:

```javascript
const request = {
  query_id: `ui-${crypto.randomUUID()}`,
  query: queryInput.value,
  budget_profile: "balanced",
  include_trace: true,
  mode: modeSelect.value,
};
```

The page renders:

- selected papers, titles, IDs, RRF/optional scores, and source ranks;
- high/partial relevance evidence and citation edges;
- actual usage;
- partial state, planner fallback, dependency statuses, safe warnings, and stop reason;
- execution mode, snapshot capture time/set identity, config hash, and run ID;
- typed safe error code/detail for non-200 responses.

The page contains no service-side search form handler and no import of `evaluation.runner.PipelineResult`.

**Steps:**

- [ ] Replace UI tests with static asset and unified route tests.
- [ ] Add failing tests asserting `app.js` posts JSON to `/v1/search`, sends mode explicitly, handles loading/error/partial states, and never submits a manifest path.
- [ ] Add a packaging test that all three assets are included in built distributions.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/ui/test_app.py tests/api/test_app.py tests/test_packaging.py -q
```

Expected initial result: FAIL because the current UI directly calls an evaluation-only service.

- [ ] Implement semantic accessible HTML with query input, mode selector, status region, result list, provenance panel, and diagnostic panel.
- [ ] Implement browser `fetch("/v1/search", ...)` with cancellation and stale-response protection.
- [ ] Render all user/external text with DOM `textContent`, never HTML interpolation.
- [ ] Install UI routes/assets into the unified FastAPI app.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/ui/test_app.py tests/api/test_app.py tests/test_packaging.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/ui src/paper_search/api/app.py tests/ui/test_app.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/ui src/paper_search/api/app.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/ui/static/index.html src/paper_search/ui/static/app.js src/paper_search/ui/static/styles.css src/paper_search/ui/app.py src/paper_search/ui/__init__.py src/paper_search/api/app.py pyproject.toml tests/ui/test_app.py tests/api/test_app.py tests/test_packaging.py
git commit -m "feat: serve browser UI through canonical API"
```

---

### Task 3: Add the stable `paper-search serve` lifecycle

**Files:**

- Create: `tests/cli/test_serve.py`
- Create: `tests/integration/test_serve_process.py`
- Modify: `src/paper_search/cli.py`
- Modify: `src/paper_search/application/composition.py`
- Modify: `src/paper_search/application/artifacts.py`
- Modify: `tests/test_packaging.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ServerApplicationBundle:
    replay: ApplicationBundle
    live_factory: Callable[[], ApplicationBundle] | None
    capture_artifact_factory: ArtifactFactory
    service_router: SearchServiceRouter


class CompositionRoot:
    @classmethod
    def compose_server(
        cls,
        *,
        replay_lock_path: Path,
        snapshot_manifest_path: Path,
        artifact_root: Path,
        capture_output_root: Path,
        live_authorized: bool,
        environ: Mapping[str, str] | None = None,
    ) -> ServerApplicationBundle: ...
```

Command:

```text
paper-search serve
  --lock REPLAY_LOCK
  --mode replay
  --snapshot-manifest MANIFEST
  --capture-output-root RUNS
  [--allow-live]
  [--host 127.0.0.1]
  [--port 8000]
```

The replay lock may set `runtime_allow_live` true or false. `--allow-live` is rejected if the lock forbids live. The replay service and manifest binding never change during process lifetime. Each live request receives a new capture session/service and publishes exactly one validated request run before HTTP success.

**Steps:**

- [ ] Add failing parser tests for required arguments, replay-only mode, loopback default, invalid live authorization, and safe error output.
- [ ] Add process tests for startup readiness, replay search, graceful client cleanup, signal shutdown, port conflict, and import-without-side-effects.
- [ ] Add live-factory tests proving each request is isolated and capture publication precedes response completion.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/cli/test_serve.py tests/integration/test_serve_process.py tests/test_packaging.py -q
```

Expected initial result: FAIL because `serve` and server composition are absent.

- [ ] Implement `compose_server()` from the same replay/live constructors used by smoke/evaluate.
- [ ] Use FastAPI lifespan to close HTTP clients, snapshot readers, and ledger handles.
- [ ] Add the Uvicorn invocation only inside the command handler.
- [ ] Ensure module import reads no credentials and opens no network/file ledger.
- [ ] Re-run focused tests and CLI help.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/cli/test_serve.py tests/integration/test_serve_process.py tests/test_packaging.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file paper-search serve --help
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/cli.py src/paper_search/application/composition.py tests/cli/test_serve.py tests/integration/test_serve_process.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/cli.py src/paper_search/application/composition.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/cli.py src/paper_search/application/composition.py src/paper_search/application/artifacts.py tests/cli/test_serve.py tests/integration/test_serve_process.py tests/test_packaging.py
git commit -m "feat: add unified serve command"
```

---

### Task 4: Wire explicit experiment identities and async optional stages

**Files:**

- Create: `src/paper_search/application/experiments.py`
- Create: `src/paper_search/graph/provider_stage.py`
- Create: `src/paper_search/ranking/llm_stage.py`
- Create: `tests/application/test_experiments.py`
- Modify: `src/paper_search/application/composition.py`
- Modify: `src/paper_search/pipeline/orchestrator.py`
- Modify: `src/paper_search/config.py`
- Modify: `configs/base.yaml`
- Modify: `configs/ablations.yaml`
- Modify: `src/paper_search/evaluation/ablations.py`
- Modify: `tests/evaluation/test_ablations.py`
- Modify: `tests/integration/test_orchestrator.py`
- Modify: `tests/integration/test_evolution_strategies.py`

**Interfaces:**

```python
ExperimentName = Literal[
    "main-baseline",
    "embedding",
    "citation-expansion",
    "llm-rerank",
    "fixed-two-round",
    "adaptive-evolution",
]


class ExperimentFlags(DomainModel):
    embedding: bool = False
    citation_expansion: bool = False
    constraint_reranking: bool = False
    fixed_two_round: bool = False
    adaptive_evolution: bool = False


class ExperimentDefinition(DomainModel):
    name: ExperimentName
    flags: ExperimentFlags
    strategy: Literal[
        "fixed-one-round", "fixed-two-round", "adaptive-evolution"
    ]


@dataclass(frozen=True)
class ExperimentComponents:
    embedding_ranker: EmbeddingRankingStage | None
    citation_expander: AsyncCitationExpansionStage | None
    constraint_reranker: AsyncConstraintRerankingStage | None
    evolution_strategy: EvolutionStrategy


def load_experiment_definition(
    name: ExperimentName,
    *,
    ablation_config: Path,
) -> ExperimentDefinition: ...


def build_experiment_components(
    definition: ExperimentDefinition,
    *,
    dependencies: ExperimentDependencyFactory,
) -> ExperimentComponents: ...
```

Async stage protocols:

```python
class AsyncCitationExpansionStage(Protocol):
    async def expand(
        self,
        seeds: list[Paper],
        *,
        controller: HardBudgetController,
    ) -> CitationExpansionResult: ...


class AsyncConstraintRerankingStage(Protocol):
    async def rerank(
        self,
        papers: list[Paper],
        constraints: list[str],
        *,
        controller: HardBudgetController,
    ) -> ConstraintRerankResult: ...
```

Exact registry behavior:

- `main-baseline`: all false, fixed one round;
- `embedding`: embedding true only;
- `citation-expansion`: citation expansion true only;
- `llm-rerank`: constraint/LLM reranking true only;
- `fixed-two-round`: fixed two round true only;
- `adaptive-evolution`: adaptive evolution true only.

`EvolutionCoordinator` wraps the shared single-round executor for multi-round strategies. It is not a post-retrieval stage.

**Steps:**

- [ ] Add failing registry tests for exact names, exact flags, forbidden combinations, main-baseline negative construction, and no lazy optional dependency import for baseline.
- [ ] Change the two wrong YAML expectations to require `citation_expansion: true` and `llm_rerank`/`constraint_reranking: true` for their named cases.
- [ ] Add async/budget tests for Provider citation calls and LLM rerank calls, including capture/replay refs, failure degradation, and no nested event loop.
- [ ] Add evolution tests proving fixed-two/adaptive wrap the shared executor.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/application/test_experiments.py tests/evaluation/test_ablations.py tests/integration/test_orchestrator.py tests/integration/test_evolution_strategies.py -q
```

Expected initial result: FAIL because the registry/adapters are absent and two YAML flags are false.

- [ ] Implement exact registry parsing and component construction.
- [ ] Treat baseline planning and bounded multi-source routing as mandatory composition behavior; do not interpret legacy ablation keys such as `query_planning=false` or `multi_source=false` as permission to disable them.
- [ ] Await optional Provider/LLM stages inside the orchestrator under the same request budget.
- [ ] Route multi-round definitions through `EvolutionCoordinator` and the shared one-round executor.
- [ ] Keep `configs/base.yaml` at `experiment: main-baseline`.
- [ ] Re-run focused tests and static checks.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/application/test_experiments.py tests/evaluation/test_ablations.py tests/integration/test_orchestrator.py tests/integration/test_evolution_strategies.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check src/paper_search/application/experiments.py src/paper_search/graph/provider_stage.py src/paper_search/ranking/llm_stage.py src/paper_search/pipeline/orchestrator.py src/paper_search/evaluation/ablations.py tests/application/test_experiments.py tests/evaluation/test_ablations.py
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src/paper_search/application/experiments.py src/paper_search/graph/provider_stage.py src/paper_search/ranking/llm_stage.py src/paper_search/pipeline/orchestrator.py
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add src/paper_search/application/experiments.py src/paper_search/application/composition.py src/paper_search/graph/provider_stage.py src/paper_search/ranking/llm_stage.py src/paper_search/pipeline/orchestrator.py src/paper_search/config.py configs/base.yaml configs/ablations.yaml src/paper_search/evaluation/ablations.py tests/application/test_experiments.py tests/evaluation/test_ablations.py tests/integration/test_orchestrator.py tests/integration/test_evolution_strategies.py
git commit -m "feat: wire default-off experiment modules"
```

---

### Task 5: Add dual-mode process end-to-end coverage

**Files:**

- Create: `tests/e2e/test_dual_mode_serve.py`
- Modify: `tests/integration/test_serve_process.py`
- Modify: `tests/integration/test_mock_server_process.py` only to mark it legacy, not to remove coverage.

**Interfaces:**

The process fixture starts `paper-search serve` on an OS-assigned loopback port using a sealed replay fixture. Fake-live uses in-process deterministic HTTP transports and a fixture pricing policy; it does not contact the internet.

**Steps:**

- [ ] Add a replay E2E test with socket/name-resolution tripwires for all outbound destinations except the loopback test client.
- [ ] Assert UI and direct API produce the same selected IDs and visible provenance for the same request.
- [ ] Assert live returns 403 when lock permission, server flag, or request mode is missing.
- [ ] Assert authorized fake-live publishes exactly one validated request capture before HTTP 200.
- [ ] Assert failed/interrupted capture never appears complete and never returns HTTP 200.
- [ ] Assert a replay request on a live-capable server remains replay-only.
- [ ] Assert repeated replay produces byte-identical canonical business projection.
- [ ] Run focused tests.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/e2e/test_dual_mode_serve.py tests/integration/test_serve_process.py tests/integration/test_mock_server_process.py -q
```

Expected initial result: FAIL until the complete API/UI/server lifecycle is wired.

- [ ] Fix only defects revealed by the E2E tests; do not introduce an alternate test-only production composition.
- [ ] Re-run focused tests and the full suite.

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/e2e/test_dual_mode_serve.py tests/integration/test_serve_process.py tests/integration/test_mock_server_process.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add tests/e2e/test_dual_mode_serve.py tests/integration/test_serve_process.py tests/integration/test_mock_server_process.py
git commit -m "test: cover dual-mode service end to end"
```

---

### Task 6: Perform real-browser Gate 5 acceptance

**Files:**

- No source changes unless the browser check reveals a reproducible defect.
- Access-controlled create: one live request capture if the user authorizes the live demonstration.

**Interfaces:**

The browser acceptance record contains:

```text
server command and bound snapshot_set_id
replay request query_id and run_id
visible response-field checklist
browser console error count
browser network request summary
optional authorized live capture run_id and verify-run exit
```

It contains no query text, credential, raw snapshot, or private absolute path.

**Preconditions:**

- Phase 3 dev/validation formal evidence is available.
- Replay capture/manifest pair passes `verify-run` and `compare-replay`.
- The user explicitly authorizes any real live browser request and its hard budget.

**Steps:**

- [ ] Start the unified server in replay mode on loopback using a verified capture.

```powershell
paper-search serve --lock runs/dev-capture/replay.lock.yaml --mode replay --snapshot-manifest runs/dev-capture/snapshot-manifest.json --capture-output-root runs --host 127.0.0.1 --port 8000
```

Expected result: process remains healthy and `/health/ready` reports replay mode and the bound snapshot set.

- [ ] Open `http://127.0.0.1:8000/` in a real browser, submit one representative replay query, and verify papers, evidence, sources, usage, partial/fallback state, warnings, snapshot time, config hash, and run ID are visible.
- [ ] Inspect browser console and network panel: no JavaScript error, one `/v1/search` POST, and no direct external dependency request.
- [ ] Repeat the replay query and verify visible business content is unchanged.
- [ ] If live demonstration is approved, restart with `--allow-live`, select live in the UI, submit one bounded request, and verify its request capture passes:

```powershell
paper-search verify-run runs/live-api-capture
```

Expected result: exit 0; the response visibly identifies live mode and any bounded degradation.

- [ ] Stop the server gracefully and verify no client/ledger lock remains open.
- [ ] Record screenshots and safe run IDs outside source control unless the user explicitly requests committed documentation assets.

---

### Task 7: Reconcile delivery documentation after Gates 0–5

**Files:**

- Modify: `README.md`
- Modify: `docs/architecture/current-system.md`
- Modify: `docs/demo/demo-runbook.md`
- Modify: `docs/deployment/new-environment-checklist.md`
- Conditional modify: `docs/limitations-and-risks.md`
- Conditional modify: `data/manifest.json`
- Conditional modify: `data/README.md`

**Interfaces:**

Documentation claims map only to machine evidence:

```text
V2 frozen status            -> passing Gate0Report
replay/live runtime support -> passing Gates 1 and 2
formal dev/validation       -> verify-run + compare-replay for Gates 3 and 4
API/UI delivery             -> Gate 5 process and browser acceptance
optional promotion          -> separately approved Gate 6 evidence
```

**Steps:**

- [ ] Replace mock-only runtime statements with the verified replay, authorized live, evaluate, verify, compare, and serve commands.
- [ ] Document the one-service architecture and three live authorization keys.
- [ ] Document access control for snapshots, predictions, failures, business results, gold labels, and validation claims.
- [ ] Report only measured aggregate results from validated runs; do not extrapolate or invent performance claims.
- [ ] Update limitations only where a named Gate supplies evidence that the limitation is resolved.
- [ ] Update data status only when Gate 0 report is passing and hashes match.
- [ ] Run a contradiction and secret scan.

```powershell
rg -n -i "mock-only|waiting_for_human_label_freeze|authorization:|bearer |api[_-]?key\s*[:=]|private-gate0|^[A-Z]:\\\\" README.md docs data/manifest.json data/README.md
```

Expected result: no stale status claim, credential value, private root, or absolute local path. Secret environment variable names may remain in deployment instructions without values.

- [ ] Run final engineering verification.

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file paper-search --help
```

Expected result: all commands exit 0.

- [ ] Commit.

```powershell
git add README.md docs/architecture/current-system.md docs/demo/demo-runbook.md docs/deployment/new-environment-checklist.md docs/limitations-and-risks.md data/manifest.json data/README.md
git commit -m "docs: describe integrated baseline delivery"
```

---

### Task 8: Execute Gate 6 optional-module ablations without changing baseline defaults

**Files:**

- Access-controlled create: experiment run directories and aggregate reports.
- No change to `configs/base.yaml` during evidence generation.
- A later promotion commit is allowed only after separate user approval.

**Interfaces:**

Each optional-module evidence bundle records:

```python
class PromotionEvidence(DomainModel):
    experiment: ExperimentName
    dev_run_ids: tuple[NonEmptyStr, NonEmptyStr, NonEmptyStr]
    median_macro_f1_delta: Decimal
    bootstrap_samples: Literal[1000]
    bootstrap_95_lower_bound: Decimal
    validation_macro_f1_drop: Decimal
    policy_sha256: Sha256
    passed: bool
```

**Steps:**

- [ ] Verify Gates 0–5 and the main baseline are complete before running optional cases.
- [ ] For each named optional experiment, run three same-configuration dev comparisons against the same frozen input, snapshot set, budget rules, and measurement policy.
- [ ] Use the existing 1,000-sample bootstrap implementation and report:

```text
median_dev_macro_f1_delta >= +0.01
bootstrap_95_percent_lower_bound >= -0.005
validation_macro_f1_drop <= 0.01
```

- [ ] Keep each module default-off when any rule fails or evidence is incomplete.
- [ ] If a module passes, present its complete evidence and request a separate promotion decision before changing `configs/base.yaml` or a validation lock.
- [ ] Preserve all non-promoted experiment artifacts and report them as non-promoted, not failed implementation.

---

## Phase 4 Exit Evidence

- Gate 5 demonstrates the same replay result through API and a real browser, plus an explicitly authorized captured live request when approved.
- Every public error/status/readiness field follows the normative contract.
- Browser UI posts only canonical requests and exposes no filesystem/snapshot switching.
- The baseline constructs no optional stage.
- Named experiment identities construct only their intended components.
- Gate 6 evidence remains separate from baseline delivery and from any promotion decision.
