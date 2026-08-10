# Query Evolution Prompt Contract Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the locked Query Evolution prompt-delivery contract, prove it with offline tests, and add a three-query LLM-only canary that must pass before a new 55-query probe can be authorized.

**Architecture:** Move deterministic prompt parsing/rendering into one shared LLM module, bind the selected prompt artifact into a v2 probe lock, and make the live analyzer honor any explicitly bound prompt. Extend the diagnostic CLI with a gold-blind canary lock and runner that reuse the strict Query Evolution validator, snapshot store, pricing, controller, and ledger while never constructing OpenAlex dependencies.

**Tech Stack:** Python 3.11, Pydantic 2, PyYAML, `httpx`, SQLite, pytest, Ruff, mypy strict.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-10-query-evolution-prompt-contract-canary-design.md` and the unchanged Gate contracts in `docs/superpowers/specs/2026-08-09-query-evolution-bounded-probe-design.md`.
- Use the existing worktree and branch; preserve untracked `data/budget_ledger.sqlite3` and `deliverables/`.
- Use TDD: observe each new test fail for the intended reason before implementing the minimum fix.
- Do not relax `QueryEvolutionProposal`, mechanical validation, Gate A/B/C, retrieval merging, filtering, RRF, or evidence privacy.
- Do not add compatibility parsing, LLM repair, rules fallback, prompt auto-tuning, or unrelated refactoring.
- Offline implementation and verification must not read `D:\AI Projects\Projects\.env` or access the network.
- A later instruction to “execute the plan” authorizes Tasks 1–5 only. Task 6 requires separate authorization for three DeepSeek logical operations with at most nine retry-inclusive requests; Task 7 requires another authorization for the 55-query DeepSeek/OpenAlex probe.
- The canary reads no gold, identifier map, availability evidence, or OpenAlex credentials; it creates exactly three `evolve` reservations and zero search reservations, allows at most nine LLM attempts, and has a 600-second global timeout.
- Any canary failure stops the workflow. Do not change the prompt, replace samples, or rerun without a new decision.
- Keep the existing failed 55-query run and ledger receipts unchanged.

---

### Task 1: Centralize and harden prompt artifact rendering

**Files:**
- Create: `src/paper_search/llm/prompt_artifacts.py`
- Create: `tests/unit/test_prompt_artifacts.py`
- Modify: `configs/prompts/query_evolve.yaml`
- Modify: `src/paper_search/application/composition.py`
- Test: `tests/integration/test_application_composition.py`

**Interfaces:**
- Produces: `load_prompt_artifact(prompt_bytes: bytes) -> PromptArtifact`
- Produces: `render_prompt_system_message(prompt_bytes: bytes) -> str`
- Consumed later by: probe preflight, probe run, canary run, and application composition.

- [ ] **Step 1: Write failing prompt artifact tests**

Add tests that require deterministic rendering for both existing prompt files, reject malformed YAML and wrong field types, and require the Query Evolution message to contain the exact field names, both enum sets, generated/no-op structures, and the ban on outer wrappers.

```python
def test_query_evolve_message_contains_complete_contract() -> None:
    message = render_prompt_system_message(
        Path("configs/prompts/query_evolve.yaml").read_bytes()
    )

    assert '"subqueries"' in message
    assert '"text"' in message
    assert '"source_facets"' in message
    assert '"strategy"' in message
    assert '"no_op_reason"' in message
    for value in (
        "synonym",
        "entity_alias",
        "facet_combination",
        "task_decomposition",
        "insufficient_grounded_facets",
        "no_novel_query",
    ):
        assert value in message
    assert "Do not return payload or prompt_name wrappers" in message
```

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_prompt_artifacts.py -q
```

Expected: FAIL because `paper_search.llm.prompt_artifacts` does not exist.

- [ ] **Step 2: Implement the strict shared loader**

Use a focused model and two public functions:

```python
EvolutionStrategy = Literal[
    "synonym", "entity_alias", "facet_combination", "task_decomposition"
]
NoOpReason = Literal["insufficient_grounded_facets", "no_novel_query"]


class PromptArtifact(DomainModel):
    name: NonEmptyStr
    version: NonEmptyStr
    temperature: Literal[0]
    response_model: NonEmptyStr
    instructions: list[NonEmptyStr]
    strategies: list[EvolutionStrategy] | None = None
    no_op_reasons: list[NoOpReason] | None = None

    @model_validator(mode="after")
    def validate_query_evolve_contract(self) -> PromptArtifact:
        if self.name == "query_evolve":
            if self.version != "query-evolve-v1":
                raise ValueError("query_evolve prompt version must be query-evolve-v1")
            if self.response_model != "QueryEvolutionProposal":
                raise ValueError("query_evolve response model mismatch")
            if self.strategies != [
                "synonym",
                "entity_alias",
                "facet_combination",
                "task_decomposition",
            ]:
                raise ValueError("query_evolve strategy enum mismatch")
            if self.no_op_reasons != [
                "insufficient_grounded_facets",
                "no_novel_query",
            ]:
                raise ValueError("query_evolve no-op enum mismatch")
        elif self.strategies is not None or self.no_op_reasons is not None:
            raise ValueError("evolution enums require query_evolve")
        return self


def load_prompt_artifact(prompt_bytes: bytes) -> PromptArtifact:
    try:
        raw = yaml.safe_load(prompt_bytes)
    except yaml.YAMLError as error:
        raise ValueError("invalid prompt artifact") from error
    if not isinstance(raw, dict):
        raise ValueError("invalid prompt artifact")
    return PromptArtifact.model_validate(raw)


def render_prompt_system_message(prompt_bytes: bytes) -> str:
    artifact = load_prompt_artifact(prompt_bytes)
    lines = [
        "Respond with a JSON object.",
        f"The JSON object must match the {artifact.response_model} contract.",
        *(f"- {item}" for item in artifact.instructions),
    ]
    return "\n".join(lines)
```

Replace the private `_prompt_system_message` implementation in `application/composition.py` with an import and call to `render_prompt_system_message`; do not change its application-visible output for `query_analyze.yaml`.

- [ ] **Step 3: Expand the Query Evolution artifact with the exact contract**

Keep the existing name/version/enums and replace vague output instructions with explicit, single-schema requirements. The YAML instructions must include these two compact examples verbatim:

```yaml
  - 'Generated form: {"subqueries":[{"text":"string","source_facets":["exact payload facet"],"strategy":"synonym"}],"no_op_reason":null}'
  - 'No-op form: {"subqueries":[],"no_op_reason":"insufficient_grounded_facets"}'
  - Top-level keys must be exactly subqueries and no_op_reason.
  - Each subquery must contain exactly text, source_facets, and strategy.
  - Do not return payload or prompt_name wrappers, Markdown, or extra fields.
```

- [ ] **Step 4: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_prompt_artifacts.py tests/integration/test_application_composition.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/llm/prompt_artifacts.py src/paper_search/application/composition.py tests/unit/test_prompt_artifacts.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/llm/prompt_artifacts.py src/paper_search/application/composition.py
git add -- configs/prompts/query_evolve.yaml src/paper_search/llm/prompt_artifacts.py src/paper_search/application/composition.py tests/unit/test_prompt_artifacts.py tests/integration/test_application_composition.py
git commit -m "fix: bind deterministic prompt contracts"
```

Expected: focused tests pass; Ruff and mypy exit `0`.

---

### Task 2: Bind the selected prompt into a v2 probe lock and live request

**Files:**
- Modify: `src/paper_search/llm/snapshot_adapters.py`
- Modify: `scripts/probe_query_evolution.py`
- Modify: `tests/unit/test_llm_snapshot_adapters.py`
- Modify: `tests/integration/test_query_evolution_probe.py`

**Interfaces:**
- Consumes: `load_prompt_artifact` and `render_prompt_system_message` from Task 1.
- Produces: `ProbePromptBinding(path, sha256, name, version)`.
- Changes: `preflight_probe(..., prompt_config: Path, probe_run_id: str) -> ProbeLock`.
- Produces: `_load_locked_prompt(lock: ProbeLock) -> tuple[bytes, str]`.

- [ ] **Step 1: Write failing adapter and lock tests**

First add `prompt_name: str = "query_analyze"` to the existing `_live` test helper and pass it to `analyzer.generate_json`. Then add a `query_evolve` request test that fails under the current name guard:

```python
def test_live_analyzer_sends_bound_instructions_for_query_evolve(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=_response_bytes({"subqueries": [], "no_op_reason": "no_novel_query"}),
            request=request,
        )

    asyncio.run(
        _live(
            tmp_path,
            httpx.MockTransport(handler),
            SettlementRecorder(),
            prompt_name="query_evolve",
            prompt_instructions="Respond with the QueryEvolutionProposal JSON contract.",
        )
    )

    assert seen[0]["messages"][0]["content"] == (
        "Respond with the QueryEvolutionProposal JSON contract."
    )
```

Extend integration tests so a copied prompt at a caller-supplied repository-relative path is stored in the lock, while prompt path traversal, hash drift, name drift, and version drift fail before an HTTP transport is invoked.

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_llm_snapshot_adapters.py::test_live_analyzer_sends_bound_instructions_for_query_evolve tests/integration/test_query_evolution_probe.py -q
```

Expected: FAIL because `query_evolve` receives only the generic JSON message and `ProbeLock` has no prompt binding.

- [ ] **Step 2: Remove the adapter’s prompt-name special case**

Make the request call use the bound value directly:

```python
response = await self._client.request_response(
    prompt_name=prompt_name,
    payload=payload,
    prompt_instructions=self._prompt_instructions,
)
```

Keep the default `None` behavior unchanged. Do not infer or load prompt files inside the adapter.

- [ ] **Step 3: Upgrade the probe lock and preflight plumbing**

Replace the v1 prompt scalar fields with:

```python
class ProbePromptBinding(DomainModel):
    path: SafeRelativePath
    sha256: Sha256
    name: Literal["query_evolve"]
    version: Literal["query-evolve-v1"]


```

Define `SafeRunId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]` and use it for probe and canary run IDs. Change `ProbeLock.schema_version` from `query-evolution-probe-lock-v1` to `query-evolution-probe-lock-v2`, remove `prompt_sha256` and `prompt_version`, and add `prompt: ProbePromptBinding`. Keep every existing frozen-source, queue, budget, endpoint, limit, ledger, output and self-hash field unchanged. Change `_build_lock_payload` and `preflight_probe` to receive `prompt_config` and `probe_run_id`. Resolve the prompt under `ROOT`, reject paths outside `ROOT`, parse it through Task 1, and write the actual path/hash/name/version. Set `expected_run_directory = f"runs/_diag_query_evolution_{probe_run_id}"`. Wire the CLI’s existing `--prompt-config` argument into this function instead of merely checking that the path exists.

- [ ] **Step 4: Verify the locked artifact before reservations or network**

Implement:

```python
def _load_locked_prompt(lock: ProbeLock) -> tuple[bytes, str]:
    path = (ROOT / lock.prompt.path).resolve(strict=True)
    root = ROOT.resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("locked prompt path is invalid")
    prompt_bytes = path.read_bytes()
    if _sha256_bytes(prompt_bytes) != lock.prompt.sha256:
        raise ValueError("locked prompt hash mismatch")
    artifact = load_prompt_artifact(prompt_bytes)
    if artifact.name != lock.prompt.name or artifact.version != lock.prompt.version:
        raise ValueError("locked prompt identity mismatch")
    return prompt_bytes, render_prompt_system_message(prompt_bytes)
```

Call this from `run_probe` before ledger reservations and before client construction. Pass the returned system message to `LiveCaptureLLMAnalyzer`; use `lock.prompt.sha256` and `lock.prompt.version` for live and replay identities.

- [ ] **Step 5: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_llm_snapshot_adapters.py tests/integration/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/llm/snapshot_adapters.py scripts/probe_query_evolution.py tests/unit/test_llm_snapshot_adapters.py tests/integration/test_query_evolution_probe.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts/probe_query_evolution.py
git add -- src/paper_search/llm/snapshot_adapters.py scripts/probe_query_evolution.py tests/unit/test_llm_snapshot_adapters.py tests/integration/test_query_evolution_probe.py
git commit -m "fix: lock query evolution prompt delivery"
```

---

### Task 3: Add deterministic, gold-blind canary selection and lock contracts

**Files:**
- Modify: `scripts/probe_query_evolution.py`
- Modify: `tests/integration/test_query_evolution_probe.py`

**Interfaces:**
- Produces: `build_probe_context(query_id: str, raw_record: Mapping[str, object]) -> QueryEvolutionContext`.
- Produces: `select_canary_query_ids(lock: ProbeLock, raw_records: Mapping[str, Mapping[str, object]]) -> tuple[str, str, str]`.
- Produces: `preflight_canary(probe_lock_path: Path, ledger_path: Path, canary_run_id: str, output_path: Path) -> CanaryLock`.

- [ ] **Step 1: Write failing selector and privacy tests**

Use synthetic contexts with payload lengths that force an unambiguous minimum, median, and maximum. Add a tie case ordered by query ID, and monkeypatch `read_jsonl`/`IdentifierMap.from_path` to raise if canary preflight attempts to read gold or identifier data.

```python
def test_canary_selects_minimum_median_and_maximum_canonical_payload() -> None:
    selected = select_canary_query_ids(lock, raw_records)
    ranked = sorted(
        lock.query_ids,
        key=lambda query_id: (
            len(_canonical_json(build_probe_context(query_id, raw_records[query_id]).model_dump(mode="json"))),
            query_id,
        ),
    )
    assert selected == (ranked[0], ranked[len(ranked) // 2], ranked[-1])
    assert len(set(selected)) == 3
```

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/integration/test_query_evolution_probe.py -k canary -q
```

Expected: FAIL because the canary interfaces do not exist.

- [ ] **Step 2: Extract one context builder and implement the selector**

Move the duplicated frozen `QuerySpec`/`SearchPlan`/candidate-count construction used by live capture and replay into `build_probe_context`. Serialize each context with the existing `_canonical_json`, rank by `(byte_length, query_id)`, select indices `0`, `len(ranked) // 2`, and `len(ranked) - 1`, and fail if the result does not contain exactly three distinct IDs.

- [ ] **Step 3: Define a self-hashed canary lock and preflight**

Use strict diagnostic-only models:

```python
CanaryReason = Literal[
    "passed",
    "canary_preflight_failed",
    "prompt_binding_failed",
    "contract_canary_failed",
    "canary_dependency_failed",
    "canary_accounting_failed",
    "canary_snapshot_failed",
    "canary_cancelled",
]


class CanaryLimits(DomainModel):
    query_count: Literal[3]
    llm_logical_operations: Literal[3]
    llm_attempts: Literal[9]
    global_timeout_seconds: Literal[600]


class CanaryLock(DomainModel):
    schema_version: Literal["query-evolution-contract-canary-lock-v1"]
    canary_run_id: SafeRunId
    source_probe_lock_sha256: Sha256
    source_run_id: NonEmptyStr
    source_hashes: dict[str, Sha256]
    probe_code_sha256: Sha256
    prompt: ProbePromptBinding
    model_id: Literal["deepseek-v4-flash"]
    endpoint: Literal["https://api.deepseek.com/v1"]
    query_ids: tuple[str, str, str]
    limits: CanaryLimits
    evolve_estimate: UsageEstimate
    ledger_checkpoint_sha256: Sha256
    expected_run_directory: SafeRelativePath
    lock_sha256: Sha256
```

`preflight_canary` loads and verifies the v2 probe lock, verifies only frozen business/execution source hashes and the locked prompt, validates and stores the explicit `canary_run_id`, copies the source run ID and current probe code hash, selects the three IDs, copies the locked `evolve` estimate, records the 3/9/600 limits and current ledger checkpoint, and atomically writes canonical JSON. Set `expected_run_directory = f"runs/_diag_query_evolution_{canary_run_id}"`. It must not accept gold, identifier-map, availability, or env paths.

- [ ] **Step 4: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/integration/test_query_evolution_probe.py -k "canary or preflight" -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check scripts/probe_query_evolution.py tests/integration/test_query_evolution_probe.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy scripts/probe_query_evolution.py
git add -- scripts/probe_query_evolution.py tests/integration/test_query_evolution_probe.py
git commit -m "feat: add query evolution canary lock"
```

---

### Task 4: Implement the three-query LLM-only canary runner

**Files:**
- Modify: `scripts/probe_query_evolution.py`
- Modify: `tests/integration/test_query_evolution_probe.py`

**Interfaces:**
- Consumes: `CanaryLock`, locked prompt loader, context builder, existing controller/pricer/ledger/snapshot components.
- Produces: `run_canary(lock_path: Path, runtime: CanaryRuntime) -> None`.
- CLI: `canary-preflight --probe-lock ... --ledger ... --out ...` and `canary-run --lock ... --env-file ... --ledger ... --allow-live`.

- [ ] **Step 1: Write failing mocked runner tests**

Use `httpx.MockTransport` only. Cover exactly three valid generated/no-op outcomes; one strict-schema failure; dependency, accounting, snapshot and 600-second cancellation failures; source run/hash, probe code, prompt, fixed-limit or checkpoint drift before dispatch; three terminal ledger receipts; sealed snapshot refs; atomic evidence writes; no retry beyond nine total attempts; and zero construction or invocation of `LiveCaptureSearchProvider`.

Add a secret-boundary test that supplies only `LLM_API_KEY` in a temporary env file and makes every OpenAlex environment lookup raise. The canary must still succeed.

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/integration/test_query_evolution_probe.py -k canary_run -q
```

Expected: FAIL because `run_canary` and CLI commands do not exist.

- [ ] **Step 2: Split credential loading by dependency**

Replace the all-or-nothing `_load_secrets` with:

```python
def _load_llm_secret(env_file: Path) -> str:
    values = dotenv_values(env_file)
    value = values.get("LLM_API_KEY") or os.environ.get("LLM_API_KEY")
    if not value:
        raise ValueError("LLM_API_KEY is missing from the authorized env file")
    return value


def _load_openalex_secrets(env_file: Path) -> tuple[tuple[str, ...], str | None]:
    values = dotenv_values(env_file)
    names = ("OPENALEX_API_KEY", *(f"OPENALEX_API_KEY_{index}" for index in range(2, 8)))
    keys = tuple(value for name in names if (value := values.get(name) or os.environ.get(name)))
    if not keys:
        raise ValueError("OPENALEX_API_KEY is missing from the authorized env file")
    return keys, values.get("OPENALEX_MAILTO") or os.environ.get("OPENALEX_MAILTO")
```

The full probe calls both functions; canary calls only `_load_llm_secret`.

- [ ] **Step 3: Implement reservations, capture, evidence, and stop logic**

Before network, verify the canary lock self-hash, source run and hashes, probe code hash, prompt binding, 3/9/600 limits, output non-existence, and ledger checkpoint. Reserve exactly one `evolve` slot per selected query under the canary run ID. Wrap the three-operation batch in `asyncio.timeout(600)`. Reuse `QueryEvolutionGenerator` with `LiveCaptureLLMAnalyzer(prompt_instructions=locked_message)` and settle every slot with actual usage.

Require the locked run directory not to exist, create it, and copy the validated external canary lock into it byte-for-byte before capture. This avoids partial-run ambiguity.

Write only:

```text
canary.lock.json
outcomes.jsonl
snapshots/
result.json
```

`result.json` must contain the fixed reason, three terminal counts, aggregate usage, manifest hash/set ID, ledger checkpoint, and `promoted: true|false`. Set `promoted=true` only when all three outcomes are `generated` or `no_op`, all receipts are terminal, and every snapshot ref belongs to the sealed manifest. Any other state uses the matching fixed reason and stops; do not create a full probe lock.

- [ ] **Step 4: Wire CLI boundaries and verify mocked behavior**

`canary-run` returns exit `2` without `--allow-live`, exit `1` for fixed failures including `canary_preflight_failed` and `canary_cancelled`, and `0` only when `promoted=true`. Automated tests must patch transports and must not read the real `.env`.

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/integration/test_query_evolution_probe.py -k canary -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_query_evolution.py tests/unit/test_llm_snapshot_adapters.py tests/integration/test_query_evolution_probe.py -q
git add -- scripts/probe_query_evolution.py tests/integration/test_query_evolution_probe.py
git commit -m "feat: run bounded query evolution canary"
```

---

### Task 5: Complete offline verification and record the authorization boundary

**Files:**
- Modify: `HANDOFF.md`
- Modify: `docs/retrieval-roadmap.md`

**Interfaces:**
- Produces: an offline-verified implementation and a handoff that stops before any real provider call.

- [ ] **Step 1: Run the complete offline quality gate**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_prompt_artifacts.py tests/unit/test_llm_snapshot_adapters.py tests/unit/test_query_evolution.py tests/integration/test_application_composition.py tests/integration/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check .
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts/probe_query_evolution.py
git diff --check
```

Expected: all tests pass; Ruff, mypy, and diff check exit `0`. If a failure appears, diagnose it before changing code and do not proceed to preflight.

- [ ] **Step 2: Build only the offline v2 source lock and canary lock**

Use unique, explicit run IDs and no env file:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/probe_query_evolution.py preflight `
  --run runs/dev-20260809T061903Z-9bd861e90299 `
  --gold data/dev/gold.jsonl `
  --id-map data/identifier-map.json `
  --availability docs/evidence/gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json `
  --prompt-config configs/prompts/query_evolve.yaml `
  --budget-config configs/budget_balanced.yaml `
  --pricing-policy data/annotation_work/pricing_v1.yaml `
  --ledger data/budget_ledger.sqlite3 `
  --probe-run-id contract-v2-source-20260810 `
  --out runs/_locks/query_evolution_contract-v2-source-20260810/probe.lock.json

& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/probe_query_evolution.py canary-preflight `
  --probe-lock runs/_locks/query_evolution_contract-v2-source-20260810/probe.lock.json `
  --ledger data/budget_ledger.sqlite3 `
  --canary-run-id contract-canary-20260810 `
  --out runs/_locks/query_evolution_contract-canary-20260810/canary.lock.json
```

Expected: both locks validate; canary lock contains exactly three deterministic query IDs, source run/code bindings, 3/9/600 limits, and the current unchanged ledger checkpoint. There is no reservation, `.env` read, or network request.

- [ ] **Step 3: Update the handoff and commit**

Record exact test counts, lock hashes, selected-sample rule, and the boundary that real canary and full probe have not run. Preserve old failed evidence and untracked user files.

```powershell
git status --short
git diff -- docs/evidence runs/candidate.lock.yaml data/budget_ledger.sqlite3
git add -- HANDOFF.md docs/retrieval-roadmap.md
git commit -m "docs: prepare query evolution contract canary"
```

Stop and request authorization for Task 6.

---

### Task 6: Authorization gate — run the real three-query canary

**Files:**
- Private ignored evidence only: `runs/_diag_query_evolution_contract-canary-20260810/`
- Modify after result: `HANDOFF.md`
- Modify after result: `docs/retrieval-roadmap.md`

**Authorization:** Do not execute this task unless the user explicitly authorizes the three-query DeepSeek canary after Task 5 completes.

- [ ] **Step 1: Run one LLM-only canary**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/probe_query_evolution.py canary-run `
  --lock runs/_locks/query_evolution_contract-canary-20260810/canary.lock.json `
  --env-file 'D:\AI Projects\Projects\.env' `
  --ledger data/budget_ledger.sqlite3 `
  --allow-live
```

Expected for promotion: 3/3 strict-valid `generated` or `no_op`, three terminal receipts, no more than nine LLM attempts within 600 seconds, sealed LLM snapshots, zero OpenAlex requests, and `promoted=true`.

- [ ] **Step 2: Apply the stop rule**

If `promoted=false`, record the fixed failure reason and stop. Do not edit the prompt, replace samples, or rerun. If `promoted=true`, record hashes, aggregate usage, and the fact that this is only a contract circuit-breaker.

- [ ] **Step 3: Commit aggregate-only status**

```powershell
git add -- HANDOFF.md docs/retrieval-roadmap.md
git commit -m "docs: record query evolution contract canary"
```

Stop and request separate authorization for Task 7.

---

### Task 7: Authorization gate — rebuild and execute the full 55-query probe

**Files:**
- Private ignored evidence: `runs/_diag_query_evolution_contract-v2-full-20260810/`
- Aggregate evidence and status files only after a completed run.

**Authorization:** Do not execute this task unless Task 6 produced `promoted=true` and the user separately authorizes the 55-query DeepSeek/OpenAlex probe.

- [ ] **Step 1: Rebuild the full lock against the post-canary ledger checkpoint**

Do not reuse the source or canary lock because their ledger checkpoint predates canary usage.

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/probe_query_evolution.py preflight `
  --run runs/dev-20260809T061903Z-9bd861e90299 `
  --gold data/dev/gold.jsonl `
  --id-map data/identifier-map.json `
  --availability docs/evidence/gold-bottleneck-attribution-2026-08-09-doi-contract-retry3.json `
  --prompt-config configs/prompts/query_evolve.yaml `
  --budget-config configs/budget_balanced.yaml `
  --pricing-policy data/annotation_work/pricing_v1.yaml `
  --ledger data/budget_ledger.sqlite3 `
  --probe-run-id contract-v2-full-20260810 `
  --out runs/_locks/query_evolution_contract-v2-full-20260810/probe.lock.json
```

Expected: the lock records the post-canary project checkpoint and a new self-hash.

- [ ] **Step 2: Execute one complete capture → replay → Gate evaluation**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/probe_query_evolution.py run `
  --lock runs/_locks/query_evolution_contract-v2-full-20260810/probe.lock.json `
  --env-file 'D:\AI Projects\Projects\.env' `
  --ledger data/budget_ledger.sqlite3 `
  --allow-live
```

- [ ] **Step 3: Enforce the existing Gate conclusion**

- Gate A failure: B/C remain `not_evaluated`; diagnose evidence integrity only.
- Gate B failure: reject the current Query Evolution retrieval hypothesis; no prompt variant, ranking change, or formal capture.
- Gate B pass and Gate C failure: record recall/ranking bottleneck; no formal capture.
- Gate C pass: record `capture_candidate`; formal capture still requires separate design and authorization.

Commit only aggregate, privacy-safe evidence and status updates after verifying snapshots, replay hash, ledger terminal states, and `git diff --check`.

---

## Final self-review checklist

- [ ] Every requirement in the 2026-08-10 design maps to a task above.
- [ ] The prompt renderer is shared; probe code does not import a private composition helper or duplicate rendering.
- [ ] The caller-supplied prompt path reaches lock construction and is verified before reservations/network.
- [ ] Bound instructions reach `query_evolve`; unbound analyzer behavior and `query_analyze` remain compatible.
- [ ] The canary selection is deterministic min/median/max over canonical payload bytes with query-ID tie-breaking.
- [ ] Canary preflight and run accept no gold, identifier-map, availability, or OpenAlex inputs.
- [ ] The canary creates exactly three LLM ledger slots and zero search slots.
- [ ] The canary lock binds source run/hash, probe code, prompt, ledger checkpoint, estimate, and 3/9/600 limits.
- [ ] Existing schema/mechanical validation and Gate A/B/C are unchanged.
- [ ] Automated tests use mocked transports and never read the real `.env`.
- [ ] Task 5 stops before network; Tasks 6 and 7 each require a distinct explicit authorization.
- [ ] Old failed snapshots, receipts, untracked ledger, and `deliverables/` remain intact.
