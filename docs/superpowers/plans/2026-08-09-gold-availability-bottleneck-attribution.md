# Gold Availability and Bottleneck Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one aggregate-only diagnostic that exactly probes OpenAlex availability for 134 normalized gold works and attributes all 139 normalized query–work associations to retrieval, filtering, or Top-50 loss.

**Architecture:** Add one standalone diagnostic script with three boundaries: sealed-run offline attribution, an injected exact-work OpenAlex probe, and ledger-backed orchestration/output. Reuse existing normalization, pricing, and ledger contracts; do not change production retrieval, filtering, ranking, locks, or formal-run workflows.

**Tech Stack:** Python 3.12, Pydantic v2 domain models, httpx async client/MockTransport, existing `IdentifierMap`, `ActualCostPricer`, `SQLiteBudgetLedger`, pytest, JSON/JSONL sealed artifacts.

## Global Constraints

- Work in `D:\AI Projects\.worktrees\week3` on `codex/project-document-handoff`.
- Treat `docs/superpowers/specs/2026-08-09-gold-availability-bottleneck-attribution-design.md` as the authoritative contract; this plan only specifies implementation order and verification.
- Use source run `runs/dev-20260809T061903Z-9bd861e90299`; require complete/passed status, formal validity, quality pass, and passed zero-valued `provenance-failures`.
- Recompute and require exactly 60 queries, 143 raw gold identifiers, 139 normalized query–work associations, and 134 normalized unique works (128 DOI, 6 OpenAlex IDs).
- Normalize only through existing `normalize_paper_id` and `IdentifierMap.resolve`. Probe only terminal DOI/OpenAlex IDs through OpenAlex single-work GET with `select=id,doi`.
- Never search by arXiv ID, title, author, abstract, full text, or query text.
- Do not run readiness, capture, replay, compare, validation, or candidate-lock rebuild.
- Do not modify production providers, filtering, fusion, selector, ranking, `data/manifest.json`, historical runs, or `runs/candidate.lock.yaml`.
- Use only the contiguous process-environment sequence `OPENALEX_API_KEY`, `OPENALEX_API_KEY_2` through the highest numbered key present. Never read or print `.env` or keys.
- Use `runs/.ledger/formal.sqlite3` through `SQLiteBudgetLedger`; no direct SQL or pricing bypass.
- Reserve one aggregate ledger entry for at most 402 HTTP attempts and settle/fail it with actual priced attempts.
- Never persist raw responses, per-work records, gold/query/paper/request IDs, titles, URLs, secrets, or free text.
- Output must satisfy the exact `gold-bottleneck-attribution-v1` schema, key allowlists, value allowlists, totals, and privacy rules in the design.
- Global input, Gate, provenance, authentication, budget, ledger, or artifact-integrity failures exit nonzero without a report.
- Only per-work `unknown_transient`, `invalid_identifier`, or `integrity_failure` may produce an incomplete aggregate report; then `diagnostic_complete=false` and `recommended_direction=null`.
- Preserve the existing untracked `data/budget_ledger.sqlite3` and `deliverables/`.

---

### Task 1: Implement sealed offline attribution and aggregate report assembly

**Files:**
- Create: `scripts/analyze_gold_bottlenecks.py`
- Create: `tests/scripts/test_analyze_gold_bottlenecks.py`

**Interfaces:**

- `AvailabilityStatus`: exactly `available`, `exact_not_found`, `unknown_transient`, `invalid_identifier`, or `integrity_failure`.
- `PipelineStage`: exactly `selected_top50`, `ranked_outside_top50`, `filtered_out`, or `not_retrieved`.
- `GoldIndex`: immutable counts, terminal-identifier counts, and query-to-work sets.
- `OfflineContext`: immutable `GoldIndex`, one stage per normalized association, and six input hashes.
- `build_gold_index(gold_path: Path, identifier_map: IdentifierMap) -> GoldIndex`.
- `load_offline_context(run: Path, gold_path: Path, id_map_path: Path) -> OfflineContext`.
- `assemble_report(context: OfflineContext, probe: ProbeBatch, usage: DiagnosticUsage) -> dict[str, object]`.
- `assert_safe_report(payload: Mapping[str, object]) -> None`.

- [ ] **Step 1: Add one grouped RED test set for all offline contracts**

Use compact fixtures and parameterization. Cover:

- the fixed 60/143/139/134/128/6 counts without hard-coding them into implementation;
- query-local and global deduplication through existing normalization/resolution;
- rejection of unresolved/non-terminal identifiers;
- source run, Gate, formal-validity, quality, provenance, input-hash, and source-SHA failures;
- exact equality of gold/execution/business-result query sets and `selected ⊆ post-filter ⊆ retrieved` for every query;
- mutually exclusive association stages: selected Top-50, ranked outside Top-50, filtered out, not retrieved;
- preliminary fixed-run stage totals: 14 retrieved, 14 post-filter, 8 selected, hence 0 filter loss and 6 ranking loss;
- availability-to-association reuse, cross-tab totals, query coverage, completeness, tie handling, and allowed decision enum;
- exact top-level/nested schema keys, extra-key rejection, forbidden-key scanning, and value-level privacy rejection;
- atomic preservation of an existing destination if that individual write fails.

Representative test shape:

```python
def test_fixed_inputs_produce_expected_offline_denominators() -> None:
    context = load_offline_context(SOURCE_RUN, GOLD_PATH, ID_MAP_PATH)
    assert (
        context.gold_index.query_count,
        context.gold_index.raw_gold_identifier_count,
        context.gold_index.normalized_query_work_count,
        context.gold_index.unique_work_count,
    ) == (60, 143, 139, 134)
    assert context.gold_index.terminal_identifier_counts == {"doi": 128, "openalex": 6}

```

Implement the direction matrix as a parameterized test covering all four allowed directions plus tied, incomplete, and zero-loss null outcomes.

Run once and confirm RED:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "offline or report or direction or privacy or atomic" -q
```

Expected: failure because the script and pure contracts do not exist.

- [ ] **Step 2: Implement the pure offline boundary**

Implementation order:

1. Load and validate the sealed run and six fixed input hashes before any later side effect.
2. Build the three denominators with existing normalization and identifier-map resolution.
3. Validate query-set equality and the ordered-set invariant, then derive each normalized query–work association's single pipeline stage from `executions.jsonl` and `business-results.jsonl`.
4. Assemble only aggregate counts using the exact schema and direction rules from the design.
5. Validate key and value allowlists recursively before rendering JSON or Markdown; serialize JSON with `allow_nan=False`.
6. Provide an atomic single-file writer; do not add cross-file rollback or backup machinery.

Keep provider records and identifiers internal and ephemeral. The Markdown renderer receives only the already validated aggregate payload.

- [ ] **Step 3: Run one GREEN verification and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "offline or report or direction or privacy or atomic" -q
.\.venv\Scripts\ruff.exe check scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
git diff --check
```

Expected: selected tests and Ruff pass; diff check is clean.

```powershell
git add -- scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
git commit -m "feat: add offline gold bottleneck attribution"
```

---

### Task 2: Add the bounded exact OpenAlex probe, ledger transaction, and CLI

**Files:**
- Modify: `scripts/analyze_gold_bottlenecks.py`
- Modify: `tests/scripts/test_analyze_gold_bottlenecks.py`

**Interfaces:**

- `ProbeCounters`: immutable aggregate HTTP-attempt, status, and timeout counts.
- `ProbeBatch`: immutable in-memory work-status map and `ProbeCounters`.
- `ProbeGlobalError`: safe reason code plus the number of attempts already made.
- `collect_openalex_keys(environ: Mapping[str, str]) -> tuple[SecretStr, ...]`.
- `exact_work_endpoint(identifier: str) -> str`.
- `probe_openalex_exact(work_ids: Sequence[str], *, client: httpx.AsyncClient, keys: Sequence[SecretStr], sleep: SleepFn, clock: ClockFn) -> ProbeBatch`.
- `run_diagnostic` consumes the fixed paths plus injected client, environment, sleeper, and clock, and returns `DiagnosticRunResult(diagnostic_run_id, payload)`.

- [ ] **Step 1: Add one grouped RED test set for probe and orchestration**

Use `httpx.MockTransport`, an injected no-op sleeper/clock, a temporary ledger, and synthetic secrets. Cover:

- contiguous environment-key discovery; missing or gapped keys fail before reservation/network;
- DOI URL and OpenAlex ID endpoint construction, with only `api_key` and `select=id,doi` parameters;
- classification of 200 exact match, 404, identifier mismatch, timeout, 429, and 5xx;
- at most three total attempts per work; retry only timeout/429/5xx; valid `Retry-After` else 1s/2s, capped at 10s;
- low-quota 429 key rotation; no retry for 401/403/other 4xx or non-timeout client errors;
- deterministic lexical request order for normalized resolved IDs;
- hard aggregate limit of 402 attempts and no fuzzy/search endpoint;
- all offline validation before reservation and HTTP;
- one priced reservation with `query_id="aggregate-gold-availability"`, estimated usage 402, and `DEV_RUN_CAP_CNY`;
- successful settlement with actual attempts; global post-dispatch failure records actual attempts through `ledger.fail` and writes no output;
- input hashes recomputed immediately before publication; drift prevents both outputs;
- CLI stdout restricted to schema, completion, `143/139/134`, and direction.

Representative orchestration assertion:

```python
def test_run_diagnostic_uses_one_aggregate_receipt(tmp_path: Path) -> None:
    result = asyncio.run(
        run_diagnostic(
            run=_synthetic_run(tmp_path),
            gold_path=_synthetic_gold(tmp_path),
            id_map_path=_synthetic_map(tmp_path),
            ledger_path=tmp_path / "formal.sqlite3",
            pricing_path=TEST_PRICING,
            client=_mock_success_client(),
            environ={"OPENALEX_API_KEY": "synthetic-key"},
            sleep=_no_sleep,
            clock=_fixed_clock,
        )
    )
    report = SQLiteBudgetLedger(
        tmp_path / "formal.sqlite3",
        clock=_fixed_clock,
    ).report(result.diagnostic_run_id)
    assert len(report.receipts) == 1
    assert report.actual.search_api_calls == result.payload["usage"]["http_attempts"]
```

Run once and confirm RED:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "keys or endpoint or probe or retry or diagnostic or cli or ledger" -q
```

Expected: failure because online and orchestration contracts are absent.

- [ ] **Step 2: Implement probe, aggregate ledger lifecycle, and CLI**

Probe rules:

1. Use only `/works/{W-id}` or `/works/{URL-encoded DOI URL}`.
2. Limit each work to one request plus two eligible retries.
3. A 200 response is `available` only when returned `id`/`doi` agrees with the requested normalized terminal identifier; mismatch is `integrity_failure`.
4. A final 404 is `exact_not_found`; exhausted timeout/429/5xx is `unknown_transient`.
5. 401/403, other non-404 4xx, non-timeout client errors, attempt-limit breaches, and orchestration failures are global failures.

`run_diagnostic` must perform this single lifecycle:

1. Validate offline inputs and initial hashes.
2. Collect keys, load `data/annotation_work/pricing_v1.yaml`, open `SQLiteBudgetLedger`, and capture `project_checkpoint()` before reservation.
3. Price 402 OpenAlex requests via `ActualCostPricer` using dependency `openalex`, adapter `openalex-works-v1`, and `UsageActual(search_api_calls=402)`.
4. Reserve one aggregate receipt; probe in lexical normalized-ID order; settle or fail with priced actual attempts.
5. Capture the post-settlement checkpoint, then assemble and validate the aggregate payload.
6. Recompute all input hashes and fail closed on drift.
7. Atomically write each validated output independently.

CLI arguments are limited to:

```text
--run PATH
--gold PATH
--id-map PATH
--ledger PATH
--pricing-policy PATH
--out-json PATH
--out-report PATH
```

Do not add key, `.env`, query, alternate-source, retry-grid, or live-capture options.

- [ ] **Step 3: Run one GREEN verification and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -q
.\.venv\Scripts\ruff.exe check scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
git diff --check
```

Expected: all diagnostic tests and Ruff pass; diff check is clean. No real network request has occurred.

```powershell
git add -- scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
git commit -m "feat: add ledger-backed OpenAlex diagnostic"
```

---

### Task 3: Run the one authorized probe, validate evidence, and freeze the decision

**Files:**
- Create: `docs/evidence/gold-bottleneck-attribution-2026-08-09.json`
- Create: `docs/gold-bottleneck-attribution-2026-08-09.md`
- Modify: `docs/retrieval-roadmap.md`
- Modify: `docs/experiment-decisions.md`
- Modify: `HANDOFF.md`

- [ ] **Step 1: Perform a non-secret, non-network preflight**

```powershell
if (-not $env:OPENALEX_API_KEY) { throw 'OPENALEX_API_KEY is not set in this process' }
if (-not (Test-Path -LiteralPath 'runs/.ledger/formal.sqlite3')) { throw 'formal ledger is missing' }
if (-not (Test-Path -LiteralPath 'data/annotation_work/pricing_v1.yaml')) { throw 'pricing policy is missing' }
git status --short
```

Expected: required paths and primary key exist without printing secret values; status contains no unexpected source/lock changes. Preserve `data/budget_ledger.sqlite3` and `deliverables/`.

- [ ] **Step 2: Execute exactly one authorized online diagnostic**

```powershell
.\.venv\Scripts\python.exe scripts/analyze_gold_bottlenecks.py `
  --run runs/dev-20260809T061903Z-9bd861e90299 `
  --gold data/dev/gold.jsonl `
  --id-map data/identifier-map.json `
  --ledger runs/.ledger/formal.sqlite3 `
  --pricing-policy data/annotation_work/pricing_v1.yaml `
  --out-json docs/evidence/gold-bottleneck-attribution-2026-08-09.json `
  --out-report docs/gold-bottleneck-attribution-2026-08-09.md
```

Expected stdout only:

```text
schema_version=gold-bottleneck-attribution-v1
diagnostic_complete=<true|false>
counts=143/139/134
recommended_direction=<allowed-enum|null>
```

Do not rerun automatically. Built-in retries are the complete authorization scope.

- [ ] **Step 3: Validate the artifact independently and record only the measured result**

Run a read-only assertion using `assert_safe_report` that parses the JSON and verifies:

- exact schema keys and recursive privacy allowlists;
- 60/143/139/134/128/6 denominator counts;
- availability total 134, pipeline and cross-tab totals 139, and query coverage ranges;
- HTTP status plus timeout totals equal `http_attempts <= 402`;
- incomplete/tied evidence has `recommended_direction=null`;
- ledger before/after checkpoints and aggregate actual usage are present.

Expected output: `gold_bottleneck_artifact: valid`.

If validation fails, stop without updating project-state documents. If the report is valid but incomplete, retain it and document the reason; do not rerun or choose a direction.

Update the Markdown report, roadmap, experiment decisions, and handoff with only:

1. denominator definitions and aggregate counts;
2. completeness/reason codes;
3. the uniquely selected direction, tie, or incomplete result;
4. limitations: frozen dev, exact-ID only, one provider, no live capture.

Link to the evidence instead of copying the full matrix. Do not implement the selected direction in this task.

- [ ] **Step 4: Run final verification and audit scope**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -q
& 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src tests scripts
& 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src scripts/analyze_gold_bottlenecks.py
git diff --check
git status --short
```

Expected:

- focused and full pytest pass;
- Ruff passes;
- mypy has no errors in this task's files (the known baseline may remain only in unchanged `query/parser.py`, `application/readiness.py`, `retrieval/snapshot_adapters.py`, and `llm/snapshot_adapters.py`);
- no production source, candidate lock, manifest, or historical run changed;
- no readiness/capture/replay/compare/validation command ran;
- only the two pre-existing untracked paths remain outside planned changes.

- [ ] **Step 5: Commit evidence and project-state updates**

```powershell
git add -- docs/evidence/gold-bottleneck-attribution-2026-08-09.json docs/gold-bottleneck-attribution-2026-08-09.md docs/retrieval-roadmap.md docs/experiment-decisions.md HANDOFF.md
git commit -m "docs: record gold bottleneck attribution"
```

Final completion requires three implementation-boundary commits, valid aggregate evidence, and either one evidence-backed next direction or an explicit incomplete/tie stop condition.
