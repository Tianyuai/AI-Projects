# Gold Availability and Bottleneck Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one aggregate-only diagnostic that exactly probes OpenAlex availability for 134 normalized gold works and attributes all 139 normalized query–work associations to retrieval, filtering, or Top-50 loss.

**Architecture:** Add one standalone diagnostic script with three internal boundaries: pure sealed-run attribution, an injected exact-work HTTP probe, and ledger-backed orchestration/output. The script reuses existing identifier normalization, pricing, and SQLite ledger contracts but does not change production providers, retrieval, filtering, fusion, ranking, locks, or formal-run workflows.

**Tech Stack:** Python 3.12, Pydantic v2 domain models, httpx async client/MockTransport, existing `IdentifierMap`, `ActualCostPricer`, `SQLiteBudgetLedger`, pytest, JSON/JSONL sealed artifacts.

## Global Constraints

- Execute in `D:\AI Projects\.worktrees\week3` on branch `codex/project-document-handoff`.
- Treat `docs/superpowers/specs/2026-08-09-gold-availability-bottleneck-attribution-design.md` as the authority.
- Use source run `runs/dev-20260809T061903Z-9bd861e90299`; require complete/passed run, formal validity, quality pass, and passed zero-valued `provenance-failures`.
- Recompute and require exactly 60 queries, 143 raw gold identifiers, 139 normalized query–work associations, and 134 normalized unique works.
- Resolve all gold through existing `normalize_paper_id` and `IdentifierMap.resolve`; do not create another normalization implementation.
- Allow only terminal DOI and OpenAlex IDs. Never search by arXiv ID, title, author, abstract, full text, or query text.
- Use OpenAlex single-work GET only; select only `id,doi`.
- Do not run readiness, live capture, replay, compare, validation, or candidate-lock rebuild.
- Do not modify production provider, filtering, fusion, selector, ranking, `data/manifest.json`, historical runs, or `runs/candidate.lock.yaml`.
- Use only API keys already present in the process environment; do not read, print, parse, or commit `.env`.
- Preserve existing untracked `data/budget_ledger.sqlite3` and `deliverables/`.
- Use `runs/.ledger/formal.sqlite3` through `SQLiteBudgetLedger`; never issue direct SQL or bypass pricing.
- Reserve one aggregate ledger entry for at most 402 HTTP attempts, then settle/fail it with actual priced attempts.
- Never persist raw responses, per-work statuses, gold IDs, query IDs, paper IDs, request IDs, titles, URLs, or free text.
- JSON schema is exactly `gold-bottleneck-attribution-v1`; reject extra keys and disallowed string values before writing.
- Any global input, Gate, provenance, authentication, budget, ledger, or artifact-integrity failure exits nonzero without a report.
- A partial online report is permitted only for `unknown_transient`, `invalid_identifier`, or `integrity_failure`; it must set `diagnostic_complete=false` and `recommended_direction=null`.

---

### Task 1: Build pure gold indexing and sealed pipeline attribution

**Files:**
- Create: `scripts/analyze_gold_bottlenecks.py`
- Create: `tests/scripts/test_analyze_gold_bottlenecks.py`

**Interfaces:**
- Consumes: `Sequence[EvaluationQuery]`, `IdentifierMap`, and one sealed run directory.
- Produces:
  - `GoldIndex(raw_identifier_count, query_to_work_ids, unique_work_ids, doi_work_count, openalex_work_count, invalid_identifier_count)`;
  - `OfflineContext(source_run_id, source_git_sha, input_hashes, gold_index, stage_by_pair)`;
  - `build_gold_index(...) -> GoldIndex`;
  - `load_offline_context(run, gold_path, id_map_path) -> OfflineContext`.

- [ ] **Step 1: Create synthetic fixtures and write failing indexing tests**

Add local helpers and tests that contain no frozen gold values:

```python
def _gold(query_id: str, *paper_ids: str) -> EvaluationQuery:
    return EvaluationQuery(
        query_id=query_id,
        query="synthetic query",
        relevant_paper_ids=list(paper_ids),
    )


def test_build_gold_index_uses_resolved_query_and_global_deduplication(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "identifier-map.json"
    mapping.write_text(
        json.dumps(
            {
                "arxiv:1000.00001": "doi:10.1000/a",
                "arxiv:1000.00002": "openalex:W2",
            }
        ),
        encoding="utf-8",
    )
    id_map = IdentifierMap.from_path(mapping)

    index = build_gold_index(
        [
            _gold("q1", "arxiv:1000.00001", "doi:10.1000/a"),
            _gold("q2", "arxiv:1000.00001", "arxiv:1000.00002"),
        ],
        id_map,
    )

    assert index.raw_identifier_count == 4
    assert index.normalized_query_work_count == 3
    assert index.unique_work_ids == ("doi:10.1000/a", "openalex:W2")
    assert index.doi_work_count == 1
    assert index.openalex_work_count == 1
    assert index.invalid_identifier_count == 0
```

Also add `test_build_gold_index_counts_invalid_terminal_namespace` using a synthetic `s2:` terminal ID and assert it appears only in `invalid_identifier_count`.

- [ ] **Step 2: Run indexing tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "gold_index" -q
```

Expected: collection fails because `scripts.analyze_gold_bottlenecks` does not exist.

- [ ] **Step 3: Implement immutable indexing types and minimal indexing logic**

Add these exact types and signatures:

```python
AvailabilityStatus = Literal[
    "available",
    "exact_not_found",
    "unknown_transient",
    "invalid_identifier",
    "integrity_failure",
]
PipelineStage = Literal[
    "selected_top50",
    "ranked_outside_top50",
    "filtered_out",
    "not_retrieved",
]


@dataclass(frozen=True)
class GoldIndex:
    raw_identifier_count: int
    normalized_query_work_count: int
    query_to_work_ids: dict[str, tuple[str, ...]]
    unique_work_ids: tuple[str, ...]
    doi_work_count: int
    openalex_work_count: int
    invalid_identifier_count: int


def build_gold_index(
    gold: Sequence[EvaluationQuery],
    id_map: IdentifierMap,
) -> GoldIndex:
    query_to_work_ids: dict[str, tuple[str, ...]] = {}
    raw_count = 0
    for record in gold:
        raw_count += len(record.relevant_paper_ids)
        resolved = tuple(
            sorted({id_map.resolve(value) for value in record.relevant_paper_ids})
        )
        query_to_work_ids[record.query_id] = resolved
    unique = tuple(sorted({item for values in query_to_work_ids.values() for item in values}))
    doi_count = sum(item.startswith("doi:") for item in unique)
    openalex_count = sum(item.startswith("openalex:") for item in unique)
    return GoldIndex(
        raw_identifier_count=raw_count,
        normalized_query_work_count=sum(map(len, query_to_work_ids.values())),
        query_to_work_ids=query_to_work_ids,
        unique_work_ids=unique,
        doi_work_count=doi_count,
        openalex_work_count=openalex_count,
        invalid_identifier_count=len(unique) - doi_count - openalex_count,
    )
```

Do not serialize `query_to_work_ids` or `unique_work_ids`; they are in-memory inputs for later aggregation only.

- [ ] **Step 4: Run indexing tests and verify GREEN**

Run the command from Step 2.

Expected: both gold-index tests pass.

- [ ] **Step 5: Write failing sealed-run validation and stage-classification tests**

Create a synthetic run with two queries and assert all four stages:

```python
def test_load_offline_context_classifies_mutually_exclusive_stages(
    tmp_path: Path,
) -> None:
    run, gold_path, id_map_path = _write_valid_synthetic_run(tmp_path)

    context = load_offline_context(run, gold_path, id_map_path)

    assert Counter(context.stage_by_pair.values()) == {
        "selected_top50": 1,
        "ranked_outside_top50": 1,
        "filtered_out": 1,
        "not_retrieved": 1,
    }
```

Add parameterized fail-closed tests for:

```python
(
    "run_not_complete",
    "source run is not complete",
),
(
    "gate_not_passed",
    "source run gate is invalid",
),
(
    "provenance_nonzero",
    "source run provenance is invalid",
),
(
    "query_set_mismatch",
    "source run query sets do not match",
),
(
    "selected_not_subset",
    "source run stage invariant failed",
),
```

The synthetic `gates.json` must include a check with `rule_id="provenance-failures"`, `passed=true`, and measure numerator/value `0`.

- [ ] **Step 6: Run stage tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "offline_context" -q
```

Expected: FAIL because `load_offline_context` and `OfflineContext` are not defined.

- [ ] **Step 7: Implement source validation, hashes, and stage classification**

Add:

```python
@dataclass(frozen=True)
class OfflineContext:
    source_run_id: str
    source_git_sha: str
    input_hashes: dict[str, str]
    gold_index: GoldIndex
    stage_by_pair: dict[tuple[str, str], PipelineStage]


def _classify_stage(
    work_id: str,
    *,
    retrieved: AbstractSet[str],
    post_filter: AbstractSet[str],
    selected: AbstractSet[str],
) -> PipelineStage:
    if work_id in selected:
        return "selected_top50"
    if work_id in post_filter:
        return "ranked_outside_top50"
    if work_id in retrieved:
        return "filtered_out"
    return "not_retrieved"
```

`load_offline_context` must:

1. Read `run.json`, `gates.json`, `executions.jsonl`, `business-results.jsonl`, gold, and identifier map once.
2. Require run status/`gate_result` equal `complete`/`passed`; require `formal_valid` and `quality_passed` true.
3. Find exactly one `provenance-failures` check and require `passed is True`, numerator `0`, and value `0`.
4. Require identical 60-query sets across gold, executions, and business results.
5. Resolve each stage ID through the same `IdentifierMap`.
6. Require `selected <= post_filter <= retrieved` for every query.
7. Recompute and require `(60, 143, 139, 134, 128, 6, 0)` for query/raw/pair/unique/DOI/OpenAlex/invalid counts.
8. Compute fixed SHA-256 values with existing `sha256_file` for the six schema inputs.
9. Return only the in-memory context; never print identifiers.

- [ ] **Step 8: Verify Task 1 and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "gold_index or offline_context" -q
.\.venv\Scripts\ruff.exe check scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
```

Expected: all selected tests and Ruff pass.

Commit:

```powershell
git add -- scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
git commit -m "feat: add offline gold bottleneck attribution"
```

---

### Task 2: Add the exact OpenAlex probe with bounded retries

**Files:**
- Modify: `scripts/analyze_gold_bottlenecks.py`
- Modify: `tests/scripts/test_analyze_gold_bottlenecks.py`

**Interfaces:**
- Consumes: sorted resolved IDs, process environment, an injected `httpx.AsyncClient`, clock, and async sleep.
- Produces:
  - `ProbeCounters` with aggregate attempt/status counts;
  - `ProbeBatch(status_by_work, counters)` retained only in memory;
  - `ProbeGlobalError(attempts)` for post-dispatch global failures that still require ledger settlement;
  - `collect_openalex_keys(environ) -> tuple[SecretStr, ...]`;
  - `probe_openalex_exact(work_ids, *, client, keys, sleep, clock) -> ProbeBatch`.

- [ ] **Step 1: Write failing request-planning and key-policy tests**

Add:

```python
def test_collect_openalex_keys_requires_contiguous_sequence() -> None:
    assert [item.get_secret_value() for item in collect_openalex_keys({
        "OPENALEX_API_KEY": "k1",
        "OPENALEX_API_KEY_2": "k2",
    })] == ["k1", "k2"]
    with pytest.raises(ValueError, match="OpenAlex key sequence is not contiguous"):
        collect_openalex_keys({
            "OPENALEX_API_KEY": "k1",
            "OPENALEX_API_KEY_3": "k3",
        })


@pytest.mark.parametrize(
    ("work_id", "expected_suffix"),
    [
        ("openalex:W2", "/works/W2"),
        ("doi:10.1000/a", "/works/https%3A%2F%2Fdoi.org%2F10.1000%2Fa"),
    ],
)
def test_exact_work_endpoint_is_identifier_only(
    work_id: str,
    expected_suffix: str,
) -> None:
    assert exact_work_endpoint(work_id) == expected_suffix
```

Also assert invalid namespaces raise `ValueError("unsupported exact identifier")` and no endpoint includes `search`, `filter`, title, or query parameters.

- [ ] **Step 2: Run request-planning tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "openalex_keys or exact_work_endpoint" -q
```

Expected: FAIL because the functions are undefined.

- [ ] **Step 3: Implement key loading and exact endpoint planning**

Use only `Mapping[str, str]`, `SecretStr`, and `urllib.parse.quote`:

```python
def exact_work_endpoint(work_id: str) -> str:
    normalized = normalize_paper_id(work_id)
    if normalized.startswith("openalex:"):
        external = normalized.removeprefix("openalex:")
    elif normalized.startswith("doi:"):
        external = f"https://doi.org/{normalized.removeprefix('doi:')}"
    else:
        raise ValueError("unsupported exact identifier")
    return f"/works/{quote(external, safe='')}"
```

`collect_openalex_keys` must start with `OPENALEX_API_KEY`, accept `_2`, `_3`, and so on, deduplicate equal values, reject a numbered key after a gap, and never include secret values in errors or repr output.

- [ ] **Step 4: Write failing exact-response and retry tests with MockTransport**

Cover these cases with an injected client and a sleep recorder:

```python
async def test_probe_exact_classifies_matching_200() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "id": "https://openalex.org/W2",
                    "doi": "https://doi.org/10.1000/a",
                },
                request=request,
            )
        )
    )
    batch = await probe_openalex_exact(
        ("doi:10.1000/a",),
        client=client,
        keys=(SecretStr("synthetic-key"),),
        sleep=_no_sleep,
        clock=_fixed_clock,
    )
    assert list(batch.status_by_work.values()) == ["available"]
    assert batch.counters.http_200 == 1
```

Add tests for:

- matching OpenAlex ID 200;
- 200 invalid JSON or mismatched ID -> `integrity_failure`;
- 404 -> `exact_not_found` without retry;
- timeout, 429, and 5xx -> at most three attempts then `unknown_transient`;
- numeric and HTTP-date `Retry-After`, clamped to 10 seconds;
- fallback delays exactly 1 and 2 seconds;
- 429 with `x-ratelimit-remaining < 10` rotates to the next contiguous key without printing it;
- 401/403 and other 4xx raise a global error immediately;
- non-timeout `httpx.RequestError` raises globally;
- unsupported terminal namespaces become `invalid_identifier` without an HTTP attempt;
- input order is sorted and duplicate IDs issue one request sequence;
- total attempts never exceed `len(unique_ids) * 3`.

- [ ] **Step 5: Run probe tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "probe_exact or retry_after" -q
```

Expected: FAIL because probe types and behavior are not implemented.

- [ ] **Step 6: Implement the minimal in-memory probe**

Add exact immutable counters and batch types:

```python
@dataclass(frozen=True)
class ProbeCounters:
    unique_requests_planned: int
    http_attempts: int = 0
    retries: int = 0
    http_200: int = 0
    http_404: int = 0
    http_429: int = 0
    http_5xx: int = 0
    timeouts: int = 0


@dataclass(frozen=True)
class ProbeBatch:
    status_by_work: dict[str, AvailabilityStatus]
    counters: ProbeCounters


class ProbeGlobalError(RuntimeError):
    def __init__(self, code: str, attempts: int) -> None:
        super().__init__(code)
        self.code = code
        self.attempts = attempts
```

Implement a maximum of three total attempts per ID. Send only:

```python
await client.get(
    f"https://api.openalex.org{exact_work_endpoint(work_id)}",
    params={
        "api_key": keys[key_index].get_secret_value(),
        "select": "id,doi",
    },
    follow_redirects=False,
)
```

Never store `response.content`, headers, endpoint, work ID, or exceptions in output objects. Parse the current response in memory, increment counters, retain only the status mapping until aggregate assembly, then let it go out of scope.

Accumulate counts in local integers and instantiate frozen `ProbeCounters` only when returning. A global error raised after dispatch must be `ProbeGlobalError` carrying only the aggregate attempt count and a fixed safe code; it must not carry a URL, identifier, response, request ID, key, or exception text.

- [ ] **Step 7: Verify Task 2 and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "openalex_keys or exact_work_endpoint or probe_exact or retry_after" -q
.\.venv\Scripts\ruff.exe check scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
```

Expected: selected tests and Ruff pass.

Commit:

```powershell
git add -- scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
git commit -m "feat: add bounded OpenAlex exact probe"
```

---

### Task 3: Add ledger-backed orchestration and the fixed aggregate output contract

**Files:**
- Modify: `scripts/analyze_gold_bottlenecks.py`
- Modify: `tests/scripts/test_analyze_gold_bottlenecks.py`

**Interfaces:**
- Consumes: `OfflineContext`, `ProbeBatch`, `SQLiteBudgetLedger`, `ActualCostPricer`, and output paths.
- Produces:
  - `assemble_report(context, probe, *, usage) -> dict[str, object]`;
  - `assert_safe_report(payload) -> None`;
  - `DiagnosticUsage` and `DiagnosticRunResult(diagnostic_run_id, payload)`;
  - `run_diagnostic(...) -> DiagnosticRunResult`;
  - CLI writing one JSON and one Markdown report atomically.

- [ ] **Step 1: Write failing cross-tab, recommendation, and schema tests**

Use synthetic in-memory contexts and statuses:

```python
def test_assemble_report_conserves_both_denominators() -> None:
    context = _synthetic_offline_context()
    probe = _synthetic_probe_batch()

    payload = assemble_report(
        context,
        probe,
        usage=_synthetic_usage(),
    )

    assert sum(payload["availability"].values()) == payload["counts"]["unique_work_count"]
    assert sum(payload["pipeline_stages"].values()) == payload["counts"]["normalized_query_work_count"]
    assert sum(
        count
        for row in payload["cross_tab"].values()
        for count in row.values()
    ) == payload["counts"]["normalized_query_work_count"]
```

Add tests asserting:

- each cross-tab availability row has exactly four pipeline keys;
- `query_coverage` counts distinct queries, not associations;
- unique dominant bucket selects the exact direction/reason-code pair;
- tie -> `recommended_direction=None` and `largest_bucket_tie`;
- no recoverable loss -> null and `no_recoverable_loss`;
- any unknown/invalid/integrity count -> `diagnostic_complete=False`, null direction, and specific reason code;
- reason codes are deduplicated and lexically sorted.

- [ ] **Step 2: Run aggregate tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "assemble_report or recommended_direction" -q
```

Expected: FAIL because aggregate functions are undefined.

- [ ] **Step 3: Implement aggregate assembly and exact decision rules**

Define fixed constants:

```python
AVAILABILITY_STATUSES: tuple[AvailabilityStatus, ...] = (
    "available",
    "exact_not_found",
    "unknown_transient",
    "invalid_identifier",
    "integrity_failure",
)
PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    "selected_top50",
    "ranked_outside_top50",
    "filtered_out",
    "not_retrieved",
)
```

Construct every nested object from these constants, never from provider JSON. Recommendation candidates are association-level counts only:

```python
losses = {
    "new_data_source_probe": sum(
        cross_tab["exact_not_found"][stage]
        for stage in PIPELINE_STAGES
        if stage != "selected_top50"
    ),
    "retrieval_query_evolution_probe": cross_tab["available"]["not_retrieved"],
    "hard_filter_diagnosis": cross_tab["available"]["filtered_out"],
    "selector_rerank_offline": cross_tab["available"]["ranked_outside_top50"],
}
```

Choose a direction only when diagnostic completeness is true, the maximum is positive, and exactly one direction has that maximum.

- [ ] **Step 4: Write failing privacy and exact-schema tests**

Add a recursively valid payload fixture and mutate it to assert rejection of:

- any extra top-level or nested key;
- noninteger/negative count;
- invalid run ID, Git SHA, or SHA-256;
- any string containing DOI, arXiv/OpenAlex ID, URL, title, query text, request ID, or arbitrary free text;
- any direction/reason code outside the design enum;
- inconsistent availability, pipeline, cross-tab, identifier, or HTTP totals.

Also test atomic write preservation by forcing serialization failure and asserting the existing destination bytes remain unchanged.

- [ ] **Step 5: Run privacy/schema tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "safe_report or atomic" -q
```

Expected: FAIL because validator and writer are undefined.

- [ ] **Step 6: Implement a strict allow-list validator and atomic renderers**

`assert_safe_report` must validate the complete schema from the design with exact key sets and exact enum sets. Permit strings only when they match one of:

```python
RUN_ID_RE = re.compile(r"^dev-\d{8}T\d{6}Z-[0-9a-f]{12}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
```

or are the fixed schema/direction/reason/status names. Reject all other strings. Use `json.dump(..., sort_keys=True, indent=2, allow_nan=False)` to a same-directory temporary file, then `Path.replace` only after successful validation and serialization.

Define the remaining allow lists explicitly:

```python
TOP_LEVEL_KEYS = {
    "schema_version", "source_run_id", "source_git_sha", "input_hashes",
    "counts", "availability", "pipeline_stages", "cross_tab",
    "query_coverage", "usage", "diagnostic_complete",
    "recommended_direction", "reason_codes",
}
INPUT_HASH_KEYS = {
    "gold_sha256", "identifier_map_sha256", "executions_sha256",
    "business_results_sha256", "gates_sha256", "run_sha256",
}
COUNT_KEYS = {
    "query_count", "raw_gold_identifier_count",
    "normalized_query_work_count", "unique_work_count",
    "doi_work_count", "openalex_work_count",
}
USAGE_KEYS = {
    "unique_requests_planned", "http_attempts", "retries", "http_200",
    "http_404", "http_429", "http_5xx", "timeouts",
    "ledger_checkpoint_before_sha256", "ledger_checkpoint_after_sha256",
}
DIRECTIONS = {
    "new_data_source_probe", "retrieval_query_evolution_probe",
    "hard_filter_diagnosis", "selector_rerank_offline",
}
REASON_CODES = {
    "exact_not_found_dominant", "available_not_retrieved_dominant",
    "available_filtered_out_dominant", "available_ranked_out_dominant",
    "unknown_transient_present", "invalid_identifier_present",
    "integrity_failure_present", "largest_bucket_tie",
    "no_recoverable_loss",
}
```

The availability and pipeline objects use `AVAILABILITY_STATUSES` and `PIPELINE_STAGES` as their exact key sets; both cross matrices use those same two dimensions.

`render_markdown(payload)` must read only validated aggregate keys and emit these sections: conclusion, three denominators, availability table, pipeline table, cross table, recommendation/reason, usage, and evidence limits. It must not accept arbitrary prose inputs.

Add the exact orchestration result types before implementing ledger behavior:

```python
@dataclass(frozen=True)
class DiagnosticUsage:
    unique_requests_planned: int
    http_attempts: int
    retries: int
    http_200: int
    http_404: int
    http_429: int
    http_5xx: int
    timeouts: int
    ledger_checkpoint_before_sha256: str
    ledger_checkpoint_after_sha256: str


@dataclass(frozen=True)
class DiagnosticRunResult:
    diagnostic_run_id: str
    payload: dict[str, object]
```

- [ ] **Step 7: Write failing ledger transaction and CLI tests**

Use a temporary `SQLiteBudgetLedger`, test pricing policy, MockTransport, and injected environment:

```python
def test_run_diagnostic_reserves_one_aggregate_receipt_and_settles_actual(
    tmp_path: Path,
) -> None:
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
    ledger = SQLiteBudgetLedger(tmp_path / "formal.sqlite3", clock=_fixed_clock)
    report = ledger.report(result.diagnostic_run_id)
    assert len(report.receipts) == 1
    assert report.actual.search_api_calls == result.payload["usage"]["http_attempts"]
```

Add tests that:

- offline validation happens before ledger reservation or HTTP dispatch;
- estimate is exactly 402 OpenAlex requests priced by `ActualCostPricer`;
- success settles one aggregate receipt;
- a global authentication/client/network error after dispatch calls `ledger.fail` with actual attempts and writes no report;
- a 200 identifier mismatch remains the per-work `integrity_failure` status, writes only an incomplete aggregate report, and never selects a direction;
- missing key, pricing, budget, or ledger error dispatches no HTTP and writes no report;
- input hashes are recomputed immediately before output and any change prevents both files;
- CLI stdout contains only schema, completion, 143/139/134 counts, and direction.

- [ ] **Step 8: Run ledger/CLI tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -k "run_diagnostic or cli" -q
```

Expected: FAIL because orchestration and CLI are incomplete.

- [ ] **Step 9: Implement aggregate ledger transaction and CLI**

Implementation order inside `run_diagnostic`:

1. Load/validate offline context and initial hashes.
2. Collect contiguous process keys; load `data/annotation_work/pricing_v1.yaml` via `load_pricing_policy`.
3. Open `SQLiteBudgetLedger` and record `project_checkpoint()` before reservation.
4. Price 402 attempts with `ActualCostPricer(...).value_actual(dependency="openalex", model_or_adapter="openalex-works-v1", usage=UsageActual(search_api_calls=402))`.
5. Reserve one ledger receipt using an internal diagnostic run ID and `query_id="aggregate-gold-availability"`, with `run_cap_cny=DEV_RUN_CAP_CNY`.
6. Run the in-memory probe; price and settle actual attempts. On `ProbeGlobalError`, price its aggregate attempt count, call `ledger.fail`, then re-raise a constant safe error.
7. Record the after checkpoint and strip only the literal `sha256:` prefix for JSON.
8. Assemble and validate payload.
9. Recompute all six input hashes; if any differ, abort without output.
10. Validate both renderings in memory, then atomically write JSON and Markdown individually. A publication failure exits nonzero; each destination must independently preserve its prior bytes when its own write fails. Do not add a cross-file transaction or backup protocol.

CLI arguments:

```text
--run PATH
--gold PATH
--id-map PATH
--ledger PATH
--pricing-policy PATH
--out-json PATH
--out-report PATH
```

Do not add `.env`, key, query, retry-grid, alternative-source, or live-capture arguments.

- [ ] **Step 10: Verify Task 3 and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -q
.\.venv\Scripts\ruff.exe check scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
```

Expected: all diagnostic tests and Ruff pass.

Commit:

```powershell
git add -- scripts/analyze_gold_bottlenecks.py tests/scripts/test_analyze_gold_bottlenecks.py
git commit -m "feat: add ledger-backed gold availability diagnostic"
```

---

### Task 4: Execute one authorized probe and freeze the attribution decision

**Files:**
- Create: `docs/evidence/gold-bottleneck-attribution-2026-08-09.json`
- Create: `docs/gold-bottleneck-attribution-2026-08-09.md`
- Modify: `docs/retrieval-roadmap.md`
- Modify: `docs/experiment-decisions.md`
- Modify: `HANDOFF.md`

**Interfaces:**
- Consumes: tested CLI, process-provided OpenAlex keys, fixed source run, pricing policy, and formal ledger.
- Produces: one aggregate JSON artifact, one concise report, and one evidence-based next-direction decision or explicit incomplete/tie result.

- [ ] **Step 1: Perform non-secret preflight without network**

Run:

```powershell
if (-not $env:OPENALEX_API_KEY) { throw 'OPENALEX_API_KEY is not set in this process' }
if (-not (Test-Path -LiteralPath 'runs/.ledger/formal.sqlite3')) { throw 'formal ledger is missing' }
if (-not (Test-Path -LiteralPath 'data/annotation_work/pricing_v1.yaml')) { throw 'pricing policy is missing' }
git status --short
```

Expected: key presence succeeds without printing its value; the only pre-existing untracked paths remain `data/budget_ledger.sqlite3` and `deliverables/`; no source or lock changes exist.

- [ ] **Step 2: Run exactly one authorized online diagnostic**

Run:

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

Expected stdout contains only:

```text
schema_version=gold-bottleneck-attribution-v1
diagnostic_complete=<true|false>
counts=143/139/134
recommended_direction=<enum|null>
```

Do not automatically rerun if incomplete. Built-in retries are the full authorization scope for this execution.

- [ ] **Step 3: Independently validate the generated artifact**

Run a read-only assertion that imports `assert_safe_report`, parses the JSON, checks exact schema keys and totals, confirms 60/143/139/134/128/6, confirms `http_attempts <= 402`, and recursively checks that no forbidden key or string value is present.

Expected:

```text
gold_bottleneck_artifact: valid
```

If `diagnostic_complete=false`, retain the aggregate evidence but stop before selecting or implementing a direction.

- [ ] **Step 4: Record only the measured decision**

The Markdown report must contain only:

1. three denominator definitions;
2. aggregate OpenAlex availability counts;
3. aggregate pipeline and cross-tab counts;
4. diagnostic completeness and exact reason codes;
5. one recommended direction, tie, or incomplete result;
6. limitations: frozen dev, exact-ID only, one provider, no live capture.

Update roadmap, experiment decisions, and handoff with a link and one-line decision. Do not duplicate the full matrix. Do not implement the recommended direction in this task.

- [ ] **Step 5: Run focused and full verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_gold_bottlenecks.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -q
& 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src tests scripts
& 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src scripts/analyze_gold_bottlenecks.py
git diff --check
```

Expected: focused and full pytest pass; Ruff passes. Mypy may report only baseline errors in files unchanged by this task; any error in `scripts/analyze_gold_bottlenecks.py` or another task-modified Python file blocks completion.

- [ ] **Step 6: Commit evidence and project-state updates**

```powershell
git add -- docs/evidence/gold-bottleneck-attribution-2026-08-09.json docs/gold-bottleneck-attribution-2026-08-09.md docs/retrieval-roadmap.md docs/experiment-decisions.md HANDOFF.md
git commit -m "docs: record gold bottleneck attribution"
```

- [ ] **Step 7: Final audit**

Confirm:

- branch contains one commit for each task boundary;
- `git status --short` shows only pre-existing `data/budget_ledger.sqlite3` and `deliverables/`;
- `runs/candidate.lock.yaml` and production source files are unchanged;
- no live capture/replay/compare/validation command ran;
- the formal ledger advanced by exactly one aggregate diagnostic receipt;
- the final handoff states the measured next direction or why no direction can yet be selected.
