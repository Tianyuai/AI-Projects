# Query Evolution Bounded Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a gold-isolated, budget-bounded Query Evolution probe that exactly reconstructs the frozen dev baseline, captures at most one LLM proposal and two OpenAlex searches for each of 55 locked queries, replays the new evidence offline, and decides Gates A/B/C without changing production behavior.

**Architecture:** Add one strict generation module and one pure evaluation module. Keep credentials, immutable lock consumption, persistent reservations, canonical-request reuse, capture/replay orchestration, deferred gold loading, and atomic writes in a standalone diagnostic CLI. Reuse existing live/replay adapters, hard filters, deduplication, one-source RRF, metrics, pricing, ledger, and snapshot v2.

**Tech Stack:** Python 3.11, Pydantic 2, `httpx`, PyYAML, SQLite, pytest, Ruff, mypy strict.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-09-query-evolution-bounded-probe-design.md` exactly.
- Use the existing worktree and branch; preserve untracked `data/budget_ledger.sqlite3` and `deliverables/`.
- Automated work is offline: do not read `D:\AI Projects\Projects\.env`, use real network, rebuild the candidate lock, run readiness, or start formal capture/replay/validation.
- Stop implementation after the offline, zero-network `preflight`; a real bounded `run` requires separate live authorization.
- Do not modify production composition, `EvolutionCoordinator`, API, UI, `configs/ablations.yaml`, historical runs, or historical evidence.
- Fixed source: `runs/dev-20260809T061903Z-9bd861e90299`; exact baseline: 60 queries, 2,910 Top-50 outputs, 14 candidate-pool gold associations, 8 Top-50 gold associations.
- Frozen availability evidence SHA-256: `3f445486d5cf590f3f11a51930153a45916023880e856def379e0f01d053ad04`; it must retain the expected schema and 134/134 available unique works.
- Limits: 55/110 logical LLM/OpenAlex operations, 165/330 retry-inclusive attempts, 3,600-second batch timeout, 3,900-second ledger TTL.
- Capture code cannot accept or load availability/gold/identifier-map objects. `preflight` may use them to create a self-hashed private lock; `run --lock` performs only gold-free checks before network, and the deferred evaluator opens and verifies gold/id-map only after snapshot seal and offline replay.
- Use TDD and commit after each task.

---

### Task 1: Implement the strict, gold-free generation contract

**Files:**
- Create: `configs/prompts/query_evolve.yaml`
- Create: `src/paper_search/evolution/query_evolution.py`
- Create: `tests/unit/test_query_evolution.py`
- Modify: `src/paper_search/retrieval/openalex.py`
- Modify: `src/paper_search/retrieval/snapshot_adapters.py`
- Modify: `tests/unit/test_openalex.py`

**Interfaces:**
- `build_query_evolution_context(spec, plan, candidate_count, top_titles) -> QueryEvolutionContext`
- `validate_query_evolution_proposal(raw, context) -> QueryEvolutionProposal`
- `QueryEvolutionGenerator.generate(context, reservation) -> QueryEvolutionResult`

- [ ] **Step 1: Write RED tests and the fixed prompt artifact**

Prompt contract:

```yaml
name: query_evolve
version: query-evolve-v1
temperature: 0
response_model: QueryEvolutionProposal
strategies: [synonym, entity_alias, facet_combination, task_decomposition]
no_op_reasons: [insufficient_grounded_facets, no_novel_query]
instructions:
  - Use only facts and facets present in the payload.
  - Do not infer gold papers, relevance labels, new venues, new years, or unrelated entities.
  - Return zero to two complementary OpenAlex queries as strict JSON.
  - Always include no_op_reason; use null when subqueries contains items.
  - Copy every source_facets value exactly from the payload.
```

Tests must cover: strict 0–2 proposal schema; `no_op_reason` always present and `null` for non-empty proposals; legal no-op exclusivity; extra fields; deterministic facet order; empty-constraint `QuerySpec`; deterministic candidate count and Top-10 title construction; NFKC followed by the public production OpenAlex canonicalizer; `foo?`/`foo` duplicate and `*` empty cases; control-character, over-300-character, duplicate, and conflicting-year rejection; exact source-facet membership; hard filters excluded from the payload; analyzer error and invalid-schema classification; snapshot refs; and recursive absence of query ID/gold/label fields from the payload.

Run and expect import/collection failure:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_query_evolution.py -q
```

- [ ] **Step 2: Implement the minimum strict contract**

Use these models and constants:

```python
EvolutionStrategy = Literal[
    "synonym", "entity_alias", "facet_combination", "task_decomposition"
]
NoOpReason = Literal["insufficient_grounded_facets", "no_novel_query"]
MAX_QUERY_CHARS = 300


class EvolutionSeedSubquery(DomainModel):
    text: NonEmptyStr
    target_constraints: list[NonEmptyStr]


class QueryEvolutionContext(DomainModel):
    original_query: NonEmptyStr
    query_spec: QuerySpec
    seed_subqueries: list[EvolutionSeedSubquery] = Field(min_length=3, max_length=5)
    candidate_count: int = Field(strict=True, ge=0)
    top_titles: list[NonEmptyStr] = Field(max_length=10)
    facets: list[NonEmptyStr] = Field(min_length=1)
    instructions: list[NonEmptyStr]
    response_schema: Literal["query-evolution-proposal-v1"]


class EvolutionSubquery(DomainModel):
    text: NonEmptyStr
    source_facets: list[NonEmptyStr] = Field(min_length=1)
    strategy: EvolutionStrategy


class QueryEvolutionProposal(DomainModel):
    subqueries: list[EvolutionSubquery] = Field(max_length=2)
    no_op_reason: NoOpReason | None

    @model_validator(mode="after")
    def validate_no_op(self) -> QueryEvolutionProposal:
        if bool(self.subqueries) == (self.no_op_reason is not None):
            raise ValueError("no_op_reason must exist exactly for an empty proposal")
        return self
```

Make the existing OpenAlex query normalizer a public, behavior-preserving pure function and update its internal imports; do not change production request bytes. Query Evolution applies Unicode NFKC and then that function for validation and deduplication.

Build facets in this exact order: normalized `original_query`, `research_goal`, then `topics`, `methods`, `tasks`, `datasets`, `domains`, `venues`, `must_have`, `should_have`, then seed `target_constraints` in subquery order. NFKC/whitespace-normalize, casefold-deduplicate, and preserve first occurrence. `candidate_count` is the canonical-ID-unique first-round OpenAlex stream before post-filtering and Top-50 truncation. Build `top_titles` from the frozen Top-50 in order, using the same text normalization and first-occurrence deduplication. Keep `SearchPlan.inherited_hard_filters` outside the serialized context and pass it separately from the runner.

Deterministically reject only mechanically provable violations. Keep unrelated entity/venue avoidance as prompt policy; do not add a lexicon, second model, repair call, or rules fallback. Use the existing analyzer method `generate_json(prompt_name="query_evolve", payload=context.model_dump(mode="json"), reservation=reservation)`; invalid provider output is `integrity_failure`, never `no_op`.

- [ ] **Step 3: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_openalex.py tests/unit/test_query_evolution.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/evolution/query_evolution.py src/paper_search/retrieval/openalex.py src/paper_search/retrieval/snapshot_adapters.py tests/unit/test_openalex.py tests/unit/test_query_evolution.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/evolution/query_evolution.py src/paper_search/retrieval/openalex.py src/paper_search/retrieval/snapshot_adapters.py
git add -- configs/prompts/query_evolve.yaml src/paper_search/evolution/query_evolution.py src/paper_search/retrieval/openalex.py src/paper_search/retrieval/snapshot_adapters.py tests/unit/test_openalex.py tests/unit/test_query_evolution.py
git commit -m "feat: add bounded query evolution contract"
```

Expected: focused tests pass; Ruff and mypy exit `0`.

---

### Task 2: Implement exact baseline reconstruction, metrics, and Gates

**Files:**
- Create: `src/paper_search/evaluation/query_evolution_probe.py`
- Create: `tests/evaluation/test_query_evolution_probe.py`

**Interfaces:**
- `reconstruct_frozen_baseline(inputs, replay_provider) -> FrozenProbeBaseline`
- `select_probe_query_ids(baseline, gold, availability) -> tuple[str, ...]`
- `merge_probe_results(baseline, additions) -> ProbeProjection`
- `evaluate_probe(baseline, projection, gold, id_map, integrity) -> ProbeEvaluation`
- `public_probe_report(evaluation) -> PublicProbeReport`

- [ ] **Step 1: Write RED tests**

Cover: exact 60-query ordered reconstruction and 2,910 total; 14/8 baseline scores computed only after passing gold to evaluation; source/input/manifest/path, availability hash, lock hash, and post-seal gold hash mismatch fail-close; the fixed 55-query queue follows frozen 60-query order; byte-identical non-target queries; baseline results first, generated `search-1` then `search-2`; canonical ID first occurrence; exactly one `openalex` fusion source; existing hard filter, deduplication and RRF; candidate/Top-50 gold retention; MRR/NDCG and all deltas; strict three-state Gate boundaries; fixed minimal run reasons; finite JSON; recursive aggregate-only privacy; `invalid_work` warning versus request-level failure; and non-zero production estimates computed per operation type and usage dimension as `max(maximum_actual, ceil(p95 * 1.2))`, excluding unscheduled zero-usage slots.

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_query_evolution_probe.py -q
```

Expected: import/collection failure.

- [ ] **Step 2: Implement reconstruction and scoring**

Reconstruct baseline OpenAlex calls through `DependencySnapshotReader` and `ReplaySearchProvider`; do not copy raw-response parsers from `scripts/analyze_title_retention.py`. Build each projection with the production-equivalent order:

```python
def project_openalex_stream(
    spec: QuerySpec,
    baseline: Sequence[ProviderResult[list[Paper]]],
    additions: Sequence[ProviderResult[list[Paper]]],
) -> QueryProjection:
    ordered: list[Paper] = []
    seen: set[str] = set()
    for result in (*baseline, *additions):
        for paper in result.data:
            if paper.canonical_id not in seen:
                seen.add(paper.canonical_id)
                ordered.append(paper)
    filtered = apply_hard_filters(ordered, spec)
    accepted = {item.paper.canonical_id for item in filtered.accepted}
    fused = fuse_provider_results(
        {"openalex": _offline_provider_result(ordered)}, method="rrf"
    )
    return QueryProjection(
        candidate_papers=ordered,
        top50_ids=[
            item.paper.canonical_id
            for item in fused
            if item.paper.canonical_id in accepted
        ][:50],
        fusion_sources=("openalex",),
        hard_filter_rejections=len(filtered.rejected),
    )
```

Use existing `evaluate(...)` and `evaluate_ranking(...)`; calculate MRR/NDCG from this baseline rather than copying the earlier title experiment. Define `QueryTerminal` as `generated | no_op | integrity_failure | dependency_failure | accounting_failure | snapshot_failure | cancelled | not_scheduled`, `CaptureReplayMatch` as `matched | mismatched | not_evaluated`, `GateStatus` as `passed | failed | not_evaluated`, and run reasons as `preflight_failed | generation_failed | dependency_failed | accounting_failed | snapshot_failed | replay_mismatch | cancelled | gate_b_failed | gate_c_failed`. Gate predicates are exactly:

- Gate A: exact baseline and denominator, one terminal record for every locked query, capture/replay hash equality, zero integrity/provenance/unaccounted-usage failures, limits respected, aggregate-only output. A decoded OpenAlex page that retains valid data with `invalid_work` is a warning; request-level failure or retry exhaustion fails Gate A. If Gate A fails, B/C are `not_evaluated`.
- Gate B: Gate A; candidate gold `>14`; at least one newly retrieved prior `not_retrieved` association; all prior 14 candidate gold retained; capture interfaces cannot accept availability/gold/identifier maps, the network phase does not open them, and payload construction is not changed by gold. If Gate B fails, C is `not_evaluated`.
- Gate C: Gate B; Top-50 gold `>8`; all prior 8 retained; macro F1 delta `>=0.01`; recall/MRR/NDCG non-regression; hard-filter loss not increased; non-zero balanced production estimate.

- [ ] **Step 3: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/evaluation/query_evolution_probe.py tests/evaluation/test_query_evolution_probe.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/evaluation/query_evolution_probe.py
git add -- src/paper_search/evaluation/query_evolution_probe.py tests/evaluation/test_query_evolution_probe.py
git commit -m "feat: add query evolution probe evaluation"
```

---

### Task 3: Implement the bounded runner, CLI, capture, and offline replay

**Files:**
- Create: `scripts/probe_query_evolution.py`
- Create: `tests/integration/test_query_evolution_probe.py`

**Interfaces:**
- `preflight_probe(frozen_inputs, gold, id_map, availability, ledger) -> ProbeLock`
- `reserve_probe_operations(lock, ledger) -> ProbeReservations`
- `capture_probe(lock, runtime, reservations) -> CapturedProbe`
- `replay_probe(lock, replay_trace, snapshot_reader) -> ReplayedProbe`
- `run_probe(lock_path, runtime, deferred_evaluation_inputs) -> ProbeRunResult`

`deferred_evaluation_inputs` is a zero-argument loader for gold and identifier map paths; `run_probe` must not invoke it until snapshots are sealed, a complete run's replay is finished, and `capture_replay_match == "matched"`. `ReplayTrace` contains only ordered query/operation identities, canonical request identities, snapshot refs and terminals—never online normalized business fields. Neither `capture_probe` nor `replay_probe` accepts availability, gold, identifier-map objects or their paths.

- [ ] **Step 1: Write RED mocked integration and CLI tests**

Use only `httpx.MockTransport`. Cover: self-hashed immutable lock; frozen availability hash, probe-code hash and 55-query source order; `run --lock` rejecting lock/input/code/checkpoint drift before network; no availability/gold/id-map open during run before snapshot seal; unchanged hard-filter pass-through and `QuerySpec` post-filter; valid two-query and legal no-op paths; invalid LLM JSON/schema/year; OpenAlex data plus warning-only `invalid_work`; 429 success and exhausted 5xx/timeout; identical LLM and OpenAlex canonical requests dispatched once, then reused with `cache_hit=True`, zero usage/latency and no duplicate usage aggregation; cancellation; controller/ledger mismatch; partial pre-reservation failure with zero network; unused zero-usage slots; 55/110 logical and 165/330 attempt caps; 3,600-second cancellation; 3,900-second ledger TTL; snapshot confinement/seal; replay with a transport that raises if called; fixed comparable-business fields and equal hashes for complete runs; null replay hash, `not_evaluated` match and no deferred gold load for technical failures; complete failure/`not_scheduled` outcomes; three-state Gates; default preflight; live flag; env-key validation; atomic writes; existing output/run-ID rejection; no real public evidence from mocked tests; and secret non-disclosure.

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/integration/test_query_evolution_probe.py -q
```

Expected: import/collection failure.

- [ ] **Step 2: Implement immutable preflight lock, estimates, and all pre-request reservations**

Derive estimates from the frozen balanced budget, retry/timeouts, model, adapter, and pricing policy. Current locked estimates are: LLM 3 calls, 20,000 input tokens, 4,000 output tokens, 60,000 ms; each OpenAlex slot 3 calls, 60,000 ms. Price them with `ActualCostPricer`.

```python
PROBE_GLOBAL_TIMEOUT_SECONDS = 3600
PROBE_LEDGER_TTL_SECONDS = 3900
OPERATIONS = ("evolve", "search-1", "search-2")


def reserve_probe_operations(
    lock: ProbeLock,
    ledger: SQLiteBudgetLedger,
) -> dict[tuple[str, str], LedgerReservation]:
    created: dict[tuple[str, str], LedgerReservation] = {}
    try:
        for query_id in lock.query_ids:
            private_id = hashlib.sha256(query_id.encode()).hexdigest()[:16]
            for operation in OPERATIONS:
                created[(query_id, operation)] = ledger.reserve(
                    run_id=lock.probe_run_id,
                    query_id=f"{private_id}:{operation}",
                    estimate=lock.estimate_for(operation),
                    run_cap_cny=DEV_RUN_CAP_CNY,
                )
    except BaseException:
        zero = UsageActual(cost_cny=Decimal("0"))
        for reservation in created.values():
            ledger.fail(reservation, zero)
        raise
    return created
```

`preflight` validates every frozen source hash including availability, reconstructs 60/2,910/14/8, and selects the 55 queries by filtering the frozen 60-query order. Compute `probe_code_sha256` from all tracked `src/**/*.py` plus `scripts/probe_query_evolution.py`, sorted by POSIX path; append each as `UTF-8 path + NUL + decimal byte length + NUL + raw bytes` before hashing. Write it with all other hashes, the ordered queue, prompt/config/model, estimates, limits, `probe_run_id`, and ledger checkpoint to canonical JSON. Compute `lock_sha256` as `sha256:<64 lowercase hex>` over the same object with that field omitted. It makes no reservation, env read, or network request.

`run --lock <path>` validates the lock self-hash, `probe_code_sha256`, frozen run artifacts and snapshot manifest, exact 60/2,910 reconstruction, prompt/config/model and locked project checkpoint without opening availability, gold or identifier map. It derives `runs/_diag_query_evolution_<probe-run-id>/` from the lock, requires that directory not to exist, and copies the validated lock byte-for-byte into it. Then reserve all 165 logical slots before network. Construct the ledger with `reservation_ttl_seconds=3900` and each query controller with `reservation_ttl_seconds=800`. Use a separate `query-evolve-v1` `OpenAICompatibleLLMClient`/`LiveCaptureLLMAnalyzer`, pass the exact prompt artifact SHA into snapshot identity, retain DeepSeek `thinking: disabled`, and share one `DependencyCaptureStore` with the existing `LiveCaptureSearchProvider`.

- [ ] **Step 3: Implement capture, canonical-request reuse, and terminal handling**

Pass frozen `SearchPlan.inherited_hard_filters` unchanged to every generated search, retain the unchanged `QuerySpec` post-filter, and cap each search at 50 results. Maintain a runner-local memo keyed from the same canonical inputs as snapshot identity: model/endpoint/prompt version/name/payload/artifact hash for LLM, and canonicalized query/filters/limit/adapter for OpenAlex. A duplicate key does not dispatch: copy the first result's data, errors and snapshot provenance into a slot-specific `ProviderResult(cache_hit=True, usage=UsageActual(cost_cny=Decimal("0")), latency_ms=0)`, release its request reservation, and finalize its persistent slot with the same zero actual usage. Never aggregate the first request's usage again.

Mirror each dispatched adapter terminal outcome through `SQLiteBudgetLedger.finalize_controller_actual`. Accounted provider errors remain settled; cancellation/unknown usage fail-closes. Map invalid LLM output to `integrity_failure`, request-level failure/retry exhaustion to `dependency_failure`, controller/usage/ledger mismatch to `accounting_failure`, snapshot write/seal failure to `snapshot_failure`, and timeout/operator cancellation to `cancelled`; `invalid_work` on a successfully decoded page remains a warning. Release unused request reservations and fail their ledger slots with zero actual usage. On any stop, finish the in-flight receipt, terminate all remaining slots, seal snapshots where possible, and emit one outcome for every locked query; all later queries are `not_scheduled`.

- [ ] **Step 4: Implement replay, business hashing, CLI boundaries, and commit**

For a complete technical run, use sealed `DependencySnapshotReader`, `ReplayLLMAnalyzer(prompt_version="query-evolve-v1")`, and `ReplaySearchProvider(dependency="openalex")`. Use `ReplayTrace` only to locate and validate the ordered requests; re-derive `generated`/`no_op`, proposals and searches/projections from snapshots, then compare the terminal to the trace without trusting online normalized output. If any query has a technical-failure terminal, seal the available evidence but set `replay_business_sha256=None` and `capture_replay_match="not_evaluated"`; Gate A fails without synthesizing missing operations.

Define the private `ReplayComparableProbe` as: lock hash and frozen query order; then per query its ID, terminal, normalized proposal, ordered search slots containing normalized `Paper.model_dump(mode="json")` data plus stable error fields (`provider`, `code`, `retryable`), accepted/rejected canonical-ID order, candidate canonical-ID order, and Top-50 canonical-ID order. Exclude time, usage, headers, provider request IDs, snapshot refs, and paths. Canonical JSON uses sorted keys, compact separators, `allow_nan=False`, UTF-8, and a final newline. Gate A requires byte-identical capture/replay SHA-256.

Default the CLI to `preflight`; require `run --lock ... --allow-live`. The run command may accept gold and id-map paths only for a deferred loader that is invoked after a complete run's seal and matched replay, where their hashes and 14/8 baseline are rechecked before scoring; technical failures or replay mismatch never invoke it. Load only `LLM_API_KEY` plus contiguous `OPENALEX_API_KEY...` from the explicitly supplied env file; ignore model/base-URL env values. Use frozen `deepseek-v4-flash` and `https://api.deepseek.com/v1`. Private output is exactly `probe.lock.json`, `outcomes.jsonl`, `snapshots/`, and `result.json`. Only a completed real run may create aggregate-only public evidence.

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_query_evolution.py tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check scripts/probe_query_evolution.py tests/integration/test_query_evolution_probe.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts/probe_query_evolution.py
git add -- scripts/probe_query_evolution.py tests/integration/test_query_evolution_probe.py
git commit -m "feat: add bounded query evolution probe runner"
```

---

### Task 4: Verify offline, run only preflight, and record the handoff

**Files:**
- Modify: `HANDOFF.md`
- Modify: `docs/retrieval-roadmap.md`

- [ ] **Step 1: Run complete offline verification**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_query_evolution.py tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -q
$trackedPython = @(git ls-files '*.py')
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check -- $trackedPython
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts/analyze_gold_bottlenecks.py scripts/probe_query_evolution.py
git diff --check
```

Expected: no new failure beyond the documented Windows GBK packaging environment failure; focused tests, Ruff, mypy, and diff check clean. Do not claim the full suite is all-green if that environment failure remains.

- [ ] **Step 2: Run the real offline, zero-network preflight**

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
  --out runs/_diag_query_evolution_preflight/probe.lock.json
```

Expected aggregate output: `preflight_complete=true`, verified availability hash, non-zero `probe_code_sha256`, verified `lock_sha256`, 55 queries in frozen source order, 55/110 logical operations, 165/330 attempt caps, 3,600-second timeout, non-zero worst-case cost, `probe_run_id`, expected future run directory, and current project checkpoint. No `.env` read, reservation, or network request.

- [ ] **Step 3: Record state, verify scope, and commit**

Update `HANDOFF.md` and `docs/retrieval-roadmap.md` with implementation verification, exact preflight result/checkpoint, and the boundary that no live probe, lock rebuild, readiness, formal capture/replay/compare, or validation ran. The next action is one separately authorized bounded `run`, not prompt/ranking variants.

```powershell
git diff -- docs/evidence runs/candidate.lock.yaml
git status --short
git add -- HANDOFF.md docs/retrieval-roadmap.md
git commit -m "docs: prepare query evolution probe execution"
```

Expected: historical evidence and candidate lock unchanged; untracked ledger and deliverables preserved. Stop and request live authorization.

---

## Final self-review checklist

- [ ] Every design requirement maps to Tasks 1–4; production integration remains out of scope.
- [ ] `preflight` alone may read availability/gold/id-map before execution; the `run --lock` capture/replay phase reads none of them, and only the post-seal deferred evaluator verifies gold/id-map hashes.
- [ ] The lock self-hash, probe-code hash, availability hash and frozen-source query order are deterministic and fail closed.
- [ ] Generation payload and capture interfaces cannot receive availability, gold, labels, identifier maps, or `inherited_hard_filters`.
- [ ] Mechanical validation makes no false semantic guarantee.
- [ ] Query validation and OpenAlex request identity reuse the same behavior-preserving canonicalizer.
- [ ] Duplicate canonical LLM/OpenAlex requests dispatch once and leave every logical slot with an auditable terminal state.
- [ ] New results append to one OpenAlex stream; no hidden ranking variable exists.
- [ ] Every logical slot has one auditable ledger terminal state; retry attempts and global timeout are separate counters.
- [ ] Capture and replay use the same parser and produce identical `ReplayComparableProbe` bytes; unstable operational fields are excluded.
- [ ] Every locked query has one fixed terminal; capture/replay match and Gate A/B/C use their fixed three-state enums consistently.
- [ ] Public schemas reject query/generated text, IDs, titles, raw responses, secrets, and unsanitized request IDs.
- [ ] Automated verification is offline; the plan stops after zero-network `preflight`.
