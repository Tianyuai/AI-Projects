# Query Evolution Bounded Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a gold-isolated, budget-bounded Query Evolution probe that exactly reconstructs the frozen dev baseline, captures at most one LLM proposal and two OpenAlex searches for each of 55 locked queries, replays the new evidence offline, and decides Gates A/B/C without changing production behavior.

**Architecture:** Add one strict generation module and one pure evaluation module. Keep credentials, persistent reservations, capture/replay orchestration, deferred gold loading, and atomic writes in a standalone diagnostic CLI. Reuse existing live/replay adapters, hard filters, deduplication, one-source RRF, metrics, pricing, ledger, and snapshot v2.

**Tech Stack:** Python 3.11, Pydantic 2, `httpx`, PyYAML, SQLite, pytest, Ruff, mypy strict.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-09-query-evolution-bounded-probe-design.md` exactly.
- Use the existing worktree and branch; preserve untracked `data/budget_ledger.sqlite3` and `deliverables/`.
- Automated work is offline: do not read `D:\AI Projects\Projects\.env`, use real network, rebuild the candidate lock, run readiness, or start formal capture/replay/validation.
- Stop implementation after the offline, zero-network `preflight`; a real bounded `run` requires separate live authorization.
- Do not modify production composition, `EvolutionCoordinator`, API, UI, `configs/ablations.yaml`, historical runs, or historical evidence.
- Fixed source: `runs/dev-20260809T061903Z-9bd861e90299`; exact baseline: 60 queries, 2,910 Top-50 outputs, 14 candidate-pool gold associations, 8 Top-50 gold associations.
- Limits: 55/110 logical LLM/OpenAlex operations, 165/330 retry-inclusive attempts, 3,600-second batch timeout, 3,900-second ledger TTL.
- Capture code cannot accept or load gold. Gold may be used only to create the private queue lock during `preflight` and to score after the new snapshot is sealed.
- Use TDD and commit after each task.

---

### Task 1: Implement the strict, gold-free generation contract

**Files:**
- Create: `configs/prompts/query_evolve.yaml`
- Create: `src/paper_search/evolution/query_evolution.py`
- Create: `tests/unit/test_query_evolution.py`

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
  - Copy every source_facets value exactly from the payload.
```

Tests must cover: strict 0–2 proposal schema; legal no-op exclusivity; extra fields; deterministic facet order; empty-constraint `QuerySpec`; NFKC/whitespace normalization; empty, control-character, over-300-character, duplicate, and conflicting-year rejection; exact source-facet membership; unchanged hard filters; analyzer error and invalid-schema classification; snapshot refs; and recursive absence of query ID/gold fields from the payload.

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
    inherited_hard_filters: dict[str, object]
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

Normalize with Unicode NFKC plus collapsed whitespace. Deterministically reject only mechanically provable violations. Keep unrelated entity/venue avoidance as prompt policy; do not add a lexicon, second model, repair call, or rules fallback. Use the existing analyzer method `generate_json(prompt_name="query_evolve", payload=context.model_dump(mode="json"), reservation=reservation)`; invalid provider output is `integrity_failure`, never `no_op`.

- [ ] **Step 3: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_query_evolution.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/evolution/query_evolution.py tests/unit/test_query_evolution.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/evolution/query_evolution.py
git add -- configs/prompts/query_evolve.yaml src/paper_search/evolution/query_evolution.py tests/unit/test_query_evolution.py
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

Cover: exact 60-query ordered reconstruction and 2,910 total; 14/8 baseline scores computed only after passing gold to evaluation; source/input/manifest/path mismatch fail-close; fixed 55-query queue; byte-identical non-target queries; baseline results first, generated `search-1` then `search-2`; canonical ID first occurrence; exactly one `openalex` fusion source; existing hard filter, deduplication and RRF; candidate/Top-50 gold retention; MRR/NDCG and all deltas; strict Gate boundaries; Gate A suppressing B/C; fixed reason codes; finite JSON; recursive aggregate-only privacy; and non-zero production estimates computed per usage dimension as `max(maximum_actual, ceil(p95 * 1.2))`.

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

Use existing `evaluate(...)` and `evaluate_ranking(...)`; calculate MRR/NDCG from this baseline rather than copying the earlier title experiment. Gate predicates are exactly:

- Gate A: exact baseline and denominator, complete terminal states, capture/replay hash equality, zero integrity/provenance/unaccounted-usage failures, limits respected, aggregate-only output.
- Gate B: Gate A; candidate gold `>14`; at least one newly retrieved prior `not_retrieved` association; all prior 14 candidate gold retained; zero gold fields in generator payloads.
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

### Task 3: Implement bounded capture, ledger finalization, and offline replay

**Files:**
- Create: `scripts/probe_query_evolution.py`
- Create: `tests/integration/test_query_evolution_probe.py`

**Interfaces:**
- `preflight_probe(...) -> ProbeLock`
- `reserve_probe_operations(...) -> ProbeReservations`
- `capture_probe(...) -> CapturedProbe`
- `replay_probe(...) -> ReplayedProbe`
- `run_probe(...) -> ProbeRunResult`

- [ ] **Step 1: Write RED mocked integration tests**

Use only `httpx.MockTransport`. Cover: valid two-query and legal no-op paths; invalid LLM JSON/schema/year; OpenAlex data plus `invalid_work`; 429 success and exhausted 5xx/timeout; cancellation; controller/ledger mismatch; partial pre-reservation failure with zero network; unused zero-usage slots; 55/110 logical and 165/330 attempt caps; 3,600-second cancellation; 3,900-second ledger TTL; snapshot confinement/seal; replay with a transport that raises if called; deferred gold loader; equal capture/replay business hashes; and secret non-disclosure.

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/integration/test_query_evolution_probe.py -q
```

Expected: import/collection failure.

- [ ] **Step 2: Implement estimates and all pre-request reservations**

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

Recheck the locked project checkpoint, then reserve all 165 logical slots before network. Construct the ledger with `reservation_ttl_seconds=3900` and each query controller with `reservation_ttl_seconds=800`. Use a separate `query-evolve-v1` `OpenAICompatibleLLMClient`/`LiveCaptureLLMAnalyzer`, pass the exact prompt artifact SHA into snapshot identity, retain DeepSeek `thinking: disabled`, and share one `DependencyCaptureStore` with the existing `LiveCaptureSearchProvider`.

Pass frozen `SearchPlan.inherited_hard_filters` unchanged to every generated search, retain the unchanged `QuerySpec` post-filter, and cap each search at 50 results. Mirror each adapter terminal outcome through `SQLiteBudgetLedger.finalize_controller_actual`. Accounted provider errors remain settled; cancellation/unknown usage fail-closes. Release unused request reservations and fail their ledger slots with zero actual usage. On any stop, finish the in-flight receipt, terminate all remaining slots, and seal snapshots.

- [ ] **Step 3: Implement zero-network replay and business hashing**

Use sealed `DependencySnapshotReader`, `ReplayLLMAnalyzer(prompt_version="query-evolve-v1")`, and `ReplaySearchProvider(dependency="openalex")`. Re-parse proposals and rebuild searches/projections from snapshots; do not trust online normalized output. Canonical JSON uses sorted keys, compact separators, `allow_nan=False`, UTF-8, and a final newline. Gate A requires byte-identical capture/replay SHA-256.

- [ ] **Step 4: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_query_evolution.py tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check scripts/probe_query_evolution.py tests/integration/test_query_evolution_probe.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts/probe_query_evolution.py
git add -- scripts/probe_query_evolution.py tests/integration/test_query_evolution_probe.py
git commit -m "feat: add bounded query evolution probe runner"
```

---

### Task 4: Finish CLI boundaries, verify offline, and run only preflight

**Files:**
- Modify: `scripts/probe_query_evolution.py`
- Modify: `tests/integration/test_query_evolution_probe.py`
- Modify: `HANDOFF.md`
- Modify: `docs/retrieval-roadmap.md`

- [ ] **Step 1: Test and implement the thin CLI**

Default to `preflight`; require `run --allow-live`. Tests must cover: no env load or network in preflight; live flag required; base OpenAlex key plus contiguous `_2...` numbering; missing/gapped/empty keys; model/base URL env values ignored; secret redaction; atomic writes; existing output/run ID rejection; and no real public evidence from mocked tests.

`preflight` verifies frozen hashes, source status/Gate, prompt hash, 134/134 availability evidence, exact 60/2,910/14/8 reconstruction, fixed 55-query queue, ledger checkpoint, and non-zero priced estimates. It writes only a private lock and makes no reservation or network request.

`run` repeats preflight, rechecks the ledger checkpoint, requires `--allow-live`, and loads only `LLM_API_KEY` plus contiguous `OPENALEX_API_KEY...` from the explicitly supplied env file. It uses frozen `deepseek-v4-flash` and `https://api.deepseek.com/v1`. Seal snapshots before invoking the deferred gold loader. Private output under `runs/_diag_query_evolution_<run-id>/` is exactly `probe.lock.json`, `outcomes.jsonl`, `snapshots/`, and `result.json`. Only a completed real run may create `docs/evidence/query-evolution-probe-<date>.json` and `docs/query-evolution-probe-<date>.md`, after recursive aggregate-only validation.

- [ ] **Step 2: Run complete offline verification**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_query_evolution.py tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -q
$trackedPython = @(git ls-files '*.py')
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check -- $trackedPython
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts/analyze_gold_bottlenecks.py scripts/probe_query_evolution.py
git diff --check
```

Expected: no new failure beyond the documented Windows GBK packaging environment failure; focused tests, Ruff, mypy, and diff check clean. Do not claim the full suite is all-green if that environment failure remains.

- [ ] **Step 3: Run the real offline, zero-network preflight**

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

Expected aggregate output: `preflight_complete=true`, 55 queries, 55/110 logical operations, 165/330 attempt caps, 3,600-second timeout, non-zero worst-case cost, and current project checkpoint. No `.env` read, reservation, or network request.

- [ ] **Step 4: Record state, verify scope, and commit**

Update `HANDOFF.md` and `docs/retrieval-roadmap.md` with implementation verification, exact preflight result/checkpoint, and the boundary that no live probe, lock rebuild, readiness, formal capture/replay/compare, or validation ran. The next action is one separately authorized bounded `run`, not prompt/ranking variants.

```powershell
git diff -- docs/evidence runs/candidate.lock.yaml
git status --short
git add -- scripts/probe_query_evolution.py tests/integration/test_query_evolution_probe.py HANDOFF.md docs/retrieval-roadmap.md
git commit -m "docs: prepare query evolution probe execution"
```

Expected: historical evidence and candidate lock unchanged; untracked ledger and deliverables preserved. Stop and request live authorization.

---

## Final self-review checklist

- [ ] Every design requirement maps to Tasks 1–4; production integration remains out of scope.
- [ ] Generation payload and capture interfaces cannot receive gold.
- [ ] Mechanical validation makes no false semantic guarantee.
- [ ] New results append to one OpenAlex stream; no hidden ranking variable exists.
- [ ] Every logical slot has one auditable ledger terminal state; retry attempts and global timeout are separate counters.
- [ ] Capture and replay use the same parser and produce identical canonical business bytes.
- [ ] Public schemas reject query/generated text, IDs, titles, raw responses, secrets, and unsanitized request IDs.
- [ ] Automated verification is offline; the plan stops after zero-network `preflight`.
