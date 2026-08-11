# Sealed Query Recomposition Offline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one zero-network, three-variant offline comparison that determines whether Prompt v2's sealed query-result slots contain usable Top-50 ranking signal.

**Architecture:** A pure evaluation module owns the three fixed composition algorithms, verified-identifier scoring, and the four-way conclusion. A fixed-path script reuses the existing sealed-probe verification path, publishes aggregate-only canonical evidence, and cannot accept experimental parameters. The formal command runs once only after all offline gates pass.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, existing Paper Search evaluation/filter/fusion/snapshot modules, Ruff, mypy.

## Global Constraints

- Work in the existing linked worktree `D:\AI Projects\.worktrees\week3` on `codex/query-evolution-gate-contracts`; do not create another worktree.
- Preserve the user's existing changes in `HANDOFF.md`, `docs/retrieval-roadmap.md`, `data/budget_ledger.sqlite3`, `deliverables/`, and `docs/evidence/identifier-map-semantic-audit-2026-08-10.json`.
- Implement exactly `append_v2`, `round_robin_slots`, and `rrf_slots_k60`; no parameter grids, fallback variants, or per-query selection.
- Do not read `.env` or the budget ledger; do not construct network clients or run readiness/live capture/replay/compare.
- Prompt v2 materials must remain bound to `dev-20260809T061903Z-9bd861e90299`; 2026-08-10 formal and 2026-08-05 title values are external aggregate benchmarks only.
- Combination functions cannot receive Gold or `IdentifierMap`; Gold enters only after all three projections are complete.
- The formal `run` command executes at most once and never overwrites either evidence target.

---

### Task 1: Pure recomposition and scoring core

**Files:**
- Create: `src/paper_search/evaluation/query_recomposition.py`
- Create: `tests/evaluation/test_query_recomposition.py`

**Interfaces:**
- Consumes: `Paper`, `QuerySpec`, `EvaluationQuery`, `IdentifierMap`, `ProviderResult`, `apply_hard_filters()`, `fuse_provider_results()`.
- Produces:
  - `RecompositionMethod = Literal["append_v2", "round_robin_slots", "rrf_slots_k60"]`
  - `RecompositionInput(query_id: str, query_spec: QuerySpec, baseline_slots: tuple[tuple[Paper, ...], ...], addition_slots: tuple[tuple[Paper, ...], ...], retrieved_paper_ids: tuple[str, ...], post_filter_paper_ids: tuple[str, ...])`
  - `RecompositionProjection(method, retrieved_ids, post_filter_ids, selected_ids)`
  - `RecompositionRow` with the six metrics, four stage counts, `retains_append_selected_gold`, and set-invariance flags
  - `SealedQueryRecompositionReport` with schema `sealed-query-recomposition-offline-v1`, exactly three ordered rows, input hashes, 17/30 external benchmarks, and one fixed conclusion
  - `compose_append(slots: Sequence[Sequence[Paper]]) -> tuple[Paper, ...]`
  - `compose_round_robin(slots: Sequence[Sequence[Paper]]) -> tuple[Paper, ...]`
  - `compose_rrf(slots: Sequence[Sequence[Paper]]) -> tuple[Paper, ...]`
  - `project_all(inputs: Sequence[RecompositionInput]) -> dict[RecompositionMethod, dict[str, RecompositionProjection]]`
  - `build_report(*, gold: Sequence[EvaluationQuery], identifier_map: IdentifierMap, projections: Mapping[RecompositionMethod, Mapping[str, RecompositionProjection]], input_hashes: Mapping[str, str], current_formal_selected: int, legacy_title_selected: int) -> SealedQueryRecompositionReport`

- [ ] **Step 1: Write RED tests for the three fixed compositions**

Use small real `Paper` sequences to assert exact append order, rank-wise round-robin order, and existing RRF `k=60` order. Cover duplicate canonical IDs, empty slots, one slot, deterministic ties, and prove all variants return the same canonical-ID set. The expected core is:

```python
assert ids(compose_append(((a, b), (c, a)))) == ("openalex:A", "openalex:B", "openalex:C")
assert ids(compose_round_robin(((a, b), (c, d)))) == (
    "openalex:A", "openalex:C", "openalex:B", "openalex:D"
)
assert ids(compose_rrf(((a, b), (b, c))))[0] == "openalex:B"
```

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_query_recomposition.py -q
```

Expected: collection/import failure because `paper_search.evaluation.query_recomposition` does not exist.

- [ ] **Step 2: Implement the minimal pure composition functions**

Use stable canonical-ID deduplication for append/round-robin and the current fusion function for RRF:

```python
def compose_append(slots: Sequence[Sequence[Paper]]) -> tuple[Paper, ...]:
    return _stable_unique(paper for slot in slots for paper in slot)

def compose_round_robin(slots: Sequence[Sequence[Paper]]) -> tuple[Paper, ...]:
    return _stable_unique(
        slot[rank]
        for rank in range(max((len(slot) for slot in slots), default=0))
        for slot in slots
        if rank < len(slot)
    )

def compose_rrf(slots: Sequence[Sequence[Paper]]) -> tuple[Paper, ...]:
    results = {
        f"slot-{index:03d}": offline_provider_result(slot)
        for index, slot in enumerate(slots)
    }
    return tuple(item.paper for item in fuse_provider_results(results, method="rrf", rrf_k=60))
```

Do not expose `rrf_k`, weights, limits, or arbitrary methods as parameters.

- [ ] **Step 3: Add RED tests for projection, semantic scoring, and conclusions**

Assert that projection inherits authoritative retrieved/post-filter streams, only adds filtered addition IDs, selects accepted composed papers, and never reads a Gold trap. Add fixed fixtures for all four outcomes:

```python
assert classify(experiment_invariants_ok=False, usable=False, selected=19) == "integrity_failure"
assert classify(integrity_ok=True, usable=False, selected=19) == "no_usable_recomposition_signal"
assert classify(integrity_ok=True, usable=True, selected=29) == "signal_insufficient"
assert classify(integrity_ok=True, usable=True, selected=30) == "legacy_benchmark_met"
```

Also require stage conservation at 143, finite metrics, row order, candidate/post-filter set invariance, full retention of append selected Gold, and no metric regression for a usable signal.

Run the Task 1 test file again. Expected: composition tests pass; projection/report tests fail because those interfaces are missing.

- [ ] **Step 4: Implement projection and report construction, then verify GREEN**

Construct all three projections with `project_all(inputs)` before calling `build_report(gold=gold, identifier_map=identifier_map, projections=projections, input_hashes=input_hashes, current_formal_selected=17, legacy_title_selected=30)`. Keep the stage precedence `selected -> ranked outside -> filtered -> not retrieved` identical to semantic rescore. Reject any report whose row order is not the fixed three methods or whose totals differ.

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_query_recomposition.py tests/evaluation/test_semantic_rescore.py tests/evaluation/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/evaluation/query_recomposition.py tests/evaluation/test_query_recomposition.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/evaluation/query_recomposition.py
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- src/paper_search/evaluation/query_recomposition.py tests/evaluation/test_query_recomposition.py
git diff --cached --check
git commit -m "feat: add sealed query recomposition core"
```

### Task 2: Verified fixed-source CLI and safe publication

**Files:**
- Modify: `scripts/rescore_identifier_semantics.py`
- Create: `scripts/analyze_sealed_query_recomposition.py`
- Modify: `tests/scripts/test_rescore_identifier_semantics.py`
- Create: `tests/scripts/test_analyze_sealed_query_recomposition.py`

**Interfaces:**
- Consumes: Task 1 models/functions; existing identifier-generation loader, probe lock/source/result/outcome/snapshot verification, canonical JSON and privacy scanners.
- Produces:
  - `VerifiedProbeMaterials(baseline_inputs: FrozenProbeInputs, baseline_executions: dict[str, EvaluationExecutionRecord], additions: dict[str, tuple[ProviderResult[list[Paper]], ...]], binding_hashes: dict[str, str])` in `scripts.rescore_identifier_semantics`
  - `load_verified_probe_materials(run_dir: Path, expected_query_ids: tuple[str, ...]) -> VerifiedProbeMaterials` reused by both rescore and recomposition scripts
  - fixed commands `python -m scripts.analyze_sealed_query_recomposition run` and `render-markdown`
  - no-replace outputs `docs/evidence/sealed-query-recomposition-offline-2026-08-11.json` and `docs/sealed-query-recomposition-offline-2026-08-11.md`

- [ ] **Step 1: Write RED tests for shared sealed-probe materials**

Refactor tests around observable behavior, not private helper calls. Require one verified load to validate lock, bound source hashes, result match, ordered outcomes, outcome hash, manifest identity, and every referenced snapshot; then require existing `load_probe_source()` to produce the same projection as before from the returned materials.

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/scripts/test_rescore_identifier_semantics.py -q
```

Expected: new tests fail because `VerifiedProbeMaterials` and `load_verified_probe_materials()` are missing.

- [ ] **Step 2: Extract the minimal shared verified-material loader and restore GREEN**

Move existing verification statements without weakening or reordering them. `load_probe_source()` must become a thin projection over the shared result; no historical source behavior or public rescore schema changes.

Run the rescore script tests plus `tests/evaluation/test_semantic_rescore.py`. Expected: exit 0 and existing formal evidence remains byte-for-byte untouched.

- [ ] **Step 3: Write RED tests for fixed orchestration and publication**

Require this exact operation order:

```python
assert events == [
    "verified-generation",
    "gold-order",
    "verified-probe-materials",
    "compose-all-without-gold",
    "verified-external-rescore",
    "score",
    "scan-json",
    "scan-markdown",
    "write-json",
    "write-markdown",
]
```

Tests must also prove:

- generation failure causes zero probe reads;
- generation/source/replay/snapshot/privacy failures exit safely without publishing an experiment report;
- Prompt v2 remains bound to the 2026-08-09 source;
- external rescore is canonical, passed, fixed-order, and generation-hash identical;
- `append_v2` must reproduce 101/0/23/19 before other rows are interpreted;
- CLI has no path/network/env/ledger/method/weight/threshold options;
- both privacy scanners run before either write;
- either existing target prevents all writes;
- Markdown recovery reads only canonical JSON and never recomposes/rescores;
- `main()` catches only expected `OSError`/`ValueError` and emits one fixed safe error line.

Run the new script test file. Expected: import/interface failures for the missing script.

- [ ] **Step 4: Implement the fixed CLI and publication path, then verify GREEN**

Use fixed module constants only. Canonical JSON must use sorted keys, compact separators, `allow_nan=False`, UTF-8, and one trailing newline. Publish with sibling temporary files, flush/fsync, and no-replace creation. Keep `run` and `render-markdown` as the only subcommands.

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/scripts/test_analyze_sealed_query_recomposition.py tests/scripts/test_rescore_identifier_semantics.py tests/evaluation/test_query_recomposition.py tests/evaluation/test_semantic_rescore.py tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src scripts tests/evaluation/test_query_recomposition.py tests/scripts/test_analyze_sealed_query_recomposition.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- scripts/rescore_identifier_semantics.py scripts/analyze_sealed_query_recomposition.py tests/scripts/test_rescore_identifier_semantics.py tests/scripts/test_analyze_sealed_query_recomposition.py
git diff --cached --check
git commit -m "feat: publish sealed query recomposition diagnostic"
```

### Task 3: Full gate, one formal run, and evidence

**Files:**
- Create once: `docs/evidence/sealed-query-recomposition-offline-2026-08-11.json`
- Create once: `docs/sealed-query-recomposition-offline-2026-08-11.md`

**Interfaces:**
- Consumes: the fixed `run` command from Task 2.
- Produces: one validated aggregate report and one of the four terminal conclusions; no automatic follow-on experiment.

- [ ] **Step 1: Run the complete offline pre-execution gate**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -m "not online" -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src scripts tests
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts
git diff --check
```

Expected: every command exits 0. On any failure, stop before creating formal evidence.

- [ ] **Step 2: Check no-overwrite precondition and execute exactly once**

```powershell
Test-Path -LiteralPath 'docs/evidence/sealed-query-recomposition-offline-2026-08-11.json'
Test-Path -LiteralPath 'docs/sealed-query-recomposition-offline-2026-08-11.md'
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m scripts.analyze_sealed_query_recomposition run
```

Expected for the current verified inputs: both `Test-Path` calls print `False`; `run` exits 0. Never rerun `run`. If an input/binding/privacy precondition fails, expect the fixed safe error and no report. If JSON exists and Markdown alone is absent, use `render-markdown`; for every other partial/failure state, stop for human review.

- [ ] **Step 3: Validate the formal evidence and enforce the terminal outcome**

Parse JSON with `SealedQueryRecompositionReport`, rerun both public privacy scanners, compare Markdown with deterministic rendering, verify three ordered rows, 143-stage conservation, identical retrieved/post-filter sets, append 101/0/23/19, and exactly one terminal conclusion.

Apply the stop rule without further experiments. A generation/source/replay/snapshot/privacy failure is not one of the four report conclusions: it produces no report and requires human inspection. For a valid published report:

- `integrity_failure`: stop and request human inspection;
- `no_usable_recomposition_signal` or `signal_insufficient`: stop recomposition and recommend a separate title-informed retrieval design;
- `legacy_benchmark_met`: recommend a separate production-equivalent integration design, without modifying production here.

- [ ] **Step 4: Commit only the two aggregate evidence files**

```powershell
git add -- docs/evidence/sealed-query-recomposition-offline-2026-08-11.json docs/sealed-query-recomposition-offline-2026-08-11.md
git diff --cached --check
git commit -m "docs: record sealed query recomposition result"
```

- [ ] **Step 5: Run post-commit verification**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_query_recomposition.py tests/scripts/test_analyze_sealed_query_recomposition.py -q
git status --short
git log --oneline -5
```

Expected: focused tests exit 0; status contains only the user's preserved pre-existing changes; no merge or push occurs.
