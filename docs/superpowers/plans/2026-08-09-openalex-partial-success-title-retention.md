# OpenAlex Partial Success and Title Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve valid papers from error-bearing OpenAlex title-search pages, then reproducibly compare deterministic Top-50 title-retention strategies on sealed offline evidence.

**Architecture:** Make one behavior change in `LLMTitleCandidateStage`: diagnostics remain unchanged, but non-empty provider data is consumed even when the same result has structured errors. Add a standalone aggregate-only diagnostic that reconstructs the historical two-source RRF exactly, fail-closes unless all 60 stored Top-50 sequences match, then evaluates repaired RRF, weighted RRF, and reserved title-slot variants without changing production ranking.

**Tech Stack:** Python 3.12, Pydantic v2 domain models, pytest, existing OpenAlex decoder, `fuse_provider_results`, `evaluate`, `evaluate_ranking`, sealed JSON/JSONL snapshots.

## Global Constraints

- Execute in `D:\AI Projects\.worktrees\week3`; it is already a linked worktree on `codex/project-document-handoff`.
- Do not run live capture or any network request.
- Do not modify `data/`, `runs/_diag_*`, the project ledger, or historical run artifacts.
- Do not print or persist frozen query text, gold IDs, generated titles, response bodies, keys, or request IDs.
- Preserve `invalid_work` diagnostics and snapshot provenance; do not weaken the OpenAlex decoder.
- The offline source run is diagnostic-only: `runs/dev-20260805T035209Z-7af4b103f6cc`.
- Fail closed unless reconstruction exactly matches 60/60 historical `selected_paper_ids` sequences and 2,908 total results.
- Compare variants only with the same-run historical macro F1 `0.00813534234362943` and macro recall `0.09472222222222222`.
- A promotable variant must improve macro F1, retain every previously selected exact gold per query, and not reduce Recall@5/10/20, macro MRR, or macro NDCG.
- Preserve the existing untracked `data/budget_ledger.sqlite3` and `deliverables/`.

---

### Task 1: Preserve valid papers from partial-success title searches

**Files:**
- Modify: `tests/unit/test_title_candidates.py`
- Modify: `src/paper_search/retrieval/title_candidates.py:470-486`

**Interfaces:**
- Consumes: `ProviderResult[list[Paper]]` whose `data` and `errors` may both be non-empty.
- Produces: unchanged `LLMTitleCandidateStage.recall(...) -> TitleCandidateRecallResult`; valid `search.data` is retained while `_provider_diagnostic(search)` still exposes errors.

- [ ] **Step 1: Add a provider fixture that returns valid data and `invalid_work` together**

Extend `FakeTitleProvider` only through its existing constructor by adding an optional `error_queries: set[str]`, distinct from `failed_queries`. For an error query, return normal fixture data plus this error:

```python
ErrorDetail(
    code="invalid_work",
    message="synthetic invalid sibling",
    retryable=False,
    provider="openalex",
)
```

Keep `failed_queries` behavior unchanged so existing total-failure tests remain meaningful.

- [ ] **Step 2: Write the failing regression test**

Add:

```python
def test_recall_keeps_valid_papers_from_error_bearing_search() -> None:
    controller = HardBudgetController(_budget())
    paper = Paper(canonical_id="openalex:W1", title="Valid sibling")
    analyzer = FakeTitleAnalyzer({"titles": ["Partial"]})
    provider = FakeTitleProvider(
        {"Partial": [paper]},
        error_queries={"Partial"},
    )
    stage = _stage(analyzer, provider)

    result = asyncio.run(stage.recall(_spec(), controller=controller))

    assert result.status == "applied"
    assert result.provider_result.data == [paper]
    assert result.titles_searched == 1
    assert result.diagnostics[-1].errors[0].code == "invalid_work"
    assert result.provider_result.usage.search_api_calls == 1
    assert controller.committed_usage.search_api_calls == 1
```

- [ ] **Step 3: Run the new test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_title_candidates.py::test_recall_keeps_valid_papers_from_error_bearing_search -q
```

Expected: FAIL because `result.status` is `degraded` and `provider_result.data` is empty under the current unconditional `continue`.

- [ ] **Step 4: Implement the minimal behavior change**

Replace:

```python
if search.errors:
    search_errors += 1
    continue
for paper in search.data:
```

with:

```python
if search.errors:
    search_errors += 1
for paper in search.data:
```

Do not modify decoder behavior, warnings, usage settlement, exception handling, or diagnostic serialization.

- [ ] **Step 5: Verify GREEN and focused regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_title_candidates.py tests/unit/test_openalex.py -q
```

Expected: all tests pass, including total failure, adapter exception, deduplication, and invalid sibling coverage.

- [ ] **Step 6: Commit the isolated repair**

```powershell
git add src/paper_search/retrieval/title_candidates.py tests/unit/test_title_candidates.py
git commit -m "fix: preserve partial title search results"
```

---

### Task 2: Build a fail-closed offline title-retention diagnostic

**Files:**
- Create: `scripts/analyze_title_retention.py`
- Create: `tests/scripts/test_analyze_title_retention.py`

**Interfaces:**
- Consumes: `--run`, `--gold`, `--id-map`, and `--out` paths; all inputs are local.
- Produces: aggregate-only JSON with schema version `title-retention-offline-v1`, reconstruction status, historical/repaired metrics, variant metrics, guard results, and one recommended variant or `null`.
- Produces pure helpers:
  - `weighted_rrf_ids(openalex, titles, eligible_ids, *, title_weight, limit=50) -> list[str]`
  - `reserve_title_slots(baseline_ids, title_ids, eligible_ids, *, minimum, limit=50) -> list[str]`
  - `retains_baseline_golds(gold_by_query, baseline, candidate, id_map) -> bool`

- [ ] **Step 1: Write pure strategy tests first**

Create synthetic tests that contain no frozen inputs:

```python
def test_weighted_rrf_preserves_denominator_and_promotes_title_source() -> None:
    openalex = [_paper("W1"), _paper("W2")]
    titles = [_paper("W2"), _paper("W3")]
    eligible = {"openalex:W1", "openalex:W2", "openalex:W3"}

    baseline = weighted_rrf_ids(
        openalex, titles, eligible, title_weight=1.0, limit=3
    )
    boosted = weighted_rrf_ids(
        openalex, titles, eligible, title_weight=3.0, limit=3
    )

    assert baseline[0] == "openalex:W2"
    assert boosted.index("openalex:W3") < boosted.index("openalex:W1")


def test_reserve_title_slots_replaces_lowest_non_title_only() -> None:
    result = reserve_title_slots(
        ["openalex:W1", "openalex:W2", "openalex:W3"],
        ["openalex:W4", "openalex:W2"],
        {"openalex:W1", "openalex:W2", "openalex:W3", "openalex:W4"},
        minimum=2,
        limit=3,
    )
    assert result == ["openalex:W1", "openalex:W2", "openalex:W4"]


def test_reserve_title_slots_never_adds_filtered_candidate() -> None:
    result = reserve_title_slots(
        ["openalex:W1", "openalex:W2"],
        ["openalex:W3"],
        {"openalex:W1", "openalex:W2"},
        minimum=1,
        limit=2,
    )
    assert result == ["openalex:W1", "openalex:W2"]
```

Also test invalid weights, invalid slot counts, deterministic ties, duplicate IDs, and the per-query baseline-gold retention guard.

- [ ] **Step 2: Run strategy tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_title_retention.py -q
```

Expected: collection/import failure because `scripts.analyze_title_retention` does not exist.

- [ ] **Step 3: Implement weighted RRF without changing production fusion**

Use production deduplication and source-rank construction, then recompute only the score:

```python
def weighted_rrf_ids(
    openalex: Sequence[Paper],
    titles: Sequence[Paper],
    eligible_ids: AbstractSet[str],
    *,
    title_weight: float,
    limit: int = 50,
) -> list[str]:
    if not math.isfinite(title_weight) or title_weight <= 0:
        raise ValueError("title_weight must be finite and positive")
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be a positive integer")
    fused = fuse_provider_results(
        {
            "openalex": _provider_result(openalex),
            "title_candidates": _provider_result(titles),
        },
        method="rrf",
        rrf_k=60,
    )
    scored = []
    for item in fused:
        score = sum(
            (title_weight if source == "title_candidates" else 1.0)
            / (60 + rank)
            for source, rank in item.source_ranks.items()
        )
        scored.append((score, item.paper.canonical_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [paper_id for _, paper_id in scored if paper_id in eligible_ids][:limit]
```

`_provider_result` must create a zero-usage, no-error `ProviderResult[list[Paper]]` with constant non-sensitive offline provenance.

- [ ] **Step 4: Implement deterministic title-slot reservation**

```python
def reserve_title_slots(
    baseline_ids: Sequence[str],
    title_ids: Sequence[str],
    eligible_ids: AbstractSet[str],
    *,
    minimum: int,
    limit: int = 50,
) -> list[str]:
    if type(minimum) is not int or minimum < 0:
        raise ValueError("minimum must be a nonnegative integer")
    if type(limit) is not int or limit <= 0 or minimum > limit:
        raise ValueError("limit must be positive and at least minimum")
    ranked_titles = list(dict.fromkeys(
        paper_id for paper_id in title_ids if paper_id in eligible_ids
    ))
    title_set = set(ranked_titles)
    selected = list(dict.fromkeys(
        paper_id for paper_id in baseline_ids if paper_id in eligible_ids
    ))[:limit]
    needed = max(0, minimum - sum(item in title_set for item in selected))
    additions = [item for item in ranked_titles if item not in selected][:needed]
    for addition in additions:
        if len(selected) < limit:
            selected.append(addition)
            continue
        replacement = next(
            (index for index in range(len(selected) - 1, -1, -1)
             if selected[index] not in title_set),
            None,
        )
        if replacement is None:
            break
        selected[replacement] = addition
    return selected
```

- [ ] **Step 5: Implement sealed-run reconstruction**

For each execution:

1. Identify the title-generation LLM diagnostic by parsing only the sealed LLM envelope and checking that the decoded object contains a `titles` key; never print its value.
2. Decode pre-title OpenAlex refs with `decode_openalex_page` and retain valid papers even when decode errors exist.
3. Build `title_historical` from post-title refs only when that diagnostic has no errors.
4. Build `title_repaired` from every valid decoded paper in post-title refs, while counting error-bearing responses in aggregate.
5. Deduplicate each source by canonical ID in encounter order.
6. Use stored `post_filter_paper_ids` as the eligibility set.

Before variants, reconstruct historical RRF with title weight `1.0` and compare the full ordered IDs to stored `business-results.jsonl`. Raise `ValueError("historical Top-50 reconstruction mismatch")` on any mismatch or if totals differ from 60 queries / 2,908 results.

- [ ] **Step 6: Implement scoring and promotion guards**

Load gold with `read_jsonl(..., EvaluationQuery)` and the identifier map with `IdentifierMap.from_path`. Build `PredictionRecord` lists, then call both:

```python
metrics = evaluate(gold, predictions, id_map=id_map)
ranking = evaluate_ranking(gold, predictions, id_map=id_map)
```

For each candidate, include summary metrics plus aggregate exact-gold and hit-query counts. `retains_baseline_golds` must resolve IDs and verify, for every query:

```python
baseline_gold_hits <= candidate_gold_hits
```

Evaluate these variants only:

```python
TITLE_WEIGHTS = (1.0, 1.25, 1.5, 2.0, 3.0)
TITLE_SLOT_MINIMUMS = (1, 2, 3, 5, 10)
```

Do not create a weight/slot Cartesian product. Mark `promotable` only when every global constraint is true. Pick the promotable variant with highest macro F1, then macro recall, macro NDCG, macro MRR, and finally lexical variant name; otherwise emit `recommended_variant: null`.

- [ ] **Step 7: Implement aggregate-only CLI output**

CLI:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_title_retention.py `
  --run runs/dev-20260805T035209Z-7af4b103f6cc `
  --gold data/dev/gold.jsonl `
  --id-map data/identifier-map.json `
  --out docs/evidence/title-retention-offline-2026-08-09.json
```

Write JSON atomically with sorted keys, UTF-8, `allow_nan=False`, and schema version `title-retention-offline-v1`. The JSON may contain run ID, hashes, aggregate counts, parameter values, metrics, booleans, and reason codes; it must not contain query IDs, paper IDs, titles, query text, response text, request IDs, or per-query records.

- [ ] **Step 8: Verify diagnostic tests GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_analyze_title_retention.py tests/unit/test_fusion.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit the offline diagnostic implementation**

```powershell
git add scripts/analyze_title_retention.py tests/scripts/test_analyze_title_retention.py
git commit -m "feat: add offline title retention analysis"
```

---

### Task 3: Run the offline comparison and record the decision

**Files:**
- Create: `docs/evidence/title-retention-offline-2026-08-09.json`
- Create: `docs/title-retention-offline-2026-08-09.md`
- Modify: `docs/title-candidate-stage-loss-2026-08-09.md`
- Modify: `docs/experiment-decisions.md`
- Modify: `docs/retrieval-roadmap.md`
- Modify: `HANDOFF.md`

**Interfaces:**
- Consumes: the CLI and schema from Task 2.
- Produces: one aggregate JSON evidence artifact and one concise human-readable decision report.

- [ ] **Step 1: Run the analysis with no network-capable command**

Run the CLI command from Task 2. Expected:

- `reconstruction.exact_query_sequences = 60`;
- `reconstruction.query_count = 60`;
- `reconstruction.total_selected = 2908`;
- exit code 0;
- no raw query, title, gold ID, paper ID, response, or request ID in stdout or JSON.

- [ ] **Step 2: Validate the JSON artifact independently**

Run a small read-only assertion that checks the schema version, 60/60 reconstruction, finite metric values, unique variant names, allowed parameter grids, complete guard booleans, and absence of forbidden key names (`query_id`, `paper_id`, `title`, `request_id`, `response`). Expected: `title_retention_artifact: valid`.

- [ ] **Step 3: Write the decision report**

The Markdown report must contain only:

1. repaired-page delta versus historical RRF;
2. compact table of weighted RRF and slot variants;
3. guard outcomes and recommended variant or explicit no-change decision;
4. limitations: historical diagnostic run, current verifier incompatibility, dev-only parameter selection, no live evidence;
5. next action requiring separate approval if a variant is promotable.

Do not repeat implementation details already present in the design or plan.

- [ ] **Step 4: Update active project documents minimally**

Update the existing title-loss report, experiment decision table, roadmap, and handoff with the measured result. Remove superseded “next step” wording. Do not copy the full variant matrix into multiple files; link to the report.

- [ ] **Step 5: Run focused and full verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_title_candidates.py tests/unit/test_openalex.py tests/unit/test_fusion.py tests/scripts/test_analyze_title_retention.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\mypy.exe src
git diff --check
```

Expected: pytest and Ruff pass. If mypy reports only the already documented pre-existing parser errors, record them exactly and do not conflate them with this task; any new error blocks completion.

- [ ] **Step 6: Commit evidence and documentation**

```powershell
git add docs/evidence/title-retention-offline-2026-08-09.json docs/title-retention-offline-2026-08-09.md docs/title-candidate-stage-loss-2026-08-09.md docs/experiment-decisions.md docs/retrieval-roadmap.md HANDOFF.md
git commit -m "docs: record offline title retention results"
```

- [ ] **Step 7: Final repository audit**

Confirm the branch contains the three task commits, `git status --short` shows only the pre-existing untracked ledger and `deliverables/`, and no candidate lock was rebuilt. Report the exact winning/no-winning conclusion, test counts, static-check status, and commit IDs.
