# Week 2 Task 5–7 Mock-Driven Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic, fully offline Task 5–7 contracts for query analysis, multi-source retrieval/fusion, budget enforcement, and minimal orchestration.

**Architecture:** External behavior is isolated behind injected LLM and HTTP transports. Pure parsing, planning, fusion, accounting, and orchestration code composes existing domain models, deduplication, filtering, lexical ranking, and cache behavior.

**Tech Stack:** Python 3.11, Pydantic 2, httpx MockTransport, SQLite/JSON persistence, pytest, Ruff, mypy, uv.

## Global Constraints

- Start from commit `92d804fdebd15467854b22f9fa42297369b6c70e` on `codex/week2-task5-7-mock`.
- Never read or load `.env`; every test command uses `--no-env-file`.
- Never call a real LLM, OpenAlex, Semantic Scholar, or other online API.
- Never read or modify private annotations, gold, PaSa raw data, split IDs, domain labels, manifest freeze identity/status, or `docs/superpowers/specs/2026-07-15-task2-evaluation-design.md`.
- Reuse existing models, OpenAlex, cache, deduplication, filtering, and lexical ranking.
- Do not add dependencies or change public domain models/retrieval semantics without stopping for approval.
- Do not implement Task 8 endpoints, UI, graphs, embeddings, rerankers, or data-driven tuning.
- Use `apply_patch` for edits and preserve unrelated changes.
- Before and after staging, inspect staged file names; do not push, merge, create a PR, or remove the worktree.

---

### Task 1: Mock-Driven Query Analysis and Planning

**Files:**
- Create: `src/paper_search/llm/__init__.py`
- Create: `src/paper_search/llm/client.py`
- Create: `src/paper_search/query/__init__.py`
- Create: `src/paper_search/query/parser.py`
- Create: `src/paper_search/query/planner.py`
- Create: `configs/prompts/query_analyze.yaml`
- Create: `tests/unit/test_llm_client.py`
- Create: `tests/unit/test_query_parser.py`
- Create: `tests/unit/test_query_planner.py`

**Interfaces:**
- Consumes: `BudgetReservation`, `ProviderResult[dict]`, `QuerySpec`, `SearchPlan`, `QueryAnalysisResult`, and `UsageActual`.
- Produces: `OpenAICompatibleLLMClient.generate_json(...)`, `QueryParser.analyze(...)`, `rule_fallback(query)`, and `QueryPlanner.finalize(spec, plan, max_subqueries)`.

- [ ] **Step 1: Add failing LLM adapter tests**

Test an injected `httpx.MockTransport` for valid JSON, malformed JSON, empty choices, timeout, usage fields, safe provenance, and absence of an API key or authorization header in serialized errors.

- [ ] **Step 2: Verify LLM tests are RED**

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_llm_client.py -q
```

Expected: collection/import failure because `paper_search.llm.client` does not exist.

- [ ] **Step 3: Implement the minimal LLM adapter**

Define a transport-backed client that posts only to its configured absolute OpenAI-compatible endpoint with redirects disabled, parses one assistant JSON object, records input/output tokens and one LLM call, returns structured errors, and exposes no credentials in errors or provenance.

- [ ] **Step 4: Verify LLM adapter GREEN**

Run the command from Step 2. Expected: all LLM client tests pass.

- [ ] **Step 5: Add failing parser and planner tests**

Cover one repair success, repair failure followed by rules, year/exclusion fallback, Pydantic validation, deterministic 3–5 subquery output, stable IDs/order, clipping, duplicate removal, and inherited years/venues/exclusions.

- [ ] **Step 6: Verify parser/planner tests are RED**

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_query_parser.py tests/unit/test_query_planner.py -q
```

Expected: import failures because parser and planner do not exist.

- [ ] **Step 7: Implement parser, planner, prompt, and exports**

The parser accepts a callable returning `ProviderResult[dict]`, validates `QueryAnalysisResult`, invokes the repair callable at most once, and otherwise returns a deterministic rule result. The planner sorts by priority then original position, removes duplicate query text, checks inherited hard filters, adds deterministic rule subqueries if fewer than three remain, and clips to `min(max_subqueries, 5)`.

- [ ] **Step 8: Verify Task 5 GREEN and regressions**

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_llm_client.py tests/unit/test_query_parser.py tests/unit/test_query_planner.py tests/unit/test_models.py tests/unit/test_budget.py -q
uv run --no-sync --no-env-file ruff check src/paper_search/llm src/paper_search/query tests/unit/test_llm_client.py tests/unit/test_query_parser.py tests/unit/test_query_planner.py
uv run --no-sync --no-env-file mypy src/paper_search/llm src/paper_search/query
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 9: Stage-check and commit Task 5**

Inspect `git diff --cached --name-only` before and after staging. Stage only Task 5 files plus this design/plan documentation, then commit:

```text
feat: add mock-driven query planning
```

### Task 2: Semantic Scholar and Deterministic Fusion

**Files:**
- Create: `src/paper_search/retrieval/base.py`
- Create: `src/paper_search/retrieval/semantic_scholar.py`
- Modify: `src/paper_search/retrieval/__init__.py`
- Create: `src/paper_search/ranking/fusion.py`
- Modify: `src/paper_search/ranking/__init__.py`
- Create: `tests/unit/test_semantic_scholar.py`
- Create: `tests/unit/test_fusion.py`
- Create: `tests/fixtures/semantic_scholar/search.json`
- Create: `tests/fixtures/semantic_scholar/batch.json`
- Create: `tests/fixtures/semantic_scholar/references.json`
- Create: `tests/fixtures/semantic_scholar/citations.json`

**Interfaces:**
- Consumes: the existing `Paper`, `ProviderPaperId`, `CitationEdge`, `CitationExpansion`, `ProviderResult`, `BudgetReservation`, cache, and deduplicator.
- Produces: `SearchProvider` protocol, `SemanticScholarProvider.search/batch/references/citations`, `FusionMethod`, and `fuse_provider_results(...)`.

- [ ] **Step 1: Add failing Semantic Scholar contract tests and synthetic fixtures**

Use only project-original synthetic JSON. Test search, batch detail, references, citations, empty responses, missing optional fields, invalid records, 429, timeout, nonretryable provider errors, reservation call limits, stable Semantic Scholar identity, provenance, and safe errors.

- [ ] **Step 2: Verify provider tests are RED**

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_semantic_scholar.py -q
```

Expected: import failure because the provider does not exist.

- [ ] **Step 3: Implement the provider protocol and Semantic Scholar adapter**

Use an injected `httpx.AsyncClient`, fixed API host, redirects disabled, selected fields, bounded attempts within the supplied reservation, existing cache keys, safe response headers, and normalization into existing models. Reference/citation endpoints return raw provider edges only.

- [ ] **Step 4: Verify provider tests GREEN**

Run the command from Step 2. Expected: all Semantic Scholar tests pass.

- [ ] **Step 5: Add failing fusion tests**

Test RRF with fixed `k`, weighted reciprocal rank with explicit provider weights, deterministic ties, duplicate cross-source DOI merge, retained OpenAlex/Semantic Scholar IDs and sources, empty sources, and valid results surviving a sibling provider error.

- [ ] **Step 6: Verify fusion tests are RED**

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_fusion.py -q
```

Expected: import failure because fusion does not exist.

- [ ] **Step 7: Implement pure fusion and exports**

Normalize each source rank into a canonical score, merge duplicate papers with `deduplicate_papers`, accumulate source evidence, and sort by descending score then canonical ID. Reject unknown methods or invalid/non-normalized weights.

- [ ] **Step 8: Verify Task 6 GREEN and regressions**

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_semantic_scholar.py tests/unit/test_fusion.py tests/unit/test_openalex.py tests/unit/test_cache.py tests/unit/test_deduplicate.py -q
uv run --no-sync --no-env-file ruff check src/paper_search/retrieval src/paper_search/ranking tests/unit/test_semantic_scholar.py tests/unit/test_fusion.py
uv run --no-sync --no-env-file mypy src/paper_search/retrieval src/paper_search/ranking
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 9: Stage-check and commit Task 6**

Inspect staged names, stage only Task 6 files, inspect again, then commit:

```text
feat: add deterministic multi-source fusion
```

### Task 3: Atomic Budgeting, Recovery, and Minimal Orchestration

**Files:**
- Modify: `src/paper_search/control/budget.py`
- Modify: `src/paper_search/control/__init__.py`
- Create: `src/paper_search/storage/experiment.py`
- Modify: `src/paper_search/storage/__init__.py`
- Create: `src/paper_search/pipeline/__init__.py`
- Create: `src/paper_search/pipeline/orchestrator.py`
- Modify: `tests/unit/test_budget.py`
- Create: `tests/unit/test_experiment_storage.py`
- Create: `tests/integration/test_orchestrator.py`

**Interfaces:**
- Consumes: Task 5 analyzer, Task 6 `SearchProvider` and fusion, existing deduplication/filtering/lexical ranking, `SearchBudget`, `UsageEstimate`, and `UsageActual`.
- Produces: thread-safe `HardBudgetController` lifecycle/state methods, `ExperimentRecordStore`, `MinimalSearchResult`, and `MockSearchOrchestrator.run(...)`.

- [ ] **Step 1: Add failing enhanced budget tests**

Test atomic concurrent reservations with a barrier, API/LLM/token/cost limits, unknown cost separated from known totals, actual settlement, explicit release, deterministic expiry with an injected clock, soft stop, hard stop, JSON state export/import, and invalid/tampered state rejection.

- [ ] **Step 2: Verify enhanced budget tests are RED**

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_budget.py -q
```

Expected: new tests fail because locking, lifecycle, stop state, and recovery methods do not exist.

- [ ] **Step 3: Implement minimal thread-safe budget lifecycle**

Guard every read-modify-write transition with `threading.RLock`; purge expired reservations before capacity checks; add `release`, `expire_reservations`, `stop_status`, `export_state`, and `from_state`; preserve `None` whenever any included cost is unknown and expose known cost separately.

- [ ] **Step 4: Verify budget GREEN**

Run the command from Step 2. Expected: all existing and new budget tests pass.

- [ ] **Step 5: Add failing experiment storage and orchestrator tests**

Use fake analyzer/providers and a deterministic clock. Assert exact order: analyze → reserve/call/settle providers → deduplicate → filter → fuse/lexical → result. Cover provider failure, empty result, insufficient budget before any call, soft stop, hard stop, reservation settlement, structured traces, `ProviderResult`, config hash, prompt version, and stop reason.

- [ ] **Step 6: Verify orchestration tests are RED**

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_experiment_storage.py tests/integration/test_orchestrator.py -q
```

Expected: import failures because storage and orchestrator do not exist.

- [ ] **Step 7: Implement experiment storage and orchestrator**

Persist one canonical UTF-8 JSON record via temporary-file replacement. The orchestrator owns every reservation, passes it into the dependency, settles returned usage, accumulates errors/warnings, reuses existing processing functions, and returns a minimal Pydantic result without Task 8 HTTP or UI fields.

- [ ] **Step 8: Verify Task 7 GREEN and regressions**

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_budget.py tests/unit/test_experiment_storage.py tests/integration/test_orchestrator.py tests/unit/test_openalex.py tests/unit/test_deduplicate.py tests/unit/test_filter.py tests/unit/test_lexical.py -q
uv run --no-sync --no-env-file ruff check src/paper_search/control src/paper_search/storage src/paper_search/pipeline tests/unit/test_budget.py tests/unit/test_experiment_storage.py tests/integration/test_orchestrator.py
uv run --no-sync --no-env-file mypy src/paper_search/control src/paper_search/storage src/paper_search/pipeline
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 9: Stage-check and commit Task 7**

Inspect staged names, stage only Task 7 files, inspect again, then commit:

```text
feat: add budgeted mock orchestration
```

### Task 4: Joint Verification, Scope Audit, and Independent Review

**Files:**
- Modify only files required to resolve verified review findings.

**Interfaces:**
- Consumes: all Task 5–7 contracts and commits.
- Produces: fresh verification evidence and a reviewed branch with no valid Critical, Important, or Minor findings outstanding.

- [ ] **Step 1: Run joint focused tests**

```powershell
uv run --no-sync --no-env-file pytest tests/unit/test_llm_client.py tests/unit/test_query_parser.py tests/unit/test_query_planner.py tests/unit/test_semantic_scholar.py tests/unit/test_fusion.py tests/unit/test_budget.py tests/unit/test_experiment_storage.py tests/integration/test_orchestrator.py tests/unit/test_openalex.py tests/unit/test_cache.py tests/unit/test_deduplicate.py tests/unit/test_filter.py tests/unit/test_lexical.py -q
```

- [ ] **Step 2: Run full offline verification**

```powershell
uv run --no-sync --no-env-file pytest -q
uv run --no-sync --no-env-file ruff check .
uv run --no-sync --no-env-file mypy src
git diff --check
```

- [ ] **Step 3: Audit scope and secrets without reading secret files**

Inspect `git diff --name-only 92d804f...HEAD`, `git status --short`, and `git diff --cached --name-only`. Verify no protected data/document path changed. Search only the committed Task 5–7 diff for credential-shaped literals and authorization header leakage; do not search or open `.env`.

- [ ] **Step 4: Request independent code review**

Give the reviewer the exact base SHA, head SHA, design, plan, constraints, and test evidence. Require findings categorized as Critical, Important, or Minor and reject scope-expanding suggestions.

- [ ] **Step 5: Fix every valid finding with TDD**

For each behavioral issue, add a failing regression test, verify RED, make the smallest fix, verify GREEN, and commit only the review fix files.

- [ ] **Step 6: Re-run complete verification**

Repeat Steps 1–3 and confirm the branch remains unpushed with `data/manifest.json` still at `waiting_for_human_label_freeze`.
