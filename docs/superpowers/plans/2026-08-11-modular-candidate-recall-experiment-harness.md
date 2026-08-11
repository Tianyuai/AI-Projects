# Modular Candidate Recall Experiment Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Scheme B as a modular candidate-pool recall experiment harness in which a normal search-term method is added by Prompt/YAML or a manual action file, three retrieval action families are independently replaceable, and historical replay plus repeated DeepSeek generation can determine whether the new harness reproduces prior candidate-recall behavior within the approved tolerance.

**Architecture:** A strict contract package separates frozen inputs, query generation, action validation/repair, retrieval handlers, replaceable backends, candidate-pool construction, optional post-pool stages, recall-only evaluation, artifact writing, and orchestration. Gold metadata is visible only while building an Oracle generation context; Gold identifiers and `IdentifierMap` enter only the evaluator. The runner iterates registered interfaces and contains no method-specific or action-specific branches.

**Tech Stack:** Python 3.11, Pydantic v2, PyYAML, existing Paper Search LLM/OpenAlex/Semantic Scholar snapshot adapters, `HardBudgetController`, `deduplicate_papers`, pytest, Ruff, mypy strict.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-11-modular-candidate-recall-experiment-harness-design.md` exactly.
- Work in the existing linked worktree `D:\AI Projects\.worktrees\week3` on `codex/query-evolution-gate-contracts`; do not create another worktree.
- Preserve untracked user-owned `data/budget_ledger.sqlite3`, `deliverables/`, and `docs/evidence/identifier-map-semantic-audit-2026-08-10.json`.
- Phase 1 evaluates only the complete, deduplicated candidate pool. Do not add filtering, ranking, RRF, round-robin, Top-50 truncation, Precision, F1, MRR, NDCG, promotion gates, or automatic method selection.
- Automated implementation and verification are offline. Do not read `.env`, use real network, spend budget, mutate the ledger, rebuild frozen data, or rewrite historical runs/evidence.
- Oracle generation requires a hash-bound Gold document catalog with title, abstract, authors, and publication year. It must not expose DOI, OpenAlex ID, Semantic Scholar ID, canonical ID, URLs, or provider request identifiers to DeepSeek.
- Blind generation must remove the `gold_documents` field entirely, not serialize it as an empty or null field.
- The initial state is `observable_state=None`. Do not implement result-driven query adjustment in Phase 1; retain the state contract and enum only.
- Phase-1 citation expansion accepts only pre-frozen, non-Gold `seed_candidates`; it must not dynamically search, choose a new seed, and expand it in one run.
- Candidate construction must call `deduplicate_papers(papers, id_map=None)` and must not import or receive Gold or an evaluation identifier map.
- Only the evaluator may adapt `paper_search.evaluation.dataset.IdentifierMap` to the recall experiment's identifier resolver protocol.
- Existing probe scripts and formal-run code remain unchanged. Reuse production modules through adapters; do not import `scripts/*.py` as libraries.
- Exact historical replay is a prerequisite for comparing regenerated Scheme B output. If a historical method lacks provable actions/responses/denominators, report `insufficient_historical_evidence`; never synthesize missing per-query evidence.
- A live regenerated comparison is a separately authorized step after all offline tasks pass. Three valid repeats may be scheduled within at most five attempts; infrastructure failures are excluded from pass/fail.
- Use TDD and commit after each task. Each task must leave focused tests green before the next task begins.

---

## Target File Map

```text
src/paper_search/recall_experiments/
  __init__.py
  contracts.py
  recipes.py
  validation.py
  inputs/
    __init__.py
    base.py
    formal_run.py
    historical.py
  generation/
    __init__.py
    base.py
    fixed.py
    manual.py
    deepseek.py
  retrieval/
    __init__.py
    backends.py
    registry.py
    text_search.py
    title_search.py
    citation_expand.py
  candidate_pool.py
  stages.py
  evaluator.py
  artifacts.py
  runner.py
  composition.py

configs/recall_experiments/
  samples/dev-smoke-3.yaml
  methods/manual-oracle-smoke.yaml
  methods/scheme-b-oracle.yaml
  methods/scheme-b-blind.yaml
  historical/query-rewrite.yaml
  historical/llm-query-variants.yaml
  historical/query-evolution.yaml
  historical/title-candidates.yaml
  historical/citation-expansion.yaml

configs/prompts/recall/
  scheme-b-oracle.yaml
  scheme-b-blind.yaml

tests/recall_experiments/
  test_contracts.py
  test_recipes.py
  test_inputs.py
  test_generation.py
  test_retrieval_registry.py
  test_text_search.py
  test_title_search.py
  test_citation_expand.py
  test_candidate_pool.py
  test_evaluator.py
  test_artifacts.py
  test_runner.py
  test_historical_replay.py
  test_cli.py
```

Existing files modified only at integration points:

```text
src/paper_search/cli.py
tests/integration/test_smoke_cli.py
```

No method-specific standalone script is added. A future ordinary method adds only one method recipe and, for DeepSeek generation, one prompt artifact.

---

### Task 1: Establish strict recall contracts and action validation

**Files:**
- Create: `src/paper_search/recall_experiments/__init__.py`
- Create: `src/paper_search/recall_experiments/contracts.py`
- Create: `src/paper_search/recall_experiments/validation.py`
- Create: `tests/recall_experiments/test_contracts.py`

**Interfaces:**
- `GoldVisibility = Literal["oracle", "blind", "historical"]`
- `ObservableSearchState = Literal["zero_results", "low_yield", "broad_noisy", "facet_gap", "duplicate_saturation", "entity_ambiguity", "provider_failure", "adequate"]`
- `GoldDocument(title, abstract, authors, publication_year)` with no identifier fields
- `SeedCandidate(paper: Paper)` validated as a real provider-normalized candidate
- `RecallGenerationContext(query_id, original_query, query_spec, seed_queries, seed_candidates, observable_state, gold_documents)`
- Discriminated actions `TextSearchAction`, `TitleSearchAction`, and `CitationExpandAction`
- `RecallActionBatch(actions)`
- `RetrievalActionResult(action_id, action_type, hits, usage, provenance, errors, infrastructure_failure)`
- `CandidatePool`, `CandidatePoolEntry`, `CandidateSourceEvidence`
- `validate_action_batch(raw, context, allowed_actions, max_actions) -> RecallActionBatch`

- [ ] **Step 1: Write RED contract tests**

Cover strict extra-field rejection, discriminated payload validation, deterministic action order, duplicate action IDs, whitespace/NFKC duplicate search text, empty and over-300-character search text, illegal action type, year conflicts with `QuerySpec`, missing payload fields, action-count limits, and citation direction restricted to `references | citations | both`.

Add privacy boundary tests:

```python
oracle = oracle_context_fixture()
assert "doi" not in oracle.model_dump_json().casefold()
assert "openalex" not in oracle.model_dump_json().casefold()

blind_payload = generation_payload(oracle, visibility="blind")
assert "gold_documents" not in blind_payload
```

Add seed integrity tests requiring every `CitationExpandPayload.seed_canonical_id` to be present in `context.seed_candidates`. Resolved Gold-versus-seed isolation is tested later at the evaluator preflight boundary, because action validation does not receive Gold IDs or an identifier map.

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_contracts.py -q
```

Expected: collection/import failure because the new package does not exist.

- [ ] **Step 2: Implement the minimum closed models**

Use `DomainModel` so models are frozen and `extra="forbid"`. Use a Pydantic discriminated union rather than a single payload with optional fields:

```python
class TextSearchPayload(DomainModel):
    query_text: NonEmptyStr


class TextSearchAction(ActionBase):
    action_type: Literal["text_search"]
    payload: TextSearchPayload


RecallSearchAction = Annotated[
    TextSearchAction | TitleSearchAction | CitationExpandAction,
    Field(discriminator="action_type"),
]
```

Keep action strategy as an open non-empty string so new prompt strategies do not require Python changes. Keep action types closed to the three implemented handlers for Phase 1.

- [ ] **Step 3: Implement mechanical validation and structured errors**

Define `ActionValidationIssue(code, field_path, message)` and `ActionValidationFailure(issues, previous_output, allowed_change_scope)`. The allowed error codes are exactly:

```text
invalid_json
duplicate_action
empty_action
action_too_long
disallowed_action_type
year_conflict
missing_required_field
unknown_seed_candidate
action_limit_exceeded
```

Validation performs canonical whitespace/NFKC normalization before duplicate checks, validates explicit years against `QuerySpec.year_from/year_to`, and never uses Gold hits or candidate recall.

- [ ] **Step 4: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_contracts.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/contracts.py src/paper_search/recall_experiments/validation.py tests/recall_experiments/test_contracts.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/contracts.py src/paper_search/recall_experiments/validation.py
git add -- src/paper_search/recall_experiments/__init__.py src/paper_search/recall_experiments/contracts.py src/paper_search/recall_experiments/validation.py tests/recall_experiments/test_contracts.py
git diff --cached --check
git commit -m "feat: add candidate recall contracts"
```

Expected: all commands exit `0`.

---

### Task 2: Add strict recipes and declarative method configuration

**Files:**
- Create: `src/paper_search/recall_experiments/recipes.py`
- Create: `tests/recall_experiments/test_recipes.py`
- Create: `configs/recall_experiments/methods/manual-oracle-smoke.yaml`

**Interfaces:**
- `RecallMethodRecipe`
- `GeneratorRecipe` discriminated as `manual_actions | fixed_actions | deepseek_prompt`
- `RetrievalRecipe`
- `EvaluationRecipe`
- `SampleBinding`
- `load_recall_recipe(path) -> LoadedRecallRecipe`
- `load_sample_binding(path) -> LoadedSampleBinding`

- [ ] **Step 1: Write RED loader and cross-field tests**

Require strict unknown-field rejection, safe relative paths, hashable canonical serialization, and these cross-field rules:

- `blind` forbids Oracle sample IDs and removes Gold documents;
- `oracle` requires a Gold-document catalog binding;
- `fixed_actions` and `manual_actions` require an actions artifact;
- `deepseek_prompt` requires model, prompt, temperature `0`, and `repair_attempts=1`;
- `allowed_actions` is non-empty and limited to registered Phase-1 action types;
- `repeat_count=3`, `max_repeat_attempts=5`, `required_passing_repeats=2` for the Scheme B comparison recipe;
- `gold_count_tolerance=1`, `macro_recall_tolerance=0.02`, and `retained_gold_min=0.90` are finite and fixed for historical comparison;
- live backends require an explicit runtime authorization flag and cannot be authorized by YAML alone.

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_recipes.py -q
```

Expected: import failure because `recipes.py` does not exist.

- [ ] **Step 2: Implement strict YAML loading and recipe locking**

Parse with `yaml.safe_load` and validate with Pydantic. `load_recall_recipe()` binds the exact recipe and prompt bytes; command-specific preflight separately binds the sample, manual/fixed actions, historical baseline, and snapshot manifests before it creates `recipe.lock.yaml`. This allows `prepare-context` to run before a user has pasted a manual action file, while `validate-actions` and `run` fail closed if that artifact is absent or changes. No loader resolves arbitrary Python import paths.

The initial manual recipe must contain only declarative values:

```yaml
method_id: manual-oracle-smoke
generator:
  type: manual_actions
  actions: runs/recall_inputs/manual-oracle-smoke/actions.json
  gold_visibility: oracle
retrieval:
  allowed_actions: [text_search, title_search, citation_expand]
  backend: snapshot_replay
  max_results_per_action: 50
  max_total_actions: 3
evaluation:
  repeat_count: 1
  max_repeat_attempts: 1
```

- [ ] **Step 3: Prove a new ordinary method needs no runner edit**

Add a test that creates a temporary `method_id`, prompt path, and action file, loads it, and later can be executed through the generic runner fixture without any method-ID registry. The only registered implementation key is generator `type`; `method_id` is data.

- [ ] **Step 4: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_recipes.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/recipes.py tests/recall_experiments/test_recipes.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/recipes.py
git add -- src/paper_search/recall_experiments/recipes.py tests/recall_experiments/test_recipes.py configs/recall_experiments/methods/manual-oracle-smoke.yaml
git diff --cached --check
```

Expected: all commands exit `0`. Then commit:

```powershell
git commit -m "feat: add declarative recall recipes"
```

---

### Task 3: Implement replaceable frozen inputs and Gold-document preflight

**Files:**
- Create: `src/paper_search/recall_experiments/inputs/__init__.py`
- Create: `src/paper_search/recall_experiments/inputs/base.py`
- Create: `src/paper_search/recall_experiments/inputs/formal_run.py`
- Create: `tests/recall_experiments/test_inputs.py`
- Create: `configs/recall_experiments/samples/dev-smoke-3.yaml`

**Interfaces:**
- `FrozenInputSource.load_queries(sample_binding) -> FrozenRecallDataset`
- `FrozenInputSource.load_historical_baseline(binding) -> HistoricalRecallBaseline | None`
- `FormalRunInputSource`
- `FrozenRecallDataset(queries, source_hashes, evaluation_materials)`
- `OpaqueEvaluationMaterials(gold_records, identifier_map_bytes, identifier_map_sha256)`; only the evaluator may parse or resolve it
- `OracleCatalogStatus = complete | incomplete | invalid`

- [ ] **Step 1: Write RED frozen-source tests**

Build temporary formal-run fixtures shaped like existing `business-results.jsonl`, `executions.jsonl`, `data/dev/gold.jsonl`, `data/identifier-map.json`, and a separate private Gold-document catalog. Require:

- query IDs are unique and retain source order;
- exact Gold records are well-formed; uniqueness after identifier resolution is deferred to evaluator preflight;
- all source hashes match the sample binding;
- a three-query sample resolves exactly the configured IDs, never “first three” implicitly;
- Oracle and Blind sample partitions do not overlap when both are declared;
- the historical comparison query IDs and Gold denominator match exactly;
- the input source verifies identifier-map bytes/hash but does not instantiate `IdentifierMap`; the bytes remain opaque outside evaluator calls and are never serialized into generation context;
- `seed_candidates` are provider-normalized `Paper` records and preserve frozen order; evaluator preflight rejects any seed that resolves to a Gold ID before generation context construction;
- Oracle fails before generation if any selected query lacks title, abstract, authors, or publication year;
- Blind loads and evaluates Gold IDs but never loads Gold-document contents.

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_inputs.py -q
```

Expected: import failure because the input adapter does not exist.

- [ ] **Step 2: Implement the protocol and formal-run adapter**

Reuse `read_jsonl`, `EvaluationQuery`, `EvaluationExecutionRecord`, `BusinessResultRecord`, and the existing snapshot-backed normalized `Paper` models. Verify the exact identifier-map bytes and hash without parsing them here. Do not call private `_load_formal_inputs()` and do not import `scripts/probe_query_evolution.py`.

Treat the Gold-document catalog as a separately replaceable, hash-bound input. The sample binding stores `gold_document_catalog.path=data/recall_freeze/dev-gold-documents.jsonl` and a `gold_document_catalog.sha256` value validated by the existing `Sha256` type and recomputed from the exact catalog bytes.

The adapter validates it but does not create it. If the current workspace lacks this catalog, the offline implementation still proceeds, but Oracle CLI preflight must return `oracle_catalog_incomplete`; no DeepSeek request is allowed.

- [ ] **Step 3: Bind the deterministic three-query smoke sample**

Select three explicit dev query IDs representing:

1. at least one historical candidate Gold hit;
2. at least one Gold association previously not retrieved;
3. a different Gold-count stratum so the quick test is not three copies of one denominator shape.

This general smoke sample is sufficient for text/title/manual workflow tests. Citation integration uses the exact non-Gold seed candidates bound by the historical Citation Expansion adapter in Task 11. If that evidence lacks a legitimate frozen seed, Citation Expansion is reported `insufficient_historical_evidence` and remains covered by synthetic handler/backend tests; do not substitute a Gold paper, live lookup, or dynamically retrieved text/title result.

The selection command is read-only and writes a reviewed YAML binding; do not choose queries based on future Scheme B results. Record exact source/gold/id-map/catalog hashes and the three IDs in `dev-smoke-3.yaml`.

- [ ] **Step 4: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_inputs.py tests/evaluation/test_dataset.py tests/evaluation/test_execution_adapter.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/inputs tests/recall_experiments/test_inputs.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/inputs
git add -- src/paper_search/recall_experiments/inputs tests/recall_experiments/test_inputs.py configs/recall_experiments/samples/dev-smoke-3.yaml
git diff --cached --check
```

Expected: offline tests pass. Then commit:

```powershell
git commit -m "feat: add frozen recall input adapter"
```

If the real Gold-document catalog is absent, the real Oracle preflight is expected to report `oracle_catalog_incomplete`; that is a data prerequisite, not a test failure.

---

### Task 4: Add replaceable search backends and an explicit action registry

**Files:**
- Create: `src/paper_search/recall_experiments/retrieval/__init__.py`
- Create: `src/paper_search/recall_experiments/retrieval/backends.py`
- Create: `src/paper_search/recall_experiments/retrieval/registry.py`
- Create: `tests/recall_experiments/test_retrieval_registry.py`

**Interfaces:**
- `SearchBackend.search(action_id, query, filters, limit) -> BackendSearchResult`
- `CitationBackend.expand(action_id, seed, direction, limit) -> BackendCitationResult`
- `RetrievalActionHandler.execute(action, context) -> RetrievalActionResult`
- `RetrievalActionRegistry.register/unregister/resolve`
- `BudgetedSearchBackend`
- `BudgetedCitationBackend`

- [ ] **Step 1: Write RED registry and backend tests**

Require explicit construction, duplicate registration rejection, unknown-action failure, unregister behavior, stable iteration order, and no global mutable registry or import-time registration. Fake backends must prove handlers can execute without importing OpenAlex, Semantic Scholar, snapshots, or local files.

Backend tests must prove one reservation per logical call, correct settle/fail/release behavior, structured infrastructure failure classification, and exact propagation of normalized `Paper`, usage, provenance, and error codes.

Run `python -m pytest tests/recall_experiments/test_retrieval_registry.py -q`. Expected: import failure for the missing registry/backend modules.

- [ ] **Step 2: Implement protocol adapters around existing providers**

`BudgetedSearchBackend` wraps a supplied `SearchProvider`—which may be `LiveCaptureSearchProvider`, `ReplaySearchProvider`, or a future local-index adapter—and owns reservations. `BudgetedCitationBackend` wraps a supplied Semantic Scholar `SearchProvider` plus `ProviderCitationExpansionStage` for `both`.

Do not construct clients in these classes. Snapshot/live/local backend selection belongs to `composition.py`.

- [ ] **Step 3: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_retrieval_registry.py tests/unit/test_retrieval_snapshot_adapters.py tests/unit/test_citation_expand.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/retrieval/backends.py src/paper_search/recall_experiments/retrieval/registry.py tests/recall_experiments/test_retrieval_registry.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/retrieval/backends.py src/paper_search/recall_experiments/retrieval/registry.py
git add -- src/paper_search/recall_experiments/retrieval/__init__.py src/paper_search/recall_experiments/retrieval/backends.py src/paper_search/recall_experiments/retrieval/registry.py tests/recall_experiments/test_retrieval_registry.py
git diff --cached --check
```

Expected: all commands exit `0`. Then commit:

```powershell
git commit -m "feat: add recall backend registry"
```

---

### Task 5: Implement and isolate the three retrieval handlers

**Files:**
- Create: `src/paper_search/recall_experiments/retrieval/text_search.py`
- Create: `src/paper_search/recall_experiments/retrieval/title_search.py`
- Create: `src/paper_search/recall_experiments/retrieval/citation_expand.py`
- Create: `tests/recall_experiments/test_text_search.py`
- Create: `tests/recall_experiments/test_title_search.py`
- Create: `tests/recall_experiments/test_citation_expand.py`

- [ ] **Step 1: Write RED text-search tests**

Assert that `TextSearchHandler` passes normalized `query_text`, inherited provider filters, and the recipe limit to `SearchBackend`; preserves raw provider rank; emits one unified action result; and never calculates recall, filters candidates, or calls another handler.

Run `python -m pytest tests/recall_experiments/test_text_search.py -q`. Expected: import failure for the missing text handler.

- [ ] **Step 2: Implement text search and verify independently**

The handler should be a thin adapter:

```python
result = await backend.search(
    action_id=action.action_id,
    query=action.payload.query_text,
    filters=context.provider_filters,
    limit=context.max_results_per_action,
)
return action_result(action, result)
```

Run only the text handler test and expect it to pass before continuing:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_text_search.py -q
```

- [ ] **Step 3: Write RED title-search tests**

Assert that `TitleSearchHandler` passes the complete title as a normal provider search query, uses the same replaceable backend, preserves provider rank, and reuses `extract_title_candidates` normalization where applicable. Tests and docstrings must not claim an exact-title endpoint or exact-title match—the existing `LLMTitleCandidateStage` calls `provider.search(title, {}, limit, reservation)`.

Run `python -m pytest tests/recall_experiments/test_title_search.py -q`. Expected: import failure for the missing title handler.

- [ ] **Step 4: Implement title search and verify independently**

Keep title-specific validation in this module; do not import `text_search.py`. Verify it before starting citation work:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_title_search.py tests/unit/test_title_candidates.py -q
```

- [ ] **Step 5: Write RED citation tests**

Cover all directions, missing Semantic Scholar ID, unknown/non-frozen seed, limit propagation, edge/paper preservation, partial provider errors, and infrastructure failures. Prove that:

- `references` calls only `provider.references` through `CitationBackend`;
- `citations` calls only `provider.citations`;
- `both` may delegate to existing `ProviderCitationExpansionStage`;
- the handler never selects a new seed from text/title results.

Run `python -m pytest tests/recall_experiments/test_citation_expand.py -q`. Expected: import failure for the missing citation handler.

- [ ] **Step 6: Implement citation expansion and aggregate verification**

Assert with an import-boundary test that the three handler modules do not import each other. Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_text_search.py tests/recall_experiments/test_title_search.py tests/recall_experiments/test_citation_expand.py tests/recall_experiments/test_retrieval_registry.py tests/unit/test_title_candidates.py tests/unit/test_citation_expand.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/retrieval tests/recall_experiments/test_text_search.py tests/recall_experiments/test_title_search.py tests/recall_experiments/test_citation_expand.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/retrieval
git add -- src/paper_search/recall_experiments/retrieval/text_search.py src/paper_search/recall_experiments/retrieval/title_search.py src/paper_search/recall_experiments/retrieval/citation_expand.py tests/recall_experiments/test_text_search.py tests/recall_experiments/test_title_search.py tests/recall_experiments/test_citation_expand.py
git diff --cached --check
```

Expected: all commands exit `0`. Then commit:

```powershell
git commit -m "feat: add modular recall handlers"
```

---

### Task 6: Build the raw candidate pool and empty future-stage pipeline

**Files:**
- Create: `src/paper_search/recall_experiments/candidate_pool.py`
- Create: `src/paper_search/recall_experiments/stages.py`
- Create: `tests/recall_experiments/test_candidate_pool.py`

**Interfaces:**
- `CandidatePoolBuilder.build(query_id, action_results) -> CandidatePool`
- `CandidateStage.apply(pool, context) -> StageResult`
- `CandidateStagePipeline.apply(pool, context) -> CandidatePool`

- [ ] **Step 1: Write RED pool tests**

Cover stable action/result/rank order, DOI/external-ID/exact-title/fuzzy-title duplicate merging through existing `deduplicate_papers`, preservation of every source evidence record, empty results, partial errors, and identical output when result order is repeated.

Install a Gold trap and an `IdentifierMap` trap that raise if accessed. The builder must remain green because its callable interface accepts neither object.

Run `python -m pytest tests/recall_experiments/test_candidate_pool.py -q`. Expected: import failure for the missing pool/stage modules.

- [ ] **Step 2: Implement the pool builder**

Flatten `Paper` objects in registered action order, call:

```python
deduplicated = deduplicate_papers(flattened, id_map=None)
```

Then attach all source evidence by duplicate-cluster membership. Do not apply `QuerySpec` hard filters, ranking, or limits after collection.

- [ ] **Step 3: Add the stage seam without Phase-1 behavior**

`CandidateStagePipeline(stages=())` returns the exact pool object. Add tests proving any non-empty stage list must be supplied explicitly and that stages cannot mutate generation or retrieval artifacts. Do not implement a concrete filter/ranker.

- [ ] **Step 4: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_candidate_pool.py tests/unit/test_deduplicate.py tests/unit/test_normalize.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/candidate_pool.py src/paper_search/recall_experiments/stages.py tests/recall_experiments/test_candidate_pool.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/candidate_pool.py src/paper_search/recall_experiments/stages.py
git add -- src/paper_search/recall_experiments/candidate_pool.py src/paper_search/recall_experiments/stages.py tests/recall_experiments/test_candidate_pool.py
git diff --cached --check
```

Expected: all commands exit `0`. Then commit:

```powershell
git commit -m "feat: build raw recall candidate pools"
```

---

### Task 7: Implement recall-only evaluation and historical tolerance logic

**Files:**
- Create: `src/paper_search/recall_experiments/evaluator.py`
- Create: `tests/recall_experiments/test_evaluator.py`

**Interfaces:**
- `CandidateRecallEvaluator.evaluate(dataset, pools) -> RecallRepeatResult`
- `CandidateRecallEvaluator.preflight(dataset) -> PreparedEvaluationContext`
- `IdentifierResolver.resolve(value) -> str`
- `compare_exact_replay(current, historical) -> HistoricalReplayComparison`
- `compare_regenerated(repeats, historical, policy) -> RegeneratedComparison`
- terminal conclusions `passed | failed | insufficient_valid_repeats | insufficient_historical_evidence`

- [ ] **Step 1: Write RED metric tests**

Use synthetic alias mappings to prove:

- per-query Gold count and hit count use unique resolved associations;
- candidate recall is `hits / gold`, with frozen data forbidding zero-Gold queries;
- macro candidate recall is the mean of per-query candidate recall;
- aggregate Gold association count is the sum of per-query unique hits;
- historical Gold retention is the association-level intersection, not candidate-set overlap;
- no candidate precision, Top-K, ranking, F1, MRR, or NDCG field exists in the public report model.

`preflight()` is the only place that calls `IdentifierMap.from_bytes`. It rejects duplicate resolved Gold associations and any `seed_candidate` that resolves to a Gold ID, then returns separate generation-safe query contexts and private scoring data. The runner passes only generation-safe contexts to generators/handlers/pool construction.

Do not call `paper_search.evaluation.metrics.evaluate()` because it computes out-of-scope ranking and precision metrics. Reuse only the identifier normalization/resolution contract.

Run `python -m pytest tests/recall_experiments/test_evaluator.py -q`. Expected: import failure for the missing evaluator.

- [ ] **Step 2: Implement exact historical replay comparison**

If per-query historical candidate sets exist, require identical normalized candidate-ID sets, per-query hits, total hits, and macro recall. If only aggregate historical evidence exists, compare only the explicit aggregate fields and set `per_query_comparison="not_provable"`.

- [ ] **Step 3: Implement regenerated-repeat policy**

For each valid repeat, pass only when all are true:

```python
abs(current.gold_association_count - historical.gold_association_count) <= 1
abs(current.macro_candidate_recall - historical.macro_candidate_recall) <= 0.02
current.historical_gold_retention >= 0.90
```

Exclude `infrastructure_failure` attempts. Require three valid repeats within five scheduled attempts and at least two passing repeats. Report min/median/max for hits and macro recall. If fewer than three valid repeats exist after five attempts, return `insufficient_valid_repeats` without a sixth attempt.

- [ ] **Step 4: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_evaluator.py tests/evaluation/test_dataset.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/evaluator.py tests/recall_experiments/test_evaluator.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/evaluator.py
git add -- src/paper_search/recall_experiments/evaluator.py tests/recall_experiments/test_evaluator.py
git diff --cached --check
```

Expected: all commands exit `0`. Then commit:

```powershell
git commit -m "feat: evaluate candidate recall compatibility"
```

---

### Task 8: Add fixed/manual generation, immutable artifacts, and the generic runner

**Files:**
- Create: `src/paper_search/recall_experiments/generation/__init__.py`
- Create: `src/paper_search/recall_experiments/generation/base.py`
- Create: `src/paper_search/recall_experiments/generation/fixed.py`
- Create: `src/paper_search/recall_experiments/generation/manual.py`
- Create: `src/paper_search/recall_experiments/artifacts.py`
- Create: `src/paper_search/recall_experiments/runner.py`
- Create: `tests/recall_experiments/test_generation.py`
- Create: `tests/recall_experiments/test_artifacts.py`
- Create: `tests/recall_experiments/test_runner.py`

**Interfaces:**
- `QueryGenerator.generate(context) -> GenerationResult`
- `FixedActionGenerator`
- `ManualActionGenerator`
- `RecallArtifactWriter`
- `RecallExperimentRunner.run(request) -> RecallExperimentResult`

- [ ] **Step 1: Write RED fixed/manual generator tests**

Require exact query coverage, no unknown IDs, one immutable action batch per query/repeat, strict validation before retrieval, and byte-identical fixed replay. Manual generation reads a user-prepared JSON artifact; it performs no LLM call.

Run `python -m pytest tests/recall_experiments/test_generation.py -q`. Expected: import failure for the missing generation modules.

- [ ] **Step 2: Implement generators and generation artifact boundary**

Both generators return the same `GenerationResult`. Write `generation/<repeat>/<query_id>.json` before calling any handler. Once written, the runner cannot replace the action batch based on retrieval results.

- [ ] **Step 3: Write RED artifact tests**

Require this exact layout:

```text
recipe.lock.yaml
sample-manifest.json
generation/<repeat>/<query_id>.json
retrieval/<repeat>/<query_id>.json
candidate-pools/<repeat>/<query_id>.json
recall-report.json
```

Require canonical UTF-8 JSON with sorted keys and one trailing newline; atomic writes; a new run directory; no overwrite; recipe/sample/input hashes; and sanitized errors with no secrets or authorization headers.

Run `python -m pytest tests/recall_experiments/test_artifacts.py -q`. Expected: import failure for the missing artifact writer.

- [ ] **Step 4: Write RED runner tests**

Use fake input source, generator, registry, handlers, stage pipeline, evaluator, and writer. Assert exact event order:

```python
assert events == [
    "load-and-verify-inputs",
    "evaluator-preflight-and-gold-seed-isolation",
    "build-generation-context",
    "generate-and-validate",
    "write-generation",
    "resolve-and-execute-actions",
    "write-retrieval",
    "build-candidate-pool",
    "apply-empty-stages",
    "write-candidate-pool",
    "evaluate-recall",
    "write-report",
]
```

Add a temporary method with an arbitrary `method_id` and existing `text_search` action; it must run without a method branch. Add an unknown handler test that fails before any retrieval artifact is written.

Run `python -m pytest tests/recall_experiments/test_runner.py -q`. Expected: import failure for the missing runner.

- [ ] **Step 5: Implement the branch-free runner**

The runner iterates actions and calls `registry.resolve(action.action_type)`. It contains no literals for Query Rewrite, Query Variants, Query Evolution, Title Candidates, Citation Expansion, method IDs, metric formulas, provider constructors, or generator constructors.

- [ ] **Step 6: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_contracts.py tests/recall_experiments/test_generation.py tests/recall_experiments/test_artifacts.py tests/recall_experiments/test_runner.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/generation src/paper_search/recall_experiments/artifacts.py src/paper_search/recall_experiments/runner.py tests/recall_experiments/test_generation.py tests/recall_experiments/test_artifacts.py tests/recall_experiments/test_runner.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/generation src/paper_search/recall_experiments/artifacts.py src/paper_search/recall_experiments/runner.py
git add -- src/paper_search/recall_experiments/generation/__init__.py src/paper_search/recall_experiments/generation/base.py src/paper_search/recall_experiments/generation/fixed.py src/paper_search/recall_experiments/generation/manual.py src/paper_search/recall_experiments/artifacts.py src/paper_search/recall_experiments/runner.py tests/recall_experiments/test_generation.py tests/recall_experiments/test_artifacts.py tests/recall_experiments/test_runner.py
git diff --cached --check
```

Expected: all commands exit `0`. Then commit:

```powershell
git commit -m "feat: orchestrate reusable recall experiments"
```

---

### Task 9: Add DeepSeek generation with Oracle/Blind isolation and one repair

**Files:**
- Create: `src/paper_search/recall_experiments/generation/deepseek.py`
- Modify: `tests/recall_experiments/test_generation.py`
- Create: `configs/prompts/recall/scheme-b-oracle.yaml`
- Create: `configs/prompts/recall/scheme-b-blind.yaml`
- Create: `configs/recall_experiments/methods/scheme-b-oracle.yaml`
- Create: `configs/recall_experiments/methods/scheme-b-blind.yaml`

**Interfaces:**
- `DeepSeekPromptGenerator`
- `RecallPromptArtifact`
- `render_recall_prompt()`
- `build_generation_payload(context, visibility)`
- `build_repair_payload(failure)`

- [ ] **Step 1: Write RED payload and privacy tests**

Use a recording fake analyzer. Oracle payload must include only query data, seed data, allowed action schema, and Gold title/abstract/authors/year. Recursively reject keys or values containing canonical identifiers, DOI, OpenAlex, Semantic Scholar, URLs, provider request IDs, prior Gold hits, or evaluation results.

Blind payload must not contain a `gold_documents` key at any nesting level. Historical mode must preserve the historical source's recorded visibility and may not silently upgrade Blind to Oracle.

Run `python -m pytest tests/recall_experiments/test_generation.py -q`. Expected: new DeepSeek-specific tests fail because `deepseek.py` is absent.

- [ ] **Step 2: Write RED one-repair tests**

Parameterize all structured validation failures from Task 1. The first invalid response triggers exactly one repair call containing error codes, field paths, previous output, and the allowed change scope. A second invalid response returns `generation_failure`. Provider/auth/rate-limit/network errors return `infrastructure_failure` and do not consume the semantic repair attempt.

- [ ] **Step 3: Implement through existing LLM adapters**

Inject an analyzer compatible with `LiveCaptureLLMAnalyzer` or `ReplayLLMAnalyzer`; do not instantiate `OpenAICompatibleLLMClient` here. Bind prompt bytes and SHA-256 into the existing snapshot identity. Keep temperature `0`, model `deepseek-v4-flash`, and `repair_attempts=1` locked by recipe.

- [ ] **Step 4: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_generation.py tests/unit/test_llm_snapshot_adapters.py tests/unit/test_prompt_artifacts.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/generation/deepseek.py tests/recall_experiments/test_generation.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/generation/deepseek.py
git add -- src/paper_search/recall_experiments/generation/deepseek.py tests/recall_experiments/test_generation.py configs/prompts/recall/scheme-b-oracle.yaml configs/prompts/recall/scheme-b-blind.yaml configs/recall_experiments/methods/scheme-b-oracle.yaml configs/recall_experiments/methods/scheme-b-blind.yaml
git diff --cached --check
```

Expected: all commands exit `0`. Then commit:

```powershell
git commit -m "feat: generate recall actions with deepseek"
```

---

### Task 10: Add composition and a three-step CLI for fast feasibility testing

**Files:**
- Create: `src/paper_search/recall_experiments/composition.py`
- Modify: `src/paper_search/cli.py`
- Create: `tests/recall_experiments/test_cli.py`
- Modify: `tests/integration/test_smoke_cli.py`

**CLI:**

```text
paper-search recall prepare-context
paper-search recall validate-actions
paper-search recall run
paper-search recall compare
paper-search recall inventory-history
```

- [ ] **Step 1: Write RED CLI and authorization tests**

Require:

- `prepare-context` verifies frozen inputs and writes sanitized per-query contexts without LLM or retrieval;
- `validate-actions` validates manually pasted DeepSeek output and writes generation artifacts without retrieval;
- `run` can consume validated manual/fixed actions or use the DeepSeek generator selected by recipe;
- replay is default; live generation/search requires `--allow-live` and an authorized recipe/backend combination;
- `prepare-context`, `validate-actions`, and replay never open `.env`, ledger, or network;
- Oracle catalog incompleteness fails before analyzer construction;
- output JSON reports safe terminal codes and paths, not raw provider errors or secrets.

Run `python -m pytest tests/recall_experiments/test_cli.py -q`. Expected: parser/composition tests fail because the recall commands are absent.

- [ ] **Step 2: Implement explicit composition registries**

Composition owns three local registries:

```python
generator_factories = {
    "manual_actions": build_manual_generator,
    "fixed_actions": build_fixed_generator,
    "deepseek_prompt": build_deepseek_generator,
}
handler_registry = RetrievalActionRegistry()
handler_registry.register("text_search", build_text_handler(runtime))
handler_registry.register("title_search", build_title_handler(runtime))
handler_registry.register("citation_expand", build_citation_handler(runtime))
```

Backend factories wrap the existing `LiveCaptureSearchProvider` / `ReplaySearchProvider` and Semantic Scholar citation provider. The runner receives only completed interfaces.

- [ ] **Step 3: Verify the intended fast workflow offline**

With the three-query binding:

```powershell
paper-search recall prepare-context --recipe configs/recall_experiments/methods/manual-oracle-smoke.yaml --sample configs/recall_experiments/samples/dev-smoke-3.yaml --out runs/_recall_prepare_manual_oracle
paper-search recall validate-actions --recipe configs/recall_experiments/methods/manual-oracle-smoke.yaml --contexts runs/_recall_prepare_manual_oracle --actions runs/recall_inputs/manual-oracle-smoke/actions.json --out runs/_recall_validate_manual_oracle
paper-search recall run --recipe configs/recall_experiments/methods/manual-oracle-smoke.yaml --sample configs/recall_experiments/samples/dev-smoke-3.yaml --out runs/_recall_smoke_manual_oracle
```

Expected workflow: the user can copy the prepared Oracle context to DeepSeek, paste returned actions into one JSON file, validate them, and test candidate recall without writing a new script or enabling an LLM client in the harness.

If required Gold metadata or replay responses are absent, stop with the corresponding preflight code; do not switch to live automatically.

- [ ] **Step 4: Verify and commit**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_cli.py tests/recall_experiments tests/integration/test_smoke_cli.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/composition.py src/paper_search/cli.py tests/recall_experiments/test_cli.py tests/integration/test_smoke_cli.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/composition.py src/paper_search/cli.py
git add -- src/paper_search/recall_experiments/composition.py src/paper_search/cli.py tests/recall_experiments/test_cli.py tests/integration/test_smoke_cli.py
git diff --cached --check
```

Expected: all commands exit `0`. Then commit:

```powershell
git commit -m "feat: expose candidate recall experiment cli"
```

---

### Task 11: Normalize and exactly replay the five historical candidate-pool methods

**Files:**
- Create: `src/paper_search/recall_experiments/inputs/historical.py`
- Create: `tests/recall_experiments/test_historical_replay.py`
- Create: the five `configs/recall_experiments/historical/*.yaml` bindings listed in the file map

**Included methods:** Query Rewrite, LLM Query Variants, Query Evolution, Title Candidates, Citation Expansion.

**Excluded methods:** Embedding, RRF, round-robin, filters, identifier rescore, final ranking, and any method that changes only candidate ordering/selection.

- [ ] **Step 1: Inventory historical evidence read-only**

`inventory-history` inspects only explicitly configured paths and emits:

```text
method_id
source_run_id
source_hashes
query_ids_available
gold_denominator_available
actions_available
provider_responses_available
per_query_candidate_ids_available
aggregate_recall_available
evidence_level = exact | aggregate_only | insufficient
```

Do not scan arbitrary run directories and guess identities. Each binding names exact immutable files and hashes.

- [ ] **Step 2: Write RED normalization tests for each source schema**

Each historical adapter must produce the same `FixedActionGenerator` input, `FrozenRecallDataset`, and optional `HistoricalRecallBaseline`. Source-specific parsing ends here; the generic runner and evaluator remain unchanged.

For Query Evolution, reuse public models/functions from `paper_search.evaluation.query_evolution_probe` and its sealed source records, not `scripts/probe_query_evolution.py`. For Title Candidates and Citation Expansion, reuse normalized `Paper`/snapshot records from their sealed run evidence. Query Rewrite and LLM Query Variants bindings must identify their exact action slots; if those slots were not preserved, mark them `insufficient`.

Run `python -m pytest tests/recall_experiments/test_historical_replay.py -q`. Expected: import failure for the missing historical adapter.

- [ ] **Step 3: Execute exact offline replay for every provable method**

For `evidence_level=exact`, require exact normalized candidate-ID sets, per-query hits, total Gold associations, and macro candidate recall. For `aggregate_only`, compare only the stored aggregate fields and label per-query equality `not_provable`. Any mismatch blocks Scheme B comparison for that method.

All five bindings must reach either:

- `exact_replay_passed`; or
- `insufficient_historical_evidence` with the precise missing artifact recorded.

No method may be silently omitted. Scheme B cannot receive an overall `passed` conclusion unless at least two historical methods are exactly replayable, including at least one text-search method and one non-text action family; otherwise the terminal conclusion is `insufficient_historical_evidence`.

- [ ] **Step 4: Verify and commit**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_historical_replay.py -q
paper-search recall inventory-history --config-root configs/recall_experiments/historical --out runs/_recall_history_inventory
paper-search recall compare --config-root configs/recall_experiments/historical --out runs/_recall_history_compare
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/recall_experiments/inputs/historical.py tests/recall_experiments/test_historical_replay.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src/paper_search/recall_experiments/inputs/historical.py
git diff -- docs/evidence runs
git add -- src/paper_search/recall_experiments/inputs/historical.py tests/recall_experiments/test_historical_replay.py configs/recall_experiments/historical
git diff --cached --check
```

Expected: tests and commands exit `0`; historical files are byte-identical; inventory/compare artifacts remain untracked diagnostic outputs. Then commit only adapters, tests, and bindings:

```powershell
git commit -m "feat: replay historical candidate recall methods"
```

---

### Task 12: Run the offline module/aggregate acceptance gate

**Files:**
- Modify after verified results: `HANDOFF.md`
- Modify after verified results: `docs/retrieval-roadmap.md`

- [ ] **Step 1: Run module tests separately**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_contracts.py tests/recall_experiments/test_recipes.py tests/recall_experiments/test_inputs.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_retrieval_registry.py tests/recall_experiments/test_text_search.py tests/recall_experiments/test_title_search.py tests/recall_experiments/test_citation_expand.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_candidate_pool.py tests/recall_experiments/test_evaluator.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments/test_generation.py tests/recall_experiments/test_artifacts.py tests/recall_experiments/test_runner.py tests/recall_experiments/test_cli.py -q
```

Expected: every module group exits `0`, making failures attributable to one boundary.

- [ ] **Step 2: Run aggregate offline tests**

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/recall_experiments -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -m "not online" -q
$trackedPython = @(git ls-files '*.py')
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check -- $trackedPython
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts
git diff --check
```

Expected: all commands exit `0`. Investigate any existing environment-specific failure before claiming full green.

- [ ] **Step 3: Check architecture and scope mechanically**

Use import/AST tests and repository search to confirm:

- handlers do not import one another;
- `candidate_pool.py` does not import `evaluator.py`, Gold models, or `IdentifierMap`;
- `runner.py` contains no method IDs, action-type branches, provider constructors, metric formulas, or historical constants;
- no Phase-1 report model contains precision/F1/MRR/NDCG/Top-K fields;
- no new standalone method script exists;
- old scripts, historical evidence, formal runs, ledger, and user-owned files have no diff.

- [ ] **Step 4: Commit verified handoff state**

Only after the offline gate passes, record implemented capabilities, exact historical replay coverage, any missing Oracle catalog/evidence, and the boundary that no live Scheme B repeat has run. Commit documentation separately:

```powershell
git commit -m "docs: prepare scheme b recall comparison"
```

---

### Task 13: Execute the separately authorized Scheme B regenerated comparison

**Preconditions:**

- Tasks 1–12 are green.
- Oracle Gold-document catalog is complete and hash-bound.
- The chosen historical method has `exact_replay_passed`.
- Same query IDs, Gold denominator, Gold visibility, prompt method, model, temperature, and search budget are locked.
- Network, DeepSeek/OpenAlex/Semantic Scholar use, budget, ledger path, and output root are explicitly authorized.

- [ ] **Step 1: Run a three-query live canary**

Run `prepare-context` first and inspect the sanitized Oracle payload. Then run one live repeat on `dev-smoke-3.yaml`. Verify generation JSON, repair count, retrieval provenance, candidate-pool contents, usage settlement, snapshot seal, and offline replay equality before expanding the sample.

The canary asks only whether the harness executes correctly; it does not decide Scheme B feasibility.

- [ ] **Step 2: Schedule up to five attempts to obtain three valid repeats**

For each repeat, freeze the generated actions before retrieval and write all six artifact groups. Do not adjust search terms after seeing results. Classify provider/auth/rate-limit/snapshot/ledger failures as `infrastructure_failure`; exclude them from pass/fail but count them toward the five scheduling attempts.

- [ ] **Step 3: Compare against every eligible historical method**

Use only matched query sets and denominators. A repeat passes a method when Gold count delta is at most `±1`, macro candidate recall delta at most `0.02`, and historical Gold retention at least `0.90`. Require at least two passing repeats among three valid repeats.

The final report must separate:

- harness exact-replay correctness;
- Scheme B regenerated compatibility per historical method;
- infrastructure failures;
- evidence-insufficient methods;
- Oracle-only limitation.

It must not claim that any historical method is good, that Oracle performance generalizes, or that filtering/ranking is solved.

- [ ] **Step 4: Stop at the candidate-recall decision**

If Scheme B passes, the next design task is a separate candidate filtering/ranking stage evaluation. If it fails, revise only the recipe/prompt or selected action handler/backend and rerun the three-query/manual workflow before another full comparison. Do not add a new monolithic script.

---

## Definition of Done

- [ ] A normal text-search method can be added by Prompt/YAML without creating or editing a standalone experiment script.
- [ ] Manual DeepSeek output can be validated and tested on three frozen queries before any in-harness LLM generation is enabled.
- [ ] Oracle sees complete Gold content but no Gold identifiers; Blind receives no Gold document field.
- [ ] Text, title, and citation handlers pass independent unit tests and can be registered/unregistered/replaced independently.
- [ ] Frozen input source and search backend can be replaced independently.
- [ ] Candidate pool is complete, deduplicated, provenance-preserving, unfiltered, unranked, and Gold-independent.
- [ ] The post-pool stage seam exists and is empty in Phase 1.
- [ ] Evaluator reports only candidate recall and historical compatibility fields.
- [ ] Fixed historical replay is exact wherever evidence supports exactness; missing evidence is explicit.
- [ ] DeepSeek generation uses one structured repair attempt and no result-driven prompt rewrite.
- [ ] Three valid regenerated repeats are obtained within at most five attempts, or the terminal result is `insufficient_valid_repeats`.
- [ ] Scheme B compatibility uses the approved `±1`, `0.02`, `0.90`, and `2/3` thresholds.
- [ ] All module, aggregate, lint, type, privacy, and architecture checks pass without modifying user-owned or historical artifacts.

## Final Self-Review Checklist

- [ ] Every requirement in the approved design maps to a task and test above.
- [ ] No placeholder, TODO, “implement later,” or undefined interface remains in the Phase-1 path.
- [ ] Generator, handler, backend, stage, evaluator, and input-source types are consistent at every boundary.
- [ ] Live authorization is runtime-only and cannot be enabled by recipe data.
- [ ] Gold metadata completeness is checked before DeepSeek construction; no identifier can enter its payload.
- [ ] Exact replay and regenerated comparison are distinct conclusions.
- [ ] Historical methods that only affect filtering/ranking are excluded, while all five candidate-pool methods are explicitly accounted for.
- [ ] The plan ends at candidate-pool recall; filtering/ranking starts only under a new approved design.
