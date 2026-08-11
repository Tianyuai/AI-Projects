# Identifier Relation-Correctness Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox notation for execution tracking.

**Goal:** Replace the false 100% provider-coverage requirement with a strict relation-correctness gate, publish one trustworthy v2 identifier-map generation from the already sealed evidence, and only then run the deferred offline semantic rescore.

**Architecture:** Keep capture, snapshots, ledger, and the semantic classifier unchanged. Extend the offline semantic models with exact v2 aggregate/private contracts; make the existing rebuild script consume every sealed evidence ref exactly once and publish no-replace artifacts under a public-audit-last transaction; add one reusable strict loader that Task 4 uses to verify all source and artifact bytes before parsing the map.

**Tech Stack:** Python 3.12, Pydantic domain models, pytest, Ruff, mypy, canonical JSON, SHA-256, existing dependency-snapshot reader.

**Global Constraints:**

- [ ] Use only `data/dev/gold.jsonl`, `data/annotation_work/identifier_semantics/identity-evidence.json`, and its bound sealed snapshot manifest; perform no readiness, capture, ledger, `.env`, OpenAlex, or Semantic Scholar operation.
- [ ] Preserve `HANDOFF.md`, `docs/retrieval-roadmap.md`, `data/budget_ledger.sqlite3`, `deliverables/`, and the existing failed public audit unless a later, separately authorized handoff update names them.
- [ ] Use fresh v2 output paths; never overwrite or auto-clean an output or publication lock.
- [ ] Stop for human intervention on input-integrity failure, decoder regression, real semantic mismatch/conflict, stale publication state, or any Task 4 integrity/retrieval/ranking/budget guard failure. Do not respond by repeating capture.
- [ ] Follow RED → GREEN → focused regression for every code task and commit only named files after verification.

### Task 1: Encode audit-v2 and relation-conservation contracts

**Files:**

- Modify: `src/paper_search/evaluation/identifier_semantics.py`
- Modify: `tests/evaluation/test_identifier_semantics.py`
- Modify: `scripts/rebuild_dev_identifier_map.py`
- Modify: `tests/scripts/test_rebuild_dev_identifier_map.py`

**Interfaces:**

- Advance `IdentifierMapSemanticAudit` to exact schema `identifier-map-semantic-audit-v2` with separate `input_hashes` and `artifact_hashes` plus anchor/provider coverage counts.
- Add an exact-schema private relation-audit envelope so private bytes can be validated and hashed as one generation artifact.
- Keep `classify_relation()` and snapshot decoding behavior unchanged.

- [ ] **Step 1: Add RED tests for the corrected candidate universe.**

  Cover: anchor-only groups pass without synthetic rows; each canonical evidence ref creates one candidate; duplicate, malformed, unsorted, outside-gold, unconsumed, or undecodable refs raise the same value-free input error; incomplete/mismatched candidates remain audited and fail promotion; every alias-conflict contributor remains present.

  Run:

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py -q
  ```

  Expected: new tests fail against the v1/synthetic-placeholder behavior.

- [ ] **Step 2: Implement the smallest pure rebuild change.**

  In `rebuild_dev_map()`:

  1. normalize unique gold arXiv IDs and build exactly one verified DataCite anchor each;
  2. require canonical, sorted, unique `(arxiv_id, alias)` evidence refs contained in gold;
  3. consume each ref exactly once through the existing sealed decoder and classify before filtering;
  4. remove `provider_identity_missing` relation synthesis;
  5. preserve all conflict contributors while changing their state/reason;
  6. construct candidate-map bytes only from verified anchors and aliases.

- [ ] **Step 3: Enforce v2 count/hash equations in model validators.**

  Serialize zeroes for every supported state, proof kind, and reason; require anchor/provider/relation conservation and `provider_identity_missing_group_count = gold_group_count - provider_identity_group_count`. Bind exactly the three source hashes and the canonical candidate-map/private-audit hashes. A passed status must imply all real relations are verified and every required anchor is verified.

- [ ] **Step 4: Add the fixed sealed-evidence regression.**

  With the three design-recorded hashes, assert `141` gold groups, `141` verified anchors, `90` provider groups, `51` missing-provider groups, `90` candidates, and `231` relations. If those hashes match but counts differ, raise decoder regression before any publication. Other hashes use only general equations.

- [ ] **Step 5: Run focused tests and commit.**

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py -q
  git add -- src/paper_search/evaluation/identifier_semantics.py tests/evaluation/test_identifier_semantics.py scripts/rebuild_dev_identifier_map.py tests/scripts/test_rebuild_dev_identifier_map.py
  git commit -m "fix: enforce identifier relation correctness"
  ```

  Expected: focused tests pass; no protected user-owned path is staged.

### Task 2: Make publication and consumption transactional

**Files:**

- Modify: `src/paper_search/evaluation/identifier_semantics.py`
- Modify: `tests/evaluation/test_identifier_semantics.py`
- Modify: `scripts/rebuild_dev_identifier_map.py`
- Modify: `tests/scripts/test_rebuild_dev_identifier_map.py`

**Interfaces:**

- Add `load_verified_identifier_map(...) -> IdentifierMap`, the sole Task 4 entry point for audit/map/private-audit verification.
- Replace overwriting writes with an exclusive sibling publication lock and atomic no-replace writes; public audit is the final generation marker.

- [ ] **Step 1: Add RED publication tests.**

  Prove pairwise-distinct normalized targets, absent formal targets, exclusive lock acquisition, no-replace races, and all four outcome paths. Input/decoder failures write nothing; semantic failure writes private audit then failed public marker and no map; success writes private audit and map, re-verifies both, then writes passed public marker last. Simulated interruption before the public marker must not create a recognized generation.

- [ ] **Step 2: Add RED loader tests.**

  Before reading `status`, reject duplicate public keys, wrong/extra/missing v2 fields, noncanonical bytes, and recursive privacy violations. Then reject non-passed audit or mismatched gold/evidence/manifest/private/map hash. Verify raw map hash before strict duplicate-key/exact-shape parsing and canonical-byte comparison.

- [ ] **Step 3: Implement lock, no-replace publisher, and strict loader.**

  Keep errors value-free. Release a normally completed lock only after the durable public marker; never auto-remove a stale lock or residual generation. Expose no network, prediction, query, validation, manual-alias, environment override, or bypass parameter.

- [ ] **Step 4: Run focused and adjacent regression suites.**

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py tests/scripts/test_capture_identifier_identity.py tests/storage/test_dependency_snapshot.py tests/storage/test_budget_ledger.py -q
  & 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src/paper_search/evaluation/identifier_semantics.py scripts/rebuild_dev_identifier_map.py tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py
  & 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src/paper_search/evaluation/identifier_semantics.py scripts/rebuild_dev_identifier_map.py
  ```

- [ ] **Step 5: Commit the publication boundary.**

  ```powershell
  git add -- src/paper_search/evaluation/identifier_semantics.py tests/evaluation/test_identifier_semantics.py scripts/rebuild_dev_identifier_map.py tests/scripts/test_rebuild_dev_identifier_map.py
  git commit -m "fix: publish verified identifier generations"
  ```

### Task 3: Rebuild one fresh v2 generation offline

**Fresh targets:**

- Private map: `data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v2.json`
- Private audit: `data/annotation_work/identifier_semantics/relation-audit.v2.json`
- Public marker: `docs/evidence/identifier-map-semantic-audit-2026-08-11.json`

- [ ] **Step 1: Confirm inputs are the approved sealed generation and all targets/lock are absent.**

  Compute the three source hashes and require the design-recorded values. If any differs, stop; do not substitute a recent file silently.

- [ ] **Step 2: Run the offline builder once.**

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/rebuild_dev_identifier_map.py --gold data/dev/gold.jsonl --evidence data/annotation_work/identifier_semantics/identity-evidence.json --snapshot-root data/annotation_work/identifier_semantics/snapshots --out-map data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v2.json --out-private-audit data/annotation_work/identifier_semantics/relation-audit.v2.json --out-public-audit docs/evidence/identifier-map-semantic-audit-2026-08-11.json
  ```

- [ ] **Step 3: Independently load the completed generation.**

  Require a canonical privacy-safe `passed` public marker, all five bound hashes, `141/141/90/51/90/231`, zero mismatch/unresolved/conflict relations, and strict canonical map bytes. If any check fails, stop and report the exact human-intervention category; do not start Task 4.

### Task 4: Implement and run the deferred sealed-run rescore

**Files:**

- Create: `scripts/rescore_identifier_semantics.py`
- Create: `tests/scripts/test_rescore_identifier_semantics.py`
- Modify only if a shared report model is needed: `src/paper_search/evaluation/identifier_semantics.py`, `tests/evaluation/test_identifier_semantics.py`
- Produce only after success: `docs/evidence/identifier-map-semantic-rescore-2026-08-11.json`, `docs/identifier-map-semantic-rescore-2026-08-11.md`

- [ ] **Step 1: Add RED tests for the Task 4 guard and source bindings.**

  Require `load_verified_identifier_map()` before opening run business data; reject failed/non-v2/tampered generations and cross-run stage inputs. Independently bind each formal run's `run.json`, `predictions.jsonl`, `executions.jsonl`, and `business-results.jsonl`; retain the existing exact-hash boundary for the legacy title run and the matched capture/replay boundary for the query-evolution probe.

- [ ] **Step 2: Implement the offline aggregate rescorer and privacy-safe publisher.**

  Recompute metrics and funnel stages from each source independently. Emit only aggregate JSON/Markdown; do not mutate runs, predictions, snapshots, locks, or ledgers and do not access the network.

- [ ] **Step 3: Run focused tests, static checks, and commit.**

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rescore_identifier_semantics.py -q
  & 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src/paper_search/evaluation/identifier_semantics.py scripts/rescore_identifier_semantics.py tests/evaluation/test_identifier_semantics.py tests/scripts/test_rescore_identifier_semantics.py
  & 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src/paper_search/evaluation/identifier_semantics.py scripts/rescore_identifier_semantics.py
  git add -- src/paper_search/evaluation/identifier_semantics.py tests/evaluation/test_identifier_semantics.py scripts/rescore_identifier_semantics.py tests/scripts/test_rescore_identifier_semantics.py
  git commit -m "feat: rescore sealed runs with verified identities"
  ```

- [ ] **Step 4: Execute the already bound offline comparison.**

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\paper-search.exe' verify-run runs/dev-20260810T104256Z-d9e89476d484
  & 'D:\AI Projects\Projects\.venv\Scripts\paper-search.exe' verify-run runs/dev-20260809T061903Z-9bd861e90299
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/rescore_identifier_semantics.py --gold data/dev/gold.jsonl --id-map data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v2.json --private-audit data/annotation_work/identifier_semantics/relation-audit.v2.json --audit docs/evidence/identifier-map-semantic-audit-2026-08-11.json --identity-evidence data/annotation_work/identifier_semantics/identity-evidence.json --snapshot-root data/annotation_work/identifier_semantics/snapshots --formal-run runs/dev-20260810T104256Z-d9e89476d484 --formal-run runs/dev-20260809T061903Z-9bd861e90299 --legacy-run runs/dev-20260805T035209Z-7af4b103f6cc --legacy-evidence docs/evidence/title-retention-offline-2026-08-09.json --query-evolution-probe runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810 --out-json docs/evidence/identifier-map-semantic-rescore-2026-08-11.json --out-report docs/identifier-map-semantic-rescore-2026-08-11.md
  ```

  Both formal verifications must pass first. The rescore covers those runs, the exact-hash-bound legacy title run, and the matched query-evolution probe using the new v2 generation. Publish only aggregate privacy-checked outputs. Require all 12 direct arXiv hits to be counted; on any integrity, retrieval, ranking, or budget guard failure, stop this improvement direction and request human review.

### Task 5: Final verification and handoff decision

- [ ] Run the complete repository test suite, Ruff, and mypy. Report exact pass/fail counts and distinguish pre-existing failures from regressions; do not claim completion if any relevant failure remains.

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -q
  & 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check .
  & 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src scripts
  ```

- [ ] Inspect `git diff --check`, `git status --short`, and the commits from Tasks 1, 2, and 4. Confirm protected user-owned paths and sealed inputs were neither staged nor modified.
- [ ] Review the rescore funnel once. If trustworthy evidence identifies a retrieval/ranking bottleneck, propose one bounded next experiment; otherwise stop and request a human decision. Never return automatically to identity capture.
- [ ] Update `HANDOFF.md` and `docs/retrieval-roadmap.md` only after separate user authorization, because both currently contain user-owned edits.
