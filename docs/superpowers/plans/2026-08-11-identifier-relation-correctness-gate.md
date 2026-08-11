# Identifier Relation-Correctness Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the false provider-coverage requirement with the approved relation-correctness Gate and publish one strictly verifiable v2 identifier generation from the existing sealed evidence.

**Architecture:** Keep capture, snapshots, ledger, and semantic classification unchanged. Extend the offline semantic models with the closed v2 public/private schemas, rebuild actual relations from read-once sealed bytes, and publish through an exclusive public-audit-last transaction. A strict path-based loader verifies the completed generation and becomes the only identifier input permitted for the later rescore task.

**Tech Stack:** Python 3.11, Pydantic domain models, pytest, Ruff, mypy, canonical JSON, SHA-256, existing dependency-snapshot reader.

## Global Constraints

- [ ] Perform no readiness, capture, ledger mutation, `.env` access, OpenAlex request, or Semantic Scholar request.
- [ ] Preserve `HANDOFF.md`, `docs/retrieval-roadmap.md`, `data/budget_ledger.sqlite3`, `deliverables/`, and `docs/evidence/identifier-map-semantic-audit-2026-08-10.json`.
- [ ] Do not modify the collector, sealed inputs, shared snapshot contract, ledger, classifier, historical runs, or old identifier map.
- [ ] Use fresh v2 targets and the exact sibling lock `docs/evidence/identifier-map-semantic-audit-2026-08-11.json.lock`; never overwrite, reuse, or auto-clean a target or lock.
- [ ] Stop for human intervention on input-integrity failure, decoder regression, semantic mismatch/conflict, stale publication state, or failed strict loading. Never respond by repeating capture.
- [ ] Follow RED → GREEN → focused regression and commit only the files named by each task.

---

### Task 1: Encode the closed v2 schemas and corrected relation set

**Files:**

- Modify: `src/paper_search/evaluation/identifier_semantics.py`
- Modify: `tests/evaluation/test_identifier_semantics.py`
- Modify: `scripts/rebuild_dev_identifier_map.py`
- Modify: `tests/scripts/test_rebuild_dev_identifier_map.py`

**Interfaces:**

- Produce `IdentifierMapSemanticAudit` with literal schema `identifier-map-semantic-audit-v2` and exactly the fields/enums in the approved design.
- Add `relation_kind: Literal["required_anchor", "provider_candidate"]` to each private row.
- Add `PrivateRelationAuditV2(schema_version, scope, relations)` with extra fields forbidden and rows unique/sorted by `(relation_kind order, arxiv_id, alias)`.
- Keep `classify_relation()` and all proof rules unchanged.

- [ ] **Step 1: Add RED schema and relation-conservation tests.**

  Add `test_v2_models_reject_missing_extra_unknown_enum_and_boolean_counts`, `test_v2_counts_include_all_zero_enum_keys`, `test_anchor_only_group_passes_without_provider_placeholder`, `test_failed_anchor_is_semantic_failure_not_decoder_regression`, `test_each_evidence_ref_creates_one_provider_candidate`, `test_failed_provider_candidates_are_retained_and_stop_promotion`, `test_alias_conflict_retains_every_contributor`, and `test_anchor_and_same_alias_provider_candidate_remain_distinct`. Together they cover the closed schemas, strict non-boolean counts, all count/set/hash equations, and the required `relation_kind` distinction.

  Run:

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py -q
  ```

  Expected: the new tests fail because the implementation still emits audit v1, an unwrapped private list, and synthetic `provider_identity_missing` rows.

- [ ] **Step 2: Implement the exact models and canonical serializers.**

  Use Pydantic models derived from `DomainModel` so extra fields are forbidden. Use strict non-negative integer fields that reject booleans, plus the design's exact hash pattern, enums, nullable `proof_kind`, identifier normalization, row invariants, and canonical JSON definition. Keep cross-artifact equations in the pure builder validator, where private rows are available; model validators enforce only equations provable from one model.

- [ ] **Step 3: Implement the corrected pure relation construction.**

  In `rebuild_dev_map()` create one `required_anchor` row per normalized unique gold arXiv ID, consume each canonical unique `(arxiv_id, alias)` evidence ref exactly once into one `provider_candidate` row, classify before filtering, and remove provider-missing row synthesis. Resolve conflicts across all provider candidates sharing one normalized alias across distinct arXiv IDs, retaining every contributor. Build map bytes only from verified anchors and aliases.

- [ ] **Step 4: Add the sealed-input decoder regression.**

  When and only when the three source hashes equal the approved values, require `141` gold groups, `141` required-anchor rows, `90` provider groups, `51` missing-provider groups, `90` provider candidates, and `231` total rows. Raise the value-free decoder-regression error before publication if any fixed count differs. Evaluate `verified_anchor_count` only in the semantic Gate so an anchor mismatch produces the specified failed private/public audits rather than a decoder rejection.

- [ ] **Step 5: Verify and commit Task 1.**

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py -q
  git add -- src/paper_search/evaluation/identifier_semantics.py tests/evaluation/test_identifier_semantics.py scripts/rebuild_dev_identifier_map.py tests/scripts/test_rebuild_dev_identifier_map.py
  git commit -m "fix: enforce identifier relation correctness"
  ```

  Expected: selected tests pass and no protected path is staged.

### Task 2: Enforce read-once transactional publication and strict loading

**Files:**

- Modify: `src/paper_search/evaluation/identifier_semantics.py`
- Modify: `tests/evaluation/test_identifier_semantics.py`
- Modify: `scripts/rebuild_dev_identifier_map.py`
- Modify: `tests/scripts/test_rebuild_dev_identifier_map.py`

**Interfaces:**

- Add `VerifiedIdentifierGeneration(audit: IdentifierMapSemanticAudit, identifier_map: IdentifierMap)`.
- Add `load_verified_identifier_generation(*, audit_path: Path, gold_path: Path, evidence_path: Path, snapshot_manifest_path: Path, private_audit_path: Path, map_path: Path) -> VerifiedIdentifierGeneration`.
- The CLI retains exactly its existing six arguments. Its sibling publication-lock path is derived as `<out-public-audit>.lock` and is not configurable.

- [ ] **Step 1: Add RED publisher and read-once tests.**

  Add `test_publication_paths_and_lock_are_exclusive`, `test_inputs_are_opened_once_after_lock`, `test_only_referenced_responses_are_opened_once`, `test_shared_response_is_opened_once`, `test_replacing_input_after_read_cannot_change_decoding`, `test_no_replace_race_writes_no_marker`, and one named test for each of the four design outcome paths. Assert the manifest is opened once, unreferenced responses are never opened, shared referenced responses are opened once, and interruption immediately before the public write leaves an unloadable generation.

- [ ] **Step 2: Add RED loader and attack-surface tests.**

  Add `test_loader_rejects_public_audit_before_opening_other_paths`, parameterized public/private/map corruption tests, and `test_cli_exposes_only_the_six_authorized_paths`. The first test records `Path.open` order; corruption cases cover duplicate keys, missing/extra/wrong fields, noncanonical bytes, privacy violations, failed status, all five hash mismatches, duplicate raw/normalized map keys, extra structure, chains, and cycles. Monkeypatch `socket.create_connection` to fail and set sentinel `.env`/override variables; a successful fixture rebuild must neither call the socket nor change output bytes. Because the parser has only the six exact paths and function signatures accept no predictions, queries, validation data, historical runs, manual aliases, overrides, or sidecars, those inputs have no callable entry point.

- [ ] **Step 3: Implement publication from immutable byte buffers.**

  Acquire the exclusive lock before reading inputs. Read gold, evidence, and manifest once; build an entry index from that manifest buffer; then read only distinct response entries named by evidence refs, once each. Pass those same bytes through hashing, validation, decoding, relation construction, and serialization. Use atomic exclusive no-replace writes. All four controlled outcomes release the lock after completing their specified writes or no-write rejection; only process interruption or failure to make the final marker durable leaves the lock and residuals for human archival.

- [ ] **Step 4: Implement the strict loader in the specified order.**

  Apply recursive privacy validation only to the public audit. Validate private bytes with duplicate-key rejection, closed schema, canonical reserialization, hashes, row invariants, and cross-artifact relation consistency. Verify the three source hashes and raw private/map hashes before parsing private rows or map entries; then recompute all public equations and require the map to equal the exact map reconstructed from verified private rows. Return only the validated audit and strict `IdentifierMap`; expose no bypass parameter.

- [ ] **Step 5: Run focused and adjacent regressions.**

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py -q
  & 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src/paper_search/evaluation/identifier_semantics.py scripts/rebuild_dev_identifier_map.py tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py
  & 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src/paper_search/evaluation/identifier_semantics.py scripts/rebuild_dev_identifier_map.py
  ```

  Expected: all selected tests and affected-file static checks pass.

- [ ] **Step 6: Commit Task 2.**

  ```powershell
  git add -- src/paper_search/evaluation/identifier_semantics.py tests/evaluation/test_identifier_semantics.py scripts/rebuild_dev_identifier_map.py tests/scripts/test_rebuild_dev_identifier_map.py
  git commit -m "fix: publish verified identifier generations"
  ```

### Task 3: Publish and independently verify one sealed v2 generation

**Fresh targets:**

- Private map: `data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v2.json`
- Private audit: `data/annotation_work/identifier_semantics/relation-audit.v2.json`
- Public marker: `docs/evidence/identifier-map-semantic-audit-2026-08-11.json`
- Publication lock: `docs/evidence/identifier-map-semantic-audit-2026-08-11.json.lock`

- [ ] **Step 1: Perform an informational preflight without authorizing publication.**

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -c "from pathlib import Path; import hashlib; paths=['data/dev/gold.jsonl','data/annotation_work/identifier_semantics/identity-evidence.json','data/annotation_work/identifier_semantics/snapshots/snapshot-manifest.json']; print(*[f'{p}=sha256:{hashlib.sha256(Path(p).read_bytes()).hexdigest()}' for p in paths], sep='\n')"
  @( 'data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v2.json','data/annotation_work/identifier_semantics/relation-audit.v2.json','docs/evidence/identifier-map-semantic-audit-2026-08-11.json','docs/evidence/identifier-map-semantic-audit-2026-08-11.json.lock') | ForEach-Object { "$_=$([bool](Test-Path -LiteralPath $_))" }
  ```

  Expected: `dev_gold=sha256:24009cf03ad069131793b9a190024e239082277bd0e48149a1efbbbb7978e215`, `identity_evidence=sha256:e4567d4b7641871ed538c18f5625cd7037e3014065e7311d3a76e81d4e4c61d4`, `snapshot_manifest=sha256:a0c0cd67543582e02365a2adfb3464a6f33fa96be5304d4ccac8dd031867943b`, and every target reports `False`. Any mismatch is a stop; the builder still repeats the authoritative checks after locking.

- [ ] **Step 2: Run the offline builder exactly once.**

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' scripts/rebuild_dev_identifier_map.py --gold data/dev/gold.jsonl --evidence data/annotation_work/identifier_semantics/identity-evidence.json --snapshot-root data/annotation_work/identifier_semantics/snapshots --out-map data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v2.json --out-private-audit data/annotation_work/identifier_semantics/relation-audit.v2.json --out-public-audit docs/evidence/identifier-map-semantic-audit-2026-08-11.json
  ```

  Expected: exit `0`; no network or ledger activity; no publication lock remains after the durable passed marker.

- [ ] **Step 3: Independently load the completed generation.**

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -c "from pathlib import Path; from paper_search.evaluation.identifier_semantics import load_verified_identifier_generation as load; g=load(audit_path=Path('docs/evidence/identifier-map-semantic-audit-2026-08-11.json'),gold_path=Path('data/dev/gold.jsonl'),evidence_path=Path('data/annotation_work/identifier_semantics/identity-evidence.json'),snapshot_manifest_path=Path('data/annotation_work/identifier_semantics/snapshots/snapshot-manifest.json'),private_audit_path=Path('data/annotation_work/identifier_semantics/relation-audit.v2.json'),map_path=Path('data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v2.json')); print(g.audit.status,g.audit.gold_group_count,g.audit.verified_anchor_count,g.audit.provider_identity_group_count,g.audit.provider_identity_missing_group_count,g.audit.provider_candidate_count,g.audit.relation_count)"
  ```

  Expected: `passed 141 141 90 51 90 231`. Any other result stops before downstream rescore work.

- [ ] **Step 4: Commit only the privacy-validated public marker.**

  ```powershell
  git add -- docs/evidence/identifier-map-semantic-audit-2026-08-11.json
  git diff --cached --name-only
  git commit -m "docs: record verified identifier generation"
  ```

  Expected: the staged list contains only the 2026-08-11 public marker. Private map/audit files and protected paths remain unstaged.

### Task 4: Final scope verification and downstream handoff

- [ ] **Step 1: Re-run the one combined relevant gate.**

  ```powershell
  & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py tests/scripts/test_capture_dev_identifier_identity.py tests/unit/test_dependency_snapshot.py tests/unit/test_budget_ledger.py -q
  & 'D:\AI Projects\Projects\.venv\Scripts\ruff.exe' check src/paper_search/evaluation/identifier_semantics.py scripts/rebuild_dev_identifier_map.py tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py
  & 'D:\AI Projects\Projects\.venv\Scripts\mypy.exe' src/paper_search/evaluation/identifier_semantics.py scripts/rebuild_dev_identifier_map.py
  git diff --check 9ce9c48..HEAD -- src/paper_search/evaluation/identifier_semantics.py scripts/rebuild_dev_identifier_map.py tests/evaluation/test_identifier_semantics.py tests/scripts/test_rebuild_dev_identifier_map.py docs/evidence/identifier-map-semantic-audit-2026-08-11.json
  git status --short
  ```

  Expected: all relevant checks pass; only the previously protected user-owned paths and private/generated ignored artifacts remain outside commits.

- [ ] **Step 2: Stop at the explicit downstream boundary.**

  Do not implement, run, or edit the rescore plan in this task. After the v2 loader succeeds, report that Task 4 of `docs/superpowers/plans/2026-08-10-identifier-map-semantic-recovery.md` must be separately revised to consume `load_verified_identifier_generation()` and freeze its source bindings, aggregate report schema, metric formulas, 12-direct-hit counting rule, and integrity/retrieval/ranking/budget stop conditions. Wait for separate approval before modifying that plan.

- [ ] **Step 3: Leave handoff files untouched pending authorization.**

  Do not update `HANDOFF.md` or `docs/retrieval-roadmap.md` because both contain user-owned edits. Report the three implementation commits, exact verification counts, public audit path/hash, private artifact paths/hashes, and the downstream stop/continue decision in the task response.
