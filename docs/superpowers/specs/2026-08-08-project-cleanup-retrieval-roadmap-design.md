# Project Cleanup and Retrieval Roadmap Design

**Date:** 2026-08-08  
**Workspace:** `D:\AI Projects\.worktrees\week3`  
**Status:** Approved design; implementation has not started

## 1. Goals

This change has two goals:

1. Remove rebuildable, superseded, or abandoned local work products while preserving formal evaluation evidence and frozen data.
2. Replace the current retrieval improvement plan with an evidence-driven roadmap that does not repeat experiments already rejected by project data.

The cleanup must leave one clear project entry point, one active retrieval roadmap, and one durable experiment decision log.

## 2. Non-goals

- Do not delete or rewrite formal `runs/` evidence, dependency snapshots, or `_diag_*` artifacts.
- Do not modify frozen `data/`, including `data/manifest.json`.
- Do not read or print `.env`.
- Do not call external providers, run validation, or start a live capture.
- Do not change retrieval implementation code as part of the cleanup.
- Do not treat a rejected experiment as permanently impossible; reopening requires a materially different hypothesis and explicit evidence.

## 3. Approved Cleanup Boundary

### 3.1 Delete rebuildable local artifacts

Delete these exact paths:

- `.mypy_cache/`
- `.ruff_cache/`
- `.uv-cache/`
- `.pdf-check/`
- `.sheet-build/`
- `.superpowers/sdd/`
- `.gate0-report.json`

`.gate0-report.json` is a stale failed report from 2026-07-30. The current formal projection is `data/gate0_evidence.json`.

Add repository-root ignore rules for `.uv-cache/`, `.pdf-check/`, `.sheet-build/`, and `.superpowers/` so these local products do not return to `git status`. Existing ignore rules for Python caches remain in force.

### 3.2 Delete superseded work products

Delete these exact paths:

- `outputs/annotation_status_20260729/`
- `deliverables/初赛提交包_20260805/`
- `deliverables/VivaAI_材料交接包_20260805.zip`
- `docs/PROJECT_HANDOFF_TASK4.md`
- `docs/superpowers/plans/2026-07-28-task10-experiment-ablation.md`
- `docs/superpowers/plans/2026-07-28-week3-task10-experimentation.md`
- `docs/superpowers/plans/2026-07-28-week3-task9-embedding-ranking.md`
- `docs/superpowers/plans/2026-07-29-data-freeze-v2.md`

The four plan files are currently untracked and superseded by later integrated plans. The Task 4 handoff is tracked but obsolete because `HANDOFF.md` is the active recovery entry point.

### 3.3 Preserve evidence and current deliverables

Preserve without content changes:

- `runs/`, including failed captures, snapshots, ledger files, and every `_diag_*` artifact
- `data/`, including frozen gold, manifests, identifier maps, and readiness evidence
- `.venv/`
- Git-tracked tests and fixtures
- Git-tracked historical designs and implementation plans not listed in section 3.2
- `HANDOFF.md`
- The 2026-08-06 submission, demo, and project-document sources

## 4. Naming and Documentation Structure

Keep the conventional root entry points:

- `README.md`
- `PRD.md`
- `HANDOFF.md`

Apply these renames:

- `docs/improvement-plan-2026-08-07.md` → `docs/retrieval-roadmap.md`
- `deliverables/初赛提交包_20260806/` → `deliverables/submission/`
- `deliverables/演示包_20260806/` → `deliverables/demo/`
- `deliverables/项目文档_20260806/` → `deliverables/project-docs/`

Create `docs/experiment-decisions.md` as the authoritative record of completed diagnostic experiments. After its valid evidence is transferred, delete `academic_retrieval_v3_optimization_plan.md`.

Dates and version provenance move into each deliverable README or document metadata instead of remaining in active directory names. Search the repository after every rename and update live references in documentation and scripts. Do not rewrite frozen evidence merely to replace a path string.

Update `HANDOFF.md` to describe the post-cleanup paths, current baseline state, approved roadmap, rejected experiments, and lock consequences. Do not create a second project-status document.

## 5. Experiment Decision Record

`docs/experiment-decisions.md` must contain one row per completed approach with its sample, metric, result, decision, and reopening condition. At minimum it records:

| Approach | Existing evidence | Decision | Reopen only if |
| --- | --- | --- | --- |
| Citation expansion | 0/20 gold recall in the completed probe | Rejected | Candidate seeds, graph source, or expansion mechanism materially changes and an offline graph test is positive |
| Topic retrieval | No gold in top 50; ceiling probe found one item only at rank 180 | Rejected | Topic mapping or corpus indexing materially changes and a new ceiling probe is positive |
| Embedding reranking | Gold top-50 fell from 13 to 6; F1 also regressed | Rejected | Candidate pool, representation, or training objective changes and offline F1 beats the capture order |
| Query rewrites | Zero-hit queries were not recovered | Rejected | The generator uses new evidence unavailable to the prior rewrite experiment |
| LLM query variants | Gold top-50 fell from 13 to 8–10 with high request overhead | Rejected | A bounded probe demonstrates positive exact-ID recall while preserving existing hits |
| Title candidates | Only completed approach with positive recall signal; union pool reached 41 gold across 24 queries | Continue | Active optimization focuses on candidate retention and selection before generating more titles |

The record must not include frozen query text. Reopening a rejected path requires a written, falsifiable hypothesis and a low-cost offline or bounded probe before a formal capture.

## 6. Retrieval Roadmap

### 6.1 Phase 0: Establish a clean formal baseline

After cleanup and lock renewal, run the formal sequence:

1. Refresh provider readiness.
2. Run one dev live capture.
3. Run `verify-run`.
4. Run zero-network replay.
5. Run `compare-replay`.

The baseline is accepted only when the quality gate passes, `provenance_failures=0`, and capture/replay business results agree. No new full online experiment starts before this baseline exists. Offline diagnostics that do not change the system may proceed earlier.

### 6.2 Phase 1: Run two bounded diagnostics

#### Exact identifier availability

Use the frozen DOI, arXiv, and OpenAlex identifiers for direct read-only lookup. Report only aggregate availability and per-query reason codes. Gold identifiers must never be transformed into retrieval queries or used to improve a prediction directly.

This is distinct from the existing P0 title-search probe: that probe measured whether generated titles retrieved gold, not whether every gold identifier exists in OpenAlex.

If OpenAlex coverage is materially incomplete, consider another source only for the missing identifier classes. If coverage is high, continue optimizing query/title generation and candidate retention.

#### Candidate attrition

Measure exact gold retention at these boundaries:

1. generated title list
2. OpenAlex title verification results
3. merged candidate pool
4. RRF-ranked pool
5. final `selected_paper_ids`

Choose the next implementation target from the largest observed loss. This diagnostic is the priority because prior data shows a title-candidate union pool of 41 gold across 24 queries but only about 10 gold across 9 queries in final output.

### 6.3 Phase 2: Optimize title-candidate retention and output selection

Compare 10 and 20 generated titles on the same frozen dev input. Do not assume that more generated titles improve final F1.

Inspect title verification rank, fusion contribution, and final truncation. Run the PRD-defined offline grid:

- `K ∈ {10, 20, 30, 50}`
- score threshold from `0.45` through `0.75` in `0.05` increments

Select by macro F1. Use precision, recall, Recall@K, MRR, and NDCG as safeguards and explanations. Freeze the selected K/threshold combination before any validation attempt. A combination with no offline gain does not receive a live capture.

This replaces the previous unsupported recommendation to set output size directly to 10–15.

### 6.4 Phase 3: Treat Query Evolution as a conditional experiment

Do not enable the current `fixed_two_round` path directly. It currently:

- rebuilds the query with rule fallback instead of using the production DeepSeek `QuerySpec`
- disables the title-candidate component under its experiment identity
- estimates all second-round provider, token, cost, and elapsed usage as zero
- overlaps with previously negative query-rewrite and LLM-variant experiments

Before a Query Evolution capture, redesign the experiment so it uses the production analysis, composes with the selected title-candidate baseline, and has non-zero estimates derived from real calls. Compare rule and LLM generation in a bounded probe, one at a time. Only a variant with positive exact-ID recall that preserves existing hits may advance to formal capture.

### 6.5 Later conditional work

- Add Query Type routing only after per-type error analysis shows a stable and actionable difference.
- Add Selector or LLM reranking only after recall has materially improved.
- Add another scholarly source only if exact-identifier availability shows an OpenAlex coverage gap.
- Keep citation, topic, embedding, and ordinary rewrite experiments closed unless their reopening conditions in the decision record are met.

Every promoted change uses one independent experiment identity, one configuration change, a rebuilt lock, and a capture/replay/compare evidence chain.

## 7. Implementation Order

1. Record the approved design in this document and commit it alone.
2. Capture a pre-delete inventory with exact paths, file counts, and sizes.
3. Delete only the approved cache and superseded paths.
4. Rename the approved active deliverable directories and roadmap.
5. Create `docs/experiment-decisions.md` from existing aggregate evidence.
6. Rewrite `docs/retrieval-roadmap.md` to match section 6.
7. Update `HANDOFF.md`, `.gitignore`, README references, scripts, and deliverable READMEs.
8. Delete `academic_retrieval_v3_optimization_plan.md` after its decisions are represented in the decision record.
9. Search for stale paths, stale priorities, and contradictory experiment states.
10. Commit cleanup and documentation changes.
11. Rebuild and validate the next candidate lock against the final commit and current ledger checkpoint only after separate operator authorization for the next formal run.

## 8. Destructive-action Safety

Before each recursive delete, resolve the literal target to an absolute path and verify that it is inside `D:\AI Projects\.worktrees\week3`. Use literal paths, not globs or environment-variable-derived targets. Delete no sibling path and no workspace root.

The deleted deliverables and outputs are untracked and cannot be restored through Git. Their deletion was explicitly approved. If a target differs from the approved inventory at execution time, stop and reassess before deletion.

If any validation fails, stop the cleanup sequence. Do not continue deleting unrelated targets and do not start external evaluation.

## 9. Verification

The cleanup is complete only when all of these checks pass:

- Every approved deletion target is absent.
- `runs/`, `_diag_*`, `data/`, `.venv/`, and the three renamed 2026-08-06 deliverables are present.
- `rg` finds no live reference to the old roadmap or deliverable paths, except an intentional historical statement in the decision record.
- `HANDOFF.md`, `docs/retrieval-roadmap.md`, and `docs/experiment-decisions.md` agree on priorities and rejected experiments.
- `.gitignore` covers the approved local build directories.
- `git diff --check` passes.
- Documentation/path checks and any focused non-network tests pass.
- No `.env` content, frozen query text, or validation gold is added to tracked output.

## 10. Candidate-lock Consequence

The existing v21 candidate lock binds source commit `c22abf9`. Committing this design and the cleanup changes changes `HEAD`, so v21 must not be used for the next formal capture.

After the cleanup commit, the next lock must bind:

- the final cleanup commit SHA
- the then-current project ledger checkpoint
- current configuration and policy hashes
- a new approval reference issued for that formal run

Lock rebuilding does not itself authorize a live run. The cleanup task must not automatically refresh readiness, contact providers, or start capture.
