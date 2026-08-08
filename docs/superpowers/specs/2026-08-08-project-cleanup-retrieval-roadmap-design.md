# Project Cleanup and Retrieval Roadmap Design

**Date:** 2026-08-08  
**Workspace:** `D:\AI Projects\.worktrees\week3`

## 1. Objective

Remove rebuildable or superseded work products, simplify active names, and replace the current improvement plan with a roadmap driven by existing experiment evidence.

Formal run evidence, diagnostic artifacts, frozen data, and current deliverables remain protected. This cleanup does not call external providers, read `.env`, run validation, or change retrieval code.

## 2. Cleanup Boundary

Delete these rebuildable local artifacts:

- `.mypy_cache/`
- `.ruff_cache/`
- `.uv-cache/`
- `.pdf-check/`
- `.sheet-build/`
- `.superpowers/sdd/`
- `.gate0-report.json`

Delete these superseded work products:

- `outputs/annotation_status_20260729/`
- `deliverables/初赛提交包_20260805/`
- `deliverables/VivaAI_材料交接包_20260805.zip`
- `docs/PROJECT_HANDOFF_TASK4.md`
- `docs/superpowers/plans/2026-07-28-task10-experiment-ablation.md`
- `docs/superpowers/plans/2026-07-28-week3-task10-experimentation.md`
- `docs/superpowers/plans/2026-07-28-week3-task9-embedding-ranking.md`
- `docs/superpowers/plans/2026-07-29-data-freeze-v2.md`

After transferring its valid conclusions to the experiment decision record, delete `academic_retrieval_v3_optimization_plan.md`.

Preserve:

- all of `runs/`, including failed runs, snapshots, ledger files, and `_diag_*`
- all of `data/`
- `.venv/`
- tracked tests, fixtures, historical designs, and implementation plans not listed above
- `HANDOFF.md`
- the 2026-08-06 submission, demo, and project-document sources

Add root ignore rules for `.uv-cache/`, `.pdf-check/`, `.sheet-build/`, and `.superpowers/`.

## 3. Naming and Active Documents

Apply these renames:

- `docs/improvement-plan-2026-08-07.md` → `docs/retrieval-roadmap.md`
- `deliverables/初赛提交包_20260806/` → `deliverables/submission/`
- `deliverables/演示包_20260806/` → `deliverables/demo/`
- `deliverables/项目文档_20260806/` → `deliverables/project-docs/`

Keep `README.md`, `PRD.md`, and `HANDOFF.md` as the standard entry points. Update live path references after renaming. Keep version dates in the existing submission and demo READMEs, and add `deliverables/project-docs/README.md` for the project-document sources.

Create `docs/experiment-decisions.md` as the authoritative record of tested retrieval approaches. Update `HANDOFF.md` to reflect the cleaned structure, active roadmap, rejected experiments, and lock status. Do not create another project-status document.

## 4. Experiment Decisions

The decision record must capture the aggregate evidence below without frozen query text:

| Approach | Existing result | Decision and reopening condition |
| --- | --- | --- |
| Citation expansion | 0/20 gold recall | Rejected; reopen only with materially different seeds, graph source, or expansion method and a positive bounded probe |
| Topic retrieval | No gold in top 50; ceiling hit only at rank 180 | Rejected; reopen only after a materially different mapping/indexing method passes a new ceiling probe |
| Embedding reranking | Gold top-50 fell from 13 to 6 and F1 regressed | Rejected; reopen only if the candidate pool, representation, or training objective changes and offline F1 improves |
| Query rewrites | No zero-hit query recovery | Rejected; reopen only when the generator uses evidence unavailable to the prior experiment |
| LLM query variants | Gold top-50 fell from 13 to 8–10 with high request overhead | Rejected; reopen only after a bounded exact-ID recall gain that preserves existing hits |
| Title candidates | Only positive recall signal; union pool reached 41 gold across 24 queries | Continue; diagnose candidate loss before generating more titles |

Any rejected approach requires a new falsifiable hypothesis and a low-cost positive probe before formal capture.

## 5. Retrieval Roadmap

### Phase 0: Clean baseline

After cleanup and lock renewal, run readiness → dev capture → verify → replay → compare. Accept the baseline only when the quality gate passes, `provenance_failures=0`, and capture/replay business results agree.

### Phase 1: Diagnose availability and candidate loss

1. Use DOI, arXiv ID, and OpenAlex ID for direct read-only availability checks. Report aggregate reason codes only; never turn gold identifiers into retrieval queries.
2. Count exact gold retention at generated titles, OpenAlex verification, merged pool, RRF pool, and final `selected_paper_ids`.

The existing P0 probe measured title-search recall, not exact identifier availability. The next implementation target is the stage with the largest measured candidate loss.

### Phase 2: Improve title-candidate retention and selection

Compare 10 and 20 titles on the same frozen dev input. Inspect verification rank, fusion contribution, and final truncation. Run the PRD-defined offline grid:

- `K ∈ {10, 20, 30, 50}`
- score threshold `0.45–0.75` in `0.05` increments

Select by macro F1, with precision, recall, Recall@K, MRR, and NDCG as safeguards. Freeze the selected combination before validation. Variants without offline gain do not receive live capture.

### Phase 3: Query Evolution only after redesign

Do not directly enable the current `fixed_two_round`. It uses rule fallback instead of the production DeepSeek `QuerySpec`, disables title candidates under its experiment identity, and estimates second-round usage as zero.

Before a live experiment, make it compose with the production analysis and selected title-candidate baseline, and derive non-zero budget estimates from real calls. Compare rule and LLM generation one at a time in a bounded probe. Advance only if exact-ID recall improves without losing existing hits.

Query Type routing, Selector/LLM reranking, and new data sources remain conditional on the preceding diagnostics. Every promoted change remains a single-variable experiment with its own configuration, lock, and capture/replay/compare evidence.

## 6. Implementation and Checks

1. Resolve each approved destructive target as a literal path inside the workspace, then delete only those targets.
2. Rename the approved files and directories, create the decision record, and rewrite the roadmap and handoff.
3. Update `.gitignore`, READMEs, scripts, and other live path references.
4. Verify that approved targets are absent, protected paths remain present, old active paths have no stale references, and the three active documents agree.
5. Run `git diff --check` and focused non-network document/path checks, then commit the cleanup.

The deleted untracked deliverables cannot be restored through Git; their deletion was explicitly approved. If an exact target differs from this design at execution time, stop before deleting it.

## 7. Candidate Lock

The current v21 lock binds commit `c22abf9`. The design and cleanup commits change `HEAD`, so v21 must not be used again.

After cleanup, rebuild the next lock against the final commit and current ledger checkpoint under a new run approval. Lock rebuilding and live capture are outside this cleanup and require separate authorization.
