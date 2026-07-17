# Week 1 Collaborator Checklist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the outdated Task 2 onboarding with an immediately executable Week 1 collaborator checklist and publish it on the collaboration branch.

**Architecture:** Keep the stable `docs/TEAMMATE_ONBOARDING.md` entry point and replace its contents. Organize work as gated packages so the collaborator can distinguish preparation, human labeling, data freeze, formal baseline, audit, and Week 1 gate activities without relying on prior conversations.

**Tech Stack:** Markdown, Git, PowerShell, repository Task 2/Task 4 data contracts.

## Global Constraints

- Never read, print, search, parse, or copy `.env` contents.
- Do not touch `docs/superpowers/specs/2026-07-15-task2-evaluation-design.md`.
- Do not commit restricted PaSa data, real queries, gold labels, or human annotations.
- Do not claim a frozen dataset or Week 1 gate without complete, hash-consistent evidence.
- Modify files with `apply_patch` except mechanical formatting when the sandbox prevents it.
- Check the staged file list before and after every staging operation.
- Push only `codex/week1-collaboration`; do not create a PR, merge, force-push, or clean branches/worktrees.

---

### Task 1: Replace and Publish the Week 1 Collaborator Checklist

**Files:**
- Modify: `docs/TEAMMATE_ONBOARDING.md`
- Verify: `docs/superpowers/specs/2026-07-17-week1-collaborator-checklist-design.md`

**Interfaces:**
- Consumes: PRD Week 1 Task 2/Task 4 status, `data/README.md` frozen partition contract, and Task 4 acceptance boundaries.
- Produces: one authoritative collaborator checklist at `docs/TEAMMATE_ONBOARDING.md`.

- [ ] **Step 1: Replace the old onboarding text**

Write sections for current status and red lines, startup checks, data preparation, 90 type/domain labels, 40 constraint labels, 20 independent overlap labels, real manifest freeze, formal baseline, fuzzy-dedup audit, Recall/cost/failure audit, Week 1 gate criteria, and reporting templates.

- [ ] **Step 2: Verify checklist completeness**

Run:

```powershell
rg -n "90 条|40 条|20 条|0.80|98%|0.02|zero_answer_policy|gold_sha256|ids_sha256|Week 1" docs/TEAMMATE_ONBOARDING.md
```

Expected: every required quantity, threshold, hash field, and gate term appears in an actionable section.

- [ ] **Step 3: Verify scope and safety**

Run:

```powershell
git diff --name-only 6e4128e...HEAD
git diff --check
git status --short
git grep -n -I -E "OPENALEX_API_KEY=.+|HF_TOKEN=.+|Authorization:[[:space:]]*(Bearer|Basic)[[:space:]]+" -- docs/TEAMMATE_ONBOARDING.md docs/superpowers/specs/2026-07-17-week1-collaborator-checklist-design.md
```

Expected: only the design, plan, and onboarding files differ from the Task 4 baseline; whitespace and secret scans are clean; the protected Task 2 design is absent.

- [ ] **Step 4: Run independent reader review**

Give a fresh reviewer only `docs/TEAMMATE_ONBOARDING.md` and ask whether a collaborator can determine immediate actions, blockers, deliverables, forbidden Git content, freeze evidence, and Week 1 gate conditions. Resolve every valid ambiguity before committing.

- [ ] **Step 5: Stage with before/after checks and commit**

Run:

```powershell
git diff --cached --name-only
git add -- docs/TEAMMATE_ONBOARDING.md docs/superpowers/plans/2026-07-17-week1-collaborator-checklist-implementation.md
git diff --cached --name-only
git diff --cached --check
git commit -m "docs: update week one collaborator checklist"
```

Expected: exactly the onboarding and implementation plan are staged for this commit; commit succeeds.

- [ ] **Step 6: Push the collaboration branch**

Run:

```powershell
git push -u origin codex/week1-collaboration
```

Expected: remote branch `origin/codex/week1-collaboration` is created or fast-forwarded without force.
