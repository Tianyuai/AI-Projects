# Week 1 Collaborator Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the collaborator's prepared Task 2 metadata reproducible across Git checkouts, truthfully update Week 1 status, and publish the annotation-work-package v1 start notice without claiming a formal data freeze.

**Architecture:** Keep collaborator commit `d6adb6e` unchanged as the prepared-data baseline. Add a repository-level LF checkout contract for exact-byte hashed metadata, verify it through Git's own attribute resolver and byte hashes, then update only the authoritative PRD and collaborator onboarding status. Human labels, gold, formal approval, online baseline, and the Week 1 gate remain separate later stages.

**Tech Stack:** Git attributes, Python 3.11, pytest, PowerShell, Markdown, deterministic UTF-8 JSON.

## Global Constraints

- Never read, print, search, parse, or copy `.env` contents.
- Offline commands use `uv run --no-sync --no-env-file`.
- Do not touch `docs/superpowers/specs/2026-07-15-task2-evaluation-design.md`.
- Do not commit PaSa raw data, real queries, gold, private work packages, or human labels.
- Do not modify manifest hashes to accommodate checkout newline conversion.
- Do not change `data/manifest.json` from `waiting_for_human_label_freeze` to `frozen`.
- Do not use an LLM to produce the 90/40/20 human annotations.
- Every behavior/configuration change follows RED → minimal GREEN → refactor → verification.
- Check staged files before and after every `git add`.
- Do not push, create a PR, merge to `main`, force-push, delete a worktree, or clean branches without explicit approval.

---

## File Map

- Create `.gitattributes`: pin exact-byte, Git-tracked Task 2 evidence files to LF checkout bytes.
- Modify `tests/evaluation/test_prepare_data.py`: verify Git resolves `text` and `eol=lf` for manifest, split ID, and freeze-report paths.
- Modify `PRD.md`: mark only prepared 60/30/50 metadata complete and update remaining blocker descriptions.
- Modify `docs/TEAMMATE_ONBOARDING.md`: publish the annotation-work-package v1 identity and start notice while retaining the formal-freeze blocker.
- Do not modify the collaborator's `data/manifest.json`, six ID lists, adapter change, or adapter regression test.

---

### Task 1: Exact-Byte Git Checkout Contract

**Files:**
- Create: `.gitattributes`
- Modify: `tests/evaluation/test_prepare_data.py`
- Mechanically normalize without semantic diff: `data/manifest.json`, `data/splits/*.ids.json`

**Interfaces:**
- Consumes: Git attribute resolution for tracked data paths.
- Produces: LF worktree bytes identical to Git blobs for hashed manifest, split ID, and future safe freeze-report files.

- [ ] **Step 1: Add the Git-attribute regression test**

Add `import subprocess` beside the standard-library imports in `tests/evaluation/test_prepare_data.py`, then add:

```python
def test_repository_pins_hashed_data_checkout_bytes_to_lf() -> None:
    paths = (
        "data/manifest.json",
        "data/splits/dev.ids.json",
        "data/freeze_reports/example.json",
    )
    result = subprocess.run(
        ["git", "check-attr", "text", "eol", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "data/manifest.json: text: set",
        "data/manifest.json: eol: lf",
        "data/splits/dev.ids.json: text: set",
        "data/splits/dev.ids.json: eol: lf",
        "data/freeze_reports/example.json: text: set",
        "data/freeze_reports/example.json: eol: lf",
    ]
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/evaluation/test_prepare_data.py::test_repository_pins_hashed_data_checkout_bytes_to_lf -q
```

Expected: FAIL because the current repository has no `.gitattributes` and Git reports `text: unspecified` / `eol: unspecified`.

- [ ] **Step 3: Add the minimal LF contract**

Create `.gitattributes` with exactly:

```gitattributes
data/manifest.json text eol=lf
data/splits/*.json text eol=lf
data/freeze_reports/*.json text eol=lf
```

- [ ] **Step 4: Mechanically restore current tracked data files to LF bytes**

Run this only for the seven exact paths already delivered by the collaborator:

```powershell
$paths = @(
  'data/manifest.json',
  'data/splits/dev.ids.json',
  'data/splits/validation.ids.json',
  'data/splits/simulated_test.ids.json',
  'data/splits/type_domain_annotation.ids.json',
  'data/splits/constraint_annotation.ids.json',
  'data/splits/overlap_annotation.ids.json'
)
foreach ($path in $paths) {
  $absolute = Join-Path (Get-Location) $path
  $text = [IO.File]::ReadAllText($absolute)
  $normalized = $text.Replace("`r`n", "`n")
  [IO.File]::WriteAllText($absolute, $normalized, [Text.UTF8Encoding]::new($false))
}
```

Expected: `git diff --name-only` lists only `.gitattributes` and the test because the normalized data bytes equal their existing Git blobs.

- [ ] **Step 5: Run the new test and verify GREEN**

Run the Step 2 command again.

Expected: `1 passed`.

- [ ] **Step 6: Verify all prepared metadata hashes from worktree bytes**

Run a safe script that prints only aggregate results:

```powershell
@'
import hashlib
import json
from pathlib import Path

root = Path("data")
manifest = json.loads((root / "manifest.json").read_bytes())
for section in ("partitions", "work_packages"):
    for name, entry in manifest[section].items():
        payload = (root / entry["ids_path"]).read_bytes()
        actual = "sha256:" + hashlib.sha256(payload).hexdigest()
        assert actual == entry["ids_sha256"], name
print("all_manifest_id_hashes_match=true")
'@ | & 'D:\Dev\Python311\python.exe' -
```

Expected: `all_manifest_id_hashes_match=true`.

- [ ] **Step 7: Run Task 1 focused verification**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/evaluation/test_official_adapter.py `
  tests/evaluation/test_prepare_data.py `
  tests/evaluation/test_freeze.py -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check `
  tests/evaluation/test_prepare_data.py
```

Expected: both commands exit `0`.

- [ ] **Step 8: Stage-check and commit Task 1**

```powershell
git diff --cached --name-only
git add -- .gitattributes tests/evaluation/test_prepare_data.py
git diff --cached --name-only
git diff --cached --check
git commit -m "fix: preserve Task 2 evidence bytes"
```

Expected: exactly `.gitattributes` and `tests/evaluation/test_prepare_data.py` are committed. The seven metadata files remain byte-identical to their existing blobs and are not staged.

---

### Task 2: Truthful Week 1 Status and Annotation Start Notice

**Files:**
- Modify: `PRD.md`
- Modify: `docs/TEAMMATE_ONBOARDING.md`

**Interfaces:**
- Consumes: collaborator commit `d6adb6e`, PaSa revision `232428b0c867268c3b8ded90db4d98c1b30501d6`, prepared manifest blob SHA-256 `sha256:b5eaa1bc83d3a22b655c9f7e3e1ffa506176e49781cc3c2c618307d7c541ae21`.
- Produces: one unambiguous annotation-work-package v1 identity and a truthful PRD checklist state.

- [ ] **Step 1: Update the PRD prepared-data status**

In `PRD.md`:

1. Change the Task 2 current-status sentence to state that the fixed revision, manifest, 60/30/50 ID lists, and 90/40/20 work-package IDs are prepared and independently checked, while human labels and formal freeze remain pending.
2. Change only the 60/30/50 preparation checkbox at the current line 787 from `[ ]` to `[x]` and append evidence `d6adb6e`.
3. Keep the 90, 40, and 20 human-label checkboxes unchecked.
4. Keep baseline, formal freeze, fuzzy-dedup audit, and Week 1 gate unchecked; replace their outdated “协作者数据集尚未冻结” blocker with “准备 manifest/ID 已固定，真人标签、gold 与正式冻结证据仍缺失”.

- [ ] **Step 2: Update the onboarding header and current goal**

In `docs/TEAMMATE_ONBOARDING.md`:

- change `最后更新` to `2026-07-21`;
- replace “当前仓库没有正式 data/manifest.json” with a statement that a prepared manifest and safe ID lists exist, but the manifest remains `waiting_for_human_label_freeze` and cannot drive a formal baseline;
- retain every `.env`, private-data, human-label, and no-fake-freeze red line.

- [ ] **Step 3: Publish the exact annotation-work-package v1 notice**

Replace the “完成后等待主负责人明确发布” block with:

````markdown
### 主负责人通知（2026-07-21）

```text
Task 2 标注工作包 v1 已冻结（仅标注输入）
source commit: d6adb6e1f1ab12c40cf87315951de1cfe9742121
dataset revision: 232428b0c867268c3b8ded90db4d98c1b30501d6
prepared manifest sha256: sha256:b5eaa1bc83d3a22b655c9f7e3e1ffa506176e49781cc3c2c618307d7c541ae21
counts: dev=60, validation=30, simulated_test=50, type_domain=90, constraints=40, overlap=20
```

这条通知只固定标注输入并授权开始真人独立标注，不代表数据集已正式冻结。当前 manifest 必须继续保持 `waiting_for_human_label_freeze`；正式 `frozen` 仍需完整 gold、90/40/20 私密标签、kappa、逐分区 zero-answer policy 和主负责人显式批准。
````

- [ ] **Step 4: Update the immediate-action sequence**

In section 15, replace “等待通知” with instructions to verify the exact source commit, revision, manifest SHA-256, local private-source hashes, and then begin 90/40/20 independent human work. Keep “正式冻结批准前，不运行 baseline，不修改 manifest 状态”.

- [ ] **Step 5: Verify documentation completeness and contradictions**

```powershell
rg -n "d6adb6e|b5eaa1bc|标注工作包 v1 已冻结（仅标注输入）|waiting_for_human_label_freeze|90 条|40 条|20 条|0.80|zero_answer_policy|正式冻结批准前" `
  PRD.md docs/TEAMMATE_ONBOARDING.md
rg -n "当前仓库没有正式.*manifest|协作者数据集尚未冻结" `
  PRD.md docs/TEAMMATE_ONBOARDING.md
```

Expected: the first command finds every required identity, count, threshold, and blocker; the second command returns no matches.

- [ ] **Step 6: Stage-check and commit Task 2**

```powershell
git diff --cached --name-only
git add -- PRD.md docs/TEAMMATE_ONBOARDING.md
git diff --cached --name-only
git diff --cached --check
git commit -m "docs: launch Task 2 human annotation"
```

Expected: exactly the PRD and onboarding files are committed.

---

### Task 3: Fresh Checkout, Full Verification, and Independent Review

**Files:**
- Verify only: all Task 1–2 files and collaborator commit range `5b8800f..HEAD`.
- Modify only if a valid review finding first receives a failing regression test or is a documentation-only correction.

**Interfaces:**
- Consumes: committed LF contract, prepared metadata, truthful docs.
- Produces: review-clean local branch ready for an explicitly authorized push and human annotation handoff.

- [ ] **Step 1: Verify a fresh Git archive preserves exact hashes**

Export `HEAD` to a verified system-temp directory, recompute all six ID hashes against `data/manifest.json`, and delete only that verified temp directory. Do not load `.env` or inspect private data.

Expected: all manifest ID hashes match in the extracted snapshot.

- [ ] **Step 2: Run focused Week 1 verification**

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest `
  tests/evaluation/test_official_adapter.py `
  tests/evaluation/test_prepare_data.py `
  tests/evaluation/test_annotation.py `
  tests/evaluation/test_freeze.py `
  tests/evaluation/test_runner.py `
  tests/integration/test_week1_pipeline.py -q
```

Expected: exit `0`.

- [ ] **Step 3: Run full repository verification**

```powershell
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest -q
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file ruff check .
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file mypy src
git diff --check 5b8800f..HEAD
```

Expected: all commands exit `0`; only `tests/integration/test_openalex_live.py` may skip because the offline process has no `OPENALEX_API_KEY`.

- [ ] **Step 4: Run scope, protected-file, and secret audits**

```powershell
git diff --name-only 5b8800f..HEAD
git status --short --branch
git status --short -- docs/superpowers/specs/2026-07-15-task2-evaluation-design.md
git grep -n -I -E "OPENALEX_API_KEY=.+|HF_TOKEN=.+|Authorization:[[:space:]]*(Bearer|Basic)[[:space:]]+" `
  -- . ":(exclude).env*"
```

Expected: only collaborator delivery, approved design/plan, LF contract/test, PRD, and onboarding differ; protected-file status is empty; no credential value is found.

- [ ] **Step 5: Request independent review**

Reviewer scope:

```text
Review 5b8800f..HEAD against
docs/superpowers/specs/2026-07-21-week1-collaborator-integration-design.md.
Prioritize exact-byte checkout reproducibility, manifest/ID hash mismatch,
incorrect completion checkboxes, false frozen/gate claims, restricted-data leakage,
adapter deduplication semantics, and tests that can pass falsely.
Report Critical/Important/Minor findings and Ready Yes/No. Do not modify files,
read .env, or read the protected Task 2 design document.
```

- [ ] **Step 6: Resolve valid findings and re-run Steps 1–4**

For code/config behavior, first add a failing regression test and confirm RED. For documentation-only ambiguity, apply the smallest exact edit. Re-run fresh archive, focused/full tests, Ruff, mypy, diff, scope, protected-file, and secret checks after every accepted finding.

- [ ] **Step 7: Final handoff**

Report local commit SHAs, exact verification counts, the single permitted online skip if present, independent-review severity counts, current manifest status, and the human-only next actions. Do not push until the user explicitly authorizes it.

---

## Execution Checkpoints

1. Checkpoint A: Task 1 RED/GREEN, worktree-byte hashes, focused tests, exact two-file commit.
2. Checkpoint B: Task 2 truthful documentation verification and exact two-file commit.
3. Checkpoint C: fresh archive, full verification, scope/secret checks, independent review, final handoff.

At every checkpoint inspect `git status --short --branch`, staged files before and after staging, and the protected Task 2 design status. Never use passing synthetic tests or the annotation-work-package notification to claim the real dataset is formally frozen.
