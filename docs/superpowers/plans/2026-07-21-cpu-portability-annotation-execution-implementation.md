# CPU-First Portability and Annotation Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository reproducible on Windows/Linux CPU machines while retaining an explicit CUDA extra, and give both human annotators a secret-safe pre-freeze validation workflow.

**Architecture:** Keep one uv lock with CPU torch as the no-extra default and a conditional `cuda` extra. Split health into blocking CPU/core evidence and non-blocking accelerator evidence. Keep private JSONL outside Git; validate it with a standalone CLI that emits only counts, exact hashes, and generic status, while formal cross-rater agreement remains in the freeze audit.

**Tech Stack:** Python 3.11, uv 0.11, PyTorch 2.5.1, Pydantic 2, pytest, Ruff, mypy, UTF-8 JSON/JSONL.

## Global Constraints

- Never read, print, search, parse, or copy `.env` contents.
- Offline tests use `uv run --no-sync --no-env-file`.
- Do not read, modify, format, stage, or commit `docs/superpowers/specs/2026-07-15-task2-evaluation-design.md`.
- Do not modify `data/manifest.json`, the six prepared ID lists, gold, sampling rules, or dataset revision.
- Do not commit raw PaSa data, real queries, private annotation files, screenshots, or logs containing restricted text.
- Do not generate, complete, or rewrite human labels with an LLM.
- Keep manifest status `waiting_for_human_label_freeze`; do not run a formal online baseline.
- Windows/Linux Python 3.11 CPU is required; CUDA is optional; ROCm is not claimed.
- Use strict RED → minimal GREEN → refactor → verification for every behavior/configuration change.
- Inspect staged files before and after every `git add`.
- Do not push, create a PR, merge, force-push, delete a worktree, or clean branches without explicit authorization.

---

## File Map

- Modify `pyproject.toml`: CPU default torch source, conditional CUDA extra, hardware marker.
- Modify `uv.lock`: one reproducible CPU-default/CUDA-extra resolution.
- Create `tests/test_packaging.py`: assert dependency/source contract without installing packages.
- Modify `src/paper_search/health.py`: CPU/core and optional accelerator health semantics.
- Rewrite `tests/test_health.py`: deterministic hardware-independent tests.
- Create `data/domain_labels.v1.json`: safe versioned controlled vocabulary.
- Modify `src/paper_search/evaluation/annotation.py`: controlled domain type and private validation CLI.
- Modify `src/paper_search/evaluation/__init__.py`: export only stable annotation validation types/functions.
- Modify `tests/evaluation/test_annotation.py`: vocabulary, private validation, no-leak CLI tests.
- Create `README.md`: CPU quickstart and optional CUDA verification.
- Modify `docs/TEAMMATE_ONBOARDING.md`: fresh CPU setup, 90/40/20 exact workflow, validator commands.
- Modify `data/README.md`: replace stale wait instruction with the published input-only notice and validation flow.

---

### Task 1: CPU Default and Explicit CUDA Extra

**Files:**
- Create: `tests/test_packaging.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: uv 0.11 conditional source resolution.
- Produces: no-extra CPU torch resolution; `--extra cuda` CUDA torch resolution; `hardware` pytest marker.

- [ ] **Step 1: Add the dependency-contract test**

Create `tests/test_packaging.py`:

```python
from __future__ import annotations

import tomllib
from pathlib import Path


def test_torch_defaults_to_cpu_and_cuda_is_explicit() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional = project["project"]["optional-dependencies"]
    sources = project["tool"]["uv"]["sources"]["torch"]
    indexes = {item["name"]: item["url"] for item in project["tool"]["uv"]["index"]}

    assert optional["cuda"] == ["torch==2.5.1"]
    assert sources == [
        {"index": "pytorch-cu121", "extra": "cuda"},
        {"index": "pytorch-cpu"},
    ]
    assert indexes == {
        "pytorch-cpu": "https://download.pytorch.org/whl/cpu",
        "pytorch-cu121": "https://download.pytorch.org/whl/cu121",
    }


def test_hardware_tests_are_explicitly_marked() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "hardware: requires an explicitly selected accelerator environment" in (
        project["tool"]["pytest"]["ini_options"]["markers"]
    )
```

- [ ] **Step 2: Run RED**

Run:

```powershell
$env:UV_PROJECT_ENVIRONMENT='D:\AI Projects\Projects\.venv'
& 'D:\Dev\uv\uv.exe' run --no-sync --no-env-file pytest tests/test_packaging.py -q
```

Expected: FAIL because there is no `cuda` optional dependency, torch has one unconditional CUDA source, and the hardware marker is absent.

- [ ] **Step 3: Apply the minimal pyproject configuration**

Add:

```toml
[project.optional-dependencies]
cuda = [
    "torch==2.5.1",
]
```

Replace the torch source/index section with:

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu121", extra = "cuda" },
    { index = "pytorch-cpu" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

Add the exact pytest marker string from Step 1.

- [ ] **Step 4: Verify GREEN before locking**

Run the Step 2 command. Expected: `2 passed`.

- [ ] **Step 5: Regenerate and inspect the single lock**

Run:

```powershell
& 'D:\Dev\uv\uv.exe' lock
& 'D:\Dev\uv\uv.exe' lock --check
& 'D:\Dev\uv\uv.exe' sync --locked --dry-run
& 'D:\Dev\uv\uv.exe' sync --locked --extra cuda --dry-run
```

Expected: all commands exit `0`; default dry-run selects CPU torch; CUDA dry-run selects `2.5.1+cu121`; neither command proposes both torch variants in one environment. If the conditional source syntax is rejected or one resolution contains both variants, stop Task 1 and do not commit a fallback configuration.

- [ ] **Step 6: Verify the CPU wheel identities safely**

Run a script that parses `uv.lock` and prints only torch source/version plus wheel platform filenames. Expected: CPU source is present for Windows and Linux, CUDA source is tied to the `cuda` resolution, and no credentials appear.

- [ ] **Step 7: Focused verification and commit**

Run the packaging tests and `ruff check tests/test_packaging.py`. Check staged files before/after; stage exactly `pyproject.toml`, `uv.lock`, and `tests/test_packaging.py`; run `git diff --cached --check`; commit:

```text
fix: make CPU the default torch profile
```

---

### Task 2: Device-Independent Health Contract

**Files:**
- Modify: `src/paper_search/health.py`
- Modify: `tests/test_health.py`

**Interfaces:**
- Produces: `collect_local_health(matrix_size: int = 64, require_accelerator: str | None = None) -> dict[str, Any]`.
- CLI: `python -m paper_search.health [--require-accelerator cuda]`.

- [ ] **Step 1: Replace live-machine assertions with RED contract tests**

Tests must cover:

```python
def test_cpu_only_runtime_is_ready(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = health.collect_local_health(matrix_size=4)
    assert report["status"] == "ready"
    assert report["core"]["matrix_smoke"]["finite"] is True
    assert report["accelerator"]["status"] == "unavailable"
    assert report["errors"] == []


def test_required_cuda_is_blocking_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = health.collect_local_health(matrix_size=4, require_accelerator="cuda")
    assert report["status"] == "degraded"
    assert report["errors"] == ["accelerator_required:cuda"]
```

Add deterministic fake-runtime tests for CUDA available/smoke failure and CLI tests for default exit `0`, required-CUDA exit `1`, invalid arguments, and secret-free output. Remove assertions for `2.5.1+cu121`, `12.1`, and `RTX 3050 Ti`.

- [ ] **Step 2: Run RED**

Run `pytest tests/test_health.py -q`. Expected: CPU-only test fails because current code records `cuda_unavailable` as blocking and report lacks `core`/`accelerator`.

- [ ] **Step 3: Implement minimal layered reporting**

Use private helpers `_cpu_smoke(torch_module, matrix_size)` and `_cuda_smoke(torch_module, matrix_size)`. Always run the CPU smoke. Populate:

```python
{
    "status": "ready" | "degraded",
    "python": {...},
    "core": {
        "torch_version": str,
        "matrix_smoke": {"shape": [n, n], "finite": bool, "checksum": float},
        "dependencies": {...},
    },
    "accelerator": {
        "backend": "cuda",
        "status": "available" | "unavailable" | "error",
        "build": str | None,
        "device": str | None,
        "matrix_smoke": dict[str, object] | None,
    },
    "errors": list[str],
}
```

Only CPU/dependency failures and an explicitly required unavailable/error CUDA backend enter `errors`.

- [ ] **Step 4: Add argparse CLI**

`main(argv: Sequence[str] | None = None) -> int` accepts `--require-accelerator cuda`, prints sorted JSON, and returns nonzero only for `degraded`.

- [ ] **Step 5: Run GREEN and static checks**

Run health tests, full `tests/test_health.py` under default CPU semantics, Ruff on both files, and mypy on `health.py`. Expected: all exit `0` without using real CUDA.

- [ ] **Step 6: Commit**

Stage exactly the health implementation/test; commit:

```text
fix: make accelerator health optional
```

---

### Task 3: Frozen Domain Vocabulary

**Files:**
- Create: `data/domain_labels.v1.json`
- Modify: `src/paper_search/evaluation/annotation.py`
- Modify: `tests/evaluation/test_annotation.py`

**Interfaces:**
- Produces: `DOMAIN_LABELS: tuple[str, ...]` and a Pydantic domain value restricted to the exact v1 list.

- [ ] **Step 1: Add RED tests**

Assert the safe JSON artifact has `version == "domain-labels-v1"`, exact ordered labels from the design, unique entries, and definitions for every entry. Add model tests proving `information-retrieval` and `other` pass while `search-systems` and uppercase/spaced values fail.

- [ ] **Step 2: Run RED**

Run `pytest tests/evaluation/test_annotation.py -k domain -q`. Expected: FAIL because the artifact and controlled membership validation do not exist.

- [ ] **Step 3: Add the safe artifact**

Create a UTF-8 JSON file containing only:

```json
{
  "version": "domain-labels-v1",
  "labels": ["artificial-intelligence", "machine-learning", "natural-language-processing", "information-retrieval", "computer-vision", "speech-audio", "robotics", "data-mining", "knowledge-graphs", "recommender-systems", "human-computer-interaction", "software-engineering", "computer-systems", "networks-security", "databases", "theory-algorithms", "computational-biology", "computational-social-science", "scientific-computing", "multidisciplinary", "other"],
  "definitions": {
    "artificial-intelligence": "General AI topics not more precisely covered by another label",
    "machine-learning": "Learning algorithms, training, optimization, and generalization",
    "natural-language-processing": "Computational methods for human language",
    "information-retrieval": "Search, ranking, indexing, and retrieval evaluation",
    "computer-vision": "Image and video understanding or generation",
    "speech-audio": "Speech, music, and general audio processing",
    "robotics": "Embodied agents, control, navigation, and manipulation",
    "data-mining": "Pattern discovery and analytical mining of structured data",
    "knowledge-graphs": "Structured knowledge representation and graph reasoning",
    "recommender-systems": "Personalized ranking and recommendation",
    "human-computer-interaction": "Human-facing interaction, usability, and interface research",
    "software-engineering": "Software construction, testing, maintenance, and developer tools",
    "computer-systems": "Operating, distributed, cloud, and high-performance systems",
    "networks-security": "Computer networks, privacy, and security",
    "databases": "Database systems, transactions, and data management",
    "theory-algorithms": "Computational theory, complexity, and algorithms",
    "computational-biology": "Computational biology, medicine, and bioinformatics",
    "computational-social-science": "Computational study of social and behavioral phenomena",
    "scientific-computing": "Numerical and computational methods for physical sciences",
    "multidisciplinary": "Multiple domains are equally central",
    "other": "No frozen label describes the primary domain"
  }
}
```

- [ ] **Step 4: Enforce membership in Pydantic models**

Keep kebab-case validation and add an `AfterValidator` that rejects values not in `DOMAIN_LABELS`. Do not read the JSON artifact at import time; the test binds the safe artifact to the code constant.

- [ ] **Step 5: Run GREEN and commit**

Run annotation tests, Ruff, mypy; stage exactly three files; commit:

```text
feat: freeze Task 2 domain vocabulary
```

---

### Task 4: Secret-Safe Annotation Validation CLI

**Files:**
- Modify: `src/paper_search/evaluation/annotation.py`
- Modify: `src/paper_search/evaluation/__init__.py`
- Modify: `tests/evaluation/test_annotation.py`

**Interfaces:**
- Produces: `validate_annotation_file(labels_path: Path, ids_path: Path, *, kind: Literal["type-domain", "constraints"]) -> AnnotationValidationSummary`.
- CLI: `python -m paper_search.evaluation.annotation --kind <kind> --labels <private.jsonl> --ids <safe.ids.json>`.

- [ ] **Step 1: Add RED unit and CLI tests**

Cover valid type-domain and constraint files; invalid UTF-8; blank/malformed lines; unknown fields; duplicate/missing/extra IDs; wrong record kind; controlled-domain failure; private filename/query/annotator/sentinel not present in stdout/stderr. Expected summary:

```json
{"count":90,"ids_match":true,"kind":"type-domain","sha256":"sha256:<64 lowercase hex>","status":"valid"}
```

Invalid input returns exit `1` and exactly `annotation validation failed` on stderr, with no stdout.

- [ ] **Step 2: Run RED**

Run the new tests. Expected: FAIL because the summary, validator, and CLI do not exist.

- [ ] **Step 3: Implement the validator**

Read exact bytes once, decode UTF-8, validate one JSON object per non-empty line using `TypeDomainAnnotationRecord` or `AnnotationRecord`, reject duplicates, load the safe ID JSON list, and require equal set and length. Collapse `OSError`, decode, JSON, Pydantic, duplicate, and ID errors to `ValueError("private annotations are invalid")`.

`AnnotationValidationSummary` contains only `status`, `kind`, `count`, `sha256`, and `ids_match`; it must never include a path or record value.

- [ ] **Step 4: Implement the CLI**

Use argparse with required `--kind`, `--labels`, and `--ids`. Print `model_dump_json()` only on success; print the fixed generic error on failure. Do not catch `SystemExit` from invalid CLI syntax.

- [ ] **Step 5: Run GREEN and regression checks**

Run all annotation tests, freeze tests that load private evidence, Ruff, and mypy. Expected: all exit `0`; freeze behavior remains unchanged.

- [ ] **Step 6: Commit**

Stage exactly annotation implementation/export/test; commit:

```text
feat: validate private annotations safely
```

---

### Task 5: Third-Party and Human-Annotation Documentation

**Files:**
- Create: `README.md`
- Modify: `docs/TEAMMATE_ONBOARDING.md`
- Modify: `data/README.md`
- Modify: `tests/test_packaging.py`

**Interfaces:**
- Consumes: CPU profile, health CLI, controlled vocabulary, annotation validation CLI.
- Produces: copyable CPU and CUDA commands plus exact 90/40/20 responsibilities.

- [ ] **Step 1: Add RED documentation-contract tests**

Assert README/onboarding contain:

- CPU default install command without `--all-groups`;
- optional `--extra cuda` command;
- default and required-CUDA health commands;
- `--no-env-file` offline validation;
- three annotation validator invocations;
- fresh clone/worktree requirement;
- 90/40/20 roles, stable distinct annotators, `domain-labels-v1`, hash-first handoff, kappa `0.80`;
- explicit statement that the notification is input-only and manifest remains `waiting_for_human_label_freeze`.

- [ ] **Step 2: Run RED**

Run the documentation-contract tests. Expected: FAIL because README is absent and onboarding still uses `uv sync --all-groups`.

- [ ] **Step 3: Write README CPU quickstart**

Document fresh clone, Python 3.11, `uv sync --locked`, default health, offline test commands, expected OpenAlex skip, and prepared hash verification. Put CUDA under an explicitly optional section using `uv sync --locked --extra cuda` and `--require-accelerator cuda`.

- [ ] **Step 4: Update collaborator docs**

Replace `uv sync --all-groups` with the CPU default. Add these exact commands; all three label files remain under the ignored local directory:

```powershell
uv run --no-sync --no-env-file python -m paper_search.evaluation.annotation `
  --kind type-domain `
  --labels data/annotation_work/type_domain_labels.jsonl `
  --ids data/splits/type_domain_annotation.ids.json
uv run --no-sync --no-env-file python -m paper_search.evaluation.annotation `
  --kind constraints `
  --labels data/annotation_work/constraint_labels.jsonl `
  --ids data/splits/constraint_annotation.ids.json
uv run --no-sync --no-env-file python -m paper_search.evaluation.annotation `
  --kind constraints `
  --labels data/annotation_work/overlap_labels.jsonl `
  --ids data/splits/overlap_annotation.ids.json
```

Explain that the collaborator produces the 90- and 40-record files, the main owner produces the independent 20-record file, and the collaborator's overlap ratings are the corresponding rows inside the 40-record file.

Update `data/README.md` to reflect the already published input-only v1 notification instead of instructing the collaborator to wait for it.

- [ ] **Step 5: Run GREEN and commit**

Run documentation-contract tests, `git diff --check`, and the scoped secret scan. Stage exactly README, onboarding, data README, and packaging test; commit:

```text
docs: document portable annotation workflow
```

---

### Task 6: Fresh CPU Verification and Independent Review

**Files:**
- Verify only; modify only for a validated finding using a new RED test first.

- [ ] **Step 1: Verify dependency resolutions**

Run `uv lock --check`, CPU dry-run, CUDA-extra dry-run, and safe torch resolution summaries. Default must resolve CPU; CUDA extra must resolve CUDA; neither may contain both variants.

- [ ] **Step 2: Verify a fresh CPU environment**

Create a verified temporary environment outside the project worktree, sync from the lock without `.env`, run the health CLI, import retrieval dependencies, and delete only the verified temporary target. The health JSON must be `ready` even when CUDA is unavailable.

- [ ] **Step 3: Run focused tests**

Run packaging, health, annotation, prepare-data, and freeze tests with `--no-env-file`.

- [ ] **Step 4: Run full checks**

Run full offline pytest, Ruff, mypy, and `git diff --check 5b8800f..HEAD`. Only the explicit online OpenAlex test may skip for a missing process key; hardware tests are not part of default pytest.

- [ ] **Step 5: Run scope and safety audits**

List `5b8800f..HEAD`, confirm protected-file status/diff empty, confirm staged empty, verify prepared manifest/six ID hashes, and scan tracked executable/data/ordinary docs while excluding `.env*`, all superpowers plans, and all superpowers specs.

- [ ] **Step 6: Independent review**

Review `65e20ae..HEAD` against `docs/superpowers/specs/2026-07-21-cpu-portability-annotation-execution-design.md`. Prioritize CPU/CUDA resolution truthfulness, hardware-independent tests, secret leakage, annotation ID/schema false passes, domain vocabulary drift, false human-label/frozen claims, and third-party commands. Report Critical/Important/Minor and Ready Yes/No.

- [ ] **Step 7: Resolve findings and hand off locally**

For every valid behavior/config finding, add a failing regression test and confirm RED before the minimal fix; rerun Steps 1–5 and re-review. Report commit SHAs and fresh evidence. Keep the branch/worktree local until the user explicitly authorizes push/PR/merge.

## Execution Checkpoints

1. Checkpoint A: Task 1 dependency contract, lock, CPU/CUDA dry-runs, exact commit.
2. Checkpoint B: Task 2 health RED/GREEN and device-independent commit.
3. Checkpoint C: Tasks 3–4 domain/validator RED/GREEN and exact commits.
4. Checkpoint D: Task 5 documentation contract and exact commit.
5. Checkpoint E: fresh CPU verification, full checks, safety audits, independent review.
