# Task 2 Freeze Approval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, secret-safe audit and explicit one-way approval command that turns a fully verified Task 2 prepared manifest into the frozen manifest consumed by the existing Week 1 runner.

**Architecture:** `paper_search.evaluation.freeze` owns pure audit models/functions, explicit per-partition policy parsing, private-label validation, safe report assembly, and a guarded atomic manifest replacement. The CLI defaults to audit-only; `--approve` plus a confined report path authorizes the one-way state transition. The existing runner remains a read-only frozen-manifest consumer.

**Tech Stack:** Python 3.11, Pydantic 2, pathlib, argparse, hashlib, deterministic UTF-8 JSON/JSONL, pytest, Ruff, mypy strict.

## Global Constraints

- Never read, print, search, parse, or copy `.env` contents.
- Offline commands use `uv run --no-sync --no-env-file`; this prevents active `.env` loading but does not clear inherited process variables or guarantee network isolation.
- Do not touch `docs/superpowers/specs/2026-07-15-task2-evaluation-design.md`.
- Do not commit `data/manifest.json`, real PaSa data, real queries, gold, or human annotation files.
- Use synthetic fixtures only.
- Every production change follows RED → minimal GREEN → refactor → verification.
- Frozen state requires explicit policies for every manifest partition; no default policy is allowed.
- `status: frozen` in the current manifest is the only authoritative freeze state.
- Check staged files before and after every `git add`.
- Do not merge, create a PR, force-push, delete a worktree, or clean branches without explicit approval.

---

## File Map

- Modify `src/paper_search/evaluation/annotation.py`: add the minimal type/domain label model.
- Modify `src/paper_search/evaluation/__init__.py`: export stable public freeze/annotation contracts only if existing export style requires it.
- Create `src/paper_search/evaluation/freeze.py`: audit models, prepared-manifest validation, work-package/annotation validation, policy parsing, safe report generation, CLI, and guarded approval.
- Modify `tests/evaluation/test_annotation.py`: type/domain label Schema tests.
- Create `tests/evaluation/test_freeze.py`: all audit, policy, security, CLI, TOCTOU, and approval tests.
- Modify `tests/integration/test_week1_pipeline.py`: prove an approved synthetic manifest is accepted by the existing runner resolver.
- Modify `data/README.md`: document audit-only and approval commands, private file boundaries, and authoritative state.
- Modify `docs/TEAMMATE_ONBOARDING.md`: replace manual freeze wording with exact command workflow.

---

### Task 1: Minimal Type/Domain Annotation Contract

**Files:**
- Modify: `src/paper_search/evaluation/annotation.py`
- Test: `tests/evaluation/test_annotation.py`

**Interfaces:**
- Consumes: existing `QueryType`, `DomainLabel`, `DomainModel`, `NonEmptyStr`.
- Produces: `TypeDomainAnnotationRecord(query_id, query_type, domain, annotator)`.

- [ ] **Step 1: Write failing Schema tests**

Add tests proving the approved four fields are accepted, whitespace-only IDs/annotators fail, invalid query types/domains fail, records are frozen, and constraint-only or unknown fields are rejected:

```python
def test_type_domain_annotation_accepts_only_the_minimal_frozen_contract() -> None:
    record = TypeDomainAnnotationRecord.model_validate(
        {
            "query_id": "q1",
            "query_type": "method",
            "domain": "information-retrieval",
            "annotator": "member-b",
        }
    )
    assert record.query_id == "q1"
    with pytest.raises(ValidationError):
        record.domain = "changed"
    with pytest.raises(ValidationError):
        TypeDomainAnnotationRecord.model_validate(
            {**record.model_dump(), "research_goal": "not allowed"}
        )
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_annotation.py -k type_domain_annotation -v
```

Expected: collection fails because `TypeDomainAnnotationRecord` does not exist.

- [ ] **Step 3: Implement the minimal model**

Add next to `AnnotationRecord`:

```python
class TypeDomainAnnotationRecord(DomainModel):
    query_id: NonEmptyStr
    query_type: QueryType
    domain: DomainLabel
    annotator: NonEmptyStr
```

Do not add query text or constraint fields.

- [ ] **Step 4: Run Task 1 verification and confirm GREEN**

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_annotation.py -v
uv run --no-sync --no-env-file ruff check src/paper_search/evaluation/annotation.py tests/evaluation/test_annotation.py
uv run --no-sync --no-env-file mypy src/paper_search/evaluation/annotation.py
```

Expected: all commands exit `0`.

- [ ] **Step 5: Stage-check and commit Task 1**

```powershell
git diff --cached --name-only
git add -- src/paper_search/evaluation/annotation.py tests/evaluation/test_annotation.py
git diff --cached --name-only
git diff --cached --check
git commit -m "feat: add type-domain annotation contract"
```

---

### Task 2: Prepared Manifest, Partition, and Work-Package Audit

**Files:**
- Create: `src/paper_search/evaluation/freeze.py`
- Create: `tests/evaluation/test_freeze.py`

**Interfaces:**
- Consumes: `EvaluationQuery`, `read_jsonl()`, prepared manifest bytes, `data_root`, and explicit `Mapping[str, ZeroAnswerPolicy]`.
- Produces: `PartitionFreezeAudit`, `FreezeAuditReport`, and a read-only `FreezeAuditResult` without a final report path or frozen-manifest bytes.

- [ ] **Step 1: Write failing valid-candidate audit test**

Create a synthetic prepared tree with small counts but the same manifest shape as `prepare_task2_data.py`. Assert audit recomputes exact hashes and produces frozen partition fields without writing files:

```python
def test_audit_candidate_builds_safe_result_without_writing(tmp_path: Path) -> None:
    fixture = _prepared_tree(tmp_path)
    result = audit_freeze_candidate(
        data_root=fixture.data_root,
        type_domain_labels_path=fixture.type_domain_labels,
        constraint_labels_path=fixture.constraint_labels,
        overlap_labels_path=fixture.overlap_labels,
        policies={"dev": "reject", "validation": "reject", "simulated_test": "allow"},
    )
    assert result.report.partitions["dev"].labels_complete is True
    assert result.report.partitions["dev"].gold_sha256.startswith("sha256:")
    assert result.report.approval_requested is False
    assert (fixture.data_root / "manifest.json").read_bytes() == result.prepared_manifest_bytes
```

- [ ] **Step 2: Run the audit test and verify RED**

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_freeze.py::test_audit_candidate_builds_safe_result_without_writing -v
```

Expected: collection fails because `paper_search.evaluation.freeze` does not exist.

- [ ] **Step 3: Implement deterministic models and serializers**

Define:

```python
ZeroAnswerPolicy = Literal["reject", "allow"]


class PartitionFreezeAudit(DomainModel):
    count: PositiveInt
    gold_path: NonEmptyStr
    gold_sha256: NonEmptyStr
    ids_path: NonEmptyStr
    ids_sha256: NonEmptyStr
    zero_answer_policy: ZeroAnswerPolicy
    labels_complete: Literal[True]


class FreezeAuditReport(DomainModel):
    prepared_manifest_sha256: NonEmptyStr
    dataset_revision: NonEmptyStr
    source_file_count: PositiveInt
    type_domain_count: PositiveInt
    type_domain_sha256: NonEmptyStr
    constraint_count: PositiveInt
    constraint_sha256: NonEmptyStr
    overlap_count: PositiveInt
    overlap_sha256: NonEmptyStr
    agreement: AgreementReport
    partitions: dict[str, PartitionFreezeAudit]
    approval_requested: bool


@dataclass(frozen=True)
class FreezeAuditResult:
    prepared_manifest_bytes: bytes
    frozen_manifest_payload: dict[str, object]
    report: FreezeAuditReport


@dataclass(frozen=True)
class FreezeApprovalPlan:
    prepared_manifest_bytes: bytes
    frozen_manifest_bytes: bytes
    report_bytes: bytes
    report: FreezeAuditReport
```

Serialize JSON with UTF-8, sorted keys, indentation, `allow_nan=False`, and one trailing newline. Hashes use the existing `sha256:<hex>` convention.

- [ ] **Step 4: Implement prepared manifest and confined-path validation**

Require `status == "waiting_for_human_label_freeze"`, fixed nonempty identity fields, dictionary sections, and relative paths confined below `data_root`. Validate each raw source's byte count and SHA-256. Error messages exposed by CLI must be fixed and must not contain absolute paths.

- [ ] **Step 5: Implement partition validation**

For each manifest partition, validate positive count, nonempty gold, Schema-valid gold JSONL, unique nonempty ID JSON, exact count, ordered equality, exact hashes, and explicit zero-answer policy. If a prepared partition already declares a hash, reject mismatch rather than replacing it.

- [ ] **Step 6: Implement work-package identity validation**

Validate source/ID hashes from the manifest, then enforce:

```text
overlap IDs ⊆ constraint IDs ⊆ dev IDs
type/domain IDs == dev IDs ∪ validation IDs
```

Counts come from the prepared manifest so small synthetic fixtures remain possible; the production PaSa manifest still declares 90/40/20.

- [ ] **Step 7: Add rejection tests and verify GREEN**

Cover invalid status, identity field, path escape, missing file, raw hash/byte count, empty gold, count mismatch, duplicate IDs, ordered mismatch, existing hash mismatch, invalid subset, and source hash mismatch.

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_freeze.py -k "manifest or partition or work_package or path" -v
```

Expected: all selected tests pass.

- [ ] **Step 8: Run Task 2 static verification and commit**

```powershell
uv run --no-sync --no-env-file ruff check src/paper_search/evaluation/freeze.py tests/evaluation/test_freeze.py
uv run --no-sync --no-env-file mypy src/paper_search/evaluation/freeze.py
git diff --cached --name-only
git add -- src/paper_search/evaluation/freeze.py tests/evaluation/test_freeze.py
git diff --cached --name-only
git diff --cached --check
git commit -m "feat: audit Task 2 freeze candidates"
```

---

### Task 3: Human Labels, Agreement Gate, and Explicit Policies

**Files:**
- Modify: `src/paper_search/evaluation/freeze.py`
- Modify: `tests/evaluation/test_freeze.py`

**Interfaces:**
- Consumes: private type/domain, constraint, and overlap JSONL files; prepared work-package IDs; `compare_annotations()`.
- Produces: safe hashes/counts, `AgreementReport`, and a plan only when both categorical fields meet `0.80`.

- [ ] **Step 1: Write failing private-label alignment tests**

Test exact 90/40/20-equivalent fixture coverage, unique IDs, valid schemas, nonempty annotators, overlap extraction from the constraint file, and safe file hashes. Include failures for missing, duplicate, extra, and wrong-set IDs.

- [ ] **Step 2: Run and verify RED**

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_freeze.py -k "human_labels or annotation_alignment" -v
```

Expected: failures show the audit does not yet validate private labels.

- [ ] **Step 3: Implement private JSONL parsing and exact alignment**

Use `read_jsonl()` with `TypeDomainAnnotationRecord` and `AnnotationRecord`. Reject duplicate IDs before set comparison. Never put parsed content, annotator values, private paths, or mismatched IDs in report/error messages.

- [ ] **Step 4: Write and pass kappa gate tests**

Assert `query_type` and `domain` use the existing `compare_annotations(..., fields=("query_type", "domain"))`. Any `accepted is False` rejects the plan with the safe message `human annotation agreement is below threshold`.

- [ ] **Step 5: Write and pass explicit-policy parser tests**

Implement:

```python
def parse_zero_answer_policies(
    values: Sequence[str], partition_names: Collection[str]
) -> dict[str, ZeroAnswerPolicy]: ...
```

Reject missing, duplicate, unknown, malformed, or non-`reject`/`allow` entries. Do not provide a default.

- [ ] **Step 6: Verify the report is content-safe**

Use sentinel query text, paper ID, annotator, secret, and absolute private path. Assert none appears in `plan.report_bytes`, stdout-safe summary, or fixed error messages; only exact file hashes and counts may identify label inputs.

- [ ] **Step 7: Run Task 3 verification and commit**

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_freeze.py -k "label or agreement or policy or secret" -v
uv run --no-sync --no-env-file ruff check src/paper_search/evaluation/freeze.py tests/evaluation/test_freeze.py
uv run --no-sync --no-env-file mypy src/paper_search/evaluation/freeze.py
git diff --cached --name-only
git add -- src/paper_search/evaluation/freeze.py tests/evaluation/test_freeze.py
git diff --cached --name-only
git diff --cached --check
git commit -m "feat: gate Task 2 freeze on human evidence"
```

---

### Task 4: Audit-Only CLI

**Files:**
- Modify: `src/paper_search/evaluation/freeze.py`
- Modify: `src/paper_search/evaluation/__init__.py` only if public exports are consistent with the package style.
- Modify: `tests/evaluation/test_freeze.py`

**Interfaces:**
- Consumes: explicit file/path/policy CLI arguments.
- Produces: `main(argv: Sequence[str] | None = None) -> int`, safe JSON stdout on success, fixed stderr and exit `2` on expected failure.

- [ ] **Step 1: Write failing parser and audit-only tests**

Assert required options are exactly:

```text
--data-root
--type-domain-labels
--constraint-labels
--overlap-labels
--zero-answer-policy (repeatable)
--approve
--report
```

Without `--approve`, a valid invocation prints a safe report with `approval_requested: false`, returns `0`, and leaves manifest/report bytes unchanged or absent.

- [ ] **Step 2: Run and verify RED**

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_freeze.py -k "cli and audit" -v
```

- [ ] **Step 3: Implement the CLI boundary**

Use `argparse`. Read only explicit paths. Do not read environment variables. Catch only expected input/validation/I/O exceptions and print fixed messages such as:

```text
freeze audit failed: prepared data is invalid
freeze audit failed: private annotations are invalid
freeze audit failed: human annotation agreement is below threshold
freeze approval failed
```

Do not print raw exception repr.

- [ ] **Step 4: Enforce approval argument pairing**

Reject `--approve` without `--report` and `--report` without `--approve`. Confine approved reports below `<data-root>/freeze_reports/`. Audit-only accepts no report output.

- [ ] **Step 5: Run Task 4 verification and commit**

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_freeze.py -k cli -v
uv run --no-sync --no-env-file ruff check src/paper_search/evaluation/freeze.py tests/evaluation/test_freeze.py
uv run --no-sync --no-env-file mypy src/paper_search/evaluation/freeze.py
git diff --cached --name-only
git add -- src/paper_search/evaluation/freeze.py src/paper_search/evaluation/__init__.py tests/evaluation/test_freeze.py
git diff --cached --name-only
git diff --cached --check
git commit -m "feat: add Task 2 freeze audit CLI"
```

If `__init__.py` did not change, omit it from `git add`.

---

### Task 5: Guarded One-Way Approval and Runner Integration

**Files:**
- Modify: `src/paper_search/evaluation/freeze.py`
- Modify: `tests/evaluation/test_freeze.py`
- Modify: `tests/integration/test_week1_pipeline.py`

**Interfaces:**
- Consumes: `FreezeAuditResult`, exact current prepared manifest bytes, confined report path, explicit approval.
- Produces: complete safe report plus atomically replaced frozen manifest accepted by `_resolve_frozen_split()`.

- [ ] **Step 1: Write failing approval-plan and success tests**

Implement and test this explicit pure boundary before file writes:

```python
def build_approval_plan(
    audit: FreezeAuditResult, *, report_relative_path: str
) -> FreezeApprovalPlan: ...
```

Assert it changes the report to `approval_requested: true`, computes final report bytes/hash, and adds the confined report path/hash to exact frozen manifest bytes without mutating the audit result. Then assert approval writes deterministic report bytes first, replaces manifest with exact `plan.frozen_manifest_bytes`, includes prepared manifest hash, leaves no temp files, and runner resolution returns the expected frozen identity.

- [ ] **Step 2: Write failing preflight and TOCTOU tests**

Cover:

- current manifest bytes changed after audit;
- report path escapes `data/freeze_reports`;
- different existing report;
- already frozen manifest with different policies or label hashes;
- mutation immediately before manifest replacement;
- simulated report write failure;
- simulated manifest replacement failure;
- identical orphan report reuse;
- identical already-frozen invocation is idempotent.

- [ ] **Step 3: Run and verify RED**

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_freeze.py -k "approve or frozen or toctou or overwrite" -v
```

- [ ] **Step 4: Implement report preflight and atomic write**

Reuse `write_frozen_bytes()` for a new report or identical report. A different existing report raises `FileExistsError`. The report is complete but non-authoritative until manifest replacement succeeds.

- [ ] **Step 5: Implement guarded atomic manifest replacement**

Immediately before replacement, require current manifest bytes to equal `plan.prepared_manifest_bytes`. Write a same-directory temp file with exclusive creation, flush and `os.fsync()`, revalidate current bytes again, then call `os.replace()`. Always remove uncommitted temp files in `finally`.

This is a guarded atomic replace, not a cross-process filesystem transaction. Tests must prove all mutations observable at the injected pre-replace boundary are rejected.

- [ ] **Step 6: Implement already-frozen idempotency**

If current bytes equal `plan.frozen_manifest_bytes` and the exact report already exists, return success without rewriting. Any other frozen content is immutable and rejected.

- [ ] **Step 7: Add runner integration test and verify GREEN**

Approve a synthetic tree, then call the existing runner's frozen split resolver. Assert it accepts the generated manifest and preserves split, Git SHA, gold hash, manifest hash, revision, and policy.

Run:

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_freeze.py tests/integration/test_week1_pipeline.py -v
```

- [ ] **Step 8: Run Task 5 static verification and commit**

```powershell
uv run --no-sync --no-env-file ruff check src/paper_search/evaluation/freeze.py tests/evaluation/test_freeze.py tests/integration/test_week1_pipeline.py
uv run --no-sync --no-env-file mypy src/paper_search/evaluation/freeze.py
git diff --cached --name-only
git add -- src/paper_search/evaluation/freeze.py tests/evaluation/test_freeze.py tests/integration/test_week1_pipeline.py
git diff --cached --name-only
git diff --cached --check
git commit -m "feat: approve Task 2 frozen manifests"
```

---

### Task 6: Operator Documentation and Final Evidence

**Files:**
- Modify: `data/README.md`
- Modify: `docs/TEAMMATE_ONBOARDING.md`
- Modify only if final review finds a valid defect: Task 1–5 implementation/test files.

**Interfaces:**
- Consumes: completed CLI and safety contract.
- Produces: exact operator commands, private-data boundaries, truthful project state, and a review-clean branch.

- [ ] **Step 1: Update data operator documentation**

Document audit-only first, explicit per-partition policies, private label path handling, `--approve`/`--report`, frozen manifest authority, orphan complete report semantics, idempotency, and prohibited Git content. Never include a real private path or credential value.

- [ ] **Step 2: Update collaborator onboarding**

Replace manual manifest editing with the exact audit and approval commands. State that only the main responsible person runs `--approve`; collaborator supplies private files and verifies safe hashes/counts.

- [ ] **Step 3: Run focused verification**

```powershell
uv run --no-sync --no-env-file pytest tests/evaluation/test_annotation.py tests/evaluation/test_freeze.py tests/evaluation/test_runner.py tests/integration/test_week1_pipeline.py -v
```

- [ ] **Step 4: Run full repository verification**

```powershell
uv run --no-sync --no-env-file pytest -q
uv run --no-sync --no-env-file ruff check .
uv run --no-sync --no-env-file mypy src
git diff --check 8289fed...HEAD
```

Expected: all commands exit `0`; only the explicitly online OpenAlex test may skip when the process lacks a key.

- [ ] **Step 5: Run scope and secret audits**

```powershell
git diff --name-only 8289fed...HEAD
git status --short
git grep -n -I -E "OPENALEX_API_KEY=.+|HF_TOKEN=.+|Authorization:[[:space:]]*(Bearer|Basic)[[:space:]]+" -- . ":(exclude).env*"
```

Expected: only approved Task 2 freeze files and truthful docs appear; the protected Task 2 design is absent; no secret-bearing value is found.

- [ ] **Step 6: Request independent review**

Reviewer scope:

```text
Review 8289fed...HEAD against
docs/superpowers/specs/2026-07-17-task2-freeze-approval-design.md.
Prioritize false freeze, label/ID misalignment, unsafe policy defaults, path escape,
hash mismatch, overwrite/TOCTOU, partial or misleading evidence, private-content
leakage, CLI error leakage, runner compatibility, and tests that can pass falsely.
Report Critical/Important/Minor findings with file and line references. Do not modify.
```

- [ ] **Step 7: Resolve every valid finding with TDD**

For each finding, first add a failing test, run it to prove RED, implement the smallest fix, rerun focused tests for GREEN, then repeat Steps 3–5.

- [ ] **Step 8: Stage-check and commit documentation/evidence**

```powershell
git diff --cached --name-only
git add -- data/README.md docs/TEAMMATE_ONBOARDING.md
git diff --cached --name-only
git diff --cached --check
git commit -m "docs: document Task 2 freeze approval"
```

If review fixes exist, stage and commit only their exact implementation/test files separately before the documentation commit.

---

## Execution Order and Checkpoints

Execute Tasks 1–2 as checkpoint A, Tasks 3–4 as checkpoint B, Task 5 as checkpoint C, and Task 6 as the final verification/review checkpoint. At every checkpoint:

1. inspect `git status --short`;
2. inspect staged files before and after staging;
3. confirm the protected Task 2 design is absent;
4. report exact RED and GREEN commands/results;
5. never use synthetic passing evidence to claim real human data is frozen.

Do not run the real approval command in this implementation plan. The delivered CLI is exercised only on synthetic fixtures; real approval waits for the collaborator's private 90/40/20 files and the main responsible person's explicit policy choices.
