# Query Evolution Prompt v2 Novelty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the locked Query Evolution prompt to v2 so DeepSeek is explicitly instructed not to repeat original/seed queries and to return `no_novel_query` when no grounded novel query exists.

**Architecture:** Make the artifact loader and new preflight path v2-only while allowing the lock data model to parse historical v1 evidence. Preserve the existing `QueryEvolutionProposal` schema and `_canonical_query()` validator; use unit/integration tests to prove prompt rendering, new-lock v2 identity, stale-v1 rejection, duplicate rejection, and pre-dispatch failure.

**Tech Stack:** Python 3.11, Pydantic v2, PyYAML, pytest, Ruff, mypy, PowerShell.

## Global Constraints

- Follow the approved design in `docs/superpowers/specs/2026-08-10-query-evolution-prompt-v2-novelty-design.md`.
- Do not modify `QueryEvolutionProposal`, `_canonical_query()`, retry, timeout, budget, ledger, snapshot, selection, or promotion behavior.
- Do not add automatic deduplication, repair calls, fallback, or silent conversion of duplicates to no-op.
- Do not read `.env`, construct a real provider client, send network requests, create repository lock files, or run live canary/full probe.
- Do not modify `runs/_diag_query_evolution_contract-canary-20260810/`, `data/budget_ledger.sqlite3`, or `deliverables/`.
- Use `D:\AI Projects\Projects\.venv\Scripts\python.exe` for Python checks.
- Use TDD for the prompt/version behavior: observe the focused tests fail before changing production/config files.

## File Structure

- Modify `configs/prompts/query_evolve.yaml`: v2 identity and the four novelty/no-op instructions.
- Modify `src/paper_search/llm/prompt_artifacts.py`: require `query-evolve-v2` for `query_evolve` artifacts.
- Modify `scripts/probe_query_evolution.py`: allow `ProbePromptBinding` to parse historical v1 and current v2 while new preflight remains v2-only through the artifact loader.
- Modify `tests/unit/test_prompt_artifacts.py`: exact rendered-message and v1-rejection coverage.
- Modify `tests/unit/test_query_evolution.py`: characterize original/seed/same-batch duplicate rejection.
- Modify `tests/integration/test_query_evolution_probe.py`: v2 lock fixtures and pre-dispatch version/hash drift guards.

---

### Task 1: Bind and render the prompt v2 contract

**Files:**
- Modify: `configs/prompts/query_evolve.yaml:1-20`
- Modify: `src/paper_search/llm/prompt_artifacts.py:25-31`
- Modify: `scripts/probe_query_evolution.py:104-108`
- Test: `tests/unit/test_prompt_artifacts.py:32-107`
- Test: `tests/integration/test_query_evolution_probe.py:38-64,400-424,548-569`

**Interfaces:**
- Consumes: `load_prompt_artifact(prompt_bytes: bytes) -> PromptArtifact`, `render_prompt_system_message(prompt_bytes: bytes) -> str`, and `ProbePromptBinding`.
- Produces: a `query-evolve-v2` artifact/new lock binding, historical v1 lock readability, and the existing deterministic system-message renderer.

- [ ] **Step 1: Record protected local-state fingerprints**

Run:

```powershell
Get-FileHash -LiteralPath data/budget_ledger.sqlite3 -Algorithm SHA256
git status --short
```

Expected ledger SHA-256:

```text
628571402F58F9DD790CC00DD30173652D4102C2E591791A5A264219B80D4293
```

Expected status contains only user-owned untracked `data/budget_ledger.sqlite3` and `deliverables/`; stop if any protected path has an unexpected tracked change.

- [ ] **Step 2: Write the failing v2 prompt and lock-binding tests**

In `tests/unit/test_prompt_artifacts.py`, add this focused identity test:

```python
def test_query_evolve_artifact_requires_v2() -> None:
    prompt_bytes = Path("configs/prompts/query_evolve.yaml").read_bytes()
    artifact = load_prompt_artifact(prompt_bytes)

    assert artifact.version == "query-evolve-v2"
    with pytest.raises(ValidationError, match="query_evolve prompt version must be query-evolve-v2"):
        load_prompt_artifact(
            prompt_bytes.replace(b"version: query-evolve-v2", b"version: query-evolve-v1")
        )
```

Update the exact `evolve_message` expectation in `test_render_prompt_system_message_is_deterministic_for_existing_prompt_files()` by inserting these four rendered lines immediately after the existing zero-to-two instruction:

```python
"- Before returning, verify that each generated text is novel after canonicalization against original_query, every seed_subqueries text, and earlier generated subqueries.",
"- Case, Unicode, whitespace, or punctuation-only changes do not make a query novel.",
"- Return only valid novel subqueries; one valid subquery is allowed and duplicate candidates must be omitted.",
'- No-novel form: {"subqueries":[],"no_op_reason":"no_novel_query"}',
```

In `tests/integration/test_query_evolution_probe.py`:

```python
# _synthetic_probe_lock(...)
version="query-evolve-v2",

# test_preflight_stores_caller_supplied_prompt_binding
assert lock.prompt.version == "query-evolve-v2"

# test_preflight_rejects_prompt_version_drift
DEFAULT_PROMPT_CONFIG.read_text(encoding="utf-8").replace(
    "version: query-evolve-v2",
    "version: query-evolve-v1",
)
```

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_prompt_artifacts.py::test_query_evolve_artifact_requires_v2 tests/unit/test_prompt_artifacts.py::test_render_prompt_system_message_is_deterministic_for_existing_prompt_files tests/integration/test_query_evolution_probe.py::test_preflight_stores_caller_supplied_prompt_binding -q
```

Expected: FAIL because the current YAML, artifact validator, and `ProbePromptBinding` still require `query-evolve-v1`. If the tests pass, stop and correct the tests before implementation.

- [ ] **Step 3: Implement the minimal v2 artifact and version guards**

In `configs/prompts/query_evolve.yaml`, change the version and insert the same four instructions after the existing zero-to-two instruction:

```yaml
version: query-evolve-v2
```

```yaml
  - Before returning, verify that each generated text is novel after canonicalization against original_query, every seed_subqueries text, and earlier generated subqueries.
  - Case, Unicode, whitespace, or punctuation-only changes do not make a query novel.
  - Return only valid novel subqueries; one valid subquery is allowed and duplicate candidates must be omitted.
  - 'No-novel form: {"subqueries":[],"no_op_reason":"no_novel_query"}'
```

In `src/paper_search/llm/prompt_artifacts.py`, replace only the query-evolve version check:

```python
if self.version != "query-evolve-v2":
    raise ValueError(
        "query_evolve prompt version must be query-evolve-v2"
    )
```

In `scripts/probe_query_evolution.py`, expand only the version literal so historical locks remain readable:

```python
class ProbePromptBinding(DomainModel):
    path: SafeRelativePath
    sha256: Sha256
    name: Literal["query_evolve"]
    version: Literal["query-evolve-v1", "query-evolve-v2"]
```

Do not change `ProbeLock`/`CanaryLock` schema versions or any runner logic.

- [ ] **Step 4: Verify GREEN for prompt rendering and lock binding**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_prompt_artifacts.py tests/integration/test_query_evolution_probe.py -k "prompt or binding" -q
```

Expected: PASS; the rendered system message is exact, preflight stores v2, and a v1 copy is rejected.

- [ ] **Step 5: Commit the v2 contract**

Run:

```powershell
git add -- configs/prompts/query_evolve.yaml src/paper_search/llm/prompt_artifacts.py scripts/probe_query_evolution.py tests/unit/test_prompt_artifacts.py tests/integration/test_query_evolution_probe.py
git diff --cached --check
git diff --cached --name-only
git commit -m "fix: require query evolution prompt v2"
```

Expected staged files: exactly the five paths listed above. Stop if protected evidence, ledger, or `deliverables/` appears.

---

### Task 2: Prove novelty invariants and pre-dispatch stopping

**Files:**
- Test: `tests/unit/test_query_evolution.py:343-436`
- Test: `tests/integration/test_query_evolution_probe.py:572-613,1061-1127`

**Interfaces:**
- Consumes: existing `validate_query_evolution_proposal(raw, context)`, `run_probe(...)`, and `run_canary(...)` behavior.
- Produces: characterization evidence that original/seed/same-batch duplicates remain integrity failures and stale prompt bindings cannot reach reservation or dispatch.

- [ ] **Step 1: Add original and seed duplicate characterization tests**

Add beside `test_validate_query_evolution_proposal_rejects_mechanical_violations`:

```python
@pytest.mark.parametrize(
    "text",
    [
        "ＧＮＮｓ for node classification",
        "graph neural networks",
    ],
    ids=["original-query-after-nfkc", "seed-query"],
)
def test_validate_query_evolution_proposal_rejects_existing_queries(
    text: str,
) -> None:
    with pytest.raises(ValueError, match="duplicate subquery text after canonicalization"):
        validate_query_evolution_proposal(
            {
                "subqueries": [
                    {
                        "text": text,
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "synonym",
                    }
                ],
                "no_op_reason": None,
            },
            _context(),
        )
```

The existing `foo?`/`foo` case remains the same-batch canonical-duplicate proof. Do not modify production validator code.

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_query_evolution.py -k "existing_queries or mechanical_violations" -q
```

Expected: PASS against the unchanged validator. Failure means the approved design assumption is wrong; stop rather than changing the validator.

- [ ] **Step 2: Strengthen stale-version/hash guards without real reservations**

In the parameter list for `test_canary_run_rejects_lock_drift_before_dispatch`, add:

```python
(
    "prompt-version",
    lambda payload: payload["prompt"].__setitem__("version", "query-evolve-v1"),  # type: ignore[index]
    "prompt_binding_failed",
),
```

Use an explicit runtime ledger path and assert it was never created:

```python
ledger_path = tmp_path / "runtime-ledger.sqlite3"

probe.run_canary(
    lock_path,
    ProbeRuntime(
        allow_live=True,
        env_file=env_file,
        ledger_path=ledger_path,
    ),
)

assert not ledger_path.exists()
```

In `test_run_rejects_locked_prompt_hash_drift_before_live_probe`, likewise assign `ledger_path = tmp_path / "runtime-ledger.sqlite3"`, pass it to `ProbeRuntime`, and assert `not ledger_path.exists()` together with `live_probe_started is False`.

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/integration/test_query_evolution_probe.py -k "prompt_version_drift or prompt_hash_drift or lock_drift_before_dispatch" -q
```

Expected: PASS with zero LLM usage, no network client construction, and no runtime ledger file.

- [ ] **Step 3: Recheck the archived failure read-only**

Run this offline script; it prints no query text or query ID:

```powershell
@'
import json
from pathlib import Path

import scripts.probe_query_evolution as probe
from paper_search.evolution.query_evolution import _canonical_query

run = Path("runs/_diag_query_evolution_contract-canary-20260810")
canary = probe.load_canary_lock(Path("runs/_locks/query_evolution_contract-20260810/canary.lock.json"))
source = probe.load_probe_lock(Path("runs/_locks/query_evolution_contract-v2-source-20260810/probe.lock.json"))
records = probe._load_canary_raw_records(source)
manifest = json.loads((run / "snapshots/snapshot-manifest.json").read_bytes())
response_by_entry = {
    item["entry_id"]: run / "snapshots" / item["response_path"]
    for item in manifest["entries"]
}
outcomes = [
    json.loads(line)
    for line in (run / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
]
failed = next(item for item in outcomes if item["terminal"] == "integrity_failure")
context = probe.build_probe_context(failed["query_id"], records[failed["query_id"]])
response = json.loads(response_by_entry[failed["snapshot_refs"][0]["entry_id"]].read_bytes())
raw = json.loads(response["choices"][0]["message"]["content"])
outputs = [_canonical_query(item["text"]).casefold() for item in raw["subqueries"]]
existing = [
    _canonical_query(context.original_query).casefold(),
    *[_canonical_query(item.text).casefold() for item in context.seed_subqueries],
]
print("terminal", failed["terminal"])
print("output_count", len(outputs))
print("matches_existing", [sum(value == item for item in existing) for value in outputs])
'@ | & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -
```

Expected:

```text
terminal integrity_failure
output_count 2
matches_existing [1, 1]
```

- [ ] **Step 4: Run focused static and test gates**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/unit/test_prompt_artifacts.py tests/unit/test_query_evolution.py tests/integration/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/llm/prompt_artifacts.py scripts/probe_query_evolution.py tests/unit/test_prompt_artifacts.py tests/unit/test_query_evolution.py tests/integration/test_query_evolution_probe.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts/probe_query_evolution.py
git diff --check
```

Expected: all commands exit `0`.

- [ ] **Step 5: Run the full offline suite and verify protected fingerprints**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -q
Get-FileHash -LiteralPath data/budget_ledger.sqlite3 -Algorithm SHA256
```

Expected: pytest exits `0`; ledger SHA-256 remains:

```text
628571402F58F9DD790CC00DD30173652D4102C2E591791A5A264219B80D4293
```

Recompute the archived-evidence directory fingerprint:

```powershell
$root=(Resolve-Path 'runs/_diag_query_evolution_contract-canary-20260810').Path
$records=Get-ChildItem -LiteralPath $root -Recurse -File | Sort-Object FullName | ForEach-Object { $rel=$_.FullName.Substring($root.Length+1).Replace('\','/'); $hash=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(); "$rel`t$hash" }
$bytes=[System.Text.Encoding]::UTF8.GetBytes(($records -join "`n"))
$sha=[System.Security.Cryptography.SHA256]::Create()
Write-Output $records.Count
($sha.ComputeHash($bytes) | ForEach-Object ToString x2) -join ''
```

Expected: 7 files and fingerprint:

```text
084f60681b66240c97ac8037aa6740ef237e4b09294d1d1a29cf34a1238e0f59
```

- [ ] **Step 6: Commit invariant coverage and stop**

Run:

```powershell
git add -- tests/unit/test_query_evolution.py tests/integration/test_query_evolution_probe.py
git diff --cached --check
git diff --cached --name-only
git commit -m "test: preserve query evolution novelty boundaries"
git status --short
```

Expected: the commit contains only the two test files. Final status retains user-owned untracked `data/budget_ledger.sqlite3` and `deliverables/` and contains no new `runs/` artifact.

## Completion Boundary

Stop after the two implementation commits and offline verification. Do not rebuild source/canary locks, run readiness/preflight against the persistent ledger, read `.env`, or start a live canary. Those actions require a separate decision and explicit authorization.
