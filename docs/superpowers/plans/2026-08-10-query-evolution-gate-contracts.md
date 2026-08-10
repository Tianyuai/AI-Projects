# Query Evolution Gate Contracts Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the bounded-probe Gate A denominator, separate retrieved coverage from selected ranking, and count canonical gold associations once.

**Architecture:** Extend the existing frozen-query and projection models with a retrieved-ID stream while leaving selected-paper ranking unchanged. Evaluate Gate B from retrieved IDs, Gate C from Top-50 IDs, and derive preflight 14/8 counts from frozen evidence.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, Ruff, mypy, PowerShell.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-10-query-evolution-gate-contracts-design.md`.
- Do not change ranking, RRF, filters, prompt, provider, retry, timeout, budget, ledger, snapshot, or threshold behavior.
- Do not read `.env`, send network requests, rebuild locks, run live work, or rewrite sealed evidence.
- Preserve user-owned `data/budget_ledger.sqlite3` and `deliverables/`.
- Use `D:\AI Projects\Projects\.venv\Scripts\python.exe` for Python checks.
- Add failing tests before production changes.

---

### Task 1: Repair evaluation stream and Gate semantics

**Files:**
- Modify: `src/paper_search/evaluation/query_evolution_probe.py`
- Modify: `tests/evaluation/test_query_evolution_probe.py`

**Interfaces:**
- `FrozenQueryRecord.retrieved_paper_ids: list[str]` stores the complete frozen retrieval stream.
- `QueryProjection.retrieved_ids: list[str]` stores frozen retrieval IDs plus deduplicated addition IDs.
- `count_gold_associations(gold, identifiers_by_query, id_map) -> int` counts unique resolved query-paper pairs.

- [ ] **Step 1: Write focused failing tests**

Update the `_record()` fixture to accept an explicit retrieved stream, defaulting to the selected papers' canonical IDs. Add tests proving:

```python
def test_gate_a_uses_locked_execution_count_not_sixty_query_baseline() -> None:
    # 60-record / 2910-selected baseline, but 55 locked and 55 terminal.
    # Expected Gate A: passed.

def test_gate_a_rejects_terminal_count_below_locked_count() -> None:
    # locked=55, terminal=54.
    # Expected Gate A: failed.

def test_gold_associations_are_unique_after_identifier_resolution() -> None:
    # Two aliases resolve to one paper within one query.
    # Expected baseline/candidate count: one association.

def test_retrieved_but_unselected_gold_is_retained_not_new() -> None:
    # W2 is in retrieved IDs but absent from selected papers; an addition returns W2.
    # Expected newly_retrieved_count: zero.

def test_true_retrieval_gain_does_not_change_top50_without_ranking_gain() -> None:
    # Add one truly new retrieved gold after a full selected list.
    # Expected retrieved gain one and Top-50 gain zero.
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_query_evolution_probe.py -k "locked_execution_count or terminal_count_below or unique_after_identifier or retrieved_but_unselected or true_retrieval_gain" -q
```

Expected: failures because the models have no retrieved stream, Gate A compares terminals with 60, and counts use selected IDs plus alias-inflated sums.

- [ ] **Step 3: Implement the minimal dual-stream model**

In `FrozenQueryRecord`, require:

```python
retrieved_paper_ids: list[str]
```

In `QueryProjection`, add:

```python
retrieved_ids: list[str]
```

Change `_project_query()` to preserve current selected ordering while separately deduplicating `retrieved_paper_ids` followed by every addition paper canonical ID. Pass each record's retrieved IDs from `FrozenProbeBaseline.total_selected` and `merge_probe_results()`.

Add a resolved-pair helper and public count wrapper:

```python
def _gold_associations(
    gold: Sequence[EvaluationQuery],
    identifiers_by_query: Mapping[str, Sequence[str]],
    id_map: IdentifierMap | None,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for record in gold:
        available = {_resolved(value, id_map) for value in identifiers_by_query[record.query_id]}
        pairs.update(
            (record.query_id, resolved)
            for identifier in record.relevant_paper_ids
            if (resolved := _resolved(identifier, id_map)) in available
        )
    return pairs

def count_gold_associations(
    gold: Sequence[EvaluationQuery],
    identifiers_by_query: Mapping[str, Sequence[str]],
    id_map: IdentifierMap | None,
) -> int:
    return len(_gold_associations(gold, identifiers_by_query, id_map))
```

In `evaluate_probe()`, use `retrieved_ids` for Gate B counts/newness/retention and `top50_ids` for Gate C counts/retention. In `_gate_a()`, accept execution counts only when both are absent or both are positive and equal; do not compare them with `baseline.query_count`.

- [ ] **Step 4: Verify GREEN and commit**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/evaluation/query_evolution_probe.py tests/evaluation/test_query_evolution_probe.py
```

Then commit only the evaluation module and its test:

```powershell
git add -- src/paper_search/evaluation/query_evolution_probe.py tests/evaluation/test_query_evolution_probe.py
git diff --cached --check
git commit -m "fix: separate query evolution retrieval and ranking gates"
```

---

### Task 2: Wire runtime counts and derive preflight baselines

**Files:**
- Modify: `scripts/probe_query_evolution.py`
- Modify: `tests/integration/test_query_evolution_probe.py`

**Interfaces:**
- `_frozen_inputs()` reads `executions.jsonl[*].retrieved_paper_ids` into each `FrozenQueryRecord`.
- `_build_lock_payload()` derives `baseline_candidate_gold_count` and `baseline_top50_gold_count` through `count_gold_associations()`.
- Runtime integrity records `locked_query_count=len(lock.query_ids)` and `terminal_count=len(replayed)`.

- [ ] **Step 1: Write failing integration tests**

Extend the offline preflight test to assert derived `14/8`. Add fixture-level tests that reject reconstructed baseline count drift and assert the runtime source contains the 55-query locked count rather than the 60-query denominator. Keep all fixtures offline and temporary.

- [ ] **Step 2: Verify RED**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/integration/test_query_evolution_probe.py -k "preflight_reconstructs_fixed_queue or baseline_gold_count_drift or locked_execution_count" -q
```

Expected: failures because preflight embeds 14/8 constants, frozen records omit retrieved IDs, or runtime uses 60 as the locked execution count.

- [ ] **Step 3: Implement minimal runtime wiring**

Import `count_gold_associations`. Build resolved candidate/top50 counts from frozen `retrieved_paper_ids` and `selected_paper_ids`; require exactly 14 and 8 before writing a lock. Populate `FrozenQueryRecord.retrieved_paper_ids` from executions and set:

```python
locked_query_count=len(lock.query_ids)
terminal_count=len(replayed)
```

Do not change lock fields, schema versions, thresholds, or run output schemas.

- [ ] **Step 4: Run focused and full offline verification**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check src/paper_search/evaluation/query_evolution_probe.py scripts/probe_query_evolution.py tests/evaluation/test_query_evolution_probe.py tests/integration/test_query_evolution_probe.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy src scripts/probe_query_evolution.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -m "not online" -q
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 5: Recompute sealed evidence read-only**

Use the sealed run `runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810` and its existing outcomes/snapshots to reconstruct additions without network. Assert only aggregate values:

```text
gate_a=passed
gate_b=passed
gate_c=failed
baseline_candidate_gold_count=14
candidate_candidate_gold_count=15
baseline_top50_gold_count=8
candidate_top50_gold_count=8
newly_retrieved_count=1
accepted_added_gold_positions=[55, 63]
```

Verify `git status --short` still contains only user-owned untracked `data/budget_ledger.sqlite3` and `deliverables/` plus planned tracked changes.

- [ ] **Step 6: Commit runtime repair**

```powershell
git add -- scripts/probe_query_evolution.py tests/integration/test_query_evolution_probe.py
git diff --cached --check
git commit -m "fix: align query evolution probe gate contracts"
```

Stop after offline verification and the two repair commits. Do not rebuild a lock or run live work.
