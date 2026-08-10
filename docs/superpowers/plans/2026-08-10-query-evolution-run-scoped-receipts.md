# Query Evolution Run-Scoped Receipts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent historical project receipts from causing false Query Evolution accounting failures while preserving the ledger's project-wide audit contract.

**Architecture:** Keep `SQLiteBudgetLedger.report()` unchanged. Add one private probe helper that filters `LedgerReport.receipts` by `run_id`, then use it for canary/full-probe recovery, canary finalization, and promotion checks.

**Tech Stack:** Python 3.12, SQLite, Pydantic v2, pytest, httpx MockTransport, Ruff, mypy.

## Global Constraints

- Do not change `SQLiteBudgetLedger.report()`, `LedgerReport`, project checkpoint, budget caps, evidence schema, prompt, validation, retry, or timeout behavior.
- Do not rewrite `runs/_diag_query_evolution_contract-canary-20260810/` or any historical ledger receipt.
- Do not read `.env`, construct real provider clients, send network requests, or run live canary/full probe.
- Preserve untracked `data/budget_ledger.sqlite3` and `deliverables/`.
- Use TDD: observe the historical-receipt regression tests fail before modifying production code.

---

## File Structure

- Modify `scripts/probe_query_evolution.py`: define the run-scoped receipt selector and replace the five run-local uses of project-wide receipts.
- Modify `tests/integration/test_query_evolution_probe.py`: seed unrelated historical receipts and cover canary execution, canary recovery/cleanup, and full-probe recovery.
- Read only `src/paper_search/control/ledger.py`: retain and verify its project-wide `LedgerReport.receipts` contract.
- Read only `runs/_diag_query_evolution_contract-canary-20260810/outcomes.jsonl`: verify the archived outcome classification without rewriting evidence.

---

### Task 1: Filter Query Evolution receipts by run and close the offline regression

**Files:**
- Modify: `scripts/probe_query_evolution.py:23-30,504-540,805-864,1220-1240`
- Modify: `tests/integration/test_query_evolution_probe.py:1-250,673-930`
- Verify only: `src/paper_search/control/ledger.py:864-952`

**Interfaces:**
- Consumes: `SQLiteBudgetLedger.report(run_id: str) -> LedgerReport`, whose `receipts` remain project-wide.
- Produces: `_run_receipts(ledger: SQLiteBudgetLedger, run_id: str) -> list[LedgerReceipt]`.
- Preserves: `reserve_probe_operations`, `reserve_canary_operations`, `_finalize_canary_reservations`, and `run_canary` public behavior except removal of false cross-run accounting failures.

- [ ] **Step 1: Record the user-ledger fingerprint and add shared historical-ledger test setup**

Before editing, record the SHA-256 shown by this read-only command in the task notes:

```powershell
Get-FileHash data/budget_ledger.sqlite3 -Algorithm SHA256
```

Add `Decimal`, allow `_write_canary_lock` to bind a caller-supplied ledger, and add one deterministic historical receipt helper:

```python
from decimal import Decimal


def _seed_settled_receipt(
    ledger: probe.SQLiteBudgetLedger,
    *,
    run_id: str = "prior-run",
    query_id: str = "prior:evolve",
) -> probe.LedgerReservation:
    estimate = probe.UsageEstimate(llm_calls=1, cost_cny=Decimal("0.01"))
    actual = probe.UsageActual(llm_calls=1, cost_cny=Decimal("0.001"))
    reservation = ledger.reserve(
        run_id=run_id,
        query_id=query_id,
        estimate=estimate,
        run_cap_cny=probe.DEV_RUN_CAP_CNY,
    )
    ledger.checkpoint_actual(reservation, actual)
    ledger.settle(reservation, actual)
    return reservation


def _write_canary_lock(
    tmp_path: Path,
    label: str,
    *,
    ledger_path: Path | None = None,
) -> tuple[Path, probe.CanaryLock, Path]:
```

Immediately after the function signature, derive the canary ledger path:

```python
canary_ledger_path = ledger_path or tmp_path / f"{label}-canary-ledger.sqlite3"
```

In the existing `probe.preflight_canary(...)` call, replace only its `ledger_path` argument:

```python
ledger_path=canary_ledger_path,
```

- [ ] **Step 2: Write the failing historical-receipt tests**

Add a parameterized end-to-end mocked canary test for both promotion and strict-contract failure:

```python
@pytest.mark.parametrize(
    ("behaviors", "expected_reason", "expected_promoted"),
    [
        pytest.param(
            ["generated", "no_op", "generated"],
            "passed",
            True,
            id="promoted",
        ),
        pytest.param(
            ["generated", "integrity", "generated"],
            "contract_canary_failed",
            False,
            id="contract-failed",
        ),
    ],
)
def test_canary_uses_only_current_run_receipts_when_project_has_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behaviors: list[str],
    expected_reason: str,
    expected_promoted: bool,
) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = probe.SQLiteBudgetLedger(
        ledger_path,
        reservation_ttl_seconds=probe.PROBE_LEDGER_TTL_SECONDS,
    )
    prior = _seed_settled_receipt(ledger)
    lock_path, lock, run_dir = _write_canary_lock(
        tmp_path,
        "project-history",
        ledger_path=ledger_path,
    )
    env_file = tmp_path / "canary.env"
    env_file.write_text("LLM_API_KEY=test-llm-key\n", encoding="utf-8")
    _install_mock_llm_client(
        monkeypatch,
        _mock_llm_transport(behaviors, []),
    )
    _install_openalex_guards(monkeypatch)

    probe.run_canary(
        lock_path,
        ProbeRuntime(
            allow_live=True,
            env_file=env_file,
            ledger_path=ledger_path,
        ),
    )

    result = _read_result(run_dir)
    receipts = ledger.report(lock.canary_run_id).receipts
    current = [receipt for receipt in receipts if receipt.run_id == lock.canary_run_id]
    assert result["reason"] == expected_reason
    assert result["promoted"] is expected_promoted
    assert len(receipts) == 4
    assert len(current) == 3
    assert all(receipt.state == "settled" for receipt in current)
    assert next(receipt for receipt in receipts if receipt.reservation_id == prior.reservation_id).state == "settled"
```

Add direct recovery tests that prove current-run reservations are restored while unrelated history is ignored:

```python
def test_canary_reservation_recovery_ignores_other_runs(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = probe.SQLiteBudgetLedger(ledger_path)
    _seed_settled_receipt(ledger)
    _, lock, _ = _write_canary_lock(
        tmp_path,
        "canary-resume",
        ledger_path=ledger_path,
    )
    created = probe.reserve_canary_operations(lock, ledger)

    restored = probe.reserve_canary_operations(lock, ledger)

    assert set(restored) == set(created)
    assert {
        key: value.reservation_id for key, value in restored.items()
    } == {
        key: value.reservation_id for key, value in created.items()
    }


def test_full_probe_reservation_recovery_ignores_other_runs(tmp_path: Path) -> None:
    ledger = probe.SQLiteBudgetLedger(tmp_path / "ledger.sqlite3")
    _seed_settled_receipt(ledger)
    lock = _synthetic_probe_lock(("q1", "q2"))
    created = probe.reserve_probe_operations(lock, ledger)

    restored = probe.reserve_probe_operations(lock, ledger)

    assert {
        key: value.reservation_id for key, value in restored.values.items()
    } == {
        key: value.reservation_id for key, value in created.values.items()
    }
```

Update the existing accounting-failure and cancellation tests to seed one prior terminal receipt before `_write_canary_lock(..., ledger_path=ledger_path)`, then assert only the three current-run receipts are checked and all are terminal. Do not weaken their fixed-reason assertions.

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest `
  tests/integration/test_query_evolution_probe.py `
  -k "project_has_history or recovery_ignores_other_runs or marks_accounting_failure or marks_timeout" -q
```

Expected: FAIL because project-wide receipt counts overwrite `passed`, `contract_canary_failed`, and `canary_cancelled`, and both recovery calls reject unrelated history.

- [ ] **Step 4: Implement the single run-scoped selector and replace all five misuse sites**

Import `LedgerReceipt` and add the helper near the reservation functions:

```python
from paper_search.control.ledger import (
    DEV_RUN_CAP_CNY,
    LedgerReceipt,
    LedgerReservation,
    LedgerReservationError,
    SQLiteBudgetLedger,
)


def _run_receipts(
    ledger: SQLiteBudgetLedger,
    run_id: str,
) -> list[LedgerReceipt]:
    return [
        receipt
        for receipt in ledger.report(run_id).receipts
        if receipt.run_id == run_id
    ]
```

Replace only these run-local reads:

```python
# reserve_probe_operations
receipts = _run_receipts(ledger, lock.probe_run_id)

# _finalize_canary_reservations, initial and final reads
receipts = {
    receipt.reservation_id: receipt
    for receipt in _run_receipts(ledger, lock.canary_run_id)
}
final_receipts = _run_receipts(ledger, lock.canary_run_id)

# reserve_canary_operations
receipts = _run_receipts(ledger, lock.canary_run_id)

# run_canary promotion check
receipts = _run_receipts(ledger, lock.canary_run_id)
```

Keep the existing `LedgerReservationError` handling for an unknown run. Do not change any count, state, operation-ID, actual-usage, snapshot, or promotion predicate.

- [ ] **Step 5: Run focused GREEN verification**

Run:

```powershell
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest `
  tests/integration/test_query_evolution_probe.py -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest `
  tests/unit/test_budget_ledger.py -q
```

Expected: all tests pass; the ledger test proving project-wide receipt history remains unchanged.

- [ ] **Step 6: Run the complete offline quality gate**

Run:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m pytest -q
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m ruff check `
  scripts/probe_query_evolution.py `
  tests/integration/test_query_evolution_probe.py
& 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -m mypy `
  src scripts/probe_query_evolution.py
git diff --check
```

Expected: all tests pass; Ruff, mypy, and diff check exit `0`. Do not run full-repository Ruff because the untouched user-owned `deliverables/project-docs/edit_docx.py` has a known unrelated F401.

- [ ] **Step 7: Reclassify archived outcomes read-only and commit**

Run this read-only check; it must not write into the archived run directory:

```powershell
@'
import json
import scripts.probe_query_evolution as probe
from pathlib import Path

run_dir = Path("runs/_diag_query_evolution_contract-canary-20260810")
stored = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
outcomes = [
    json.loads(line)
    for line in (run_dir / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
]
print("stored_reason", stored["reason"])
print("reclassified_reason", probe._classify_canary_outcomes(outcomes))
print("promoted", stored["promoted"])
'@ | & 'D:\AI Projects\Projects\.venv\Scripts\python.exe' -
```

Expected:

```text
stored_reason canary_accounting_failed
reclassified_reason contract_canary_failed
promoted False
```

Then verify scope and commit. The second ledger hash must exactly match the value recorded in Step 1:

```powershell
Get-FileHash data/budget_ledger.sqlite3 -Algorithm SHA256
git status --short
git diff -- runs/_diag_query_evolution_contract-canary-20260810
git add -- scripts/probe_query_evolution.py tests/integration/test_query_evolution_probe.py
git diff --cached --check
git diff --cached --name-only
git commit -m "fix: scope query evolution receipts by run"
```

Expected: the user-ledger SHA-256 is unchanged; archived evidence has no diff; `git status --short` still lists the pre-existing untracked `data/budget_ledger.sqlite3` and `deliverables/`; `git diff --cached --name-only` lists only the two planned files. Stop after the commit. Do not rebuild a lock or request live authorization as part of this task.

---

## Completion Boundary

This plan ends after the offline fix, full verification, read-only historical classification, and one code/test commit. Any new canary lock, live canary, or 55-query probe requires a separate design decision and explicit authorization.
