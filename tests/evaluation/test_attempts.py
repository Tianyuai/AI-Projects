from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from paper_search.evaluation.attempts import (
    ValidationAttemptClaim,
    ValidationAttemptConflictError,
    ValidationAttemptStore,
    dispatch_with_validation_claim,
)


NOW = datetime(2026, 8, 2, tzinfo=UTC)
LOCK_A = "sha256:" + "a" * 64
LOCK_B = "sha256:" + "b" * 64
LOCK_C = "sha256:" + "c" * 64


def test_claim_writes_exact_initial_state_and_digest_path(tmp_path: Path) -> None:
    store = ValidationAttemptStore(tmp_path)

    claim = store.claim(
        validation_lock_sha256=LOCK_A,
        run_id="validation-1",
        claimed_at=NOW,
    )

    assert claim == ValidationAttemptClaim(
        validation_lock_sha256=LOCK_A,
        run_id="validation-1",
        claimed_at=NOW,
        state="claimed",
        completed_at=None,
        incident_ref=None,
    )
    path = tmp_path / "validation-attempts" / f"{'a' * 64}.claim"
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "claimed"


def test_concurrent_claim_is_exclusive_and_maps_conflict_code(tmp_path: Path) -> None:
    store = ValidationAttemptStore(tmp_path)
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def claim(run_id: str) -> None:
        barrier.wait()
        try:
            store.claim(validation_lock_sha256=LOCK_A, run_id=run_id, claimed_at=NOW)
        except ValidationAttemptConflictError as error:
            assert error.code == "validation_attempt_conflict"
            outcomes.append("conflict")
        else:
            outcomes.append("claimed")

    threads = [threading.Thread(target=claim, args=(f"run-{index}",)) for index in (1, 2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["claimed", "conflict"]


def test_concurrent_terminal_transition_commits_exactly_once(tmp_path: Path) -> None:
    store = ValidationAttemptStore(tmp_path)
    store.claim(validation_lock_sha256=LOCK_A, run_id="validation-1", claimed_at=NOW)
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def transition(target: str) -> None:
        barrier.wait()
        try:
            store.transition(
                validation_lock_sha256=LOCK_A,
                target=target,
                completed_at=NOW + timedelta(minutes=1),
                incident_ref=None if target == "complete" else "INC-race",
            )
        except ValidationAttemptConflictError:
            outcomes.append("conflict")
        else:
            outcomes.append(target)

    threads = [
        threading.Thread(target=transition, args=(target,))
        for target in ("complete", "failed")
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert "conflict" in outcomes
    assert len(outcomes) == 2


@pytest.mark.parametrize("target", ["complete", "failed", "interrupted"])
def test_every_legal_terminal_transition_persists_across_restart(tmp_path: Path, target: str) -> None:
    store = ValidationAttemptStore(tmp_path)
    store.claim(validation_lock_sha256=LOCK_A, run_id="validation-1", claimed_at=NOW)
    incident = None if target == "complete" else f"incident-{target}"

    transitioned = store.transition(
        validation_lock_sha256=LOCK_A,
        target=target,
        completed_at=NOW + timedelta(minutes=1),
        incident_ref=incident,
    )
    restarted = ValidationAttemptStore(tmp_path).read(LOCK_A)

    assert transitioned.state == target
    assert restarted == transitioned
    with pytest.raises(ValidationAttemptConflictError, match="terminal"):
        store.transition(
            validation_lock_sha256=LOCK_A,
            target="complete",
            completed_at=NOW + timedelta(minutes=2),
        )


@pytest.mark.parametrize("target", ["failed", "interrupted"])
def test_unsuccessful_terminal_transition_requires_incident_reference(tmp_path: Path, target: str) -> None:
    store = ValidationAttemptStore(tmp_path)
    store.claim(validation_lock_sha256=LOCK_A, run_id="validation-1", claimed_at=NOW)

    with pytest.raises(ValueError, match="incident_ref"):
        store.transition(
            validation_lock_sha256=LOCK_A,
            target=target,
            completed_at=NOW + timedelta(minutes=1),
        )


def test_malformed_claim_fails_closed(tmp_path: Path) -> None:
    attempts = tmp_path / "validation-attempts"
    attempts.mkdir()
    (attempts / f"{'a' * 64}.claim").write_text('{"state":"claimed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="malformed validation attempt claim"):
        ValidationAttemptStore(tmp_path).read(LOCK_A)


def test_superseding_hash_requires_prior_incident_and_same_hash_never_retries(tmp_path: Path) -> None:
    store = ValidationAttemptStore(tmp_path)
    store.claim(validation_lock_sha256=LOCK_A, run_id="validation-1", claimed_at=NOW)
    store.transition(
        validation_lock_sha256=LOCK_A,
        target="failed",
        completed_at=NOW + timedelta(minutes=1),
        incident_ref="INC-42",
    )

    with pytest.raises(ValidationAttemptConflictError, match="supersedes"):
        store.claim(
            validation_lock_sha256=LOCK_B,
            run_id="validation-2",
            claimed_at=NOW + timedelta(minutes=2),
        )
    replacement = store.claim(
        validation_lock_sha256=LOCK_B,
        run_id="validation-2",
        claimed_at=NOW + timedelta(minutes=2),
        supersedes_validation_lock_sha256=LOCK_A,
        incident_ref="INC-42",
    )
    assert replacement.validation_lock_sha256 == LOCK_B
    with pytest.raises(ValidationAttemptConflictError):
        store.claim(
            validation_lock_sha256=LOCK_A,
            run_id="validation-retry",
            claimed_at=NOW + timedelta(minutes=3),
        )


def test_concurrent_supersessions_consume_one_prior_incident(tmp_path: Path) -> None:
    store = ValidationAttemptStore(tmp_path)
    store.claim(validation_lock_sha256=LOCK_A, run_id="validation-1", claimed_at=NOW)
    store.transition(
        validation_lock_sha256=LOCK_A,
        target="failed",
        completed_at=NOW + timedelta(minutes=1),
        incident_ref="INC-race",
    )
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def supersede(lock_hash: str) -> None:
        barrier.wait()
        try:
            store.claim(
                validation_lock_sha256=lock_hash,
                run_id=f"validation-{lock_hash[-1]}",
                claimed_at=NOW + timedelta(minutes=2),
                supersedes_validation_lock_sha256=LOCK_A,
                incident_ref="INC-race",
            )
        except ValidationAttemptConflictError:
            outcomes.append("conflict")
        else:
            outcomes.append("claimed")

    threads = [
        threading.Thread(target=supersede, args=(lock_hash,))
        for lock_hash in (LOCK_B, LOCK_C)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["claimed", "conflict"]


def test_replay_dispatch_bypasses_attempt_claim(tmp_path: Path) -> None:
    store = ValidationAttemptStore(tmp_path)
    events: list[str] = []

    result = dispatch_with_validation_claim(
        execution_mode="replay",
        offline_preflight=lambda: events.append("preflight"),
        reserve_run_budget=lambda: events.append("reserve"),
        store=store,
        validation_lock_sha256=LOCK_A,
        run_id="replay-1",
        claimed_at=NOW,
        dispatch=lambda: events.append("dispatch") or "ok",
    )

    assert result == "ok"
    assert events == ["preflight", "reserve", "dispatch"]
    assert not (tmp_path / "validation-attempts").exists()


def test_keyboard_interrupt_transitions_created_claim_to_interrupted(tmp_path: Path) -> None:
    store = ValidationAttemptStore(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        dispatch_with_validation_claim(
            execution_mode="live",
            offline_preflight=lambda: None,
            reserve_run_budget=lambda: None,
            store=store,
            validation_lock_sha256=LOCK_A,
            run_id="validation-1",
            claimed_at=NOW,
            dispatch=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

    claim = store.read(LOCK_A)
    assert claim.state == "interrupted"
    assert claim.incident_ref == "automatic-interruption:validation-1"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), asyncio.CancelledError()])
def test_on_claim_interruption_transitions_created_claim(
    tmp_path: Path,
    interruption: BaseException,
) -> None:
    store = ValidationAttemptStore(tmp_path)

    def interrupt() -> None:
        raise interruption

    with pytest.raises(type(interruption)):
        dispatch_with_validation_claim(
            execution_mode="live",
            offline_preflight=lambda: None,
            reserve_run_budget=lambda: None,
            store=store,
            validation_lock_sha256=LOCK_A,
            run_id="validation-1",
            claimed_at=NOW,
            on_claim=interrupt,
            dispatch=lambda: None,
        )

    assert store.read(LOCK_A).state == "interrupted"


def test_dispatch_exception_transitions_created_claim_to_failed(tmp_path: Path) -> None:
    store = ValidationAttemptStore(tmp_path)

    with pytest.raises(RuntimeError, match="dispatch failed"):
        dispatch_with_validation_claim(
            execution_mode="live",
            offline_preflight=lambda: None,
            reserve_run_budget=lambda: None,
            store=store,
            validation_lock_sha256=LOCK_A,
            run_id="validation-1",
            claimed_at=NOW,
            dispatch=lambda: (_ for _ in ()).throw(RuntimeError("dispatch failed")),
        )

    claim = store.read(LOCK_A)
    assert claim.state == "failed"
    assert claim.incident_ref == "automatic-failure:validation-1"


def test_read_recovers_complete_terminal_sibling_after_crash(tmp_path: Path) -> None:
    store = ValidationAttemptStore(tmp_path)
    store.claim(validation_lock_sha256=LOCK_A, run_id="validation-1", claimed_at=NOW)
    terminal = ValidationAttemptClaim(
        validation_lock_sha256=LOCK_A,
        run_id="validation-1",
        claimed_at=NOW,
        state="interrupted",
        completed_at=NOW + timedelta(minutes=1),
        incident_ref="INC-crash",
    )
    terminal_path = (
        tmp_path / "validation-attempts" / f".{'a' * 64}.terminal"
    )
    terminal_path.write_text(
        terminal.model_dump_json() + "\n",
        encoding="utf-8",
    )

    assert ValidationAttemptStore(tmp_path).read(LOCK_A) == terminal
    assert not terminal_path.exists()


def test_store_has_no_delete_or_reset_api() -> None:
    public = {name for name in dir(ValidationAttemptStore) if not name.startswith("_")}
    assert "delete" not in public
    assert "reset" not in public
