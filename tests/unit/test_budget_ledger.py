from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from paper_search.domain.models import (
    BudgetReservation,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.control.budget import HardBudgetController, ReservationError
from paper_search.control.ledger import (
    LedgerBudgetExceededError,
    LedgerReservationError,
    LedgerSoftStopError,
    SQLiteBudgetLedger,
)
from paper_search.evaluation.attempts import (
    ValidationAttemptStore,
    dispatch_with_validation_claim,
)


NOW = datetime(2026, 8, 2, 6, 0, tzinfo=UTC)
VALIDATION_LOCK_SHA256 = "sha256:" + "a" * 64


def test_validation_preflight_and_run_reserve_precede_claim_and_dispatch(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite3")
    store = ValidationAttemptStore(tmp_path)
    events: list[str] = []

    def reserve() -> None:
        ledger.reserve(
            run_id="validation-order",
            query_id="batch",
            estimate=_estimate("0.10"),
            run_cap_cny=Decimal("9.00"),
        )
        events.append("reserve")

    def dispatch() -> str:
        assert store.read(VALIDATION_LOCK_SHA256).state == "claimed"
        events.append("dispatch")
        return "sent"

    result = dispatch_with_validation_claim(
        execution_mode="live",
        offline_preflight=lambda: events.append("preflight"),
        reserve_run_budget=reserve,
        store=store,
        validation_lock_sha256=VALIDATION_LOCK_SHA256,
        run_id="validation-order",
        claimed_at=NOW,
        dispatch=dispatch,
        on_claim=lambda: events.append("claim"),
    )

    assert result == "sent"
    assert events == ["preflight", "reserve", "claim", "dispatch"]


def _ledger(
    path: Path,
    *,
    clock: object | None = None,
    soft: Decimal = Decimal("160.00"),
    hard: Decimal = Decimal("200.00"),
    replay: bool = False,
) -> SQLiteBudgetLedger:
    return SQLiteBudgetLedger(
        path,
        clock=clock or (lambda: NOW),
        reservation_ttl_seconds=30,
        project_soft_stop_cny=soft,
        project_hard_cap_cny=hard,
        replay=replay,
    )


def _estimate(cost: str) -> UsageEstimate:
    return UsageEstimate(search_api_calls=1, cost_cny=Decimal(cost))


def _actual(cost: str | None) -> UsageActual:
    return UsageActual(
        search_api_calls=1,
        cost_cny=Decimal(cost) if cost is not None else None,
    )


def _controller_request(
    *,
    clock: Callable[[], datetime],
    ttl_seconds: int = 120,
    estimate_cost: str = "0.30",
) -> tuple[HardBudgetController, BudgetReservation]:
    controller = HardBudgetController(
        SearchBudget(
            max_search_api_calls=2,
            target_search_api_calls=1,
            max_llm_calls=1,
            target_llm_calls=0,
            max_total_tokens=1,
            max_cost_cny=1.0,
            max_elapsed_seconds=10,
            soft_deadline_seconds=9,
        ),
        formal_live=True,
        reservation_ttl_seconds=ttl_seconds,
        clock=clock,
    )
    request = controller.reserve(
        "provider.search",
        UsageEstimate(
            search_api_calls=1,
            cost_cny=Decimal(estimate_cost),
        ),
    )
    controller.mark_dispatched(request)
    return controller, request


def test_concurrent_reserve_is_atomic_across_sqlite_connections(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = _ledger(path)
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def reserve(query_id: str) -> None:
        barrier.wait()
        try:
            ledger.reserve(
                run_id="run-concurrent",
                query_id=query_id,
                estimate=_estimate("0.20"),
                run_cap_cny=Decimal("0.30"),
            )
        except LedgerBudgetExceededError:
            outcomes.append("rejected")
        else:
            outcomes.append("reserved")

    threads = [
        threading.Thread(target=reserve, args=("q1",)),
        threading.Thread(target=reserve, args=("q2",)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["rejected", "reserved"]
    assert ledger.report("run-concurrent").reserved.cost_cny == Decimal("0.20")


def test_settlement_is_exactly_once_and_stored_as_integer_micro_cny(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = _ledger(path)
    reservation = ledger.reserve(
        run_id="run-exact",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )

    ledger.settle(reservation, _actual("0.10"))
    with pytest.raises(LedgerReservationError, match="already terminal"):
        ledger.settle(reservation, _actual("0.10"))

    report = ledger.report("run-exact")
    assert report.actual.cost_cny == Decimal("0.10")
    assert report.project_actual_cny == Decimal("0.10")
    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            "SELECT actual_cost_micro_cny FROM reservations"
        ).fetchone()
    assert stored == (100_000,)


def test_concurrent_terminal_transition_commits_exactly_once(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite3")
    reservation = ledger.reserve(
        run_id="run-terminal-race",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def settle() -> None:
        barrier.wait()
        try:
            ledger.settle(reservation, _actual("0.10"))
        except LedgerReservationError:
            outcomes.append("rejected")
        else:
            outcomes.append("settled")

    threads = [threading.Thread(target=settle), threading.Thread(target=settle)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["rejected", "settled"]
    assert ledger.report("run-terminal-race").project_actual_cny == Decimal(
        "0.10"
    )


def test_unknown_terminal_cost_is_retained_as_failed_and_fails_closed(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite3")
    reservation = ledger.reserve(
        run_id="run-unknown",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )

    ledger.fail(reservation, _actual(None))

    report = ledger.report("run-unknown")
    assert report.actual.search_api_calls == 1
    assert report.actual.cost_cny is None
    assert report.within_caps is False
    with pytest.raises(LedgerReservationError, match="already terminal"):
        ledger.fail(reservation, _actual(None))


def test_request_and_run_caps_reject_before_reservation(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite3")

    with pytest.raises(LedgerBudgetExceededError, match="request"):
        ledger.reserve(
            run_id="run-request",
            query_id="q1",
            estimate=_estimate("0.300001"),
            run_cap_cny=Decimal("1.00"),
        )

    ledger.reserve(
        run_id="run-cap",
        query_id="q1",
        estimate=_estimate("0.20"),
        run_cap_cny=Decimal("0.30"),
    )
    with pytest.raises(LedgerBudgetExceededError, match="run"):
        ledger.reserve(
            run_id="run-cap",
            query_id="q2",
            estimate=_estimate("0.20"),
            run_cap_cny=Decimal("0.30"),
        )


def test_project_soft_stop_and_hard_cap_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = _ledger(path, soft=Decimal("0.15"), hard=Decimal("0.20"))
    reservation = ledger.reserve(
        run_id="run-project",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    ledger.settle(reservation, _actual("0.15"))

    with pytest.raises(LedgerSoftStopError):
        ledger.reserve(
            run_id="run-soft-stop",
            query_id="q1",
            estimate=_estimate("0.01"),
            run_cap_cny=Decimal("1.00"),
        )

    overage = _ledger(
        tmp_path / "hard.sqlite3",
        soft=Decimal("0.15"),
        hard=Decimal("0.20"),
    )
    hard_reservation = overage.reserve(
        run_id="run-hard",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    overage.fail(hard_reservation, _actual("0.21"))
    assert overage.report("run-hard").within_caps is False
    with pytest.raises(LedgerBudgetExceededError, match="project hard"):
        overage.reserve(
            run_id="run-hard-next",
            query_id="q1",
            estimate=_estimate("0.01"),
            run_cap_cny=Decimal("1.00"),
        )


def test_restart_recovery_marks_expired_reservation_failed_without_releasing_spend(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    reservation = _ledger(path).reserve(
        run_id="run-restart",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )

    restarted = _ledger(path, clock=lambda: NOW + timedelta(minutes=1))
    report = restarted.report("run-restart")

    assert report.reserved == UsageEstimate(cost_cny=Decimal("0"))
    assert report.actual == UsageActual(search_api_calls=1, cost_cny=Decimal("0.10"))
    assert report.project_actual_cny == Decimal("0.10")
    with pytest.raises(LedgerReservationError, match="already terminal"):
        restarted.settle(reservation, _actual("0.05"))


def test_replay_is_run_local_zero_spend_and_does_not_mutate_live_project(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    live = _ledger(path)
    reservation = live.reserve(
        run_id="live-run",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    live.settle(reservation, _actual("0.10"))
    before = live.report("live-run")

    replay = _ledger(path, replay=True)
    replay_reservation = replay.reserve(
        run_id="replay-run",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    assert replay.report("replay-run").actual == UsageActual(
        cost_cny=Decimal("0")
    )
    replay.settle(replay_reservation, _actual("0.10"))

    replay_report = replay.report("replay-run")
    after = live.report("live-run")
    assert replay_report.actual == UsageActual(cost_cny=Decimal("0"))
    assert replay_report.project_actual_cny == Decimal("0")
    assert after == before


def test_live_project_caps_and_policy_metadata_are_persistent_and_exact_match(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    first = _ledger(path, soft=Decimal("10.00"), hard=Decimal("20.00"))
    first.reserve(
        run_id="metadata-run",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    _ledger(path, soft=Decimal("10.00"), hard=Decimal("20.00"))

    with pytest.raises(LedgerReservationError, match="metadata"):
        _ledger(path, soft=Decimal("11.00"), hard=Decimal("21.00"))

    with sqlite3.connect(path) as connection:
        metadata = connection.execute(
            """
            SELECT schema_version, policy_version,
                   project_soft_stop_micro_cny, project_hard_cap_micro_cny
            FROM ledger_metadata WHERE singleton_id = 1
            """
        ).fetchone()
    assert metadata == (2, "budget-ledger-policy-v1", 10_000_000, 20_000_000)


def test_live_project_caps_cannot_raise_fixed_production_boundaries(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="fixed production boundaries"):
        _ledger(
            tmp_path / "ledger.sqlite3",
            soft=Decimal("161.00"),
            hard=Decimal("201.00"),
        )


def test_concurrent_instances_cannot_initialize_different_project_caps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def initialize(soft: str, hard: str) -> None:
        barrier.wait()
        try:
            _ledger(path, soft=Decimal(soft), hard=Decimal(hard))
        except LedgerReservationError:
            outcomes.append("mismatch")
        else:
            outcomes.append("initialized")

    threads = [
        threading.Thread(target=initialize, args=("10.00", "20.00")),
        threading.Thread(target=initialize, args=("11.00", "21.00")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["initialized", "mismatch"]


def test_controller_final_actual_checkpoint_survives_crash_and_recovers_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    current = NOW
    ledger = _ledger(path, clock=lambda: current)
    ledger_reservation = ledger.reserve(
        run_id="crash-run",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    controller = HardBudgetController(
        SearchBudget(
            max_search_api_calls=2,
            target_search_api_calls=1,
            max_llm_calls=1,
            target_llm_calls=0,
            max_total_tokens=1,
            max_cost_cny=1.0,
            max_elapsed_seconds=10,
            soft_deadline_seconds=9,
        ),
        formal_live=True,
        clock=lambda: current,
    )
    request = controller.reserve(
        "provider.search",
        UsageEstimate(search_api_calls=1, cost_cny=Decimal("0.30")),
    )
    controller.mark_dispatched(request)
    final_actual = UsageActual(
        search_api_calls=1,
        cost_cny=Decimal("0.25"),
    )
    controller.settle(request, final_actual)

    ledger.checkpoint_actual(ledger_reservation, final_actual)
    ledger.checkpoint_actual(ledger_reservation, final_actual)
    current += timedelta(minutes=1)
    recovered = _ledger(path, clock=lambda: current)
    first = recovered.report("crash-run")
    second = _ledger(path, clock=lambda: current).report("crash-run")

    assert first.actual == final_actual
    assert first.project_actual_cny == Decimal("0.25")
    assert second == first
    recovered.finalize_controller_actual(
        ledger_reservation=ledger_reservation,
        controller=controller,
        request_reservation=request,
        actual=final_actual,
    )
    recovered.finalize_controller_actual(
        ledger_reservation=ledger_reservation,
        controller=controller,
        request_reservation=request,
        actual=final_actual,
    )
    with pytest.raises(LedgerReservationError, match="already terminal"):
        recovered.settle(ledger_reservation, final_actual)


def test_checkpoint_actual_is_immediately_authoritative_for_caps_and_report(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite3")
    reservation = ledger.reserve(
        run_id="authoritative-run",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("0.30"),
    )
    final_actual = _actual("0.25")

    ledger.checkpoint_actual(reservation, final_actual)

    report = ledger.report("authoritative-run")
    assert report.reserved == UsageEstimate(cost_cny=Decimal("0"))
    assert report.actual == final_actual
    assert report.project_actual_cny == Decimal("0.25")
    with pytest.raises(LedgerBudgetExceededError, match="run"):
        ledger.reserve(
            run_id="authoritative-run",
            query_id="q2",
            estimate=_estimate("0.20"),
            run_cap_cny=Decimal("0.30"),
        )


def test_checkpoint_actual_is_authoritative_after_reopen_before_ttl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = _ledger(path)
    reservation = ledger.reserve(
        run_id="reopen-before-ttl",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("0.30"),
    )
    ledger.checkpoint_actual(reservation, _actual("0.25"))

    reopened = _ledger(path, clock=lambda: NOW + timedelta(seconds=1))
    report = reopened.report("reopen-before-ttl")

    assert report.reserved == UsageEstimate(cost_cny=Decimal("0"))
    assert report.actual == _actual("0.25")
    assert report.project_actual_cny == Decimal("0.25")


def test_finalize_controller_does_not_commit_controller_before_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite3")
    ledger_reservation = ledger.reserve(
        run_id="crash-before-checkpoint",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    controller = HardBudgetController(
        SearchBudget(
            max_search_api_calls=2,
            target_search_api_calls=1,
            max_llm_calls=1,
            target_llm_calls=0,
            max_total_tokens=1,
            max_cost_cny=1.0,
            max_elapsed_seconds=10,
            soft_deadline_seconds=9,
        ),
        formal_live=True,
        clock=lambda: NOW,
    )
    request = controller.reserve(
        "provider.search",
        UsageEstimate(search_api_calls=1, cost_cny=Decimal("0.30")),
    )
    controller.mark_dispatched(request)

    def crash_before_checkpoint(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("crash before checkpoint")

    monkeypatch.setattr(ledger, "checkpoint_actual", crash_before_checkpoint)
    with pytest.raises(RuntimeError, match="before checkpoint"):
        ledger.finalize_controller_actual(
            ledger_reservation=ledger_reservation,
            controller=controller,
            request_reservation=request,
            actual=_actual("0.25"),
        )

    assert controller.committed_usage == UsageActual()
    assert controller.reserved_usage.search_api_calls == 1
    report = ledger.report("crash-before-checkpoint")
    assert report.reserved.cost_cny == Decimal("0.10")
    assert report.actual == UsageActual(cost_cny=Decimal("0"))


def test_finalize_controller_recovers_after_checkpoint_and_is_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = _ledger(path)
    ledger_reservation = ledger.reserve(
        run_id="crash-after-checkpoint",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    controller = HardBudgetController(
        SearchBudget(
            max_search_api_calls=2,
            target_search_api_calls=1,
            max_llm_calls=1,
            target_llm_calls=0,
            max_total_tokens=1,
            max_cost_cny=1.0,
            max_elapsed_seconds=10,
            soft_deadline_seconds=9,
        ),
        formal_live=True,
        clock=lambda: NOW,
    )
    request = controller.reserve(
        "provider.search",
        UsageEstimate(search_api_calls=1, cost_cny=Decimal("0.30")),
    )
    controller.mark_dispatched(request)
    original_settle = controller.settle

    def crash_after_checkpoint(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("crash after checkpoint")

    monkeypatch.setattr(controller, "settle", crash_after_checkpoint)
    with pytest.raises(RuntimeError, match="after checkpoint"):
        ledger.finalize_controller_actual(
            ledger_reservation=ledger_reservation,
            controller=controller,
            request_reservation=request,
            actual=_actual("0.25"),
        )

    reopened = _ledger(path, clock=lambda: NOW + timedelta(seconds=1))
    assert reopened.report("crash-after-checkpoint").actual == _actual("0.25")
    monkeypatch.setattr(controller, "settle", original_settle)
    reopened.finalize_controller_actual(
        ledger_reservation=ledger_reservation,
        controller=controller,
        request_reservation=request,
        actual=_actual("0.25"),
    )
    reopened.finalize_controller_actual(
        ledger_reservation=ledger_reservation,
        controller=controller,
        request_reservation=request,
        actual=_actual("0.25"),
    )

    assert controller.committed_usage == _actual("0.25")
    assert reopened.report("crash-after-checkpoint").actual == _actual("0.25")


def test_finalize_controller_recovers_crash_after_controller_before_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite3")
    ledger_reservation = ledger.reserve(
        run_id="controller-final-gap",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    controller = HardBudgetController(
        SearchBudget(
            max_search_api_calls=2,
            target_search_api_calls=1,
            max_llm_calls=1,
            target_llm_calls=0,
            max_total_tokens=1,
            max_cost_cny=1.0,
            max_elapsed_seconds=10,
            soft_deadline_seconds=9,
        ),
        formal_live=True,
        clock=lambda: NOW,
    )
    request = controller.reserve(
        "provider.search",
        UsageEstimate(search_api_calls=1, cost_cny=Decimal("0.30")),
    )
    controller.mark_dispatched(request)
    original_settle = ledger.settle
    crashed = False

    def crash_once(*args: object, **kwargs: object) -> None:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("crash after controller final")
        original_settle(*args, **kwargs)

    monkeypatch.setattr(ledger, "settle", crash_once)
    with pytest.raises(RuntimeError, match="after controller final"):
        ledger.finalize_controller_actual(
            ledger_reservation=ledger_reservation,
            controller=controller,
            request_reservation=request,
            actual=_actual("0.25"),
        )

    assert ledger.report("controller-final-gap").actual == _actual("0.25")
    ledger.finalize_controller_actual(
        ledger_reservation=ledger_reservation,
        controller=controller,
        request_reservation=request,
        actual=_actual("0.25"),
    )
    ledger.finalize_controller_actual(
        ledger_reservation=ledger_reservation,
        controller=controller,
        request_reservation=request,
        actual=_actual("0.25"),
    )

    assert controller.committed_usage == _actual("0.25")
    assert ledger.report("controller-final-gap").actual == _actual("0.25")


def test_finalize_retry_after_controller_ttl_does_not_terminalize_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = NOW
    path = tmp_path / "ledger.sqlite3"
    ledger = _ledger(path, clock=lambda: current)
    ledger_reservation = ledger.reserve(
        run_id="controller-ttl",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    controller, request = _controller_request(
        clock=lambda: current,
        ttl_seconds=2,
    )
    original_settle = controller.settle

    def crash_after_checkpoint(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("crash after checkpoint")

    monkeypatch.setattr(controller, "settle", crash_after_checkpoint)
    with pytest.raises(RuntimeError, match="after checkpoint"):
        ledger.finalize_controller_actual(
            ledger_reservation=ledger_reservation,
            controller=controller,
            request_reservation=request,
            actual=_actual("0.25"),
        )
    monkeypatch.setattr(controller, "settle", original_settle)
    current += timedelta(seconds=3)

    for _ in range(2):
        with pytest.raises(ReservationError, match="unknown or already settled"):
            ledger.finalize_controller_actual(
                ledger_reservation=ledger_reservation,
                controller=controller,
                request_reservation=request,
                actual=_actual("0.25"),
            )

    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            "SELECT state, checkpoint_present FROM reservations"
        ).fetchone()
    assert stored == ("reserved", 1)
    assert controller.committed_usage == UsageActual()
    assert ledger.report("controller-ttl").actual == _actual("0.25")


def test_finalize_repeated_permanent_reservation_error_keeps_ledger_prepared(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = _ledger(path)
    ledger_reservation = ledger.reserve(
        run_id="permanent-error",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    controller, request = _controller_request(
        clock=lambda: NOW,
        estimate_cost="0.20",
    )

    for _ in range(2):
        with pytest.raises(ReservationError, match="cost_cny exceeds"):
            ledger.finalize_controller_actual(
                ledger_reservation=ledger_reservation,
                controller=controller,
                request_reservation=request,
                actual=_actual("0.25"),
            )

    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            "SELECT state, checkpoint_present FROM reservations"
        ).fetchone()
    assert stored == ("reserved", 1)
    assert controller.committed_usage == UsageActual()


def test_finalize_rejects_mismatched_request_reservation_on_every_retry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = _ledger(path)
    ledger_reservation = ledger.reserve(
        run_id="mismatched-request",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    controller, request = _controller_request(clock=lambda: NOW)
    mismatched = request.model_copy(update={"action": "provider.other"})

    for _ in range(2):
        with pytest.raises(ReservationError, match="does not match"):
            ledger.finalize_controller_actual(
                ledger_reservation=ledger_reservation,
                controller=controller,
                request_reservation=mismatched,
                actual=_actual("0.25"),
            )

    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            "SELECT state, checkpoint_present FROM reservations"
        ).fetchone()
    assert stored == ("reserved", 1)
    assert controller.committed_usage == UsageActual()


def test_stale_recovery_preserves_checkpoint_until_controller_reconciles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = NOW
    path = tmp_path / "ledger.sqlite3"
    ledger = _ledger(path, clock=lambda: current)
    ledger_reservation = ledger.reserve(
        run_id="stale-prepared",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    controller, request = _controller_request(clock=lambda: NOW)
    original_settle = controller.settle

    def crash_after_checkpoint(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("crash after checkpoint")

    monkeypatch.setattr(controller, "settle", crash_after_checkpoint)
    with pytest.raises(RuntimeError, match="after checkpoint"):
        ledger.finalize_controller_actual(
            ledger_reservation=ledger_reservation,
            controller=controller,
            request_reservation=request,
            actual=_actual("0.25"),
        )
    current += timedelta(minutes=1)
    reopened = _ledger(path, clock=lambda: current)

    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            "SELECT state, checkpoint_present FROM reservations"
        ).fetchone()
    assert stored == ("reserved", 1)
    assert reopened.report("stale-prepared").actual == _actual("0.25")

    monkeypatch.setattr(controller, "settle", original_settle)
    reopened.finalize_controller_actual(
        ledger_reservation=ledger_reservation,
        controller=controller,
        request_reservation=request,
        actual=_actual("0.25"),
    )
    assert reopened.report("stale-prepared").actual == _actual("0.25")


def test_finalize_rejects_opposite_terminal_mode_with_identical_usage(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path / "ledger.sqlite3")
    ledger_reservation = ledger.reserve(
        run_id="opposite-mode",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    controller, request = _controller_request(clock=lambda: NOW)
    actual = _actual("0.25")
    ledger.finalize_controller_actual(
        ledger_reservation=ledger_reservation,
        controller=controller,
        request_reservation=request,
        actual=actual,
    )

    with pytest.raises(LedgerReservationError, match="terminal mode"):
        ledger.finalize_controller_actual(
            ledger_reservation=ledger_reservation,
            controller=controller,
            request_reservation=request,
            actual=actual,
            failed=True,
        )

    assert controller.committed_usage == actual
    with sqlite3.connect(tmp_path / "ledger.sqlite3") as connection:
        assert connection.execute("SELECT state FROM reservations").fetchone() == (
            "settled",
        )


def test_malformed_restored_receipt_cannot_terminalize_prepared_ledger(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = _ledger(path)
    ledger_reservation = ledger.reserve(
        run_id="forged-restore",
        query_id="q1",
        estimate=_estimate("0.10"),
        run_cap_cny=Decimal("1.00"),
    )
    forged_actual = _actual("0.25")
    ledger.checkpoint_actual(ledger_reservation, forged_actual)

    legacy_controller, legacy_request = _controller_request(clock=lambda: NOW)
    legacy_controller.settle(legacy_request, _actual("0.10"))
    legacy_state = legacy_controller.export_state()
    legacy_state["version"] = 3
    legacy_state.pop("terminal_outcomes")
    legacy_state.pop("terminal_outcomes_complete")
    partial = HardBudgetController.from_state(
        legacy_controller.budget,
        legacy_state,
        clock=lambda: NOW,
    )
    forged_controller, forged_request = _controller_request(clock=lambda: NOW)
    forged_state = partial.export_state()
    forged_state["terminal_outcomes"].append(
        {
            "reservation": forged_request.model_dump(mode="json"),
            "mode": "settled",
            "actual": forged_actual.model_dump(mode="json"),
        }
    )

    try:
        restored = HardBudgetController.from_state(
            forged_controller.budget,
            forged_state,
            clock=lambda: NOW,
        )
    except ValueError:
        pass
    else:
        ledger.finalize_controller_actual(
            ledger_reservation=ledger_reservation,
            controller=restored,
            request_reservation=forged_request,
            actual=forged_actual,
        )

    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            "SELECT state, checkpoint_present FROM reservations"
        ).fetchone()
    assert stored == ("reserved", 1)
def test_report_preserves_reservation_creation_order(tmp_path: Path) -> None:
    ledger = SQLiteBudgetLedger(tmp_path / "ledger.sqlite3")
    for query_id in ("q2", "q1"):
        reservation = ledger.reserve(
            run_id="ordered-run",
            query_id=query_id,
            estimate=UsageEstimate(cost_cny=Decimal("0.1")),
            run_cap_cny=Decimal("18"),
        )
        ledger.settle(reservation, UsageActual(cost_cny=Decimal("0.1")))

    report = ledger.report("ordered-run")

    assert [receipt.query_id for receipt in report.receipts] == ["q2", "q1"]
