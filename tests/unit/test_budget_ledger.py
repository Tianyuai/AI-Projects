from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from paper_search.domain.models import UsageActual, UsageEstimate
from paper_search.domain.models import SearchBudget
from paper_search.control.budget import HardBudgetController
from paper_search.control.ledger import (
    LedgerBudgetExceededError,
    LedgerReservationError,
    LedgerSoftStopError,
    SQLiteBudgetLedger,
)


NOW = datetime(2026, 8, 2, 6, 0, tzinfo=UTC)


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
    with pytest.raises(LedgerReservationError, match="already terminal"):
        recovered.settle(ledger_reservation, final_actual)
