"""Persistent hierarchical request, run, and project budget accounting."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Literal
from uuid import uuid4

from paper_search.domain.models import (
    DomainModel,
    MoneyCny,
    NonEmptyStr,
    UsageActual,
    UsageEstimate,
)


Clock = Callable[[], datetime]
_MICRO_CNY = Decimal("0.000001")
REQUEST_HARD_CAP_CNY = Decimal("0.30")
DEV_RUN_CAP_CNY = Decimal("18.00")
VALIDATION_RUN_CAP_CNY = Decimal("9.00")
PROJECT_SOFT_STOP_CNY = Decimal("160.00")
PROJECT_HARD_CAP_CNY = Decimal("200.00")


class LedgerError(RuntimeError):
    """Base class for fixed ledger failures."""


class LedgerReservationError(LedgerError):
    """A reservation is unknown, mismatched, duplicate, or already terminal."""


class LedgerBudgetExceededError(LedgerError):
    """A request, run, or project hard cap would be exceeded."""


class LedgerSoftStopError(LedgerError):
    """The project has reached its operator-defined soft stop."""


class LedgerReservation(DomainModel):
    reservation_id: NonEmptyStr
    run_id: NonEmptyStr
    query_id: NonEmptyStr
    estimate: UsageEstimate
    state: Literal["reserved", "settled", "failed"]


class LedgerReport(DomainModel):
    run_id: NonEmptyStr
    reserved: UsageEstimate
    actual: UsageActual
    run_cap_cny: MoneyCny
    project_actual_cny: MoneyCny
    project_soft_stop_cny: MoneyCny
    project_hard_cap_cny: MoneyCny
    within_caps: bool


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_micro(value: Decimal) -> int:
    decimal_value = Decimal(value)
    if not decimal_value.is_finite() or decimal_value < 0:
        raise ValueError("CNY amount must be finite and non-negative")
    if decimal_value.quantize(_MICRO_CNY) != decimal_value:
        raise ValueError("CNY amount must use at most six decimal places")
    return int(decimal_value / _MICRO_CNY)


def _from_micro(value: int) -> Decimal:
    return (Decimal(value) * _MICRO_CNY).quantize(_MICRO_CNY)


def _usage_values(usage: UsageEstimate) -> tuple[int, int, int, int, int, int | None]:
    return (
        usage.search_api_calls,
        usage.llm_calls,
        usage.input_tokens,
        usage.output_tokens,
        usage.elapsed_ms,
        _to_micro(usage.cost_cny) if usage.cost_cny is not None else None,
    )


def _usage_from_row(
    row: sqlite3.Row,
    *,
    prefix: str,
    actual: bool,
) -> UsageEstimate:
    cost_micro = row[f"{prefix}_cost_micro_cny"]
    values = {
        "search_api_calls": row[f"{prefix}_search_api_calls"],
        "llm_calls": row[f"{prefix}_llm_calls"],
        "input_tokens": row[f"{prefix}_input_tokens"],
        "output_tokens": row[f"{prefix}_output_tokens"],
        "elapsed_ms": row[f"{prefix}_elapsed_ms"],
        "cost_cny": _from_micro(cost_micro) if cost_micro is not None else None,
    }
    model = UsageActual if actual else UsageEstimate
    return model.model_validate(values)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_cap_micro_cny INTEGER NOT NULL CHECK (run_cap_micro_cny >= 0),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reservations (
    reservation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    query_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('reserved', 'settled', 'failed')),
    estimate_search_api_calls INTEGER NOT NULL CHECK (estimate_search_api_calls >= 0),
    estimate_llm_calls INTEGER NOT NULL CHECK (estimate_llm_calls >= 0),
    estimate_input_tokens INTEGER NOT NULL CHECK (estimate_input_tokens >= 0),
    estimate_output_tokens INTEGER NOT NULL CHECK (estimate_output_tokens >= 0),
    estimate_elapsed_ms INTEGER NOT NULL CHECK (estimate_elapsed_ms >= 0),
    estimate_cost_micro_cny INTEGER NOT NULL CHECK (estimate_cost_micro_cny >= 0),
    actual_search_api_calls INTEGER NOT NULL DEFAULT 0 CHECK (actual_search_api_calls >= 0),
    actual_llm_calls INTEGER NOT NULL DEFAULT 0 CHECK (actual_llm_calls >= 0),
    actual_input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (actual_input_tokens >= 0),
    actual_output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (actual_output_tokens >= 0),
    actual_elapsed_ms INTEGER NOT NULL DEFAULT 0 CHECK (actual_elapsed_ms >= 0),
    actual_cost_micro_cny INTEGER CHECK (
        actual_cost_micro_cny IS NULL OR actual_cost_micro_cny >= 0
    ),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    terminal_at TEXT,
    UNIQUE (run_id, query_id)
);

CREATE INDEX IF NOT EXISTS ix_reservations_run_state
ON reservations(run_id, state);
"""


class SQLiteBudgetLedger:
    """SQLite-backed atomic hierarchy over request, run, and project spend."""

    def __init__(
        self,
        path: str | Path,
        *,
        project_soft_stop_cny: MoneyCny = PROJECT_SOFT_STOP_CNY,
        project_hard_cap_cny: MoneyCny = PROJECT_HARD_CAP_CNY,
        reservation_ttl_seconds: int = 300,
        clock: Clock = _utc_now,
        replay: bool = False,
    ) -> None:
        if type(reservation_ttl_seconds) is not int or reservation_ttl_seconds <= 0:
            raise ValueError("reservation_ttl_seconds must be a positive integer")
        self._soft_micro = _to_micro(Decimal(project_soft_stop_cny))
        self._hard_micro = _to_micro(Decimal(project_hard_cap_cny))
        if self._soft_micro >= self._hard_micro:
            raise ValueError("project soft stop must be below project hard cap")
        self._clock = clock
        self._ttl = reservation_ttl_seconds
        self._replay = replay
        self._memory_lock = threading.RLock()
        self._memory: sqlite3.Connection | None = None
        if replay:
            self._path: Path | None = None
            self._memory = sqlite3.connect(
                ":memory:",
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
            self._configure(self._memory, wal=False)
            self._memory.executescript(_SCHEMA)
        else:
            self._path = Path(path).resolve()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._connection() as connection:
                connection.executescript(_SCHEMA)
            with self._immediate() as connection:
                self._recover_locked(connection)

    @staticmethod
    def _configure(connection: sqlite3.Connection, *, wal: bool) -> None:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        if wal:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._memory is not None:
            with self._memory_lock:
                yield self._memory
            return
        if self._path is None:
            raise RuntimeError("ledger path is unavailable")
        connection = sqlite3.connect(
            self._path,
            timeout=30,
            isolation_level=None,
        )
        try:
            self._configure(connection, wal=True)
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _recover_locked(self, connection: sqlite3.Connection) -> None:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("ledger clock must return a timezone-aware datetime")
        if self._replay:
            connection.execute(
                """
                UPDATE reservations
                SET state = 'failed',
                    actual_search_api_calls = 0,
                    actual_llm_calls = 0,
                    actual_input_tokens = 0,
                    actual_output_tokens = 0,
                    actual_elapsed_ms = 0,
                    actual_cost_micro_cny = 0,
                    terminal_at = ?
                WHERE state = 'reserved' AND expires_at <= ?
                """,
                (now.isoformat(), now.isoformat()),
            )
        else:
            connection.execute(
                """
                UPDATE reservations
                SET state = 'failed',
                    actual_search_api_calls = estimate_search_api_calls,
                    actual_llm_calls = estimate_llm_calls,
                    actual_input_tokens = estimate_input_tokens,
                    actual_output_tokens = estimate_output_tokens,
                    actual_elapsed_ms = estimate_elapsed_ms,
                    actual_cost_micro_cny = estimate_cost_micro_cny,
                    terminal_at = ?
                WHERE state = 'reserved' AND expires_at <= ?
                """,
                (now.isoformat(), now.isoformat()),
            )

    @staticmethod
    def _project_totals(connection: sqlite3.Connection) -> tuple[int, int, int]:
        row = connection.execute(
            """
            SELECT
                COALESCE(SUM(CASE
                    WHEN state IN ('settled', 'failed')
                    THEN actual_cost_micro_cny ELSE 0 END), 0) AS actual_micro,
                COALESCE(SUM(CASE
                    WHEN state = 'reserved'
                    THEN estimate_cost_micro_cny ELSE 0 END), 0) AS reserved_micro,
                COALESCE(SUM(CASE
                    WHEN state IN ('settled', 'failed')
                         AND actual_cost_micro_cny IS NULL
                    THEN 1 ELSE 0 END), 0) AS unknown_count
            FROM reservations
            """
        ).fetchone()
        if row is None:
            return 0, 0, 0
        return row["actual_micro"], row["reserved_micro"], row["unknown_count"]

    @staticmethod
    def _run_totals(
        connection: sqlite3.Connection,
        run_id: str,
    ) -> tuple[int, int, int]:
        row = connection.execute(
            """
            SELECT
                COALESCE(SUM(CASE
                    WHEN state IN ('settled', 'failed')
                    THEN actual_cost_micro_cny ELSE 0 END), 0) AS actual_micro,
                COALESCE(SUM(CASE
                    WHEN state = 'reserved'
                    THEN estimate_cost_micro_cny ELSE 0 END), 0) AS reserved_micro,
                COALESCE(SUM(CASE
                    WHEN state IN ('settled', 'failed')
                         AND actual_cost_micro_cny IS NULL
                    THEN 1 ELSE 0 END), 0) AS unknown_count
            FROM reservations
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return 0, 0, 0
        return row["actual_micro"], row["reserved_micro"], row["unknown_count"]

    def reserve(
        self,
        *,
        run_id: str,
        query_id: str,
        estimate: UsageEstimate,
        run_cap_cny: MoneyCny,
    ) -> LedgerReservation:
        if not run_id.strip() or not query_id.strip():
            raise ValueError("run_id and query_id must not be empty")
        estimate = UsageEstimate.model_validate(estimate.model_dump(mode="python"))
        if estimate.cost_cny is None:
            raise LedgerReservationError("live reservation requires a known cost estimate")
        estimate_micro = _to_micro(estimate.cost_cny)
        request_cap_micro = _to_micro(REQUEST_HARD_CAP_CNY)
        if not self._replay and estimate_micro > request_cap_micro:
            raise LedgerBudgetExceededError("request cost hard cap exceeded")
        run_cap_micro = _to_micro(Decimal(run_cap_cny))
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("ledger clock must return a timezone-aware datetime")
        reservation = LedgerReservation(
            reservation_id=str(uuid4()),
            run_id=run_id,
            query_id=query_id,
            estimate=estimate,
            state="reserved",
        )
        with self._immediate() as connection:
            self._recover_locked(connection)
            existing_run = connection.execute(
                "SELECT run_cap_micro_cny FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if existing_run is None:
                connection.execute(
                    "INSERT INTO runs(run_id, run_cap_micro_cny, created_at) VALUES (?, ?, ?)",
                    (run_id, run_cap_micro, now.isoformat()),
                )
            elif existing_run["run_cap_micro_cny"] != run_cap_micro:
                raise LedgerReservationError("run cap does not match the existing run")

            duplicate = connection.execute(
                "SELECT 1 FROM reservations WHERE run_id = ? AND query_id = ?",
                (run_id, query_id),
            ).fetchone()
            if duplicate is not None:
                raise LedgerReservationError(
                    "query already has a ledger reservation"
                )

            run_actual, run_reserved, run_unknown = self._run_totals(
                connection,
                run_id,
            )
            if run_unknown:
                raise LedgerBudgetExceededError("run has unknown committed cost")
            if not self._replay and run_actual + run_reserved + estimate_micro > run_cap_micro:
                raise LedgerBudgetExceededError("run cost hard cap exceeded")

            project_actual, project_reserved, project_unknown = self._project_totals(
                connection
            )
            projected = project_actual + project_reserved + estimate_micro
            if not self._replay:
                if project_unknown or projected > self._hard_micro:
                    raise LedgerBudgetExceededError("project hard cap exceeded")
                if project_actual >= self._soft_micro or projected >= self._soft_micro:
                    raise LedgerSoftStopError("project soft stop reached")

            values = _usage_values(estimate)
            try:
                connection.execute(
                    """
                    INSERT INTO reservations(
                        reservation_id, run_id, query_id, state,
                        estimate_search_api_calls, estimate_llm_calls,
                        estimate_input_tokens, estimate_output_tokens,
                        estimate_elapsed_ms, estimate_cost_micro_cny,
                        created_at, expires_at
                    ) VALUES (?, ?, ?, 'reserved', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reservation.reservation_id,
                        run_id,
                        query_id,
                        *values,
                        now.isoformat(),
                        (now + timedelta(seconds=self._ttl)).isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                raise LedgerReservationError(
                    "query already has a ledger reservation"
                ) from None
        return reservation

    def _terminal(
        self,
        reservation: LedgerReservation,
        actual: UsageActual,
        *,
        state: Literal["settled", "failed"],
    ) -> None:
        if reservation.state != "reserved":
            raise LedgerReservationError("reservation is already terminal")
        actual = UsageActual.model_validate(actual.model_dump(mode="python"))
        if state == "settled" and actual.cost_cny is None and not self._replay:
            raise LedgerReservationError("settlement requires a known actual cost")
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("ledger clock must return a timezone-aware datetime")
        with self._immediate() as connection:
            self._recover_locked(connection)
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation.reservation_id,),
            ).fetchone()
            if row is None:
                raise LedgerReservationError("reservation is unknown")
            if row["state"] != "reserved":
                raise LedgerReservationError("reservation is already terminal")
            stored = LedgerReservation(
                reservation_id=row["reservation_id"],
                run_id=row["run_id"],
                query_id=row["query_id"],
                estimate=_usage_from_row(row, prefix="estimate", actual=False),
                state="reserved",
            )
            if stored != reservation:
                raise LedgerReservationError("reservation does not match stored state")
            if self._replay:
                values = _usage_values(UsageActual(cost_cny=Decimal("0")))
            else:
                values = _usage_values(actual)
            connection.execute(
                """
                UPDATE reservations
                SET state = ?,
                    actual_search_api_calls = ?, actual_llm_calls = ?,
                    actual_input_tokens = ?, actual_output_tokens = ?,
                    actual_elapsed_ms = ?, actual_cost_micro_cny = ?,
                    terminal_at = ?
                WHERE reservation_id = ? AND state = 'reserved'
                """,
                (
                    state,
                    *values,
                    now.isoformat(),
                    reservation.reservation_id,
                ),
            )

    def settle(
        self,
        reservation: LedgerReservation,
        actual: UsageActual,
    ) -> None:
        self._terminal(reservation, actual, state="settled")

    def fail(
        self,
        reservation: LedgerReservation,
        actual: UsageActual,
    ) -> None:
        self._terminal(reservation, actual, state="failed")

    @staticmethod
    def _aggregate_usage(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        state: str,
        actual: bool,
    ) -> UsageEstimate:
        prefix = "actual" if actual else "estimate"
        row = connection.execute(
            f"""
            SELECT
                COALESCE(SUM({prefix}_search_api_calls), 0) AS search_api_calls,
                COALESCE(SUM({prefix}_llm_calls), 0) AS llm_calls,
                COALESCE(SUM({prefix}_input_tokens), 0) AS input_tokens,
                COALESCE(SUM({prefix}_output_tokens), 0) AS output_tokens,
                COALESCE(SUM({prefix}_elapsed_ms), 0) AS elapsed_ms,
                COALESCE(SUM({prefix}_cost_micro_cny), 0) AS cost_micro,
                COALESCE(SUM(CASE WHEN {prefix}_cost_micro_cny IS NULL THEN 1 ELSE 0 END), 0)
                    AS unknown_count
            FROM reservations
            WHERE run_id = ? AND state {state}
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("ledger aggregate query returned no row")
        cost = None if row["unknown_count"] else _from_micro(row["cost_micro"])
        values = {
            "search_api_calls": row["search_api_calls"],
            "llm_calls": row["llm_calls"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "elapsed_ms": row["elapsed_ms"],
            "cost_cny": cost,
        }
        model = UsageActual if actual else UsageEstimate
        return model.model_validate(values)

    def report(self, run_id: str) -> LedgerReport:
        with self._immediate() as connection:
            self._recover_locked(connection)
            run = connection.execute(
                "SELECT run_cap_micro_cny FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise LedgerReservationError("run is unknown")
            reserved = self._aggregate_usage(
                connection,
                run_id=run_id,
                state="= 'reserved'",
                actual=False,
            )
            actual = self._aggregate_usage(
                connection,
                run_id=run_id,
                state="IN ('settled', 'failed')",
                actual=True,
            )
            project_actual, project_reserved, project_unknown = (
                self._project_totals(connection)
            )
            over_request = connection.execute(
                """
                SELECT COUNT(*)
                FROM reservations
                WHERE run_id = ? AND state IN ('settled', 'failed')
                  AND actual_cost_micro_cny > ?
                """,
                (run_id, _to_micro(REQUEST_HARD_CAP_CNY)),
            ).fetchone()[0]
            run_cap_micro = run["run_cap_micro_cny"]
            run_known_cost = (
                _to_micro(actual.cost_cny) if actual.cost_cny is not None else 0
            )
            reserved_cost = _to_micro(reserved.cost_cny or Decimal("0"))
            within_caps = (
                actual.cost_cny is not None
                and not project_unknown
                and not over_request
                and run_known_cost + reserved_cost <= run_cap_micro
                and project_actual + project_reserved <= self._hard_micro
            )
            return LedgerReport(
                run_id=run_id,
                reserved=reserved,
                actual=UsageActual.model_validate(actual.model_dump()),
                run_cap_cny=_from_micro(run_cap_micro),
                project_actual_cny=(
                    Decimal("0") if self._replay else _from_micro(project_actual)
                ),
                project_soft_stop_cny=_from_micro(self._soft_micro),
                project_hard_cap_cny=_from_micro(self._hard_micro),
                within_caps=within_caps,
            )

    def close(self) -> None:
        if self._memory is not None:
            with self._memory_lock:
                self._memory.close()
                self._memory = None


__all__ = [
    "DEV_RUN_CAP_CNY",
    "LedgerBudgetExceededError",
    "LedgerError",
    "LedgerReport",
    "LedgerReservation",
    "LedgerReservationError",
    "LedgerSoftStopError",
    "PROJECT_HARD_CAP_CNY",
    "PROJECT_SOFT_STOP_CNY",
    "REQUEST_HARD_CAP_CNY",
    "SQLiteBudgetLedger",
    "VALIDATION_RUN_CAP_CNY",
]
