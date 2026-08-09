"""Persistent hierarchical request, run, and project budget accounting."""

from __future__ import annotations

import hashlib
import json
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
    BudgetReservation,
    DomainModel,
    MoneyCny,
    NonEmptyStr,
    NonNegativeInt,
    Sha256,
    UsageActual,
    UsageEstimate,
)
from paper_search.control.budget import HardBudgetController, ReservationError


Clock = Callable[[], datetime]
_MICRO_CNY = Decimal("0.000001")
REQUEST_HARD_CAP_CNY = Decimal("0.30")
DEV_RUN_CAP_CNY = Decimal("18.00")
VALIDATION_RUN_CAP_CNY = Decimal("9.00")
PROJECT_SOFT_STOP_CNY = Decimal("160.00")
PROJECT_HARD_CAP_CNY = Decimal("200.00")
_SCHEMA_VERSION = 2
_POLICY_VERSION = "budget-ledger-policy-v1"


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


class LedgerReceipt(DomainModel):
    reservation_id: NonEmptyStr
    run_id: NonEmptyStr
    query_id: NonEmptyStr
    estimate: UsageEstimate
    state: Literal["reserved", "settled", "failed"]
    actual: UsageActual | None


class LedgerReport(DomainModel):
    policy_version: Literal["budget-ledger-policy-v1"] = "budget-ledger-policy-v1"
    run_id: NonEmptyStr
    reserved: UsageEstimate
    actual: UsageActual
    run_cap_cny: MoneyCny
    project_actual_cny: MoneyCny
    project_soft_stop_cny: MoneyCny
    project_hard_cap_cny: MoneyCny
    within_caps: bool
    receipts: list[LedgerReceipt] = []
    project_receipt_count: NonNegativeInt = 0
    project_receipts_sha256: Sha256 = (
        "sha256:37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"
    )


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
CREATE TABLE IF NOT EXISTS ledger_metadata (
    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    schema_version INTEGER NOT NULL,
    policy_version TEXT NOT NULL,
    request_cap_micro_cny INTEGER NOT NULL,
    project_soft_stop_micro_cny INTEGER NOT NULL,
    project_hard_cap_micro_cny INTEGER NOT NULL
);

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
    checkpoint_present INTEGER NOT NULL DEFAULT 0 CHECK (checkpoint_present IN (0, 1)),
    checkpoint_search_api_calls INTEGER NOT NULL DEFAULT 0 CHECK (checkpoint_search_api_calls >= 0),
    checkpoint_llm_calls INTEGER NOT NULL DEFAULT 0 CHECK (checkpoint_llm_calls >= 0),
    checkpoint_input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (checkpoint_input_tokens >= 0),
    checkpoint_output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (checkpoint_output_tokens >= 0),
    checkpoint_elapsed_ms INTEGER NOT NULL DEFAULT 0 CHECK (checkpoint_elapsed_ms >= 0),
    checkpoint_cost_micro_cny INTEGER CHECK (
        checkpoint_cost_micro_cny IS NULL OR checkpoint_cost_micro_cny >= 0
    ),
    checkpointed_at TEXT,
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
        if not replay and (
            self._soft_micro > _to_micro(PROJECT_SOFT_STOP_CNY)
            or self._hard_micro > _to_micro(PROJECT_HARD_CAP_CNY)
        ):
            raise ValueError("project caps cannot exceed fixed production boundaries")
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
            with self._immediate() as connection:
                self._initialize_locked(connection)
        else:
            self._path = Path(path).resolve()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._immediate() as connection:
                self._initialize_locked(connection)

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
            self._configure(connection, wal=False)
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
                WHERE state = 'reserved' AND checkpoint_present = 0
                  AND expires_at <= ?
                """,
                (now.isoformat(), now.isoformat()),
            )
        else:
            connection.execute(
                """
                UPDATE reservations
                SET state = 'failed',
                    actual_search_api_calls = CASE checkpoint_present
                        WHEN 1 THEN checkpoint_search_api_calls ELSE estimate_search_api_calls END,
                    actual_llm_calls = CASE checkpoint_present
                        WHEN 1 THEN checkpoint_llm_calls ELSE estimate_llm_calls END,
                    actual_input_tokens = CASE checkpoint_present
                        WHEN 1 THEN checkpoint_input_tokens ELSE estimate_input_tokens END,
                    actual_output_tokens = CASE checkpoint_present
                        WHEN 1 THEN checkpoint_output_tokens ELSE estimate_output_tokens END,
                    actual_elapsed_ms = CASE checkpoint_present
                        WHEN 1 THEN checkpoint_elapsed_ms ELSE estimate_elapsed_ms END,
                    actual_cost_micro_cny = CASE checkpoint_present
                        WHEN 1 THEN checkpoint_cost_micro_cny ELSE estimate_cost_micro_cny END,
                    terminal_at = ?
                WHERE state = 'reserved' AND checkpoint_present = 0
                  AND expires_at <= ?
                """,
                (now.isoformat(), now.isoformat()),
            )

    def _initialize_locked(self, connection: sqlite3.Connection) -> None:
        for statement in _SCHEMA.split(";"):
            if statement.strip():
                connection.execute(statement)
        expected = (
            _SCHEMA_VERSION,
            _POLICY_VERSION,
            _to_micro(REQUEST_HARD_CAP_CNY),
            self._soft_micro,
            self._hard_micro,
        )
        row = connection.execute(
            """
            SELECT schema_version, policy_version, request_cap_micro_cny,
                   project_soft_stop_micro_cny, project_hard_cap_micro_cny
            FROM ledger_metadata WHERE singleton_id = 1
            """
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO ledger_metadata(
                    singleton_id, schema_version, policy_version,
                    request_cap_micro_cny, project_soft_stop_micro_cny,
                    project_hard_cap_micro_cny
                ) VALUES (1, ?, ?, ?, ?, ?)
                """,
                expected,
            )
        elif tuple(row) != expected:
            raise LedgerReservationError("ledger metadata does not match configured policy")
        self._recover_locked(connection)

    @staticmethod
    def _project_totals(connection: sqlite3.Connection) -> tuple[int, int, int]:
        row = connection.execute(
            """
            SELECT
                COALESCE(SUM(CASE
                    WHEN state IN ('settled', 'failed')
                    THEN actual_cost_micro_cny ELSE 0 END), 0) AS actual_micro,
                COALESCE(SUM(CASE
                    WHEN state = 'reserved' AND checkpoint_present = 0
                    THEN estimate_cost_micro_cny ELSE 0 END), 0) AS reserved_micro,
                COALESCE(SUM(CASE
                    WHEN state IN ('settled', 'failed')
                         AND actual_cost_micro_cny IS NULL
                    THEN 1
                    WHEN state = 'reserved' AND checkpoint_present = 1
                         AND checkpoint_cost_micro_cny IS NULL
                    THEN 1 ELSE 0 END), 0) AS unknown_count
            FROM reservations
            """
        ).fetchone()
        if row is None:
            return 0, 0, 0
        checkpointed = connection.execute(
            """
            SELECT COALESCE(SUM(checkpoint_cost_micro_cny), 0)
            FROM reservations
            WHERE state = 'reserved' AND checkpoint_present = 1
            """
        ).fetchone()[0]
        return (
            row["actual_micro"] + checkpointed,
            row["reserved_micro"],
            row["unknown_count"],
        )

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
                    WHEN state = 'reserved' AND checkpoint_present = 0
                    THEN estimate_cost_micro_cny ELSE 0 END), 0) AS reserved_micro,
                COALESCE(SUM(CASE
                    WHEN state IN ('settled', 'failed')
                         AND actual_cost_micro_cny IS NULL
                    THEN 1
                    WHEN state = 'reserved' AND checkpoint_present = 1
                         AND checkpoint_cost_micro_cny IS NULL
                    THEN 1 ELSE 0 END), 0) AS unknown_count
            FROM reservations
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return 0, 0, 0
        checkpointed = connection.execute(
            """
            SELECT COALESCE(SUM(checkpoint_cost_micro_cny), 0)
            FROM reservations
            WHERE run_id = ? AND state = 'reserved' AND checkpoint_present = 1
            """,
            (run_id,),
        ).fetchone()[0]
        return (
            row["actual_micro"] + checkpointed,
            row["reserved_micro"],
            row["unknown_count"],
        )

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

    def checkpoint_actual(
        self,
        reservation: LedgerReservation,
        actual: UsageActual,
    ) -> None:
        """Durably checkpoint controller-final usage before ledger terminalization."""

        if reservation.state != "reserved":
            raise LedgerReservationError("reservation is already terminal")
        actual = UsageActual.model_validate(actual.model_dump(mode="python"))
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("ledger clock must return a timezone-aware datetime")
        checkpoint = actual
        values = _usage_values(checkpoint)
        with self._immediate() as connection:
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation.reservation_id,),
            ).fetchone()
            if row is None:
                raise LedgerReservationError("reservation is unknown")
            stored = LedgerReservation(
                reservation_id=row["reservation_id"],
                run_id=row["run_id"],
                query_id=row["query_id"],
                estimate=_usage_from_row(row, prefix="estimate", actual=False),
                state="reserved",
            )
            if stored != reservation:
                raise LedgerReservationError("reservation does not match stored state")
            if row["checkpoint_present"]:
                previous = UsageActual.model_validate(
                    _usage_from_row(row, prefix="checkpoint", actual=True).model_dump()
                )
                if previous != checkpoint:
                    raise LedgerReservationError("actual checkpoint does not match")
            elif row["state"] != "reserved":
                terminal = UsageActual.model_validate(
                    _usage_from_row(row, prefix="actual", actual=True).model_dump()
                )
                if terminal != checkpoint:
                    raise LedgerReservationError("reservation is already terminal")
                return
            else:
                connection.execute(
                    """
                    UPDATE reservations
                    SET checkpoint_present = 1,
                        checkpoint_search_api_calls = ?, checkpoint_llm_calls = ?,
                        checkpoint_input_tokens = ?, checkpoint_output_tokens = ?,
                        checkpoint_elapsed_ms = ?, checkpoint_cost_micro_cny = ?,
                        checkpointed_at = ?
                    WHERE reservation_id = ? AND state = 'reserved'
                    """,
                    (*values, now.isoformat(), reservation.reservation_id),
                )
            self._recover_locked(connection)

    def _coordinated_actual_state(
        self,
        reservation: LedgerReservation,
        actual: UsageActual,
    ) -> Literal["reserved", "checkpointed", "settled", "failed"]:
        expected = actual
        with self._immediate() as connection:
            self._recover_locked(connection)
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation.reservation_id,),
            ).fetchone()
            if row is None:
                raise LedgerReservationError("reservation is unknown")
            stored = LedgerReservation(
                reservation_id=row["reservation_id"],
                run_id=row["run_id"],
                query_id=row["query_id"],
                estimate=_usage_from_row(row, prefix="estimate", actual=False),
                state="reserved",
            )
            if stored != reservation:
                raise LedgerReservationError("reservation does not match stored state")
            if row["state"] != "reserved":
                terminal = UsageActual.model_validate(
                    _usage_from_row(row, prefix="actual", actual=True).model_dump()
                )
                if terminal != expected:
                    raise LedgerReservationError("terminal actual does not match")
                return "settled" if row["state"] == "settled" else "failed"
            if not row["checkpoint_present"]:
                return "reserved"
            checkpoint = UsageActual.model_validate(
                _usage_from_row(row, prefix="checkpoint", actual=True).model_dump()
            )
            if checkpoint != expected:
                raise LedgerReservationError("actual checkpoint does not match")
            return "checkpointed"

    @staticmethod
    def _controller_outcome_matches(
        controller: HardBudgetController,
        reservation: BudgetReservation,
        actual: UsageActual,
        mode: Literal["settled", "failed"],
    ) -> bool:
        outcome = controller.terminal_outcome(reservation)
        if outcome is None:
            return False
        terminal_mode, terminal_actual = outcome
        if terminal_mode != mode or terminal_actual != actual:
            raise LedgerReservationError(
                "controller terminal outcome does not match requested outcome"
            )
        return True

    def finalize_controller_actual(
        self,
        *,
        ledger_reservation: LedgerReservation,
        controller: HardBudgetController,
        request_reservation: BudgetReservation,
        actual: UsageActual,
        failed: bool = False,
    ) -> None:
        """Checkpoint first, then terminalize controller and ledger exactly once.

        The durable checkpoint is the prepared outbox state. Therefore controller
        finalization can never precede persistent actual usage, and a retry after
        either later transition can safely finish the ledger from that checkpoint.
        """

        actual = UsageActual.model_validate(actual.model_dump(mode="python"))
        requested_mode: Literal["settled", "failed"] = (
            "failed" if failed else "settled"
        )
        prior = self._coordinated_actual_state(ledger_reservation, actual)
        if prior in {"settled", "failed"}:
            if prior != requested_mode:
                raise LedgerReservationError(
                    "ledger terminal mode conflicts with requested terminal mode"
                )
            if not self._controller_outcome_matches(
                controller,
                request_reservation,
                actual,
                requested_mode,
            ):
                raise LedgerReservationError(
                    "controller has no matching terminal outcome"
                )
            return
        self.checkpoint_actual(ledger_reservation, actual)
        try:
            if failed:
                controller.fail_closed(request_reservation, actual)
            else:
                controller.settle(request_reservation, actual)
        except ReservationError:
            if not self._controller_outcome_matches(
                controller,
                request_reservation,
                actual,
                requested_mode,
            ):
                raise
        if not self._controller_outcome_matches(
            controller,
            request_reservation,
            actual,
            requested_mode,
        ):
            raise LedgerReservationError(
                "controller has no matching terminal outcome"
            )
        if failed:
            self.fail(ledger_reservation, actual)
        else:
            self.settle(ledger_reservation, actual)

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
                values = _usage_values(actual)
            else:
                if row["checkpoint_present"]:
                    checkpoint = UsageActual.model_validate(
                        _usage_from_row(
                            row,
                            prefix="checkpoint",
                            actual=True,
                        ).model_dump()
                    )
                    if checkpoint != actual:
                        raise LedgerReservationError(
                            "terminal actual does not match checkpoint"
                        )
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
        actual: bool,
    ) -> UsageEstimate:
        if actual:
            expressions = {
                field: (
                    f"CASE WHEN state = 'reserved' THEN checkpoint_{field} "
                    f"ELSE actual_{field} END"
                )
                for field in (
                    "search_api_calls",
                    "llm_calls",
                    "input_tokens",
                    "output_tokens",
                    "elapsed_ms",
                    "cost_micro_cny",
                )
            }
            predicate = (
                "state IN ('settled', 'failed') OR "
                "(state = 'reserved' AND checkpoint_present = 1)"
            )
        else:
            expressions = {
                field: f"estimate_{field}"
                for field in (
                    "search_api_calls",
                    "llm_calls",
                    "input_tokens",
                    "output_tokens",
                    "elapsed_ms",
                    "cost_micro_cny",
                )
            }
            predicate = "state = 'reserved' AND checkpoint_present = 0"
        cost_expression = expressions["cost_micro_cny"]
        row = connection.execute(
            f"""
            SELECT
                COALESCE(SUM({expressions['search_api_calls']}), 0) AS search_api_calls,
                COALESCE(SUM({expressions['llm_calls']}), 0) AS llm_calls,
                COALESCE(SUM({expressions['input_tokens']}), 0) AS input_tokens,
                COALESCE(SUM({expressions['output_tokens']}), 0) AS output_tokens,
                COALESCE(SUM({expressions['elapsed_ms']}), 0) AS elapsed_ms,
                COALESCE(SUM({cost_expression}), 0) AS cost_micro,
                COALESCE(SUM(CASE WHEN {cost_expression} IS NULL THEN 1 ELSE 0 END), 0)
                    AS unknown_count
            FROM reservations
            WHERE run_id = ? AND ({predicate})
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
                actual=False,
            )
            actual = self._aggregate_usage(
                connection,
                run_id=run_id,
                actual=True,
            )
            project_actual, project_reserved, project_unknown = (
                self._project_totals(connection)
            )
            over_request = connection.execute(
                """
                SELECT COUNT(*)
                FROM reservations
                WHERE run_id = ? AND (
                    (state IN ('settled', 'failed') AND actual_cost_micro_cny > ?)
                    OR (state = 'reserved' AND checkpoint_present = 1
                        AND checkpoint_cost_micro_cny > ?)
                )
                """,
                (
                    run_id,
                    _to_micro(REQUEST_HARD_CAP_CNY),
                    _to_micro(REQUEST_HARD_CAP_CNY),
                ),
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
            receipt_rows = connection.execute(
                "SELECT * FROM reservations ORDER BY rowid",
            ).fetchall()
            receipts = [self._receipt_from_row(row) for row in receipt_rows]
            receipt_bytes = (
                json.dumps(
                    [receipt.model_dump(mode="json") for receipt in receipts],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
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
                receipts=receipts,
                project_receipt_count=len(receipts),
                project_receipts_sha256=(
                    f"sha256:{hashlib.sha256(receipt_bytes).hexdigest()}"
                ),
            )

    def project_checkpoint(self) -> tuple[int, Sha256]:
        """Return the append-only receipt count and canonical history root."""
        with self._immediate() as connection:
            self._recover_locked(connection)
            rows = connection.execute(
                "SELECT * FROM reservations ORDER BY rowid"
            ).fetchall()
            receipts = [self._receipt_from_row(row) for row in rows]
            receipt_bytes = (
                json.dumps(
                    [receipt.model_dump(mode="json") for receipt in receipts],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            return (
                len(receipts),
                f"sha256:{hashlib.sha256(receipt_bytes).hexdigest()}",
            )

    @staticmethod
    def _receipt_from_row(row: sqlite3.Row) -> LedgerReceipt:
        return LedgerReceipt(
            reservation_id=row["reservation_id"],
            run_id=row["run_id"],
            query_id=row["query_id"],
            estimate=_usage_from_row(row, prefix="estimate", actual=False),
            state=row["state"],
            actual=(
                None
                if row["state"] == "reserved" and not row["checkpoint_present"]
                else UsageActual.model_validate(
                    _usage_from_row(
                        row,
                        prefix=(
                            "checkpoint" if row["state"] == "reserved" else "actual"
                        ),
                        actual=True,
                    ).model_dump()
                )
            ),
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
    "LedgerReceipt",
    "LedgerReservation",
    "LedgerReservationError",
    "LedgerSoftStopError",
    "PROJECT_HARD_CAP_CNY",
    "PROJECT_SOFT_STOP_CNY",
    "REQUEST_HARD_CAP_CNY",
    "SQLiteBudgetLedger",
    "VALIDATION_RUN_CAP_CNY",
]
