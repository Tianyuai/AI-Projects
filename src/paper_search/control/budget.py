"""Hard reservation accounting shared by all external providers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import threading
from uuid import uuid4

from paper_search.domain.models import (
    BudgetReservation,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)


class BudgetExceededError(RuntimeError):
    """Raised before an operation that would exceed a hard limit."""


class ReservationError(RuntimeError):
    """Raised when a reservation is invalid or cannot be settled safely."""


Clock = Callable[[], datetime]


def _aggregate(usages: Iterable[UsageEstimate]) -> UsageEstimate:
    items = list(usages)
    costs = [item.cost_cny for item in items]
    known_costs = [value for value in costs if value is not None]
    cost = sum(known_costs) if costs and len(known_costs) == len(costs) else None
    return UsageEstimate(
        search_api_calls=sum(item.search_api_calls for item in items),
        llm_calls=sum(item.llm_calls for item in items),
        input_tokens=sum(item.input_tokens for item in items),
        output_tokens=sum(item.output_tokens for item in items),
        cost_cny=cost,
        elapsed_ms=sum(item.elapsed_ms for item in items),
    )


def _known_cost(usages: Iterable[UsageEstimate]) -> Decimal:
    return sum(
        (item.cost_cny for item in usages if item.cost_cny is not None),
        Decimal("0"),
    )


class HardBudgetController:
    """Reserve estimates before calls and replace them with actual usage after calls.

    Expiry handling, persistence, soft deadlines and concurrent atomic reservation are
    deliberately deferred to Task 7. This Task 1 controller enforces hard limits in one
    process and prevents a caller from settling more usage than it reserved.
    """

    def __init__(
        self,
        budget: SearchBudget,
        *,
        reservation_ttl_seconds: int = 120,
        clock: Clock | None = None,
        formal_live: bool = False,
    ) -> None:
        if reservation_ttl_seconds <= 0:
            raise ValueError("reservation_ttl_seconds must be positive")
        self._budget = SearchBudget.model_validate(budget.model_dump())
        self.reservation_ttl_seconds = reservation_ttl_seconds
        self.formal_live = formal_live
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._reservations: dict[str, BudgetReservation] = {}
        self._committed: list[UsageActual] = []
        self._committed_actions: list[str] = []
        self._fail_closed = False

    @property
    def budget(self) -> SearchBudget:
        return self._budget

    @property
    def reserved_usage(self) -> UsageEstimate:
        with self._lock:
            self._expire_locked()
            return _aggregate(item.reserved for item in self._reservations.values())

    @property
    def committed_usage(self) -> UsageActual:
        with self._lock:
            summary = _aggregate(self._committed)
            return UsageActual.model_validate(summary.model_dump())

    @property
    def known_committed_cost_cny(self) -> Decimal:
        with self._lock:
            return _known_cost(self._committed)

    @property
    def unknown_cost_actions(self) -> list[str]:
        with self._lock:
            return [
                action
                for action, usage in zip(self._committed_actions, self._committed, strict=True)
                if usage.cost_cny is None
            ]

    def reserve(self, action: str, estimate: UsageEstimate) -> BudgetReservation:
        with self._lock:
            self._expire_locked()
            if self.stop_status() == "hard_stop":
                raise BudgetExceededError("budget controller is in hard-stop state")
            if estimate.llm_calls > 0 and estimate.cost_cny is None:
                raise ReservationError("LLM reservations require a known cost estimate")
            candidate = [
                *self._committed,
                *(r.reserved for r in self._reservations.values()),
                estimate,
            ]
            self._check_hard_limits(candidate)
            reservation = BudgetReservation(
                reservation_id=str(uuid4()),
                action=action,
                reserved=estimate,
                expires_at=self._clock() + timedelta(seconds=self.reservation_ttl_seconds),
            )
            self._reservations[reservation.reservation_id] = reservation
            return reservation

    def can_reserve(self, estimate: UsageEstimate) -> bool:
        """Return whether an estimate currently fits without creating a reservation."""
        with self._lock:
            self._expire_locked()
            if self.stop_status() == "hard_stop":
                return False
            if estimate.llm_calls > 0 and estimate.cost_cny is None:
                raise ReservationError("LLM reservations require a known cost estimate")
            candidate = [
                *self._committed,
                *(item.reserved for item in self._reservations.values()),
                estimate,
            ]
            try:
                self._check_hard_limits(candidate)
            except BudgetExceededError:
                return False
            return True

    def settle(self, reservation: BudgetReservation, actual: UsageActual) -> None:
        with self._lock:
            self._expire_locked()
            active = self._reservations.get(reservation.reservation_id)
            if active is None:
                raise ReservationError("reservation is unknown or already settled")
            if active != reservation:
                raise ReservationError("reservation does not match the active reservation")
            if self.formal_live and actual.cost_cny is None:
                raise ReservationError("formal live settlement requires a valued actual cost")
            self._ensure_within_reservation(active.reserved, actual)

            other_reserved = [
                item.reserved
                for reservation_id, item in self._reservations.items()
                if reservation_id != reservation.reservation_id
            ]
            self._check_hard_limits([*self._committed, *other_reserved, actual])
            del self._reservations[reservation.reservation_id]
            self._committed.append(actual)
            self._committed_actions.append(active.action)

    def release(self, reservation: BudgetReservation) -> None:
        """Release an unused active reservation."""
        with self._lock:
            self._expire_locked()
            active = self._reservations.get(reservation.reservation_id)
            if active != reservation:
                raise ReservationError("reservation is unknown or does not match the active reservation")
            del self._reservations[reservation.reservation_id]

    def expire_reservations(self) -> list[str]:
        """Release expired reservations and return their stable IDs."""
        with self._lock:
            return self._expire_locked()

    def fail_closed(self, reservation: BudgetReservation) -> None:
        """Block all future calls after a provider reports unaccountable usage."""
        with self._lock:
            active = self._reservations.get(reservation.reservation_id)
            if active != reservation:
                raise ReservationError("reservation is unknown or does not match the active reservation")
            self._fail_closed = True

    def stop_status(self) -> str:
        """Return deterministic continue, soft-stop, or hard-stop state."""
        with self._lock:
            self._expire_locked()
            if self._fail_closed:
                return "hard_stop"
            total = _aggregate([*self._committed, *(r.reserved for r in self._reservations.values())])
            hard = (
                total.search_api_calls >= self._budget.max_search_api_calls
                or total.llm_calls >= self._budget.max_llm_calls
                or total.input_tokens + total.output_tokens >= self._budget.max_total_tokens
                or total.elapsed_ms >= self._budget.max_elapsed_seconds * 1000
                or _known_cost([*self._committed, *(r.reserved for r in self._reservations.values())])
                >= self._budget.max_cost_cny
            )
            if hard:
                return "hard_stop"
            if total.elapsed_ms >= self._budget.soft_deadline_seconds * 1000:
                return "soft_stop"
            return "continue"

    def export_state(self) -> dict[str, object]:
        """Return JSON-round-trippable reservation and committed usage state."""
        with self._lock:
            self._expire_locked()
            return {
                "version": 1,
                "reservation_ttl_seconds": self.reservation_ttl_seconds,
                "formal_live": self.formal_live,
                "fail_closed": self._fail_closed,
                "reservations": [item.model_dump(mode="json") for item in self._reservations.values()],
                "committed": [
                    {"action": action, "usage": usage.model_dump(mode="json")}
                    for action, usage in zip(self._committed_actions, self._committed, strict=True)
                ],
            }

    @classmethod
    def from_state(
        cls,
        budget: SearchBudget,
        state: Mapping[str, object],
        *,
        clock: Clock | None = None,
    ) -> HardBudgetController:
        """Restore validated state without bypassing budget accounting invariants."""
        reservation_ttl_seconds = state.get("reservation_ttl_seconds")
        fail_closed = state.get("fail_closed")
        formal_live = state.get("formal_live")
        if (
            state.get("version") != 1
            or not isinstance(reservation_ttl_seconds, int)
            or not isinstance(fail_closed, bool)
            or not isinstance(formal_live, bool)
        ):
            raise ValueError("invalid budget controller state")
        controller = cls(
            budget,
            reservation_ttl_seconds=reservation_ttl_seconds,
            clock=clock,
            formal_live=formal_live,
        )
        reservations = state.get("reservations")
        committed = state.get("committed")
        if not isinstance(reservations, list) or not isinstance(committed, list):
            raise ValueError("invalid budget controller state")
        try:
            restored_reservations = [BudgetReservation.model_validate(raw) for raw in reservations]
            if len({item.reservation_id for item in restored_reservations}) != len(
                restored_reservations
            ):
                raise ValueError("duplicate reservation IDs")
            controller._reservations = {
                item.reservation_id: item for item in restored_reservations
            }
            controller._committed = []
            controller._committed_actions = []
            for raw in committed:
                if not isinstance(raw, Mapping) or not isinstance(raw.get("action"), str):
                    raise ValueError("invalid committed usage state")
                controller._committed_actions.append(raw["action"])
                controller._committed.append(UsageActual.model_validate(raw.get("usage")))
            controller._check_hard_limits(
                [*controller._committed, *(item.reserved for item in controller._reservations.values())]
            )
            controller._fail_closed = fail_closed
        except (TypeError, ValueError) as error:
            raise ValueError("invalid budget controller state") from error
        controller._expire_locked()
        return controller

    def _expire_locked(self) -> list[str]:
        now = self._clock()
        expired = [
            reservation_id
            for reservation_id, reservation in self._reservations.items()
            if reservation.expires_at <= now
        ]
        for reservation_id in expired:
            del self._reservations[reservation_id]
        return sorted(expired)

    def _check_hard_limits(self, usages: list[UsageEstimate]) -> None:
        total = _aggregate(usages)
        checks = (
            (
                total.search_api_calls,
                self._budget.max_search_api_calls,
                "search API calls",
            ),
            (total.llm_calls, self._budget.max_llm_calls, "LLM calls"),
            (
                total.input_tokens + total.output_tokens,
                self._budget.max_total_tokens,
                "total tokens",
            ),
            (
                total.elapsed_ms,
                self._budget.max_elapsed_seconds * 1000,
                "elapsed time",
            ),
        )
        for value, limit, label in checks:
            if value > limit:
                raise BudgetExceededError(f"{label} hard limit exceeded: {value} > {limit}")
        known_cost = _known_cost(usages)
        if known_cost > self._budget.max_cost_cny:
            raise BudgetExceededError(
                f"cost hard limit exceeded: {known_cost} > {self._budget.max_cost_cny}"
            )

    @staticmethod
    def _ensure_within_reservation(reserved: UsageEstimate, actual: UsageActual) -> None:
        checks = (
            (actual.search_api_calls, reserved.search_api_calls, "search_api_calls"),
            (actual.llm_calls, reserved.llm_calls, "llm_calls"),
            (actual.input_tokens, reserved.input_tokens, "input_tokens"),
            (actual.output_tokens, reserved.output_tokens, "output_tokens"),
            (actual.elapsed_ms, reserved.elapsed_ms, "elapsed_ms"),
        )
        for value, limit, label in checks:
            if value > limit:
                raise ReservationError(f"actual {label} exceeds its reservation")
        if (
            reserved.cost_cny is not None
            and actual.cost_cny is not None
            and actual.cost_cny > reserved.cost_cny
        ):
            raise ReservationError("actual cost_cny exceeds its reservation")
