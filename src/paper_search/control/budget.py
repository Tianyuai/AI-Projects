"""Hard reservation accounting shared by all external providers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
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


def _known_cost(usages: Iterable[UsageEstimate]) -> float:
    return sum(item.cost_cny for item in usages if item.cost_cny is not None)


class HardBudgetController:
    """Reserve estimates before calls and replace them with actual usage after calls.

    Expiry handling, persistence, soft deadlines and concurrent atomic reservation are
    deliberately deferred to Task 7. This Task 1 controller enforces hard limits in one
    process and prevents a caller from settling more usage than it reserved.
    """

    def __init__(self, budget: SearchBudget, *, reservation_ttl_seconds: int = 120) -> None:
        if reservation_ttl_seconds <= 0:
            raise ValueError("reservation_ttl_seconds must be positive")
        self._budget = SearchBudget.model_validate(budget.model_dump())
        self.reservation_ttl_seconds = reservation_ttl_seconds
        self._reservations: dict[str, BudgetReservation] = {}
        self._committed: list[UsageActual] = []

    @property
    def budget(self) -> SearchBudget:
        return self._budget

    @property
    def reserved_usage(self) -> UsageEstimate:
        return _aggregate(item.reserved for item in self._reservations.values())

    @property
    def committed_usage(self) -> UsageActual:
        summary = _aggregate(self._committed)
        return UsageActual.model_validate(summary.model_dump())

    def reserve(self, action: str, estimate: UsageEstimate) -> BudgetReservation:
        if estimate.llm_calls > 0 and estimate.cost_cny is None:
            raise ReservationError("LLM reservations require a known cost estimate")
        candidate = [*self._committed, *(r.reserved for r in self._reservations.values()), estimate]
        self._check_hard_limits(candidate)
        reservation = BudgetReservation(
            reservation_id=str(uuid4()),
            action=action,
            reserved=estimate,
            expires_at=datetime.now(UTC) + timedelta(seconds=self.reservation_ttl_seconds),
        )
        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def settle(self, reservation: BudgetReservation, actual: UsageActual) -> None:
        active = self._reservations.get(reservation.reservation_id)
        if active is None:
            raise ReservationError("reservation is unknown or already settled")
        if active != reservation:
            raise ReservationError("reservation does not match the active reservation")
        if actual.llm_calls > 0 and actual.cost_cny is None:
            raise ReservationError("actual LLM usage requires a known cost")
        self._ensure_within_reservation(active.reserved, actual)

        other_reserved = [
            item.reserved
            for reservation_id, item in self._reservations.items()
            if reservation_id != reservation.reservation_id
        ]
        self._check_hard_limits([*self._committed, *other_reserved, actual])
        del self._reservations[reservation.reservation_id]
        self._committed.append(actual)

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
