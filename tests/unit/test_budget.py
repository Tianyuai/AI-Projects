import importlib
import threading
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from paper_search.domain.models import SearchBudget, UsageActual, UsageEstimate


def budget_api() -> tuple[type, type[Exception], type[Exception]]:
    try:
        module = importlib.import_module("paper_search.control.budget")
    except ModuleNotFoundError:
        pytest.fail("paper_search.control.budget must be implemented")
    assert hasattr(module, "HardBudgetController"), "HardBudgetController must be implemented"
    assert hasattr(module, "BudgetExceededError"), "BudgetExceededError must be implemented"
    assert hasattr(module, "ReservationError"), "ReservationError must be implemented"
    return module.HardBudgetController, module.BudgetExceededError, module.ReservationError


def budget_preflight_api() -> tuple[type, type[Exception], type[Exception]]:
    controller_type, exceeded_error, reservation_error = budget_api()
    assert hasattr(controller_type, "can_reserve"), "HardBudgetController.can_reserve must be implemented"
    return controller_type, exceeded_error, reservation_error


def make_budget(**updates: object) -> SearchBudget:
    values = {
        "max_search_api_calls": 2,
        "target_search_api_calls": 1,
        "max_llm_calls": 1,
        "target_llm_calls": 1,
        "max_total_tokens": 100,
        "max_cost_cny": 1.0,
        "max_elapsed_seconds": 2,
        "soft_deadline_seconds": 1,
    }
    values.update(updates)
    return SearchBudget.model_validate(values)


def test_reserve_rejects_usage_above_hard_limit_without_changing_state() -> None:
    controller_type, exceeded_error, _ = budget_api()
    controller = controller_type(make_budget())
    first = controller.reserve("openalex.search", UsageEstimate(search_api_calls=1))

    with pytest.raises(exceeded_error):
        controller.reserve("openalex.next_page", UsageEstimate(search_api_calls=2))

    assert first.action == "openalex.search"
    assert controller.reserved_usage.search_api_calls == 1
    assert controller.committed_usage.search_api_calls == 0


def test_token_limit_combines_input_and_output_tokens() -> None:
    controller_type, exceeded_error, _ = budget_api()
    controller = controller_type(make_budget())
    controller.reserve("query.analyze", UsageEstimate(input_tokens=60, output_tokens=30))

    with pytest.raises(exceeded_error):
        controller.reserve("query.repair", UsageEstimate(input_tokens=5, output_tokens=6))


def test_settle_releases_estimate_and_commits_actual_usage() -> None:
    controller_type, _, _ = budget_api()
    controller = controller_type(make_budget())
    reservation = controller.reserve(
        "llm.generate",
        UsageEstimate(
            llm_calls=1,
            input_tokens=50,
            output_tokens=20,
            cost_cny=0.5,
            elapsed_ms=500,
        ),
    )

    controller.settle(
        reservation,
        UsageActual(
            llm_calls=1,
            input_tokens=40,
            output_tokens=10,
            cost_cny=0.4,
            elapsed_ms=400,
        ),
    )

    assert controller.reserved_usage.llm_calls == 0
    assert controller.reserved_usage.input_tokens == 0
    assert controller.committed_usage.llm_calls == 1
    assert controller.committed_usage.input_tokens == 40
    assert controller.committed_usage.output_tokens == 10
    assert controller.committed_usage.cost_cny == pytest.approx(0.4)


def test_settle_rejects_usage_above_reservation_and_keeps_it_active() -> None:
    controller_type, _, reservation_error = budget_api()
    controller = controller_type(make_budget())
    reservation = controller.reserve("openalex.search", UsageEstimate(search_api_calls=1))

    with pytest.raises(reservation_error):
        controller.settle(reservation, UsageActual(search_api_calls=2))

    assert controller.reserved_usage.search_api_calls == 1
    assert controller.committed_usage.search_api_calls == 0


def test_reservation_cannot_be_settled_twice() -> None:
    controller_type, _, reservation_error = budget_api()
    controller = controller_type(make_budget())
    reservation = controller.reserve("openalex.search", UsageEstimate(search_api_calls=1))
    controller.settle(reservation, UsageActual(search_api_calls=1))

    with pytest.raises(reservation_error):
        controller.settle(reservation, UsageActual(search_api_calls=1))


def test_budget_accounting_inputs_are_immutable_after_settlement() -> None:
    controller_type, _, _ = budget_api()
    budget = make_budget()
    controller = controller_type(budget)
    estimate = UsageEstimate(llm_calls=1, cost_cny=0.5)
    actual = UsageActual(llm_calls=1, cost_cny=0.4)
    reservation = controller.reserve("llm.generate", estimate)
    controller.settle(reservation, actual)

    with pytest.raises(ValidationError):
        actual.llm_calls = 0
    with pytest.raises(ValidationError):
        estimate.cost_cny = 0.0
    with pytest.raises(ValidationError):
        budget.max_llm_calls = 100
    with pytest.raises(AttributeError):
        controller.budget = make_budget(max_llm_calls=100, target_llm_calls=1)

    assert controller.committed_usage.llm_calls == 1
    assert controller.committed_usage.cost_cny == pytest.approx(0.4)


def test_llm_reservation_requires_known_cost() -> None:
    controller_type, _, reservation_error = budget_api()
    controller = controller_type(make_budget())

    with pytest.raises(reservation_error):
        controller.reserve("llm.generate", UsageEstimate(llm_calls=1, cost_cny=None))

    assert controller.reserved_usage.llm_calls == 0


def test_can_reserve_matches_reserve_without_mutating_state() -> None:
    controller_type, _, _ = budget_preflight_api()
    controller = controller_type(make_budget(max_search_api_calls=1, target_search_api_calls=1))
    estimate = UsageEstimate(search_api_calls=1)

    assert controller.can_reserve(estimate) is True
    assert controller.reserved_usage == UsageEstimate()
    reservation = controller.reserve("test", estimate)
    assert reservation.reserved == estimate
    assert controller.can_reserve(UsageEstimate(search_api_calls=1)) is False


def test_can_reserve_expires_at_limit_reservation_without_replacing_it() -> None:
    controller_type, _, _ = budget_preflight_api()
    current = datetime(2026, 7, 29, tzinfo=UTC)
    controller = controller_type(
        make_budget(max_search_api_calls=1, target_search_api_calls=1),
        clock=lambda: current,
        reservation_ttl_seconds=1,
    )
    controller.reserve("expiring", UsageEstimate(search_api_calls=1))
    current += timedelta(seconds=2)

    assert controller.can_reserve(UsageEstimate(search_api_calls=1)) is True
    assert controller.reserved_usage == UsageEstimate()


@pytest.mark.parametrize(
    ("estimate", "budget_updates"),
    [
        (UsageEstimate(search_api_calls=1), {"max_search_api_calls": 1, "target_search_api_calls": 1}),
        (UsageEstimate(llm_calls=1, cost_cny=0.1), {"max_llm_calls": 1, "target_llm_calls": 1}),
        (UsageEstimate(input_tokens=60, output_tokens=40), {"max_total_tokens": 100}),
        (UsageEstimate(elapsed_ms=2_000), {"max_elapsed_seconds": 2}),
        (UsageEstimate(cost_cny=1.0), {"max_cost_cny": 1.0}),
    ],
)
def test_can_reserve_allows_exact_hard_limits(
    estimate: UsageEstimate, budget_updates: dict[str, object]
) -> None:
    controller_type, _, _ = budget_preflight_api()
    controller = controller_type(make_budget(**budget_updates))

    assert controller.can_reserve(estimate) is True
    assert controller.reserved_usage == UsageEstimate()


@pytest.mark.parametrize(
    ("reserved", "estimate", "budget_updates"),
    [
        (
            UsageEstimate(search_api_calls=1),
            UsageEstimate(search_api_calls=1),
            {"max_search_api_calls": 1, "target_search_api_calls": 1},
        ),
        (
            UsageEstimate(llm_calls=1, cost_cny=0.1),
            UsageEstimate(llm_calls=1, cost_cny=0.1),
            {"max_llm_calls": 1, "target_llm_calls": 1},
        ),
        (
            UsageEstimate(input_tokens=50, output_tokens=50),
            UsageEstimate(input_tokens=1),
            {"max_total_tokens": 100},
        ),
        (
            UsageEstimate(elapsed_ms=2_000),
            UsageEstimate(elapsed_ms=1),
            {"max_elapsed_seconds": 2},
        ),
        (
            UsageEstimate(cost_cny=1.0),
            UsageEstimate(cost_cny=0.1),
            {"max_cost_cny": 1.0},
        ),
    ],
)
def test_can_reserve_accounts_for_active_reservations_in_all_hard_limits(
    reserved: UsageEstimate, estimate: UsageEstimate, budget_updates: dict[str, object]
) -> None:
    controller_type, _, _ = budget_preflight_api()
    controller = controller_type(make_budget(**budget_updates))
    controller.reserve("active", reserved)

    assert controller.can_reserve(estimate) is False


def test_can_reserve_accounts_for_committed_usage() -> None:
    controller_type, _, _ = budget_preflight_api()
    controller = controller_type(make_budget(max_search_api_calls=1, target_search_api_calls=1))
    reservation = controller.reserve("committed", UsageEstimate(search_api_calls=1))
    controller.settle(reservation, UsageActual(search_api_calls=1))

    assert controller.can_reserve(UsageEstimate(search_api_calls=1)) is False


def test_can_reserve_returns_false_after_fail_closed() -> None:
    controller_type, _, _ = budget_preflight_api()
    controller = controller_type(make_budget())
    reservation = controller.reserve("provider.search", UsageEstimate(search_api_calls=1))
    controller.fail_closed(reservation)

    assert controller.can_reserve(UsageEstimate()) is False


def test_can_reserve_propagates_unknown_llm_cost_error() -> None:
    controller_type, _, reservation_error = budget_preflight_api()
    controller = controller_type(make_budget())

    with pytest.raises(reservation_error, match="LLM reservations require a known cost estimate"):
        controller.can_reserve(UsageEstimate(llm_calls=1))


def test_unknown_actual_llm_cost_does_not_release_reservation() -> None:
    controller_type, _, reservation_error = budget_api()
    controller = controller_type(make_budget())
    reservation = controller.reserve(
        "llm.generate",
        UsageEstimate(llm_calls=1, cost_cny=0.5),
    )

    controller.settle(reservation, UsageActual(llm_calls=1, cost_cny=None))

    assert controller.reserved_usage.llm_calls == 0
    assert controller.committed_usage.llm_calls == 1
    assert controller.committed_usage.cost_cny is None
    assert controller.unknown_cost_actions == ["llm.generate"]


def test_release_expiry_stop_status_and_recovery_are_deterministic() -> None:
    controller_type, exceeded_error, _ = budget_api()
    current = datetime(2026, 7, 23, tzinfo=UTC)
    controller = controller_type(make_budget(), clock=lambda: current, reservation_ttl_seconds=1)
    reservation = controller.reserve("provider.search", UsageEstimate(search_api_calls=1))

    controller.release(reservation)
    assert controller.reserved_usage.search_api_calls == 0
    expiring = controller.reserve("provider.search", UsageEstimate(search_api_calls=1))
    current += timedelta(seconds=2)
    assert controller.expire_reservations() == [expiring.reservation_id]
    assert controller.reserved_usage.search_api_calls == 0

    soft = controller.reserve("slow", UsageEstimate(elapsed_ms=1_000))
    controller.settle(soft, UsageActual(elapsed_ms=1_000))
    assert controller.stop_status() == "soft_stop"
    hard = controller.reserve("slower", UsageEstimate(elapsed_ms=1_000))
    controller.settle(hard, UsageActual(elapsed_ms=1_000))
    assert controller.stop_status() == "hard_stop"
    with pytest.raises(exceeded_error):
        controller.reserve("blocked", UsageEstimate(search_api_calls=1))

    restored = controller_type.from_state(make_budget(), controller.export_state(), clock=lambda: current)
    assert restored.committed_usage == controller.committed_usage
    assert restored.stop_status() == "hard_stop"


def test_unknown_cost_is_separate_from_known_cost_after_settlement() -> None:
    controller_type, _, _ = budget_api()
    controller = controller_type(make_budget())
    known = controller.reserve("known", UsageEstimate(search_api_calls=1, cost_cny=0.25))
    controller.settle(known, UsageActual(search_api_calls=1, cost_cny=0.2))
    unknown = controller.reserve("unknown", UsageEstimate(search_api_calls=1))
    controller.settle(unknown, UsageActual(search_api_calls=1, cost_cny=None))

    assert controller.committed_usage.cost_cny is None
    assert controller.known_committed_cost_cny == pytest.approx(0.2)
    assert controller.unknown_cost_actions == ["unknown"]


def test_concurrent_reservations_are_atomic() -> None:
    controller_type, exceeded_error, _ = budget_api()
    controller = controller_type(make_budget(max_search_api_calls=1, target_search_api_calls=1))
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def reserve() -> None:
        barrier.wait()
        try:
            controller.reserve("parallel", UsageEstimate(search_api_calls=1))
        except exceeded_error:
            outcomes.append("blocked")
        else:
            outcomes.append("reserved")

    threads = [threading.Thread(target=reserve), threading.Thread(target=reserve)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["blocked", "reserved"]
    assert controller.reserved_usage.search_api_calls == 1


def test_failed_settlement_can_force_a_fail_closed_hard_stop() -> None:
    controller_type, exceeded_error, reservation_error = budget_api()
    controller = controller_type(make_budget())
    reservation = controller.reserve("provider.search", UsageEstimate(search_api_calls=1))

    with pytest.raises(reservation_error):
        controller.settle(reservation, UsageActual(search_api_calls=2))
    controller.fail_closed(reservation)

    assert controller.stop_status() == "hard_stop"
    with pytest.raises(exceeded_error):
        controller.reserve("provider.retry", UsageEstimate(search_api_calls=1))


def test_recovery_rejects_duplicate_reservation_ids() -> None:
    controller_type, _, _ = budget_api()
    controller = controller_type(make_budget())
    reservation = controller.reserve("provider.search", UsageEstimate(search_api_calls=1))
    state = controller.export_state()
    state["reservations"] = [reservation.model_dump(mode="json"), reservation.model_dump(mode="json")]

    with pytest.raises(ValueError, match="invalid budget controller state"):
        controller_type.from_state(make_budget(), state)
