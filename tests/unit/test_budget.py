import importlib

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


def test_unknown_actual_llm_cost_does_not_release_reservation() -> None:
    controller_type, _, reservation_error = budget_api()
    controller = controller_type(make_budget())
    reservation = controller.reserve(
        "llm.generate",
        UsageEstimate(llm_calls=1, cost_cny=0.5),
    )

    with pytest.raises(reservation_error):
        controller.settle(reservation, UsageActual(llm_calls=1, cost_cny=None))

    assert controller.reserved_usage.llm_calls == 1
    assert controller.reserved_usage.cost_cny == pytest.approx(0.5)
    assert controller.committed_usage.llm_calls == 0
