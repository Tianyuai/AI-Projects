from paper_search.control.budget import HardBudgetController
import scripts.validate_semantic_action_llm_live as validation_script


def test_llm_only_validation_budget_starts_in_continue_state() -> None:
    assert hasattr(validation_script, "_validation_budget")
    controller = HardBudgetController(
        validation_script._validation_budget(),  # type: ignore[attr-defined]  # noqa: SLF001
        formal_live=True,
    )

    assert controller.stop_status() == "continue"
