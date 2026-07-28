from pathlib import Path

import pytest
import yaml

from paper_search.evaluation import ExperimentAggregate
from paper_search.evaluation.ablations import AblationCase, AblationReport, run_ablations


REQUIRED_CASE_NAMES = [
    "baseline",
    "query-planning",
    "multi-source",
    "embedding",
    "citation-expansion",
    "llm-rerank",
    "fixed-two-round",
    "adaptive-evolution",
    "low-budget",
    "balanced",
]
PUBLIC_MODULES = (
    "query_planning",
    "multi_source",
    "embedding",
    "citation_expansion",
    "llm_rerank",
    "fixed_two_round",
    "adaptive_evolution",
    "low_budget",
    "balanced",
)


def _aggregate_for(name: str) -> ExperimentAggregate:
    return ExperimentAggregate(
        query_count=3,
        macro_f1=0.4 if name == "baseline" else 0.5,
        macro_recall=0.6,
        search_api_calls=4,
        llm_calls=0,
        input_tokens=0,
        output_tokens=0,
        cost_cny=0.0,
        latency_ms=25.0,
        failure_count=0,
    )


def _modules(**overrides: bool) -> dict[str, bool]:
    modules = {name: False for name in PUBLIC_MODULES}
    modules.update(overrides)
    return modules


def _case(name: str, **overrides: bool) -> AblationCase:
    return AblationCase(name=name, modules=_modules(**overrides))


def _report_payload(
    *,
    name: str = "baseline",
    modules: dict[str, bool] | None = None,
) -> dict[str, object]:
    return {
        "split": "dev",
        "phase": "selection_only",
        "cases": [
            {
                "name": name,
                "modules": _modules() if modules is None else modules,
                "aggregate": _aggregate_for("baseline").model_dump(mode="json"),
            }
        ],
    }


def _load_ablation_matrix() -> dict[str, dict[str, bool]]:
    path = Path("configs/ablations.yaml")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert isinstance(raw, dict)
    for case_name, modules in raw.items():
        assert isinstance(case_name, str)
        assert isinstance(modules, dict)
    return raw


def test_run_ablations_preserves_case_order_and_calls_evaluator_once_per_case() -> None:
    calls: list[str] = []

    def evaluator(case: AblationCase) -> ExperimentAggregate:
        calls.append(case.name)
        return _aggregate_for(case.name)

    report = run_ablations(
        cases=[
            _case("baseline"),
            _case("embedding", embedding=True),
        ],
        split="dev",
        evaluator=evaluator,
    )

    assert isinstance(report, AblationReport)
    assert calls == ["baseline", "embedding"]
    assert report.split == "dev"
    assert [case.name for case in report.cases] == ["baseline", "embedding"]
    assert report.phase == "tuning"
    assert report.cases[0].modules["embedding"] is False
    assert report.cases[1].modules["embedding"] is True


def test_validation_ablation_requires_selection_only_phase() -> None:
    with pytest.raises(ValueError, match="selection_only"):
        run_ablations(
            cases=[_case("baseline")],
            split="validation",
            evaluator=lambda case: _aggregate_for(case.name),
        )


def test_tuning_is_dev_only() -> None:
    with pytest.raises(ValueError, match="dev"):
        run_ablations(
            cases=[_case("baseline")],
            split="holdout",
            phase="tuning",
            evaluator=lambda case: _aggregate_for(case.name),
        )


@pytest.mark.parametrize("name", ["", "  ", "../private", "unsafe name"])
def test_case_names_reject_blank_and_unsafe_values(name: str) -> None:
    with pytest.raises(ValueError):
        AblationCase(name=name, modules=_modules())


def test_run_ablations_rejects_duplicate_case_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        run_ablations(
            cases=[_case("baseline"), _case("baseline")],
            split="dev",
            evaluator=lambda case: _aggregate_for(case.name),
        )


def test_direct_report_construction_rejects_unsafe_case_names() -> None:
    with pytest.raises(ValueError, match="safe public identifiers"):
        AblationReport(**_report_payload(name="../private"))


def test_report_deserialization_rejects_private_module_keys() -> None:
    modules = _modules()
    modules["private_notes"] = True

    with pytest.raises(ValueError, match="public boolean flags"):
        AblationReport.model_validate(_report_payload(modules=modules))


def test_report_deserialization_rejects_missing_public_module_flags() -> None:
    modules = _modules()
    modules.pop("balanced")

    with pytest.raises(ValueError, match="every public boolean flag"):
        AblationReport.model_validate(_report_payload(modules=modules))


def test_direct_report_construction_rejects_duplicate_case_names() -> None:
    duplicate_payload = {
        "split": "dev",
        "phase": "selection_only",
        "cases": [
            {
                "name": "baseline",
                "modules": _modules(),
                "aggregate": _aggregate_for("baseline").model_dump(mode="json"),
            },
            {
                "name": "baseline",
                "modules": _modules(embedding=True),
                "aggregate": _aggregate_for("embedding").model_dump(mode="json"),
            },
        ],
    }

    with pytest.raises(ValueError, match="duplicate"):
        AblationReport.model_validate(duplicate_payload)


def test_report_serializes_only_public_case_metadata_and_aggregate_results() -> None:
    report = run_ablations(
        cases=[_case("balanced", balanced=True)],
        split="dev",
        phase="selection_only",
        evaluator=lambda case: _aggregate_for(case.name),
    )

    assert report.model_dump(mode="json") == {
        "split": "dev",
        "phase": "selection_only",
        "cases": [
            {
                "name": "balanced",
                "modules": _modules(balanced=True),
                "aggregate": _aggregate_for("balanced").model_dump(mode="json"),
            }
        ],
    }


def test_yaml_matrix_lists_required_cases_and_safe_public_booleans() -> None:
    matrix = _load_ablation_matrix()

    assert list(matrix) == REQUIRED_CASE_NAMES
    for modules in matrix.values():
        assert set(modules) == set(PUBLIC_MODULES)
        assert all(isinstance(value, bool) for value in modules.values())

    assert matrix["baseline"]["embedding"] is False
    assert matrix["embedding"]["embedding"] is True
    assert matrix["citation-expansion"]["citation_expansion"] is False
    assert matrix["llm-rerank"]["llm_rerank"] is False

    for case_name, modules in matrix.items():
        if case_name != "embedding":
            assert modules["embedding"] is False
        assert modules["citation_expansion"] is False
        assert modules["llm_rerank"] is False


def test_readme_documents_offline_injected_owner_only_ablation_policy() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "offline and injected by default" in readme
    assert "does not call APIs or load `.env`" in readme
    assert "owner_only_provisional" in readme
