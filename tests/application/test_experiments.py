from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_search.application.experiments import (
    ExperimentDefinition,
    ExperimentFlags,
    build_experiment_components,
    load_experiment_definition,
)


ABLATION_CONFIG = Path("configs/ablations.yaml")
EXPECTED = {
    "main-baseline": ({}, "fixed-one-round"),
    "embedding": ({"embedding": True}, "fixed-one-round"),
    "citation-expansion": ({"citation_expansion": True}, "fixed-one-round"),
    "llm-rerank": ({"constraint_reranking": True}, "fixed-one-round"),
    "fixed-two-round": ({"fixed_two_round": True}, "fixed-two-round"),
    "adaptive-evolution": (
        {"adaptive_evolution": True},
        "adaptive-evolution",
    ),
}


@pytest.mark.parametrize(("name", "expected"), EXPECTED.items())
def test_registry_loads_only_the_exact_named_flag(
    name: str,
    expected: tuple[dict[str, bool], str],
) -> None:
    enabled, strategy = expected

    definition = load_experiment_definition(name, ablation_config=ABLATION_CONFIG)

    expected_flags = {
        "embedding": False,
        "citation_expansion": False,
        "constraint_reranking": False,
        "fixed_two_round": False,
        "adaptive_evolution": False,
        **enabled,
    }
    assert definition.name == name
    assert definition.flags.model_dump() == expected_flags
    assert definition.strategy == strategy


@pytest.mark.parametrize(
    "definition",
    [
        {
            "name": "main-baseline",
            "flags": {"embedding": True},
            "strategy": "fixed-one-round",
        },
        {
            "name": "embedding",
            "flags": {"embedding": True, "citation_expansion": True},
            "strategy": "fixed-one-round",
        },
        {
            "name": "fixed-two-round",
            "flags": {"fixed_two_round": True},
            "strategy": "adaptive-evolution",
        },
        {
            "name": "embedding",
            "flags": {"embedding": 1},
            "strategy": "fixed-one-round",
        },
    ],
)
def test_definition_rejects_non_exact_flag_or_strategy_combinations(
    definition: dict[str, object],
) -> None:
    with pytest.raises(
        ValidationError,
        match="exact experiment definition|valid boolean",
    ):
        ExperimentDefinition.model_validate(definition)


class DependencyTrap:
    def build_embedding_ranker(self) -> object:
        raise AssertionError("baseline must not construct embedding")

    def build_citation_expander(self) -> object:
        raise AssertionError("baseline must not construct citation expansion")

    def build_constraint_reranker(self) -> object:
        raise AssertionError("baseline must not construct LLM reranking")


def test_main_baseline_constructs_no_optional_dependencies() -> None:
    definition = ExperimentDefinition(
        name="main-baseline",
        flags=ExperimentFlags(),
        strategy="fixed-one-round",
    )

    components = build_experiment_components(
        definition,
        dependencies=DependencyTrap(),
    )

    assert components.embedding_ranker is None
    assert components.citation_expander is None
    assert components.constraint_reranker is None
    assert components.evolution_strategy == "fixed_one_round"


def test_registry_import_does_not_import_optional_dependency_adapters() -> None:
    script = """
import json
import sys
import paper_search.application.experiments

optional = {
    "paper_search.graph.provider_stage",
    "paper_search.ranking.llm_stage",
    "paper_search.ranking.sentence_transformer",
}
print(json.dumps(sorted(optional.intersection(sys.modules))))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_baseline_composition_import_does_not_load_optional_adapters() -> None:
    script = """
import json
import sys
import paper_search.application.composition

optional = {
    "paper_search.graph.provider_stage",
    "paper_search.ranking.llm_stage",
}
print(json.dumps(sorted(optional.intersection(sys.modules))))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
