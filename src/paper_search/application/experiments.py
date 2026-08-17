"""Stable production identity for the canonical end-to-end system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import StrictBool, model_validator

from paper_search.domain.models import DomainModel


ExperimentName = Literal["main-baseline"]
ExperimentStrategy = Literal["fixed-one-round"]


class OptionalStageUnavailableError(RuntimeError):
    """Compatibility error used by shared provider adapters."""


class ExperimentFlags(DomainModel):
    """Read-only legacy evidence fields; production requires every flag off."""

    embedding: StrictBool = False
    citation_expansion: StrictBool = False
    constraint_reranking: StrictBool = False
    title_candidates: StrictBool = False
    fixed_two_round: StrictBool = False
    adaptive_evolution: StrictBool = False


def expected_experiment_flags(name: str) -> ExperimentFlags:
    if name != "main-baseline":
        raise ValueError("unknown experiment name")
    return ExperimentFlags()


class ExperimentDefinition(DomainModel):
    name: ExperimentName
    flags: ExperimentFlags
    strategy: ExperimentStrategy

    @model_validator(mode="after")
    def validate_exact_definition(self) -> ExperimentDefinition:
        if self.flags != ExperimentFlags() or self.strategy != "fixed-one-round":
            raise ValueError("exact experiment definition is required")
        return self


@dataclass(frozen=True)
class ExperimentComponents:
    embedding_ranker: None = None
    citation_expander: None = None
    constraint_reranker: None = None
    title_candidate_stage: None = None
    evolution_strategy: Literal["fixed_one_round"] = "fixed_one_round"


class ExperimentDependencyFactory(Protocol):
    """Compatibility marker for composition callers; no optional builders remain."""


def load_experiment_definition(
    name: str,
    *,
    ablation_config: Path | None = None,
) -> ExperimentDefinition:
    del ablation_config
    expected_experiment_flags(name)
    return ExperimentDefinition(
        name="main-baseline",
        flags=ExperimentFlags(),
        strategy="fixed-one-round",
    )


def build_experiment_components(
    definition: ExperimentDefinition,
    *,
    dependencies: ExperimentDependencyFactory | Any,
) -> ExperimentComponents:
    del dependencies
    ExperimentDefinition.model_validate(definition)
    return ExperimentComponents()


__all__ = [
    "ExperimentComponents",
    "ExperimentDefinition",
    "ExperimentDependencyFactory",
    "ExperimentFlags",
    "ExperimentName",
    "OptionalStageUnavailableError",
    "build_experiment_components",
    "expected_experiment_flags",
    "load_experiment_definition",
]
