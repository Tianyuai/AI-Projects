"""Exact default-off experiment identities and injected component selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import yaml
from pydantic import StrictBool, model_validator

from paper_search.domain.models import DomainModel
from paper_search.evolution import EvolutionStrategy

if TYPE_CHECKING:
    from paper_search.graph.provider_stage import AsyncCitationExpansionStage
    from paper_search.ranking.embedding import EmbeddingRankingStage
    from paper_search.ranking.llm_stage import AsyncConstraintRerankingStage


ExperimentName = Literal[
    "main-baseline",
    "embedding",
    "citation-expansion",
    "llm-rerank",
    "fixed-two-round",
    "adaptive-evolution",
]
ExperimentStrategy = Literal[
    "fixed-one-round",
    "fixed-two-round",
    "adaptive-evolution",
]


class OptionalStageUnavailableError(RuntimeError):
    """An optional stage reported an expected, evidence-bearing outage."""


class ExperimentFlags(DomainModel):
    embedding: StrictBool = False
    citation_expansion: StrictBool = False
    constraint_reranking: StrictBool = False
    fixed_two_round: StrictBool = False
    adaptive_evolution: StrictBool = False


_DEFINITIONS: dict[ExperimentName, tuple[ExperimentFlags, ExperimentStrategy]] = {
    "main-baseline": (ExperimentFlags(), "fixed-one-round"),
    "embedding": (ExperimentFlags(embedding=True), "fixed-one-round"),
    "citation-expansion": (
        ExperimentFlags(citation_expansion=True),
        "fixed-one-round",
    ),
    "llm-rerank": (
        ExperimentFlags(constraint_reranking=True),
        "fixed-one-round",
    ),
    "fixed-two-round": (
        ExperimentFlags(fixed_two_round=True),
        "fixed-two-round",
    ),
    "adaptive-evolution": (
        ExperimentFlags(adaptive_evolution=True),
        "adaptive-evolution",
    ),
}


class ExperimentDefinition(DomainModel):
    name: ExperimentName
    flags: ExperimentFlags
    strategy: ExperimentStrategy

    @model_validator(mode="after")
    def validate_exact_definition(self) -> ExperimentDefinition:
        expected_flags, expected_strategy = _DEFINITIONS[self.name]
        if self.flags != expected_flags or self.strategy != expected_strategy:
            raise ValueError("exact experiment definition is required")
        return self


@dataclass(frozen=True)
class ExperimentComponents:
    embedding_ranker: EmbeddingRankingStage | None
    citation_expander: AsyncCitationExpansionStage | None
    constraint_reranker: AsyncConstraintRerankingStage | None
    evolution_strategy: EvolutionStrategy


class ExperimentDependencyFactory(Protocol):
    def build_embedding_ranker(self) -> EmbeddingRankingStage: ...

    def build_citation_expander(self) -> AsyncCitationExpansionStage: ...

    def build_constraint_reranker(self) -> AsyncConstraintRerankingStage: ...


def _optional_flags(raw: object) -> ExperimentFlags:
    if not isinstance(raw, dict):
        raise ValueError("ablation case must contain a mapping")
    public = {
        "embedding": raw.get("embedding"),
        "citation_expansion": raw.get("citation_expansion"),
        "constraint_reranking": raw.get("llm_rerank"),
        "fixed_two_round": raw.get("fixed_two_round"),
        "adaptive_evolution": raw.get("adaptive_evolution"),
    }
    if any(not isinstance(value, bool) for value in public.values()):
        raise ValueError("experiment flags must be explicit booleans")
    return ExperimentFlags.model_validate(public)


def load_experiment_definition(
    name: ExperimentName,
    *,
    ablation_config: Path,
) -> ExperimentDefinition:
    if name not in _DEFINITIONS:
        raise ValueError("unknown experiment name")
    try:
        raw = yaml.safe_load(ablation_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("invalid ablation registry") from error
    if not isinstance(raw, dict):
        raise ValueError("ablation registry must contain a mapping")
    case_name = "baseline" if name == "main-baseline" else name
    flags = _optional_flags(raw.get(case_name))
    expected_flags, strategy = _DEFINITIONS[name]
    if flags != expected_flags:
        raise ValueError("ablation registry does not match exact experiment definition")
    return ExperimentDefinition(name=name, flags=flags, strategy=strategy)


def build_experiment_components(
    definition: ExperimentDefinition,
    *,
    dependencies: ExperimentDependencyFactory,
) -> ExperimentComponents:
    definition = ExperimentDefinition.model_validate(definition)
    strategy_map: dict[ExperimentStrategy, EvolutionStrategy] = {
        "fixed-one-round": "fixed_one_round",
        "fixed-two-round": "fixed_two_round",
        "adaptive-evolution": "adaptive",
    }
    return ExperimentComponents(
        embedding_ranker=(
            dependencies.build_embedding_ranker()
            if definition.flags.embedding
            else None
        ),
        citation_expander=(
            dependencies.build_citation_expander()
            if definition.flags.citation_expansion
            else None
        ),
        constraint_reranker=(
            dependencies.build_constraint_reranker()
            if definition.flags.constraint_reranking
            else None
        ),
        evolution_strategy=strategy_map[definition.strategy],
    )


__all__ = [
    "ExperimentComponents",
    "ExperimentDefinition",
    "ExperimentDependencyFactory",
    "ExperimentFlags",
    "ExperimentName",
    "OptionalStageUnavailableError",
    "build_experiment_components",
    "load_experiment_definition",
]
