from __future__ import annotations

from typing import Literal

import yaml
from pydantic import model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr

EvolutionStrategy = Literal[
    "synonym", "entity_alias", "facet_combination", "task_decomposition"
]
NoOpReason = Literal["insufficient_grounded_facets", "no_novel_query"]


class PromptArtifact(DomainModel):
    name: NonEmptyStr
    version: NonEmptyStr
    temperature: Literal[0]
    response_model: NonEmptyStr
    instructions: list[NonEmptyStr]
    strategies: list[EvolutionStrategy] | None = None
    no_op_reasons: list[NoOpReason] | None = None

    @model_validator(mode="after")
    def validate_query_evolve_contract(self) -> PromptArtifact:
        if self.name == "query_evolve":
            if self.version != "query-evolve-v2":
                raise ValueError(
                    "query_evolve prompt version must be query-evolve-v2"
                )
            if self.response_model != "QueryEvolutionProposal":
                raise ValueError("query_evolve response model mismatch")
            if self.strategies != [
                "synonym",
                "entity_alias",
                "facet_combination",
                "task_decomposition",
            ]:
                raise ValueError("query_evolve strategy enum mismatch")
            if self.no_op_reasons != [
                "insufficient_grounded_facets",
                "no_novel_query",
            ]:
                raise ValueError("query_evolve no-op enum mismatch")
        elif self.strategies is not None or self.no_op_reasons is not None:
            raise ValueError("evolution enums require query_evolve")
        return self


def load_prompt_artifact(prompt_bytes: bytes) -> PromptArtifact:
    try:
        raw = yaml.safe_load(prompt_bytes)
    except yaml.YAMLError as error:
        raise ValueError("invalid prompt artifact") from error
    if not isinstance(raw, dict):
        raise ValueError("invalid prompt artifact")
    return PromptArtifact.model_validate(raw)


def render_prompt_system_message(prompt_bytes: bytes) -> str:
    artifact = load_prompt_artifact(prompt_bytes)
    lines = [
        "Respond with a JSON object.",
        f"The JSON object must match the {artifact.response_model} contract.",
        *(f"- {item}" for item in artifact.instructions),
    ]
    return "\n".join(lines)
