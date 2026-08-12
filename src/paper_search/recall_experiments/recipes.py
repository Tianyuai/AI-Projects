"""Strict, declarative configuration for candidate-recall experiments.

Loading a recipe is deliberately offline: it binds the recipe and (where
applicable) prompt bytes, but does not resolve actions, snapshots, datasets,
or call a backend.  Those command-specific inputs are bound by later
preflight steps.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import yaml
from pydantic import Field, field_validator, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, SafeRelativePath, Sha256
from paper_search.recall_experiments.contracts import SeedCandidate
from paper_search.recall_experiments.contracts import ActionType, GoldVisibility


RecallBackend = Literal["live_provider", "snapshot_replay"]
CandidatePoolPolicy = Literal["production-dedup-v1", "canonical-id-first-v1"]


class _GeneratorRecipeBase(DomainModel):
    gold_visibility: GoldVisibility


class ManualActionsGeneratorRecipe(_GeneratorRecipeBase):
    type: Literal["manual_actions"]
    actions: SafeRelativePath


class FixedActionsGeneratorRecipe(_GeneratorRecipeBase):
    type: Literal["fixed_actions"]
    actions: SafeRelativePath


class DeepSeekPromptGeneratorRecipe(_GeneratorRecipeBase):
    type: Literal["deepseek_prompt"]
    prompt: SafeRelativePath
    model: NonEmptyStr
    temperature: Annotated[int, Field(strict=True, ge=0, le=0)]
    max_generated_actions: Annotated[int, Field(strict=True, gt=0)]
    repair_attempts: Annotated[int, Field(strict=True, ge=1, le=1)]


GeneratorRecipe: TypeAlias = Annotated[
    ManualActionsGeneratorRecipe | FixedActionsGeneratorRecipe | DeepSeekPromptGeneratorRecipe,
    Field(discriminator="type"),
]


class RetrievalRecipe(DomainModel):
    allowed_actions: list[ActionType] = Field(min_length=1)
    backend: RecallBackend
    max_results_per_action: Annotated[int, Field(strict=True, gt=0)]
    max_total_actions: Annotated[int, Field(strict=True, gt=0)]


class CandidatePoolRecipe(DomainModel):
    policy_version: CandidatePoolPolicy = "production-dedup-v1"


class EvaluationRecipe(DomainModel):
    repeat_count: Annotated[int, Field(strict=True, gt=0)]
    max_repeat_attempts: Annotated[int, Field(strict=True, gt=0)]
    compare_with: NonEmptyStr | None = None
    gold_count_tolerance: Annotated[int, Field(strict=True, ge=0)] | None = None
    macro_recall_tolerance: Annotated[
        float, Field(strict=True, ge=0, allow_inf_nan=False)
    ] | None = None
    retained_gold_min: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)] | None = None
    required_passing_repeats: Annotated[int, Field(strict=True, gt=0)] | None = None

    @model_validator(mode="after")
    def validate_comparison_settings(self) -> EvaluationRecipe:
        if self.max_repeat_attempts < self.repeat_count:
            raise ValueError("max_repeat_attempts must not be less than repeat_count")
        comparison_values = (
            self.gold_count_tolerance,
            self.macro_recall_tolerance,
            self.retained_gold_min,
            self.required_passing_repeats,
        )
        if self.compare_with is None:
            if any(value is not None for value in comparison_values):
                raise ValueError("historical comparison thresholds require compare_with")
            return self
        if (
            self.repeat_count != 3
            or self.max_repeat_attempts != 5
            or self.gold_count_tolerance != 1
            or self.macro_recall_tolerance != 0.02
            or self.retained_gold_min != 0.90
            or self.required_passing_repeats != 2
        ):
            raise ValueError(
                "historical comparison requires repeat_count=3, max_repeat_attempts=5, "
                "gold_count_tolerance=1, macro_recall_tolerance=0.02, "
                "retained_gold_min=0.90, and required_passing_repeats=2"
            )
        return self


class RecallMethodRecipe(DomainModel):
    method_id: NonEmptyStr
    generator: GeneratorRecipe
    retrieval: RetrievalRecipe
    candidate_pool: CandidatePoolRecipe = Field(default_factory=CandidatePoolRecipe)
    evaluation: EvaluationRecipe

    @model_validator(mode="after")
    def validate_recipe_constraints(self) -> RecallMethodRecipe:
        if isinstance(self.generator, DeepSeekPromptGeneratorRecipe):
            if self.generator.max_generated_actions > self.retrieval.max_total_actions:
                raise ValueError("max_generated_actions must not exceed max_total_actions")
        if (
            self.candidate_pool.policy_version == "canonical-id-first-v1"
            and self.evaluation.compare_with is None
        ):
            raise ValueError("canonical-id-first-v1 is only permitted for historical comparison")
        return self

    def canonical_bytes(self) -> bytes:
        """Return a stable serialization suitable for recipe identity and locking."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")

    def __hash__(self) -> int:
        return hash(self.canonical_bytes())


class ArtifactBinding(DomainModel):
    path: SafeRelativePath
    sha256: Sha256


class HistoricalBaselineBinding(DomainModel):
    """Hash-bound records used only for an approved historical comparison."""

    query_ids: list[NonEmptyStr] = Field(min_length=1)
    gold_associations: ArtifactBinding
    business_results: ArtifactBinding
    executions: ArtifactBinding


class FormalRunInputBinding(DomainModel):
    """Replaceable, offline formal-run inputs for a query sample."""

    gold_associations: ArtifactBinding
    identifier_map: ArtifactBinding
    bound_paper_sources: list[ArtifactBinding] = Field(default_factory=list)
    seed_candidates: list[SeedCandidate] = Field(default_factory=list)
    historical_baseline: HistoricalBaselineBinding | None = None


class SampleBinding(DomainModel):
    """Recipe-independent declarative binding for a frozen query sample.

    The input adapter later verifies catalog/source hashes and resolves opaque
    evaluation identifiers.  These ID lists only support cross-artifact
    preflight and are never generation inputs.
    """

    sample_id: NonEmptyStr
    query_ids: list[NonEmptyStr] = Field(min_length=1)
    gold_document_catalog: ArtifactBinding | None = None
    gold_document_catalog_manifest: ArtifactBinding | None = None
    gold_ids: list[NonEmptyStr] = Field(default_factory=list)
    seed_canonical_ids: list[NonEmptyStr] = Field(default_factory=list)
    legacy_candidate_pool_policy: Literal["canonical-id-first-v1"] | None = None
    frozen_inputs: FormalRunInputBinding | None = None

    @field_validator("query_ids")
    @classmethod
    def validate_unique_query_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("query_ids must be unique")
        return values


class LoadedRecallRecipe(DomainModel):
    recipe: RecallMethodRecipe
    recipe_path: Path
    recipe_bytes: bytes
    recipe_sha256: Sha256
    prompt_bytes: bytes | None = None
    prompt_sha256: Sha256 | None = None


class LoadedSampleBinding(DomainModel):
    binding: SampleBinding
    binding_path: Path
    binding_bytes: bytes
    binding_sha256: Sha256


def load_recall_recipe(path: str | Path) -> LoadedRecallRecipe:
    """Load and bind a declarative recipe without resolving execution artifacts."""
    recipe_path = Path(path)
    recipe_bytes = recipe_path.read_bytes()
    recipe = RecallMethodRecipe.model_validate(_load_yaml_mapping(recipe_bytes, recipe_path))
    prompt_bytes: bytes | None = None
    prompt_sha256: Sha256 | None = None
    if isinstance(recipe.generator, DeepSeekPromptGeneratorRecipe):
        prompt_path = _workspace_path(recipe.generator.prompt)
        prompt_bytes = prompt_path.read_bytes()
        prompt_sha256 = _sha256(prompt_bytes)
        prompt_payload = _load_yaml_mapping(prompt_bytes, prompt_path)
        if prompt_payload.get("model") != recipe.generator.model:
            raise ValueError("recipe generator model does not match prompt artifact model")
    return LoadedRecallRecipe(
        recipe=recipe,
        recipe_path=recipe_path.resolve(),
        recipe_bytes=recipe_bytes,
        recipe_sha256=_sha256(recipe_bytes),
        prompt_bytes=prompt_bytes,
        prompt_sha256=prompt_sha256,
    )


def load_sample_binding(path: str | Path) -> LoadedSampleBinding:
    """Load and bind a sample declaration without reading its frozen sources."""
    binding_path = Path(path)
    binding_bytes = binding_path.read_bytes()
    return LoadedSampleBinding(
        binding=SampleBinding.model_validate(_load_yaml_mapping(binding_bytes, binding_path)),
        binding_path=binding_path.resolve(),
        binding_bytes=binding_bytes,
        binding_sha256=_sha256(binding_bytes),
    )


def authorize_live_backend(recipe: RecallMethodRecipe, *, allow_live: bool) -> None:
    """Require the runtime-only flag before a live backend may be used."""
    if recipe.retrieval.backend == "live_provider" and not allow_live:
        raise PermissionError("live_provider requires explicit runtime authorization")


def validate_recipe_sample_preflight(
    recipe: RecallMethodRecipe,
    sample: SampleBinding,
    *,
    blind_sample: SampleBinding | None = None,
) -> None:
    """Validate recipe/sample invariants that no independent loader can prove.

    This remains offline and does not parse catalog or identifier-map content.
    The frozen-input/evaluator layers later repeat these checks after resolving
    the actual identifiers.
    """
    if recipe.generator.gold_visibility == "oracle" and (
        sample.gold_document_catalog is None or sample.gold_document_catalog_manifest is None
    ):
        raise ValueError("oracle generation requires Gold-document catalog and manifest bindings")
    if (
        recipe.candidate_pool.policy_version == "canonical-id-first-v1"
        and sample.legacy_candidate_pool_policy != "canonical-id-first-v1"
    ):
        raise ValueError("historical recipe binding must prove the legacy policy")
    if set(sample.gold_ids).intersection(sample.seed_canonical_ids):
        raise ValueError("Gold IDs and citation seeds must not overlap")
    if blind_sample is not None and set(sample.query_ids).intersection(blind_sample.query_ids):
        raise ValueError("Oracle and Blind sample query IDs must not overlap")


def _load_yaml_mapping(raw: bytes, path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def _workspace_path(relative_path: SafeRelativePath) -> Path:
    root = Path.cwd().resolve()
    resolved = (root / relative_path).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("recipe path must resolve within the workspace") from error
    return resolved


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"
