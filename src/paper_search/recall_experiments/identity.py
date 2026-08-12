"""Strict, versioned execution identity for comparable recall reports."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, Field, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, Sha256


def _reject_zero_sha256(value: str) -> str:
    if value == "sha256:" + "0" * 64:
        raise ValueError("execution identity SHA-256 must not be zero")
    return value


IdentitySha256 = Annotated[Sha256, AfterValidator(_reject_zero_sha256)]
StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]


class ReplayRuntimeIdentity(DomainModel):
    backend_identity: Literal["sealed_dependency_snapshot"]
    budget_policy: Literal["recall-replay-v1"]
    pricing_provenance: Literal["snapshot_bound_usage"]
    snapshot_manifest_sha256: IdentitySha256


class SnapshotUnavailableRuntimeIdentity(DomainModel):
    identity_schema_version: Literal["candidate-recall-unavailable-runtime-v1"]
    backend_identity: Literal["snapshot_unavailable"]


class LiveDependencyIdentity(DomainModel):
    identity_schema_version: Literal["live-dependency-runtime-identity-v1"]
    provider: NonEmptyStr
    dependency: Literal["llm", "openalex", "semantic_scholar"]
    adapter: NonEmptyStr
    model: NonEmptyStr | None
    version: NonEmptyStr
    endpoints: tuple[NonEmptyStr, ...]
    operations: tuple[NonEmptyStr, ...]
    pricing_policy_sha256: IdentitySha256
    controller_policy_sha256: IdentitySha256

    @model_validator(mode="after")
    def validate_dependency_surface(self) -> LiveDependencyIdentity:
        if not self.endpoints or not self.operations:
            raise ValueError("live dependency identity requires endpoints and operations")
        if len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError("live dependency endpoints must be unique")
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("live dependency operations must be unique")
        return self


class LiveRuntimeIdentity(DomainModel):
    identity_schema_version: Literal["candidate-recall-live-runtime-v1"]
    controller_policy_sha256: IdentitySha256
    pricing_policy_sha256: IdentitySha256
    dependencies: dict[
        Literal["search", "citation", "llm"], LiveDependencyIdentity
    ]

    @model_validator(mode="after")
    def validate_shared_policies(self) -> LiveRuntimeIdentity:
        if set(self.dependencies) != {"search", "citation", "llm"}:
            raise ValueError("live runtime must bind search, citation, and LLM dependencies")
        controller_hashes = {
            item.controller_policy_sha256 for item in self.dependencies.values()
        }
        pricing_hashes = {
            item.pricing_policy_sha256 for item in self.dependencies.values()
        }
        if controller_hashes != {self.controller_policy_sha256}:
            raise ValueError("live dependency controller policies do not match")
        if pricing_hashes != {self.pricing_policy_sha256}:
            raise ValueError("live dependency pricing policies do not match")
        if self.dependencies["llm"].dependency != "llm":
            raise ValueError("live LLM dependency identity is invalid")
        if self.dependencies["citation"].dependency != "semantic_scholar":
            raise ValueError("live citation dependency identity is invalid")
        return self


class ExecutionIdentity(DomainModel):
    """Complete comparison identity emitted by ``_execution_identity``."""

    identity_schema_version: Literal["candidate-recall-execution-identity-v1"]
    method_id: NonEmptyStr
    recipe_sha256: IdentitySha256
    sample_sha256: IdentitySha256
    prompt_sha256: IdentitySha256 | None
    generator_type: Literal["manual_actions", "fixed_actions", "deepseek_prompt"]
    generator_model: NonEmptyStr | None
    retrieval_backend: Literal["snapshot_replay", "live_provider"]
    snapshot_manifest_sha256: IdentitySha256 | None
    actions_sha256: IdentitySha256 | None
    max_total_actions: StrictPositiveInt
    max_results_per_action: StrictPositiveInt
    candidate_pool_policy_version: Literal[
        "production-dedup-v1", "canonical-id-first-v1"
    ]
    repeat_count: Annotated[int, Field(strict=True, gt=0, le=3)]
    max_repeat_attempts: Annotated[int, Field(strict=True, gt=0, le=5)]
    live_authorized: Annotated[bool, Field(strict=True)]
    runtime: (
        ReplayRuntimeIdentity
        | SnapshotUnavailableRuntimeIdentity
        | LiveRuntimeIdentity
    )

    @model_validator(mode="after")
    def validate_conditional_bindings(self) -> ExecutionIdentity:
        if self.max_repeat_attempts < self.repeat_count:
            raise ValueError("max repeat attempts must cover repeat count")
        if self.generator_type in {"manual_actions", "fixed_actions"}:
            if self.prompt_sha256 is not None or self.generator_model is not None:
                raise ValueError("non-LLM generators must not bind prompt/model identity")
            if self.actions_sha256 is None:
                raise ValueError("fixed/manual generators require action identity")
        elif (
            self.prompt_sha256 is None
            or self.generator_model is None
            or self.actions_sha256 is not None
        ):
            raise ValueError("DeepSeek generator requires prompt/model and no action artifact")

        if self.retrieval_backend == "snapshot_replay":
            if self.live_authorized:
                raise ValueError("snapshot replay must not use live authorization")
            if isinstance(self.runtime, ReplayRuntimeIdentity):
                if (
                    self.snapshot_manifest_sha256 is None
                    or self.snapshot_manifest_sha256
                    != self.runtime.snapshot_manifest_sha256
                ):
                    raise ValueError("snapshot manifest identity must match replay runtime")
            elif not (
                isinstance(self.runtime, SnapshotUnavailableRuntimeIdentity)
                and self.snapshot_manifest_sha256 is None
            ):
                raise ValueError("snapshot replay runtime identity is invalid")
        elif (
            not self.live_authorized
            or self.snapshot_manifest_sha256 is not None
            or not isinstance(self.runtime, LiveRuntimeIdentity)
        ):
            raise ValueError("live retrieval requires authorized, snapshot-free live runtime")
        return self


__all__ = ["ExecutionIdentity"]
