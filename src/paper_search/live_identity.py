from __future__ import annotations

from typing import Literal, Protocol

from pydantic import ConfigDict, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, Sha256

__all__ = [
    "LiveDependencyEvidence",
    "LiveIdentityController",
    "LiveProviderDescriptor",
    "SelfIdentifyingLiveDependency",
]


class _FrozenIdentityModel(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class LiveProviderDescriptor(_FrozenIdentityModel):
    identity_schema_version: Literal["live-provider-descriptor-v1"]
    provider: NonEmptyStr
    dependency: Literal["openalex", "semantic_scholar", "llm"]
    adapter: NonEmptyStr
    version: NonEmptyStr
    model: NonEmptyStr | None
    endpoints: tuple[NonEmptyStr, ...]
    operations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def validate_surface(self) -> LiveProviderDescriptor:
        if not self.endpoints or len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError("live provider endpoints must be nonempty and unique")
        if not self.operations or len(set(self.operations)) != len(self.operations):
            raise ValueError("live provider operations must be nonempty and unique")
        return self


class LiveDependencyEvidence(_FrozenIdentityModel):
    identity_schema_version: Literal["live-dependency-evidence-v1"]
    provider: LiveProviderDescriptor
    pricing_policy_sha256: Sha256
    controller_policy_sha256: Sha256
    formal_live: Literal[True]


class LiveIdentityController(Protocol):
    @property
    def policy_fingerprint(self) -> str: ...

    @property
    def formal_live(self) -> bool: ...


class SelfIdentifyingLiveDependency(Protocol):
    @property
    def live_identity_evidence(self) -> LiveDependencyEvidence: ...
