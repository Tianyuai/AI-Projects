"""Fail-closed manifest and artifact verification for trained action rankers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from paper_search.domain.models import (
    DomainModel,
    NonEmptyStr,
    SafeRelativePath,
    Sha256,
    UnitFloat,
)


class ModelBundleManifest(DomainModel):
    schema_version: Literal["action-ranker-bundle-v1"] = "action-ranker-bundle-v1"
    model_id: NonEmptyStr
    base_model: NonEmptyStr
    framework: Literal["transformers"]
    training_data_manifest_sha256: Sha256
    confidence_threshold: UnitFloat
    device_priority: list[Literal["cuda", "cpu"]] = Field(min_length=1)
    license: NonEmptyStr
    artifacts: dict[SafeRelativePath, Sha256]

    @model_validator(mode="after")
    def validate_artifacts_and_devices(self) -> ModelBundleManifest:
        if not self.artifacts:
            raise ValueError("model bundle must declare at least one artifact")
        if len(self.device_priority) != len(set(self.device_priority)):
            raise ValueError("device priority entries must be unique")
        if "cpu" not in self.device_priority:
            raise ValueError("model bundle must declare a CPU fallback")
        return self


class VerifiedModelBundle(DomainModel):
    model_id: NonEmptyStr
    bundle_sha256: Sha256
    verified_artifacts: list[SafeRelativePath]


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def verify_model_bundle(
    root: Path,
    manifest: ModelBundleManifest,
) -> VerifiedModelBundle:
    manifest = ModelBundleManifest.model_validate(manifest)
    resolved_root = root.resolve(strict=True)
    verified: list[str] = []
    for relative_path, expected_hash in sorted(manifest.artifacts.items()):
        artifact = (resolved_root / relative_path).resolve(strict=True)
        if not artifact.is_relative_to(resolved_root) or not artifact.is_file():
            raise ValueError("model artifact must be a regular file inside the bundle")
        actual_hash = _sha256(artifact.read_bytes())
        if actual_hash != expected_hash:
            raise ValueError(f"artifact hash mismatch: {relative_path}")
        verified.append(relative_path)
    manifest_bytes = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return VerifiedModelBundle(
        model_id=manifest.model_id,
        bundle_sha256=_sha256(manifest_bytes),
        verified_artifacts=verified,
    )


__all__ = ["ModelBundleManifest", "VerifiedModelBundle", "verify_model_bundle"]
