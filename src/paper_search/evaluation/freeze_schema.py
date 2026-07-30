"""Versioned, fail-closed schemas for frozen evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from paper_search.domain.models import (
    DomainModel,
    NonEmptyStr,
    SafeRelativePath,
    Sha256,
    validate_safe_relative_path,
)


PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class _FreezeModel(DomainModel):
    """Strict normative model base used only by the freeze contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class LegacySourceFileV1(_FreezeModel):
    path: NonEmptyStr
    raw_path: SafeRelativePath
    row_count: PositiveInt
    byte_count: PositiveInt
    sha256: Sha256


class LegacyPartitionV1(_FreezeModel):
    count: PositiveInt
    gold_path: SafeRelativePath
    gold_sha256: Sha256
    ids_path: SafeRelativePath
    ids_sha256: Sha256
    zero_answer_policy: Literal["reject", "allow"]
    labels_complete: Literal[True]


class LegacyWorkPackageV1(_FreezeModel):
    count: PositiveInt
    ids_path: SafeRelativePath
    ids_sha256: Sha256
    source_path: SafeRelativePath | None = None
    source_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_source_binding(self) -> LegacyWorkPackageV1:
        if (self.source_path is None) != (self.source_sha256 is None):
            raise ValueError("legacy work package source binding is invalid")
        return self


class FreezeManifestV1(_FreezeModel):
    """In-memory projection of the exact approved, unversioned legacy format."""

    schema_version: Literal["paper-search-freeze-v1"]
    repo_id: Literal["CarlanLark/pasa-dataset"]
    revision: Literal["232428b0c867268c3b8ded90db4d98c1b30501d6"]
    license: Literal["CC-BY-NC-SA-4.0"]
    access: Literal["gated-hugging-face-dataset"]
    random_seed: Literal[20260714]
    sampling_algorithm: Literal["answer-count-largest-remainder-v1"]
    status: Literal["frozen"]
    source_files: list[LegacySourceFileV1]
    partitions: dict[Literal["dev", "validation", "simulated_test"], LegacyPartitionV1]
    work_package_sampling: Literal["answer-count-largest-remainder-v1-seeded-offsets"]
    work_packages: dict[
        Literal["type_domain", "constraints", "overlap"], LegacyWorkPackageV1
    ]
    prepared_manifest_sha256: Sha256
    freeze_report_path: SafeRelativePath
    freeze_report_sha256: Sha256

    @model_validator(mode="after")
    def validate_exact_legacy_shape(self) -> FreezeManifestV1:
        if set(self.partitions) != {"dev", "validation", "simulated_test"}:
            raise ValueError("legacy partitions are invalid")
        if {
            name: partition.count for name, partition in self.partitions.items()
        } != {"dev": 60, "validation": 30, "simulated_test": 50}:
            raise ValueError("legacy partition counts are invalid")
        if set(self.work_packages) != {"type_domain", "constraints", "overlap"}:
            raise ValueError("legacy work packages are invalid")
        expected_sources = {
            "AutoScholarQuery/dev.jsonl": ("raw/AutoScholarQuery/dev.jsonl", 1000),
            "AutoScholarQuery/test.jsonl": ("raw/AutoScholarQuery/test.jsonl", 1000),
            "RealScholarQuery/test.jsonl": ("raw/RealScholarQuery/test.jsonl", 50),
        }
        if len(self.source_files) != len(expected_sources) or {
            item.path: (item.raw_path, item.row_count) for item in self.source_files
        } != expected_sources:
            raise ValueError("legacy source files are invalid")
        if {
            name: package.count for name, package in self.work_packages.items()
        } != {"type_domain": 90, "constraints": 40, "overlap": 20}:
            raise ValueError("legacy work package counts are invalid")
        if (
            self.work_packages["type_domain"].source_path is None
            or self.work_packages["constraints"].source_path is None
            or self.work_packages["overlap"].source_path is not None
        ):
            raise ValueError("legacy work packages are invalid")
        expected_package_paths = {
            "type_domain": (
                "annotation_work/type_domain_source.jsonl",
                "splits/type_domain_annotation.ids.json",
            ),
            "constraints": (
                "annotation_work/constraints_source.jsonl",
                "splits/constraint_annotation.ids.json",
            ),
            "overlap": (None, "splits/overlap_annotation.ids.json"),
        }
        if {
            name: (package.source_path, package.ids_path)
            for name, package in self.work_packages.items()
        } != expected_package_paths:
            raise ValueError("legacy work package paths are invalid")
        return self


class FrozenPartitionV2(_FreezeModel):
    name: Literal["dev", "validation"]
    path: SafeRelativePath
    query_count: PositiveInt
    sha256: Sha256
    zero_answer_policy: Literal["allow", "forbid"]


class IdentifierMapBindingV2(_FreezeModel):
    path: SafeRelativePath
    sha256: Sha256
    entry_count: PositiveInt


class FreezeApprovalReportV2(_FreezeModel):
    schema_version: Literal["freeze-approval-v2"]
    approval_requested: Literal[True]
    approved_at: datetime
    approver_ref: NonEmptyStr
    audit_sha256: Sha256
    partition_hashes: dict[Literal["dev", "validation"], Sha256]
    identifier_map_sha256: Sha256

    @model_validator(mode="after")
    def validate_partition_hashes(self) -> FreezeApprovalReportV2:
        if set(self.partition_hashes) != {"dev", "validation"}:
            raise ValueError("approval partition hashes are invalid")
        return self


class FreezeApprovalBindingV2(_FreezeModel):
    report_path: SafeRelativePath
    report_sha256: Sha256
    approved_at: datetime
    approver_ref: NonEmptyStr


class FreezeManifestV2(_FreezeModel):
    schema_version: Literal["paper-search-freeze-v2"]
    dataset_revision: NonEmptyStr
    created_at: datetime
    annotation_status: Literal["frozen"]
    freeze_status: Literal["approved"]
    partitions: list[FrozenPartitionV2]
    gold_sha256: Sha256
    identifier_map: IdentifierMapBindingV2
    partition_immutability: Literal["content_addressed"]
    approval: FreezeApprovalBindingV2

    @model_validator(mode="after")
    def validate_partition_identity(self) -> FreezeManifestV2:
        names = [partition.name for partition in self.partitions]
        if names != ["dev", "validation"]:
            raise ValueError("V2 partitions must be dev then validation")
        if len({partition.path for partition in self.partitions}) != 2:
            raise ValueError("V2 partition paths must be unique")
        if len({partition.sha256 for partition in self.partitions}) != 2:
            raise ValueError("V2 partition hashes must be unique")
        return self


FreezeManifest: TypeAlias = Annotated[
    FreezeManifestV1 | FreezeManifestV2,
    Field(discriminator="schema_version"),
]
_FREEZE_MANIFEST_ADAPTER: TypeAdapter[FreezeManifest] = TypeAdapter(FreezeManifest)


def sha256_bytes(content: bytes) -> Sha256:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_gold_set_sha256(dev_sha256: Sha256, validation_sha256: Sha256) -> Sha256:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "paper-search-gold-set-v1",
                "partitions": [
                    {"name": "dev", "sha256": dev_sha256},
                    {"name": "validation", "sha256": validation_sha256},
                ],
            }
        )
    )


def _confined_path(data_root: Path, relative_path: str) -> Path:
    root = data_root.resolve()
    candidate = root / Path(validate_safe_relative_path(relative_path))
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("freeze manifest is invalid") from None
    return resolved


def read_confined_bytes(data_root: Path, relative_path: str) -> bytes:
    """Read one regular, non-escaping artifact through a single descriptor."""
    path = _confined_path(data_root, relative_path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError("freeze manifest is invalid") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("freeze manifest is invalid")
        content = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            content.extend(chunk)
        return bytes(content)
    except OSError:
        raise ValueError("freeze manifest is invalid") from None
    finally:
        os.close(descriptor)


def parse_freeze_manifest_bytes(content: bytes) -> FreezeManifest:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("freeze manifest is invalid") from None
    if not isinstance(payload, dict):
        raise ValueError("freeze manifest is invalid")
    try:
        if payload.get("schema_version") == "paper-search-freeze-v1":
            raise ValueError("freeze manifest is invalid")
        if "schema_version" in payload:
            return _FREEZE_MANIFEST_ADAPTER.validate_json(content, strict=True)
        if payload.get("status") != "frozen":
            raise ValueError("freeze manifest is invalid")
        return FreezeManifestV1.model_validate(
            {"schema_version": "paper-search-freeze-v1", **payload}, strict=True
        )
    except ValidationError:
        raise ValueError("freeze manifest is invalid") from None


def load_freeze_manifest(path: Path, *, data_root: Path) -> FreezeManifest:
    root = data_root.resolve()
    try:
        relative_path = path.resolve().relative_to(root).as_posix()
    except ValueError:
        raise ValueError("freeze manifest is invalid") from None
    return parse_freeze_manifest_bytes(read_confined_bytes(root, relative_path))
