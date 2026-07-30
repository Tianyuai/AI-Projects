"""Versioned, fail-closed schemas for frozen evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Iterator, Literal, TypeAlias

from pydantic import (
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

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

    @field_validator("approved_at")
    @classmethod
    def normalize_approved_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approval timestamp must be timezone-aware")
        return value.astimezone(UTC)

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

    @field_validator("approved_at")
    @classmethod
    def normalize_approved_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approval timestamp must be timezone-aware")
        return value.astimezone(UTC)


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

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("manifest timestamp must be timezone-aware")
        return value.astimezone(UTC)

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


def canonical_document_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_object(content: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("freeze manifest is invalid")
            result[key] = value
        return result

    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError("freeze manifest is invalid") from None
    if not isinstance(value, dict):
        raise ValueError("freeze manifest is invalid")
    return value


@dataclass
class BoundArtifact:
    """Exact bytes and identity read through one still-open trusted descriptor."""

    relative_path: SafeRelativePath
    descriptor: int
    content: bytes
    sha256: Sha256
    device: int
    inode: int

    def close(self) -> None:
        os.close(self.descriptor)


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    content = bytearray()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return bytes(content)


def _open_posix_confined(data_root: Path, relative_path: SafeRelativePath) -> int:
    root = data_root.resolve()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        descriptor = os.open(root, flags)
        descriptors.append(descriptor)
        parts = relative_path.split("/")
        for index, part in enumerate(parts):
            final = index == len(parts) - 1
            next_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if not final:
                next_flags |= getattr(os, "O_DIRECTORY", 0)
            next_descriptor = os.open(part, next_flags, dir_fd=descriptor)
            descriptors.append(next_descriptor)
            descriptor = next_descriptor
        return descriptors.pop()
    except OSError:
        raise ValueError("freeze manifest is invalid") from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _normalized_windows_path(value: str) -> str:
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.abspath(value))


def _open_windows_confined(data_root: Path, relative_path: SafeRelativePath) -> int:
    import ctypes
    import msvcrt

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [("file_attributes", ctypes.c_uint32), ("reparse_tag", ctypes.c_uint32)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.restype = ctypes.c_void_p
    get_handle_info = kernel32.GetFileInformationByHandleEx
    get_final_path = kernel32.GetFinalPathNameByHandleW
    invalid = ctypes.c_void_p(-1).value
    open_reparse = 0x00200000
    backup_semantics = 0x02000000
    handles: list[int] = []
    try:
        root_handle = create_file(
            str(data_root.resolve()),
            0x80000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            backup_semantics | open_reparse,
            None,
        )
        if root_handle == invalid:
            raise OSError
        handles.append(int(root_handle))
        buffer = ctypes.create_unicode_buffer(32768)
        if not get_final_path(ctypes.c_void_p(root_handle), buffer, len(buffer), 0):
            raise OSError
        root_final = _normalized_windows_path(buffer.value)
        candidate = data_root.resolve()
        for index, part in enumerate(relative_path.split("/")):
            candidate = candidate / part
            final = index == len(relative_path.split("/")) - 1
            handle = create_file(
                str(candidate),
                0x80000000,
                0x00000001 | 0x00000002 | 0x00000004,
                None,
                3,
                (backup_semantics if not final else 0) | open_reparse,
                None,
            )
            if handle == invalid:
                raise OSError
            handles.append(int(handle))
            info = FileAttributeTagInfo()
            if not get_handle_info(
                ctypes.c_void_p(handle), 9, ctypes.byref(info), ctypes.sizeof(info)
            ) or info.file_attributes & 0x00000400:
                raise OSError
            if not get_final_path(ctypes.c_void_p(handle), buffer, len(buffer), 0):
                raise OSError
            final_path = _normalized_windows_path(buffer.value)
            if final_path != root_final and not final_path.startswith(root_final + "\\"):
                raise OSError
        handle = handles.pop()
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except OSError:
        raise ValueError("freeze manifest is invalid") from None
    finally:
        for handle in reversed(handles):
            kernel32.CloseHandle(ctypes.c_void_p(handle))


@contextmanager
def open_confined_artifact(
    data_root: Path,
    relative_path: SafeRelativePath,
) -> Iterator[BoundArtifact]:
    """Open each path component safely and keep the evidence descriptor alive."""
    try:
        normalized = validate_safe_relative_path(relative_path)
        descriptor = (
            _open_windows_confined(data_root, normalized)
            if os.name == "nt"
            else _open_posix_confined(data_root, normalized)
        )
    except (OSError, ValueError):
        raise ValueError("freeze manifest is invalid") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("freeze manifest is invalid")
        content = _read_descriptor(descriptor)
        yield BoundArtifact(
            relative_path=normalized,
            descriptor=descriptor,
            content=content,
            sha256=sha256_bytes(content),
            device=metadata.st_dev,
            inode=metadata.st_ino,
        )
    except OSError:
        raise ValueError("freeze manifest is invalid") from None
    finally:
        os.close(descriptor)


def read_confined_bytes(data_root: Path, relative_path: SafeRelativePath) -> bytes:
    with open_confined_artifact(data_root, relative_path) as artifact:
        return artifact.content


def _open_posix_report_parent(data_root: Path, relative_path: SafeRelativePath) -> tuple[int, str]:
    parts = relative_path.split("/")
    root = data_root.resolve()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root, flags)
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _write_descriptor(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        offset += os.write(descriptor, content[offset:])
    os.fsync(descriptor)


def publish_confined_bytes_no_overwrite(
    data_root: Path,
    relative_path: SafeRelativePath,
    content: bytes,
) -> Literal["created", "matched"]:
    """Atomically create a confined report or accept only its exact bytes."""
    try:
        normalized = validate_safe_relative_path(relative_path)
    except ValueError:
        raise ValueError("freeze manifest is invalid") from None
    if os.name == "nt":
        import ctypes
        import msvcrt

        class FileAttributeTagInfo(ctypes.Structure):
            _fields_ = [("file_attributes", ctypes.c_uint32), ("reparse_tag", ctypes.c_uint32)]

        root = data_root.resolve()
        target = root.joinpath(*normalized.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.restype = ctypes.c_void_p
        get_handle_info = kernel32.GetFileInformationByHandleEx
        get_final_path = kernel32.GetFinalPathNameByHandleW
        invalid = ctypes.c_void_p(-1).value
        root_handle = create_file(
            str(root),
            0x80000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x02000000 | 0x00200000,
            None,
        )
        if root_handle == invalid:
            raise ValueError("freeze manifest is invalid")
        buffer = ctypes.create_unicode_buffer(32768)
        if not get_final_path(ctypes.c_void_p(root_handle), buffer, len(buffer), 0):
            kernel32.CloseHandle(ctypes.c_void_p(root_handle))
            raise ValueError("freeze manifest is invalid")
        root_final = _normalized_windows_path(buffer.value)
        handle = create_file(
            str(target),
            0x40000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            1,
            0x00200000,
            None,
        )
        if handle == invalid:
            kernel32.CloseHandle(ctypes.c_void_p(root_handle))
            try:
                with open_confined_artifact(root, normalized) as artifact:
                    if artifact.content == content:
                        return "matched"
            except ValueError:
                pass
            raise FileExistsError("refusing to overwrite frozen file") from None
        info = FileAttributeTagInfo()
        if not get_handle_info(
            ctypes.c_void_p(handle), 9, ctypes.byref(info), ctypes.sizeof(info)
        ) or info.file_attributes & 0x00000400 or not get_final_path(
            ctypes.c_void_p(handle), buffer, len(buffer), 0
        ):
            kernel32.CloseHandle(ctypes.c_void_p(handle))
            kernel32.CloseHandle(ctypes.c_void_p(root_handle))
            raise ValueError("freeze manifest is invalid")
        target_final = _normalized_windows_path(buffer.value)
        if target_final != root_final and not target_final.startswith(root_final + "\\"):
            kernel32.CloseHandle(ctypes.c_void_p(handle))
            kernel32.CloseHandle(ctypes.c_void_p(root_handle))
            raise ValueError("freeze manifest is invalid")
        try:
            descriptor = msvcrt.open_osfhandle(
                int(handle), os.O_WRONLY | getattr(os, "O_BINARY", 0)
            )
        except OSError:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
            kernel32.CloseHandle(ctypes.c_void_p(root_handle))
            raise ValueError("freeze manifest is invalid") from None
        try:
            _write_descriptor(descriptor, content)
        finally:
            os.close(descriptor)
            kernel32.CloseHandle(ctypes.c_void_p(root_handle))
        try:
            with open_confined_artifact(root, normalized) as artifact:
                if artifact.content != content:
                    raise ValueError("freeze manifest is invalid")
        except ValueError:
            raise ValueError("freeze manifest is invalid") from None
        return "created"
    try:
        parent_descriptor, filename = _open_posix_report_parent(data_root, normalized)
    except OSError:
        raise ValueError("freeze manifest is invalid") from None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(filename, flags, 0o600, dir_fd=parent_descriptor)
        except FileExistsError:
            with open_confined_artifact(data_root, normalized) as artifact:
                if artifact.content == content:
                    return "matched"
            raise FileExistsError("refusing to overwrite frozen file") from None
        try:
            _write_descriptor(descriptor, content)
        finally:
            os.close(descriptor)
        return "created"
    finally:
        os.close(parent_descriptor)


def parse_freeze_manifest_bytes(content: bytes) -> FreezeManifest:
    payload = _json_object(content)
    try:
        if payload.get("schema_version") == "paper-search-freeze-v1":
            raise ValueError("freeze manifest is invalid")
        if "schema_version" in payload:
            manifest = _FREEZE_MANIFEST_ADAPTER.validate_json(content, strict=True)
            if not isinstance(manifest, FreezeManifestV2) or (
                canonical_document_bytes(manifest.model_dump(mode="json")) != content
            ):
                raise ValueError("freeze manifest is invalid")
            return manifest
        if payload.get("status") != "frozen":
            raise ValueError("freeze manifest is invalid")
        return FreezeManifestV1.model_validate(
            {"schema_version": "paper-search-freeze-v1", **payload}, strict=True
        )
    except ValidationError:
        raise ValueError("freeze manifest is invalid") from None


def _validated_report_bytes(content: bytes) -> FreezeApprovalReportV2:
    _json_object(content)
    try:
        report = FreezeApprovalReportV2.model_validate_json(content, strict=True)
    except ValidationError:
        raise ValueError("freeze manifest is invalid") from None
    if canonical_document_bytes(report.model_dump(mode="json")) != content:
        raise ValueError("freeze manifest is invalid")
    return report


def _validate_v2_bindings(data_root: Path, manifest: FreezeManifestV2) -> None:
    with open_confined_artifact(data_root, manifest.approval.report_path) as report_artifact:
        if report_artifact.sha256 != manifest.approval.report_sha256:
            raise ValueError("freeze manifest is invalid")
        report = _validated_report_bytes(report_artifact.content)
    if (
        report.approved_at != manifest.approval.approved_at
        or report.approver_ref != manifest.approval.approver_ref
        or report.identifier_map_sha256 != manifest.identifier_map.sha256
    ):
        raise ValueError("freeze manifest is invalid")
    partition_hashes: dict[str, Sha256] = {}
    for partition in manifest.partitions:
        with open_confined_artifact(data_root, partition.path) as artifact:
            if artifact.sha256 != partition.sha256:
                raise ValueError("freeze manifest is invalid")
            try:
                rows = [_json_object(line) for line in artifact.content.splitlines() if line]
                query_ids = [row.get("query_id") for row in rows]
            except ValueError:
                raise ValueError("freeze manifest is invalid") from None
            if (
                len(rows) != partition.query_count
                or not all(isinstance(query_id, str) and query_id for query_id in query_ids)
                or len(set(query_ids)) != len(query_ids)
            ):
                raise ValueError("freeze manifest is invalid")
            partition_hashes[partition.name] = artifact.sha256
    with open_confined_artifact(data_root, manifest.identifier_map.path) as artifact:
        if artifact.sha256 != manifest.identifier_map.sha256:
            raise ValueError("freeze manifest is invalid")
        value = _json_object(artifact.content)
        if len(value) != manifest.identifier_map.entry_count:
            raise ValueError("freeze manifest is invalid")
    if (
        report.partition_hashes != partition_hashes
        or manifest.gold_sha256
        != canonical_gold_set_sha256(partition_hashes["dev"], partition_hashes["validation"])
    ):
        raise ValueError("freeze manifest is invalid")


def load_freeze_manifest(path: Path, *, data_root: Path) -> FreezeManifest:
    root = data_root.resolve()
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError:
        raise ValueError("freeze manifest is invalid") from None
    with open_confined_artifact(root, relative_path) as artifact:
        manifest = parse_freeze_manifest_bytes(artifact.content)
    if isinstance(manifest, FreezeManifestV2):
        _validate_v2_bindings(root, manifest)
    return manifest
