"""Versioned, fail-closed schemas for frozen evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
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
FreezeEvidenceReason: TypeAlias = Literal[
    "manifest_invalid",
    "approval_invalid",
    "partition_hash_mismatch",
    "partition_count_mismatch",
    "identifier_map_missing",
    "identifier_map_hash_mismatch",
    "identifier_map_coverage_failed",
]


class FreezeEvidenceError(ValueError):
    """One sanitized, structured failure from the shared freeze validation pass."""

    def __init__(
        self,
        reason: FreezeEvidenceReason,
        *additional_reasons: FreezeEvidenceReason,
    ) -> None:
        super().__init__("freeze manifest is invalid")
        self.reasons = tuple(dict.fromkeys((reason, *additional_reasons)))
        self.reason = self.reasons[0]


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


def parse_json_object_bytes(content: bytes) -> dict[str, object]:
    """Parse one UTF-8 JSON object while rejecting duplicate object keys."""
    return _json_object(content)


_WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
        "com¹",
        "com²",
        "com³",
        "lpt¹",
        "lpt²",
        "lpt³",
    }
)
_WINDOWS_NAMESPACE_PREFIXES = ("\\\\?\\", "\\\\.\\", "\\??\\", "\\\\??\\")


def _validate_stable_windows_component(component: str) -> None:
    if component.endswith((".", " ")):
        raise ValueError("freeze manifest is invalid")
    basename = component.split(".", 1)[0].rstrip(" .").casefold()
    if basename in _WINDOWS_RESERVED_BASENAMES:
        raise ValueError("freeze manifest is invalid")


def validate_stable_path(path: Path) -> None:
    """Reject Windows spellings whose ordinary namespace identity is unstable."""
    if os.name != "nt":
        return
    try:
        raw = str(path)
        namespace_spelling = raw.replace("/", "\\").casefold()
        if namespace_spelling.startswith(_WINDOWS_NAMESPACE_PREFIXES):
            raise ValueError("freeze manifest is invalid")
        colon_positions = [index for index, value in enumerate(raw) if value == ":"]
        if colon_positions and not (
            colon_positions == [1]
            and len(raw) >= 3
            and raw[0].isalpha()
            and raw[2] in {"/", "\\"}
        ):
            raise ValueError("freeze manifest is invalid")
        components: tuple[str, ...] = path.parts[1:] if path.anchor else path.parts
        if namespace_spelling.startswith("\\\\"):
            authority = raw.replace("/", "\\")[2:].split("\\")[:2]
            if len(authority) != 2 or not all(authority):
                raise ValueError("freeze manifest is invalid")
            components = (*authority, *components)
        for component in components:
            _validate_stable_windows_component(component)
    except (OSError, RuntimeError, ValueError):
        raise ValueError("freeze manifest is invalid") from None


def _validated_lexical_root(path: Path) -> Path:
    """Return one absolute root only when no lexical component redirects."""
    try:
        absolute = path.absolute()
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current /= part
            try:
                metadata = os.lstat(current)
            except FileNotFoundError:
                break
            attributes = getattr(metadata, "st_file_attributes", 0)
            if stat.S_ISLNK(metadata.st_mode) or attributes & getattr(
                stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400
            ):
                raise ValueError("freeze manifest is invalid")
        resolved = path.resolve()
        if os.path.normcase(os.path.normpath(str(resolved))) != os.path.normcase(
            os.path.normpath(str(absolute))
        ):
            raise ValueError("freeze manifest is invalid")
        return absolute
    except (OSError, RuntimeError, ValueError):
        raise ValueError("freeze manifest is invalid") from None


def validate_lexical_parent(path: Path) -> None:
    """Reject lexical parent components that redirect before target lookup."""
    _validated_lexical_root(path.parent)


@dataclass
class BoundArtifact:
    """Exact bytes and identity read through one still-open trusted descriptor."""

    data_root: Path
    relative_path: SafeRelativePath
    descriptor: int
    content: bytes
    sha256: Sha256
    device: int
    inode: int
    ancestor_descriptors: list[int] = field(default_factory=list)
    path_identities: tuple[tuple[int, int], ...] = ()
    windows_final_paths: tuple[str, ...] = ()
    windows_file_id: tuple[int, bytes] | None = None
    replaceable_manifest: bool = False
    _closed: bool = field(default=False, init=False, repr=False)

    def verify_path_identity(self) -> None:
        """Rehash the descriptor, then reject any lexical/path identity change."""
        if self._closed:
            raise ValueError("freeze manifest is invalid")
        try:
            metadata = os.fstat(self.descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino) != (self.device, self.inode)
                or sha256_bytes(_read_descriptor(self.descriptor)) != self.sha256
                or _validated_lexical_root(self.data_root) != self.data_root
            ):
                raise ValueError("freeze manifest is invalid")
        except (OSError, RuntimeError, ValueError):
            raise ValueError("freeze manifest is invalid") from None
        if os.name == "nt":
            _verify_windows_artifact_identity(self)
            return
        try:
            root_identity = os.stat(self.data_root, follow_symlinks=False)
            expected = self.path_identities
            if (
                not expected
                or (root_identity.st_dev, root_identity.st_ino) != expected[0]
            ):
                raise ValueError("freeze manifest is invalid")
            parts = self.relative_path.split("/")
            for index, part in enumerate(parts):
                parent = self.ancestor_descriptors[index]
                observed = os.stat(part, dir_fd=parent, follow_symlinks=False)
                if (observed.st_dev, observed.st_ino) != expected[index + 1]:
                    raise ValueError("freeze manifest is invalid")
        except OSError:
            raise ValueError("freeze manifest is invalid") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self.descriptor)
        finally:
            if os.name == "nt":
                import ctypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                for handle in reversed(self.ancestor_descriptors):
                    kernel32.CloseHandle(ctypes.c_void_p(handle))
            else:
                for descriptor in reversed(self.ancestor_descriptors):
                    os.close(descriptor)


PartitionRows: TypeAlias = tuple[
    tuple[Literal["dev", "validation"], tuple[dict[str, object], ...]], ...
]


@dataclass(frozen=True)
class ValidatedFreezeEvidence:
    """One validated freeze whose exact source artifacts remain descriptor-bound."""

    manifest: FreezeManifest | None
    manifest_artifact: BoundArtifact
    approval_artifact: BoundArtifact | None = None
    partition_artifacts: tuple[BoundArtifact, ...] = ()
    partition_artifact_names: tuple[Literal["dev", "validation"], ...] = ()
    identifier_map_artifact: BoundArtifact | None = None
    partition_rows: PartitionRows = ()
    reasons: tuple[FreezeEvidenceReason, ...] = ()

    @property
    def artifacts(self) -> tuple[BoundArtifact, ...]:
        optional = (
            (self.approval_artifact,) if self.approval_artifact is not None else ()
        )
        identifier = (
            (self.identifier_map_artifact,)
            if self.identifier_map_artifact is not None
            else ()
        )
        return (
            self.manifest_artifact,
            *optional,
            *self.partition_artifacts,
            *identifier,
        )

    @property
    def artifact_identity_checks(
        self,
    ) -> tuple[tuple[BoundArtifact, FreezeEvidenceReason], ...]:
        approval = (
            ((self.approval_artifact, "approval_invalid"),)
            if self.approval_artifact is not None
            else ()
        )
        partitions = tuple(
            (artifact, "partition_hash_mismatch")
            for artifact in self.partition_artifacts
        )
        identifier = (
            ((self.identifier_map_artifact, "identifier_map_hash_mismatch"),)
            if self.identifier_map_artifact is not None
            else ()
        )
        return (
            (self.manifest_artifact, "manifest_invalid"),
            *approval,
            *partitions,
            *identifier,
        )


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


def _open_posix_confined(
    data_root: Path, relative_path: SafeRelativePath
) -> tuple[int, list[int], tuple[tuple[int, int], ...]]:
    root = data_root
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        descriptor = os.open(root, flags)
        descriptors.append(descriptor)
        parts = relative_path.split("/")
        for index, part in enumerate(parts):
            is_final = index == len(parts) - 1
            next_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            if not is_final:
                next_flags |= getattr(os, "O_DIRECTORY", 0)
            next_descriptor = os.open(part, next_flags, dir_fd=descriptor)
            descriptors.append(next_descriptor)
            descriptor = next_descriptor
        identities = tuple(
            (metadata.st_dev, metadata.st_ino)
            for metadata in (os.fstat(item) for item in descriptors)
        )
        final_descriptor = descriptors.pop()
        ancestors = descriptors
        descriptors = []
        return final_descriptor, ancestors, identities
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


def _windows_file_id(handle: int) -> tuple[int, bytes]:
    import ctypes

    class FileId128(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_byte * 16)]

    class FileIdInfo(ctypes.Structure):
        _fields_ = [
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", FileId128),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    result = FileIdInfo()
    if not kernel32.GetFileInformationByHandleEx(
        ctypes.c_void_p(handle),
        18,  # FileIdInfo
        ctypes.byref(result),
        ctypes.sizeof(result),
    ):
        raise OSError(ctypes.get_last_error(), "GetFileInformationByHandleEx failed")
    return result.volume_serial_number, bytes(result.file_id.identifier)


def _delete_windows_handle(handle: int) -> bool:
    """Mark the exact open file object for deletion without resolving its path."""
    import ctypes

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("delete_file", ctypes.c_ubyte)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    disposition = FileDispositionInfo(1)
    return bool(
        kernel32.SetFileInformationByHandle(
            ctypes.c_void_p(handle),
            4,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        )
    )


def _delete_windows_path_if_file_id(
    path: Path,
    expected_file_id: tuple[int, bytes],
) -> None:
    """Best-effort deletion of only the pathname entry owned by one FILE_ID."""
    import ctypes

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [("file_attributes", ctypes.c_uint32), ("reparse_tag", ctypes.c_uint32)]

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("delete_file", ctypes.c_ubyte)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.restype = ctypes.c_void_p
    get_handle_info = kernel32.GetFileInformationByHandleEx
    get_final_path = kernel32.GetFinalPathNameByHandleW
    set_handle_info = kernel32.SetFileInformationByHandle
    invalid = ctypes.c_void_p(-1).value
    handle = create_file(
        str(path),
        0x80000000 | 0x00010000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x00200000,
        None,
    )
    if handle == invalid:
        return
    try:
        attributes = FileAttributeTagInfo()
        buffer = ctypes.create_unicode_buffer(32768)
        if (
            not get_handle_info(
                ctypes.c_void_p(handle),
                9,
                ctypes.byref(attributes),
                ctypes.sizeof(attributes),
            )
            or attributes.file_attributes & 0x00000400
            or not get_final_path(ctypes.c_void_p(handle), buffer, len(buffer), 0)
            or _normalized_windows_path(buffer.value)
            != _normalized_windows_path(str(path))
            or _windows_file_id(int(handle)) != expected_file_id
        ):
            return
        disposition = FileDispositionInfo(1)
        set_handle_info(
            ctypes.c_void_p(handle),
            4,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        )
    except OSError:
        return
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def _open_windows_confined(
    data_root: Path,
    relative_path: SafeRelativePath,
    *,
    allow_target_delete_share: bool = False,
    allow_target_write_share: bool = False,
) -> tuple[int, list[int], tuple[str, ...], tuple[int, bytes]]:
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
            str(data_root),
            0x80000000,
            0x00000001 | 0x00000002,
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
        final_paths = [root_final]
        candidate = data_root
        for index, part in enumerate(relative_path.split("/")):
            candidate = candidate / part
            final = index == len(relative_path.split("/")) - 1
            share_mode = 0x00000001
            if not final or allow_target_write_share:
                share_mode |= 0x00000002
            if final and allow_target_delete_share:
                share_mode |= 0x00000004
            handle = create_file(
                str(candidate),
                0x80000000,
                share_mode,
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
            final_paths.append(final_path)
        handle = handles[-1]
        try:
            descriptor = msvcrt.open_osfhandle(
                handle, os.O_RDONLY | getattr(os, "O_BINARY", 0)
            )
        except OSError:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
            handles.pop()
            raise
        handles.pop()
        try:
            file_id = _windows_file_id(msvcrt.get_osfhandle(descriptor))
        except OSError:
            os.close(descriptor)
            raise
        ancestors = handles
        for ancestor in reversed(ancestors):
            kernel32.CloseHandle(ctypes.c_void_p(ancestor))
        handles = []
        return descriptor, [], (final_paths[-1],), file_id
    except OSError:
        raise ValueError("freeze manifest is invalid") from None
    finally:
        for handle in reversed(handles):
            kernel32.CloseHandle(ctypes.c_void_p(handle))


def _verify_windows_artifact_identity(artifact: BoundArtifact) -> None:
    import msvcrt

    if len(artifact.windows_final_paths) != 1 or artifact.windows_file_id is None:
        raise ValueError("freeze manifest is invalid")
    try:
        retained_id = _windows_file_id(msvcrt.get_osfhandle(artifact.descriptor))
        descriptor, ancestors, paths, current_id = _open_windows_confined(
            artifact.data_root,
            artifact.relative_path,
            allow_target_delete_share=artifact.replaceable_manifest,
        )
    except OSError:
        raise ValueError("freeze manifest is invalid") from None
    try:
        if (
            ancestors
            or paths != artifact.windows_final_paths
            or retained_id != artifact.windows_file_id
            or current_id != retained_id
        ):
            raise ValueError("freeze manifest is invalid")
    finally:
        os.close(descriptor)


@contextmanager
def open_confined_artifact(
    data_root: Path,
    relative_path: SafeRelativePath,
    *,
    replaceable_manifest: bool = False,
    allow_target_write_share: bool = False,
) -> Iterator[BoundArtifact]:
    """Open each path component safely and keep the evidence descriptor alive."""
    try:
        normalized = validate_safe_relative_path(relative_path)
        validate_stable_path(data_root.joinpath(*normalized.split("/")))
        root = _validated_lexical_root(data_root)
        if replaceable_manifest and normalized != "manifest.json":
            raise ValueError("freeze manifest is invalid")
        path_identities: tuple[tuple[int, int], ...] = ()
        windows_final_paths: tuple[str, ...] = ()
        windows_file_id: tuple[int, bytes] | None = None
        if os.name == "nt":
            descriptor, ancestors, windows_final_paths, windows_file_id = _open_windows_confined(
                root,
                normalized,
                allow_target_delete_share=replaceable_manifest,
                allow_target_write_share=allow_target_write_share,
            )
        else:
            descriptor, ancestors, path_identities = _open_posix_confined(
                root, normalized
            )
    except (OSError, RuntimeError, ValueError):
        raise ValueError("freeze manifest is invalid") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("freeze manifest is invalid")
        content = _read_descriptor(descriptor)
        artifact = BoundArtifact(
            data_root=root,
            relative_path=normalized,
            descriptor=descriptor,
            content=content,
            sha256=sha256_bytes(content),
            device=metadata.st_dev,
            inode=metadata.st_ino,
            ancestor_descriptors=ancestors,
            path_identities=path_identities,
            windows_final_paths=windows_final_paths,
            windows_file_id=windows_file_id,
            replaceable_manifest=replaceable_manifest,
        )
    except (OSError, RuntimeError, ValueError):
        for ancestor in reversed(ancestors):
            if os.name == "nt":
                import ctypes

                ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(
                    ctypes.c_void_p(ancestor)
                )
            else:
                os.close(ancestor)
        os.close(descriptor)
        raise ValueError("freeze manifest is invalid") from None
    try:
        yield artifact
    finally:
        artifact.close()


def read_confined_bytes(data_root: Path, relative_path: SafeRelativePath) -> bytes:
    with open_confined_artifact(data_root, relative_path) as artifact:
        return artifact.content


def _open_posix_report_parent(data_root: Path, relative_path: SafeRelativePath) -> tuple[int, str]:
    parts = relative_path.split("/")
    root = data_root
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
        validate_stable_path(data_root.joinpath(*normalized.split("/")))
        root = _validated_lexical_root(data_root)
    except ValueError:
        raise ValueError("freeze manifest is invalid") from None
    if os.name == "nt":
        import ctypes
        import msvcrt

        class FileAttributeTagInfo(ctypes.Structure):
            _fields_ = [("file_attributes", ctypes.c_uint32), ("reparse_tag", ctypes.c_uint32)]

        target = root.joinpath(*normalized.split("/"))
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.restype = ctypes.c_void_p
        get_handle_info = kernel32.GetFileInformationByHandleEx
        get_final_path = kernel32.GetFinalPathNameByHandleW
        invalid = ctypes.c_void_p(-1).value
        root_handle = create_file(
            str(root),
            0x80000000,
            0x00000001 | 0x00000002,
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
        handles = [int(root_handle)]
        temporary_path: Path | None = None
        retained_temp_id: tuple[int, bytes] | None = None
        try:
            parent = root
            for part in normalized.split("/")[:-1]:
                candidate = parent / part
                try:
                    os.mkdir(candidate)
                except FileExistsError:
                    pass
                handle = create_file(
                    str(candidate),
                    0x80000000,
                    0x00000001 | 0x00000002,
                    None,
                    3,
                    0x02000000 | 0x00200000,
                    None,
                )
                if handle == invalid:
                    raise ValueError("freeze manifest is invalid")
                handles.append(int(handle))
                info = FileAttributeTagInfo()
                if not get_handle_info(
                    ctypes.c_void_p(handle), 9, ctypes.byref(info), ctypes.sizeof(info)
                ) or info.file_attributes & 0x00000400 or not get_final_path(
                    ctypes.c_void_p(handle), buffer, len(buffer), 0
                ):
                    raise ValueError("freeze manifest is invalid")
                final_path = _normalized_windows_path(buffer.value)
                if final_path != root_final and not final_path.startswith(root_final + "\\"):
                    raise ValueError("freeze manifest is invalid")
                parent = candidate
            temporary_path = target.with_name(
                f".{target.name}.{secrets.token_hex(16)}.tmp"
            )
            handle = create_file(
                str(temporary_path),
                0x40000000 | 0x00010000,
                0x00000001 | 0x00000002 | 0x00000004,
                None,
                1,
                0x00200000,
                None,
            )
            if handle == invalid:
                raise ValueError("freeze manifest is invalid")
            descriptor: int | None = None
            try:
                retained_temp_id = _windows_file_id(int(handle))
                descriptor = msvcrt.open_osfhandle(
                    int(handle), os.O_WRONLY | getattr(os, "O_BINARY", 0)
                )
                handle = invalid
                _write_descriptor(descriptor, content)
                temporary_relative = temporary_path.relative_to(root).as_posix()
                (
                    current_temp_descriptor,
                    current_temp_ancestors,
                    _,
                    current_temp_id,
                ) = _open_windows_confined(
                    root,
                    temporary_relative,
                    allow_target_delete_share=True,
                    allow_target_write_share=True,
                )
                try:
                    if (
                        current_temp_ancestors
                        or current_temp_id != retained_temp_id
                    ):
                        raise ValueError("freeze manifest is invalid")
                finally:
                    os.close(current_temp_descriptor)
                create_hard_link = kernel32.CreateHardLinkW
                if not create_hard_link(str(target), str(temporary_path), None):
                    try:
                        with open_confined_artifact(root, normalized) as artifact:
                            if artifact.content == content:
                                return "matched"
                    except ValueError:
                        pass
                    raise FileExistsError("refusing to overwrite frozen file") from None
                with open_confined_artifact(
                    root, normalized, allow_target_write_share=True
                ) as artifact:
                    if (
                        artifact.content != content
                        or artifact.windows_file_id != retained_temp_id
                    ):
                        raise ValueError("freeze manifest is invalid")
                return "created"
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                elif handle != invalid:
                    _delete_windows_handle(int(handle))
                    kernel32.CloseHandle(ctypes.c_void_p(handle))
        finally:
            try:
                if temporary_path is not None and retained_temp_id is not None:
                    _delete_windows_path_if_file_id(
                        temporary_path,
                        retained_temp_id,
                    )
            finally:
                for open_handle in reversed(handles):
                    kernel32.CloseHandle(ctypes.c_void_p(open_handle))
    try:
        parent_descriptor, filename = _open_posix_report_parent(root, normalized)
    except (OSError, RuntimeError):
        raise ValueError("freeze manifest is invalid") from None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        temporary_name = f".{filename}.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_descriptor)
        except FileExistsError:  # pragma: no cover - a random retry is impractical to force
            raise RuntimeError("freeze manifest is invalid") from None
        temporary_identity: os.stat_result | None = None
        try:
            temporary_identity = os.fstat(descriptor)
            status: Literal["created", "matched"]
            try:
                _write_descriptor(descriptor, content)
                observed = os.stat(
                    temporary_name, dir_fd=parent_descriptor, follow_symlinks=False
                )
                if (observed.st_dev, observed.st_ino) != (
                    temporary_identity.st_dev,
                    temporary_identity.st_ino,
                ):
                    raise ValueError("freeze manifest is invalid")
                try:
                    os.link(
                        temporary_name,
                        filename,
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    with open_confined_artifact(root, normalized) as artifact:
                        if artifact.content == content:
                            status = "matched"
                        else:
                            raise FileExistsError("refusing to overwrite frozen file")
                else:
                    final_identity = os.stat(
                        filename, dir_fd=parent_descriptor, follow_symlinks=False
                    )
                    if (final_identity.st_dev, final_identity.st_ino) != (
                        temporary_identity.st_dev,
                        temporary_identity.st_ino,
                    ):
                        raise ValueError("freeze manifest is invalid")
                    status = "created"
            finally:
                try:
                    os.close(descriptor)
                finally:
                    try:
                        observed = os.stat(
                            temporary_name,
                            dir_fd=parent_descriptor,
                            follow_symlinks=False,
                        )
                        if (observed.st_dev, observed.st_ino) == (
                            temporary_identity.st_dev,
                            temporary_identity.st_ino,
                        ):
                            os.unlink(temporary_name, dir_fd=parent_descriptor)
                    except FileNotFoundError:
                        pass
        finally:
            try:
                if temporary_identity is None:
                    os.close(descriptor)
            finally:
                os.fsync(parent_descriptor)
        return status
    finally:
        os.close(parent_descriptor)


@contextmanager
def publish_confined_artifact_no_overwrite(
    data_root: Path,
    relative_path: SafeRelativePath,
    content: bytes,
) -> Iterator[tuple[Literal["created", "matched"], BoundArtifact]]:
    """Publish a report and retain its final pathname-bound evidence descriptor."""
    status = publish_confined_bytes_no_overwrite(data_root, relative_path, content)
    with open_confined_artifact(data_root, relative_path) as artifact:
        if artifact.content != content:
            raise ValueError("freeze manifest is invalid")
        yield status, artifact


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


def _validate_v2_bindings(
    data_root: Path,
    manifest: FreezeManifestV2,
    *,
    stack: ExitStack,
    manifest_artifact: BoundArtifact,
) -> ValidatedFreezeEvidence:
    reasons: list[FreezeEvidenceReason] = []

    def add_reason(reason: FreezeEvidenceReason) -> None:
        if reason not in reasons:
            reasons.append(reason)

    report_artifact: BoundArtifact | None = None
    report: FreezeApprovalReportV2 | None = None
    try:
        report_artifact = stack.enter_context(
            open_confined_artifact(data_root, manifest.approval.report_path)
        )
    except ValueError:
        add_reason("approval_invalid")
    else:
        if report_artifact.sha256 != manifest.approval.report_sha256:
            add_reason("approval_invalid")
        else:
            try:
                report = _validated_report_bytes(report_artifact.content)
            except ValueError:
                add_reason("approval_invalid")
            else:
                if (
                    report.approved_at != manifest.approval.approved_at
                    or report.approver_ref != manifest.approval.approver_ref
                    or report.identifier_map_sha256 != manifest.identifier_map.sha256
                ):
                    add_reason("approval_invalid")
    partition_hashes: dict[str, Sha256] = {}
    partition_artifacts: list[BoundArtifact] = []
    partition_artifact_names: list[Literal["dev", "validation"]] = []
    partition_rows: list[
        tuple[Literal["dev", "validation"], tuple[dict[str, object], ...]]
    ] = []
    for partition in manifest.partitions:
        try:
            artifact = stack.enter_context(
                open_confined_artifact(data_root, partition.path)
            )
        except ValueError:
            add_reason("partition_hash_mismatch")
            continue
        partition_artifacts.append(artifact)
        partition_artifact_names.append(partition.name)
        if artifact.sha256 != partition.sha256:
            add_reason("partition_hash_mismatch")
            continue
        try:
            lines = artifact.content.splitlines()
            if not lines or any(not line for line in lines):
                raise ValueError("partition rows are invalid")
            rows = [_json_object(line) for line in lines]
            query_ids = [row.get("query_id") for row in rows]
        except ValueError:
            add_reason("partition_count_mismatch")
            continue
        partition_hashes[partition.name] = artifact.sha256
        if (
            len(rows) != partition.query_count
            or not all(
                isinstance(query_id, str) and query_id.strip() for query_id in query_ids
            )
            or len(set(query_ids)) != len(query_ids)
        ):
            add_reason("partition_count_mismatch")
        partition_rows.append((partition.name, tuple(rows)))
    identifier_artifact: BoundArtifact | None = None
    try:
        identifier_artifact = stack.enter_context(
            open_confined_artifact(data_root, manifest.identifier_map.path)
        )
    except ValueError:
        add_reason("identifier_map_missing")
    else:
        if identifier_artifact.sha256 != manifest.identifier_map.sha256:
            add_reason("identifier_map_hash_mismatch")
        else:
            try:
                value = _json_object(identifier_artifact.content)
            except ValueError:
                add_reason("identifier_map_coverage_failed")
            else:
                if (
                    len(value) != manifest.identifier_map.entry_count
                    or not value
                    or not all(
                        isinstance(key, str)
                        and key.strip()
                        and isinstance(target, str)
                        and target.strip()
                        for key, target in value.items()
                    )
                ):
                    add_reason("identifier_map_coverage_failed")
    if report is not None and len(partition_hashes) == len(manifest.partitions):
        if report.partition_hashes != partition_hashes:
            add_reason("approval_invalid")
    if len(partition_hashes) == len(manifest.partitions) and (
        manifest.gold_sha256
        != canonical_gold_set_sha256(
            partition_hashes["dev"], partition_hashes["validation"]
        )
    ):
        add_reason("manifest_invalid")
    return ValidatedFreezeEvidence(
        manifest=manifest,
        manifest_artifact=manifest_artifact,
        approval_artifact=report_artifact,
        partition_artifacts=tuple(partition_artifacts),
        partition_artifact_names=tuple(partition_artifact_names),
        identifier_map_artifact=identifier_artifact,
        partition_rows=tuple(partition_rows),
        reasons=tuple(reasons),
    )


@contextmanager
def open_validated_freeze_evidence(
    path: Path,
    *,
    data_root: Path,
) -> Iterator[ValidatedFreezeEvidence]:
    """Validate one freeze and retain every exact evidence descriptor through exit."""
    try:
        root = _validated_lexical_root(data_root)
        relative_path = path.absolute().relative_to(root).as_posix()
    except (OSError, RuntimeError, ValueError):
        raise FreezeEvidenceError("manifest_invalid") from None
    with ExitStack() as stack:
        try:
            artifact = stack.enter_context(
                open_confined_artifact(root, relative_path)
            )
        except ValueError:
            raise FreezeEvidenceError("manifest_invalid") from None
        try:
            manifest = parse_freeze_manifest_bytes(artifact.content)
        except ValueError:
            evidence = ValidatedFreezeEvidence(
                manifest=None,
                manifest_artifact=artifact,
                reasons=("manifest_invalid",),
            )
        else:
            if isinstance(manifest, FreezeManifestV2):
                evidence = _validate_v2_bindings(
                    root,
                    manifest,
                    stack=stack,
                    manifest_artifact=artifact,
                )
            else:
                evidence = ValidatedFreezeEvidence(
                    manifest=manifest,
                    manifest_artifact=artifact,
                )
        try:
            yield evidence
        finally:
            identity_failures = list(evidence.reasons)
            for bound_artifact, reason in evidence.artifact_identity_checks:
                try:
                    bound_artifact.verify_path_identity()
                except (OSError, RuntimeError, ValueError):
                    identity_failures.append(reason)
            if identity_failures:
                raise FreezeEvidenceError(
                    identity_failures[0],
                    *identity_failures[1:],
                ) from None


def load_freeze_manifest(path: Path, *, data_root: Path) -> FreezeManifest:
    with open_validated_freeze_evidence(path, data_root=data_root) as evidence:
        if evidence.reasons or evidence.manifest is None:
            raise FreezeEvidenceError(
                *(evidence.reasons or ("manifest_invalid",)),
            )
        return evidence.manifest
