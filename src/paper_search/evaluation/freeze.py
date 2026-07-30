from __future__ import annotations

import argparse

import errno
import hashlib
import json
import os
import secrets
import stat
import sys
import tempfile
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, NoReturn, TypeVar, cast

from pydantic import BaseModel, PositiveInt, ValidationError

from paper_search.domain.models import DomainModel, NonEmptyStr, SafeRelativePath, Sha256
from paper_search.evaluation.annotation import (
    AgreementReport,
    AnnotationRecord,
    TypeDomainAnnotationRecord,
    compare_annotations,
)
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    write_frozen_bytes,
)
from paper_search.evaluation.freeze_schema import (
    FreezeApprovalBindingV2,
    FreezeApprovalReportV2,
    FreezeManifestV1,
    FreezeManifestV2,
    FrozenPartitionV2,
    IdentifierMapBindingV2,
    LegacyPartitionV1,
    BoundArtifact,
    canonical_gold_set_sha256,
    open_confined_artifact,
    parse_json_object_bytes,
    parse_freeze_manifest_bytes,
    publish_confined_artifact_no_overwrite,
    sha256_bytes,
)


PASA_REPO_ID = "CarlanLark/pasa-dataset"
PASA_REVISION = "232428b0c867268c3b8ded90db4d98c1b30501d6"
RANDOM_SEED = 20260714
SAMPLING_ALGORITHM = "answer-count-largest-remainder-v1"
PREPARED_STATUS = "waiting_for_human_label_freeze"
AGREEMENT_THRESHOLD = 0.80


@dataclass(frozen=True)
class FreezeExpectations:
    """Exact production data shape required before approval."""

    source_row_counts: tuple[tuple[str, int], ...]
    partition_counts: tuple[tuple[str, int], ...]
    work_package_counts: tuple[tuple[str, int], ...]


OFFICIAL_EXPECTATIONS = FreezeExpectations(
    source_row_counts=(
        ("AutoScholarQuery/dev.jsonl", 1000),
        ("AutoScholarQuery/test.jsonl", 1000),
        ("RealScholarQuery/test.jsonl", 50),
    ),
    partition_counts=(
        ("dev", 60),
        ("validation", 30),
        ("simulated_test", 50),
    ),
    work_package_counts=(
        ("type_domain", 90),
        ("constraints", 40),
        ("overlap", 20),
    ),
)


def _expectation_map(pairs: tuple[tuple[str, int], ...]) -> dict[str, int]:
    expected: dict[str, int] = {}
    for name, count in pairs:
        if not name.strip() or count <= 0 or name in expected:
            raise ValueError("prepared data is invalid")
        expected[name] = count
    if not expected:
        raise ValueError("prepared data is invalid")
    return expected


ZeroAnswerPolicy = Literal["reject", "allow"]


def parse_zero_answer_policies(
    values: Sequence[str],
    partition_names: Collection[str],
) -> dict[str, ZeroAnswerPolicy]:
    """Parse exactly one explicit zero-answer policy for every partition."""
    expected = set(partition_names)
    if not expected or any(not name.strip() for name in expected):
        raise ValueError("zero-answer policies are invalid")
    parsed: dict[str, ZeroAnswerPolicy] = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError("zero-answer policies are invalid")
        name, policy = value.split("=", maxsplit=1)
        if not name or name not in expected or name in parsed or policy not in ("reject", "allow"):
            raise ValueError("zero-answer policies are invalid")
        parsed[name] = cast(ZeroAnswerPolicy, policy)
    if set(parsed) != expected:
        raise ValueError("zero-answer policies are invalid")
    return parsed


class PartitionFreezeAudit(DomainModel):
    """Verified identity and policy for one frozen evaluation partition."""

    count: PositiveInt
    gold_path: NonEmptyStr
    gold_sha256: NonEmptyStr
    ids_path: NonEmptyStr
    ids_sha256: NonEmptyStr
    zero_answer_policy: ZeroAnswerPolicy
    labels_complete: Literal[True]


class FreezeAuditReport(DomainModel):
    """Content-safe summary of a Task 2 freeze candidate."""

    prepared_manifest_sha256: NonEmptyStr
    dataset_revision: NonEmptyStr
    source_file_count: PositiveInt
    type_domain_count: PositiveInt
    type_domain_sha256: NonEmptyStr
    constraint_count: PositiveInt
    constraint_sha256: NonEmptyStr
    overlap_count: PositiveInt
    overlap_sha256: NonEmptyStr
    agreement: AgreementReport
    partitions: dict[str, PartitionFreezeAudit]
    approval_requested: bool


@dataclass(frozen=True)
class _EvidenceIdentity:
    path: Path
    sha256: str


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class _FileState:
    identity: _FileIdentity
    regular: bool


@dataclass(frozen=True)
class FreezeAuditResult:
    """Pure audit output; it contains no approved report path or final bytes."""

    prepared_manifest_bytes: bytes
    frozen_manifest_payload: dict[str, object]
    report: FreezeAuditReport
    evidence: tuple[_EvidenceIdentity, ...] = ()


@dataclass(frozen=True)
class FreezeApprovalPlan:
    """Exact report and manifest bytes authorized for one approval attempt."""

    prepared_manifest_bytes: bytes
    frozen_manifest_bytes: bytes
    report_bytes: bytes
    report: FreezeAuditReport
    evidence: tuple[_EvidenceIdentity, ...]


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + chr(10)
    ).encode("utf-8")


def _load_json_bytes(content: bytes) -> object:
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("prepared data is invalid") from None


ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_jsonl_evidence(
    path: Path,
    model_type: type[ModelT],
    *,
    private: bool = False,
) -> tuple[list[ModelT], bytes, _EvidenceIdentity]:
    error_message = "private annotations are invalid" if private else "prepared data is invalid"
    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        raise ValueError(error_message) from None
    records: list[ModelT] = []
    seen_query_ids: set[str] = set()
    try:
        for line in text.splitlines():
            if not line.strip():
                raise ValueError
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError
            record = model_type.model_validate(payload)
            query_id = getattr(record, "query_id", None)
            if isinstance(query_id, str):
                if query_id in seen_query_ids:
                    raise ValueError
                seen_query_ids.add(query_id)
            records.append(record)
    except (json.JSONDecodeError, ValidationError, ValueError):
        raise ValueError(error_message) from None
    identity = _EvidenceIdentity(path=path.resolve(), sha256=_sha256_bytes(content))
    return records, content, identity


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("prepared data is invalid")
    return cast(dict[str, object], value)


def _nonempty_string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("prepared data is invalid")
    return value


def _positive_integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("prepared data is invalid")
    return value


def _confined_path(data_root: Path, relative_path: object) -> tuple[str, Path]:
    value = _nonempty_string(relative_path)
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("prepared data is invalid")
    root = data_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("prepared data is invalid") from None
    return value, resolved


def _read_ids(
    data_root: Path,
    path_value: object,
) -> tuple[str, list[str], bytes, _EvidenceIdentity]:
    relative_path, path = _confined_path(data_root, path_value)
    try:
        content = path.read_bytes()
        payload = _load_json_bytes(content)
    except OSError:
        raise ValueError("prepared data is invalid") from None
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(item, str) and item.strip() for item in payload)
    ):
        raise ValueError("prepared data is invalid")
    identifiers = cast(list[str], payload)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("prepared data is invalid")
    identity = _EvidenceIdentity(path=path.resolve(), sha256=_sha256_bytes(content))
    return relative_path, identifiers, content, identity


def _validate_source_files(
    data_root: Path,
    value: object,
    expected_row_counts: Mapping[str, int],
) -> tuple[int, list[_EvidenceIdentity]]:
    if not isinstance(value, list) or len(value) != len(expected_row_counts):
        raise ValueError("prepared data is invalid")
    seen_paths: set[str] = set()
    evidence: list[_EvidenceIdentity] = []
    allowed_fields = {"path", "raw_path", "row_count", "byte_count", "sha256"}
    for raw_item in value:
        item = _mapping(raw_item)
        if set(item) != allowed_fields:
            raise ValueError("prepared data is invalid")
        source_path = _nonempty_string(item.get("path"))
        if source_path in seen_paths or source_path not in expected_row_counts:
            raise ValueError("prepared data is invalid")
        seen_paths.add(source_path)
        raw_relative, path = _confined_path(data_root, item.get("raw_path"))
        expected_rows = expected_row_counts[source_path]
        if raw_relative != f"raw/{source_path}" or item.get("row_count") != expected_rows:
            raise ValueError("prepared data is invalid")
        expected_bytes = _positive_integer(item.get("byte_count"))
        expected_hash = _nonempty_string(item.get("sha256"))
        try:
            content = path.read_bytes()
        except OSError:
            raise ValueError("prepared data is invalid") from None
        content_hash = _sha256_bytes(content)
        row_count = len(content.splitlines())
        if (
            len(content) != expected_bytes
            or row_count != expected_rows
            or content_hash != expected_hash
        ):
            raise ValueError("prepared data is invalid")
        evidence.append(_EvidenceIdentity(path=path.resolve(), sha256=content_hash))
    if seen_paths != set(expected_row_counts):
        raise ValueError("prepared data is invalid")
    return len(value), evidence


def _validate_partitions(
    data_root: Path,
    value: object,
    policies: Mapping[str, ZeroAnswerPolicy],
    expected_counts: Mapping[str, int],
) -> tuple[
    dict[str, PartitionFreezeAudit],
    dict[str, dict[str, object]],
    dict[str, list[str]],
    list[_EvidenceIdentity],
]:
    partitions = _mapping(value)
    if set(partitions) != set(expected_counts) or set(policies) != set(partitions):
        raise ValueError("prepared data is invalid")
    audits: dict[str, PartitionFreezeAudit] = {}
    frozen: dict[str, dict[str, object]] = {}
    ids_by_partition: dict[str, list[str]] = {}
    evidence: list[_EvidenceIdentity] = []
    allowed_fields = {"count", "gold_path", "gold_sha256", "ids_path", "ids_sha256"}
    for name, raw_partition in partitions.items():
        partition = _mapping(raw_partition)
        if not set(partition).issubset(allowed_fields) or not {
            "count",
            "gold_path",
            "ids_path",
            "ids_sha256",
        }.issubset(partition):
            raise ValueError("prepared data is invalid")
        count = _positive_integer(partition.get("count"))
        if count != expected_counts[name]:
            raise ValueError("prepared data is invalid")
        policy = policies[name]
        if policy not in ("reject", "allow"):
            raise ValueError("prepared data is invalid")
        gold_relative, gold_path = _confined_path(data_root, partition.get("gold_path"))
        ids_relative, identifiers, ids_content, ids_identity = _read_ids(
            data_root, partition.get("ids_path")
        )
        records, gold_content, gold_identity = _read_jsonl_evidence(
            gold_path,
            EvaluationQuery,
        )
        if not records or len(records) != count or len(identifiers) != count:
            raise ValueError("prepared data is invalid")
        if [record.query_id for record in records] != identifiers:
            raise ValueError("prepared data is invalid")
        if policy == "reject" and any(not record.relevant_paper_ids for record in records):
            raise ValueError("prepared data is invalid")
        gold_hash = _sha256_bytes(gold_content)
        ids_hash = _sha256_bytes(ids_content)
        declared_gold_hash = partition.get("gold_sha256")
        if declared_gold_hash is not None and declared_gold_hash != gold_hash:
            raise ValueError("prepared data is invalid")
        if partition.get("ids_sha256") != ids_hash:
            raise ValueError("prepared data is invalid")
        audit = PartitionFreezeAudit(
            count=count,
            gold_path=gold_relative,
            gold_sha256=gold_hash,
            ids_path=ids_relative,
            ids_sha256=ids_hash,
            zero_answer_policy=policy,
            labels_complete=True,
        )
        audits[name] = audit
        frozen[name] = audit.model_dump(mode="json")
        ids_by_partition[name] = identifiers
        evidence.extend((gold_identity, ids_identity))
    return audits, frozen, ids_by_partition, evidence


def _validate_work_packages(
    data_root: Path,
    value: object,
    partition_ids: Mapping[str, list[str]],
    expected_counts: Mapping[str, int],
) -> tuple[list[str], list[str], list[str], list[_EvidenceIdentity]]:
    packages = _mapping(value)
    if set(packages) != set(expected_counts):
        raise ValueError("prepared data is invalid")

    package_ids: dict[str, list[str]] = {}
    evidence: list[_EvidenceIdentity] = []
    for name, raw_package in packages.items():
        package = _mapping(raw_package)
        expected_fields = (
            {"count", "ids_path", "ids_sha256"}
            if name == "overlap"
            else {"count", "source_path", "source_sha256", "ids_path", "ids_sha256"}
        )
        if set(package) != expected_fields:
            raise ValueError("prepared data is invalid")
        count = _positive_integer(package.get("count"))
        if count != expected_counts[name]:
            raise ValueError("prepared data is invalid")
        _, identifiers, ids_content, ids_identity = _read_ids(data_root, package.get("ids_path"))
        if len(identifiers) != count or package.get("ids_sha256") != _sha256_bytes(ids_content):
            raise ValueError("prepared data is invalid")
        evidence.append(ids_identity)
        if name != "overlap":
            _, source_path = _confined_path(data_root, package.get("source_path"))
            try:
                source_content = source_path.read_bytes()
            except OSError:
                raise ValueError("prepared data is invalid") from None
            source_hash = _sha256_bytes(source_content)
            if package.get("source_sha256") != source_hash:
                raise ValueError("prepared data is invalid")
            if len(source_content.splitlines()) != count:
                raise ValueError("prepared data is invalid")
            evidence.append(_EvidenceIdentity(path=source_path.resolve(), sha256=source_hash))
        package_ids[name] = identifiers

    required_partitions = {"dev", "validation"}
    if not required_partitions.issubset(partition_ids):
        raise ValueError("prepared data is invalid")
    type_domain = package_ids["type_domain"]
    constraints = package_ids["constraints"]
    overlap = package_ids["overlap"]
    if not set(overlap).issubset(constraints):
        raise ValueError("prepared data is invalid")
    if not set(constraints).issubset(partition_ids["dev"]):
        raise ValueError("prepared data is invalid")
    if set(type_domain) != set(partition_ids["dev"]) | set(partition_ids["validation"]):
        raise ValueError("prepared data is invalid")
    return type_domain, constraints, overlap, evidence


def _private_records(
    type_domain_labels_path: Path,
    constraint_labels_path: Path,
    overlap_labels_path: Path,
    type_domain_ids: list[str],
    constraint_ids: list[str],
    overlap_ids: list[str],
) -> tuple[
    list[TypeDomainAnnotationRecord],
    list[AnnotationRecord],
    list[AnnotationRecord],
    AgreementReport,
    bytes,
    bytes,
    bytes,
    list[_EvidenceIdentity],
]:
    type_domain, type_domain_bytes, type_domain_identity = _read_jsonl_evidence(
        type_domain_labels_path,
        TypeDomainAnnotationRecord,
        private=True,
    )
    constraints, constraint_bytes, constraint_identity = _read_jsonl_evidence(
        constraint_labels_path,
        AnnotationRecord,
        private=True,
    )
    overlap, overlap_bytes, overlap_identity = _read_jsonl_evidence(
        overlap_labels_path,
        AnnotationRecord,
        private=True,
    )
    if (
        {record.query_id for record in type_domain} != set(type_domain_ids)
        or len(type_domain) != len(type_domain_ids)
        or {record.query_id for record in constraints} != set(constraint_ids)
        or len(constraints) != len(constraint_ids)
        or {record.query_id for record in overlap} != set(overlap_ids)
        or len(overlap) != len(overlap_ids)
    ):
        raise ValueError("private annotations are invalid")
    type_domain_by_id = {record.query_id: record for record in type_domain}
    if any(
        record.query_type != type_domain_by_id[record.query_id].query_type
        or record.domain != type_domain_by_id[record.query_id].domain
        for record in constraints
    ):
        raise ValueError("private annotations are invalid")
    annotator_sets = (
        {record.annotator for record in type_domain},
        {record.annotator for record in constraints},
        {record.annotator for record in overlap},
    )
    if any(len(annotators) != 1 for annotators in annotator_sets):
        raise ValueError("private annotations are invalid")
    type_domain_annotator = next(iter(annotator_sets[0]))
    constraint_annotator = next(iter(annotator_sets[1]))
    overlap_annotator = next(iter(annotator_sets[2]))
    if (
        type_domain_annotator != constraint_annotator
        or overlap_annotator == constraint_annotator
    ):
        raise ValueError("private annotations are invalid")
    overlap_set = set(overlap_ids)
    first_rater = [record for record in constraints if record.query_id in overlap_set]
    first_by_id = {record.query_id: record for record in first_rater}
    if any(first_by_id[record.query_id].annotator == record.annotator for record in overlap):
        raise ValueError("private annotations are invalid")
    agreement = compare_annotations(
        first_rater,
        overlap,
        fields=("query_type", "domain"),
    )
    if any(
        field.threshold != AGREEMENT_THRESHOLD
        or field.kappa < AGREEMENT_THRESHOLD
        or not field.accepted
        for field in agreement.fields.values()
    ):
        raise ValueError("human annotation agreement is below threshold")
    return (
        type_domain,
        constraints,
        overlap,
        agreement,
        type_domain_bytes,
        constraint_bytes,
        overlap_bytes,
        [type_domain_identity, constraint_identity, overlap_identity],
    )


def audit_freeze_candidate(
    *,
    data_root: Path,
    type_domain_labels_path: Path,
    constraint_labels_path: Path,
    overlap_labels_path: Path,
    policies: Mapping[str, ZeroAnswerPolicy],
) -> FreezeAuditResult:
    """Audit a prepared Task 2 tree without writing any file."""
    manifest_path = data_root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError:
        raise ValueError("prepared data is invalid") from None
    manifest = _mapping(_load_json_bytes(manifest_bytes))
    expected_identity: dict[str, object] = {
        "repo_id": PASA_REPO_ID,
        "revision": PASA_REVISION,
        "license": "CC-BY-NC-SA-4.0",
        "access": "gated-hugging-face-dataset",
        "random_seed": RANDOM_SEED,
        "sampling_algorithm": SAMPLING_ALGORITHM,
        "status": PREPARED_STATUS,
        "work_package_sampling": "answer-count-largest-remainder-v1-seeded-offsets",
    }
    expected_fields = {
        *expected_identity,
        "source_files",
        "partitions",
        "work_packages",
    }
    if set(manifest) != expected_fields or any(
        manifest.get(key) != expected for key, expected in expected_identity.items()
    ):
        raise ValueError("prepared data is invalid")

    source_counts = _expectation_map(OFFICIAL_EXPECTATIONS.source_row_counts)
    partition_counts = _expectation_map(OFFICIAL_EXPECTATIONS.partition_counts)
    work_package_counts = _expectation_map(OFFICIAL_EXPECTATIONS.work_package_counts)
    source_file_count, source_evidence = _validate_source_files(
        data_root,
        manifest.get("source_files"),
        source_counts,
    )
    (
        partition_audits,
        frozen_partitions,
        partition_ids,
        partition_evidence,
    ) = _validate_partitions(
        data_root,
        manifest.get("partitions"),
        policies,
        partition_counts,
    )
    (
        type_domain_ids,
        constraint_ids,
        overlap_ids,
        package_evidence,
    ) = _validate_work_packages(
        data_root,
        manifest.get("work_packages"),
        partition_ids,
        work_package_counts,
    )
    (
        type_domain,
        constraints,
        overlap,
        agreement,
        type_domain_bytes,
        constraint_bytes,
        overlap_bytes,
        private_evidence,
    ) = _private_records(
        type_domain_labels_path,
        constraint_labels_path,
        overlap_labels_path,
        type_domain_ids,
        constraint_ids,
        overlap_ids,
    )

    prepared_manifest_hash = _sha256_bytes(manifest_bytes)
    report = FreezeAuditReport(
        prepared_manifest_sha256=prepared_manifest_hash,
        dataset_revision=PASA_REVISION,
        source_file_count=source_file_count,
        type_domain_count=len(type_domain),
        type_domain_sha256=_sha256_bytes(type_domain_bytes),
        constraint_count=len(constraints),
        constraint_sha256=_sha256_bytes(constraint_bytes),
        overlap_count=len(overlap),
        overlap_sha256=_sha256_bytes(overlap_bytes),
        agreement=agreement,
        partitions=partition_audits,
        approval_requested=False,
    )
    frozen_manifest: dict[str, object] = {
        "repo_id": PASA_REPO_ID,
        "revision": PASA_REVISION,
        "license": "CC-BY-NC-SA-4.0",
        "access": "gated-hugging-face-dataset",
        "random_seed": RANDOM_SEED,
        "sampling_algorithm": SAMPLING_ALGORITHM,
        "status": "frozen",
        "source_files": manifest["source_files"],
        "partitions": frozen_partitions,
        "work_package_sampling": ("answer-count-largest-remainder-v1-seeded-offsets"),
        "work_packages": manifest["work_packages"],
        "prepared_manifest_sha256": prepared_manifest_hash,
    }
    return FreezeAuditResult(
        prepared_manifest_bytes=manifest_bytes,
        frozen_manifest_payload=frozen_manifest,
        report=report,
        evidence=tuple(
            [
                *source_evidence,
                *partition_evidence,
                *package_evidence,
                *private_evidence,
            ]
        ),
    )


def _normalized_report_relative_path(value: str) -> str:
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or len(candidate.parts) < 2
        or candidate.parts[0] != "freeze_reports"
        or ".." in candidate.parts
    ):
        raise ValueError("freeze approval failed")
    return candidate.as_posix()


def build_approval_plan(
    audit: FreezeAuditResult,
    *,
    report_relative_path: str,
) -> FreezeApprovalPlan:
    """Bind an approved safe report to exact final frozen-manifest bytes."""
    relative_path = _normalized_report_relative_path(report_relative_path)
    report = audit.report.model_copy(update={"approval_requested": True})
    report_bytes = _json_bytes(report.model_dump(mode="json"))
    frozen_manifest = _mapping(_load_json_bytes(_json_bytes(audit.frozen_manifest_payload)))
    frozen_manifest["freeze_report_path"] = relative_path
    frozen_manifest["freeze_report_sha256"] = _sha256_bytes(report_bytes)
    return FreezeApprovalPlan(
        prepared_manifest_bytes=audit.prepared_manifest_bytes,
        frozen_manifest_bytes=_json_bytes(frozen_manifest),
        report_bytes=report_bytes,
        report=report,
        evidence=audit.evidence,
    )


def _validated_plan_report_path(
    data_root: Path,
    plan: FreezeApprovalPlan,
) -> Path:
    frozen_manifest = _mapping(_load_json_bytes(plan.frozen_manifest_bytes))
    if _json_bytes(frozen_manifest) != plan.frozen_manifest_bytes:
        raise RuntimeError("freeze approval failed")
    relative_path = _normalized_report_relative_path(
        _nonempty_string(frozen_manifest.get("freeze_report_path"))
    )
    if frozen_manifest.get("freeze_report_sha256") != _sha256_bytes(plan.report_bytes):
        raise RuntimeError("freeze approval failed")
    if _json_bytes(plan.report.model_dump(mode="json")) != plan.report_bytes:
        raise RuntimeError("freeze approval failed")
    if plan.report.approval_requested is not True:
        raise RuntimeError("freeze approval failed")
    if frozen_manifest.get("prepared_manifest_sha256") != _sha256_bytes(
        plan.prepared_manifest_bytes
    ):
        raise RuntimeError("freeze approval failed")
    report_path = data_root / relative_path
    if _confined_report_relative_path(data_root, report_path) != relative_path:
        raise RuntimeError("freeze approval failed")
    return report_path


def _evidence_matches(evidence: Sequence[_EvidenceIdentity]) -> bool:
    for identity in evidence:
        try:
            content = identity.path.read_bytes()
        except OSError:
            return False
        if _sha256_bytes(content) != identity.sha256:
            return False
    return True


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return f"sha256:{digest.hexdigest()}"


def _normalized_windows_path(value: str) -> str:
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.abspath(value))


@contextmanager
def _stable_evidence_files(
    evidence: Sequence[_EvidenceIdentity | BoundArtifact],
) -> Iterator[None]:
    bound = [item for item in evidence if isinstance(item, BoundArtifact)]
    pathname_evidence = [item for item in evidence if isinstance(item, _EvidenceIdentity)]
    if any(
        _sha256_descriptor(item.descriptor) != item.sha256
        for item in bound
    ):
        raise RuntimeError("freeze approval failed")
    for item in bound:
        try:
            item.verify_path_identity()
        except ValueError:
            raise RuntimeError("freeze approval failed") from None
    if not pathname_evidence:
        yield
        if any(
            _sha256_descriptor(item.descriptor) != item.sha256
            for item in bound
        ):
            raise RuntimeError("freeze approval failed")
        for item in bound:
            try:
                item.verify_path_identity()
            except ValueError:
                raise RuntimeError("freeze approval failed") from None
        return
    evidence = pathname_evidence
    descriptors: list[int] = []
    try:
        if os.name == "nt":
            import ctypes
            import msvcrt

            class FileAttributeTagInfo(ctypes.Structure):
                _fields_ = [
                    ("file_attributes", ctypes.c_uint32),
                    ("reparse_tag", ctypes.c_uint32),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_file = kernel32.CreateFileW
            create_file.restype = ctypes.c_void_p
            get_handle_info = kernel32.GetFileInformationByHandleEx
            get_final_path = kernel32.GetFinalPathNameByHandleW
            invalid_handle = ctypes.c_void_p(-1).value

            for identity in evidence:
                handle = create_file(
                    str(identity.path),
                    0x80000000,
                    0x00000001,
                    None,
                    3,
                    0x00200000,
                    None,
                )
                if handle == invalid_handle:
                    raise RuntimeError("freeze approval failed")
                info = FileAttributeTagInfo()
                if not get_handle_info(
                    ctypes.c_void_p(handle),
                    9,
                    ctypes.byref(info),
                    ctypes.sizeof(info),
                ):
                    kernel32.CloseHandle(ctypes.c_void_p(handle))
                    raise RuntimeError("freeze approval failed")
                if info.file_attributes & 0x00000400:
                    kernel32.CloseHandle(ctypes.c_void_p(handle))
                    raise RuntimeError("freeze approval failed")
                path_buffer = ctypes.create_unicode_buffer(32768)
                path_length = int(
                    get_final_path(
                        ctypes.c_void_p(handle),
                        path_buffer,
                        len(path_buffer),
                        0,
                    )
                )
                if (
                    path_length == 0
                    or path_length >= len(path_buffer)
                    or _normalized_windows_path(path_buffer.value)
                    != _normalized_windows_path(str(identity.path))
                ):
                    kernel32.CloseHandle(ctypes.c_void_p(handle))
                    raise RuntimeError("freeze approval failed")
                try:
                    descriptor = msvcrt.open_osfhandle(
                        int(handle),
                        os.O_RDONLY | getattr(os, "O_BINARY", 0),
                    )
                except OSError:
                    kernel32.CloseHandle(ctypes.c_void_p(handle))
                    raise RuntimeError("freeze approval failed") from None
                descriptors.append(descriptor)
        else:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            for identity in evidence:
                descriptor = os.open(identity.path, flags)
                descriptors.append(descriptor)
                proc_path = Path(f"/proc/self/fd/{descriptor}")
                if proc_path.exists() and proc_path.resolve() != identity.path:
                    raise RuntimeError("freeze approval failed")

        if any(
            _sha256_descriptor(descriptor) != identity.sha256
            for descriptor, identity in zip(descriptors, evidence, strict=True)
        ):
            raise RuntimeError("freeze approval failed")
        yield
        if any(_sha256_descriptor(item.descriptor) != item.sha256 for item in bound):
            raise RuntimeError("freeze approval failed")
        if any(
            _sha256_descriptor(descriptor) != identity.sha256
            for descriptor, identity in zip(descriptors, evidence, strict=True)
        ):
            raise RuntimeError("freeze approval failed")
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@contextmanager
def _stable_bound_artifact_hash(
    artifact: BoundArtifact | None,
) -> Iterator[None]:
    """Hold a replaceable artifact by descriptor without binding its old path."""
    if artifact is not None and (
        _sha256_descriptor(artifact.descriptor) != artifact.sha256
    ):
        raise RuntimeError("freeze approval failed")
    yield
    if artifact is not None and (
        _sha256_descriptor(artifact.descriptor) != artifact.sha256
    ):
        raise RuntimeError("freeze approval failed")


def _manifest_recovery_matches(
    backup_path: Path,
    *,
    expected_identity: _FileIdentity,
    expected_bytes: bytes,
    expected_sha256: str,
    retained_manifest: BoundArtifact | None,
) -> bool:
    try:
        if retained_manifest is not None and (
            _sha256_descriptor(retained_manifest.descriptor) != expected_sha256
        ):
            return False
        if not _state_matches_owned_file(
            _probe_file_state(backup_path),
            expected_identity,
        ):
            return False
        content = backup_path.read_bytes()
    except OSError:
        return False
    return content == expected_bytes and _sha256_bytes(content) == expected_sha256


def _canonical_manifest_document(content: bytes) -> dict[str, object]:
    try:
        payload = parse_json_object_bytes(content)
    except ValueError:
        raise RuntimeError("freeze approval failed") from None
    if _json_bytes(payload) != content:
        raise RuntimeError("freeze approval failed")
    return payload


def _canonical_freeze_manifest(content: bytes) -> FreezeManifestV1 | FreezeManifestV2:
    _canonical_manifest_document(content)
    try:
        manifest = parse_freeze_manifest_bytes(content)
    except ValueError:
        raise RuntimeError("freeze approval failed") from None
    return manifest


def _owned_manifest_matches(
    path: Path,
    *,
    expected_identity: _FileIdentity,
    expected_bytes: bytes,
    expected_sha256: str,
) -> bool:
    if not _state_matches_owned_file(
        _probe_file_state(path),
        expected_identity,
    ):
        return False
    try:
        content = path.read_bytes()
        _canonical_manifest_document(content)
    except (OSError, RuntimeError):
        return False
    return content == expected_bytes and _sha256_bytes(content) == expected_sha256


def _rename_no_overwrite(
    source: Path,
    target: Path,
) -> Literal["moved", "collision", "failed"]:
    if os.name == "nt":
        try:
            os.rename(source, target)
        except FileExistsError:
            return "collision"
        except OSError:
            return "failed"
        return "moved"

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            return "failed"
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, target_bytes, 0x00000004)
    else:
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            return "failed"
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            source_bytes,
            -100,
            target_bytes,
            0x00000001,
        )
    if result == 0:
        return "moved"
    return "collision" if ctypes.get_errno() == errno.EEXIST else "failed"


def _move_manifest_to_quarantine_no_overwrite(
    manifest_path: Path,
) -> Path | None:
    for _ in range(8):
        quarantine = manifest_path.with_name(
            f".{manifest_path.name}.rollback.{secrets.token_hex(16)}.tmp"
        )
        outcome = _rename_no_overwrite(manifest_path, quarantine)
        if outcome == "moved":
            return quarantine
        if outcome == "failed":
            return None
    return None


def _quarantine_published_manifest(
    manifest_path: Path,
    temporary_path: Path,
) -> Path | None:
    """Move the current owner aside before deciding whether it is ours."""
    quarantine = _move_manifest_to_quarantine_no_overwrite(manifest_path)
    if quarantine is None:
        return None
    try:
        own_published_inode = quarantine.samefile(temporary_path)
    except OSError:
        return quarantine
    if own_published_inode:
        return quarantine
    try:
        os.link(quarantine, manifest_path)
    except OSError:
        return quarantine
    try:
        if not manifest_path.samefile(quarantine):
            return quarantine
    except OSError:
        return quarantine
    return quarantine


def _file_state(metadata: os.stat_result) -> _FileState:
    return _FileState(
        identity=_FileIdentity(
            device=metadata.st_dev,
            inode=metadata.st_ino,
        ),
        regular=stat.S_ISREG(metadata.st_mode),
    )


def _probe_file_state(path: Path) -> _FileState | None:
    try:
        return _file_state(path.stat(follow_symlinks=False))
    except OSError:
        return None


def _state_matches_owned_file(
    state: _FileState | None,
    expected_identity: _FileIdentity,
) -> bool:
    return bool(
        state is not None
        and state.identity == expected_identity
        and state.regular
    )


def _move_owned_file_to_cleanup_quarantine(path: Path) -> Path | None:
    for _ in range(8):
        try:
            quarantine = path.with_name(
                f"{path.name}.cleanup.{secrets.token_hex(16)}.tmp"
            )
            outcome = _rename_no_overwrite(path, quarantine)
        except Exception:
            return None
        if outcome == "moved":
            return quarantine
        if outcome == "failed":
            return None
    return None


def _quarantine_owned_file(
    path: Path,
    expected_identity: _FileIdentity | None,
) -> Path | None:
    if expected_identity is None:
        return None
    quarantine = _move_owned_file_to_cleanup_quarantine(path)
    if quarantine is None:
        return None
    if not _state_matches_owned_file(
        _probe_file_state(quarantine),
        expected_identity,
    ):
        try:
            _rename_no_overwrite(quarantine, path)
        except Exception:
            pass
        return None
    return quarantine


def _best_effort_close_file(file: BinaryIO) -> None:
    try:
        file.close()
    except Exception:
        return


def _best_effort_close_descriptor(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        return


def _delete_matching_manifest_temporary_windows(
    candidate: Path,
    expected_sha256: str,
) -> None:
    import ctypes
    import msvcrt

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("file_attributes", ctypes.c_uint32),
            ("reparse_tag", ctypes.c_uint32),
        ]

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = [("delete_file", ctypes.c_ubyte)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.restype = ctypes.c_void_p
    get_handle_info = kernel32.GetFileInformationByHandleEx
    get_final_path = kernel32.GetFinalPathNameByHandleW
    set_handle_info = kernel32.SetFileInformationByHandle
    invalid_handle = ctypes.c_void_p(-1).value
    handle = create_file(
        str(candidate),
        0x80000000 | 0x00010000,
        0x00000001,
        None,
        3,
        0x00200000,
        None,
    )
    if handle == invalid_handle:
        return
    descriptor: int | None = None
    try:
        attributes = FileAttributeTagInfo()
        if not get_handle_info(
            ctypes.c_void_p(handle),
            9,
            ctypes.byref(attributes),
            ctypes.sizeof(attributes),
        ) or attributes.file_attributes & 0x00000400:
            return
        path_buffer = ctypes.create_unicode_buffer(32768)
        path_length = int(
            get_final_path(
                ctypes.c_void_p(handle),
                path_buffer,
                len(path_buffer),
                0,
            )
        )
        if (
            path_length == 0
            or path_length >= len(path_buffer)
            or _normalized_windows_path(path_buffer.value)
            != _normalized_windows_path(str(candidate))
        ):
            return
        try:
            descriptor = msvcrt.open_osfhandle(
                int(handle),
                os.O_RDONLY | getattr(os, "O_BINARY", 0),
            )
        except OSError:
            return
        handle = None
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > 1024 * 1024
            or _sha256_descriptor(descriptor) != expected_sha256
        ):
            return
        disposition = FileDispositionInfo(1)
        set_handle_info(
            ctypes.c_void_p(msvcrt.get_osfhandle(descriptor)),
            4,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        )
    except OSError:
        return
    finally:
        if descriptor is not None:
            _best_effort_close_descriptor(descriptor)
        elif handle not in (None, invalid_handle):
            kernel32.CloseHandle(ctypes.c_void_p(handle))


def _delete_matching_manifest_temporary(
    candidate: Path,
    expected_sha256: str,
) -> None:
    if os.name == "nt":
        _delete_matching_manifest_temporary_windows(candidate, expected_sha256)


def _cleanup_matching_manifest_temporaries(
    data_root: Path,
    *,
    prepared_manifest_sha256: str,
    frozen_manifest_sha256: str,
) -> None:
    for candidate in data_root.glob(".manifest.json.*.tmp"):
        expected_sha256 = (
            prepared_manifest_sha256
            if candidate.name.startswith(".manifest.json.prepared.")
            else frozen_manifest_sha256
        )
        _delete_matching_manifest_temporary(candidate, expected_sha256)


def _replace_manifest_guarded(
    manifest_path: Path,
    prepared_bytes: bytes,
    frozen_bytes: bytes,
    *,
    before_manifest_replace: Callable[[], object] | None = None,
    evidence: Sequence[_EvidenceIdentity | BoundArtifact] = (),
    bound_manifest: BoundArtifact | None = None,
) -> None:
    prepared_sha256 = _sha256_bytes(prepared_bytes)
    frozen_sha256 = _sha256_bytes(frozen_bytes)
    _canonical_manifest_document(frozen_bytes)
    if bound_manifest is not None:
        if (
            bound_manifest.content != prepared_bytes
            or bound_manifest.sha256 != prepared_sha256
            or _sha256_descriptor(bound_manifest.descriptor) != bound_manifest.sha256
        ):
            raise RuntimeError("freeze approval failed")
        try:
            bound_manifest.verify_path_identity()
        except ValueError:
            raise RuntimeError("freeze approval failed") from None
    else:
        try:
            if manifest_path.read_bytes() != prepared_bytes:
                raise RuntimeError("freeze approval failed")
        except OSError:
            raise RuntimeError("freeze approval failed") from None

    stable_evidence = tuple(item for item in evidence if item is not bound_manifest)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=manifest_path.parent,
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    owned_descriptor: int | None = descriptor
    temporary_file: BinaryIO | None = None
    temporary_identity: _FileIdentity | None = None
    backup_candidate: Path | None = None
    backup_path: Path | None = None
    backup_identity: _FileIdentity | None = None
    source_manifest_state: _FileState | None = None
    manifest_taken = False
    committed = False
    try:
        temporary_state = _file_state(os.fstat(descriptor))
        if not temporary_state.regular:
            raise RuntimeError("freeze approval failed")
        temporary_identity = temporary_state.identity
        temporary_file = cast(BinaryIO, os.fdopen(descriptor, "wb"))
        owned_descriptor = None
        temporary_file.write(frozen_bytes)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
        temporary_file.close()
        temporary_file = None
        if before_manifest_replace is not None:
            before_manifest_replace()
        with (
            _stable_evidence_files(stable_evidence),
            _stable_bound_artifact_hash(bound_manifest),
        ):
            try:
                source_manifest_state = _probe_file_state(manifest_path)
                if source_manifest_state is None or not source_manifest_state.regular:
                    raise RuntimeError("freeze approval failed")
                if bound_manifest is None:
                    current_manifest = manifest_path.read_bytes()
                    if (
                        current_manifest != prepared_bytes
                        or _sha256_bytes(current_manifest) != prepared_sha256
                    ):
                        raise RuntimeError("freeze approval failed")
                if bound_manifest is not None and source_manifest_state.identity != (
                    _FileIdentity(
                        device=bound_manifest.device,
                        inode=bound_manifest.inode,
                    )
                ):
                    raise RuntimeError("freeze approval failed")
                for _ in range(8):
                    backup_candidate = manifest_path.with_name(
                        f".{manifest_path.name}.prepared."
                        f"{secrets.token_hex(16)}.tmp"
                    )
                    outcome = _rename_no_overwrite(
                        manifest_path,
                        backup_candidate,
                    )
                    if outcome == "collision":
                        continue
                    if outcome == "failed":
                        raise RuntimeError("freeze approval failed")
                    backup_path = backup_candidate
                    manifest_taken = True
                    backup_state = _probe_file_state(backup_path)
                    if not _state_matches_owned_file(
                        backup_state,
                        source_manifest_state.identity
                    ):
                        raise RuntimeError("freeze approval failed")
                    backup_identity = source_manifest_state.identity
                    break
                else:
                    raise RuntimeError("freeze approval failed")
                if backup_identity is None:
                    raise RuntimeError("freeze approval failed")
                if not _manifest_recovery_matches(
                    backup_path,
                    expected_identity=backup_identity,
                    expected_bytes=prepared_bytes,
                    expected_sha256=prepared_sha256,
                    retained_manifest=bound_manifest,
                ):
                    raise RuntimeError("freeze approval failed")
                try:
                    os.link(temporary_path, manifest_path)
                except OSError:
                    if not (
                        temporary_identity is not None
                        and backup_identity is not None
                        and _owned_manifest_matches(
                            manifest_path,
                            expected_identity=temporary_identity,
                            expected_bytes=frozen_bytes,
                            expected_sha256=frozen_sha256,
                        )
                        and _manifest_recovery_matches(
                            backup_path,
                            expected_identity=backup_identity,
                            expected_bytes=prepared_bytes,
                            expected_sha256=prepared_sha256,
                            retained_manifest=bound_manifest,
                        )
                    ):
                        raise
            except OSError:
                raise RuntimeError("freeze approval failed") from None
        if (
            temporary_identity is None
            or backup_identity is None
            or not _owned_manifest_matches(
                manifest_path,
                expected_identity=temporary_identity,
                expected_bytes=frozen_bytes,
                expected_sha256=frozen_sha256,
            )
            or not _manifest_recovery_matches(
                backup_path,
                expected_identity=backup_identity,
                expected_bytes=prepared_bytes,
                expected_sha256=prepared_sha256,
                retained_manifest=bound_manifest,
            )
        ):
            raise RuntimeError("freeze approval failed")
        committed = True
    except BaseException:
        if (
            not manifest_taken
            and backup_candidate is not None
            and source_manifest_state is not None
            and _state_matches_owned_file(
                _probe_file_state(backup_candidate),
                source_manifest_state.identity,
            )
        ):
            backup_path = backup_candidate
            backup_identity = source_manifest_state.identity
            manifest_taken = True
        if not committed and manifest_taken:
            _quarantine_published_manifest(manifest_path, temporary_path)
        raise
    finally:
        if temporary_file is not None:
            _best_effort_close_file(temporary_file)
        if owned_descriptor is not None:
            _best_effort_close_descriptor(owned_descriptor)
        _quarantine_owned_file(temporary_path, temporary_identity)
        if committed and backup_path is not None:
            _quarantine_owned_file(backup_path, backup_identity)


@contextmanager
def _exclusive_freeze_lock(data_root: Path) -> Iterator[None]:
    lock_path = data_root / ".task2-freeze.lock"
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.restype = ctypes.c_void_p
        handle = create_file(
            str(lock_path.resolve()),
            0x40000000 | 0x00010000,
            0,
            None,
            1,
            0x04000000,
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            raise RuntimeError("freeze approval failed")
        try:
            yield
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
        return

    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        raise RuntimeError("freeze approval failed") from None
    try:
        yield
    finally:
        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = lock_path.stat()
            same_file = (
                descriptor_stat.st_dev == path_stat.st_dev
                and descriptor_stat.st_ino == path_stat.st_ino
            )
        except OSError:
            same_file = False
        os.close(descriptor)
        if same_file:
            lock_path.unlink(missing_ok=True)


@contextmanager
def _stable_report_parent(data_root: Path, report_path: Path) -> Iterator[None]:
    root = data_root.resolve()
    parent = report_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent_relative = parent.resolve().relative_to(root)
    except (OSError, ValueError):
        raise RuntimeError("freeze approval failed") from None

    directories = [root]
    current = root
    for part in parent_relative.parts:
        current = current / part
        directories.append(current)

    handles: list[int] = []
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create_file = kernel32.CreateFileW
            create_file.restype = ctypes.c_void_p
            invalid_handle = ctypes.c_void_p(-1).value
            get_attributes = kernel32.GetFileAttributesW
            get_attributes.restype = ctypes.c_uint32

            for directory in directories:
                absolute = str(directory)
                attributes = int(get_attributes(absolute))
                if attributes == 0xFFFFFFFF or attributes & 0x00000400:
                    raise RuntimeError("freeze approval failed")
                handle = create_file(
                    absolute,
                    0x80000000 | 0x00010000,
                    0x00000001 | 0x00000002,
                    None,
                    3,
                    0x02000000 | 0x00200000,
                    None,
                )
                if handle == invalid_handle:
                    raise RuntimeError("freeze approval failed")
                handles.append(int(handle))
        else:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            for directory in directories:
                handles.append(os.open(directory, flags))

        if (
            _confined_report_relative_path(data_root, report_path)
            != report_path.resolve().relative_to(root).as_posix()
        ):
            raise RuntimeError("freeze approval failed")
        yield
    finally:
        if os.name == "nt":
            if handles:
                import ctypes

                close_handle = ctypes.WinDLL(
                    "kernel32",
                    use_last_error=True,
                ).CloseHandle
                for handle in reversed(handles):
                    close_handle(ctypes.c_void_p(handle))
        else:
            for handle in reversed(handles):
                os.close(handle)


def _approve_freeze_locked(
    *,
    data_root: Path,
    plan: FreezeApprovalPlan,
    before_manifest_replace: Callable[[], object] | None,
) -> Literal["created", "matched"]:
    report_path = _validated_plan_report_path(data_root, plan)
    if not _evidence_matches(plan.evidence):
        raise RuntimeError("freeze approval failed")
    manifest_path = data_root / "manifest.json"
    try:
        current_manifest = manifest_path.read_bytes()
    except OSError:
        raise RuntimeError("freeze approval failed") from None

    if current_manifest == plan.frozen_manifest_bytes:
        if report_path.is_file() and report_path.read_bytes() == plan.report_bytes:
            _cleanup_matching_manifest_temporaries(
                data_root,
                prepared_manifest_sha256=plan.report.prepared_manifest_sha256,
                frozen_manifest_sha256=_sha256_bytes(plan.frozen_manifest_bytes),
            )
            return "matched"
        raise RuntimeError("freeze approval failed")
    if current_manifest != plan.prepared_manifest_bytes:
        raise RuntimeError("freeze approval failed")
    if report_path.exists() and report_path.read_bytes() != plan.report_bytes:
        raise FileExistsError("freeze approval failed")

    missing_parents: list[Path] = []
    candidate = report_path.parent
    while not candidate.exists():
        missing_parents.append(candidate)
        candidate = candidate.parent
    try:
        with _stable_report_parent(data_root, report_path):
            try:
                _confined_report_relative_path(data_root, report_path)
            except ValueError:
                raise RuntimeError("freeze approval failed") from None
            write_frozen_bytes(report_path, plan.report_bytes)
            try:
                if (
                    _validated_plan_report_path(data_root, plan).resolve() != report_path.resolve()
                    or not report_path.is_file()
                    or report_path.read_bytes() != plan.report_bytes
                ):
                    raise RuntimeError("freeze approval failed")
            except (OSError, ValueError):
                raise RuntimeError("freeze approval failed") from None
    except BaseException:
        for created_parent in missing_parents:
            try:
                created_parent.rmdir()
            except OSError:
                break
        raise
    _replace_manifest_guarded(
        manifest_path,
        plan.prepared_manifest_bytes,
        plan.frozen_manifest_bytes,
        before_manifest_replace=before_manifest_replace,
        evidence=plan.evidence,
    )
    return "created"


def approve_freeze(
    *,
    data_root: Path,
    plan: FreezeApprovalPlan,
    before_manifest_replace: Callable[[], object] | None = None,
) -> Literal["created", "matched"]:
    """Approve under an exclusive lock after revalidating all bound evidence."""
    with _exclusive_freeze_lock(data_root):
        return _approve_freeze_locked(
            data_root=data_root,
            plan=plan,
            before_manifest_replace=before_manifest_replace,
        )


def _migration_evidence(
    data_root: Path,
    relative_path: SafeRelativePath,
    expected_sha256: str,
    *,
    stack: ExitStack,
) -> BoundArtifact:
    try:
        artifact = stack.enter_context(open_confined_artifact(data_root, relative_path))
    except ValueError:
        raise ValueError("freeze migration failed") from None
    if artifact.sha256 != expected_sha256:
        raise ValueError("freeze migration failed")
    return artifact


def _migration_json(content: bytes) -> object:
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("freeze migration failed") from None


def _migration_identifiers(content: bytes, expected_count: int) -> list[str]:
    value = _migration_json(content)
    if (
        not isinstance(value, list)
        or len(value) != expected_count
        or not all(isinstance(item, str) and item.strip() for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("freeze migration failed")
    return cast(list[str], value)


def _migration_partition(
    data_root: Path,
    partition: LegacyPartitionV1,
    *,
    stack: ExitStack,
) -> tuple[bytes, BoundArtifact, BoundArtifact]:
    gold_path = partition.gold_path
    gold_hash = partition.gold_sha256
    ids_path = partition.ids_path
    ids_hash = partition.ids_sha256
    count = partition.count
    policy = partition.zero_answer_policy
    gold_evidence = _migration_evidence(data_root, gold_path, gold_hash, stack=stack)
    ids_evidence = _migration_evidence(data_root, ids_path, ids_hash, stack=stack)
    identifiers = _migration_identifiers(ids_evidence.content, count)
    try:
        rows = [
            _mapping(_migration_json(line)) for line in gold_evidence.content.splitlines() if line
        ]
        records = [EvaluationQuery.model_validate(row) for row in rows]
    except (ValidationError, ValueError):
        raise ValueError("freeze migration failed") from None
    if (
        not records
        or len(records) != count
        or [record.query_id for record in records] != identifiers
        or len({record.query_id for record in records}) != count
        or (policy == "reject" and any(not record.relevant_paper_ids for record in records))
    ):
        raise ValueError("freeze migration failed")
    return gold_evidence.content, gold_evidence, ids_evidence


def _migration_source_evidence(
    data_root: Path,
    v1: FreezeManifestV1,
    *,
    stack: ExitStack,
) -> list[BoundArtifact]:
    evidence: list[BoundArtifact] = []
    for source in v1.source_files:
        identity = _migration_evidence(
            data_root, source.raw_path, source.sha256, stack=stack
        )
        if (
            len(identity.content) != source.byte_count
            or len(identity.content.splitlines()) != source.row_count
        ):
            raise ValueError("freeze migration failed")
        evidence.append(identity)
    for name, package in v1.work_packages.items():
        ids_identity = _migration_evidence(
            data_root, package.ids_path, package.ids_sha256, stack=stack
        )
        _migration_identifiers(ids_identity.content, package.count)
        evidence.append(ids_identity)
        if name == "overlap":
            if package.source_path is not None or package.source_sha256 is not None:
                raise ValueError("freeze migration failed")
            continue
        if package.source_path is None or package.source_sha256 is None:
            raise ValueError("freeze migration failed")
        source_identity = _migration_evidence(
            data_root, package.source_path, package.source_sha256, stack=stack
        )
        if len(source_identity.content.splitlines()) != package.count:
            raise ValueError("freeze migration failed")
        evidence.append(source_identity)
    return evidence


def _validated_v1_report(
    data_root: Path,
    v1: FreezeManifestV1,
    *,
    stack: ExitStack,
) -> BoundArtifact:
    report_identity = _migration_evidence(
        data_root,
        v1.freeze_report_path,
        v1.freeze_report_sha256,
        stack=stack,
    )
    try:
        report = FreezeAuditReport.model_validate(_migration_json(report_identity.content))
    except ValidationError:
        raise ValueError("freeze migration failed") from None
    if (
        _json_bytes(report.model_dump(mode="json")) != report_identity.content
        or report.approval_requested is not True
        or report.prepared_manifest_sha256 != v1.prepared_manifest_sha256
        or report.dataset_revision != v1.revision
        or set(report.partitions) != set(v1.partitions)
    ):
        raise ValueError("freeze migration failed")
    for name, partition in v1.partitions.items():
        audit_partition = report.partitions[name]
        if (
            audit_partition.count != partition.count
            or audit_partition.gold_path != partition.gold_path
            or audit_partition.gold_sha256 != partition.gold_sha256
            or audit_partition.ids_path != partition.ids_path
            or audit_partition.ids_sha256 != partition.ids_sha256
            or audit_partition.zero_answer_policy != partition.zero_answer_policy
            or audit_partition.labels_complete is not True
        ):
            raise ValueError("freeze migration failed")
    return report_identity


def _validated_identifier_map(
    data_root: Path,
    identifier_map: IdentifierMapBindingV2,
    *,
    stack: ExitStack,
) -> tuple[bytes, BoundArtifact]:
    identity = _migration_evidence(
        data_root, identifier_map.path, identifier_map.sha256, stack=stack
    )
    try:
        value = parse_json_object_bytes(identity.content)
    except ValueError:
        raise ValueError("freeze migration failed") from None
    if (
        not isinstance(value, dict)
        or len(value) != identifier_map.entry_count
        or not value
        or not all(
            isinstance(key, str)
            and key.strip()
            and isinstance(target, str)
            and target.strip()
            for key, target in value.items()
        )
    ):
        raise ValueError("freeze migration failed")
    return identity.content, identity


def _migration_manifest(
    v1: FreezeManifestV1,
    *,
    approval: FreezeApprovalReportV2,
    identifier_map: IdentifierMapBindingV2,
    dataset_revision: str,
    approval_report_path: SafeRelativePath,
    dev_gold_sha256: str,
    validation_gold_sha256: str,
) -> tuple[FreezeManifestV2, bytes, bytes]:
    if not dataset_revision.strip() or approval.approval_requested is not True:
        raise ValueError("freeze migration failed")
    if approval.partition_hashes != {
        "dev": dev_gold_sha256,
        "validation": validation_gold_sha256,
    } or approval.identifier_map_sha256 != identifier_map.sha256:
        raise ValueError("freeze migration failed")
    report_bytes = _json_bytes(approval.model_dump(mode="json"))
    manifest = FreezeManifestV2(
        schema_version="paper-search-freeze-v2",
        dataset_revision=dataset_revision,
        created_at=approval.approved_at,
        annotation_status="frozen",
        freeze_status="approved",
        partitions=[
            FrozenPartitionV2(
                name="dev",
                path=v1.partitions["dev"].gold_path,
                query_count=v1.partitions["dev"].count,
                sha256=dev_gold_sha256,
                zero_answer_policy=(
                    "forbid"
                    if v1.partitions["dev"].zero_answer_policy == "reject"
                    else "allow"
                ),
            ),
            FrozenPartitionV2(
                name="validation",
                path=v1.partitions["validation"].gold_path,
                query_count=v1.partitions["validation"].count,
                sha256=validation_gold_sha256,
                zero_answer_policy=(
                    "forbid"
                    if v1.partitions["validation"].zero_answer_policy == "reject"
                    else "allow"
                ),
            ),
        ],
        gold_sha256=canonical_gold_set_sha256(dev_gold_sha256, validation_gold_sha256),
        identifier_map=identifier_map,
        partition_immutability="content_addressed",
        approval=FreezeApprovalBindingV2(
            report_path=approval_report_path,
            report_sha256=sha256_bytes(report_bytes),
            approved_at=approval.approved_at,
            approver_ref=approval.approver_ref,
        ),
    )
    return manifest, report_bytes, _json_bytes(manifest.model_dump(mode="json"))


def _bound_artifacts_share_identity(
    first: BoundArtifact,
    second: BoundArtifact,
) -> bool:
    if os.name == "nt":
        return (
            first.windows_file_id is not None
            and first.windows_file_id == second.windows_file_id
        )
    return (first.device, first.inode) == (second.device, second.inode)


def _artifact_matches_content(
    artifact: BoundArtifact,
    *,
    expected_bytes: bytes,
    expected_sha256: str,
) -> bool:
    try:
        artifact.verify_path_identity()
        descriptor_sha256 = _sha256_descriptor(artifact.descriptor)
    except (OSError, ValueError):
        return False
    return (
        artifact.content == expected_bytes
        and artifact.sha256 == expected_sha256
        and descriptor_sha256 == expected_sha256
    )


def _validated_v1_recovery(
    data_root: Path,
    recovery: BoundArtifact,
    *,
    expected_sha256: Sha256,
    stack: ExitStack,
) -> tuple[FreezeManifestV1, list[BoundArtifact]]:
    if not _artifact_matches_content(
        recovery,
        expected_bytes=recovery.content,
        expected_sha256=expected_sha256,
    ):
        raise ValueError("freeze recovery failed")
    try:
        manifest = _canonical_freeze_manifest(recovery.content)
    except RuntimeError:
        raise ValueError("freeze recovery failed") from None
    if not isinstance(manifest, FreezeManifestV1):
        raise ValueError("freeze recovery failed")
    evidence = [recovery, _validated_v1_report(data_root, manifest, stack=stack)]
    for name in ("dev", "validation", "simulated_test"):
        _, gold_evidence, ids_evidence = _migration_partition(
            data_root,
            manifest.partitions[name],
            stack=stack,
        )
        evidence.extend((gold_evidence, ids_evidence))
    evidence.extend(_migration_source_evidence(data_root, manifest, stack=stack))
    return manifest, evidence


def recover_freeze_manifest(
    *,
    data_root: Path,
    recovery_path: SafeRelativePath,
    expected_sha256: Sha256,
) -> Literal["created", "matched"]:
    """Validate and publish one approved legacy V1 recovery without overwrite."""
    root = data_root.resolve()
    manifest_path = root / "manifest.json"
    if recovery_path == "manifest.json":
        raise ValueError("freeze recovery failed")
    created = False
    try:
        with _exclusive_freeze_lock(root):
            with ExitStack() as stack:
                try:
                    recovery = stack.enter_context(
                        open_confined_artifact(root, recovery_path)
                    )
                    _, evidence = _validated_v1_recovery(
                        root,
                        recovery,
                        expected_sha256=expected_sha256,
                        stack=stack,
                    )
                except ValueError:
                    raise ValueError("freeze recovery failed") from None
                final_artifact: BoundArtifact | None = None
                try:
                    with _stable_evidence_files(evidence):
                        if manifest_path.exists():
                            try:
                                existing = stack.enter_context(
                                    open_confined_artifact(root, "manifest.json")
                                )
                            except ValueError:
                                raise FileExistsError(
                                    "freeze recovery failed"
                                ) from None
                            if not _artifact_matches_content(
                                existing,
                                expected_bytes=recovery.content,
                                expected_sha256=expected_sha256,
                            ):
                                raise FileExistsError("freeze recovery failed")
                            with _stable_evidence_files((existing,)):
                                return "matched"
                        try:
                            os.link(
                                root.joinpath(*recovery.relative_path.split("/")),
                                manifest_path,
                            )
                        except OSError as error:
                            try:
                                final_artifact = stack.enter_context(
                                    open_confined_artifact(root, "manifest.json")
                                )
                            except ValueError:
                                if manifest_path.exists():
                                    raise FileExistsError(
                                        "freeze recovery failed"
                                    ) from None
                                raise RuntimeError(
                                    "freeze recovery failed"
                                ) from error
                            if not (
                                _bound_artifacts_share_identity(
                                    recovery,
                                    final_artifact,
                                )
                                and _artifact_matches_content(
                                    final_artifact,
                                    expected_bytes=recovery.content,
                                    expected_sha256=expected_sha256,
                                )
                            ):
                                if isinstance(error, FileExistsError):
                                    raise FileExistsError(
                                        "freeze recovery failed"
                                    ) from None
                                raise RuntimeError(
                                    "freeze recovery failed"
                                ) from error
                            created = True
                        else:
                            created = True
                            try:
                                final_artifact = stack.enter_context(
                                    open_confined_artifact(root, "manifest.json")
                                )
                            except ValueError:
                                raise RuntimeError(
                                    "freeze recovery failed"
                                ) from None
                        if final_artifact is None or not (
                            _bound_artifacts_share_identity(
                                recovery,
                                final_artifact,
                            )
                            and _artifact_matches_content(
                                final_artifact,
                                expected_bytes=recovery.content,
                                expected_sha256=expected_sha256,
                            )
                        ):
                            raise RuntimeError("freeze recovery failed")
                        with _stable_evidence_files((final_artifact,)):
                            pass
                except BaseException:
                    if created:
                        _move_manifest_to_quarantine_no_overwrite(manifest_path)
                    raise
                return "created"
    except FileExistsError:
        raise FileExistsError("freeze recovery failed") from None
    except ValueError:
        raise ValueError("freeze recovery failed") from None
    except RuntimeError:
        raise RuntimeError("freeze recovery failed") from None
    except OSError:
        raise RuntimeError("freeze recovery failed") from None


def migrate_v1_to_v2(
    v1: FreezeManifestV1,
    *,
    data_root: Path,
    approval: FreezeApprovalReportV2,
    identifier_map: IdentifierMapBindingV2,
    dataset_revision: str,
    approval_report_path: SafeRelativePath,
) -> FreezeManifestV2:
    """Verify an approved legacy freeze and atomically publish its V2 successor."""
    root = data_root.resolve()
    manifest_path = root / "manifest.json"
    if approval_report_path == "manifest.json":
        raise ValueError("freeze migration failed")
    with _exclusive_freeze_lock(root):
        with ExitStack() as stack:
            try:
                current_artifact = stack.enter_context(
                    open_confined_artifact(
                        root, "manifest.json", replaceable_manifest=True
                    )
                )
                current_bytes = current_artifact.content
                current = parse_freeze_manifest_bytes(current_bytes)
            except ValueError:
                raise ValueError("freeze migration failed") from None
            if not isinstance(current, (FreezeManifestV1, FreezeManifestV2)):
                raise ValueError("freeze migration failed")
            if isinstance(current, FreezeManifestV1) and current != v1:
                raise ValueError("freeze migration failed")
            legacy_report_evidence = _validated_v1_report(root, v1, stack=stack)
            partition_content: dict[str, bytes] = {}
            evidence: list[BoundArtifact] = [current_artifact, legacy_report_evidence]
            for name in ("dev", "validation", "simulated_test"):
                content, gold_evidence, ids_evidence = _migration_partition(
                    root, v1.partitions[name], stack=stack
                )
                partition_content[name] = content
                evidence.extend((gold_evidence, ids_evidence))
            evidence.extend(_migration_source_evidence(root, v1, stack=stack))
            _, identifier_evidence = _validated_identifier_map(
                root, identifier_map, stack=stack
            )
            evidence.append(identifier_evidence)
            dev_hash = sha256_bytes(partition_content["dev"])
            validation_hash = sha256_bytes(partition_content["validation"])
            if approval.audit_sha256 != v1.freeze_report_sha256:
                raise ValueError("freeze migration failed")
            migrated, approval_bytes, migrated_bytes = _migration_manifest(
                v1,
                approval=approval,
                identifier_map=identifier_map,
                dataset_revision=dataset_revision,
                approval_report_path=approval_report_path,
                dev_gold_sha256=dev_hash,
                validation_gold_sha256=validation_hash,
            )
            if isinstance(current, FreezeManifestV2):
                try:
                    existing_report = stack.enter_context(
                        open_confined_artifact(root, approval_report_path)
                    )
                except ValueError:
                    raise ValueError("freeze migration failed") from None
                evidence.append(existing_report)
                with _stable_evidence_files(evidence):
                    if not (
                        current == migrated
                        and current_bytes == migrated_bytes
                        and existing_report.content == approval_bytes
                    ):
                        raise ValueError("freeze migration failed")
                return migrated
            try:
                _, published_report = stack.enter_context(
                    publish_confined_artifact_no_overwrite(
                        root, approval_report_path, approval_bytes
                    )
                )
            except (OSError, ValueError, FileExistsError):
                raise RuntimeError("freeze migration failed") from None
            evidence.append(published_report)
            _replace_manifest_guarded(
                manifest_path,
                current_bytes,
                migrated_bytes,
                evidence=evidence,
                bound_manifest=current_artifact,
            )
            return migrated


def _prepared_partition_names(data_root: Path) -> set[str]:
    try:
        manifest_bytes = (data_root / "manifest.json").read_bytes()
    except OSError:
        raise ValueError("prepared data is invalid") from None
    manifest = _mapping(_load_json_bytes(manifest_bytes))
    partitions = _mapping(manifest.get("partitions"))
    if not partitions:
        raise ValueError("prepared data is invalid")
    return set(partitions)


def _confined_report_relative_path(data_root: Path, report_path: Path) -> str:
    data_root_resolved = data_root.resolve()
    report_root = (data_root_resolved / "freeze_reports").resolve()
    resolved = report_path.resolve()
    try:
        report_relative = resolved.relative_to(report_root)
        data_relative = resolved.relative_to(data_root_resolved)
    except ValueError:
        raise ValueError("freeze approval failed") from None
    if report_relative == Path("."):
        raise ValueError("freeze approval failed")
    return data_relative.as_posix()


def _match_existing_approval(
    *,
    data_root: Path,
    report_relative_path: str,
    type_domain_labels_path: Path,
    constraint_labels_path: Path,
    overlap_labels_path: Path,
    policies: Mapping[str, ZeroAnswerPolicy],
) -> FreezeAuditReport | None:
    manifest_path = data_root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError:
        raise ValueError("prepared data is invalid") from None
    manifest = _mapping(_load_json_bytes(manifest_bytes))
    if manifest.get("status") != "frozen":
        return None
    expected_identity: dict[str, object] = {
        "repo_id": PASA_REPO_ID,
        "revision": PASA_REVISION,
        "license": "CC-BY-NC-SA-4.0",
        "access": "gated-hugging-face-dataset",
        "random_seed": RANDOM_SEED,
        "sampling_algorithm": SAMPLING_ALGORITHM,
        "status": "frozen",
        "work_package_sampling": "answer-count-largest-remainder-v1-seeded-offsets",
    }
    expected_fields = {
        *expected_identity,
        "source_files",
        "partitions",
        "work_packages",
        "prepared_manifest_sha256",
        "freeze_report_path",
        "freeze_report_sha256",
    }
    if (
        set(manifest) != expected_fields
        or any(manifest.get(key) != expected for key, expected in expected_identity.items())
        or manifest.get("freeze_report_path") != report_relative_path
        or _json_bytes(manifest) != manifest_bytes
    ):
        raise ValueError("prepared data is invalid")

    report_path = data_root / report_relative_path
    try:
        report_bytes = report_path.read_bytes()
        report = FreezeAuditReport.model_validate(_load_json_bytes(report_bytes))
    except (OSError, ValidationError):
        raise ValueError("prepared data is invalid") from None
    if (
        _json_bytes(report.model_dump(mode="json")) != report_bytes
        or manifest.get("freeze_report_sha256") != _sha256_bytes(report_bytes)
        or report.approval_requested is not True
        or manifest.get("prepared_manifest_sha256") != report.prepared_manifest_sha256
    ):
        raise ValueError("prepared data is invalid")

    source_counts = _expectation_map(OFFICIAL_EXPECTATIONS.source_row_counts)
    partition_counts = _expectation_map(OFFICIAL_EXPECTATIONS.partition_counts)
    work_package_counts = _expectation_map(OFFICIAL_EXPECTATIONS.work_package_counts)
    source_file_count, _ = _validate_source_files(
        data_root,
        manifest.get("source_files"),
        source_counts,
    )
    frozen_partition_payload = _mapping(manifest.get("partitions"))
    prepared_partitions: dict[str, dict[str, object]] = {}
    expected_frozen_fields = {
        "count",
        "gold_path",
        "gold_sha256",
        "ids_path",
        "ids_sha256",
        "zero_answer_policy",
        "labels_complete",
    }
    for name, raw_partition in frozen_partition_payload.items():
        partition = _mapping(raw_partition)
        if (
            set(partition) != expected_frozen_fields
            or partition.get("zero_answer_policy") != policies.get(name)
            or partition.get("labels_complete") is not True
        ):
            raise ValueError("prepared data is invalid")
        prepared_partitions[name] = {
            key: partition[key]
            for key in (
                "count",
                "gold_path",
                "gold_sha256",
                "ids_path",
                "ids_sha256",
            )
        }
    partition_audits, frozen_partitions, partition_ids, _ = _validate_partitions(
        data_root,
        prepared_partitions,
        policies,
        partition_counts,
    )
    if frozen_partitions != frozen_partition_payload:
        raise ValueError("prepared data is invalid")
    type_domain_ids, constraint_ids, overlap_ids, _ = _validate_work_packages(
        data_root,
        manifest.get("work_packages"),
        partition_ids,
        work_package_counts,
    )
    (
        type_domain,
        constraints,
        overlap,
        agreement,
        type_domain_bytes,
        constraint_bytes,
        overlap_bytes,
        _,
    ) = _private_records(
        type_domain_labels_path,
        constraint_labels_path,
        overlap_labels_path,
        type_domain_ids,
        constraint_ids,
        overlap_ids,
    )
    expected_report = FreezeAuditReport(
        prepared_manifest_sha256=_nonempty_string(manifest.get("prepared_manifest_sha256")),
        dataset_revision=PASA_REVISION,
        source_file_count=source_file_count,
        type_domain_count=len(type_domain),
        type_domain_sha256=_sha256_bytes(type_domain_bytes),
        constraint_count=len(constraints),
        constraint_sha256=_sha256_bytes(constraint_bytes),
        overlap_count=len(overlap),
        overlap_sha256=_sha256_bytes(overlap_bytes),
        agreement=agreement,
        partitions=partition_audits,
        approval_requested=True,
    )
    if report != expected_report:
        raise ValueError("prepared data is invalid")
    _cleanup_matching_manifest_temporaries(
        data_root,
        prepared_manifest_sha256=report.prepared_manifest_sha256,
        frozen_manifest_sha256=_sha256_bytes(manifest_bytes),
    )
    return report


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise ValueError("freeze approval failed")


def _build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Audit and approve Task 2 data freeze")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--type-domain-labels", type=Path, required=True)
    parser.add_argument("--constraint-labels", type=Path, required=True)
    parser.add_argument("--overlap-labels", type=Path, required=True)
    parser.add_argument(
        "--zero-answer-policy",
        action="append",
        required=True,
        dest="zero_answer_policies",
    )
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def _audit_error_message(error: ValueError) -> str:
    reason = str(error)
    if reason == "private annotations are invalid":
        return "freeze audit failed: private annotations are invalid"
    if reason == "human annotation agreement is below threshold":
        return "freeze audit failed: human annotation agreement is below threshold"
    return "freeze audit failed: prepared data is invalid"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the secret-safe Task 2 freeze audit CLI."""
    try:
        args = _build_parser().parse_args(argv)
    except ValueError:
        print("freeze approval failed", file=sys.stderr)
        return 2
    if args.approve != (args.report is not None):
        print("freeze approval failed", file=sys.stderr)
        return 2
    report_relative_path: str | None = None
    if args.report is not None:
        try:
            report_relative_path = _confined_report_relative_path(
                args.data_root,
                args.report,
            )
        except ValueError:
            print("freeze approval failed", file=sys.stderr)
            return 2
    try:
        partition_names = _prepared_partition_names(args.data_root)
        policies = parse_zero_answer_policies(
            args.zero_answer_policies,
            partition_names,
        )
        existing_report = None
        if args.approve and report_relative_path is not None:
            existing_report = _match_existing_approval(
                data_root=args.data_root,
                report_relative_path=report_relative_path,
                type_domain_labels_path=args.type_domain_labels,
                constraint_labels_path=args.constraint_labels,
                overlap_labels_path=args.overlap_labels,
                policies=policies,
            )
        if existing_report is None:
            result = audit_freeze_candidate(
                data_root=args.data_root,
                type_domain_labels_path=args.type_domain_labels,
                constraint_labels_path=args.constraint_labels,
                overlap_labels_path=args.overlap_labels,
                policies=policies,
            )
    except ValueError as error:
        print(_audit_error_message(error), file=sys.stderr)
        return 2

    if existing_report is not None:
        output_report = existing_report
    else:
        output_report = result.report
        if args.approve:
            if report_relative_path is None:
                print("freeze approval failed", file=sys.stderr)
                return 2
            try:
                plan = build_approval_plan(
                    result,
                    report_relative_path=report_relative_path,
                )
                approve_freeze(data_root=args.data_root, plan=plan)
            except (OSError, RuntimeError, ValueError):
                print("freeze approval failed", file=sys.stderr)
                return 2
            output_report = plan.report
    print(
        json.dumps(
            output_report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
