from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import PositiveInt

from paper_search.domain.models import DomainModel, NonEmptyStr
from paper_search.evaluation.annotation import (
    AgreementReport,
    AnnotationRecord,
    TypeDomainAnnotationRecord,
    compare_annotations,
)
from paper_search.evaluation.dataset import EvaluationQuery, read_jsonl


PASA_REPO_ID = "CarlanLark/pasa-dataset"
PASA_REVISION = "232428b0c867268c3b8ded90db4d98c1b30501d6"
RANDOM_SEED = 20260714
SAMPLING_ALGORITHM = "answer-count-largest-remainder-v1"
PREPARED_STATUS = "waiting_for_human_label_freeze"

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
        if (
            not name
            or name not in expected
            or name in parsed
            or policy not in ("reject", "allow")
        ):
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
class FreezeAuditResult:
    """Pure audit output; it contains no approved report path or final bytes."""

    prepared_manifest_bytes: bytes
    frozen_manifest_payload: dict[str, object]
    report: FreezeAuditReport


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _load_json_bytes(content: bytes) -> object:
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("prepared data is invalid") from None


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


def _read_ids(data_root: Path, path_value: object) -> tuple[str, list[str], bytes]:
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
    return relative_path, identifiers, content


def _validate_source_files(data_root: Path, value: object) -> int:
    if not isinstance(value, list) or not value:
        raise ValueError("prepared data is invalid")
    for raw_item in value:
        item = _mapping(raw_item)
        _, path = _confined_path(data_root, item.get("raw_path"))
        expected_rows = _positive_integer(item.get("row_count"))
        expected_bytes = _positive_integer(item.get("byte_count"))
        expected_hash = _nonempty_string(item.get("sha256"))
        try:
            content = path.read_bytes()
        except OSError:
            raise ValueError("prepared data is invalid") from None
        row_count = len(content.splitlines())
        if (
            len(content) != expected_bytes
            or row_count != expected_rows
            or _sha256_bytes(content) != expected_hash
        ):
            raise ValueError("prepared data is invalid")
    return len(value)


def _validate_partitions(
    data_root: Path,
    value: object,
    policies: Mapping[str, ZeroAnswerPolicy],
) -> tuple[dict[str, PartitionFreezeAudit], dict[str, dict[str, object]], dict[str, list[str]]]:
    partitions = _mapping(value)
    if not partitions or set(policies) != set(partitions):
        raise ValueError("prepared data is invalid")
    audits: dict[str, PartitionFreezeAudit] = {}
    frozen: dict[str, dict[str, object]] = {}
    ids_by_partition: dict[str, list[str]] = {}
    for name, raw_partition in partitions.items():
        partition = _mapping(raw_partition)
        count = _positive_integer(partition.get("count"))
        policy = policies[name]
        if policy not in ("reject", "allow"):
            raise ValueError("prepared data is invalid")
        gold_relative, gold_path = _confined_path(data_root, partition.get("gold_path"))
        ids_relative, identifiers, ids_content = _read_ids(
            data_root, partition.get("ids_path")
        )
        try:
            gold_content = gold_path.read_bytes()
            records = read_jsonl(gold_path, EvaluationQuery)
        except (OSError, ValueError):
            raise ValueError("prepared data is invalid") from None
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
    return audits, frozen, ids_by_partition


def _validate_work_packages(
    data_root: Path,
    value: object,
    partition_ids: Mapping[str, list[str]],
) -> tuple[list[str], list[str], list[str]]:
    packages = _mapping(value)
    if set(packages) != {"type_domain", "constraints", "overlap"}:
        raise ValueError("prepared data is invalid")

    package_ids: dict[str, list[str]] = {}
    for name, raw_package in packages.items():
        package = _mapping(raw_package)
        count = _positive_integer(package.get("count"))
        _, identifiers, ids_content = _read_ids(data_root, package.get("ids_path"))
        if len(identifiers) != count or package.get("ids_sha256") != _sha256_bytes(ids_content):
            raise ValueError("prepared data is invalid")
        if name != "overlap":
            _, source_path = _confined_path(data_root, package.get("source_path"))
            try:
                source_content = source_path.read_bytes()
            except OSError:
                raise ValueError("prepared data is invalid") from None
            if package.get("source_sha256") != _sha256_bytes(source_content):
                raise ValueError("prepared data is invalid")
            if len(source_content.splitlines()) != count:
                raise ValueError("prepared data is invalid")
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
    return type_domain, constraints, overlap


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
]:
    try:
        type_domain = read_jsonl(type_domain_labels_path, TypeDomainAnnotationRecord)
        constraints = read_jsonl(constraint_labels_path, AnnotationRecord)
        overlap = read_jsonl(overlap_labels_path, AnnotationRecord)
    except (OSError, ValueError):
        raise ValueError("private annotations are invalid") from None
    if (
        {record.query_id for record in type_domain} != set(type_domain_ids)
        or len(type_domain) != len(type_domain_ids)
        or {record.query_id for record in constraints} != set(constraint_ids)
        or len(constraints) != len(constraint_ids)
        or {record.query_id for record in overlap} != set(overlap_ids)
        or len(overlap) != len(overlap_ids)
    ):
        raise ValueError("private annotations are invalid")
    overlap_set = set(overlap_ids)
    first_rater = [record for record in constraints if record.query_id in overlap_set]
    agreement = compare_annotations(
        first_rater,
        overlap,
        fields=("query_type", "domain"),
    )
    if any(not field.accepted for field in agreement.fields.values()):
        raise ValueError("human annotation agreement is below threshold")
    return type_domain, constraints, overlap, agreement


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
        "random_seed": RANDOM_SEED,
        "sampling_algorithm": SAMPLING_ALGORITHM,
        "status": PREPARED_STATUS,
    }
    if any(manifest.get(key) != expected for key, expected in expected_identity.items()):
        raise ValueError("prepared data is invalid")

    source_file_count = _validate_source_files(data_root, manifest.get("source_files"))
    partition_audits, frozen_partitions, partition_ids = _validate_partitions(
        data_root,
        manifest.get("partitions"),
        policies,
    )
    type_domain_ids, constraint_ids, overlap_ids = _validate_work_packages(
        data_root,
        manifest.get("work_packages"),
        partition_ids,
    )
    type_domain, constraints, overlap, agreement = _private_records(
        type_domain_labels_path,
        constraint_labels_path,
        overlap_labels_path,
        type_domain_ids,
        constraint_ids,
        overlap_ids,
    )
    try:
        type_domain_bytes = type_domain_labels_path.read_bytes()
        constraint_bytes = constraint_labels_path.read_bytes()
        overlap_bytes = overlap_labels_path.read_bytes()
    except OSError:
        raise ValueError("private annotations are invalid") from None

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
    frozen_manifest = manifest.copy()
    frozen_manifest["status"] = "frozen"
    frozen_manifest["prepared_manifest_sha256"] = prepared_manifest_hash
    frozen_manifest["partitions"] = frozen_partitions
    return FreezeAuditResult(
        prepared_manifest_bytes=manifest_bytes,
        frozen_manifest_payload=frozen_manifest,
        report=report,
    )
