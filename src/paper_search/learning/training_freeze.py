"""Deterministic PaSa training freeze with cross-role contamination removal."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, Sha256
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.evaluation.official_adapter import PaSaRecord, adapt_pasa_record
from paper_search.learning.data_isolation import (
    DatasetExample,
    DatasetIsolationIssue,
    DatasetPartition,
    DatasetRole,
    DatasetRoleRegistry,
)


PASA_TRAINING_EXPECTED_COUNTS = {
    "AutoScholarQuery/train.jsonl": 33_551,
    "AutoScholarQuery/dev.jsonl": 1_000,
    "AutoScholarQuery/test.jsonl": 1_000,
    "RealScholarQuery/test.jsonl": 50,
}
_PARTITION_BINDINGS: tuple[
    tuple[str, str, str, DatasetRole], ...
] = (
    ("AutoScholarQuery/train.jsonl", "pasa", "auto_train", "training"),
    ("AutoScholarQuery/dev.jsonl", "pasa", "auto_dev", "development"),
    ("AutoScholarQuery/test.jsonl", "pasa", "auto_test", "final_test"),
    ("RealScholarQuery/test.jsonl", "pasa", "real_test", "final_test"),
)


def _pasa_scope() -> list[Literal["pasa"]]:
    return ["pasa"]


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(payloads: list[dict[str, object]]) -> bytes:
    return (
        "".join(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for payload in payloads
        )
    ).encode("utf-8")


class FrozenFile(DomainModel):
    path: NonEmptyStr
    byte_count: int = Field(strict=True, ge=0)
    row_count: int = Field(strict=True, ge=0)
    sha256: Sha256


class TrainingFreezeManifest(DomainModel):
    schema_version: Literal["query-policy-training-freeze-v1"] = (
        "query-policy-training-freeze-v1"
    )
    pasa_revision: NonEmptyStr
    datasets_in_scope: list[Literal["pasa"]] = Field(default_factory=_pasa_scope)
    asta_access_status: Literal[
        "authorized", "unauthorized", "not_requested", "excluded_by_scope"
    ]
    source_counts: dict[NonEmptyStr, int]
    frozen_counts: dict[NonEmptyStr, int]
    exclusion_policy: Literal["role-priority-direct-cross-role-v2"] = (
        "role-priority-direct-cross-role-v2"
    )
    training_isolation_verified: Literal[True] = True
    excluded_training_count: int = Field(strict=True, ge=0)
    excluded_development_count: int = Field(strict=True, ge=0)
    final_cross_role_issue_count: Literal[0] = 0
    transitive_component_warning_count: int = Field(strict=True, ge=0)
    excluded_training_ids_sha256: Sha256
    excluded_development_ids_sha256: Sha256
    isolation_issue_counts: dict[NonEmptyStr, int]
    source_files: dict[NonEmptyStr, FrozenFile]
    partition_files: dict[NonEmptyStr, FrozenFile]
    isolation_report: FrozenFile


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _normalized_query(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def _tokens(value: str) -> frozenset[str]:
    return frozenset(_normalized_query(value).split())


def _parse_records(content: bytes, source_path: str) -> list[PaSaRecord]:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise ValueError(f"{source_path}: invalid UTF-8") from None
    records: list[PaSaRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{source_path}:{line_number}: blank line")
        try:
            raw = json.loads(line)
            record = PaSaRecord.model_validate(raw)
        except (json.JSONDecodeError, ValueError):
            raise ValueError(
                f"{source_path}:{line_number}: invalid PaSa record"
            ) from None
        if record.qid in seen:
            raise ValueError(f"{source_path}: duplicate query ID: {record.qid}")
        seen.add(record.qid)
        records.append(record)
    return records


def _partition_from_records(
    records: list[PaSaRecord],
    *,
    dataset: str,
    split: str,
    role: DatasetRole,
    revision: str,
) -> DatasetPartition:
    source = "RealScholarQuery" if split == "real_test" else "AutoScholarQuery"
    examples: list[DatasetExample] = []
    for record in records:
        adapted = adapt_pasa_record(
            record,
            source=source,
            split=split,
            revision=revision,
        )
        examples.append(
            DatasetExample(
                query_id=adapted.query_id,
                query=adapted.query,
                gold_paper_ids=adapted.relevant_paper_ids,
            )
        )
    return DatasetPartition(
        dataset=dataset,
        split=split,
        role=role,
        revision=revision,
        examples=examples,
    )


def _indexed_isolation(
    registry: DatasetRoleRegistry,
    *,
    near_duplicate_threshold: float = 0.9,
) -> tuple[
    list[DatasetIsolationIssue],
    set[tuple[str, str]],
    set[tuple[str, str]],
]:
    entries = [
        (partition, example)
        for partition in registry.partitions
        for example in partition.examples
    ]
    disjoint = _DisjointSet(len(entries))
    issues: list[DatasetIsolationIssue] = []
    issue_keys: set[tuple[str, int, int]] = set()

    def add_group(code: str, values: dict[str, list[int]]) -> None:
        for shared_value, indexes in values.items():
            if len(indexes) < 2:
                continue
            anchor = indexes[0]
            for index in indexes[1:]:
                disjoint.union(anchor, index)
            for offset, left in enumerate(indexes):
                for right in indexes[offset + 1 :]:
                    left_partition, left_example = entries[left]
                    right_partition, right_example = entries[right]
                    if left_partition.role == right_partition.role:
                        continue
                    ordered = (min(left, right), max(left, right))
                    key = (code, *ordered)
                    if key in issue_keys:
                        continue
                    issue_keys.add(key)
                    issues.append(
                        DatasetIsolationIssue(
                            code=code,
                            left_partition=left_partition.identity,
                            left_query_id=left_example.query_id,
                            right_partition=right_partition.identity,
                            right_query_id=right_example.query_id,
                            shared_values=[shared_value],
                        )
                    )

    exact: dict[str, list[int]] = defaultdict(list)
    gold: dict[str, list[int]] = defaultdict(list)
    source: dict[str, list[int]] = defaultdict(list)
    token_sets: list[frozenset[str]] = []
    for index, (_, example) in enumerate(entries):
        exact[_normalized_query(example.query)].append(index)
        for paper_id in sorted(set(example.gold_paper_ids)):
            gold[paper_id].append(index)
        for component in sorted(set(example.source_components)):
            source[component].append(index)
        token_sets.append(_tokens(example.query))
    add_group("exact_query_overlap", exact)
    add_group("gold_paper_overlap", gold)
    add_group("source_component_overlap", source)

    token_frequency = Counter(token for tokens in token_sets for token in tokens)
    prefix_index: dict[str, list[int]] = defaultdict(list)
    for right, tokens in enumerate(token_sets):
        if not tokens:
            continue
        ordered_tokens = sorted(tokens, key=lambda token: (token_frequency[token], token))
        prefix_length = len(tokens) - math.ceil(near_duplicate_threshold * len(tokens)) + 1
        candidates: set[int] = set()
        for token in ordered_tokens[:prefix_length]:
            candidates.update(prefix_index[token])
        for left in candidates:
            left_partition, left_example = entries[left]
            right_partition, right_example = entries[right]
            if left_partition.role == right_partition.role:
                continue
            left_tokens = token_sets[left]
            if not left_tokens:
                continue
            similarity = len(left_tokens & tokens) / len(left_tokens | tokens)
            if similarity < near_duplicate_threshold:
                continue
            if _normalized_query(left_example.query) == _normalized_query(right_example.query):
                continue
            key = ("near_duplicate_query", left, right)
            if key in issue_keys:
                continue
            issue_keys.add(key)
            disjoint.union(left, right)
            issues.append(
                DatasetIsolationIssue(
                    code="near_duplicate_query",
                    left_partition=left_partition.identity,
                    left_query_id=left_example.query_id,
                    right_partition=right_partition.identity,
                    right_query_id=right_example.query_id,
                    shared_values=[
                        _normalized_query(left_example.query),
                        _normalized_query(right_example.query),
                    ],
                )
            )
        for token in ordered_tokens[:prefix_length]:
            prefix_index[token].append(right)

    roles_by_root: dict[int, set[DatasetRole]] = defaultdict(set)
    for index, (partition, _) in enumerate(entries):
        roles_by_root[disjoint.find(index)].add(partition.role)
    component_candidates = {
        (partition.identity, example.query_id)
        for index, (partition, example) in enumerate(entries)
        if partition.role == "training"
        and roles_by_root[disjoint.find(index)].difference({"training"})
    }
    training_partitions = {
        partition.identity
        for partition in registry.partitions
        if partition.role == "training"
    }
    excluded: set[tuple[str, str]] = set()
    for issue in issues:
        if issue.left_partition in training_partitions:
            excluded.add((issue.left_partition, issue.left_query_id))
        if issue.right_partition in training_partitions:
            excluded.add((issue.right_partition, issue.right_query_id))
    transitive_warnings = component_candidates.difference(excluded)
    issues.sort(
        key=lambda item: (
            item.left_partition,
            item.left_query_id,
            item.right_partition,
            item.right_query_id,
            item.code,
            tuple(item.shared_values),
        )
    )
    return issues, excluded, transitive_warnings


def _file_record(path: str, content: bytes, row_count: int) -> FrozenFile:
    return FrozenFile(
        path=path,
        byte_count=len(content),
        row_count=row_count,
        sha256=_sha256(content),
    )


def freeze_pasa_training_data(
    *,
    raw_root: Path,
    private_output_root: Path,
    manifest_path: Path,
    revision: str,
    asta_access_status: Literal[
        "authorized", "unauthorized", "not_requested", "excluded_by_scope"
    ],
    expected_counts: dict[str, int] | None = None,
) -> TrainingFreezeManifest:
    """Freeze PaSa with final-test > development > training role priority."""
    expected = PASA_TRAINING_EXPECTED_COUNTS if expected_counts is None else expected_counts
    partitions: list[DatasetPartition] = []
    source_files: dict[str, FrozenFile] = {}
    for source_path, dataset, split, role in _PARTITION_BINDINGS:
        try:
            content = (raw_root / source_path).read_bytes()
        except OSError as error:
            raise ValueError(f"missing PaSa source file: {source_path}") from error
        records = _parse_records(content, source_path)
        required_count = expected.get(source_path)
        if required_count is None or len(records) != required_count:
            raise ValueError(
                f"{source_path}: expected {required_count}, found {len(records)}"
            )
        source_files[source_path] = _file_record(source_path, content, len(records))
        partitions.append(
            _partition_from_records(
                records,
                dataset=dataset,
                split=split,
                role=role,
                revision=revision,
            )
        )

    registry = DatasetRoleRegistry(partitions=partitions)
    issues, _, _ = _indexed_isolation(registry)
    development_partitions = {
        partition.identity
        for partition in partitions
        if partition.role == "development"
    }
    final_test_partitions = {
        partition.identity
        for partition in partitions
        if partition.role == "final_test"
    }
    excluded_development: set[tuple[str, str]] = set()
    for issue in issues:
        if (
            issue.left_partition in development_partitions
            and issue.right_partition in final_test_partitions
        ):
            excluded_development.add((issue.left_partition, issue.left_query_id))
        if (
            issue.right_partition in development_partitions
            and issue.left_partition in final_test_partitions
        ):
            excluded_development.add((issue.right_partition, issue.right_query_id))
    development_clean_partitions = [
        partition.model_copy(
            update={
                "examples": [
                    example
                    for example in partition.examples
                    if (partition.identity, example.query_id)
                    not in excluded_development
                ]
            }
        )
        for partition in partitions
    ]
    _, excluded_training, transitive_warnings = _indexed_isolation(
        DatasetRoleRegistry(partitions=development_clean_partitions)
    )
    clean_partitions = [
        partition.model_copy(
            update={
                "examples": [
                    example
                    for example in partition.examples
                    if (partition.identity, example.query_id) not in excluded_training
                ]
            }
        )
        for partition in development_clean_partitions
    ]
    final_issues, remaining_training_conflicts, _ = _indexed_isolation(
        DatasetRoleRegistry(partitions=clean_partitions)
    )
    if remaining_training_conflicts or final_issues:
        raise RuntimeError("final cross-role isolation postcondition failed")
    issue_counts = Counter(issue.code for issue in issues)
    partition_files: dict[str, FrozenFile] = {}
    frozen_counts: dict[str, int] = {}
    source_counts = registry.partition_counts()
    for partition in sorted(clean_partitions, key=lambda item: item.identity):
        examples = partition.examples
        rows = [
            {
                "dataset": partition.dataset,
                "split": partition.split,
                "role": partition.role,
                "revision": partition.revision,
                **example.model_dump(mode="json"),
            }
            for example in examples
        ]
        content = _jsonl_bytes(rows)
        relative = f"partitions/{partition.dataset}_{partition.split}.jsonl"
        write_frozen_bytes(private_output_root / relative, content)
        partition_files[partition.identity] = _file_record(
            relative, content, len(rows)
        )
        frozen_counts[partition.identity] = len(rows)

    issue_payload = [issue.model_dump(mode="json") for issue in issues]
    report_content = _json_bytes(
        {
            "schema_version": "query-policy-isolation-report-v1",
            "issues": issue_payload,
                "excluded_training": [
                    {"partition": partition, "query_id": query_id}
                    for partition, query_id in sorted(excluded_training)
                ],
                "excluded_development": [
                    {"partition": partition, "query_id": query_id}
                    for partition, query_id in sorted(excluded_development)
            ],
            "transitive_component_warnings": [
                {"partition": partition, "query_id": query_id}
                for partition, query_id in sorted(transitive_warnings)
            ],
        }
    )
    report_relative = "isolation-report.json"
    write_frozen_bytes(private_output_root / report_relative, report_content)
    excluded_bytes = _json_bytes(
        [
            {"partition": partition, "query_id": query_id}
            for partition, query_id in sorted(excluded_training)
        ]
    )
    excluded_development_bytes = _json_bytes(
        [
            {"partition": partition, "query_id": query_id}
            for partition, query_id in sorted(excluded_development)
        ]
    )
    manifest = TrainingFreezeManifest(
        pasa_revision=revision,
        asta_access_status=asta_access_status,
        source_counts=source_counts,
        frozen_counts=frozen_counts,
        excluded_training_count=len(excluded_training),
        excluded_development_count=len(excluded_development),
        transitive_component_warning_count=len(transitive_warnings),
        excluded_training_ids_sha256=_sha256(excluded_bytes),
        excluded_development_ids_sha256=_sha256(excluded_development_bytes),
        isolation_issue_counts=dict(sorted(issue_counts.items())),
        source_files=source_files,
        partition_files=partition_files,
        isolation_report=_file_record(report_relative, report_content, len(issues)),
    )
    write_frozen_bytes(
        manifest_path,
        _json_bytes(manifest.model_dump(mode="json")),
    )
    return manifest


__all__ = [
    "PASA_TRAINING_EXPECTED_COUNTS",
    "TrainingFreezeManifest",
    "freeze_pasa_training_data",
]
