"""Source-aware train/development/test isolation contracts."""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr, UnitFloat


DatasetRole = Literal["training", "development", "final_test"]
IsolationIssueCode = Literal[
    "exact_query_overlap",
    "near_duplicate_query",
    "gold_paper_overlap",
    "source_component_overlap",
]


def _normalized_query(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def _query_tokens(value: str) -> frozenset[str]:
    return frozenset(_normalized_query(value).split())


class DatasetExample(DomainModel):
    query_id: NonEmptyStr
    query: NonEmptyStr
    gold_paper_ids: list[NonEmptyStr] = Field(default_factory=list)
    source_components: list[NonEmptyStr] = Field(default_factory=list)


class DatasetPartition(DomainModel):
    dataset: NonEmptyStr
    split: NonEmptyStr
    role: DatasetRole
    revision: NonEmptyStr
    examples: list[DatasetExample] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_test_role_and_ids(self) -> DatasetPartition:
        split_tokens = set(re.split(r"[^a-z0-9]+", self.split.casefold()))
        if "test" in split_tokens and self.role != "final_test":
            raise ValueError("official test partitions must be final_test")
        query_ids = [example.query_id for example in self.examples]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query IDs must be unique within a dataset partition")
        return self

    @property
    def identity(self) -> str:
        return f"{self.dataset}/{self.split}"


class DatasetRoleRegistry(DomainModel):
    partitions: list[DatasetPartition]

    @model_validator(mode="after")
    def validate_partition_identities(self) -> DatasetRoleRegistry:
        identities = [partition.identity for partition in self.partitions]
        if len(identities) != len(set(identities)):
            raise ValueError("dataset partition identities must be unique")
        return self

    def partition_counts(self) -> dict[str, int]:
        return {
            partition.identity: len(partition.examples)
            for partition in sorted(self.partitions, key=lambda item: item.identity)
        }


class DatasetIsolationIssue(DomainModel):
    code: IsolationIssueCode
    left_partition: NonEmptyStr
    left_query_id: NonEmptyStr
    right_partition: NonEmptyStr
    right_query_id: NonEmptyStr
    shared_values: list[NonEmptyStr] = Field(default_factory=list)


class DatasetIsolationReport(DomainModel):
    safe_for_training: bool
    partition_counts: dict[NonEmptyStr, int]
    role_counts: dict[DatasetRole, int]
    issues: list[DatasetIsolationIssue] = Field(default_factory=list)


def _similarity(left: str, right: str) -> float:
    left_tokens = _query_tokens(left)
    right_tokens = _query_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def audit_dataset_isolation(
    registry: DatasetRoleRegistry,
    *,
    near_duplicate_threshold: UnitFloat = 0.9,
) -> DatasetIsolationReport:
    """Find cross-role leakage while preserving per-source partition counts."""
    registry = DatasetRoleRegistry.model_validate(registry)
    entries = [
        (partition, example)
        for partition in registry.partitions
        for example in partition.examples
    ]
    issues: list[DatasetIsolationIssue] = []
    for (left_partition, left), (right_partition, right) in combinations(entries, 2):
        if left_partition.role == right_partition.role:
            continue
        common = {
            "gold_paper_overlap": sorted(
                set(left.gold_paper_ids).intersection(right.gold_paper_ids)
            ),
            "source_component_overlap": sorted(
                set(left.source_components).intersection(right.source_components)
            ),
        }
        left_normalized = _normalized_query(left.query)
        right_normalized = _normalized_query(right.query)
        if left_normalized == right_normalized:
            common["exact_query_overlap"] = [left_normalized]
        elif _similarity(left.query, right.query) >= near_duplicate_threshold:
            common["near_duplicate_query"] = [left_normalized, right_normalized]
        for raw_code, values in common.items():
            if not values:
                continue
            issues.append(
                DatasetIsolationIssue(
                    code=raw_code,
                    left_partition=left_partition.identity,
                    left_query_id=left.query_id,
                    right_partition=right_partition.identity,
                    right_query_id=right.query_id,
                    shared_values=values,
                )
            )
    issues.sort(
        key=lambda item: (
            item.left_partition,
            item.left_query_id,
            item.right_partition,
            item.right_query_id,
            item.code,
        )
    )
    role_counts = Counter(
        partition.role
        for partition in registry.partitions
        for _ in partition.examples
    )
    roles: tuple[DatasetRole, ...] = (
        "training",
        "development",
        "final_test",
    )
    return DatasetIsolationReport(
        safe_for_training=not issues,
        partition_counts=registry.partition_counts(),
        role_counts={role: role_counts.get(role, 0) for role in roles},
        issues=issues,
    )


def assert_training_safe(report: DatasetIsolationReport) -> None:
    report = DatasetIsolationReport.model_validate(report)
    if not report.safe_for_training or report.issues:
        raise ValueError("dataset isolation audit failed")


def load_dataset_role_registry(path: Path) -> DatasetRoleRegistry:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("invalid dataset role registry") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("partitions"), list):
        raise ValueError("dataset role registry must contain a partitions list")
    return DatasetRoleRegistry.model_validate(raw)


__all__ = [
    "DatasetExample",
    "DatasetIsolationIssue",
    "DatasetIsolationReport",
    "DatasetPartition",
    "DatasetRole",
    "DatasetRoleRegistry",
    "assert_training_safe",
    "audit_dataset_isolation",
    "load_dataset_role_registry",
]
