"""Frozen task-slot labels and a bounded residual over an unchanged B0 ranker."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from paper_search.evaluation.predictions import paper_evaluation_id
from paper_search.learning.candidates import query_content_terms
from paper_search.learning.cpu_document_ranker import (
    DocumentCandidateEvidence,
    DocumentRankingQuery,
)
from paper_search.learning.query_constraint_annotations import query_sha256


_TASK_SLOT_FAMILIES = frozenset(
    {"task_slot_reliability", "task_slot_title", "task_slot_abstract"}
)
_TASK_SLOT_RELIABILITY_COMPONENTS = frozenset(
    {
        "existence_cardinality",
        "confidence",
        "provenance_status",
        "baseline_interaction",
    }
)
_STATUS_WEIGHTS = {
    "reviewed": 1.0,
    "reused_existing": 0.95,
    "base_accepted": 0.9,
    "runtime_deterministic": 0.85,
    "review_failed_fallback": 0.5,
    "unresolved": 0.0,
}


class BaselineDocumentRanker(Protocol):
    def rank(
        self,
        query: str,
        candidates: Sequence[DocumentCandidateEvidence],
    ) -> Sequence[DocumentCandidateEvidence]: ...


@dataclass(frozen=True)
class FrozenTaskValue:
    normalized_value: str
    confidence: float


@dataclass(frozen=True)
class FrozenTaskSlotLabel:
    query_id: str
    query_sha256: str
    role: str
    split: str
    tasks: tuple[FrozenTaskValue, ...]
    ambiguous_fields: tuple[str, ...]
    task_label_status: str

    @property
    def task_present(self) -> bool:
        return bool(self.tasks)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def multi_task(self) -> bool:
        return len(self.tasks) > 1

    @property
    def task_confidence_min(self) -> float:
        return min((task.confidence for task in self.tasks), default=0.0)

    @property
    def task_confidence_mean(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(task.confidence for task in self.tasks) / len(self.tasks)

    @property
    def task_ambiguous(self) -> bool:
        return any(field.casefold() in {"task", "tasks"} for field in self.ambiguous_fields)

    @property
    def missing_or_unresolved(self) -> bool:
        return self.task_label_status == "unresolved"

    @property
    def reliability_weight(self) -> float:
        if self.task_ambiguous or self.missing_or_unresolved or not self.tasks:
            return 0.0
        return _STATUS_WEIGHTS[self.task_label_status] * self.task_confidence_mean


class FrozenTaskSlotLabelStore:
    """Read-only query-hash lookup with an explicit train/dev role boundary."""

    def __init__(self, labels_by_hash: Mapping[str, FrozenTaskSlotLabel]) -> None:
        self._labels_by_hash = dict(labels_by_hash)

    @classmethod
    def from_jsonl(cls, *paths: Path) -> FrozenTaskSlotLabelStore:
        if not paths:
            raise ValueError("task-slot labels require at least one JSONL path")
        return cls.from_jsonl_bytes(*(path.read_bytes() for path in paths))

    @classmethod
    def from_jsonl_bytes(cls, *payloads: bytes) -> FrozenTaskSlotLabelStore:
        """Load frozen labels from verified bytes without reopening mutable paths."""

        if not payloads:
            raise ValueError("task-slot labels require at least one JSONL payload")
        labels: dict[str, FrozenTaskSlotLabel] = {}
        query_ids: set[str] = set()
        for payload_index, payload in enumerate(payloads, start=1):
            try:
                content = payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("invalid task-slot JSONL encoding") from error
            for line_number, raw_line in enumerate(content.splitlines(), start=1):
                if not raw_line.strip():
                    continue
                try:
                    raw = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "invalid task-slot JSONL at payload "
                        f"{payload_index}:{line_number}"
                    ) from error
                label = _parse_label(raw)
                if label.query_id in query_ids:
                    raise ValueError("task-slot query ids must be unique")
                if label.query_sha256 in labels:
                    raise ValueError("task-slot query hashes must be unique")
                query_ids.add(label.query_id)
                labels[label.query_sha256] = label
        return cls(labels)

    def for_scoring_query(self, query: str) -> FrozenTaskSlotLabel | None:
        return self._labels_by_hash.get(query_sha256(query))

    def for_training_query(self, query: str) -> FrozenTaskSlotLabel | None:
        label = self.for_scoring_query(query)
        if label is not None and (label.role != "training" or label.split != "auto_train"):
            raise ValueError("development task-slot label cannot be used during fit")
        return label

    def __len__(self) -> int:
        return len(self._labels_by_hash)


def _parse_label(raw: Mapping[str, Any]) -> FrozenTaskSlotLabel:
    required = {
        "query_id",
        "query_sha256",
        "role",
        "split",
        "tasks",
        "ambiguous_fields",
        "task_label_status",
    }
    if not required.issubset(raw):
        raise ValueError("task-slot label is missing required fields")
    status = str(raw["task_label_status"])
    if status not in _STATUS_WEIGHTS:
        raise ValueError(f"unsupported task-slot label status: {status}")
    role = str(raw["role"])
    split = str(raw["split"])
    if (role, split) not in {("training", "auto_train"), ("development", "auto_dev")}:
        raise ValueError("task-slot role and split are inconsistent")
    tasks: list[FrozenTaskValue] = []
    for item in raw["tasks"]:
        value = " ".join(str(item["normalized_value"]).split()).strip()
        confidence = float(item.get("confidence", 0.0))
        if not value or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("invalid frozen task value")
        tasks.append(FrozenTaskValue(value, confidence))
    return FrozenTaskSlotLabel(
        query_id=str(raw["query_id"]),
        query_sha256=str(raw["query_sha256"]),
        role=role,
        split=split,
        tasks=tuple(tasks),
        ambiguous_fields=tuple(str(value) for value in raw["ambiguous_fields"]),
        task_label_status=status,
    )


def _lexical_support(tasks: Sequence[FrozenTaskValue], text: str) -> float:
    if not tasks:
        return 0.0
    document_terms = set(query_content_terms(text))
    supports: list[float] = []
    for task in tasks:
        task_terms = set(query_content_terms(task.normalized_value))
        supports.append(
            len(task_terms & document_terms) / len(task_terms) if task_terms else 0.0
        )
    return sum(supports) / len(supports)


def task_slot_candidate_features(
    label: FrozenTaskSlotLabel | None,
    candidate: DocumentCandidateEvidence,
    *,
    baseline_rank: int,
    families: Set[str],
    reliability_components: Set[str] | None = None,
) -> dict[str, float]:
    """Return missing-safe task-slot features for one fixed B0 candidate."""

    enabled = frozenset(families)
    unsupported = enabled - _TASK_SLOT_FAMILIES
    if unsupported:
        raise ValueError(f"unsupported task-slot feature families: {sorted(unsupported)}")
    if baseline_rank <= 0:
        raise ValueError("task-slot baseline rank must be positive")
    components = (
        _TASK_SLOT_RELIABILITY_COMPONENTS
        if reliability_components is None
        else frozenset(reliability_components)
    )
    unsupported_components = components - _TASK_SLOT_RELIABILITY_COMPONENTS
    if unsupported_components:
        raise ValueError(
            "unsupported task-slot reliability components: "
            f"{sorted(unsupported_components)}"
        )
    if label is None or not enabled or label.reliability_weight <= 0.0:
        return {}
    reliability = label.reliability_weight
    values: dict[str, float] = {}
    if "existence_cardinality" in components:
        values.update(
            {
                "task-slot-present": float(label.task_present),
                "task-slot-count": min(label.task_count, 4) / 4.0,
                "task-slot-multi": float(label.multi_task),
            }
        )
    if "confidence" in components:
        values.update(
            {
                "task-slot-confidence-min": label.task_confidence_min,
                "task-slot-confidence-mean": label.task_confidence_mean,
                "task-slot-reliability": reliability,
            }
        )
    if "baseline_interaction" in components:
        values["task-slot-baseline-reciprocal"] = reliability / baseline_rank
    if "provenance_status" in components:
        values["task-slot-source-support"] = (
            reliability * min(candidate.support_count, 6) / 6.0
        )
        values[f"task-slot-status={label.task_label_status}"] = (
            reliability / baseline_rank
        )
    if "task_slot_title" in enabled:
        values["task-slot-title-match"] = reliability * _lexical_support(
            label.tasks, candidate.paper.title
        )
    if "task_slot_abstract" in enabled:
        abstract = candidate.paper.abstract or ""
        values["task-slot-abstract-match"] = reliability * _lexical_support(
            label.tasks, abstract
        )
        values["task-slot-abstract-missing"] = reliability * float(
            not abstract.strip()
        )
        values["task-slot-title-abstract-support"] = reliability * max(
            _lexical_support(label.tasks, candidate.paper.title),
            _lexical_support(label.tasks, abstract),
        )
    return values


def _index(name: str, dimension: int) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _hashed(values: Mapping[str, float], dimension: int) -> dict[int, float]:
    output: dict[int, float] = {}
    for name, value in values.items():
        index = _index(name, dimension)
        output[index] = output.get(index, 0.0) + value
    return output


def _score(weights: np.ndarray, values: Mapping[int, float]) -> float:
    return float(sum(weights[index] * value for index, value in values.items()))


class TaskSlotResidualRanker:
    """Learn only task-slot residual weights over a frozen B0 ordering."""

    model_id = "task-slot-residual-document-ranker-v1"

    def __init__(
        self,
        *,
        baseline_ranker: BaselineDocumentRanker,
        label_store: FrozenTaskSlotLabelStore,
        feature_families: Set[str],
        reliability_components: Set[str] | None = None,
        dimension: int = 2048,
        epochs: int = 12,
        learning_rate: float = 0.05,
        l2: float = 1e-6,
        maximum_residual: float = 0.35,
        hard_negative_limit: int = 100,
    ) -> None:
        families = frozenset(feature_families)
        unsupported = families - _TASK_SLOT_FAMILIES
        if not families or unsupported:
            raise ValueError(
                f"unsupported task-slot feature families: {sorted(unsupported)}"
            )
        components = (
            _TASK_SLOT_RELIABILITY_COMPONENTS
            if reliability_components is None
            else frozenset(reliability_components)
        )
        unsupported_components = components - _TASK_SLOT_RELIABILITY_COMPONENTS
        if not components or unsupported_components:
            raise ValueError(
                "unsupported task-slot reliability components: "
                f"{sorted(unsupported_components)}"
            )
        if dimension <= 0 or epochs <= 0 or hard_negative_limit <= 0:
            raise ValueError("task-slot ranker sizes must be positive")
        if learning_rate <= 0 or l2 < 0 or not 0 < maximum_residual <= 1:
            raise ValueError("invalid task-slot ranker optimization settings")
        self.baseline_ranker = baseline_ranker
        self.label_store = label_store
        self.feature_families = families
        self.reliability_components = components
        self.dimension = dimension
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.maximum_residual = maximum_residual
        self.hard_negative_limit = hard_negative_limit
        self.weights = np.zeros(dimension, dtype=np.float64)
        self.last_fit_query_count = 0

    def _b0_rows(
        self, query: str, candidates: Sequence[DocumentCandidateEvidence]
    ) -> list[DocumentCandidateEvidence]:
        input_rows = list(candidates)
        rows = list(self.baseline_ranker.rank(query, input_rows))
        input_ids = [paper_evaluation_id(row.paper) for row in input_rows]
        output_ids = [paper_evaluation_id(row.paper) for row in rows]
        if len(output_ids) != len(input_ids) or set(output_ids) != set(input_ids):
            raise ValueError("B0 ranker changed task-slot candidate identity")
        return rows

    def fit(self, queries: Sequence[DocumentRankingQuery]) -> int:
        pairs: list[tuple[dict[int, float], dict[int, float]]] = []
        used_queries = 0
        for raw_query in queries:
            query = DocumentRankingQuery.model_validate(raw_query)
            label = self.label_store.for_training_query(query.query)
            if label is None or label.reliability_weight <= 0.0:
                continue
            rows = self._b0_rows(query.query, query.candidates)
            gold = set(query.gold_paper_ids)
            positives: list[dict[int, float]] = []
            negatives: list[dict[int, float]] = []
            for rank, candidate in enumerate(rows, start=1):
                values = _hashed(
                    task_slot_candidate_features(
                        label,
                        candidate,
                        baseline_rank=rank,
                        families=self.feature_families,
                        reliability_components=self.reliability_components,
                    ),
                    self.dimension,
                )
                if paper_evaluation_id(candidate.paper) in gold:
                    positives.append(values)
                elif len(negatives) < self.hard_negative_limit:
                    negatives.append(values)
            query_pairs = [
                (positive, negative)
                for positive in positives
                for negative in negatives
                if positive != negative
            ]
            if query_pairs:
                used_queries += 1
                pairs.extend(query_pairs)
        for _epoch in range(self.epochs):
            for positive, negative in pairs:
                difference = dict(positive)
                for index, value in negative.items():
                    difference[index] = difference.get(index, 0.0) - value
                margin = _score(self.weights, difference)
                gradient_scale = 1.0 / (1.0 + math.exp(min(60.0, margin)))
                for index, value in difference.items():
                    self.weights[index] += self.learning_rate * (
                        gradient_scale * value - self.l2 * self.weights[index]
                    )
        self.last_fit_query_count = used_queries
        return len(pairs)

    def rank(
        self,
        query: str,
        candidates: Sequence[DocumentCandidateEvidence],
    ) -> list[DocumentCandidateEvidence]:
        rows = self._b0_rows(query, candidates)
        label = self.label_store.for_scoring_query(query)
        if label is None or label.reliability_weight <= 0.0 or not rows:
            return rows
        divisor = max(1, len(rows) - 1)
        combined: list[float] = []
        for rank, candidate in enumerate(rows, start=1):
            values = _hashed(
                task_slot_candidate_features(
                    label,
                    candidate,
                    baseline_rank=rank,
                    families=self.feature_families,
                    reliability_components=self.reliability_components,
                ),
                self.dimension,
            )
            residual = self.maximum_residual * math.tanh(_score(self.weights, values))
            combined.append(1.0 - (rank - 1) / divisor + residual)
        return [
            rows[index]
            for index in sorted(
                range(len(rows)), key=lambda index: (-combined[index], index)
            )
        ]

    def deployment_manifest_fields(self) -> dict[str, object]:
        return {
            "schema_version": "task-slot-residual-document-ranker-manifest-v1",
            "model_id": self.model_id,
            "feature_families": sorted(self.feature_families),
            "reliability_components": sorted(self.reliability_components),
            "dimension": self.dimension,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "maximum_residual": self.maximum_residual,
            "hard_negative_limit": self.hard_negative_limit,
            "training_query_count_with_usable_task_slot": self.last_fit_query_count,
            "development_labels_used_for_training": False,
        }

    def weights_bytes(self) -> bytes:
        return self.weights.astype("<f8", copy=False).tobytes(order="C")


__all__ = [
    "FrozenTaskSlotLabel",
    "FrozenTaskSlotLabelStore",
    "FrozenTaskValue",
    "TaskSlotResidualRanker",
    "task_slot_candidate_features",
]
