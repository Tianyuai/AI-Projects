"""Deterministic CPU-only baseline for query-term retention."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable
from pathlib import Path

import numpy as np
from pydantic import Field

from paper_search.domain.models import DomainModel, UnitFloat
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.weak_labels import QueryTermLabel


class BinaryMetrics(DomainModel):
    threshold: UnitFloat
    accuracy: UnitFloat
    precision: UnitFloat
    recall: UnitFloat
    f1: UnitFloat
    positive_prediction_rate: UnitFloat
    evaluated_count: int = Field(strict=True, ge=0)


class CpuBaselineSummary(DomainModel):
    schema_version: str = "query-term-cpu-baseline-v1"
    model_id: str = "hashed-logistic-term-ranker-v1"
    train_count: int = Field(strict=True, ge=0)
    development_count: int = Field(strict=True, ge=0)
    dimension: int = Field(strict=True, gt=0)
    epochs: int = Field(strict=True, gt=0)
    seed: int
    threshold_selected_on: str = "development"
    learned: BinaryMetrics
    all_negative: BinaryMetrics
    length_rule: BinaryMetrics
    train_labels_sha256: str
    development_labels_sha256: str
    model_sha256: str


def _stable_index(value: str, dimension: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _features(row: QueryTermLabel, dimension: int) -> tuple[int, ...]:
    term = row.action_text.casefold()
    values = {
        "bias",
        f"term={term}",
        f"prefix2={term[:2]}",
        f"prefix3={term[:3]}",
        f"suffix2={term[-2:]}",
        f"suffix3={term[-3:]}",
        f"length={min(len(term), 15)}",
        f"position={min(row.query_term_index, 12)}",
    }
    values.update(
        f"char3={term[index:index + 3]}"
        for index in range(max(0, len(term) - 2))
    )
    return tuple(sorted({_stable_index(value, dimension) for value in values}))


@dataclass
class HashedLogisticTermRanker:
    dimension: int = 4096
    epochs: int = 8
    learning_rate: float = 0.15
    l2: float = 1e-6
    seed: int = 17

    def __post_init__(self) -> None:
        if self.dimension <= 0 or self.epochs <= 0:
            raise ValueError("dimension and epochs must be positive")
        self.weights = np.zeros(self.dimension, dtype=np.float64)

    def fit(self, rows: list[QueryTermLabel]) -> None:
        validated = [QueryTermLabel.model_validate(row) for row in rows]
        rng = np.random.default_rng(self.seed)
        indexes = np.arange(len(validated))
        step = 0
        for _ in range(self.epochs):
            rng.shuffle(indexes)
            for row_index in indexes:
                row = validated[int(row_index)]
                feature_indexes = _features(row, self.dimension)
                score = float(self.weights[list(feature_indexes)].sum())
                probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
                target = 1.0 if row.label == "positive" else 0.0
                rate = self.learning_rate / math.sqrt(1.0 + step / 1000.0)
                gradient = target - probability
                for feature_index in feature_indexes:
                    self.weights[feature_index] = (
                        self.weights[feature_index] * (1.0 - rate * self.l2)
                        + rate * gradient
                    )
                step += 1

    def predict_proba(self, rows: list[QueryTermLabel]) -> list[float]:
        result: list[float] = []
        for raw in rows:
            row = QueryTermLabel.model_validate(raw)
            score = float(self.weights[list(_features(row, self.dimension))].sum())
            result.append(
                1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
            )
        return result


def evaluate_probabilities(
    labels: Iterable[bool],
    probabilities: Iterable[float],
    *,
    threshold: float,
) -> BinaryMetrics:
    pairs = list(zip(labels, probabilities, strict=True))
    true_positive = sum(label and score >= threshold for label, score in pairs)
    false_positive = sum(not label and score >= threshold for label, score in pairs)
    false_negative = sum(label and score < threshold for label, score in pairs)
    true_negative = len(pairs) - true_positive - false_positive - false_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        threshold=threshold,
        accuracy=(true_positive + true_negative) / len(pairs) if pairs else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        positive_prediction_rate=(true_positive + false_positive) / len(pairs) if pairs else 0.0,
        evaluated_count=len(pairs),
    )


def select_f1_threshold(labels: list[bool], probabilities: list[float]) -> float:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be non-empty and aligned")
    candidates = sorted(set(probabilities), reverse=True)
    scored = [
        (evaluate_probabilities(labels, probabilities, threshold=value).f1, value)
        for value in candidates
    ]
    return max(scored, key=lambda item: (item[0], item[1]))[1]


def _load_labels(path: Path) -> tuple[list[QueryTermLabel], bytes]:
    content = path.read_bytes()
    rows = [
        QueryTermLabel.model_validate(json.loads(line))
        for line in content.decode("utf-8").splitlines()
    ]
    if not rows:
        raise ValueError(f"label file is empty: {path}")
    return rows, content


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def run_cpu_baseline_experiment(
    *,
    train_path: Path,
    development_path: Path,
    result_path: Path,
    model_path: Path,
    dimension: int = 4096,
    epochs: int = 8,
    seed: int = 17,
) -> CpuBaselineSummary:
    train, train_content = _load_labels(train_path)
    development, development_content = _load_labels(development_path)
    if any(row.role != "training" for row in train):
        raise ValueError("training label file contains a non-training role")
    if any(row.role != "development" for row in development):
        raise ValueError("development label file contains a non-development role")
    ranker = HashedLogisticTermRanker(
        dimension=dimension,
        epochs=epochs,
        seed=seed,
    )
    ranker.fit(train)
    probabilities = ranker.predict_proba(development)
    labels = [row.label == "positive" for row in development]
    threshold = select_f1_threshold(labels, probabilities)
    model_content = ranker.weights.astype("<f8", copy=False).tobytes()
    write_frozen_bytes(model_path, model_content)
    summary = CpuBaselineSummary(
        train_count=len(train),
        development_count=len(development),
        dimension=dimension,
        epochs=epochs,
        seed=seed,
        learned=evaluate_probabilities(labels, probabilities, threshold=threshold),
        all_negative=evaluate_probabilities(
            labels, [0.0] * len(labels), threshold=1.0
        ),
        length_rule=evaluate_probabilities(
            labels,
            [1.0 if len(row.action_text) >= 8 else 0.0 for row in development],
            threshold=0.5,
        ),
        train_labels_sha256=_sha256(train_content),
        development_labels_sha256=_sha256(development_content),
        model_sha256=_sha256(model_content),
    )
    result_content = (
        json.dumps(
            summary.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    write_frozen_bytes(result_path, result_content)
    return summary


__all__ = [
    "BinaryMetrics",
    "CpuBaselineSummary",
    "HashedLogisticTermRanker",
    "evaluate_probabilities",
    "run_cpu_baseline_experiment",
    "select_f1_threshold",
]
