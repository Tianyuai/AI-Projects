"""Deterministic CPU-only action ranker implementing the policy scorer protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from pydantic import Field

from paper_search.domain.models import DomainModel, Sha256, UnitFloat
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.action_labels import ActionWeakLabel
from paper_search.learning.contracts import PolicyActionCandidate, QueryPolicyInput
from paper_search.learning.cpu_baseline import (
    BinaryMetrics,
    evaluate_probabilities,
    select_f1_threshold,
)


class CpuActionExperimentSummary(DomainModel):
    schema_version: str = "cpu-action-ranker-experiment-v1"
    model_id: str = "cpu-action-ranker-v1"
    train_count: int = Field(strict=True, ge=0)
    development_count: int = Field(strict=True, ge=0)
    development_query_count: int = Field(strict=True, ge=0)
    dimension: int = Field(strict=True, gt=0)
    epochs: int = Field(strict=True, gt=0)
    seed: int
    threshold_selected_on: str = "development"
    learned: BinaryMetrics
    anchor_only: BinaryMetrics
    rule_all_candidates: BinaryMetrics
    learned_non_anchor: BinaryMetrics
    anchor_non_anchor: BinaryMetrics
    learned_top1_positive_rate: UnitFloat
    anchor_top1_positive_rate: UnitFloat
    train_labels_sha256: Sha256
    development_labels_sha256: Sha256
    model_sha256: Sha256


def _index(value: str, dimension: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _feature_indexes(
    *,
    query_kind: str,
    action: PolicyActionCandidate,
    dimension: int,
) -> tuple[int, ...]:
    text = action.text.casefold()
    terms = text.split()
    values = {
        "bias",
        f"kind={query_kind}",
        f"type={action.action_type}",
        f"origin={action.origin}",
        f"kind-type={query_kind}:{action.action_type}",
        f"term-count={min(len(terms), 20)}",
        f"char-count={min(len(text), 120) // 10}",
    }
    for term in terms:
        values.add(f"term={term}")
        values.add(f"prefix3={term[:3]}")
        values.add(f"suffix3={term[-3:]}")
    values.update(
        f"bigram={left}_{right}" for left, right in zip(terms, terms[1:])
    )
    return tuple(sorted({_index(value, dimension) for value in values}))


class CpuActionRanker:
    model_id = "cpu-action-ranker-v1"

    def __init__(
        self,
        *,
        dimension: int = 8192,
        epochs: int = 8,
        learning_rate: float = 0.12,
        l2: float = 1e-6,
        seed: int = 17,
        confidence_threshold: float = 0.5,
    ) -> None:
        if dimension <= 0 or epochs <= 0:
            raise ValueError("dimension and epochs must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be between zero and one")
        self.dimension = dimension
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.seed = seed
        self.confidence_threshold = confidence_threshold
        self.weights = np.zeros(dimension, dtype=np.float64)

    def fit(self, rows: list[ActionWeakLabel]) -> None:
        validated = [ActionWeakLabel.model_validate(row) for row in rows]
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(validated))
        step = 0
        for _ in range(self.epochs):
            rng.shuffle(order)
            for row_index in order:
                row = validated[int(row_index)]
                indexes = _feature_indexes(
                    query_kind=row.query_kind,
                    action=row.action,
                    dimension=self.dimension,
                )
                score = float(self.weights[list(indexes)].sum())
                probability = 1.0 / (
                    1.0 + math.exp(-max(-30.0, min(30.0, score)))
                )
                target = 1.0 if row.label == "positive" else 0.0
                rate = self.learning_rate / math.sqrt(1.0 + step / 1000.0)
                gradient = target - probability
                for index in indexes:
                    self.weights[index] = (
                        self.weights[index] * (1.0 - rate * self.l2)
                        + rate * gradient
                    )
                step += 1

    def score(
        self,
        request: QueryPolicyInput,
        candidates: list[PolicyActionCandidate],
    ) -> list[float]:
        request = QueryPolicyInput.model_validate(request)
        probabilities: list[float] = []
        for raw_candidate in candidates:
            candidate = PolicyActionCandidate.model_validate(raw_candidate)
            indexes = _feature_indexes(
                query_kind=request.query_kind,
                action=candidate,
                dimension=self.dimension,
            )
            score = float(self.weights[list(indexes)].sum())
            probabilities.append(
                1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
            )
        return probabilities

    def predict_label_probabilities(
        self, rows: list[ActionWeakLabel]
    ) -> list[float]:
        probabilities: list[float] = []
        for raw_row in rows:
            row = ActionWeakLabel.model_validate(raw_row)
            indexes = _feature_indexes(
                query_kind=row.query_kind,
                action=row.action,
                dimension=self.dimension,
            )
            score = float(self.weights[list(indexes)].sum())
            probabilities.append(
                1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
            )
        return probabilities

    def save(self, path: Path) -> None:
        write_frozen_bytes(path, self.weights.astype("<f8", copy=False).tobytes())

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        dimension: int,
        confidence_threshold: float,
    ) -> CpuActionRanker:
        content = path.read_bytes()
        expected_bytes = dimension * np.dtype("<f8").itemsize
        if len(content) != expected_bytes:
            raise ValueError("CPU action ranker weight size mismatch")
        ranker = cls(
            dimension=dimension,
            confidence_threshold=confidence_threshold,
        )
        ranker.weights = np.frombuffer(content, dtype="<f8").astype(
            np.float64, copy=True
        )
        return ranker


def _load_labels(path: Path) -> tuple[list[ActionWeakLabel], bytes]:
    content = path.read_bytes()
    rows = [
        ActionWeakLabel.model_validate(json.loads(line))
        for line in content.decode("utf-8").splitlines()
    ]
    if not rows:
        raise ValueError(f"action label file is empty: {path}")
    return rows, content


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _top1_positive_rate(
    rows: list[ActionWeakLabel], scores: list[float]
) -> float:
    grouped: dict[str, list[tuple[ActionWeakLabel, float]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        grouped[row.query_id].append((row, score))
    positives = 0
    for candidates in grouped.values():
        top = max(
            candidates,
            key=lambda item: (
                item[1],
                item[0].action.origin == "original_query",
                item[0].action.action_id,
            ),
        )
        positives += top[0].label == "positive"
    return positives / len(grouped) if grouped else 0.0


def run_cpu_action_experiment(
    *,
    train_path: Path,
    development_path: Path,
    model_path: Path,
    result_path: Path,
    dimension: int = 8192,
    epochs: int = 8,
    seed: int = 17,
) -> CpuActionExperimentSummary:
    train, train_content = _load_labels(train_path)
    development, development_content = _load_labels(development_path)
    if any(row.role != "training" for row in train):
        raise ValueError("training action labels contain a non-training role")
    if any(row.role != "development" for row in development):
        raise ValueError("development action labels contain a non-development role")
    ranker = CpuActionRanker(
        dimension=dimension,
        epochs=epochs,
        seed=seed,
    )
    ranker.fit(train)
    learned_scores = ranker.predict_label_probabilities(development)
    labels = [row.label == "positive" for row in development]
    threshold = select_f1_threshold(labels, learned_scores)
    anchor_scores = [
        1.0 if row.action.origin == "original_query" else 0.0
        for row in development
    ]
    rule_scores = [
        1.0 if row.action.origin == "original_query" else 0.5
        for row in development
    ]
    non_anchor_indexes = [
        index
        for index, row in enumerate(development)
        if row.action.origin != "original_query"
    ]
    non_anchor_labels = [labels[index] for index in non_anchor_indexes]
    non_anchor_scores = [learned_scores[index] for index in non_anchor_indexes]
    non_anchor_threshold = select_f1_threshold(
        non_anchor_labels, non_anchor_scores
    )
    ranker.save(model_path)
    model_content = model_path.read_bytes()
    summary = CpuActionExperimentSummary(
        train_count=len(train),
        development_count=len(development),
        development_query_count=len({row.query_id for row in development}),
        dimension=dimension,
        epochs=epochs,
        seed=seed,
        learned=evaluate_probabilities(
            labels, learned_scores, threshold=threshold
        ),
        anchor_only=evaluate_probabilities(
            labels, anchor_scores, threshold=0.5
        ),
        rule_all_candidates=evaluate_probabilities(
            labels, rule_scores, threshold=0.5
        ),
        learned_non_anchor=evaluate_probabilities(
            non_anchor_labels,
            non_anchor_scores,
            threshold=non_anchor_threshold,
        ),
        anchor_non_anchor=evaluate_probabilities(
            non_anchor_labels,
            [0.0] * len(non_anchor_labels),
            threshold=0.5,
        ),
        learned_top1_positive_rate=_top1_positive_rate(
            development, learned_scores
        ),
        anchor_top1_positive_rate=_top1_positive_rate(
            development, anchor_scores
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
    "CpuActionExperimentSummary",
    "CpuActionRanker",
    "run_cpu_action_experiment",
]
