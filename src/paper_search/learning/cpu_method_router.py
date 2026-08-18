"""Deterministic CPU binary router for optional retrieval methods."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.candidates import query_content_terms
from paper_search.learning.method_route_labels import MethodName, MethodRouteLabel
from paper_search.learning.routing import RuleQueryRouter


_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _index(value: str, dimension: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _feature_indexes(
    *,
    method: MethodName,
    query: str,
    seed_count: int,
    dimension: int,
) -> tuple[int, ...]:
    normalized = " ".join(query.casefold().split())
    terms = query_content_terms(normalized)[:24]
    kind = RuleQueryRouter().route(query).query_kind
    seed_bucket = min(seed_count, 10)
    has_quote = '"' in normalized or "'" in normalized
    values = {
        "bias",
        f"method={method}",
        f"kind={kind}",
        f"method-kind={method}:{kind}",
        f"term-count={min(len(terms), 20)}",
        f"char-length={min(len(normalized) // 10, 20)}",
        f"has-year={bool(_YEAR.search(normalized))}",
        f"has-quote={has_quote}",
    }
    if method == "graph":
        values.add(f"seed-count={seed_bucket}")
        values.add(f"seed-present={seed_count > 0}")
    for term in terms:
        values.add(f"term={term}")
        values.add(f"method-term={method}:{term}")
        for index in range(max(0, len(term) - 2)):
            values.add(f"char3={term[index:index + 3]}")
    for left, right in zip(terms, terms[1:]):
        values.add(f"bigram={left}:{right}")
    return tuple(sorted({_index(value, dimension) for value in values}))


class CpuMethodRouter:
    """A class-balanced logistic head for one optional retrieval method."""

    model_id = "cpu-method-router-v1"

    def __init__(
        self,
        *,
        method: MethodName,
        dimension: int = 16384,
        epochs: int = 16,
        learning_rate: float = 0.08,
        l2: float = 1e-6,
        seed: int = 17,
    ) -> None:
        if dimension <= 0 or epochs <= 0:
            raise ValueError("dimension and epochs must be positive")
        self.method = method
        self.dimension = dimension
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.seed = seed
        self.weights = np.zeros(dimension, dtype=np.float64)

    def fit(self, rows: list[MethodRouteLabel]) -> int:
        validated = [MethodRouteLabel.model_validate(row) for row in rows]
        matching = [row for row in validated if row.method == self.method]
        if any(row.role == "development" for row in matching):
            raise ValueError("development labels cannot be used to fit the router")
        usable = [row for row in matching if row.routing_label != "unavailable"]
        positives = sum(row.routing_label == "beneficial" for row in usable)
        negatives = len(usable) - positives
        if positives == 0 or negatives == 0:
            raise ValueError("method labels require both beneficial and negative examples")
        positive_weight = negatives / positives
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(usable))
        step = 0
        for _ in range(self.epochs):
            rng.shuffle(order)
            for row_index in order:
                row = usable[int(row_index)]
                indexes = _feature_indexes(
                    method=self.method,
                    query=row.query,
                    seed_count=row.seed_count,
                    dimension=self.dimension,
                )
                score = float(self.weights[list(indexes)].sum())
                probability = 1.0 / (
                    1.0 + math.exp(-max(-30.0, min(30.0, score)))
                )
                target = 1.0 if row.routing_label == "beneficial" else 0.0
                weight = positive_weight if target else 1.0
                rate = self.learning_rate / math.sqrt(1.0 + step / 1000.0)
                gradient = weight * (target - probability)
                for index in indexes:
                    self.weights[index] = (
                        self.weights[index] * (1.0 - rate * self.l2)
                        + rate * gradient
                    )
                step += 1
        return len(usable)

    def predict_proba(self, query: str, *, seed_count: int = 0) -> float:
        indexes = _feature_indexes(
            method=self.method,
            query=query,
            seed_count=seed_count,
            dimension=self.dimension,
        )
        score = float(self.weights[list(indexes)].sum())
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))

    def save(self, path: Path) -> None:
        write_frozen_bytes(
            path,
            self.weights.astype("<f8", copy=False).tobytes(order="C"),
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        method: MethodName,
        dimension: int,
        epochs: int = 16,
        learning_rate: float = 0.08,
        l2: float = 1e-6,
        seed: int = 17,
    ) -> CpuMethodRouter:
        content = path.read_bytes()
        expected_bytes = dimension * np.dtype("<f8").itemsize
        if len(content) != expected_bytes:
            raise ValueError("CPU method router weight size mismatch")
        router = cls(
            method=method,
            dimension=dimension,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
        router.weights = np.frombuffer(content, dtype="<f8").astype(
            np.float64, copy=True
        )
        return router


__all__ = ["CpuMethodRouter"]
