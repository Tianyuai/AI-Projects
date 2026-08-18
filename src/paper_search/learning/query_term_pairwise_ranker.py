"""Deterministic CPU pairwise ranking for query-to-paper terminology."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from paper_search.learning.candidates import query_content_terms
from paper_search.learning.lexical_bridge import SupervisedLexicalBridge


def _index(value: str, dimension: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _feature_indexes(
    row: QueryTermCandidate, dimension: int
) -> tuple[int, ...]:
    term = row.term.casefold().strip()
    query_terms = query_content_terms(row.query)[:20]
    values = {
        "bias",
        f"term={term}",
        f"prefix3={term[:3]}",
        f"suffix3={term[-3:]}",
        f"support={min(row.neighbor_support, 12)}",
        f"sum-sim={min(round(row.similarity_sum * 10), 120)}",
        f"max-sim={min(round(row.maximum_similarity * 20), 20)}",
        f"idf={min(round(row.title_idf * 2), 30)}",
    }
    for query_term in query_terms:
        values.add(f"cross={query_term}:{term}")
    return tuple(sorted({_index(value, dimension) for value in values}))


@dataclass(frozen=True)
class QueryTermCandidate:
    query_id: str
    query: str
    term: str
    relevant: bool
    neighbor_support: int
    similarity_sum: float
    maximum_similarity: float
    title_idf: float


class QueryTermPairwiseRanker:
    def __init__(
        self,
        *,
        dimension: int = 16384,
        epochs: int = 8,
        learning_rate: float = 0.08,
        l2: float = 1e-6,
        seed: int = 17,
    ) -> None:
        if dimension <= 0 or epochs <= 0:
            raise ValueError("dimension and epochs must be positive")
        self.dimension = dimension
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.seed = seed
        self.weights = np.zeros(dimension, dtype=np.float64)

    def fit(self, rows: list[QueryTermCandidate]) -> int:
        grouped: dict[str, list[QueryTermCandidate]] = defaultdict(list)
        queries: dict[str, str] = {}
        for row in rows:
            if (
                not row.query_id
                or not row.query.strip()
                or not row.term.strip()
                or row.neighbor_support <= 0
                or row.similarity_sum < 0
                or row.maximum_similarity < 0
                or row.title_idf <= 0
            ):
                raise ValueError("invalid query-term candidate")
            if row.query_id in queries and queries[row.query_id] != row.query:
                raise ValueError("query id maps to multiple query texts")
            queries[row.query_id] = row.query
            grouped[row.query_id].append(row)
        pairs = [
            (positive, negative)
            for candidates in grouped.values()
            for positive in candidates
            if positive.relevant
            for negative in candidates
            if not negative.relevant
        ]
        if not pairs:
            raise ValueError("query-term labels contain no preference pair")
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(pairs))
        step = 0
        for _ in range(self.epochs):
            rng.shuffle(order)
            for pair_index in order:
                preferred, rejected = pairs[int(pair_index)]
                positive = set(_feature_indexes(preferred, self.dimension))
                negative = set(_feature_indexes(rejected, self.dimension))
                positive.difference_update(negative)
                negative.difference_update(
                    _feature_indexes(preferred, self.dimension)
                )
                score_difference = float(
                    self.weights[list(positive)].sum()
                    - self.weights[list(negative)].sum()
                )
                probability = 1.0 / (
                    1.0 + math.exp(-max(-30.0, min(30.0, score_difference)))
                )
                rate = self.learning_rate / math.sqrt(1.0 + step / 1000.0)
                gradient = 1.0 - probability
                for index in positive:
                    self.weights[index] = (
                        self.weights[index] * (1.0 - rate * self.l2)
                        + rate * gradient
                    )
                for index in negative:
                    self.weights[index] = (
                        self.weights[index] * (1.0 - rate * self.l2)
                        - rate * gradient
                    )
                step += 1
        return len(pairs)

    def score(self, rows: list[QueryTermCandidate]) -> list[float]:
        result = []
        for row in rows:
            raw = float(
                self.weights[list(_feature_indexes(row, self.dimension))].sum()
            )
            result.append(
                1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))
            )
        return result


def build_query_term_candidates(
    bridge: SupervisedLexicalBridge,
    *,
    query_id: str,
    query: str,
    gold_titles: tuple[str, ...] = (),
    neighbors: int = 12,
    exclude_training_index: int | None = None,
    minimum_similarity: float = 0.05,
) -> list[QueryTermCandidate]:
    if neighbors <= 0:
        raise ValueError("neighbors must be positive")
    query_vector = bridge._vectorizer.transform([query])
    similarities = np.asarray(
        (bridge._query_matrix @ query_vector.T).toarray()
    ).ravel()
    if exclude_training_index is not None:
        if not 0 <= exclude_training_index < len(similarities):
            raise ValueError("excluded training index is out of range")
        similarities = similarities.copy()
        similarities[exclude_training_index] = -np.inf
    selected = np.argsort(-similarities, kind="stable")[:neighbors]
    original_terms = set(query_content_terms(query))
    gold_terms = {
        term for title in gold_titles for term in query_content_terms(title)
    }
    support: dict[str, set[int]] = defaultdict(set)
    similarity_sum: dict[str, float] = defaultdict(float)
    maximum_similarity: dict[str, float] = defaultdict(float)
    for index in selected:
        similarity = float(similarities[index])
        if similarity < minimum_similarity:
            continue
        for term in set(bridge._title_terms[int(index)]).difference(original_terms):
            support[term].add(int(index))
            similarity_sum[term] += similarity
            maximum_similarity[term] = max(maximum_similarity[term], similarity)
    return [
        QueryTermCandidate(
            query_id=query_id,
            query=query,
            term=term,
            relevant=term in gold_terms,
            neighbor_support=len(support[term]),
            similarity_sum=similarity_sum[term],
            maximum_similarity=maximum_similarity[term],
            title_idf=bridge._title_idf[term],
        )
        for term in sorted(support)
    ]


__all__ = [
    "QueryTermCandidate",
    "QueryTermPairwiseRanker",
    "build_query_term_candidates",
]
