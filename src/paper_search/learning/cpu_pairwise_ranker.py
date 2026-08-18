"""CPU pairwise action ranker trained only from provider-observed rewards."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.candidates import query_content_terms
from paper_search.learning.contracts import PolicyActionCandidate, QueryPolicyInput
from paper_search.learning.provider_action_labels import Provider, ProviderActionLabel
from paper_search.learning.routing import RuleQueryRouter


def _index(value: str, dimension: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _feature_indexes(
    *,
    query: str,
    query_kind: str,
    provider: Provider | None,
    action: PolicyActionCandidate,
    dimension: int,
) -> tuple[int, ...]:
    query_terms = query_content_terms(query)[:20]
    action_terms = query_content_terms(action.text)[:20]
    query_set = set(query_terms)
    action_set = set(action_terms)
    shared = query_set.intersection(action_set)
    added = action_set.difference(query_set)
    dropped = query_set.difference(action_set)
    overlap_denominator = max(1, len(query_set.union(action_set)))
    overlap_bucket = round(10 * len(shared) / overlap_denominator)
    values = {
        "bias",
        f"kind={query_kind}",
        f"provider={provider or 'either'}",
        f"type={action.action_type}",
        f"search-mode={action.search_mode}",
        f"origin={action.origin}",
        f"kind-type={query_kind}:{action.action_type}",
        f"overlap-bucket={overlap_bucket}",
        f"shared-count={min(len(shared), 10)}",
        f"added-count={min(len(added), 10)}",
        f"dropped-count={min(len(dropped), 10)}",
        f"exact={query.casefold().strip() == action.text.casefold().strip()}",
    }
    for term in shared:
        values.add(f"shared={term}")
    for term in added:
        values.add(f"added={term}")
    for term in dropped:
        values.add(f"dropped={term}")
    for query_term in query_terms:
        for action_term in action_terms:
            values.add(f"cross={query_term}:{action_term}")
    return tuple(sorted({_index(value, dimension) for value in values}))


def provider_action_reward(row: ProviderActionLabel) -> float:
    assert row.action_recall is not None
    assert row.gold_association_count is not None
    assert row.novel_over_anchor_hit_count is not None
    novelty = row.novel_over_anchor_hit_count / row.gold_association_count
    return float(row.action_recall) + 0.25 * novelty


class CpuPairwiseActionRanker:
    """Rank actions within each query/provider group using real retrieval reward."""

    model_id = "cpu-pairwise-action-ranker-v1"

    def __init__(
        self,
        *,
        target_provider: Provider | None = "openalex",
        dimension: int = 16384,
        epochs: int = 12,
        learning_rate: float = 0.08,
        l2: float = 1e-6,
        seed: int = 17,
    ) -> None:
        if dimension <= 0 or epochs <= 0:
            raise ValueError("dimension and epochs must be positive")
        self.target_provider = target_provider
        self.dimension = dimension
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.seed = seed
        self.weights = np.zeros(dimension, dtype=np.float64)

    def fit(self, rows: list[ProviderActionLabel]) -> int:
        validated = [ProviderActionLabel.model_validate(row) for row in rows]
        if any(row.role == "development" for row in validated):
            raise ValueError("development labels cannot be used to fit the ranker")
        available = [
            row
            for row in validated
            if row.retrieval_status == "available"
            and (self.target_provider is None or row.provider == self.target_provider)
        ]
        grouped: dict[tuple[str, Provider], list[ProviderActionLabel]] = defaultdict(list)
        for row in available:
            grouped[(row.query_id, row.provider)].append(row)

        pairs: list[tuple[ProviderActionLabel, ProviderActionLabel]] = []
        for candidates in grouped.values():
            for left_index, left in enumerate(candidates):
                for right in candidates[left_index + 1 :]:
                    left_reward = provider_action_reward(left)
                    right_reward = provider_action_reward(right)
                    if math.isclose(left_reward, right_reward, abs_tol=1e-12):
                        continue
                    pairs.append(
                        (left, right)
                        if left_reward > right_reward
                        else (right, left)
                    )
        if not pairs:
            raise ValueError("provider labels contain no usable preference pair")

        rng = np.random.default_rng(self.seed)
        router = RuleQueryRouter()
        order = np.arange(len(pairs))
        step = 0
        for _ in range(self.epochs):
            rng.shuffle(order)
            for pair_index in order:
                preferred, rejected = pairs[int(pair_index)]
                preferred_indexes = set(
                    _feature_indexes(
                        query=preferred.query,
                        query_kind=router.route(preferred.query).query_kind,
                        provider=preferred.provider,
                        action=preferred.action,
                        dimension=self.dimension,
                    )
                )
                rejected_indexes = set(
                    _feature_indexes(
                        query=rejected.query,
                        query_kind=router.route(rejected.query).query_kind,
                        provider=rejected.provider,
                        action=rejected.action,
                        dimension=self.dimension,
                    )
                )
                positive = preferred_indexes.difference(rejected_indexes)
                negative = rejected_indexes.difference(preferred_indexes)
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

    def score(
        self,
        request: QueryPolicyInput,
        candidates: list[PolicyActionCandidate],
    ) -> list[float]:
        request = QueryPolicyInput.model_validate(request)
        probabilities: list[float] = []
        for raw_candidate in candidates:
            candidate = PolicyActionCandidate.model_validate(raw_candidate)
            provider = (
                candidate.provider_hint
                if candidate.provider_hint != "either"
                else self.target_provider
            )
            indexes = _feature_indexes(
                query=request.original_query,
                query_kind=request.query_kind,
                provider=provider,
                action=candidate,
                dimension=self.dimension,
            )
            raw_score = float(self.weights[list(indexes)].sum())
            probabilities.append(
                1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw_score))))
            )
        return probabilities

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
        target_provider: Provider | None,
        dimension: int,
        epochs: int = 12,
        learning_rate: float = 0.08,
        l2: float = 1e-6,
        seed: int = 17,
    ) -> CpuPairwiseActionRanker:
        content = path.read_bytes()
        expected_bytes = dimension * np.dtype("<f8").itemsize
        if len(content) != expected_bytes:
            raise ValueError("CPU pairwise action ranker weight size mismatch")
        ranker = cls(
            target_provider=target_provider,
            dimension=dimension,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
        ranker.weights = np.frombuffer(content, dtype="<f8").astype(
            np.float64,
            copy=True,
        )
        return ranker


__all__ = ["CpuPairwiseActionRanker", "provider_action_reward"]
