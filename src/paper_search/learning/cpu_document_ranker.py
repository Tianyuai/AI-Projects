"""Deterministic CPU pairwise ranker for papers in a frozen candidate pool."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import Field, computed_field, field_validator

from paper_search.domain.models import (
    DomainModel,
    NonEmptyStr,
    Paper,
    ProviderResult,
    UsageActual,
)
from paper_search.evaluation.dataset import normalize_paper_id, write_frozen_bytes
from paper_search.evaluation.predictions import paper_evaluation_id
from paper_search.learning.candidates import query_content_terms
from paper_search.processing.deduplicate import deduplicate_papers
from paper_search.processing.filter import apply_hard_filters
from paper_search.query.parser import rule_fallback
from paper_search.ranking.fusion import fuse_provider_results


class DocumentCandidateEvidence(DomainModel):
    paper: Paper
    baseline_score: float = Field(ge=0, allow_inf_nan=False)
    source_ranks: dict[NonEmptyStr, int]

    @field_validator("source_ranks")
    @classmethod
    def validate_source_ranks(cls, value: dict[str, int]) -> dict[str, int]:
        if not value or any(type(rank) is not int or rank <= 0 for rank in value.values()):
            raise ValueError("source ranks must be non-empty positive integers")
        return value

    @computed_field
    @property
    def support_count(self) -> int:
        return len(self.source_ranks)


class DocumentRankingQuery(DomainModel):
    query_id: NonEmptyStr
    query: NonEmptyStr
    gold_paper_ids: list[NonEmptyStr]
    candidates: list[DocumentCandidateEvidence]

    @field_validator("gold_paper_ids")
    @classmethod
    def normalize_gold(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(normalize_paper_id(value) for value in values))
        if not normalized:
            raise ValueError("document ranking query requires Gold papers")
        return normalized


def _provider_result(
    action_id: str, papers: Sequence[Paper], index: int
) -> ProviderResult[list[Paper]]:
    return ProviderResult(
        data=list(papers),
        usage=UsageActual(),
        provenance={
            "provider": action_id,
            "endpoint": "document-ranker-candidate-builder",
            "model_id": "document-ranker-candidate-builder-v1",
            "requested_at": "2026-08-18T00:00:00+08:00",
            "response_hash": "sha256:" + f"{index:064x}"[-64:],
        },
        cache_hit=True,
        latency_ms=0,
        errors=[],
    )


def build_document_candidates(
    action_results: Sequence[tuple[str, Sequence[Paper]]],
) -> list[DocumentCandidateEvidence]:
    """Build production-equivalent action-level RRF evidence."""

    action_ids = [action_id for action_id, _papers in action_results]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("document candidate action ids must be unique")
    results = {
        action_id: _provider_result(action_id, papers, index)
        for index, (action_id, papers) in enumerate(action_results, start=1)
    }
    return [
        DocumentCandidateEvidence(
            paper=item.paper,
            baseline_score=item.score,
            source_ranks=item.source_ranks,
        )
        for item in fuse_provider_results(results, method="rrf", rrf_k=60)
    ]


def build_production_document_candidates(
    query: str,
    action_results: Sequence[tuple[str, Sequence[Paper]]],
) -> list[DocumentCandidateEvidence]:
    """Mirror production's merge/filter-before-RRF candidate selection order."""

    merged = deduplicate_papers(
        [paper for _action_id, papers in action_results for paper in papers]
    )
    accepted_ids = {
        item.paper.canonical_id
        for item in apply_hard_filters(merged.papers, rule_fallback(query)).accepted
    }
    return [
        candidate
        for candidate in build_document_candidates(action_results)
        if candidate.paper.canonical_id in accepted_ids
    ]


def _index(value: str, dimension: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _add(features: dict[int, float], key: str, value: float, dimension: int) -> None:
    index = _index(key, dimension)
    features[index] = features.get(index, 0.0) + value


def document_source_family(action_id: str) -> str:
    """Normalize equivalent retrieval-action identities across receipt versions."""

    normalized = action_id.casefold()
    if "semantic" in normalized and "original" in normalized:
        return "semantic_original"
    if "boolean" in normalized:
        return "boolean"
    if "anchor" in normalized:
        return "lexical_anchor"
    if "text-" in normalized:
        return "structured_text"
    return normalized


def _feature_values(
    *,
    query: str,
    candidate: DocumentCandidateEvidence,
    baseline_rank: int,
    dimension: int,
) -> dict[int, float]:
    query_terms = query_content_terms(query)[:24]
    title_terms = query_content_terms(candidate.paper.title)[:32]
    abstract_terms = query_content_terms(candidate.paper.abstract or "")[:64]
    query_set = set(query_terms)
    title_set = set(title_terms)
    abstract_set = set(abstract_terms)
    title_shared = query_set & title_set
    abstract_shared = query_set & abstract_set
    all_shared = title_shared | abstract_shared
    features: dict[int, float] = {}
    _add(features, "bias", 1.0, dimension)
    _add(features, "baseline-reciprocal", 1.0 / baseline_rank, dimension)
    _add(features, "support-count", min(candidate.support_count, 6) / 6.0, dimension)
    _add(features, "title-overlap", len(title_shared) / max(1, len(query_set)), dimension)
    _add(
        features,
        "abstract-overlap",
        len(abstract_shared) / max(1, len(query_set)),
        dimension,
    )
    _add(features, "all-overlap-count", min(len(all_shared), 10) / 10.0, dimension)
    _add(features, f"has-abstract={bool(candidate.paper.abstract)}", 1.0, dimension)
    _add(features, f"support-bucket={min(candidate.support_count, 4)}", 1.0, dimension)
    for source, rank in candidate.source_ranks.items():
        family = document_source_family(source)
        _add(features, f"source={family}", 1.0, dimension)
        _add(features, f"source-rank={family}", 1.0 / rank, dimension)
    for term in sorted(title_shared):
        _add(features, f"title-shared={term}", 1.0, dimension)
    for term in sorted(abstract_shared):
        _add(features, f"abstract-shared={term}", 1.0, dimension)
    if candidate.paper.citation_count is not None:
        _add(
            features,
            "log-citations",
            min(math.log1p(candidate.paper.citation_count) / 12.0, 1.0),
            dimension,
        )
    return features


def _score(weights: np.ndarray, features: dict[int, float]) -> float:
    return sum(weights[index] * value for index, value in features.items())


class CpuPairwiseDocumentRanker:
    """Learn pairwise document preferences and conservatively blend with RRF."""

    model_id = "cpu-pairwise-document-ranker-v1"

    def __init__(
        self,
        *,
        dimension: int = 16384,
        epochs: int = 12,
        learning_rate: float = 0.05,
        l2: float = 1e-6,
        learned_weight: float = 0.35,
        hard_negative_limit: int = 100,
        seed: int = 29,
    ) -> None:
        if dimension <= 0 or epochs <= 0 or hard_negative_limit <= 0:
            raise ValueError("ranker sizes must be positive")
        if not 0 <= learned_weight <= 1:
            raise ValueError("learned weight must be between zero and one")
        self.dimension = dimension
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.learned_weight = learned_weight
        self.hard_negative_limit = hard_negative_limit
        self.seed = seed
        self.weights = np.zeros(dimension, dtype=np.float64)

    def fit(self, queries: Sequence[DocumentRankingQuery]) -> int:
        pairs: list[tuple[dict[int, float], dict[int, float]]] = []
        for raw_query in queries:
            query = DocumentRankingQuery.model_validate(raw_query)
            gold = set(query.gold_paper_ids)
            rows = [
                (
                    candidate,
                    paper_evaluation_id(candidate.paper) in gold,
                    _feature_values(
                        query=query.query,
                        candidate=candidate,
                        baseline_rank=rank,
                        dimension=self.dimension,
                    ),
                )
                for rank, candidate in enumerate(query.candidates, start=1)
            ]
            positives = [features for _candidate, hit, features in rows if hit]
            negatives = [
                features for _candidate, hit, features in rows if not hit
            ][: self.hard_negative_limit]
            pairs.extend(
                (positive, negative)
                for positive in positives
                for negative in negatives
            )
        if not pairs:
            raise ValueError("document ranking labels contain no preference pair")

        rng = np.random.default_rng(self.seed)
        order = np.arange(len(pairs))
        step = 0
        for _epoch in range(self.epochs):
            rng.shuffle(order)
            for pair_index in order:
                positive, negative = pairs[int(pair_index)]
                indexes = set(positive) | set(negative)
                difference = {
                    index: positive.get(index, 0.0) - negative.get(index, 0.0)
                    for index in indexes
                }
                margin = _score(self.weights, difference)
                probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, margin))))
                rate = self.learning_rate / math.sqrt(1.0 + step / 1000.0)
                gradient = 1.0 - probability
                for index, value in difference.items():
                    self.weights[index] = (
                        self.weights[index] * (1.0 - rate * self.l2)
                        + rate * gradient * value
                    )
                step += 1
        return len(pairs)

    def rank(
        self,
        query: str,
        candidates: Sequence[DocumentCandidateEvidence],
    ) -> list[DocumentCandidateEvidence]:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("document ranking query must not be empty")
        rows = list(candidates)
        learned = [
            _score(
                self.weights,
                _feature_values(
                    query=normalized_query,
                    candidate=candidate,
                    baseline_rank=rank,
                    dimension=self.dimension,
                ),
            )
            for rank, candidate in enumerate(rows, start=1)
        ]
        learned_order = sorted(range(len(rows)), key=lambda index: (-learned[index], index))
        learned_rank = {index: rank for rank, index in enumerate(learned_order, start=1)}
        final_order = sorted(
            range(len(rows)),
            key=lambda index: (
                -(
                    (1.0 - self.learned_weight) / (60 + index + 1)
                    + self.learned_weight / (60 + learned_rank[index])
                ),
                index,
            ),
        )
        return [rows[index] for index in final_order]

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
        dimension: int,
        epochs: int = 12,
        learning_rate: float = 0.05,
        l2: float = 1e-6,
        learned_weight: float = 0.35,
        hard_negative_limit: int = 100,
        seed: int = 29,
    ) -> CpuPairwiseDocumentRanker:
        return cls.load_bytes(
            path.read_bytes(),
            dimension=dimension,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            learned_weight=learned_weight,
            hard_negative_limit=hard_negative_limit,
            seed=seed,
        )

    @classmethod
    def load_bytes(
        cls,
        content: bytes,
        *,
        dimension: int,
        epochs: int = 12,
        learning_rate: float = 0.05,
        l2: float = 1e-6,
        learned_weight: float = 0.35,
        hard_negative_limit: int = 100,
        seed: int = 29,
    ) -> CpuPairwiseDocumentRanker:
        """Restore a ranker from an already verified immutable byte snapshot."""

        expected_bytes = dimension * np.dtype("<f8").itemsize
        if len(content) != expected_bytes:
            raise ValueError("CPU document ranker weight size mismatch")
        ranker = cls(
            dimension=dimension,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            learned_weight=learned_weight,
            hard_negative_limit=hard_negative_limit,
            seed=seed,
        )
        ranker.weights = np.frombuffer(content, dtype="<f8").astype(
            np.float64, copy=True
        )
        return ranker


__all__ = [
    "CpuPairwiseDocumentRanker",
    "DocumentCandidateEvidence",
    "DocumentRankingQuery",
    "build_document_candidates",
    "build_production_document_candidates",
    "document_source_family",
]
