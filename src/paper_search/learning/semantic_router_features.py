"""Gold-blind lexical retrieval-state features for semantic routing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from paper_search.domain.models import Paper
from paper_search.learning.candidates import query_content_terms


LEXICAL_ROUTE_FEATURE_NAMES = (
    "action_count_scaled",
    "empty_action_fraction",
    "mean_candidate_fraction",
    "minimum_candidate_fraction",
    "maximum_candidate_fraction",
    "union_candidate_fraction",
    "duplicate_candidate_fraction",
    "multi_action_support_fraction",
    "mean_pairwise_jaccard",
    "maximum_pairwise_jaccard",
    "unique_title_fraction",
    "mean_title_query_coverage",
    "maximum_title_query_coverage",
    "top10_title_query_coverage",
    "mean_action_best_title_coverage",
    "minimum_action_best_title_coverage",
    "query_term_count_scaled",
)

SEMANTIC_MATCH_FEATURE_NAMES = (
    "mean_query_title_similarity",
    "maximum_query_title_similarity",
    "median_query_title_similarity",
    "upper_quartile_query_title_similarity",
    "top10_query_title_similarity",
    "mean_action_best_similarity",
    "minimum_action_best_similarity",
    "similarity_standard_deviation",
)


def _safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def extract_lexical_route_features(
    query: str,
    action_hits: Sequence[Sequence[Paper]],
    *,
    maximum_results_per_action: int = 50,
) -> np.ndarray:
    """Summarize a completed lexical pool without using Gold information."""
    if not action_hits:
        raise ValueError("lexical route features require at least one action")
    if maximum_results_per_action <= 0:
        raise ValueError("maximum results per action must be positive")

    hit_sets = [set(paper.canonical_id for paper in hits) for hits in action_hits]
    counts = [len(hits) for hits in hit_sets]
    union = set().union(*hit_sets)
    total = sum(counts)
    action_count = len(action_hits)
    occurrence_counts = Counter(
        paper_id for hit_set in hit_sets for paper_id in hit_set
    )
    repeated = sum(count > 1 for count in occurrence_counts.values())

    pairwise_jaccard: list[float] = []
    for left_index, left in enumerate(hit_sets):
        for right in hit_sets[left_index + 1 :]:
            combined = left | right
            pairwise_jaccard.append(
                len(left & right) / len(combined) if combined else 0.0
            )

    papers_by_id: dict[str, Paper] = {}
    for hits in action_hits:
        for paper in hits:
            papers_by_id.setdefault(paper.canonical_id, paper)
    query_terms = set(query_content_terms(query))

    def title_coverage(paper: Paper) -> float:
        if not query_terms:
            return 0.0
        title_terms = set(query_content_terms(paper.title))
        return len(query_terms & title_terms) / len(query_terms)

    coverages = [title_coverage(paper) for paper in papers_by_id.values()]
    best_by_action = [
        max((title_coverage(paper) for paper in hits), default=0.0)
        for hits in action_hits
    ]
    normalized_titles = {
        " ".join(paper.title.casefold().split()) for paper in papers_by_id.values()
    }
    title_fraction = len(normalized_titles) / len(union) if union else 0.0
    sorted_coverages = sorted(coverages, reverse=True)
    top_coverages = sorted_coverages[:10]
    capacity = maximum_results_per_action * action_count
    features = np.asarray(
        [
            min(action_count, 10) / 10.0,
            sum(count == 0 for count in counts) / action_count,
            _safe_mean([count / maximum_results_per_action for count in counts]),
            min(counts) / maximum_results_per_action,
            max(counts) / maximum_results_per_action,
            len(union) / capacity,
            1.0 - len(union) / total if total else 0.0,
            repeated / len(union) if union else 0.0,
            _safe_mean(pairwise_jaccard),
            max(pairwise_jaccard, default=0.0),
            title_fraction,
            _safe_mean(coverages),
            max(coverages, default=0.0),
            _safe_mean(top_coverages),
            _safe_mean(best_by_action),
            min(best_by_action, default=0.0),
            min(len(query_terms), 30) / 30.0,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(features).all():
        raise ValueError("lexical route features must be finite")
    return features


def extract_semantic_match_features(
    query_embedding: np.ndarray,
    action_title_embeddings: Sequence[np.ndarray],
) -> np.ndarray:
    """Summarize frozen-encoder query/title alignment after lexical retrieval."""
    query_vector = np.asarray(query_embedding, dtype=np.float64)
    if query_vector.ndim != 1 or query_vector.size == 0:
        raise ValueError("query embedding must be a non-empty vector")
    query_norm = float(np.linalg.norm(query_vector))
    if not np.isfinite(query_norm) or query_norm == 0.0:
        raise ValueError("query embedding must have a finite non-zero norm")
    normalized_query = query_vector / query_norm

    similarities_by_action: list[np.ndarray] = []
    for raw_embeddings in action_title_embeddings:
        embeddings = np.asarray(raw_embeddings, dtype=np.float64)
        if embeddings.ndim != 2 or embeddings.shape[1] != query_vector.size:
            raise ValueError("title embeddings must be matrices matching query size")
        if not embeddings.size:
            similarities_by_action.append(np.empty(0, dtype=np.float64))
            continue
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.isfinite(embeddings).all() or np.any(norms == 0.0):
            raise ValueError("title embeddings must be finite non-zero vectors")
        similarities_by_action.append((embeddings / norms[:, None]) @ normalized_query)

    populated = [values for values in similarities_by_action if values.size]
    if not populated:
        return np.zeros(len(SEMANTIC_MATCH_FEATURE_NAMES), dtype=np.float64)
    similarities = np.concatenate(populated)
    action_best = [float(values.max()) if values.size else 0.0 for values in similarities_by_action]
    top10 = np.sort(similarities)[-10:]
    features = np.asarray(
        [
            similarities.mean(),
            similarities.max(),
            np.median(similarities),
            np.quantile(similarities, 0.75),
            top10.mean(),
            _safe_mean(action_best),
            min(action_best, default=0.0),
            similarities.std(),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(features).all():
        raise ValueError("semantic match features must be finite")
    return features


__all__ = [
    "LEXICAL_ROUTE_FEATURE_NAMES",
    "SEMANTIC_MATCH_FEATURE_NAMES",
    "extract_lexical_route_features",
    "extract_semantic_match_features",
]
