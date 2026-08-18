from __future__ import annotations

import numpy as np

from paper_search.domain.models import Paper
from paper_search.learning.semantic_router_features import (
    LEXICAL_ROUTE_FEATURE_NAMES,
    SEMANTIC_MATCH_FEATURE_NAMES,
    extract_lexical_route_features,
    extract_semantic_match_features,
)


def _paper(identifier: str, title: str) -> Paper:
    return Paper(canonical_id=identifier, title=title)


def test_lexical_route_features_are_invariant_to_action_and_hit_order() -> None:
    first = [
        _paper("openalex:1", "Graph neural networks for retrieval"),
        _paper("openalex:2", "Database indexing methods"),
    ]
    second = [
        _paper("openalex:2", "Database indexing methods"),
        _paper("openalex:3", "Semantic document retrieval"),
    ]

    forward = extract_lexical_route_features(
        "graph semantic retrieval", [first, second]
    )
    reversed_order = extract_lexical_route_features(
        "graph semantic retrieval", [list(reversed(second)), list(reversed(first))]
    )

    assert len(forward) == len(LEXICAL_ROUTE_FEATURE_NAMES)
    assert np.array_equal(forward, reversed_order)


def test_lexical_route_features_distinguish_redundant_and_complementary_pools() -> None:
    shared = [_paper("openalex:1", "Graph retrieval")]
    complementary = [_paper("openalex:2", "Semantic search")]

    redundant = extract_lexical_route_features("graph retrieval", [shared, shared])
    diverse = extract_lexical_route_features(
        "graph retrieval", [shared, complementary]
    )
    index = LEXICAL_ROUTE_FEATURE_NAMES.index("duplicate_candidate_fraction")
    support_index = LEXICAL_ROUTE_FEATURE_NAMES.index(
        "multi_action_support_fraction"
    )

    assert redundant[index] > diverse[index]
    assert redundant[support_index] > diverse[support_index]


def test_lexical_route_features_handle_empty_results_without_nonfinite_values() -> None:
    features = extract_lexical_route_features("unanswered query", [[], []])

    assert np.isfinite(features).all()
    assert features[LEXICAL_ROUTE_FEATURE_NAMES.index("empty_action_fraction")] == 1.0


def test_semantic_match_features_are_invariant_to_action_and_hit_order() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float64)
    first = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    second = np.asarray([[0.8, 0.6]], dtype=np.float64)

    forward = extract_semantic_match_features(query, [first, second])
    reversed_order = extract_semantic_match_features(
        query, [second[::-1], first[::-1]]
    )

    assert len(forward) == len(SEMANTIC_MATCH_FEATURE_NAMES)
    assert np.allclose(forward, reversed_order)


def test_semantic_match_features_capture_query_title_alignment() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float64)
    aligned = extract_semantic_match_features(
        query, [np.asarray([[1.0, 0.0], [0.8, 0.6]])]
    )
    unrelated = extract_semantic_match_features(
        query, [np.asarray([[0.0, 1.0], [-1.0, 0.0]])]
    )
    maximum = SEMANTIC_MATCH_FEATURE_NAMES.index("maximum_query_title_similarity")
    top_mean = SEMANTIC_MATCH_FEATURE_NAMES.index("top10_query_title_similarity")

    assert aligned[maximum] > unrelated[maximum]
    assert aligned[top_mean] > unrelated[top_mean]


def test_semantic_match_features_handle_empty_actions_without_nonfinite_values() -> None:
    features = extract_semantic_match_features(
        np.asarray([1.0, 0.0], dtype=np.float64),
        [np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)],
    )

    assert np.isfinite(features).all()
    assert np.array_equal(features, np.zeros(len(SEMANTIC_MATCH_FEATURE_NAMES)))
