from __future__ import annotations

import pytest
from rank_bm25 import BM25Okapi

from paper_search.domain.models import Paper, QuerySpec
from paper_search.processing import AcceptedPaper
from paper_search.ranking import rank_lexically, tokenize_text


def _query(text: str) -> QuerySpec:
    return QuerySpec(original_query=text, research_goal="Find relevant papers")


def _accepted(
    canonical_id: str = "openalex:W1",
    *,
    title: str = "Paper",
    abstract: str | None = None,
    multiplier: float = 1.0,
) -> AcceptedPaper:
    return AcceptedPaper(
        paper=Paper(canonical_id=canonical_id, title=title, abstract=abstract),
        uncertainty_reasons=[],
        score_multiplier=multiplier,
    )


def test_tokenizer_is_unicode_normalized_and_deterministic() -> None:
    assert tokenize_text("Graph-based ＡＩ, graph!") == ["graph", "based", "ai", "graph"]


def test_keyword_coverage_uses_unique_query_tokens() -> None:
    ranked = rank_lexically(
        _query("graph graph retrieval"),
        [_accepted(title="Graph methods")],
    )

    assert ranked[0].keyword_coverage == pytest.approx(0.5)


def test_bm25_raw_scores_are_retained_and_min_max_normalized() -> None:
    query = _query("graph retrieval")
    candidates = [
        _accepted("openalex:W1", title="Graph retrieval", abstract="Graph methods"),
        _accepted("openalex:W2", title="Retrieval systems"),
        _accepted("openalex:W3", title="Database indexing"),
    ]
    document_tokens = [
        tokenize_text(" ".join(filter(None, (item.paper.title, item.paper.abstract))))
        for item in candidates
    ]
    expected_raw = [
        float(value)
        for value in BM25Okapi(document_tokens).get_scores(tokenize_text(query.original_query))
    ]

    ranked = rank_lexically(query, candidates)

    by_id = {item.paper.canonical_id: item for item in ranked}
    minimum = min(expected_raw)
    maximum = max(expected_raw)
    for candidate, raw_score in zip(candidates, expected_raw, strict=True):
        score = by_id[candidate.paper.canonical_id]
        expected_normalized = (raw_score - minimum) / (maximum - minimum)
        assert score.bm25_score == pytest.approx(raw_score)
        assert score.normalized_bm25 == pytest.approx(expected_normalized)
        assert 0.0 <= score.normalized_bm25 <= 1.0
        assert score.final_score == pytest.approx(
            (0.7 * expected_normalized + 0.3 * score.keyword_coverage)
            * candidate.score_multiplier
        )


def test_equal_raw_scores_normalize_to_zero() -> None:
    ranked = rank_lexically(
        _query("graph retrieval"),
        [
            _accepted("openalex:W1", title="Graph retrieval"),
            _accepted("openalex:W2", title="Graph retrieval"),
        ],
    )

    assert ranked[0].bm25_score == pytest.approx(ranked[1].bm25_score)
    assert [item.normalized_bm25 for item in ranked] == [0.0, 0.0]


def test_uncertainty_multiplier_changes_final_order() -> None:
    candidates = [
        _accepted("openalex:W1", title="graph retrieval", multiplier=0.7),
        _accepted("openalex:W2", title="graph retrieval", multiplier=1.0),
    ]

    ranked = rank_lexically(_query("graph retrieval"), candidates)

    assert [item.paper.canonical_id for item in ranked] == ["openalex:W2", "openalex:W1"]


def test_empty_candidates_return_empty_scores() -> None:
    assert rank_lexically(_query("graph retrieval"), []) == []


def test_token_empty_documents_have_zero_scores() -> None:
    ranked = rank_lexically(_query("graph retrieval"), [_accepted(title="---")])

    assert ranked[0].bm25_score == 0.0
    assert ranked[0].normalized_bm25 == 0.0
    assert ranked[0].keyword_coverage == 0.0
    assert ranked[0].final_score == 0.0


def test_exact_ties_retain_input_order_before_canonical_id() -> None:
    candidates = [
        _accepted("openalex:W2", title="graph retrieval"),
        _accepted("openalex:W1", title="graph retrieval"),
    ]

    ranked = rank_lexically(_query("graph retrieval"), candidates)

    assert [item.paper.canonical_id for item in ranked] == ["openalex:W2", "openalex:W1"]
