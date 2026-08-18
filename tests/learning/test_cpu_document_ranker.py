from __future__ import annotations

from paper_search.domain.models import Paper
from paper_search.learning.cpu_document_ranker import (
    CpuPairwiseDocumentRanker,
    DocumentRankingQuery,
    build_document_candidates,
    build_production_document_candidates,
    document_source_family,
)


def _paper(identifier: str, title: str, abstract: str | None = None) -> Paper:
    return Paper(
        canonical_id=identifier,
        openalex_id=identifier,
        title=title,
        abstract=abstract,
        is_retracted=False,
    )


def test_document_candidates_preserve_cross_action_support_after_deduplication() -> None:
    shared = _paper("openalex:W1", "graph retrieval")
    candidates = build_document_candidates(
        [
            ("lexical", [_paper("openalex:W2", "other"), shared]),
            ("semantic", [shared]),
        ]
    )

    assert [row.paper.canonical_id for row in candidates] == [
        "openalex:W1",
        "openalex:W2",
    ]
    assert candidates[0].source_ranks == {"lexical": 2, "semantic": 1}
    assert candidates[0].support_count == 2


def test_production_document_candidates_apply_filter_without_reordering_rrf() -> None:
    kept = _paper("openalex:W5", "kept")
    retracted = Paper(
        canonical_id="openalex:W6",
        openalex_id="openalex:W6",
        title="retracted",
        is_retracted=True,
    )

    candidates = build_production_document_candidates(
        "papers about retrieval",
        [("lexical", [retracted, kept]), ("semantic", [kept])],
    )

    assert [row.paper.canonical_id for row in candidates] == ["openalex:W5"]


def test_pairwise_document_ranker_promotes_relevant_text_without_changing_pool() -> None:
    positive = _paper(
        "openalex:W10",
        "graph diffusion retrieval",
        "A method for graph diffusion in document retrieval.",
    )
    negative = _paper("openalex:W11", "unrelated image generation")
    candidates = build_document_candidates([("lexical", [negative, positive])])
    query = DocumentRankingQuery(
        query_id="train-1",
        query="graph diffusion retrieval",
        gold_paper_ids=["openalex:W10"],
        candidates=candidates,
    )
    ranker = CpuPairwiseDocumentRanker(
        dimension=2048,
        epochs=20,
        learning_rate=0.1,
        learned_weight=0.8,
        seed=7,
    )

    assert ranker.fit([query]) == 1
    ranked = ranker.rank(query.query, candidates)

    assert [row.paper.canonical_id for row in ranked] == [
        "openalex:W10",
        "openalex:W11",
    ]
    assert {row.paper.canonical_id for row in ranked} == {
        row.paper.canonical_id for row in candidates
    }


def test_document_ranker_normalizes_semantic_action_identity_across_receipts() -> None:
    assert document_source_family("semantic-backfill-original") == "semantic_original"
    assert (
        document_source_family("ceiling-candidate-semantic-original")
        == "semantic_original"
    )
    train_candidates = build_document_candidates(
        [
            ("semantic-backfill-original", [_paper("openalex:W1", "same alpha")]),
            ("ceiling-candidate-anchor", [_paper("openalex:W2", "same beta")]),
        ]
    )
    train = DocumentRankingQuery(
        query_id="train-semantic",
        query="same",
        gold_paper_ids=["openalex:W1"],
        candidates=train_candidates,
    )
    evaluation_candidates = build_document_candidates(
        [
            ("ceiling-candidate-anchor", [_paper("openalex:W3", "same gamma")]),
            ("ceiling-candidate-semantic-original", [_paper("openalex:W4", "same delta")]),
        ]
    )
    ranker = CpuPairwiseDocumentRanker(
        dimension=2048,
        epochs=20,
        learning_rate=0.1,
        learned_weight=1.0,
        seed=11,
    )

    ranker.fit([train])
    ranked = ranker.rank("same", evaluation_candidates)

    assert ranked[0].paper.canonical_id == "openalex:W4"


def test_document_ranker_save_load_preserves_ranking(tmp_path) -> None:
    positive = _paper("openalex:W20", "graph retrieval")
    negative = _paper("openalex:W21", "image synthesis")
    candidates = build_document_candidates([("lexical", [negative, positive])])
    query = DocumentRankingQuery(
        query_id="train-save",
        query="graph retrieval",
        gold_paper_ids=["openalex:W20"],
        candidates=candidates,
    )
    ranker = CpuPairwiseDocumentRanker(dimension=1024, epochs=4, seed=13)
    ranker.fit([query])
    model_path = tmp_path / "document-ranker.f64"

    ranker.save(model_path)
    loaded = CpuPairwiseDocumentRanker.load(
        model_path,
        dimension=1024,
        epochs=4,
        learned_weight=ranker.learned_weight,
        seed=13,
    )

    assert [row.paper.canonical_id for row in loaded.rank(query.query, candidates)] == [
        row.paper.canonical_id for row in ranker.rank(query.query, candidates)
    ]
