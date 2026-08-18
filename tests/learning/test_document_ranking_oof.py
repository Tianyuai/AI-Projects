from __future__ import annotations

from paper_search.domain.models import Paper
from paper_search.learning.cpu_document_ranker import (
    CpuPairwiseDocumentRanker,
    DocumentCandidateEvidence,
    DocumentRankingQuery,
    build_document_candidates,
)
from paper_search.learning.document_ranking_oof import (
    evaluate_document_ranker,
    evaluate_document_ranker_comparison,
    evaluate_document_ranking_oof,
    evaluate_document_ranking_oof_expansion,
)


def _paper(identifier: str, title: str) -> Paper:
    return Paper(
        canonical_id=identifier,
        openalex_id=identifier,
        title=title,
        is_retracted=False,
    )


def test_document_ranking_oof_trains_only_on_other_folds_and_preserves_oracle() -> None:
    rows = []
    for fold in (1, 2, 3):
        positive_id = f"openalex:W{fold}0"
        candidates = build_document_candidates(
            [
                (
                    "lexical",
                    [
                        _paper(f"openalex:W{fold}1", "unrelated image synthesis"),
                        _paper(positive_id, "graph diffusion retrieval"),
                    ],
                )
            ]
        )
        rows.append(
            (
                fold,
                DocumentRankingQuery(
                    query_id=f"q-{fold}",
                    query="graph diffusion retrieval",
                    gold_paper_ids=[positive_id],
                    candidates=candidates,
                ),
            )
        )

    result = evaluate_document_ranking_oof(
        rows,
        cutoffs=(1, 2, 10, 20, 50),
        ranker_factory=lambda seed: CpuPairwiseDocumentRanker(
            dimension=2048,
            epochs=20,
            learning_rate=0.1,
            learned_weight=0.8,
            seed=seed,
        ),
    )

    assert result["baseline"]["macro_recall_at"][1] == 0.0
    assert result["candidate"]["macro_recall_at"][1] == 1.0
    assert result["baseline"]["candidate_oracle_macro_recall"] == 1.0
    assert result["candidate"]["candidate_oracle_macro_recall"] == 1.0
    assert result["candidate_pool_identity_unchanged"] is True
    assert result["folds"]["1"]["train_query_count"] == 2
    assert result["folds"]["1"]["evaluation_query_count"] == 1


def test_trained_document_ranker_evaluation_keeps_candidate_identity() -> None:
    positive = _paper("openalex:W90", "graph diffusion retrieval")
    negative = _paper("openalex:W91", "unrelated synthesis")
    candidates = build_document_candidates([("lexical", [negative, positive])])
    query = DocumentRankingQuery(
        query_id="q-eval",
        query="graph diffusion retrieval",
        gold_paper_ids=["openalex:W90"],
        candidates=candidates,
    )
    ranker = CpuPairwiseDocumentRanker(
        dimension=2048,
        epochs=20,
        learning_rate=0.1,
        learned_weight=0.8,
        seed=19,
    )
    ranker.fit([query])

    result = evaluate_document_ranker(
        [(1, query), (2, query.model_copy(update={"query_id": "q-eval-2"})), (3, query.model_copy(update={"query_id": "q-eval-3"}))],
        ranker=ranker,
        cutoffs=(1, 10, 20, 50),
    )

    assert result["candidate_pool_identity_unchanged"] is True
    assert result["candidate"]["macro_recall_at"][1] == 1.0


def test_document_ranking_metrics_deduplicate_scorer_identity_before_cutoff() -> None:
    duplicate_one = Paper(
        canonical_id="openalex:W101",
        openalex_id="openalex:W101",
        arxiv_id="2301.00001",
        title="first version",
        is_retracted=False,
    )
    duplicate_two = Paper(
        canonical_id="openalex:W102",
        openalex_id="openalex:W102",
        arxiv_id="2301.00001",
        title="second version",
        is_retracted=False,
    )
    gold = _paper("openalex:W103", "gold")
    candidates = [
        DocumentCandidateEvidence(
            paper=paper,
            baseline_score=1.0 / (60 + rank),
            source_ranks={"lexical": rank},
        )
        for rank, paper in enumerate(
            [duplicate_one, duplicate_two, gold], start=1
        )
    ]
    query = DocumentRankingQuery(
        query_id="identity-dedup",
        query="gold",
        gold_paper_ids=["openalex:W103"],
        candidates=candidates,
    )
    ranker = CpuPairwiseDocumentRanker(learned_weight=0.0)

    result = evaluate_document_ranker(
        [
            (1, query),
            (2, query.model_copy(update={"query_id": "identity-dedup-2"})),
            (3, query.model_copy(update={"query_id": "identity-dedup-3"})),
        ],
        ranker=ranker,
        cutoffs=(2, 10, 20, 50),
    )

    assert result["baseline"]["macro_recall_at"][2] == 1.0


def test_expanded_oof_adds_only_disjoint_supplemental_training_rows() -> None:
    fit_calls: list[tuple[str, set[str]]] = []

    class RecordingRanker:
        model_id = "recording-ranker-v1"

        def __init__(self, label: str, *, reverse: bool) -> None:
            self.label = label
            self.reverse = reverse

        def fit(self, queries) -> int:
            ids = {query.query_id for query in queries}
            fit_calls.append((self.label, ids))
            return len(ids)

        def rank(self, query, candidates):
            del query
            rows = list(candidates)
            return list(reversed(rows)) if self.reverse else rows

    folded = []
    for fold in (1, 2, 3):
        positive_id = f"openalex:W{fold}0"
        papers = [
            *[
                _paper(
                    f"openalex:W{fold}{index:02d}",
                    f"unrelated topic {index}",
                )
                for index in range(1, 26)
            ],
            _paper(positive_id, "relevant"),
        ]
        candidates = [
            DocumentCandidateEvidence(
                paper=paper,
                baseline_score=1.0 / (60 + rank),
                source_ranks={"lexical": rank},
            )
            for rank, paper in enumerate(papers, start=1)
        ]
        folded.append(
            (
                fold,
                DocumentRankingQuery(
                    query_id=f"target-{fold}",
                    query="relevant",
                    gold_paper_ids=[positive_id],
                    candidates=candidates,
                ),
            )
        )
    supplemental = [
        DocumentRankingQuery(
            query_id=f"supplemental-{index}",
            query="relevant",
            gold_paper_ids=["openalex:W901"],
            candidates=build_document_candidates(
                [("lexical", [_paper("openalex:W901", "relevant")])]
            ),
        )
        for index in (1, 2)
    ]

    result = evaluate_document_ranking_oof_expansion(
        folded,
        supplemental_training_queries=supplemental,
        cutoffs=(1, 10, 20, 50),
        baseline_ranker_factory=lambda seed: RecordingRanker(
            f"baseline-{seed}", reverse=False
        ),
        candidate_ranker_factory=lambda seed: RecordingRanker(
            f"candidate-{seed}", reverse=True
        ),
    )

    for fold in (1, 2, 3):
        baseline_ids = next(
            ids for label, ids in fit_calls if label == f"baseline-{1000 + fold}"
        )
        candidate_ids = next(
            ids for label, ids in fit_calls if label == f"candidate-{2000 + fold}"
        )
        assert baseline_ids == {
            f"target-{other}" for other in (1, 2, 3) if other != fold
        }
        assert candidate_ids == baseline_ids | {
            "supplemental-1",
            "supplemental-2",
        }
        assert f"target-{fold}" not in candidate_ids
    assert result["baseline"]["macro_recall_at"][1] == 0.0
    assert result["candidate"]["macro_recall_at"][1] == 1.0
    assert result["candidate_pool_identity_unchanged"] is True
    assert result["promotion"]["promote"] is True, (
        result["promotion"],
        result["baseline"],
        result["candidate"],
    )
    assert result["supplemental_training_query_count"] == 2


def test_expanded_oof_rejects_supplemental_query_id_overlap() -> None:
    query = DocumentRankingQuery(
        query_id="overlap",
        query="relevant",
        gold_paper_ids=["openalex:W1"],
        candidates=build_document_candidates(
            [("lexical", [_paper("openalex:W1", "relevant")])]
        ),
    )

    try:
        evaluate_document_ranking_oof_expansion(
            [
                (1, query),
                (2, query.model_copy(update={"query_id": "fold-2"})),
                (3, query.model_copy(update={"query_id": "fold-3"})),
            ],
            supplemental_training_queries=[query],
        )
    except ValueError as error:
        assert "disjoint" in str(error)
    else:
        raise AssertionError("overlapping supplemental query must be rejected")


def test_trained_ranker_comparison_uses_current_model_as_baseline() -> None:
    class IdentityRanker:
        def rank(self, query, candidates):
            del query
            return list(candidates)

    class ReverseRanker:
        def rank(self, query, candidates):
            del query
            return list(reversed(candidates))

    folded = []
    for fold in (1, 2, 3):
        gold = f"openalex:W{fold}0"
        papers = [
            *[
                _paper(f"openalex:W{fold}{index:02d}", f"topic {index}")
                for index in range(1, 26)
            ],
            _paper(gold, "relevant"),
        ]
        candidates = [
            DocumentCandidateEvidence(
                paper=paper,
                baseline_score=1.0 / (60 + rank),
                source_ranks={"lexical": rank},
            )
            for rank, paper in enumerate(papers, start=1)
        ]
        folded.append(
            (
                fold,
                DocumentRankingQuery(
                    query_id=f"comparison-{fold}",
                    query="relevant",
                    gold_paper_ids=[gold],
                    candidates=candidates,
                ),
            )
        )

    result = evaluate_document_ranker_comparison(
        folded,
        baseline_ranker=IdentityRanker(),
        candidate_ranker=ReverseRanker(),
        cutoffs=(5, 10, 20, 50),
    )

    assert result["baseline"]["macro_recall_at"][10] == 0.0
    assert result["candidate"]["macro_recall_at"][10] == 1.0
    assert result["candidate_pool_identity_unchanged"] is True
    assert result["promotion"]["promote"] is True
