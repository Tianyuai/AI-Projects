"""Cross-fold evaluation for CPU document ranking over frozen candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from paper_search.evaluation.predictions import paper_evaluation_id
from paper_search.learning.cpu_document_ranker import (
    CpuPairwiseDocumentRanker,
    DocumentCandidateEvidence,
    DocumentRankingQuery,
)
from paper_search.learning.document_ranking_gate import (
    decide_document_ranking_promotion,
)


RankerFactory = Callable[[int], CpuPairwiseDocumentRanker]


def _metrics(
    queries: Sequence[DocumentRankingQuery],
    rankings: Mapping[str, Sequence[DocumentCandidateEvidence]],
    cutoffs: Sequence[int],
) -> dict[str, object]:
    if not queries:
        raise ValueError("document ranking metrics require queries")
    recall_at = {cutoff: 0.0 for cutoff in cutoffs}
    hit_query_at = {cutoff: 0 for cutoff in cutoffs}
    gold_hits_at = {cutoff: 0 for cutoff in cutoffs}
    oracle = 0.0
    for query in queries:
        gold = set(query.gold_paper_ids)
        candidate_ids = {
            paper_evaluation_id(candidate.paper) for candidate in query.candidates
        }
        oracle += len(gold & candidate_ids) / len(gold)
        ranked_ids = list(
            dict.fromkeys(
                paper_evaluation_id(candidate.paper)
                for candidate in rankings[query.query_id]
            )
        )
        for cutoff in cutoffs:
            hits = len(gold.intersection(ranked_ids[:cutoff]))
            recall_at[cutoff] += hits / len(gold)
            hit_query_at[cutoff] += int(hits > 0)
            gold_hits_at[cutoff] += hits
    count = len(queries)
    return {
        "query_count": count,
        "candidate_oracle_macro_recall": oracle / count,
        "macro_recall_at": {
            cutoff: recall_at[cutoff] / count for cutoff in cutoffs
        },
        "hit_query_at": hit_query_at,
        "gold_hits_at": gold_hits_at,
    }


def evaluate_document_ranking_oof(
    folded_queries: Sequence[tuple[int, DocumentRankingQuery]],
    *,
    cutoffs: Sequence[int] = (5, 10, 20, 50),
    ranker_factory: RankerFactory | None = None,
) -> dict[str, object]:
    """Train on two folds and rank only the held-out third fold."""

    rows = [
        (fold, DocumentRankingQuery.model_validate(query))
        for fold, query in folded_queries
    ]
    folds = sorted({fold for fold, _query in rows})
    if folds != [1, 2, 3]:
        raise ValueError("document ranking OOF requires folds 1, 2, and 3")
    if len({query.query_id for _fold, query in rows}) != len(rows):
        raise ValueError("document ranking OOF query ids must be unique")
    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not {10, 20, 50}.issubset(normalized_cutoffs):
        raise ValueError("document ranking OOF requires cutoffs 10, 20, and 50")
    factory = ranker_factory or (
        lambda seed: CpuPairwiseDocumentRanker(seed=seed)
    )

    baseline_rankings = {
        query.query_id: list(query.candidates) for _fold, query in rows
    }
    candidate_rankings: dict[str, list[DocumentCandidateEvidence]] = {}
    fold_output: dict[str, dict[str, Any]] = {}
    for fold in folds:
        train = [query for row_fold, query in rows if row_fold != fold]
        evaluation = [query for row_fold, query in rows if row_fold == fold]
        ranker = factory(1000 + fold)
        preference_pair_count = ranker.fit(train)
        for query in evaluation:
            candidate_rankings[query.query_id] = ranker.rank(
                query.query, query.candidates
            )
        baseline_fold = _metrics(
            evaluation, baseline_rankings, normalized_cutoffs
        )
        candidate_fold = _metrics(
            evaluation, candidate_rankings, normalized_cutoffs
        )
        fold_output[str(fold)] = {
            "train_query_count": len(train),
            "evaluation_query_count": len(evaluation),
            "preference_pair_count": preference_pair_count,
            "baseline": baseline_fold,
            "candidate": candidate_fold,
        }

    queries = [query for _fold, query in rows]
    baseline = _metrics(queries, baseline_rankings, normalized_cutoffs)
    candidate = _metrics(queries, candidate_rankings, normalized_cutoffs)
    identity_unchanged = all(
        {
            paper_evaluation_id(item.paper)
            for item in baseline_rankings[query.query_id]
        }
        == {
            paper_evaluation_id(item.paper)
            for item in candidate_rankings[query.query_id]
        }
        for query in queries
    )
    baseline_by_fold = {
        fold: fold_output[str(fold)]["baseline"]["macro_recall_at"]
        for fold in folds
    }
    candidate_by_fold = {
        fold: fold_output[str(fold)]["candidate"]["macro_recall_at"]
        for fold in folds
    }
    return {
        "schema_version": "cpu-document-ranking-oof-v1",
        "query_count": len(rows),
        "baseline": baseline,
        "candidate": candidate,
        "folds": fold_output,
        "candidate_pool_identity_unchanged": identity_unchanged,
        "promotion": decide_document_ranking_promotion(
            baseline=baseline["macro_recall_at"],
            candidate=candidate["macro_recall_at"],
            baseline_by_fold=baseline_by_fold,
            candidate_by_fold=candidate_by_fold,
        ),
        "test_partition_touched": False,
    }


def evaluate_document_ranking_oof_expansion(
    folded_queries: Sequence[tuple[int, DocumentRankingQuery]],
    *,
    supplemental_training_queries: Sequence[DocumentRankingQuery],
    cutoffs: Sequence[int] = (5, 10, 20, 50),
    baseline_ranker_factory: RankerFactory | None = None,
    candidate_ranker_factory: RankerFactory | None = None,
) -> dict[str, object]:
    """Compare current OOF training with a disjoint supplemental training pool."""

    rows = [
        (fold, DocumentRankingQuery.model_validate(query))
        for fold, query in folded_queries
    ]
    supplemental = [
        DocumentRankingQuery.model_validate(query)
        for query in supplemental_training_queries
    ]
    folds = sorted({fold for fold, _query in rows})
    if folds != [1, 2, 3]:
        raise ValueError("document ranking OOF expansion requires folds 1, 2, and 3")
    target_ids = [query.query_id for _fold, query in rows]
    supplemental_ids = [query.query_id for query in supplemental]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("document ranking OOF query ids must be unique")
    if not supplemental or len(set(supplemental_ids)) != len(supplemental_ids):
        raise ValueError("supplemental training query ids must be non-empty and unique")
    if set(target_ids) & set(supplemental_ids):
        raise ValueError("supplemental training queries must be disjoint from OOF queries")
    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not {10, 20, 50}.issubset(normalized_cutoffs):
        raise ValueError("document ranking OOF expansion requires cutoffs 10, 20, and 50")
    baseline_factory = baseline_ranker_factory or (
        lambda seed: CpuPairwiseDocumentRanker(seed=seed)
    )
    candidate_factory = candidate_ranker_factory or (
        lambda seed: CpuPairwiseDocumentRanker(seed=seed)
    )

    baseline_rankings: dict[str, list[DocumentCandidateEvidence]] = {}
    candidate_rankings: dict[str, list[DocumentCandidateEvidence]] = {}
    fold_output: dict[str, dict[str, Any]] = {}
    for fold in folds:
        target_train = [query for row_fold, query in rows if row_fold != fold]
        evaluation = [query for row_fold, query in rows if row_fold == fold]
        baseline_ranker = baseline_factory(1000 + fold)
        candidate_ranker = candidate_factory(2000 + fold)
        baseline_pair_count = baseline_ranker.fit(target_train)
        candidate_pair_count = candidate_ranker.fit([*target_train, *supplemental])
        for query in evaluation:
            baseline_rankings[query.query_id] = baseline_ranker.rank(
                query.query, query.candidates
            )
            candidate_rankings[query.query_id] = candidate_ranker.rank(
                query.query, query.candidates
            )
        baseline_fold = _metrics(
            evaluation, baseline_rankings, normalized_cutoffs
        )
        candidate_fold = _metrics(
            evaluation, candidate_rankings, normalized_cutoffs
        )
        fold_output[str(fold)] = {
            "target_train_query_count": len(target_train),
            "supplemental_train_query_count": len(supplemental),
            "evaluation_query_count": len(evaluation),
            "baseline_preference_pair_count": baseline_pair_count,
            "candidate_preference_pair_count": candidate_pair_count,
            "baseline": baseline_fold,
            "candidate": candidate_fold,
        }

    queries = [query for _fold, query in rows]
    baseline = _metrics(queries, baseline_rankings, normalized_cutoffs)
    candidate = _metrics(queries, candidate_rankings, normalized_cutoffs)
    identity_unchanged = all(
        {
            paper_evaluation_id(item.paper)
            for item in query.candidates
        }
        == {
            paper_evaluation_id(item.paper)
            for item in baseline_rankings[query.query_id]
        }
        == {
            paper_evaluation_id(item.paper)
            for item in candidate_rankings[query.query_id]
        }
        for query in queries
    )
    baseline_by_fold = {
        fold: fold_output[str(fold)]["baseline"]["macro_recall_at"]
        for fold in folds
    }
    candidate_by_fold = {
        fold: fold_output[str(fold)]["candidate"]["macro_recall_at"]
        for fold in folds
    }
    return {
        "schema_version": "cpu-document-ranking-oof-expansion-v1",
        "query_count": len(rows),
        "supplemental_training_query_count": len(supplemental),
        "baseline": baseline,
        "candidate": candidate,
        "folds": fold_output,
        "candidate_pool_identity_unchanged": identity_unchanged,
        "promotion": decide_document_ranking_promotion(
            baseline=baseline["macro_recall_at"],
            candidate=candidate["macro_recall_at"],
            baseline_by_fold=baseline_by_fold,
            candidate_by_fold=candidate_by_fold,
        ),
        "test_partition_touched": False,
    }


def evaluate_document_ranker(
    folded_queries: Sequence[tuple[int, DocumentRankingQuery]],
    *,
    ranker: CpuPairwiseDocumentRanker,
    cutoffs: Sequence[int] = (5, 10, 20, 50),
) -> dict[str, object]:
    """Evaluate one already-trained ranker on an untouched folded query set."""

    rows = [
        (fold, DocumentRankingQuery.model_validate(query))
        for fold, query in folded_queries
    ]
    folds = sorted({fold for fold, _query in rows})
    if folds != [1, 2, 3]:
        raise ValueError("document ranker evaluation requires folds 1, 2, and 3")
    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not {10, 20, 50}.issubset(normalized_cutoffs):
        raise ValueError("document ranker evaluation requires cutoffs 10, 20, and 50")
    baseline_rankings = {
        query.query_id: list(query.candidates) for _fold, query in rows
    }
    candidate_rankings = {
        query.query_id: ranker.rank(query.query, query.candidates)
        for _fold, query in rows
    }
    queries = [query for _fold, query in rows]
    baseline = _metrics(queries, baseline_rankings, normalized_cutoffs)
    candidate = _metrics(queries, candidate_rankings, normalized_cutoffs)
    fold_output: dict[str, dict[str, object]] = {}
    for fold in folds:
        evaluation = [query for row_fold, query in rows if row_fold == fold]
        fold_output[str(fold)] = {
            "query_count": len(evaluation),
            "baseline": _metrics(
                evaluation, baseline_rankings, normalized_cutoffs
            ),
            "candidate": _metrics(
                evaluation, candidate_rankings, normalized_cutoffs
            ),
        }
    identity_unchanged = all(
        {
            paper_evaluation_id(item.paper)
            for item in baseline_rankings[query.query_id]
        }
        == {
            paper_evaluation_id(item.paper)
            for item in candidate_rankings[query.query_id]
        }
        for query in queries
    )
    return {
        "schema_version": "cpu-document-ranking-independent-evaluation-v1",
        "query_count": len(rows),
        "baseline": baseline,
        "candidate": candidate,
        "folds": fold_output,
        "candidate_pool_identity_unchanged": identity_unchanged,
        "promotion": decide_document_ranking_promotion(
            baseline=baseline["macro_recall_at"],
            candidate=candidate["macro_recall_at"],
            baseline_by_fold={
                fold: fold_output[str(fold)]["baseline"]["macro_recall_at"]
                for fold in folds
            },
            candidate_by_fold={
                fold: fold_output[str(fold)]["candidate"]["macro_recall_at"]
                for fold in folds
            },
        ),
        "test_partition_touched": False,
    }


def evaluate_document_ranker_comparison(
    folded_queries: Sequence[tuple[int, DocumentRankingQuery]],
    *,
    baseline_ranker: Any,
    candidate_ranker: Any,
    cutoffs: Sequence[int] = (5, 10, 20, 50),
) -> dict[str, object]:
    """Compare two trained rankers on exactly the same folded candidates."""

    rows = [
        (fold, DocumentRankingQuery.model_validate(query))
        for fold, query in folded_queries
    ]
    folds = sorted({fold for fold, _query in rows})
    if folds != [1, 2, 3]:
        raise ValueError("document ranker comparison requires folds 1, 2, and 3")
    if len({query.query_id for _fold, query in rows}) != len(rows):
        raise ValueError("document ranker comparison query ids must be unique")
    normalized_cutoffs = tuple(sorted(set(cutoffs)))
    if not {10, 20, 50}.issubset(normalized_cutoffs):
        raise ValueError("document ranker comparison requires cutoffs 10, 20, and 50")
    baseline_rankings = {
        query.query_id: baseline_ranker.rank(query.query, query.candidates)
        for _fold, query in rows
    }
    candidate_rankings = {
        query.query_id: candidate_ranker.rank(query.query, query.candidates)
        for _fold, query in rows
    }
    queries = [query for _fold, query in rows]
    baseline = _metrics(queries, baseline_rankings, normalized_cutoffs)
    candidate = _metrics(queries, candidate_rankings, normalized_cutoffs)
    fold_output: dict[str, dict[str, object]] = {}
    for fold in folds:
        evaluation = [query for row_fold, query in rows if row_fold == fold]
        fold_output[str(fold)] = {
            "query_count": len(evaluation),
            "baseline": _metrics(
                evaluation, baseline_rankings, normalized_cutoffs
            ),
            "candidate": _metrics(
                evaluation, candidate_rankings, normalized_cutoffs
            ),
        }
    identity_unchanged = all(
        {
            paper_evaluation_id(item.paper)
            for item in query.candidates
        }
        == {
            paper_evaluation_id(item.paper)
            for item in baseline_rankings[query.query_id]
        }
        == {
            paper_evaluation_id(item.paper)
            for item in candidate_rankings[query.query_id]
        }
        for query in queries
    )
    return {
        "schema_version": "cpu-document-ranker-comparison-v1",
        "query_count": len(rows),
        "baseline": baseline,
        "candidate": candidate,
        "folds": fold_output,
        "candidate_pool_identity_unchanged": identity_unchanged,
        "promotion": decide_document_ranking_promotion(
            baseline=baseline["macro_recall_at"],
            candidate=candidate["macro_recall_at"],
            baseline_by_fold={
                fold: fold_output[str(fold)]["baseline"]["macro_recall_at"]
                for fold in folds
            },
            candidate_by_fold={
                fold: fold_output[str(fold)]["candidate"]["macro_recall_at"]
                for fold in folds
            },
        ),
        "test_partition_touched": False,
    }


__all__ = [
    "evaluate_document_ranker",
    "evaluate_document_ranker_comparison",
    "evaluate_document_ranking_oof",
    "evaluate_document_ranking_oof_expansion",
]
