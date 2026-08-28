from __future__ import annotations

import numpy as np

from paper_search.domain.models import Paper
from paper_search.learning.cpu_document_ranker import (
    DocumentCandidateEvidence,
    DocumentRankingQuery,
)
from paper_search.learning.fusion_activation import (
    audit_fusion_query_activation,
    build_activation_freeze,
)
from paper_search.learning.gated_feature_fusion_ranker import (
    FusionQueryContext,
    GatedFeatureFusionRanker,
    UnifiedFusionContextResolver,
    _index,
    bounded_entity_preference_pairs,
    entity_pair_signals,
    gated_family_candidate_features,
    training_candidate_eligible_for_family,
)
from paper_search.learning import gated_feature_fusion_ranker as fusion_ranker
from paper_search.retrieval.pasa_paper_database import (
    ARXIV_MISSING_YEAR_EVIDENCE_POLICY,
    PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE,
    PASA_TRAINING_GOLD_INJECTED_SOURCE,
)
from paper_search.learning.query_constraint_annotations import (
    FrozenConstraintAnnotation,
    FrozenConstraintProfileStore,
    query_sha256,
)
from paper_search.learning.query_constraint_profile import QueryConstraintProfile
from paper_search.learning.task_slot_document_ranker import FrozenTaskSlotLabelStore


class _IdentityRanker:
    def rank(self, query, candidates):
        return list(candidates)


def test_method_usage_evidence_rejects_background_and_comparison_mentions() -> None:
    context = FusionQueryContext(
        constraint_profile=QueryConstraintProfile(
            labels=["method"],
            methods=["gans"],
            confidence=1.0,
            constraint_count=1,
        )
    )
    background = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="arxiv:2304.05568",
            arxiv_id="2304.05568",
            title="Improving Diffusion Models for Scene Text Editing",
            abstract=(
                "Most previous approaches rely on style-transfer models, such as "
                "GANs. However, we propose a diffusion model with dual encoders."
            ),
        ),
        baseline_score=1.0,
        source_ranks={"openalex": 1},
    )
    comparison = background.model_copy(
        update={
            "paper": background.paper.model_copy(
                update={"abstract": "Unlike GANs, our method uses diffusion models."}
            )
        }
    )

    assert gated_family_candidate_features(
        context,
        background,
        baseline_rank=1,
        family="entity",
        constraint_text_evidence=True,
    )["entity-method-text-match"] == 0.0
    assert gated_family_candidate_features(
        context,
        comparison,
        baseline_rank=1,
        family="entity",
        constraint_text_evidence=True,
    )["entity-method-text-match"] == 0.0


def test_method_usage_evidence_accepts_affirmative_own_use() -> None:
    context = FusionQueryContext(
        constraint_profile=QueryConstraintProfile(
            labels=["method"],
            methods=["gans"],
            confidence=1.0,
            constraint_count=1,
        )
    )
    candidate = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="arxiv:2301.00001",
            arxiv_id="2301.00001",
            title="Scene Text Editing Using GANs",
            abstract="We propose a framework using GANs for scene text editing.",
        ),
        baseline_score=1.0,
        source_ranks={"openalex": 1},
    )

    assert gated_family_candidate_features(
        context,
        candidate,
        baseline_rank=1,
        family="entity",
        constraint_text_evidence=True,
    )["entity-method-text-match"] == 1.0


def test_anchored_family_weights_replace_only_selected_family() -> None:
    candidate = {
        "entity": np.array([1.0, 2.0]),
        "task_provenance": np.array([3.0, 4.0]),
    }
    production = {
        "entity": np.array([10.0, 20.0]),
        "task_provenance": np.array([30.0, 40.0]),
    }

    anchored = fusion_ranker.anchored_family_weights(
        candidate,
        production,
        families={"task_provenance"},
    )

    np.testing.assert_array_equal(anchored["entity"], candidate["entity"])
    np.testing.assert_array_equal(
        anchored["task_provenance"], production["task_provenance"]
    )
    assert anchored["entity"] is not candidate["entity"]
    assert anchored["task_provenance"] is not production["task_provenance"]


def test_year_evidence_falls_back_to_arxiv_without_overriding_declared_year() -> None:
    context = FusionQueryContext(
        constraint_profile=QueryConstraintProfile(
            labels=["year"],
            year_from=2021,
            confidence=1.0,
            constraint_count=1,
        )
    )
    derived = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="arxiv:2007.07646",
            arxiv_id="2007.07646v2",
            title="Derived year",
        ),
        baseline_score=1.0,
        source_ranks={"pasa": 1},
    )
    declared = derived.model_copy(
        update={"paper": derived.paper.model_copy(update={"publication_year": 2022})}
    )
    malformed = derived.model_copy(
        update={"paper": derived.paper.model_copy(update={"arxiv_id": "bad-id"})}
    )

    assert gated_family_candidate_features(
        context,
        derived,
        baseline_rank=1,
        family="hard_constraint",
        publication_year_evidence_policy=ARXIV_MISSING_YEAR_EVIDENCE_POLICY,
    ) == {"hard-year-compliant": 0.0, "hard-year-conflict": 1.0}
    assert gated_family_candidate_features(
        context,
        declared,
        baseline_rank=1,
        family="hard_constraint",
        publication_year_evidence_policy=ARXIV_MISSING_YEAR_EVIDENCE_POLICY,
    ) == {"hard-year-compliant": 1.0, "hard-year-conflict": 0.0}
    assert gated_family_candidate_features(
        context,
        malformed,
        baseline_rank=1,
        family="hard_constraint",
        publication_year_evidence_policy=ARXIV_MISSING_YEAR_EVIDENCE_POLICY,
    ) == {"hard-year-missing": 1.0}


def test_entity_pair_cap_preserves_each_available_signal() -> None:
    context = FusionQueryContext(
        constraint_profile=QueryConstraintProfile(
            labels=["method", "dataset"],
            methods=["m"],
            datasets=["d"],
            confidence=1.0,
            constraint_count=2,
        )
    )
    positive = {
        "entity-method-text-match": 1.0,
        "entity-dataset-text-match": 1.0,
    }
    method_only_negative = {
        "entity-method-text-match": 0.0,
        "entity-dataset-text-match": 1.0,
    }
    dataset_only_negative = {
        "entity-method-text-match": 1.0,
        "entity-dataset-text-match": 0.0,
    }

    pairs = bounded_entity_preference_pairs(
        context,
        [positive],
        [method_only_negative, method_only_negative, dataset_only_negative],
        hard_negative_limit=3,
        pair_limit=2,
    )

    assert {
        signal
        for pair in pairs
        for signal in entity_pair_signals(context, *pair)
    } == {"method", "dataset"}


def test_targeted_constraint_overlay_is_training_only_for_hard_constraints() -> None:
    candidate = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="arxiv:2301.00001",
            arxiv_id="2301.00001",
            title="Vision model using adversarial training",
            sources=["pasa_paper_database", PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE],
        ),
        baseline_score=0.5,
        source_ranks={"pasa-negation-overlay": 1},
    )

    assert training_candidate_eligible_for_family(
        candidate, "hard_constraint", is_gold=False
    )
    assert not training_candidate_eligible_for_family(
        candidate, "reliability", is_gold=False
    )
    assert not training_candidate_eligible_for_family(
        candidate, "task_provenance", is_gold=False
    )
    assert not training_candidate_eligible_for_family(
        candidate, "entity", is_gold=False
    )


def test_training_only_pasa_candidates_do_not_reactivate_source_families_at_rank_time() -> None:
    gold_injected = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="arxiv:2301.00002",
            arxiv_id="2301.00002",
            title="Training-only Gold",
            sources=["pasa_paper_database", PASA_TRAINING_GOLD_INJECTED_SOURCE],
        ),
        baseline_score=0.5,
        source_ranks={"pasa-local-original@receipt": 1},
    )
    constraint_injected = gold_injected.model_copy(
        update={
            "paper": gold_injected.paper.model_copy(
                update={
                    "sources": [
                        "pasa_paper_database",
                        PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE,
                    ]
                }
            )
        }
    )

    assert not fusion_ranker.runtime_candidate_eligible_for_family(
        gold_injected, "reliability"
    )
    assert not fusion_ranker.runtime_candidate_eligible_for_family(
        gold_injected, "task_provenance"
    )
    assert not fusion_ranker.runtime_candidate_eligible_for_family(
        constraint_injected, "reliability"
    )
    assert fusion_ranker.runtime_candidate_eligible_for_family(
        constraint_injected, "hard_constraint"
    )


def _query(*, distinguishing_negative: bool) -> DocumentRankingQuery:
    query = "papers using vision transformer methods"
    positive = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="openalex:W1",
            openalex_id="W1",
            title="Recognition Using Vision Transformer",
        ),
        baseline_score=1.0,
        source_ranks={"lexical": 1},
    )
    negative = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="openalex:W2",
            openalex_id="W2",
            title=(
                "Convolutional recognition"
                if distinguishing_negative
                else "Recognition Using Vision Transformer"
            ),
        ),
        baseline_score=0.5,
        source_ranks={"lexical": 2},
    )
    return DocumentRankingQuery(
        query_id="q1" if distinguishing_negative else "q2",
        query=query,
        gold_paper_ids=["openalex:W1"],
        candidates=[positive, negative],
    )


def _ranker() -> GatedFeatureFusionRanker:
    query = "papers using vision transformer methods"
    annotation = FrozenConstraintAnnotation(
        query_id="q1",
        query_sha256=query_sha256(query),
        role="training",
        split="auto_train",
        labels=["method"],
        methods=["vision transformer"],
        label_sources={"method": "human_review"},
        label_confidence={"method": 1.0},
        evidence={"method": ["vision transformer"]},
        status="accepted",
    )
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([annotation]),
    )
    return GatedFeatureFusionRanker(
        baseline_ranker=_IdentityRanker(),
        context_store=resolver,
        feature_families={"entity", "reliability"},
        epochs=1,
        dimension=32,
    )


def test_activation_requires_a_gold_negative_feature_difference() -> None:
    ranker = _ranker()

    effective = audit_fusion_query_activation(_query(distinguishing_negative=True), ranker)
    ineffective = audit_fusion_query_activation(
        _query(distinguishing_negative=False), ranker
    )

    assert effective["families"]["entity"]["gate_eligible"] is True
    assert effective["families"]["entity"]["effective_pair_count"] == 1
    assert ineffective["families"]["entity"]["gate_eligible"] is True
    assert ineffective["families"]["entity"]["effective_pair_count"] == 0
    assert ineffective["families"]["entity"]["reason"] == "no_feature_contrast"


def test_entity_activation_rejects_method_match_only_on_negative() -> None:
    query = _query(distinguishing_negative=True).model_copy(
        update={
            "candidates": [
                DocumentCandidateEvidence(
                    paper=Paper(
                        canonical_id="openalex:W1",
                        openalex_id="W1",
                        title="Convolutional recognition",
                    ),
                    baseline_score=1.0,
                    source_ranks={"lexical": 1},
                ),
                DocumentCandidateEvidence(
                    paper=Paper(
                        canonical_id="openalex:W2",
                        openalex_id="W2",
                title="Recognition Using Vision Transformer",
                    ),
                    baseline_score=0.5,
                    source_ranks={"lexical": 2},
                ),
            ]
        }
    )

    activation = audit_fusion_query_activation(query, _ranker())

    assert activation["families"]["entity"]["effective_pair_count"] == 0
    assert activation["families"]["entity"]["signal_effective_pair_count"] == {}


def test_method_pair_learns_positive_text_match_direction() -> None:
    ranker = _ranker()
    ranker.constraint_text_evidence = True
    query = _query(distinguishing_negative=True)

    pair_counts = ranker.fit([query])
    activation = audit_fusion_query_activation(query, ranker)

    assert pair_counts["entity"] == 1
    assert ranker.weights["entity"][_index("entity-method-text-match", 32)] > 0.0
    assert activation["families"]["entity"]["signal_effective_pair_count"] == {
        "method": 1
    }


def test_activation_freeze_separates_effective_rows_from_candidate_backfill() -> None:
    ranker = _ranker()

    report = build_activation_freeze(
        [
            _query(distinguishing_negative=True),
            _query(distinguishing_negative=False),
        ],
        ranker,
    )

    assert report["test_partition_touched"] is False
    assert report["online_requests_made"] == 0
    assert report["coverage"]["entity"] == {
        "eligible_query_count": 2,
        "effective_query_count": 1,
        "effective_pair_count": 1,
    }
    assert report["selected_query_ids_by_family"]["entity"] == ["q1"]
    assert report["candidate_backfill_queue"] == [
        {
            "query_id": "q2",
            "families": ["entity"],
            "reason": "no_feature_contrast",
        }
    ]


def test_fusion_fit_builds_pairs_from_arxiv_datacite_alias_match() -> None:
    query = _query(distinguishing_negative=True).model_copy(
        update={
            "gold_paper_ids": ["arxiv:2301.01234"],
            "candidates": [
                DocumentCandidateEvidence(
                    paper=Paper(
                        canonical_id="doi:10.48550/arxiv.2301.01234",
                        doi="10.48550/arxiv.2301.01234",
                        title="Vision transformer for recognition",
                    ),
                    baseline_score=1.0,
                    source_ranks={"lexical": 1},
                ),
                DocumentCandidateEvidence(
                    paper=Paper(
                        canonical_id="openalex:W2",
                        openalex_id="W2",
                        title="Convolutional recognition",
                    ),
                    baseline_score=0.5,
                    source_ranks={"lexical": 2},
                ),
            ],
        }
    )

    ranker = _ranker()
    pair_counts = ranker.fit([query])
    activation = audit_fusion_query_activation(query, ranker)

    assert pair_counts["reliability"] == 1
    assert activation["families"]["reliability"]["effective_pair_count"] == 1


def test_fusion_caps_pairs_per_query_and_family() -> None:
    base = _query(distinguishing_negative=True)
    candidates = list(base.candidates)
    candidates.extend(
        DocumentCandidateEvidence(
            paper=Paper(
                canonical_id=f"openalex:W{index}",
                openalex_id=f"W{index}",
                title=f"Negative {index}",
            ),
            baseline_score=1.0 / index,
            source_ranks={"lexical": index},
        )
        for index in range(3, 8)
    )
    query = base.model_copy(update={"candidates": candidates})
    ranker = _ranker()
    ranker.max_pairs_per_query_family = 3

    pair_counts = ranker.fit([query])
    activation = audit_fusion_query_activation(query, ranker)

    assert pair_counts["reliability"] == 3
    assert activation["families"]["reliability"]["effective_pair_count"] == 3


def test_fusion_uses_distinct_pair_budget_for_each_feature_family() -> None:
    base = _query(distinguishing_negative=True)
    candidates = list(base.candidates)
    candidates.extend(
        DocumentCandidateEvidence(
            paper=Paper(
                canonical_id=f"openalex:W{index}",
                openalex_id=f"W{index}",
                title=f"Negative {index}",
            ),
            baseline_score=1.0 / index,
            source_ranks={"lexical": index},
        )
        for index in range(3, 8)
    )
    query = base.model_copy(update={"candidates": candidates})
    ranker = _ranker()
    ranker.constraint_text_evidence = True
    ranker.pair_budget_by_family = {"entity": 1, "reliability": 3}

    pair_counts = ranker.fit([query])

    assert pair_counts["entity"] == 1
    assert pair_counts["reliability"] == 3


def test_gold_injected_pasa_candidate_is_excluded_from_source_family_pairs() -> None:
    query = "papers using vision transformer methods"
    positive = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="arxiv:2301.01234",
            arxiv_id="2301.01234",
            title="Recognition Using Vision Transformer",
            sources=["pasa_paper_database", PASA_TRAINING_GOLD_INJECTED_SOURCE],
        ),
        baseline_score=0.5,
        source_ranks={"pasa-local-original@receipt": 21},
    )
    negative = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="arxiv:2301.05678",
            arxiv_id="2301.05678",
            title="Convolutional recognition",
            sources=["pasa_paper_database"],
        ),
        baseline_score=1.0,
        source_ranks={"pasa-local-original@receipt": 1},
    )
    ranking_query = DocumentRankingQuery(
        query_id="q1",
        query=query,
        gold_paper_ids=["arxiv:2301.01234"],
        candidates=[positive, negative],
    )
    ranker = _ranker()

    pair_counts = ranker.fit([ranking_query])
    activation = audit_fusion_query_activation(ranking_query, ranker)

    assert pair_counts["reliability"] == 0
    assert activation["families"]["reliability"]["effective_pair_count"] == 0
    assert pair_counts["entity"] == 1
    assert activation["families"]["entity"]["effective_pair_count"] == 1


def test_legacy_unmarked_pasa_gold_is_suppressed_by_training_role() -> None:
    query = "papers using vision transformer methods"
    positive = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="arxiv:2301.01234",
            arxiv_id="2301.01234",
            title="Recognition Using Vision Transformer",
            sources=["pasa_paper_database"],
        ),
        baseline_score=0.5,
        source_ranks={"pasa-local-original@legacy": 21},
    )
    negative = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="arxiv:2301.05678",
            arxiv_id="2301.05678",
            title="Convolutional recognition",
            sources=["pasa_paper_database"],
        ),
        baseline_score=1.0,
        source_ranks={"pasa-local-original@legacy": 1},
    )
    ranking_query = DocumentRankingQuery(
        query_id="q1",
        query=query,
        gold_paper_ids=["arxiv:2301.01234"],
        candidates=[positive, negative],
    )
    ranker = _ranker()
    ranker.constraint_text_evidence = True

    pair_counts = ranker.fit([ranking_query])
    activation = audit_fusion_query_activation(ranking_query, ranker)

    assert pair_counts["reliability"] == 0
    assert activation["families"]["reliability"]["effective_pair_count"] == 0
    assert pair_counts["entity"] == 1
    assert ranker.weights["entity"][_index("entity-method-text-match", 32)] > 0.0


def test_negation_pair_learns_clean_positive_and_conflict_negative_direction() -> None:
    query = "vision models without adversarial training"
    annotation = FrozenConstraintAnnotation(
        query_id="q-negation",
        query_sha256=query_sha256(query),
        role="training",
        split="auto_train",
        labels=["negation"],
        exclusions=["adversarial training"],
        label_sources={"negation": "local_deterministic"},
        label_confidence={"negation": 1.0},
        evidence={"negation": ["adversarial training"]},
        status="accepted",
    )
    ranker = GatedFeatureFusionRanker(
        baseline_ranker=_IdentityRanker(),
        context_store=UnifiedFusionContextResolver(
            task_store=FrozenTaskSlotLabelStore({}),
            constraint_store=FrozenConstraintProfileStore([annotation]),
        ),
        feature_families={"hard_constraint"},
        epochs=1,
        dimension=2048,
        constraint_text_evidence=True,
    )
    ranking_query = DocumentRankingQuery(
        query_id="q-negation",
        query=query,
        gold_paper_ids=["openalex:W1"],
        candidates=[
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id="openalex:W1",
                    openalex_id="W1",
                    title="Vision model without adversarial training",
                ),
                baseline_score=1.0,
                source_ranks={"lexical": 1},
            ),
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id="openalex:W2",
                    openalex_id="W2",
                    title="Vision model using adversarial training",
                ),
                baseline_score=0.5,
                source_ranks={"lexical": 2},
            ),
        ],
    )

    pair_counts = ranker.fit([ranking_query])

    assert pair_counts["hard_constraint"] == 1
    assert ranker.weights["hard_constraint"][
        _index("hard-negation-clean", 2048)
    ] > 0.0
    assert ranker.weights["hard_constraint"][
        _index("hard-negation-conflict", 2048)
    ] < 0.0


def _negation_ranker(*, hard_negative_limit: int = 100) -> GatedFeatureFusionRanker:
    query = "vision models without adversarial training"
    annotation = FrozenConstraintAnnotation(
        query_id="q-negation",
        query_sha256=query_sha256(query),
        role="training",
        split="auto_train",
        labels=["negation"],
        exclusions=["adversarial training?"],
        label_sources={"negation": "local_deterministic"},
        label_confidence={"negation": 1.0},
        evidence={"negation": ["adversarial training"]},
        status="accepted",
    )
    return GatedFeatureFusionRanker(
        baseline_ranker=_IdentityRanker(),
        context_store=UnifiedFusionContextResolver(
            task_store=FrozenTaskSlotLabelStore({}),
            constraint_store=FrozenConstraintProfileStore([annotation]),
        ),
        feature_families={"hard_constraint"},
        epochs=1,
        dimension=2048,
        constraint_text_evidence=True,
        hard_negative_limit=hard_negative_limit,
    )


def test_negation_pair_uses_unknown_gold_as_zero_conflict_supervision() -> None:
    ranker = _negation_ranker()
    ranking_query = DocumentRankingQuery(
        query_id="q-negation",
        query="vision models without adversarial training",
        gold_paper_ids=["openalex:W1"],
        candidates=[
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id="openalex:W1",
                    openalex_id="W1",
                    title="A robust vision model",
                ),
                baseline_score=1.0,
                source_ranks={"lexical": 1},
            ),
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id="openalex:W2",
                    openalex_id="W2",
                    title="We use adversarial training for robust vision models",
                ),
                baseline_score=0.5,
                source_ranks={"lexical": 2},
            ),
        ],
    )

    pair_counts = ranker.fit([ranking_query])
    activation = audit_fusion_query_activation(ranking_query, ranker)

    assert pair_counts["hard_constraint"] == 1
    assert ranker.weights["hard_constraint"][
        _index("hard-negation-conflict", 2048)
    ] < 0.0
    assert ranker.weights["hard_constraint"][
        _index("hard-negation-clean", 2048)
    ] == 0.0
    assert activation["families"]["hard_constraint"][
        "signal_effective_pair_count"
    ] == {"negation": 1}


def test_negation_conflict_is_not_lost_below_b0_hard_negative_cutoff() -> None:
    ranker = _negation_ranker(hard_negative_limit=100)
    candidates = [
        DocumentCandidateEvidence(
            paper=Paper(
                canonical_id="openalex:W1",
                openalex_id="W1",
                title="A robust vision model",
            ),
            baseline_score=1.0,
            source_ranks={"lexical": 1},
        )
    ]
    candidates.extend(
        DocumentCandidateEvidence(
            paper=Paper(
                canonical_id=f"openalex:W{index}",
                openalex_id=f"W{index}",
                title=f"Neutral vision model {index}",
            ),
            baseline_score=1.0 / index,
            source_ranks={"lexical": index},
        )
        for index in range(2, 102)
    )
    candidates.append(
        DocumentCandidateEvidence(
            paper=Paper(
                canonical_id="openalex:W102",
                openalex_id="W102",
                title="We use adversarial training for robust vision models",
            ),
            baseline_score=1.0 / 102,
            source_ranks={"lexical": 102},
        )
    )
    ranking_query = DocumentRankingQuery(
        query_id="q-negation",
        query="vision models without adversarial training",
        gold_paper_ids=["openalex:W1"],
        candidates=candidates,
    )

    pair_counts = ranker.fit([ranking_query])
    activation = audit_fusion_query_activation(ranking_query, ranker)

    assert pair_counts["hard_constraint"] == 1
    assert activation["families"]["hard_constraint"]["effective_pair_count"] == 1


def test_gold_conflicting_with_negation_does_not_create_reversed_pair() -> None:
    ranker = _negation_ranker()
    ranking_query = DocumentRankingQuery(
        query_id="q-negation",
        query="vision models without adversarial training",
        gold_paper_ids=["openalex:W1"],
        candidates=[
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id="openalex:W1",
                    openalex_id="W1",
                    title="We use adversarial training for robust vision",
                ),
                baseline_score=1.0,
                source_ranks={"lexical": 1},
            ),
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id="openalex:W2",
                    openalex_id="W2",
                    title="A robust vision model",
                ),
                baseline_score=0.5,
                source_ranks={"lexical": 2},
            ),
        ],
    )

    pair_counts = ranker.fit([ranking_query])
    activation = audit_fusion_query_activation(ranking_query, ranker)

    assert pair_counts["hard_constraint"] == 0
    assert activation["families"]["hard_constraint"]["effective_pair_count"] == 0


def test_clean_gold_without_explicit_conflict_negative_is_not_trainable() -> None:
    ranker = _negation_ranker()
    ranking_query = DocumentRankingQuery(
        query_id="q-negation",
        query="vision models without adversarial training",
        gold_paper_ids=["openalex:W1"],
        candidates=[
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id="openalex:W1",
                    openalex_id="W1",
                    title="A vision model without adversarial training",
                ),
                baseline_score=1.0,
                source_ranks={"lexical": 1},
            ),
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id="openalex:W2",
                    openalex_id="W2",
                    title="A neutral vision model",
                ),
                baseline_score=0.5,
                source_ranks={"lexical": 2},
            ),
        ],
    )

    pair_counts = ranker.fit([ranking_query])
    activation = audit_fusion_query_activation(ranking_query, ranker)

    assert pair_counts["hard_constraint"] == 0
    assert activation["families"]["hard_constraint"]["effective_pair_count"] == 0


def test_negation_pair_rejects_cross_topic_conflict_candidate() -> None:
    query = "DP optimization and ERM without fairness considerations"
    annotation = FrozenConstraintAnnotation(
        query_id="q-topic",
        query_sha256=query_sha256(query),
        role="training",
        split="auto_train",
        labels=["negation"],
        exclusions=["fairness considerations"],
        label_sources={"negation": "local_deterministic"},
        label_confidence={"negation": 1.0},
        evidence={"negation": ["fairness considerations"]},
        status="accepted",
    )
    ranker = GatedFeatureFusionRanker(
        baseline_ranker=_IdentityRanker(),
        context_store=UnifiedFusionContextResolver(
            task_store=FrozenTaskSlotLabelStore({}),
            constraint_store=FrozenConstraintProfileStore([annotation]),
        ),
        feature_families={"hard_constraint"},
        epochs=1,
        dimension=2048,
        constraint_text_evidence=True,
    )
    ranking_query = DocumentRankingQuery(
        query_id="q-topic",
        query=query,
        gold_paper_ids=["openalex:W1"],
        candidates=[
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id="openalex:W1",
                    openalex_id="W1",
                    title="Differentially private ERM optimization",
                ),
                baseline_score=1.0,
                source_ranks={"lexical": 1},
            ),
            DocumentCandidateEvidence(
                paper=Paper(
                    canonical_id="openalex:W2",
                    openalex_id="W2",
                    title="Grading video interviews with fairness considerations",
                ),
                baseline_score=0.5,
                source_ranks={"lexical": 2},
            ),
        ],
    )

    pair_counts = ranker.fit([ranking_query])
    activation = audit_fusion_query_activation(ranking_query, ranker)

    assert pair_counts["hard_constraint"] == 0
    assert activation["families"]["hard_constraint"][
        "signal_effective_pair_count"
    ] == {}
