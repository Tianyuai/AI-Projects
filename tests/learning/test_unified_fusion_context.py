from __future__ import annotations

import hashlib
import json
import struct

import pytest

from paper_search.domain.models import Paper, QuerySpec
from paper_search.learning.cpu_document_ranker import DocumentCandidateEvidence
from paper_search.learning.gated_feature_fusion_ranker import (
    FUSION_FAMILIES,
    GatedFeatureFusionRanker,
    UnifiedFusionContextResolver,
    _index,
    gated_family_candidate_features,
    gated_family_eligibility,
    load_gated_feature_fusion_ranker_bytes,
)
from paper_search.learning.query_constraint_annotations import (
    FrozenConstraintAnnotation,
    FrozenConstraintProfileStore,
    query_sha256,
)
from paper_search.learning.task_slot_document_ranker import (
    FrozenTaskSlotLabel,
    FrozenTaskSlotLabelStore,
    FrozenTaskValue,
)


def _spec(query: str) -> QuerySpec:
    return QuerySpec(
        original_query=query,
        research_goal=query,
        topics=["vision"],
        methods=["vision transformer"],
        tasks=["image classification"],
        datasets=["ImageNet"],
        year_from=2020,
        year_to=2024,
        exclusions=["adversarial training"],
    )


class _IdentityRanker:
    def rank(self, query, candidates):
        del query
        return list(candidates)


def _negation_guard_candidates(*, with_contrast: bool) -> list[DocumentCandidateEvidence]:
    titles = [
        "Broad image classification survey",
        "Robust transformer evaluation",
        "Vision benchmark analysis",
        "General representation learning",
    ]
    if with_contrast:
        titles[2] = "Image classification with adversarial training"
    return [
        DocumentCandidateEvidence(
            paper=Paper(canonical_id=f"paper-{index}", title=title),
            baseline_score=1.0 / index,
            source_ranks=(
                {"lexical": 1}
                if index == 1
                else {f"source-{source}": index for source in range(6)}
            ),
        )
        for index, title in enumerate(titles, start=1)
    ]


def _negation_guard_ranker() -> GatedFeatureFusionRanker:
    ranker = GatedFeatureFusionRanker(
        baseline_ranker=_IdentityRanker(),
        context_store=UnifiedFusionContextResolver(
            task_store=FrozenTaskSlotLabelStore({}),
            constraint_store=FrozenConstraintProfileStore([]),
        ),
        feature_families=FUSION_FAMILIES,
        family_caps={family: 0.35 for family in FUSION_FAMILIES},
        constraint_text_evidence=True,
        runtime_context_scoring=True,
    )
    ranker.weights["reliability"][_index("reliability-source-support", 2048)] = 10.0
    ranker.weights["reliability"][
        _index("reliability-best-source-reciprocal", 2048)
    ] = -10.0
    return ranker


def test_negation_without_candidate_contrast_preserves_baseline_order() -> None:
    query = "image classification without adversarial training"
    candidates = _negation_guard_candidates(with_contrast=False)

    ranked = _negation_guard_ranker().rank_with_context(
        query,
        candidates,
        query_spec=_spec(query),
    )

    assert [row.paper.canonical_id for row in ranked] == [
        row.paper.canonical_id for row in candidates
    ]


def test_negation_with_candidate_contrast_keeps_fusion_active() -> None:
    query = "image classification without adversarial training"
    candidates = _negation_guard_candidates(with_contrast=True)

    ranked = _negation_guard_ranker().rank_with_context(
        query,
        candidates,
        query_spec=_spec(query),
    )

    assert [row.paper.canonical_id for row in ranked] != [
        row.paper.canonical_id for row in candidates
    ]


def test_unknown_query_uses_one_deterministic_local_context() -> None:
    query = "Image classification with vision transformers on ImageNet from 2020 to 2024 without adversarial training"
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([]),
    )

    context = resolver.for_scoring_query(query, query_spec=_spec(query))
    receipt = resolver.context_receipt(query, query_spec=_spec(query))

    assert context.task_label is not None
    assert context.task_label.task_label_status == "runtime_deterministic"
    assert [task.normalized_value for task in context.task_label.tasks] == [
        "image classification"
    ]
    assert context.constraint_profile is not None
    assert set(context.constraint_profile.labels) >= {
        "method",
        "dataset",
        "task",
        "year",
        "negation",
    }
    assert receipt["context_source"] == {
        "constraint": "local_query_spec",
        "task": "local_query_spec",
    }
    assert receipt["activated_families"] == [
        "entity",
        "hard_constraint",
        "reliability",
        "task_provenance",
    ]
    assert receipt["context_sha256"].startswith("sha256:")
    assert receipt == resolver.context_receipt(query, query_spec=_spec(query))


def test_exact_frozen_context_takes_priority_over_runtime_spec() -> None:
    query = "frozen query"
    digest = query_sha256(query)
    frozen_task = FrozenTaskSlotLabel(
        query_id="q1",
        query_sha256=digest,
        role="development",
        split="auto_dev",
        tasks=(FrozenTaskValue("frozen task", 1.0),),
        ambiguous_fields=(),
        task_label_status="reviewed",
    )
    frozen_constraint = FrozenConstraintAnnotation(
        query_id="q1",
        query_sha256=digest,
        role="development",
        split="auto_dev",
        labels=["method"],
        methods=["frozen method"],
        label_sources={"method": "human_review"},
        label_confidence={"method": 1.0},
        evidence={"method": ["frozen"]},
        status="accepted",
    )
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({digest: frozen_task}),
        constraint_store=FrozenConstraintProfileStore([frozen_constraint]),
    )

    context = resolver.for_scoring_query(query, query_spec=_spec(query))
    receipt = resolver.context_receipt(query, query_spec=_spec(query))
    local = resolver.for_local_query(query, query_spec=_spec(query))

    assert context.task_label == frozen_task
    assert context.constraint_profile is not None
    assert context.constraint_profile.methods == ["frozen method"]
    assert local.constraint_profile is not None
    assert local.constraint_profile.methods == ["vision transformer"]
    assert receipt["context_source"] == {
        "constraint": "frozen",
        "task": "frozen",
    }


def test_raw_query_parser_conservatively_extracts_explicit_slots() -> None:
    query = (
        "papers using graph neural networks for node classification on the "
        "Cora dataset after 2020 without label propagation"
    )
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([]),
        task_terms=["node classification"],
        method_terms=["graph neural networks"],
    )

    receipt = resolver.context_receipt(query)

    assert receipt["tasks"] == ["node classification"]
    assert receipt["methods"] == ["graph neural networks"]
    assert receipt["datasets"] == ["cora"]
    assert receipt["year_from"] == 2020
    assert receipt["exclusions"] == ["label propagation"]
    assert set(receipt["activated_families"]) == {
        "entity",
        "hard_constraint",
        "reliability",
        "task_provenance",
    }


def test_raw_query_parser_uses_syntax_delimited_method_and_task_slots() -> None:
    query = (
        "papers using a pair-ranker model for optimal LLM output selection "
        "without proprietary data"
    )
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([]),
        task_terms=["node classification"],
        method_terms=["graph neural networks"],
    )

    receipt = resolver.context_receipt(query)

    assert receipt["tasks"] == ["optimal llm output selection"]
    assert receipt["methods"] == ["a pair-ranker model"]
    assert receipt["exclusions"] == ["proprietary data"]
    assert receipt["activated_families"] == [
        "entity",
        "hard_constraint",
        "reliability",
        "task_provenance",
    ]


def test_raw_query_parser_uses_frozen_dataset_lexicon_without_suffix() -> None:
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([]),
        dataset_terms=["ImageNet"],
    )

    receipt = resolver.context_receipt("papers reporting accuracy on ImageNet")

    assert receipt["datasets"] == ["imagenet"]
    assert "entity" in receipt["activated_families"]


def test_raw_query_parser_rejects_question_word_task_slot() -> None:
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([]),
    )

    receipt = resolver.context_receipt("papers using transformers for what")

    assert receipt["methods"] == ["transformers"]
    assert receipt["tasks"] == []


def test_resolver_identity_tracks_explicit_slot_policy() -> None:
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([]),
    )

    assert resolver.resolver_id == "unified-local-fusion-context-v3"


def test_candidate_constraint_evidence_distinguishes_entity_and_negation() -> None:
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([]),
    )
    context = resolver.for_scoring_query("query", query_spec=_spec("query"))
    candidate = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="openalex:W1",
            openalex_id="W1",
                title="ImageNet Evaluation Using Vision Transformer",
            abstract="No adversarial training is used.",
            publication_year=2022,
        ),
        baseline_score=1.0,
        source_ranks={"lexical": 1},
    )

    entity = gated_family_candidate_features(
        context,
        candidate,
        baseline_rank=1,
        family="entity",
        constraint_text_evidence=True,
    )
    hard = gated_family_candidate_features(
        context,
        candidate,
        baseline_rank=1,
        family="hard_constraint",
        constraint_text_evidence=True,
    )

    assert gated_family_eligibility(context, "entity", gated=True)
    assert entity["entity-method-text-match"] == 1.0
    assert entity["entity-dataset-text-match"] == 1.0
    assert hard["hard-negation-conflict"] == 0.0
    assert hard["hard-negation-clean"] == 1.0
    assert hard["hard-year-compliant"] == 1.0


def test_candidate_negation_evidence_ignores_neutral_mentions() -> None:
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([]),
    )
    context = resolver.for_scoring_query("query", query_spec=_spec("query"))
    candidate = DocumentCandidateEvidence(
        paper=Paper(
            canonical_id="openalex:W2",
            openalex_id="W2",
            title="A survey of adversarial training research",
        ),
        baseline_score=1.0,
        source_ranks={"lexical": 1},
    )

    hard = gated_family_candidate_features(
        context,
        candidate,
        baseline_rank=1,
        family="hard_constraint",
        constraint_text_evidence=True,
    )

    assert hard["hard-negation-conflict"] == 0.0
    assert hard["hard-negation-clean"] == 0.0


def test_unvalidated_artifact_reports_but_does_not_score_runtime_context() -> None:
    query = "image classification with a vision transformer after 2020"
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([]),
    )
    ranker = GatedFeatureFusionRanker(
        baseline_ranker=_IdentityRanker(),
        context_store=resolver,
        feature_families=FUSION_FAMILIES,
    )

    receipt = ranker.context_receipt(query, query_spec=_spec(query))

    assert receipt is not None
    assert receipt["activated_families"] == [
        "entity",
        "hard_constraint",
        "reliability",
        "task_provenance",
    ]
    assert receipt["effective_activated_families"] == ["reliability"]
    assert receipt["runtime_context_scoring"] is False

    validated_ranker = GatedFeatureFusionRanker(
        baseline_ranker=_IdentityRanker(),
        context_store=resolver,
        feature_families=FUSION_FAMILIES,
        runtime_context_scoring=True,
    )
    validated_receipt = validated_ranker.context_receipt(
        query, query_spec=_spec(query)
    )
    assert validated_receipt is not None
    assert validated_receipt["effective_activated_families"] == receipt[
        "activated_families"
    ]


def test_artifact_loader_restores_query_family_pair_limit() -> None:
    ranker = _negation_guard_ranker()
    ranker.max_pairs_per_query_family = 32
    records: list[bytes] = []
    family_hashes: dict[str, str] = {}
    for family in sorted(ranker.feature_families):
        name = family.encode("utf-8")
        vector = ranker.weights[family].astype("<f8").tobytes(order="C")
        records.append(struct.pack("<I", len(name)) + name + vector)
        family_hashes[family] = "sha256:" + hashlib.sha256(vector).hexdigest()
    weights = b"".join(records)
    manifest = {
        **ranker.manifest_fields(),
        "family_weight_sha256": family_hashes,
        "weights_sha256": "sha256:" + hashlib.sha256(weights).hexdigest(),
    }

    restored = load_gated_feature_fusion_ranker_bytes(
        (json.dumps(manifest) + "\n").encode(),
        weights,
        baseline_ranker=ranker.baseline_ranker,
        context_store=ranker.context_store,
    )

    assert restored.max_pairs_per_query_family == 32


def test_artifact_loader_keeps_legacy_method_matching_without_new_schema() -> None:
    ranker = _negation_guard_ranker()
    records: list[bytes] = []
    family_hashes: dict[str, str] = {}
    for family in sorted(ranker.feature_families):
        name = family.encode("utf-8")
        vector = ranker.weights[family].astype("<f8").tobytes(order="C")
        records.append(struct.pack("<I", len(name)) + name + vector)
        family_hashes[family] = "sha256:" + hashlib.sha256(vector).hexdigest()
    weights = b"".join(records)
    manifest = ranker.manifest_fields()
    manifest.pop("method_usage_evidence_schema_version")
    manifest.update(
        {
            "family_weight_sha256": family_hashes,
            "weights_sha256": "sha256:" + hashlib.sha256(weights).hexdigest(),
        }
    )

    restored = load_gated_feature_fusion_ranker_bytes(
        (json.dumps(manifest) + "\n").encode(),
        weights,
        baseline_ranker=ranker.baseline_ranker,
        context_store=ranker.context_store,
    )

    assert (
        restored.method_usage_evidence_schema_version
        == "method-text-match-v0-exact-mention"
    )


def test_artifact_loader_restores_feature_family_pair_budgets() -> None:
    ranker = _negation_guard_ranker()
    ranker.pair_budget_by_family = {
        "entity": 48,
        "hard_constraint": 128,
        "reliability": 96,
        "task_provenance": 112,
    }
    records: list[bytes] = []
    family_hashes: dict[str, str] = {}
    for family in sorted(ranker.feature_families):
        name = family.encode("utf-8")
        vector = ranker.weights[family].astype("<f8").tobytes(order="C")
        records.append(struct.pack("<I", len(name)) + name + vector)
        family_hashes[family] = "sha256:" + hashlib.sha256(vector).hexdigest()
    weights = b"".join(records)
    manifest = {
        **ranker.manifest_fields(),
        "family_weight_sha256": family_hashes,
        "weights_sha256": "sha256:" + hashlib.sha256(weights).hexdigest(),
    }

    restored = load_gated_feature_fusion_ranker_bytes(
        (json.dumps(manifest) + "\n").encode(),
        weights,
        baseline_ranker=ranker.baseline_ranker,
        context_store=ranker.context_store,
    )

    assert restored.max_pairs_per_query_family is None
    assert restored.pair_budget_by_family == ranker.pair_budget_by_family


def test_development_constraint_cannot_be_used_during_fit() -> None:
    query = "development-only method query"
    annotation = FrozenConstraintAnnotation(
        query_id="q-dev",
        query_sha256=query_sha256(query),
        role="development",
        split="auto_dev",
        labels=["method"],
        methods=["development method"],
        label_sources={"method": "human_review"},
        label_confidence={"method": 1.0},
        evidence={"method": ["development method"]},
        status="accepted",
    )
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([annotation]),
    )

    with pytest.raises(
        ValueError, match="development constraint label cannot be used during fit"
    ):
        resolver.for_training_query(query)


def test_negation_gate_requires_a_nonempty_exclusion() -> None:
    query = "papers excluding an unspecified condition"
    annotation = FrozenConstraintAnnotation(
        query_id="q-empty-negation",
        query_sha256=query_sha256(query),
        role="training",
        split="auto_train",
        labels=["negation"],
        exclusions=[],
        label_sources={"negation": "local_deterministic"},
        label_confidence={"negation": 1.0},
        evidence={"negation": []},
        status="accepted",
    )
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([annotation]),
    )

    context = resolver.for_training_query(query)

    assert gated_family_eligibility(context, "hard_constraint", gated=True) is False


@pytest.mark.parametrize(
    ("query", "dataset"),
    [
        ("papers with top performance on the COCO 2017 dataset", "coco 2017"),
        ("errors in benchmarks such as CoNLL 2003 for NER", "conll 2003"),
        ("sources mentioning the Algonauts 2023 challenge", "algonauts 2023"),
    ],
)
def test_local_context_treats_named_entity_years_as_dataset_not_time(
    query: str,
    dataset: str,
) -> None:
    resolver = UnifiedFusionContextResolver(
        task_store=FrozenTaskSlotLabelStore({}),
        constraint_store=FrozenConstraintProfileStore([]),
    )

    profile = resolver.for_local_query(query).constraint_profile

    assert profile is not None
    assert profile.datasets == [dataset]
    assert "dataset" in profile.labels
    assert "year" not in profile.labels
    assert profile.year_from is None
    assert profile.year_to is None
