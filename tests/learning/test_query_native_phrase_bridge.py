from __future__ import annotations

import json

from paper_search.domain.models import Paper, QuerySpec
from paper_search.learning.cpu_document_ranker import DocumentCandidateEvidence
from paper_search.learning.query_native_phrase_bridge import (
    build_query_native_title_phrase_action_batch,
    propose_query_native_title_phrase_bridge,
)


def _candidate(
    index: int,
    title: str,
    *,
    actions: tuple[str, ...],
    source: str = "openalex",
) -> DocumentCandidateEvidence:
    source_ranks = {action: index for action in actions}
    return DocumentCandidateEvidence(
        paper=Paper(
            canonical_id=f"{source}:paper-{index}",
            title=title,
            sources=[source],
        ),
        baseline_score=sum(1.0 / (60 + rank) for rank in source_ranks.values()),
        source_ranks=source_ranks,
    )


def _supported_candidates() -> list[DocumentCandidateEvidence]:
    return [
        _candidate(
            1,
            "Graph contrastive learning for molecular representation pretraining",
            actions=("anchor-original@a",),
        ),
        _candidate(
            2,
            "Molecular representation learning with graph contrastive objectives",
            actions=("semantic-original@b",),
        ),
        _candidate(
            3,
            "Graph contrastive learning for molecular representation models",
            actions=("boolean-topic@c",),
        ),
        _candidate(
            4,
            "Molecular property prediction with graph neural networks",
            actions=("anchor-original@a",),
        ),
    ]


def _query_spec(**updates: object) -> QuerySpec:
    base = QuerySpec(
        original_query=(
            "find graph contrastive learning methods for molecular property "
            "prediction with MolCLR"
        ),
        research_goal="molecular property prediction",
        methods=["graph contrastive learning", "not present injected method"],
    )
    return base.model_copy(update=updates)


def test_bridge_combines_exact_query_entity_with_cross_title_phrase() -> None:
    proposal = propose_query_native_title_phrase_bridge(
        _query_spec(),
        _supported_candidates(),
    )

    assert proposal is not None
    assert proposal.query_anchors[0] == "graph contrastive learning"
    assert proposal.supported_phrase == "molecular representation"
    assert proposal.phrase_candidate_support == 3
    assert proposal.phrase_action_support >= 2
    assert proposal.query_text == (
        "graph contrastive learning molecular representation"
    )
    assert "not present injected method" not in proposal.query_text


def test_bridge_abstains_without_cross_action_phrase_support() -> None:
    candidates = [
        _candidate(
            index,
            "Graph contrastive learning for molecular representation models",
            actions=("anchor-original@same",),
        )
        for index in range(1, 5)
    ]

    assert propose_query_native_title_phrase_bridge(_query_spec(), candidates) is None


def test_bridge_strictly_abstains_for_negation() -> None:
    spec = _query_spec(
        original_query=(
            "find graph contrastive learning for molecular property prediction "
            "without pretraining"
        ),
        exclusions=["pretraining"],
    )

    assert (
        propose_query_native_title_phrase_bridge(spec, _supported_candidates())
        is None
    )


def test_bridge_excludes_pasa_candidate_evidence() -> None:
    candidates = [
        _candidate(
            index,
            "Graph contrastive learning for molecular representation models",
            actions=("pasa-local@a", "pasa-supplement@b"),
            source="pasa_paper_database",
        )
        for index in range(1, 5)
    ]

    assert propose_query_native_title_phrase_bridge(_query_spec(), candidates) is None


def test_bridge_action_is_one_gold_blind_lexical_action() -> None:
    proposal = propose_query_native_title_phrase_bridge(
        _query_spec(),
        _supported_candidates(),
    )
    assert proposal is not None

    batch = build_query_native_title_phrase_action_batch(proposal)
    payload = batch.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True).casefold()

    assert len(batch.actions) == 1
    assert batch.actions[0].action_id == "query-native-title-phrase-v3"
    assert batch.actions[0].payload.search_mode == "lexical"
    assert "gold" not in serialized
    assert "pasa" not in serialized


def test_bridge_does_not_create_phrase_across_title_stopwords() -> None:
    spec = QuerySpec(
        original_query="neural networks for session recommendation",
        research_goal="session recommendation",
        methods=["neural networks"],
    )
    candidates = [
        _candidate(
            1,
            "Neural networks for session based recommendation",
            actions=("anchor-original@a",),
        ),
        _candidate(
            2,
            "Neural networks for session based recommendation systems",
            actions=("semantic-original@b",),
        ),
        _candidate(
            3,
            "Neural networks for session recommendation benchmarks",
            actions=("boolean-topic@c",),
        ),
    ]

    proposal = propose_query_native_title_phrase_bridge(spec, candidates)

    assert proposal is not None
    assert proposal.supported_phrase == "session based recommendation"
    assert proposal.supported_phrase != "networks session based"


def test_bridge_rejects_all_generic_title_phrase() -> None:
    spec = QuerySpec(
        original_query="anomaly detection methods",
        research_goal="anomaly detection",
        tasks=["anomaly detection"],
    )
    candidates = [
        _candidate(
            1,
            "Deep learning for anomaly detection",
            actions=("anchor-original@a",),
        ),
        _candidate(
            2,
            "Deep learning methods for anomaly detection",
            actions=("semantic-original@b",),
        ),
        _candidate(
            3,
            "Anomaly detection benchmark evaluation",
            actions=("boolean-topic@c",),
        ),
    ]

    assert propose_query_native_title_phrase_bridge(spec, candidates) is None


def test_bridge_cleans_narrative_shell_from_query_native_anchor() -> None:
    spec = QuerySpec(
        original_query="Any works about real-world image denoising?",
        research_goal="real-world image denoising",
        tasks=["Any works about real-world image denoising"],
    )
    candidates = [
        _candidate(
            1,
            "Real world image denoising through noise removal",
            actions=("anchor-original@a",),
        ),
        _candidate(
            2,
            "Real world image denoising with noise removal",
            actions=("semantic-original@b",),
        ),
        _candidate(
            3,
            "Real world image denoising benchmark",
            actions=("boolean-topic@c",),
        ),
    ]

    proposal = propose_query_native_title_phrase_bridge(spec, candidates)

    assert proposal is not None
    assert proposal.query_anchors == ("real world image denoising",)
    assert "any works about" not in proposal.query_text


def test_bridge_prefers_complete_supported_phrase_and_preserves_its_order() -> None:
    spec = QuerySpec(
        original_query="machine translation methods",
        research_goal="machine translation",
        tasks=["machine translation"],
    )
    candidates = [
        _candidate(
            1,
            "Neural machine translation systems",
            actions=("anchor-original@a",),
        ),
        _candidate(
            2,
            "Neural machine translation evaluation",
            actions=("semantic-original@b",),
        ),
        _candidate(
            3,
            "Machine translation benchmark",
            actions=("boolean-topic@c",),
        ),
    ]

    proposal = propose_query_native_title_phrase_bridge(spec, candidates)

    assert proposal is not None
    assert proposal.supported_phrase == "neural machine translation"
    assert proposal.query_text == "neural machine translation"
