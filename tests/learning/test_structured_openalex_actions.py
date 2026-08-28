from __future__ import annotations

import json

from paper_search.domain.models import Paper, QuerySpec
from paper_search.learning.cpu_document_ranker import DocumentCandidateEvidence
from paper_search.learning.cross_vocabulary_bridge import (
    select_production_cross_vocabulary_supplement,
)


def _candidate(
    index: int,
    title: str,
    *,
    actions: tuple[str, ...],
) -> DocumentCandidateEvidence:
    source_ranks = {action: index for action in actions}
    return DocumentCandidateEvidence(
        paper=Paper(
            canonical_id=f"openalex:paper-{index}",
            title=title,
            sources=["openalex"],
        ),
        baseline_score=sum(1.0 / (60 + rank) for rank in source_ranks.values()),
        source_ranks=source_ranks,
    )


def _supported_unconstrained_candidates() -> list[DocumentCandidateEvidence]:
    return [
        _candidate(
            1,
            "Graph neural retrieval alignment modeling",
            actions=("anchor-original@a", "semantic-original@b"),
        ),
        _candidate(
            2,
            "Graph neural search alignment modeling",
            actions=("anchor-original@a",),
        ),
        _candidate(
            3,
            "Neural retrieval alignment modeling",
            actions=("semantic-original@b",),
        ),
        _candidate(
            4,
            "Graph retrieval benchmark modeling",
            actions=("boolean-topic@c",),
        ),
        _candidate(
            5,
            "Neural benchmark evaluation modeling",
            actions=("anchor-original@a",),
        ),
        _candidate(
            6,
            "Graph benchmark systems modeling",
            actions=("semantic-original@b",),
        ),
        _candidate(
            7,
            "Graph neural benchmark alignment",
            actions=("boolean-topic@c",),
        ),
    ]


def test_adaptive_supplement_compiles_explicit_entity_alias_as_or() -> None:
    spec = QuerySpec(
        original_query=(
            "Find graph attention network (GAT) methods for link prediction"
        ),
        research_goal="find methods for link prediction",
        methods=["graph attention network"],
        tasks=["link prediction"],
    )

    batch = select_production_cross_vocabulary_supplement(spec, [])

    assert len(batch.actions) == 1
    action = batch.actions[0]
    assert action.action_id == "structured-openalex-entity-alias-v1"
    assert action.payload.query_text == '("graph attention network" OR "GAT")'
    assert action.payload.search_mode == "lexical"
    serialized = json.dumps(batch.model_dump(mode="json"), sort_keys=True).casefold()
    assert "gold" not in serialized
    assert "pasa" not in serialized


def test_adaptive_supplement_compiles_grounded_method_phrase_with_proximity() -> None:
    spec = QuerySpec(
        original_query="Find methods for computing weak optimal transport plans",
        research_goal="find methods for weak optimal transport",
        methods=["weak optimal transport"],
        tasks=["computing transport plans"],
    )

    batch = select_production_cross_vocabulary_supplement(spec, [])

    assert len(batch.actions) == 1
    action = batch.actions[0]
    assert action.action_id == "structured-openalex-phrase-proximity-v1"
    assert action.payload.query_text == '"weak optimal transport"~5'


def test_adaptive_supplement_rejects_ungrounded_llm_slot() -> None:
    spec = QuerySpec(
        original_query="Find graph retrieval papers",
        research_goal="find graph retrieval papers",
        methods=["vision transformer"],
    )

    assert not select_production_cross_vocabulary_supplement(spec, []).actions


def test_adaptive_supplement_strictly_abstains_for_negation() -> None:
    spec = QuerySpec(
        original_query=(
            "Find semantic segmentation methods without ImageNet pretraining"
        ),
        research_goal="find semantic segmentation methods",
        tasks=["semantic segmentation"],
        exclusions=["ImageNet pretraining"],
    )

    assert not select_production_cross_vocabulary_supplement(spec, []).actions


def test_adaptive_supplement_preserves_validated_unconstrained_bridge() -> None:
    spec = QuerySpec(
        original_query="graph neural retrieval benchmark",
        research_goal="graph neural retrieval benchmark",
    )

    candidates = _supported_unconstrained_candidates()
    batch = select_production_cross_vocabulary_supplement(spec, candidates)
    production_batch = select_production_cross_vocabulary_supplement(spec, candidates)

    assert batch == production_batch
    assert len(batch.actions) == 1
    assert batch.actions[0].action_id == "contrastive-bridge-anchor-conditioned-v2"
