from __future__ import annotations

import json
import asyncio
from pathlib import Path

from paper_search.domain.models import Paper
from paper_search.learning.cpu_document_ranker import DocumentCandidateEvidence
from paper_search.learning.cross_vocabulary_bridge import (
    build_cross_vocabulary_action_batch,
    build_refined_cross_vocabulary_action_batch,
    propose_cross_vocabulary_bridge,
    propose_refined_cross_vocabulary_bridge,
    select_production_cross_vocabulary_supplement,
)
from paper_search.domain.models import QuerySpec
from paper_search.recall_experiments.contracts import RecallGenerationContext
from paper_search.recall_experiments.generation.fixed import FixedActionGenerator
from scripts.run_cross_vocabulary_openalex_validation import (
    _excluded_query_ids,
    _context_ids_cover_package,
    _cross_query_common_expansion_terms,
    _proposal_payload,
    _refined_strata_sample,
    _terminal_openalex_result_is_usable,
    build_parser,
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
    ]


def test_bridge_requires_cross_candidate_and_cross_action_support() -> None:
    proposal = propose_cross_vocabulary_bridge(
        "graph neural retrieval benchmark",
        _supported_candidates(),
        max_expansion_terms=2,
    )

    assert proposal is not None
    assert "alignment" in proposal.expansion_terms
    assert proposal.candidate_support["alignment"] == 3
    assert proposal.action_support["alignment"] >= 2
    assert "modeling" not in proposal.expansion_terms
    assert proposal.anchors == ("retrieval", "benchmark")
    assert proposal.query_text.split()[:2] == ["retrieval", "benchmark"]


def test_bridge_excludes_pasa_candidates_from_proposal_evidence() -> None:
    candidates = [
        _candidate(
            index,
            "Graph neural retrieval leakedterm",
            actions=("pasa-local-original", "pasa-local-supplement"),
            source="pasa_paper_database",
        )
        for index in range(1, 5)
    ]

    assert propose_cross_vocabulary_bridge(
        "graph neural retrieval benchmark", candidates
    ) is None


def test_bridge_abstains_on_single_action_noise() -> None:
    candidates = [
        _candidate(
            index,
            "Graph neural retrieval hallucination",
            actions=("anchor-original@same",),
        )
        for index in range(1, 5)
    ]

    assert propose_cross_vocabulary_bridge(
        "graph neural retrieval benchmark", candidates
    ) is None


def test_bridge_action_is_one_deterministic_blind_lexical_action() -> None:
    proposal = propose_cross_vocabulary_bridge(
        "graph neural retrieval benchmark", _supported_candidates()
    )
    assert proposal is not None

    first = build_cross_vocabulary_action_batch(proposal)
    second = build_cross_vocabulary_action_batch(proposal)
    payload = first.model_dump(mode="json")

    assert first == second
    assert len(first.actions) == 1
    assert first.actions[0].action_id == "contrastive-bridge-local-idf-v1"
    assert first.actions[0].payload.search_mode == "lexical"
    serialized = json.dumps(payload, sort_keys=True).casefold()
    assert "gold" not in serialized
    assert "pasa" not in serialized


def test_exact_context_may_safely_cover_more_auto_train_rows_than_package() -> None:
    assert _context_ids_cover_package(
        ("q-1", "q-2"),
        task_query_ids={"q-1", "q-2", "excluded-conflict"},
        constraint_query_ids={"q-1", "q-2", "excluded-conflict"},
    )
    assert not _context_ids_cover_package(
        ("q-1", "q-2"),
        task_query_ids={"q-1"},
        constraint_query_ids={"q-1", "q-2"},
    )


def test_fixed_action_override_declares_fixed_identity() -> None:
    proposal = propose_cross_vocabulary_bridge(
        "graph neural retrieval benchmark", _supported_candidates()
    )
    assert proposal is not None
    batch = build_cross_vocabulary_action_batch(proposal)
    generator = FixedActionGenerator(
        {"q-1": batch.model_dump(mode="json")},
        expected_query_ids=["q-1"],
        allowed_actions=["text_search"],
        max_actions=1,
        source_sha256="sha256:" + "1" * 64,
    )

    result = asyncio.run(
        generator.generate(
            RecallGenerationContext(
                query_id="q-1",
                original_query="graph neural retrieval benchmark",
                query_spec=QuerySpec(
                    original_query="graph neural retrieval benchmark",
                    research_goal="graph neural retrieval benchmark",
                ),
            )
        )
    )

    assert generator.generator_type == "fixed_actions"
    assert result.action_batch == batch


def test_terminal_invalid_work_warning_keeps_valid_sibling_hits() -> None:
    assert _terminal_openalex_result_is_usable([])
    assert _terminal_openalex_result_is_usable(
        [{"code": "invalid_work", "retryable": False}]
    )
    assert not _terminal_openalex_result_is_usable(
        [{"code": "provider_unavailable", "retryable": True}]
    )


def test_refined_bridge_suppresses_functional_expansion_terms() -> None:
    candidates = [
        _candidate(
            index,
            (
                "Graph neural retrieval based using via models methods "
                f"survey review analysis item{index}"
            ),
            actions=("anchor-original@a", "semantic-original@b"),
        )
        for index in range(1, 7)
    ]

    proposal = propose_refined_cross_vocabulary_bridge(
        "graph neural retrieval benchmark",
        candidates,
        profile="unconstrained",
    )

    assert proposal is None


def test_refined_bridge_accepts_cross_query_expansion_blocklist() -> None:
    candidates = [
        _candidate(
            index,
            "Graph neural retrieval learning",
            actions=("anchor-original@a", "semantic-original@b"),
        )
        for index in range(1, 7)
    ]

    proposal = propose_refined_cross_vocabulary_bridge(
        "graph neural retrieval benchmark",
        candidates,
        profile="unconstrained",
        suppressed_expansion_terms=("learning",),
    )

    assert proposal is None


def test_refined_negation_bridge_removes_exclusion_and_negation_cues() -> None:
    candidates = [
        _candidate(
            1,
            "Neural rendering objects decomposition with shape databases",
            actions=("anchor-original@a", "semantic-original@b"),
        ),
        _candidate(
            2,
            "Neural rendering objects decomposition via layers",
            actions=("anchor-original@a",),
        ),
        _candidate(
            3,
            "Objects neural rendering decomposition without supervision",
            actions=("semantic-original@b",),
        ),
        _candidate(
            4,
            "Neural rendering objects decomposition models",
            actions=("boolean-topic@c",),
        ),
    ]

    proposal = propose_refined_cross_vocabulary_bridge(
        "neural rendering objects without missing shape databases",
        candidates,
        profile="negation",
        exclusions=("shape databases",),
    )

    assert proposal is not None
    assert proposal.evidence_profile == "negation"
    assert "decomposition" in proposal.expansion_terms
    assert len(proposal.anchors) == 3
    assert not set(proposal.anchors) & {"without", "missing", "shape", "databases"}
    assert not set(proposal.expansion_terms) & {
        "without",
        "missing",
        "shape",
        "databases",
        "via",
    }


def test_refined_unconstrained_bridge_requires_stronger_anchor_conditioning() -> None:
    weak = _supported_candidates()

    assert (
        propose_refined_cross_vocabulary_bridge(
            "graph neural retrieval benchmark",
            weak,
            profile="unconstrained",
        )
        is None
    )

    strong = [
        *weak[:3],
        _candidate(
            7,
            "Graph retrieval benchmark alignment",
            actions=("boolean-topic@c",),
        ),
        *weak[3:],
    ]
    proposal = propose_refined_cross_vocabulary_bridge(
        "graph neural retrieval benchmark",
        strong,
        profile="unconstrained",
    )

    assert proposal is not None
    assert proposal.evidence_profile == "unconstrained"
    assert "alignment" in proposal.expansion_terms
    assert len(proposal.anchors) == 3
    assert proposal.title_support["alignment"] >= 4
    assert proposal.anchor_cooccurrence["alignment"] >= 4
    batch = build_refined_cross_vocabulary_action_batch(proposal)
    assert len(batch.actions) == 1
    assert batch.actions[0].action_id == "contrastive-bridge-anchor-conditioned-v2"


def test_production_bridge_schedules_only_unconstrained_queries() -> None:
    candidates = [
        *_supported_candidates()[:3],
        _candidate(
            7,
            "Graph retrieval benchmark alignment",
            actions=("boolean-topic@c",),
        ),
        *_supported_candidates()[3:],
    ]
    unconstrained = QuerySpec(
        original_query="graph neural retrieval benchmark",
        research_goal="graph neural retrieval benchmark",
    )

    batch = select_production_cross_vocabulary_supplement(
        unconstrained,
        candidates,
    )

    assert len(batch.actions) == 1
    assert batch.actions[0].action_id == "contrastive-bridge-anchor-conditioned-v2"
    assert batch.actions[0].payload.query_text == "graph neural retrieval alignment"


def test_production_bridge_strictly_abstains_for_negation_and_other_constraints() -> None:
    candidates = [
        *_supported_candidates()[:3],
        _candidate(
            7,
            "Graph retrieval benchmark alignment",
            actions=("boolean-topic@c",),
        ),
        *_supported_candidates()[3:],
    ]
    negation = QuerySpec(
        original_query="graph neural retrieval benchmark without alignment",
        research_goal="graph neural retrieval benchmark",
        exclusions=["alignment"],
    )
    method = QuerySpec(
        original_query="graph neural retrieval benchmark",
        research_goal="graph neural retrieval benchmark",
        methods=["graph transformer"],
    )

    assert not select_production_cross_vocabulary_supplement(
        negation,
        candidates,
    ).actions
    assert not select_production_cross_vocabulary_supplement(
        method,
        candidates,
    ).actions


def test_production_bridge_abstains_when_raw_query_contains_unparsed_negation() -> None:
    candidates = [
        *_supported_candidates()[:3],
        _candidate(
            7,
            "Graph retrieval benchmark alignment",
            actions=("boolean-topic@c",),
        ),
        *_supported_candidates()[3:],
    ]
    query_spec = QuerySpec(
        original_query="graph neural retrieval benchmark not using dropout",
        research_goal="graph neural retrieval benchmark",
    )

    batch = select_production_cross_vocabulary_supplement(query_spec, candidates)

    assert not batch.actions


def test_refined_bridge_anchors_are_supported_across_candidates() -> None:
    candidates = [
        _candidate(
            1,
            "Graph neural rarecontext oddmarker alignment",
            actions=("anchor-original@a",),
        ),
        _candidate(
            2,
            "Graph neural retrieval alignment",
            actions=("semantic-original@b",),
        ),
        _candidate(
            3,
            "Graph neural benchmark alignment",
            actions=("boolean-topic@c",),
        ),
        _candidate(
            4,
            "Graph retrieval benchmark alignment",
            actions=("anchor-original@a",),
        ),
        _candidate(
            5,
            "Neural retrieval benchmark systems",
            actions=("semantic-original@b",),
        ),
        _candidate(
            6,
            "Graph neural retrieval evaluation",
            actions=("boolean-topic@c",),
        ),
        _candidate(
            7,
            "Graph neural benchmark evaluation",
            actions=("anchor-original@a",),
        ),
    ]

    proposal = propose_refined_cross_vocabulary_bridge(
        "graph neural retrieval benchmark rarecontext oddmarker",
        candidates,
        profile="unconstrained",
    )

    assert proposal is not None
    assert not set(proposal.anchors) & {"rarecontext", "oddmarker"}
    assert min(proposal.anchor_support.values()) >= 2


def test_refined_proposal_payload_freezes_profile_and_support() -> None:
    proposal = propose_refined_cross_vocabulary_bridge(
        "graph neural retrieval benchmark",
        [
            *_supported_candidates()[:3],
            _candidate(
                7,
                "Graph retrieval benchmark alignment",
                actions=("boolean-topic@c",),
            ),
            *_supported_candidates()[3:],
        ],
        profile="unconstrained",
    )
    assert proposal is not None

    payload = _proposal_payload(proposal)

    assert payload["evidence_profile"] == "unconstrained"
    assert payload["title_support"] == proposal.title_support
    assert payload["anchor_cooccurrence"] == proposal.anchor_cooccurrence


def test_refined_strata_sample_prioritizes_negation_then_fills_unconstrained() -> None:
    rows = [
        {
            "query_id": f"neg-{index}",
            "signal": "negation",
            "length_bucket": "short" if index % 2 else "long",
        }
        for index in range(5)
    ]
    rows.extend(
        {
            "query_id": f"unc-{index}",
            "signal": "unconstrained",
            "length_bucket": "medium",
        }
        for index in range(8)
    )

    selected = _refined_strata_sample(rows, limit=8, negation_target=4)

    assert len(selected) == 8
    assert len({str(row["query_id"]) for row in selected}) == 8
    assert sum(row["signal"] == "negation" for row in selected) == 4
    assert sum(row["signal"] == "unconstrained" for row in selected) == 4


def test_validation_parser_accepts_refined_unconstrained_only_confirmation() -> None:
    args = build_parser().parse_args(
        ["prepare", "--refined-strata", "--unconstrained-only"]
    )

    assert args.refined_strata is True
    assert args.unconstrained_only is True


def test_excluded_query_ids_include_every_frozen_contrastive_partition(
    tmp_path: Path,
) -> None:
    recall_root = tmp_path / "data" / "training_private" / "recall_policy"
    required_empty_paths = (
        recall_root
        / "query-adaptive-high-recall-discovery128-v1"
        / "pilot-partition.jsonl",
        recall_root
        / "query-adaptive-high-recall-validation352-v1"
        / "pilot-partition.jsonl",
        recall_root
        / "contrastive-openalex-bridge-validation128-v1"
        / "partition.jsonl",
    )
    for path in required_empty_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    current_partition = (
        recall_root
        / "contrastive-openalex-bridge-nu128-v2"
        / "partition.jsonl"
    )
    current_partition.parent.mkdir(parents=True)
    current_partition.write_text(
        json.dumps({"query_id": "already-used-current-v2"}) + "\n",
        encoding="utf-8",
    )

    excluded = _excluded_query_ids(tmp_path)

    assert "already-used-current-v2" in excluded


def test_cross_query_common_terms_are_suppressed_without_gold_outcomes() -> None:
    rows = [
        {"proposal": {"expansion_terms": ["learning", "alignment"]}},
        {"proposal": {"expansion_terms": ["learning", "decomposition"]}},
        {"proposal": {"expansion_terms": ["learning", "mitigation"]}},
        {"proposal": {"expansion_terms": ["retrieval", "backdoors"]}},
    ]

    suppressed = _cross_query_common_expansion_terms(
        rows,
        min_query_count=2,
        min_query_ratio=0.5,
    )

    assert suppressed == frozenset({"learning"})
