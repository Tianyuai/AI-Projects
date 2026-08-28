from __future__ import annotations

import json

import pytest

from paper_search.domain.models import Paper
from scripts.run_openalex_depth_eligible_validation import (
    build_parser,
    classify_depth_eligibility,
    select_eligible_depth_rows,
    verified_openalex_gold_ids_from_map_bytes,
)
from scripts.run_openalex_depth_validation import build_gold_blind_request_row


def _gold(*, abstract: str | None = "Traffic forecasting with graph structure.") -> Paper:
    return Paper(
        canonical_id="arxiv:2101.00001",
        arxiv_id="2101.00001",
        title="Graph Neural Networks for Traffic Forecasting",
        abstract=abstract,
        publication_year=2021,
        sources=["pasa_paper_database"],
    )


def _proposal() -> dict[str, object]:
    return {
        "query_id": "AutoScholarQuery_train_1",
        "source_action": {
            "action_id": "policy-1",
            "action_type": "text_search",
            "payload": {
                "query_text": "graph neural networks traffic forecasting",
                "search_mode": "lexical",
            },
            "strategy": "learned_action_ranker",
        },
        "cursor": "opaque-next-cursor",
        "source_snapshot_sha256": "sha256:" + "a" * 64,
        "source_retrieval_sha256": "sha256:" + "b" * 64,
        "source_hit_count": 50,
    }


def test_depth_eligibility_requires_the_same_gold_to_pass_all_three_checks() -> None:
    result = classify_depth_eligibility(
        query="Which graph neural networks forecast road traffic?",
        action_text="graph neural networks traffic forecasting",
        gold_paper_ids=["arxiv:2101.00001"],
        pasa_gold_papers={"arxiv:2101.00001": _gold()},
        verified_openalex_gold_ids={"arxiv:2101.00001"},
    )

    assert result["category"] == "eligible"
    assert result["eligible"] is True
    assert result["eligible_gold_count"] == 1
    assert result["selected_gold_id"] == "arxiv:2101.00001"
    assert result["query_gold_title_term_overlap"] >= 2
    assert result["action_gold_title_term_overlap"] >= 2


def test_depth_eligibility_excludes_incomplete_gold_metadata() -> None:
    result = classify_depth_eligibility(
        query="Which graph neural networks forecast road traffic?",
        action_text="graph neural networks traffic forecasting",
        gold_paper_ids=["arxiv:2101.00001"],
        pasa_gold_papers={"arxiv:2101.00001": _gold(abstract=None)},
        verified_openalex_gold_ids={"arxiv:2101.00001"},
    )

    assert result["category"] == "gold_metadata_incomplete"
    assert result["eligible"] is False


def test_depth_eligibility_excludes_unverified_openalex_identity() -> None:
    result = classify_depth_eligibility(
        query="Which graph neural networks forecast road traffic?",
        action_text="graph neural networks traffic forecasting",
        gold_paper_ids=["arxiv:2101.00001"],
        pasa_gold_papers={"arxiv:2101.00001": _gold()},
        verified_openalex_gold_ids=set(),
    )

    assert result["category"] == "openalex_identity_metadata_unverified"
    assert result["eligible"] is False


def test_depth_eligibility_separates_query_and_action_vocabulary_mismatch() -> None:
    query_mismatch = classify_depth_eligibility(
        query="Protein folding with molecular simulation",
        action_text="graph neural networks traffic forecasting",
        gold_paper_ids=["arxiv:2101.00001"],
        pasa_gold_papers={"arxiv:2101.00001": _gold()},
        verified_openalex_gold_ids={"arxiv:2101.00001"},
    )
    action_mismatch = classify_depth_eligibility(
        query="Which graph neural networks forecast road traffic?",
        action_text="protein folding molecular simulation",
        gold_paper_ids=["arxiv:2101.00001"],
        pasa_gold_papers={"arxiv:2101.00001": _gold()},
        verified_openalex_gold_ids={"arxiv:2101.00001"},
    )

    assert query_mismatch["category"] == "query_gold_vocabulary_mismatch"
    assert action_mismatch["category"] == "action_gold_technical_expression_mismatch"


def test_gold_conditioned_eligibility_never_changes_the_outbound_request() -> None:
    proposal = _proposal()
    proposal["local_eligibility"] = classify_depth_eligibility(
        query="Which graph neural networks forecast road traffic?",
        action_text="graph neural networks traffic forecasting",
        gold_paper_ids=["arxiv:2101.00001"],
        pasa_gold_papers={"arxiv:2101.00001": _gold()},
        verified_openalex_gold_ids={"arxiv:2101.00001"},
    )

    request = build_gold_blind_request_row(proposal)
    serialized = json.dumps(request, sort_keys=True).casefold()

    assert request["derived_query_text"] == "graph neural networks traffic forecasting"
    assert "gold" not in serialized
    assert "2101.00001" not in serialized


def test_production_identifier_map_only_proves_openalex_or_doi_to_arxiv_links() -> None:
    payload = json.dumps(
        {
            "openalex:W1": "arxiv:2101.00001",
            "doi:10.1/example": "arxiv:2102.00002",
            "s2:abc": "arxiv:2103.00003",
            "openalex:W2": "doi:10.2/not-an-arxiv-target",
        }
    ).encode()

    assert verified_openalex_gold_ids_from_map_bytes(payload) == {
        "arxiv:2101.00001",
        "arxiv:2102.00002",
    }


def test_eligible_depth_selection_rejects_confounded_and_prior_rows() -> None:
    eligible_1 = {**_proposal(), "query_id": "AutoScholarQuery_train_1"}
    eligible_1["local_eligibility"] = {"category": "eligible"}
    eligible_1["production_identity_baseline_gold_hit"] = False
    eligible_2 = {**_proposal(), "query_id": "AutoScholarQuery_train_2"}
    eligible_2["local_eligibility"] = {"category": "eligible"}
    eligible_2["production_identity_baseline_gold_hit"] = False
    confounded = {**_proposal(), "query_id": "AutoScholarQuery_train_3"}
    confounded["local_eligibility"] = {
        "category": "query_gold_vocabulary_mismatch"
    }

    selected = select_eligible_depth_rows(
        [eligible_1, eligible_2, confounded],
        prior_query_ids={"AutoScholarQuery_train_2"},
        limit=1,
        seed="depth-eligible-v2",
    )

    assert [row["query_id"] for row in selected] == [
        "AutoScholarQuery_train_1"
    ]


def test_eligible_depth_selection_rejects_a_production_identity_baseline_hit() -> None:
    identity_hit = _proposal()
    identity_hit["local_eligibility"] = {"category": "eligible"}
    identity_hit["production_identity_baseline_gold_hit"] = True

    with pytest.raises(ValueError, match="insufficient exact depth proposals"):
        select_eligible_depth_rows(
            [identity_hit],
            prior_query_ids=set(),
            limit=1,
            seed="depth-eligible-v2",
        )


def test_eligible_depth_cli_defaults_to_a_bounded_32_query_package() -> None:
    args = build_parser().parse_args(["prepare"])

    assert args.query_count == 32
    assert args.max_raw_requests == 32
    assert args.output.name == "openalex-depth-eligible32-v2"
