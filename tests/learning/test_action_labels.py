from __future__ import annotations

import json

import pytest

from paper_search.domain.models import QuerySpec
from paper_search.learning.action_labels import build_action_labels, freeze_action_labels
from paper_search.learning.candidates import DeterministicActionCandidateGenerator
from paper_search.learning.routing import RuleQueryRouter


def test_candidate_generator_is_bounded_query_only_and_keeps_distinct_actions() -> None:
    query = "Which paper proposed graph diffusion networks for retrieval?"
    routed = RuleQueryRouter().route(query)

    actions = DeterministicActionCandidateGenerator(max_candidates=6).generate(
        routed.query_spec,
        query_kind=routed.query_kind,
    )

    assert 2 <= len(actions) <= 6
    assert actions[0].origin == "original_query"
    assert actions[0].text == query
    semantic_anchors = [
        action
        for action in actions
        if action.text == query and action.search_mode == "semantic"
    ]
    assert len(semantic_anchors) == 1
    assert semantic_anchors[0].origin == "deterministic_rule"
    assert any(action.origin == "deterministic_rule" for action in actions)
    assert any(action.action_type == "title_search" for action in actions)
    assert len(
        {
            (action.action_type, action.search_mode, action.text.casefold())
            for action in actions
        }
    ) == len(actions)


def test_candidate_generator_removes_question_scaffolding_without_losing_terms() -> None:
    query = (
        "Could you provide me some works related to representer theorems "
        "in machine learning?"
    )
    routed = RuleQueryRouter().route(query)

    actions = DeterministicActionCandidateGenerator(max_candidates=12).generate(
        routed.query_spec,
        query_kind=routed.query_kind,
    )

    texts = {
        action.text.casefold()
        for action in actions[1:]
        if action.search_mode == "lexical"
    }
    assert "representer theorems machine learning" in texts
    assert all("could" not in text and "provide" not in text for text in texts)


def test_candidate_generator_uses_explicit_query_spec_facets() -> None:
    spec = QuerySpec(
        original_query="Find work on graph retrieval with diffusion models on MS MARCO",
        research_goal="Find work on graph retrieval with diffusion models on MS MARCO",
        tasks=["graph retrieval"],
        methods=["diffusion models"],
        datasets=["MS MARCO"],
    )

    actions = DeterministicActionCandidateGenerator(max_candidates=15).generate(
        spec,
        query_kind="semantic",
    )

    texts = {action.text.casefold() for action in actions}
    assert "graph retrieval diffusion models" in texts
    assert "graph retrieval ms marco" in texts


def test_candidate_generator_preserves_parenthetical_acronym_variants() -> None:
    query = "Find graph neural networks (GNNs) for scholarly recommendation"
    routed = RuleQueryRouter().route(query)

    actions = DeterministicActionCandidateGenerator(max_candidates=12).generate(
        routed.query_spec,
        query_kind=routed.query_kind,
    )

    texts = {action.text.casefold() for action in actions}
    assert "graph neural networks gnns" in texts


def test_action_labels_use_gold_only_for_targets_and_never_serialize_it() -> None:
    query = "Which paper proposed graph diffusion networks for retrieval?"
    routed = RuleQueryRouter().route(query)
    candidates = DeterministicActionCandidateGenerator(max_candidates=6).generate(
        routed.query_spec,
        query_kind=routed.query_kind,
    )

    labels = build_action_labels(
        dataset="pasa",
        split="auto_train",
        role="training",
        query_id="q1",
        query=query,
        query_kind=routed.query_kind,
        candidates=candidates,
        gold_titles=["Graph Diffusion Networks for Information Retrieval"],
    )

    assert labels[0].action.origin == "original_query"
    assert labels[0].label == "positive"
    assert any(row.label == "positive" for row in labels[1:])
    payload = json.dumps([row.model_dump(mode="json") for row in labels])
    assert "Graph Diffusion Networks for Information Retrieval" not in payload
    assert "gold" not in payload.casefold()


def test_action_labels_reject_final_test() -> None:
    with pytest.raises(ValueError, match="final_test"):
        build_action_labels(
            dataset="pasa",
            split="auto_test",
            role="final_test",
            query_id="q1",
            query="graph retrieval",
            query_kind="semantic",
            candidates=[],
            gold_titles=["Graph Retrieval"],
        )


def test_freeze_action_labels_uses_only_frozen_partition_ids(tmp_path) -> None:
    partition = tmp_path / "partition.jsonl"
    partition.write_text(
        json.dumps(
            {
                "dataset": "pasa",
                "split": "auto_train",
                "role": "training",
                "revision": "fixed",
                "query_id": "keep",
                "query": "Which paper proposed graph diffusion?",
                "gold_paper_ids": ["arxiv:1"],
                "source_components": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid": "keep",
                        "question": "Which paper proposed graph diffusion?",
                        "answer": ["Graph Diffusion"],
                    }
                ),
                json.dumps(
                    {
                        "qid": "excluded",
                        "question": "Excluded query",
                        "answer": ["Excluded Gold"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = freeze_action_labels(
        partition_path=partition,
        source_path=source,
        output_path=tmp_path / "actions.jsonl",
        max_candidates=6,
    )

    assert manifest.query_count == 1
    assert manifest.action_count >= 2
    output = (tmp_path / "actions.jsonl").read_text(encoding="utf-8")
    assert "excluded" not in output.casefold()
    assert "Excluded Gold" not in output
