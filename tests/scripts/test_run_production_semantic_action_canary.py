from __future__ import annotations

import pytest

from scripts.run_production_semantic_action_canary import (
    _write_immutable,
    _search_budget,
    _source_papers,
    _rank_condition,
    build_openalex_request_spec,
    build_paired_model_outputs,
    online_source_ranks,
    normalized_action_identity,
    select_openalex_cases,
)


QUERY = "Which papers use graph networks for molecular property prediction?"


def _case(*, pair_eligible: bool = True) -> dict[str, object]:
    exact = {
        "action_type": "text_search",
        "priority": 1,
        "provider_hint": "either",
        "query_id": "sq-1",
        "query_type": "exact",
        "search_mode": "lexical",
        "target_constraints": [],
        "text": QUERY,
    }
    novel = {
        "action_type": "text_search",
        "priority": 2,
        "provider_hint": "openalex",
        "query_id": "sq-2",
        "query_type": "expanded",
        "search_mode": "semantic",
        "target_constraints": [],
        "text": "message passing molecular property estimation",
    }
    return {
        "case_id": "AutoScholarQuery_train_1",
        "query": QUERY,
        "stratum": "method",
        "openalex_pair_eligible": pair_eligible,
        "model_output": {
            "query_spec": {
                "original_query": QUERY,
                "research_goal": "Find relevant studies.",
                "must_have": [],
                "should_have": [],
                "exclusions": [],
                "methods": [],
                "datasets": [],
                "tasks": [],
                "topics": [],
                "domains": [],
                "venues": [],
                "year_from": None,
                "year_to": None,
                "ambiguities": [],
            },
            "search_plan": {
                "subqueries": [exact, novel],
                "inherited_hard_filters": {},
                "rationale": "paired validation fixture",
            },
        },
        "accepted_novel_model_actions": [novel],
    }


def test_build_paired_outputs_keeps_one_common_exact_action() -> None:
    case = _case()
    action = case["accepted_novel_model_actions"][0]  # type: ignore[index]

    baseline, augmented = build_paired_model_outputs(case, action)  # type: ignore[arg-type]

    assert [item["text"] for item in baseline["search_plan"]["subqueries"]] == [
        QUERY
    ]
    assert [item["text"] for item in augmented["search_plan"]["subqueries"]] == [
        QUERY,
        "message passing molecular property estimation",
    ]
    assert baseline["query_spec"] == augmented["query_spec"]


def test_build_paired_outputs_synthesizes_anchor_when_model_omits_it() -> None:
    case = _case()
    case["model_output"]["search_plan"]["subqueries"][0]["text"] = (  # type: ignore[index]
        "graph networks molecular properties"
    )
    action = case["accepted_novel_model_actions"][0]  # type: ignore[index]

    baseline, augmented = build_paired_model_outputs(case, action)  # type: ignore[arg-type]

    assert baseline["search_plan"]["subqueries"][0]["text"] == QUERY
    assert augmented["search_plan"]["subqueries"][0]["text"] == QUERY
    assert baseline["search_plan"]["subqueries"][0]["query_type"] == "exact"


def test_select_openalex_cases_rejects_completed_action_identity() -> None:
    case = _case()
    action = case["accepted_novel_model_actions"][0]  # type: ignore[index]
    completed = {normalized_action_identity(case["case_id"], action)}

    assert select_openalex_cases([case], completed, limit=12) == []


def test_select_openalex_cases_rejects_pair_ineligible_or_s2_only() -> None:
    ineligible = _case(pair_eligible=False)
    s2_only = _case()
    action = s2_only["accepted_novel_model_actions"][0]  # type: ignore[index]
    action["provider_hint"] = "semantic_scholar"

    assert select_openalex_cases([ineligible, s2_only], set(), limit=12) == []


def test_select_openalex_cases_keeps_one_new_action_without_gold_material() -> None:
    selected = select_openalex_cases([_case()], set(), limit=12)

    assert len(selected) == 1
    assert selected[0]["case_id"] == "AutoScholarQuery_train_1"
    assert "gold" not in str(selected[0]).casefold()


def test_select_openalex_cases_restores_omitted_default_action_fields() -> None:
    case = _case()
    action = case["accepted_novel_model_actions"][0]  # type: ignore[index]
    action.pop("action_type")
    action.pop("search_mode")

    selected = select_openalex_cases([case], set(), limit=12)

    assert len(selected) == 1
    assert selected[0]["selected_action"]["action_type"] == "text_search"  # type: ignore[index]
    assert selected[0]["selected_action"]["search_mode"] == "lexical"  # type: ignore[index]


def test_select_openalex_cases_requires_positive_limit() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_openalex_cases([_case()], set(), limit=0)


def test_openalex_request_spec_contains_only_gold_blind_selected_action() -> None:
    case = _case()
    case["gold_paper_ids"] = ["doi:10.1000/forbidden"]
    case["final_test"] = {"query_id": "forbidden"}

    request = build_openalex_request_spec(case)

    assert request == {
        "query_text": "message passing molecular property estimation",
        "search_mode": "semantic",
        "filters": {"_search_mode": "semantic"},
        "limit": 50,
    }
    assert "gold" not in str(request).casefold()
    assert "final_test" not in str(request).casefold()


def test_online_source_ranks_excludes_pasa_but_keeps_openalex_action_families() -> None:
    candidate = {
        "source_ranks": {
            "pasa-local-original@abc": 1,
            "ceiling-candidate-anchor@def": 7,
            "openalex:semantic-action-v2": 3,
        }
    }

    assert online_source_ranks(candidate) == {
        "ceiling-candidate-anchor@def": 7,
        "openalex:semantic-action-v2": 3,
    }


def test_immutable_json_rerun_reuses_original_creation_time(tmp_path) -> None:
    path = tmp_path / "receipt.json"

    _write_immutable(path, {"created_at": "first", "value": 1})
    _write_immutable(path, {"created_at": "second", "value": 1})

    assert '"created_at": "first"' in path.read_text(encoding="utf-8")


def test_immutable_json_rerun_rejects_substantive_change(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    _write_immutable(path, {"created_at": "first", "value": 1})

    with pytest.raises(ValueError, match="already differs"):
        _write_immutable(path, {"created_at": "second", "value": 2})


def test_openalex_only_budget_starts_in_continue_state() -> None:
    from paper_search.control.budget import HardBudgetController

    controller = HardBudgetController(_search_budget(), formal_live=True)

    assert controller.stop_status() == "continue"


def test_frozen_baseline_score_candidate_schema_loads_as_online_sources() -> None:
    row = {
        "candidates": [
            {
                "paper": {
                    "canonical_id": "openalex:W1",
                    "title": "A paper",
                    "openalex_id": "W1",
                    "sources": ["openalex"],
                },
                "baseline_score": 1 / 61,
                "source_ranks": {
                    "pasa-local-original@a": 1,
                    "ceiling-candidate-anchor@b": 2,
                },
            }
        ]
    }

    sources = _source_papers(row)

    assert list(sources) == ["ceiling-candidate-anchor@b"]
    assert sources["ceiling-candidate-anchor@b"][0][0] == 2
    assert sources["ceiling-candidate-anchor@b"][0][1].canonical_id == "openalex:W1"


def test_rank_condition_runs_current_filter_and_fair_cap_modules() -> None:
    from paper_search.domain.models import Paper, QuerySpec

    class IdentityRanker:
        def rank(self, _query, candidates):
            return list(candidates)

    paper = Paper(
        canonical_id="openalex:W1",
        title="A paper",
        openalex_id="W1",
        sources=["openalex"],
    )

    result = _rank_condition(
        query="papers",
        query_spec=QuerySpec(original_query="papers", research_goal="Find papers"),
        sources={"openalex:action": [(1, paper)]},
        ranker=IdentityRanker(),
        identifier_map=None,
        raw_cap=300,
        deduplicated_cap=200,
        output_cap=50,
    )

    assert [item.paper.canonical_id for item in result["ranked"]] == [
        "openalex:W1"
    ]
