from __future__ import annotations

import hashlib

import yaml

from scripts.run_semantic_action_replacement_live import (
    build_candidate_lock_bytes,
    candidate_action_budget_report,
    condition_comparison_label,
    parse_stratum_quotas,
    promotion_decision,
    select_disjoint_cases,
)


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def test_candidate_lock_changes_only_semantic_prompt_binding_and_approval() -> None:
    prompt = b"name: query_analyze\nversion: query-analyze-semantic-actions-v2\n"
    source = {
        "schema_version": "integrated-lock-v1",
        "lock_kind": "candidate",
        "baseline": {
            "prompt_version": "query-analyze-v1",
            "planner": {
                "prompt_config": {
                    "path": "configs/prompts/query_analyze.yaml",
                    "sha256": "sha256:" + "a" * 64,
                },
                "configured_subqueries_max": 6,
            },
            "retrieval": {"openalex_calls_max": 6},
        },
        "approval_ref": "old",
    }

    candidate = yaml.safe_load(
        build_candidate_lock_bytes(
            yaml.safe_dump(source, sort_keys=False).encode(),
            prompt_bytes=prompt,
            prompt_path="configs/prompts/query_analyze_semantic_actions_v2.yaml",
        )
    )

    assert candidate["baseline"]["prompt_version"] == (
        "query-analyze-semantic-actions-v2"
    )
    assert candidate["baseline"]["planner"]["prompt_config"] == {
        "path": "configs/prompts/query_analyze_semantic_actions_v2.yaml",
        "sha256": _sha256(prompt),
    }
    assert candidate["baseline"]["retrieval"] == source["baseline"]["retrieval"]
    assert candidate["approval_ref"].startswith("user-authorized-")


def test_candidate_lock_can_bind_single_call_protected_v3_without_other_changes() -> None:
    prompt = b"name: query_analyze\nversion: query-analyze-protected-actions-v3\n"
    source = {
        "baseline": {
            "prompt_version": "query-analyze-v1",
            "planner": {
                "prompt_config": {
                    "path": "configs/prompts/query_analyze.yaml",
                    "sha256": "sha256:" + "a" * 64,
                }
            },
        },
        "approval_ref": "old",
    }

    candidate = yaml.safe_load(
        build_candidate_lock_bytes(
            yaml.safe_dump(source, sort_keys=False).encode(),
            prompt_bytes=prompt,
            prompt_path="configs/prompts/query_analyze_protected_actions_v3.yaml",
            prompt_version="query-analyze-protected-actions-v3",
            approval_ref="user-authorized-protected-v3-live-confirmation-2026-08-28",
        )
    )

    assert candidate["baseline"]["prompt_version"] == (
        "query-analyze-protected-actions-v3"
    )
    assert candidate["baseline"]["planner"]["prompt_config"]["sha256"] == _sha256(
        prompt
    )
    assert candidate["approval_ref"].endswith("2026-08-28")


def test_select_disjoint_cases_is_gold_free_and_exactly_stratified() -> None:
    partition = []
    context = {}
    priority = {}
    strata = ["unconstrained", "method", "dataset", "year", "negation", "entity"]
    for index, stratum in enumerate(strata, start=1):
        query_id = f"q{index}"
        query = "Find papers about RL" if stratum == "entity" else f"query {stratum} {index}"
        labels = [] if stratum in {"unconstrained", "entity"} else [stratum]
        partition.append(
            {
                "query_id": query_id,
                "query": query,
                "gold_paper_ids": [f"arxiv:{index}"],
                "role": "training",
                "split": "auto_train",
            }
        )
        context[query_id] = {"labels": labels, "role": "training", "split": "auto_train"}
        priority[query_id] = {"base_gold_hit_count": 0, "base_candidate_count": 100}

    selected = select_disjoint_cases(
        partition,
        context_by_id=context,
        priority_by_id=priority,
        excluded_query_ids={"never-selected"},
        quotas={value: 1 for value in strata},
        seed="fixture",
    )

    assert {item["stratum"] for item in selected} == set(strata)
    assert all(set(item["network_payload"]) == {"query"} for item in selected)
    assert all("gold" not in str(item).casefold() for item in selected)


def test_action_budget_report_requires_llm_replacement_inside_six() -> None:
    report = candidate_action_budget_report(
        query="original query",
        subqueries=[
            {"text": "original query", "action_type": "text_search", "search_mode": "lexical"},
            {"text": "cross vocabulary formulation", "action_type": "text_search", "search_mode": "semantic"},
            {"text": "original query", "action_type": "text_search", "search_mode": "semantic"},
            {"text": "bridge action", "action_type": "text_search", "search_mode": "lexical", "query_id": "sq-supervised-lexical-bridge"},
        ],
        trace=[
            {
                "step": "supervised_query_expansion",
                "configured_action_budget": 6,
                "action_count_before": 3,
                "action_count_after": 4,
                "budget_policy": "llm-replaces-rule-fallback-before-local-bridge",
            }
        ],
    )

    assert report["within_six_action_budget"] is True
    assert report["novel_llm_action_count"] == 1
    assert report["bridge_inside_budget"] is True


def test_promotion_decision_requires_no_regression_and_strict_gain() -> None:
    passed = promotion_decision(
        {
            "query_count": 24,
            "live_replay_exact_query_count": 24,
            "candidate_action_budget_pass_query_count": 24,
            "candidate_f5_query_count": 24,
            "candidate_gold_pool_regressed_query_count": 0,
            "baseline_gold_pool_hit_query_count": 8,
            "candidate_gold_pool_hit_query_count": 10,
            "baseline_top5_hit_query_count": 6,
            "candidate_top5_hit_query_count": 6,
            "baseline_top10_hit_query_count": 7,
            "candidate_top10_hit_query_count": 8,
            "baseline_top20_hit_query_count": 8,
            "candidate_top20_hit_query_count": 9,
            "stratum_regression_count": 0,
            "candidate_llm_novel_query_count": 12,
            "candidate_replacement_observed_query_count": 4,
        }
    )
    failed = promotion_decision(
        {
            **passed["metrics"],
            "candidate_top5_hit_query_count": 5,
        }
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert "top5_regression" in failed["failed_gates"]


def test_parse_stratum_quotas_allows_explicit_zero_for_exhausted_stratum() -> None:
    quotas = parse_stratum_quotas(
        '{"unconstrained":8,"method":6,"dataset":5,"negation":0,"entity":5}'
    )

    assert sum(quotas.values()) == 24
    assert quotas["negation"] == 0


def test_condition_comparison_label_uses_frozen_execution_prompt_versions() -> None:
    def record(version: str) -> dict[str, object]:
        return {
            "live_execution": {
                "outcome": {
                    "kind": "success",
                    "response": {"prompt_version": version},
                }
            }
        }

    label = condition_comparison_label(
        [record("query-analyze-v1")],
        [record("query-analyze-protected-actions-v3")],
    )

    assert label == (
        "query-analyze-v1-versus-query-analyze-protected-actions-v3"
    )
