from __future__ import annotations

from paper_search.domain.models import UsageActual
from paper_search.recall_experiments.canary_reporting import (
    CanaryExecutionIdentity,
    CanaryInputIdentity,
    CanaryReport,
    CanaryPerQueryResult,
    CanaryRecallResult,
    compare_canary_results,
)
from paper_search.recall_experiments.identity import LiveRuntimeIdentity


def _identity() -> CanaryExecutionIdentity:
    pricing = "sha256:" + "e" * 64
    controller = "sha256:" + "f" * 64
    dependencies = {
        "search": {
            "identity_schema_version": "live-dependency-runtime-identity-v1",
            "provider": "openalex", "dependency": "openalex",
            "adapter": "openalex-works-v1", "model": None,
            "version": "live-capture-search-v1",
            "endpoints": ["https://api.openalex.org/works"], "operations": ["search"],
            "pricing_policy_sha256": pricing, "controller_policy_sha256": controller,
        },
        "citation": {
            "identity_schema_version": "live-dependency-runtime-identity-v1",
            "provider": "semantic_scholar", "dependency": "semantic_scholar",
            "adapter": "semantic-graph-v1", "model": None,
            "version": "live-capture-search-v1", "endpoints": ["https://s2.invalid"],
            "operations": ["search"], "pricing_policy_sha256": pricing,
            "controller_policy_sha256": controller,
        },
        "llm": {
            "identity_schema_version": "live-dependency-runtime-identity-v1",
            "provider": "deepseek", "dependency": "llm",
            "adapter": "openai-compatible-json", "model": "deepseek-v4-flash",
            "version": "openai-compatible-client-v1", "endpoints": ["https://deepseek.invalid"],
            "operations": ["generate_json"], "pricing_policy_sha256": pricing,
            "controller_policy_sha256": controller,
        },
    }
    return CanaryExecutionIdentity(
        identity_schema_version="recall-canary-execution-identity-v1",
        method_id="method-a", recipe_sha256="sha256:" + "a" * 64,
        input_sha256="sha256:" + "b" * 64, identifier_map_sha256=None,
        generator_type="fixed_actions", generator_model=None, prompt_sha256=None,
        actions_sha256="sha256:" + "c" * 64, allowed_actions=("text_search",),
        max_total_actions=1, max_results_per_action=5,
        candidate_pool_policy_version="production-dedup-v1",
        runtime=LiveRuntimeIdentity(
            identity_schema_version="candidate-recall-live-runtime-v1",
            controller_policy_sha256=controller, pricing_policy_sha256=pricing,
            dependencies=dependencies,
        ),
        snapshot_manifest_sha256="sha256:" + "d" * 64,
        snapshot_set_id="sha256:" + "1" * 64,
    )


def _result(*, scored: bool, ids: list[str], hits: list[str]) -> CanaryRecallResult:
    return CanaryRecallResult(
        candidate_pool_policy_version="production-dedup-v1",
        evaluation_status="available" if scored else "not_available",
        per_query=[
            CanaryPerQueryResult(
                query_id="q-1",
                candidate_pool_ids=ids,
                candidate_count=len(ids),
                evaluation_status="available" if scored else "not_available",
                gold_hit_ids=hits if scored else [],
                gold_association_count=2 if scored else None,
                gold_hit_count=len(hits) if scored else None,
                candidate_recall=len(hits) / 2 if scored else None,
            )
        ],
        gold_association_count=2 if scored else None,
        gold_hit_count=len(hits) if scored else None,
        macro_candidate_recall=len(hits) / 2 if scored else None,
    )


def test_unscored_result_keeps_the_same_metric_fields_as_scored_result() -> None:
    unscored = _result(scored=False, ids=["doi:10.1/a"], hits=[]).model_dump(mode="json")
    scored = _result(scored=True, ids=["doi:10.1/a"], hits=["doi:10.1/a"]).model_dump(
        mode="json"
    )

    assert set(unscored) == set(scored)
    assert set(unscored["per_query"][0]) == set(scored["per_query"][0])
    assert unscored["evaluation_status"] == "not_available"
    assert unscored["gold_hit_count"] is None


def test_comparison_reports_pool_overlap_and_gold_changes_with_fixed_shape() -> None:
    current = _result(
        scored=True,
        ids=["doi:10.1/a", "doi:10.1/b"],
        hits=["doi:10.1/a"],
    )
    baseline = _result(
        scored=True,
        ids=["doi:10.1/a", "doi:10.1/c"],
        hits=["doi:10.1/c"],
    )

    comparison = compare_canary_results(current, baseline, identities_match=False)

    assert comparison.evidence_level == "exploratory"
    assert comparison.per_query[0].current_candidate_count == 2
    assert comparison.per_query[0].baseline_candidate_count == 2
    assert comparison.per_query[0].intersection_count == 1
    assert comparison.per_query[0].jaccard == 1 / 3
    assert comparison.per_query[0].added_gold_hit_ids == ["doi:10.1/a"]
    assert comparison.per_query[0].lost_gold_hit_ids == ["doi:10.1/c"]


def test_unscored_comparison_keeps_gold_change_fields_empty() -> None:
    current = _result(scored=False, ids=["doi:10.1/a"], hits=[])
    baseline = _result(scored=False, ids=["doi:10.1/b"], hits=[])

    comparison = compare_canary_results(current, baseline, identities_match=True)

    assert comparison.evidence_level == "strict"
    assert comparison.per_query[0].added_gold_hit_ids == []
    assert comparison.per_query[0].lost_gold_hit_ids == []


def test_report_keeps_actions_usage_and_comparison_in_one_shape() -> None:
    report = CanaryReport(
        run_id="run-1",
        input=CanaryInputIdentity(
            input_kind="single", input_sha256="sha256:" + "a" * 64,
            evaluation_status="not_available", query_ids=("q-1",),
        ),
        execution_identity=_identity(),
        actions_by_query={"q-1": []},
        usage=UsageActual(),
        result=_result(scored=False, ids=["doi:10.1/a"], hits=[]),
        comparison=None,
    )

    payload = report.model_dump(mode="json")
    assert payload["schema_version"] == "recall-canary-report-v1"
    assert set(payload) == {
        "schema_version",
        "run_id",
        "input",
        "execution_identity",
        "actions_by_query",
        "usage",
        "result",
        "comparison",
    }
