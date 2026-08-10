from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from paper_search.domain.models import (
    Paper,
    ProviderResult,
    QuerySpec,
    UsageActual,
)
from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap
from paper_search.evaluation.query_evolution_probe import (
    calculate_production_estimates,
    FrozenProbeInputs,
    FrozenQueryRecord,
    ProbeIntegrity,
    evaluate_probe,
    merge_probe_results,
    public_probe_report,
    reconstruct_frozen_baseline,
    select_probe_query_ids,
)


def _paper(identifier: str, *, year: int = 2022) -> Paper:
    return Paper(
        canonical_id=identifier,
        title=f"Title {identifier}",
        publication_year=year,
        venue="Journal",
        openalex_id=identifier,
        is_retracted=False,
    )


def _result(papers: list[Paper], *, errors: list[dict[str, object]] | None = None) -> ProviderResult[list[Paper]]:
    return ProviderResult(
        data=papers,
        usage=UsageActual(),
        provenance={
            "provider": "openalex",
            "endpoint": "/works",
            "model_id": "offline",
            "requested_at": datetime(2026, 8, 10, tzinfo=UTC).isoformat(),
            "response_hash": "sha256:" + "1" * 64,
        },
        cache_hit=True,
        latency_ms=0,
        errors=errors or [],
    )


def _record(
    query_id: str,
    papers: list[Paper],
    *,
    retrieved_paper_ids: list[str] | None = None,
    source_index: int = 0,
) -> FrozenQueryRecord:
    return FrozenQueryRecord(
        query_id=query_id,
        query_spec=QuerySpec(original_query=query_id, research_goal="find papers"),
        baseline_results=[_result(papers)],
        retrieved_paper_ids=(
            retrieved_paper_ids
            if retrieved_paper_ids is not None
            else [paper.canonical_id for paper in papers]
        ),
        source_index=source_index,
    )


def _baseline(*records: FrozenQueryRecord):
    ordered_records = [record.model_copy(update={"source_index": index}) for index, record in enumerate(records)]
    total_selected = sum(len(result.data) for record in ordered_records for result in record.baseline_results)
    return reconstruct_frozen_baseline(
        FrozenProbeInputs(
            queries=ordered_records,
            source_run_id="dev-run",
            source_hashes={"business_results_sha256": "sha256:" + "2" * 64},
            expected_query_count=60 if len(records) == 60 else None,
            expected_total_selected=2910 if total_selected == 2910 else None,
        ),
        replay_provider=None,
    )


def test_reconstructs_frozen_order_and_2910_denominator() -> None:
    records = []
    for index in range(60):
        count = 50 if index < 30 else 47
        records.append(_record(f"q-{index:02d}", [_paper(f"openalex:W{index * 100 + n + 1}") for n in range(count)]))

    baseline = _baseline(*records)

    assert baseline.query_ids == tuple(f"q-{index:02d}" for index in range(60))
    assert baseline.query_count == 60
    assert baseline.total_selected == 2910


def test_selects_available_not_retrieved_queries_in_frozen_order() -> None:
    baseline = _baseline(
        _record("q-1", [_paper("openalex:W1")]),
        _record("q-2", [_paper("openalex:W2")]),
        _record("q-3", [_paper("openalex:W3")]),
    )
    gold = [
        EvaluationQuery(query_id="q-1", query="q-1", relevant_paper_ids=["openalex:W1", "openalex:W9"]),
        EvaluationQuery(query_id="q-2", query="q-2", relevant_paper_ids=["openalex:W2"]),
        EvaluationQuery(query_id="q-3", query="q-3", relevant_paper_ids=["openalex:W8"]),
    ]

    assert select_probe_query_ids(
        baseline,
        gold,
        {"openalex:W1": "available", "openalex:W8": "available", "openalex:W9": "not_available"},
    ) == ("q-3",)


def test_selection_excludes_gold_already_present_in_full_retrieved_stream() -> None:
    baseline = _baseline(
        _record(
            "q-1",
            [_paper("openalex:W1")],
            retrieved_paper_ids=["openalex:W1", "openalex:W9"],
        )
    )
    gold = [EvaluationQuery(query_id="q-1", query="q-1", relevant_paper_ids=["openalex:W9"])]

    assert select_probe_query_ids(baseline, gold, {"openalex:W9": "available"}) == ()


def test_merge_preserves_baseline_then_search_1_then_search_2_and_first_id() -> None:
    baseline = _baseline(_record("q-1", [_paper("openalex:W1"), _paper("openalex:W2")]))
    projection = merge_probe_results(
        baseline,
        {
            "q-1": [
                _result([_paper("openalex:W2"), _paper("openalex:W3")]),
                _result([_paper("openalex:W4")]),
            ]
        },
    )

    assert projection.by_query["q-1"].candidate_ids == (
        "openalex:W1",
        "openalex:W2",
        "openalex:W3",
        "openalex:W4",
    )
    assert projection.by_query["q-1"].fusion_sources == ("openalex",)


def test_evaluation_computes_14_8_baseline_and_gate_boundaries() -> None:
    baseline = _baseline(
        _record("q-1", [_paper("openalex:W1"), _paper("openalex:W2")]),
        _record("q-2", [_paper("openalex:W3")]),
    )
    projection = merge_probe_results(
        baseline,
        {"q-1": [_result([_paper("openalex:W4")])], "q-2": []},
    )
    gold = [
        EvaluationQuery(query_id="q-1", query="q-1", relevant_paper_ids=["openalex:W1", "openalex:W2", "openalex:W4"]),
        EvaluationQuery(query_id="q-2", query="q-2", relevant_paper_ids=["openalex:W3"]),
    ]

    evaluation = evaluate_probe(
        baseline,
        projection,
        gold,
        id_map=None,
        integrity=ProbeIntegrity(capture_replay_match="matched", balanced_production_estimate=Decimal("0.10")),
    )

    assert evaluation.baseline_candidate_gold_count == 3
    assert evaluation.candidate_candidate_gold_count == 4
    assert evaluation.newly_retrieved_count == 1
    assert evaluation.gate_a == "failed"
    assert evaluation.gate_b == "not_evaluated"
    assert evaluation.gate_c == "not_evaluated"


def test_gold_associations_are_unique_after_identifier_resolution() -> None:
    baseline = _baseline(_record("q-1", [_paper("openalex:W1")]))
    projection = merge_probe_results(baseline, {})
    gold = [
        EvaluationQuery(
            query_id="q-1",
            query="q-1",
            relevant_paper_ids=["openalex:W1", "openalex:W2"],
        )
    ]

    evaluation = evaluate_probe(
        baseline,
        projection,
        gold,
        id_map=IdentifierMap({"openalex:W2": "openalex:W1"}),
        integrity=ProbeIntegrity(capture_replay_match="matched"),
    )

    assert evaluation.baseline_candidate_gold_count == 1
    assert evaluation.candidate_candidate_gold_count == 1
    assert evaluation.baseline_top50_gold_count == 1
    assert evaluation.candidate_top50_gold_count == 1


def test_retrieved_but_unselected_gold_is_retained_not_new() -> None:
    baseline = _baseline(
        _record(
            "q-1",
            [_paper("openalex:W1")],
            retrieved_paper_ids=["openalex:W1", "openalex:W2"],
        )
    )
    projection = merge_probe_results(
        baseline,
        {"q-1": [_result([_paper("openalex:W2")])]},
    )
    gold = [
        EvaluationQuery(
            query_id="q-1",
            query="q-1",
            relevant_paper_ids=["openalex:W2"],
        )
    ]

    evaluation = evaluate_probe(
        baseline,
        projection,
        gold,
        id_map=None,
        integrity=ProbeIntegrity(capture_replay_match="matched"),
    )

    assert evaluation.baseline_candidate_gold_count == 1
    assert evaluation.candidate_candidate_gold_count == 1
    assert evaluation.newly_retrieved_count == 0
    assert evaluation.prior_candidate_gold_retained is True


def test_true_retrieval_gain_does_not_change_top50_without_ranking_gain() -> None:
    selected = [_paper(f"openalex:W{index}") for index in range(1, 51)]
    baseline = _baseline(_record("q-1", selected))
    projection = merge_probe_results(
        baseline,
        {"q-1": [_result([_paper("openalex:W51")])]},
    )
    gold = [
        EvaluationQuery(
            query_id="q-1",
            query="q-1",
            relevant_paper_ids=["openalex:W51"],
        )
    ]

    evaluation = evaluate_probe(
        baseline,
        projection,
        gold,
        id_map=None,
        integrity=ProbeIntegrity(capture_replay_match="matched"),
    )

    assert evaluation.baseline_candidate_gold_count == 0
    assert evaluation.candidate_candidate_gold_count == 1
    assert evaluation.newly_retrieved_count == 1
    assert evaluation.baseline_top50_gold_count == 0
    assert evaluation.candidate_top50_gold_count == 0


def test_invalid_request_failure_fails_gate_a_and_blocks_later_gates() -> None:
    baseline = _baseline(_record("q-1", [_paper("openalex:W1")]))
    projection = merge_probe_results(baseline, {"q-1": []})
    gold = [EvaluationQuery(query_id="q-1", query="q-1", relevant_paper_ids=["openalex:W1"])]

    evaluation = evaluate_probe(
        baseline,
        projection,
        gold,
        id_map=None,
        integrity=ProbeIntegrity(capture_replay_match="not_evaluated", request_failures=1),
    )

    assert evaluation.gate_a == "failed"
    assert evaluation.gate_b == "not_evaluated"
    assert evaluation.gate_c == "not_evaluated"


def test_hash_mismatch_fails_gate_a_without_synthesizing_gate_b_or_c() -> None:
    baseline = _baseline(_record("q-1", [_paper("openalex:W1")]))
    projection = merge_probe_results(baseline, {"q-1": []})
    gold = [EvaluationQuery(query_id="q-1", query="q-1", relevant_paper_ids=["openalex:W1"])]

    evaluation = evaluate_probe(
        baseline,
        projection,
        gold,
        id_map=None,
        integrity=ProbeIntegrity(capture_replay_match="matched", availability_hash_mismatch=True),
    )

    assert evaluation.gate_a == "failed"
    assert evaluation.gate_b == "not_evaluated"
    assert evaluation.gate_c == "not_evaluated"


def test_gate_a_compares_locked_probe_count_with_terminal_count() -> None:
    records = [
        _record(
            f"q-{index:02d}",
            [
                _paper(f"openalex:W{index * 100 + offset + 1}")
                for offset in range(50 if index < 30 else 47)
            ],
        )
        for index in range(60)
    ]
    baseline = _baseline(*records)
    projection = merge_probe_results(baseline, {})
    gold = [
        EvaluationQuery(
            query_id=record.query_id,
            query=record.query_id,
            relevant_paper_ids=[record.baseline_results[0].data[0].canonical_id],
        )
        for record in records
    ]

    evaluation = evaluate_probe(
        baseline,
        projection,
        gold,
        id_map=None,
        integrity=ProbeIntegrity(
            capture_replay_match="matched",
            locked_query_count=55,
            terminal_count=55,
        ),
    )

    assert evaluation.gate_a == "passed"


def test_gate_a_rejects_incomplete_locked_probe_terminals() -> None:
    baseline = _baseline(_record("q-1", [_paper("openalex:W1")]))
    projection = merge_probe_results(baseline, {})
    gold = [EvaluationQuery(query_id="q-1", query="q-1", relevant_paper_ids=["openalex:W1"])]

    evaluation = evaluate_probe(
        baseline,
        projection,
        gold,
        id_map=None,
        integrity=ProbeIntegrity(
            capture_replay_match="matched",
            locked_query_count=55,
            terminal_count=54,
        ),
    )

    assert evaluation.gate_a == "failed"


def test_public_report_is_aggregate_only_and_finite() -> None:
    baseline = _baseline(_record("q-1", [_paper("openalex:W1")]))
    projection = merge_probe_results(baseline, {"q-1": []})
    gold = [EvaluationQuery(query_id="q-1", query="q-1", relevant_paper_ids=["openalex:W1"])]
    evaluation = evaluate_probe(
        baseline,
        projection,
        gold,
        id_map=None,
        integrity=ProbeIntegrity(capture_replay_match="matched"),
    )

    report = public_probe_report(evaluation)
    payload = report.model_dump(mode="json")

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    assert not {key for key in keys(payload) if key in {"query_id", "paper_id", "title", "request_id", "response"}}
    assert report.gate_a in {"passed", "failed", "not_evaluated"}


def test_non_finite_estimate_is_rejected() -> None:
    with pytest.raises(ValueError):
        ProbeIntegrity(balanced_production_estimate=float("inf"))


def test_production_estimates_ignore_unscheduled_zero_usage_slots() -> None:
    estimates = calculate_production_estimates(
        {"search-1": [UsageActual(search_api_calls=3, elapsed_ms=10), UsageActual()]},
        scheduled_by_operation={"search-1": [True, False]},
    )

    assert estimates["search-1"].search_api_calls == 4
    assert estimates["search-1"].elapsed_ms == 12
