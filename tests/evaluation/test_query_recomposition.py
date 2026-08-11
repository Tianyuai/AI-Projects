from __future__ import annotations

import math
from typing import Any

import pytest

from paper_search.domain.models import Paper, QuerySpec
from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap
from paper_search.evaluation.query_recomposition import (
    RecompositionInput,
    RecompositionMethod,
    RecompositionProjection,
    SealedQueryRecompositionReport,
    build_report,
    compose_append,
    compose_round_robin,
    compose_rrf,
    project_all,
)


def _paper(identifier: str, **overrides: Any) -> Paper:
    return Paper(canonical_id=identifier, title=f"Title {identifier}", **overrides)


def _ids(papers: tuple[Paper, ...] | list[Paper]) -> tuple[str, ...]:
    return tuple(paper.canonical_id for paper in papers)


def _query_spec() -> QuerySpec:
    return QuerySpec(original_query="query", research_goal="research goal")


def _input(
    query_id: str = "q1",
    *,
    query_spec: QuerySpec | None = None,
    baseline_slots: tuple[tuple[Paper, ...], ...] = (),
    addition_slots: tuple[tuple[Paper, ...], ...] = (),
    retrieved_paper_ids: tuple[str, ...] = (),
    post_filter_paper_ids: tuple[str, ...] = (),
) -> RecompositionInput:
    return RecompositionInput(
        query_id=query_id,
        query_spec=query_spec or _query_spec(),
        baseline_slots=baseline_slots,
        addition_slots=addition_slots,
        retrieved_paper_ids=retrieved_paper_ids,
        post_filter_paper_ids=post_filter_paper_ids,
    )


def _projections_for_selected_counts(
    counts: dict[RecompositionMethod, int],
    *,
    query_id: str = "q1",
    total: int = 143,
) -> dict[RecompositionMethod, dict[str, RecompositionProjection]]:
    canonical_ids = tuple(f"openalex:W{index:04d}" for index in range(1, total + 1))
    projections: dict[RecompositionMethod, dict[str, RecompositionProjection]] = {}
    for method, selected_count in counts.items():
        projections[method] = {
            query_id: RecompositionProjection(
                method=method,
                retrieved_ids=canonical_ids,
                post_filter_ids=canonical_ids,
                selected_ids=canonical_ids[:selected_count],
            )
        }
    return projections


def _gold(total: int = 143) -> list[EvaluationQuery]:
    return [
        EvaluationQuery(
            query_id="q1",
            query="query",
            relevant_paper_ids=[f"openalex:W{index:04d}" for index in range(1, total + 1)],
        )
    ]


def test_compose_append_deduplicates_and_preserves_first_occurrence() -> None:
    a = _paper("openalex:W1")
    b = _paper("openalex:W2")
    c = _paper("openalex:W3")

    result = compose_append(((a, b), (), (c, a)))

    assert _ids(result) == ("openalex:W1", "openalex:W2", "openalex:W3")


def test_compose_append_deduplicates_by_normalized_canonical_id() -> None:
    first = _paper("https://openalex.org/W1")
    duplicate = _paper("openalex:w1")
    second = _paper("openalex:W2")

    result = compose_append(((first,), (duplicate, second)))

    assert _ids(result) == ("https://openalex.org/W1", "openalex:W2")


def test_compose_round_robin_merges_by_rank_and_skips_empty_slots() -> None:
    a = _paper("openalex:W1")
    b = _paper("openalex:W2")
    c = _paper("openalex:W3")
    d = _paper("openalex:W4")
    e = _paper("openalex:W5")

    result = compose_round_robin(((a, b), (), (c, d), (a, e)))

    assert _ids(result) == (
        "openalex:W1",
        "openalex:W3",
        "openalex:W2",
        "openalex:W4",
        "openalex:W5",
    )


def test_compose_round_robin_deduplicates_by_normalized_canonical_id() -> None:
    first = _paper("https://openalex.org/W1")
    duplicate = _paper("openalex:w1")
    second = _paper("openalex:W2")

    result = compose_round_robin(((first, second), (duplicate,)))

    assert _ids(result) == ("https://openalex.org/W1", "openalex:W2")


def test_compose_rrf_uses_fixed_k_60_and_deterministic_tie_breaks() -> None:
    a = _paper("openalex:W1")
    b = _paper("openalex:W2")
    c = _paper("openalex:W3")

    result = compose_rrf(((a,), (b, c)))

    assert _ids(result) == ("openalex:W1", "openalex:W2", "openalex:W3")

    tied = compose_rrf(((a,), (b,)))
    assert _ids(tied) == ("openalex:W1", "openalex:W2")


def test_compose_rrf_promotes_duplicate_consensus() -> None:
    a = _paper("openalex:W1")
    b = _paper("openalex:W2")
    c = _paper("openalex:W3")

    result = compose_rrf(((a, b), (b, c)))

    assert _ids(result)[0] == "openalex:W2"


def test_all_compositions_share_the_same_canonical_candidate_set() -> None:
    a = _paper("openalex:W1")
    b = _paper("openalex:W2")
    c = _paper("openalex:W3")
    d = _paper("openalex:W4")

    slots = ((a, b), (c,), (d, a))

    append_ids = set(_ids(compose_append(slots)))
    round_robin_ids = set(_ids(compose_round_robin(slots)))
    rrf_ids = set(_ids(compose_rrf(slots)))

    assert append_ids == round_robin_ids == rrf_ids == {
        "openalex:W1",
        "openalex:W2",
        "openalex:W3",
        "openalex:W4",
    }


def test_project_all_extends_authoritative_streams_and_only_selects_post_filter() -> None:
    a = _paper("openalex:W1")
    b = _paper("openalex:W2")
    c = _paper("openalex:W3")
    d = _paper("openalex:W4")

    inputs = (
        _input(
            baseline_slots=((a, b),),
            addition_slots=((c,), (d,)),
            retrieved_paper_ids=("openalex:W1", "openalex:W2", "openalex:W3"),
            post_filter_paper_ids=("openalex:W1", "openalex:W3"),
        ),
    )

    projections = project_all(inputs)

    assert set(projections) == {
        "append_v2",
        "round_robin_slots",
        "rrf_slots_k60",
    }
    for method, by_query in projections.items():
        projection = by_query["q1"]
        assert projection.retrieved_ids == (
            "openalex:W1",
            "openalex:W2",
            "openalex:W3",
            "openalex:W4",
        )
        assert projection.post_filter_ids == (
            "openalex:W1",
            "openalex:W3",
            "openalex:W4",
        )
        assert set(projection.selected_ids) <= {
            "openalex:W1",
            "openalex:W3",
            "openalex:W4",
        }
        assert projection.method == method


def test_project_all_adds_additions_to_retrieved_and_accepted_additions_to_post_filter() -> None:
    baseline = _paper("openalex:W1")
    accepted_addition = _paper("openalex:W2", publication_year=2021)
    rejected_addition = _paper("openalex:W3", publication_year=2019)
    query_spec = QuerySpec(
        original_query="query",
        research_goal="research goal",
        year_from=2020,
    )

    projections = project_all(
        (
            _input(
                query_spec=query_spec,
                baseline_slots=((baseline,),),
                addition_slots=((accepted_addition, rejected_addition),),
                retrieved_paper_ids=("openalex:W1",),
                post_filter_paper_ids=("openalex:W1",),
            ),
        )
    )

    for by_query in projections.values():
        projection = by_query["q1"]
        assert projection.retrieved_ids == (
            "openalex:W1",
            "openalex:W2",
            "openalex:W3",
        )
        assert projection.post_filter_ids == ("openalex:W1", "openalex:W2")
        assert "openalex:W2" in projection.selected_ids
        assert "openalex:W3" not in projection.selected_ids


def test_project_all_rejects_duplicate_query_ids() -> None:
    record = _input(
        baseline_slots=((_paper("openalex:W1"),),),
        retrieved_paper_ids=("openalex:W1",),
        post_filter_paper_ids=("openalex:W1",),
    )

    with pytest.raises(ValueError, match="duplicate query_id"):
        project_all((record, record))


@pytest.mark.parametrize(
    ("counts", "expected_conclusion", "expected_reason_code"),
    [
        (
            {
                "append_v2": 19,
                "round_robin_slots": 19,
                "rrf_slots_k60": 19,
            },
            "no_usable_recomposition_signal",
            "no_variant_passed_signal_gate",
        ),
        (
            {
                "append_v2": 19,
                "round_robin_slots": 29,
                "rrf_slots_k60": 29,
            },
            "signal_insufficient",
            "usable_signal_below_legacy_benchmark",
        ),
        (
            {
                "append_v2": 19,
                "round_robin_slots": 29,
                "rrf_slots_k60": 30,
            },
            "legacy_benchmark_met",
            "legacy_benchmark_met",
        ),
    ],
)
def test_build_report_orders_rows_and_classifies_conclusions(
    counts: dict[RecompositionMethod, int],
    expected_conclusion: str,
    expected_reason_code: str,
) -> None:
    projections = _projections_for_selected_counts(counts)
    report = build_report(
        gold=_gold(),
        identifier_map=IdentifierMap.from_bytes(b"{}"),
        projections=projections,
        input_hashes={"gold": "sha256:" + "1" * 64},
        current_formal_selected=17,
        legacy_title_selected=30,
    )

    assert isinstance(report, SealedQueryRecompositionReport)
    assert report.schema_version == "sealed-query-recomposition-offline-v1"
    assert [row.method for row in report.rows] == [
        "append_v2",
        "round_robin_slots",
        "rrf_slots_k60",
    ]
    assert report.conclusion == expected_conclusion
    assert report.reason_codes == (expected_reason_code,)
    assert report.current_formal_selected == 17
    assert report.legacy_title_selected == 30
    assert all(math.isfinite(row.macro_f1) for row in report.rows)
    assert all(math.isfinite(row.macro_recall) for row in report.rows)
    assert all(math.isfinite(row.micro_recall) for row in report.rows)
    assert all(math.isfinite(row.mrr) for row in report.rows)
    assert all(math.isfinite(row.ndcg) for row in report.rows)
    assert all(row.total_gold_associations == 143 for row in report.rows)
    assert report.rows[0].retains_append_selected_gold is True
    assert report.rows[1].retains_append_selected_gold is True
    assert report.rows[2].retains_append_selected_gold is True


def test_build_report_returns_integrity_failure_for_mismatched_projection() -> None:
    projections = _projections_for_selected_counts(
        {
            "append_v2": 19,
            "round_robin_slots": 29,
            "rrf_slots_k60": 30,
        }
    )
    malformed = dict(projections)
    malformed["rrf_slots_k60"] = {
        "q1": RecompositionProjection(
            method="rrf_slots_k60",
            retrieved_ids=("openalex:W0001",),
            post_filter_ids=("openalex:W0001",),
            selected_ids=("openalex:W0001",),
        )
    }

    report = build_report(
        gold=_gold(),
        identifier_map=IdentifierMap.from_bytes(b"{}"),
        projections=malformed,
        input_hashes={"gold": "sha256:" + "1" * 64},
        current_formal_selected=17,
        legacy_title_selected=30,
    )

    assert report.conclusion == "integrity_failure"
    assert report.reason_codes == ("experiment_integrity_failed",)


@pytest.mark.parametrize(
    ("conclusion", "reason_code"),
    [
        ("integrity_failure", "no_variant_passed_signal_gate"),
        ("no_usable_recomposition_signal", "experiment_integrity_failed"),
        ("signal_insufficient", "legacy_benchmark_met"),
        ("legacy_benchmark_met", "usable_signal_below_legacy_benchmark"),
    ],
)
def test_report_rejects_reason_code_that_does_not_match_conclusion(
    conclusion: str, reason_code: str
) -> None:
    report = build_report(
        gold=_gold(),
        identifier_map=IdentifierMap.from_bytes(b"{}"),
        projections=_projections_for_selected_counts(
            {
                "append_v2": 19,
                "round_robin_slots": 29,
                "rrf_slots_k60": 30,
            }
        ),
        input_hashes={"gold": "sha256:" + "1" * 64},
        current_formal_selected=17,
        legacy_title_selected=30,
    )
    payload = report.model_dump(mode="json")
    payload["conclusion"] = conclusion
    payload["reason_codes"] = [reason_code]

    with pytest.raises(ValueError, match="reason code must match conclusion"):
        SealedQueryRecompositionReport.model_validate(payload)


def test_build_report_counts_all_pipeline_stages_by_precedence() -> None:
    projections: dict[RecompositionMethod, dict[str, RecompositionProjection]] = {}
    for method in ("append_v2", "round_robin_slots", "rrf_slots_k60"):
        projections[method] = {
            "q1": RecompositionProjection(
                method=method,
                retrieved_ids=("openalex:W0001", "openalex:W0002", "openalex:W0003"),
                post_filter_ids=("openalex:W0002", "openalex:W0003"),
                selected_ids=("openalex:W0003",),
            )
        }

    report = build_report(
        gold=_gold(total=4),
        identifier_map=IdentifierMap.from_bytes(b"{}"),
        projections=projections,
        input_hashes={"gold": "sha256:" + "1" * 64},
        current_formal_selected=17,
        legacy_title_selected=30,
    )

    for row in report.rows:
        assert row.not_retrieved == 1
        assert row.filtered_out == 1
        assert row.ranked_outside_top50 == 1
        assert row.selected_top50 == 1
        assert row.usable_signal is False


@pytest.mark.parametrize(
    "malformed",
    [
        {
            "round_robin_slots": {
                "q1": RecompositionProjection(
                    method="round_robin_slots",
                    retrieved_ids=("openalex:W1",),
                    post_filter_ids=("openalex:W1",),
                    selected_ids=("openalex:W1",),
                )
            },
            "rrf_slots_k60": {
                "q1": RecompositionProjection(
                    method="rrf_slots_k60",
                    retrieved_ids=("openalex:W1",),
                    post_filter_ids=("openalex:W1",),
                    selected_ids=("openalex:W1",),
                )
            },
        },
        {
            "append_v2": {
                "q1": RecompositionProjection(
                    method="round_robin_slots",
                    retrieved_ids=("openalex:W1",),
                    post_filter_ids=("openalex:W1",),
                    selected_ids=("openalex:W1",),
                )
            },
            "round_robin_slots": {
                "q1": RecompositionProjection(
                    method="round_robin_slots",
                    retrieved_ids=("openalex:W1",),
                    post_filter_ids=("openalex:W1",),
                    selected_ids=("openalex:W1",),
                )
            },
            "rrf_slots_k60": {
                "q1": RecompositionProjection(
                    method="rrf_slots_k60",
                    retrieved_ids=("openalex:W1",),
                    post_filter_ids=("openalex:W1",),
                    selected_ids=("openalex:W1",),
                )
            },
        },
        {
            "append_v2": {
                "q1": RecompositionProjection(
                    method="append_v2",
                    retrieved_ids=("openalex:W1",),
                    post_filter_ids=("openalex:W1",),
                    selected_ids=("openalex:W2",),
                )
            },
            "round_robin_slots": {
                "q1": RecompositionProjection(
                    method="round_robin_slots",
                    retrieved_ids=("openalex:W1",),
                    post_filter_ids=("openalex:W1",),
                    selected_ids=("openalex:W2",),
                )
            },
            "rrf_slots_k60": {
                "q1": RecompositionProjection(
                    method="rrf_slots_k60",
                    retrieved_ids=("openalex:W1",),
                    post_filter_ids=("openalex:W1",),
                    selected_ids=("openalex:W2",),
                )
            },
        },
    ],
)
def test_build_report_rejects_invalid_projection_invariants(
    malformed: dict[RecompositionMethod, dict[str, RecompositionProjection]],
) -> None:
    report = build_report(
        gold=_gold(total=1),
        identifier_map=IdentifierMap.from_bytes(b"{}"),
        projections=malformed,
        input_hashes={"gold": "sha256:" + "1" * 64},
        current_formal_selected=17,
        legacy_title_selected=30,
    )

    assert report.conclusion == "integrity_failure"
