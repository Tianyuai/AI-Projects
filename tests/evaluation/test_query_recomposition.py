from __future__ import annotations

import math

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


def _paper(identifier: str) -> Paper:
    return Paper(canonical_id=identifier, title=f"Title {identifier}")


def _ids(papers: tuple[Paper, ...] | list[Paper]) -> tuple[str, ...]:
    return tuple(paper.canonical_id for paper in papers)


def _query_spec() -> QuerySpec:
    return QuerySpec(original_query="query", research_goal="research goal")


def _input(
    query_id: str = "q1",
    *,
    baseline_slots: tuple[tuple[Paper, ...], ...] = (),
    addition_slots: tuple[tuple[Paper, ...], ...] = (),
    retrieved_paper_ids: tuple[str, ...] = (),
    post_filter_paper_ids: tuple[str, ...] = (),
) -> RecompositionInput:
    return RecompositionInput(
        query_id=query_id,
        query_spec=_query_spec(),
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


def test_compose_rrf_uses_fixed_k_60_and_deterministic_tie_breaks() -> None:
    a = _paper("openalex:W1")
    b = _paper("openalex:W2")
    c = _paper("openalex:W3")

    result = compose_rrf(((a,), (b, c)))

    assert _ids(result) == ("openalex:W1", "openalex:W2", "openalex:W3")

    tied = compose_rrf(((a,), (b,)))
    assert _ids(tied) == ("openalex:W1", "openalex:W2")


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


def test_project_all_inherits_authoritative_streams_and_only_selected_top50() -> None:
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
        assert projection.retrieved_ids == ("openalex:W1", "openalex:W2", "openalex:W3")
        assert projection.post_filter_ids == ("openalex:W1", "openalex:W3")
        assert set(projection.selected_ids) <= {"openalex:W1", "openalex:W3"}
        assert projection.method == method


@pytest.mark.parametrize(
    ("counts", "expected_conclusion"),
    [
        (
            {
                "append_v2": 19,
                "round_robin_slots": 19,
                "rrf_slots_k60": 19,
            },
            "no_usable_recomposition_signal",
        ),
        (
            {
                "append_v2": 19,
                "round_robin_slots": 29,
                "rrf_slots_k60": 29,
            },
            "signal_insufficient",
        ),
        (
            {
                "append_v2": 19,
                "round_robin_slots": 29,
                "rrf_slots_k60": 30,
            },
            "legacy_benchmark_met",
        ),
    ],
)
def test_build_report_orders_rows_and_classifies_conclusions(
    counts: dict[RecompositionMethod, int],
    expected_conclusion: str,
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
