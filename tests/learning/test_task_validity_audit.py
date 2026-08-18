from __future__ import annotations

from paper_search.learning.task_validity_audit import (
    build_blind_review_packet,
    build_task_validity_cases,
    summarize_objective_task_validity_proxies,
)


def _case(role: str, cohort: str, index: int) -> dict[str, object]:
    return {
        "query_id": f"{role}-{cohort}-{index}",
        "role": role,
        "cohort": cohort,
        "fold": index % 3 + 1,
        "query": f"question {role} {cohort} {index}",
        "gold_titles": [f"gold title {index}"],
        "gold_ids": [f"arxiv:2001.{index:05d}"],
        "gold_availability": ["available"],
        "best_gold_title_token_recall": 0.0 if cohort == "miss" else 0.5,
    }


def test_blind_review_packet_is_balanced_deterministic_and_hides_outcome() -> None:
    rows = [
        _case(role, cohort, index)
        for role in ("training", "development")
        for cohort in ("miss", "hit")
        for index in range(5)
    ]
    targets = {
        "training": {"miss": 2, "hit": 2},
        "development": {"miss": 1, "hit": 1},
    }

    first = build_blind_review_packet(rows, targets=targets, seed="audit-v1")
    second = build_blind_review_packet(rows, targets=targets, seed="audit-v1")

    assert first == second
    assert len(first["review_cases"]) == 6
    assert len(first["private_key"]) == 6
    assert all("query_id" not in row and "cohort" not in row for row in first["review_cases"])
    assert all(row["adjudication"]["gold_relevance"] is None for row in first["review_cases"])
    counts = first["selection_counts"]
    assert counts["training"]["miss"] == 2
    assert counts["training"]["hit"] == 2
    assert counts["development"]["miss"] == 1
    assert counts["development"]["hit"] == 1


def test_blind_review_packet_rejects_insufficient_stratum() -> None:
    rows = [_case("training", "miss", 1)]

    try:
        build_blind_review_packet(
            rows,
            targets={"training": {"miss": 2}},
            seed="audit-v1",
        )
    except ValueError as exc:
        assert "insufficient cases" in str(exc)
    else:
        raise AssertionError("expected insufficient stratum rejection")


def test_task_validity_cases_join_raw_gold_and_objective_availability() -> None:
    raw = {
        "q-1": {
            "qid": "q-1",
            "question": "Which work studies graph diffusion retrieval?",
            "answer": ["Graph Diffusion for Retrieval"],
            "answer_arxiv_id": ["2001.00001v2"],
        }
    }

    rows = build_task_validity_cases(
        raw_by_id=raw,
        evidence_rows=[
            {"query_id": "q-1", "role": "training", "cohort": "miss", "fold": 2}
        ],
        availability_by_gold_id={"arxiv:2001.00001": "available"},
    )

    assert rows[0]["gold_ids"] == ["arxiv:2001.00001"]
    assert rows[0]["gold_availability"] == ["available"]
    assert rows[0]["best_gold_title_token_recall"] > 0


def test_task_validity_cases_mark_unchecked_development_availability() -> None:
    raw = {
        "q-dev": {
            "qid": "q-dev",
            "question": "question",
            "answer": ["A title"],
            "answer_arxiv_id": ["2101.00002"],
        }
    }

    rows = build_task_validity_cases(
        raw_by_id=raw,
        evidence_rows=[
            {"query_id": "q-dev", "role": "development", "cohort": "hit", "fold": 1}
        ],
        availability_by_gold_id={},
    )

    assert rows[0]["gold_availability"] == ["not_audited"]


def test_objective_proxy_summary_unblinds_only_after_joining_private_key() -> None:
    review_cases = [
        {
            "case_id": "case-1",
            "gold_titles": ["One"],
            "gold_availability": ["available"],
            "best_gold_title_token_recall": 0.0,
        },
        {
            "case_id": "case-2",
            "gold_titles": ["Two", "Three"],
            "gold_availability": ["available", "missing"],
            "best_gold_title_token_recall": 0.5,
        },
    ]
    private_key = [
        {"case_id": "case-1", "role": "training", "cohort": "miss", "fold": 1},
        {"case_id": "case-2", "role": "training", "cohort": "hit", "fold": 2},
    ]

    summary = summarize_objective_task_validity_proxies(
        review_cases=review_cases,
        private_key=private_key,
    )

    assert summary["training"]["miss"]["zero_title_overlap_rate"] == 1.0
    assert summary["training"]["hit"]["mean_gold_count"] == 2.0
    assert summary["training"]["hit"]["all_gold_available_rate"] == 0.0
