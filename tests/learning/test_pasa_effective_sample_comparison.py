from paper_search.learning.pasa_effective_sample_comparison import (
    compare_effective_sample_coverage,
)


def test_comparison_counts_only_new_pair_feasible_gold_injections() -> None:
    query_rows = [
        {
            "query_id": "q_existing",
            "candidate_count": 10,
            "gold_paper_count": 2,
            "gold_hit_count": 1,
            "hard_negative_candidate_count": 9,
        },
        {
            "query_id": "q_rescued",
            "candidate_count": 8,
            "gold_paper_count": 2,
            "gold_hit_count": 0,
            "hard_negative_candidate_count": 8,
        },
        {
            "query_id": "q_no_negative",
            "candidate_count": 1,
            "gold_paper_count": 1,
            "gold_hit_count": 1,
            "hard_negative_candidate_count": 0,
        },
        {
            "query_id": "q_not_strict",
            "candidate_count": 5,
            "gold_paper_count": 1,
            "gold_hit_count": 0,
            "hard_negative_candidate_count": 5,
        },
    ]
    gold_ids = {
        "q_existing": ["arxiv:1", "arxiv:2"],
        "q_rescued": ["arxiv:3", "arxiv:4"],
        "q_no_negative": ["arxiv:5"],
        "q_not_strict": ["arxiv:6"],
    }
    signal_eligibility = {
        "q_existing": {"task_provenance": True, "method": False},
        "q_rescued": {"task_provenance": True, "method": True},
        "q_no_negative": {"task_provenance": False, "method": True},
    }

    summary, queue = compare_effective_sample_coverage(
        query_rows=query_rows,
        gold_ids_by_query=gold_ids,
        strict_ready_query_ids={"q_existing", "q_rescued", "q_no_negative"},
        pasa_available_gold_ids={"arxiv:1", "arxiv:3", "arxiv:4", "arxiv:6"},
        signal_eligibility_by_query=signal_eligibility,
        shallow_candidate_threshold=5,
        minimum_hard_negatives=2,
        hard_negative_limit=100,
    )

    assert summary["strict_ready_query_count"] == 3
    assert summary["base"]["gold_hit_query_count"] == 2
    assert summary["base"]["positive_and_hard_negative_query_count"] == 1
    assert summary["openalex_pasa"]["gold_hit_query_count"] == 3
    assert summary["openalex_pasa"]["positive_and_hard_negative_query_count"] == 2
    assert summary["delta"]["positive_and_hard_negative_query_count"] == 1
    assert summary["delta"]["reliability_pair_count"] == 16
    assert summary["delta"]["task_provenance_pair_count"] == 16
    assert summary["signals"]["method"]["rescued_pair_feasible_query_count"] == 1
    assert summary["pasa"]["direct_gold_candidate_count_for_rescues"] == 2
    assert summary["pasa"]["strict_ready_ceiling_unchanged"] is True
    assert [row["query_id"] for row in queue] == ["q_no_negative", "q_rescued"]
    assert queue[0]["recommended_action"] == "targeted_pasa_lexical_negative"
    assert queue[1]["recommended_action"] == "pasa_mixed_lexical_gold_supplement"
    assert summary["projection"]["safe_training_package_materialized"] is False


def test_comparison_does_not_claim_gold_rescue_without_hard_negative() -> None:
    summary, queue = compare_effective_sample_coverage(
        query_rows=[
            {
                "query_id": "q1",
                "candidate_count": 0,
                "gold_paper_count": 1,
                "gold_hit_count": 0,
                "hard_negative_candidate_count": 0,
            }
        ],
        gold_ids_by_query={"q1": ["arxiv:1"]},
        strict_ready_query_ids={"q1"},
        pasa_available_gold_ids={"arxiv:1"},
        signal_eligibility_by_query={"q1": {"year": True}},
        shallow_candidate_threshold=5,
        minimum_hard_negatives=2,
        hard_negative_limit=100,
    )

    assert summary["openalex_pasa"]["gold_hit_query_count"] == 1
    assert summary["openalex_pasa"]["positive_and_hard_negative_query_count"] == 0
    assert summary["signals"]["year"]["rescued_pair_feasible_query_count"] == 0
    assert queue == [
        {
            "query_id": "q1",
            "base_candidate_count": 0,
            "base_gold_hit_count": 0,
            "base_hard_negative_candidate_count": 0,
            "pasa_available_gold_count": 1,
            "eligible_signals": ["year"],
            "reasons": ["missing_gold_positive", "missing_hard_negative", "shallow_candidate_pool"],
            "recommended_action": "targeted_pasa_lexical_negative_then_gold_injection",
        }
    ]
