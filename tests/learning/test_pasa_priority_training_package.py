from paper_search.domain.models import Paper
from paper_search.learning.pasa_priority_training_package import (
    build_unified_context_freeze_rows,
    build_mixed_pasa_candidates,
    merge_local_constraint_row,
    validate_priority_queue_rows,
)
from paper_search.learning.query_constraint_annotations import (
    FrozenConstraintAnnotation,
)
from paper_search.learning.query_constraint_profile import QueryConstraintProfile
from paper_search.retrieval.pasa_paper_database import (
    PASA_TRAINING_GOLD_INJECTED_SOURCE,
)


def _paper(arxiv_id: str, title: str) -> Paper:
    return Paper(
        canonical_id=f"arxiv:{arxiv_id}",
        arxiv_id=arxiv_id,
        title=title,
    )


def test_build_mixed_candidates_shares_one_source_between_gold_and_negatives() -> None:
    lexical_gold = _paper("1", "Gold found lexically")
    lexical_negative = _paper("9", "Hard lexical negative")
    appended_gold = _paper("2", "Gold appended from the local index")

    papers, audit = build_mixed_pasa_candidates(
        lexical_papers=[lexical_gold, lexical_negative],
        gold_paper_ids=["arxiv:1", "arxiv:2"],
        gold_lookup={"arxiv:1": lexical_gold, "arxiv:2": appended_gold},
    )

    assert [paper.arxiv_id for paper in papers] == ["1", "9", "2"]
    assert PASA_TRAINING_GOLD_INJECTED_SOURCE not in papers[0].sources
    assert PASA_TRAINING_GOLD_INJECTED_SOURCE in papers[2].sources
    assert audit == {
        "lexical_candidate_count": 2,
        "lexical_gold_candidate_count": 1,
        "lexical_negative_candidate_count": 1,
        "direct_gold_candidate_count": 1,
        "positive_candidate_count": 2,
        "supplement_candidate_count": 3,
        "mixed_positive_negative": True,
    }


def test_build_mixed_candidates_rejects_gold_only_source() -> None:
    gold = _paper("1", "Only Gold")

    try:
        build_mixed_pasa_candidates(
            lexical_papers=[],
            gold_paper_ids=["arxiv:1"],
            gold_lookup={"arxiv:1": gold},
        )
    except ValueError as error:
        assert str(error) == "mixed PASA candidates require both Gold and lexical negatives"
    else:
        raise AssertionError("gold-only PASA candidates must be rejected")


def test_merge_local_constraint_row_freezes_deterministic_dataset_and_negation() -> None:
    base = {
        "query_id": "q1",
        "query_sha256": "sha256:" + "1" * 64,
        "role": "training",
        "split": "auto_train",
        "labels": [],
        "methods": [],
        "datasets": [],
        "tasks": [],
        "year_from": None,
        "year_to": None,
        "exclusions": [],
        "label_sources": {},
        "label_confidence": {},
        "evidence": {},
        "status": "partial",
    }
    profile = QueryConstraintProfile(
        labels=["dataset", "negation", "year"],
        datasets=["REDS"],
        exclusions=["synthetic data"],
        year_from=2020,
        has_negation=True,
        constraint_count=3,
        confidence=1.0,
    )

    merged = merge_local_constraint_row(base, profile=profile)
    validated = FrozenConstraintAnnotation.model_validate(merged)

    assert validated.labels == ["dataset", "negation", "year"]
    assert validated.datasets == ["REDS"]
    assert validated.exclusions == ["synthetic data"]
    assert validated.year_from == 2020
    assert validated.label_sources == {
        "dataset": "local_deterministic",
        "negation": "local_deterministic",
        "year": "local_deterministic",
    }
    assert validated.status == "accepted"


def test_context_freeze_covers_exact_strict_ready_scope() -> None:
    task_rows = {
        "q1": {
            "query_id": "q1",
            "query_sha256": "sha256:" + "1" * 64,
            "role": "training",
            "split": "auto_train",
            "tasks": [],
            "ambiguous_fields": [],
            "task_label_status": "unresolved",
        },
        "q2": {
            "query_id": "q2",
            "query_sha256": "sha256:" + "2" * 64,
            "role": "training",
            "split": "auto_train",
            "tasks": [],
            "ambiguous_fields": [],
            "task_label_status": "unresolved",
        },
    }
    base_constraints = {
        query_id: {
            "query_id": query_id,
            "query_sha256": row["query_sha256"],
            "role": "training",
            "split": "auto_train",
            "labels": [],
            "methods": [],
            "datasets": [],
            "tasks": [],
            "year_from": None,
            "year_to": None,
            "exclusions": [],
            "label_sources": {},
            "label_confidence": {},
            "evidence": {},
            "status": "partial",
        }
        for query_id, row in task_rows.items()
    }
    profiles = {
        "q1": QueryConstraintProfile(
            labels=["dataset", "task"],
            datasets=["REDS"],
            tasks=["image classification"],
            constraint_count=2,
            confidence=1.0,
        ),
        "q2": QueryConstraintProfile(
            labels=[],
            constraint_count=0,
            confidence=0.0,
        ),
    }

    frozen_tasks, frozen_constraints, summary = build_unified_context_freeze_rows(
        strict_ready_query_ids=["q2", "q1"],
        partition_rows={
            "q1": {"role": "training", "split": "auto_train"},
            "q2": {"role": "training", "split": "auto_train"},
        },
        task_rows_by_query=task_rows,
        constraint_rows_by_query=base_constraints,
        local_profiles_by_query=profiles,
    )

    assert [row["query_id"] for row in frozen_tasks] == ["q1", "q2"]
    assert frozen_tasks[0]["tasks"] == [
        {
            "normalized_value": "image classification",
            "confidence": 1.0,
            "evidence_span": "image classification",
            "strength": "must",
        }
    ]
    assert frozen_tasks[0]["task_label_status"] == "runtime_deterministic"
    assert [row["query_id"] for row in frozen_constraints] == ["q1", "q2"]
    assert frozen_constraints[0]["labels"] == ["dataset", "task"]
    assert summary == {
        "query_count": 2,
        "label_query_count": {"dataset": 1, "task": 1},
        "role": "training",
        "split": "auto_train",
        "test_partition_touched": False,
    }


def test_priority_queue_accepts_only_dataset_or_task_mixed_targets() -> None:
    rows = validate_priority_queue_rows(
        [
            {
                "query_id": "q2",
                "recommended_action": "pasa_mixed_lexical_gold_supplement",
                "eligible_signals": ["task_provenance"],
            },
            {
                "query_id": "q1",
                "recommended_action": "pasa_mixed_lexical_gold_supplement",
                "eligible_signals": ["dataset"],
            },
        ],
        strict_ready_query_ids={"q1", "q2"},
        expected_count=2,
    )

    assert [row["query_id"] for row in rows] == ["q1", "q2"]

    try:
        validate_priority_queue_rows(
            [
                {
                    "query_id": "q1",
                    "recommended_action": "pasa_mixed_lexical_gold_supplement",
                    "eligible_signals": ["negation"],
                }
            ],
            strict_ready_query_ids={"q1"},
            expected_count=1,
        )
    except ValueError as error:
        assert "dataset or task_provenance" in str(error)
    else:
        raise AssertionError("non-priority gate target must be rejected")


def test_merge_local_constraints_never_overwrites_confirmed_values() -> None:
    base = {
        "query_id": "q1",
        "query_sha256": "sha256:" + "1" * 64,
        "role": "training",
        "split": "auto_train",
        "labels": ["dataset", "year"],
        "methods": [],
        "datasets": ["confirmed benchmark"],
        "tasks": [],
        "year_from": None,
        "year_to": 2019,
        "exclusions": [],
        "label_sources": {"dataset": "model", "year": "model"},
        "label_confidence": {"dataset": 0.9, "year": 1.0},
        "evidence": {"dataset": ["confirmed benchmark"], "year": ["2019"]},
        "status": "accepted",
    }
    profile = QueryConstraintProfile(
        labels=["dataset", "year"],
        datasets=["local benchmark"],
        year_from=2020,
        constraint_count=2,
        confidence=1.0,
    )

    merged = merge_local_constraint_row(base, profile=profile)

    assert merged["datasets"] == ["confirmed benchmark", "local benchmark"]
    assert merged["year_from"] is None
    assert merged["year_to"] == 2019
    assert merged["label_sources"]["dataset"] == "model"
    assert merged["label_sources"]["year"] == "model"


def test_merge_local_constraints_refreshes_deterministic_entity_years() -> None:
    base = {
        "query_id": "q1",
        "query_sha256": "sha256:" + "1" * 64,
        "role": "training",
        "split": "auto_train",
        "labels": ["dataset", "year"],
        "methods": [],
        "datasets": ["which"],
        "tasks": [],
        "year_from": 2003,
        "year_to": 2003,
        "exclusions": [],
        "label_sources": {
            "dataset": "local_deterministic",
            "year": "rule",
        },
        "label_confidence": {"dataset": 1.0, "year": 1.0},
        "evidence": {"dataset": ["which"], "year": ["2003"]},
        "status": "accepted",
    }
    profile = QueryConstraintProfile(
        labels=["dataset"],
        datasets=["conll 2003"],
        constraint_count=1,
        confidence=1.0,
    )

    merged = merge_local_constraint_row(base, profile=profile)

    assert merged["labels"] == ["dataset"]
    assert merged["datasets"] == ["conll 2003"]
    assert merged["year_from"] is None
    assert merged["year_to"] is None
    assert "year" not in merged["label_sources"]
    assert "year" not in merged["label_confidence"]
    assert "year" not in merged["evidence"]


def test_merge_local_constraints_refreshes_rule_year_direction() -> None:
    base = {
        "query_id": "q1",
        "query_sha256": "sha256:" + "1" * 64,
        "role": "training",
        "split": "auto_train",
        "labels": ["year"],
        "methods": [],
        "datasets": [],
        "tasks": [],
        "year_from": 2020,
        "year_to": 2020,
        "exclusions": [],
        "label_sources": {"year": "rule"},
        "label_confidence": {"year": 1.0},
        "evidence": {"year": ["2020"]},
        "status": "accepted",
    }
    profile = QueryConstraintProfile(
        labels=["year"],
        year_to=2020,
        constraint_count=1,
        confidence=1.0,
    )

    merged = merge_local_constraint_row(base, profile=profile)

    assert merged["year_from"] is None
    assert merged["year_to"] == 2020
    assert merged["evidence"]["year"] == ["2020"]
