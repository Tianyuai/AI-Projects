from __future__ import annotations

import pytest

from paper_search.learning.data_isolation import (
    DatasetExample,
    DatasetPartition,
    DatasetRoleRegistry,
    assert_training_safe,
    audit_dataset_isolation,
)


def _partition(
    dataset: str,
    split: str,
    role: str,
    *examples: DatasetExample,
) -> DatasetPartition:
    return DatasetPartition(
        dataset=dataset,
        split=split,
        role=role,
        revision="rev-1",
        examples=list(examples),
    )


def _example(
    query_id: str,
    query: str,
    *,
    gold: list[str],
    source_components: list[str] | None = None,
) -> DatasetExample:
    return DatasetExample(
        query_id=query_id,
        query=query,
        gold_paper_ids=gold,
        source_components=source_components or [],
    )


def test_registry_keeps_dataset_partition_counts_separate() -> None:
    registry = DatasetRoleRegistry(
        partitions=[
            _partition(
                "pasa",
                "auto_train",
                "training",
                _example("p1", "graph retrieval", gold=["paper:1"]),
            ),
            _partition(
                "asta",
                "paper_finder_validation",
                "development",
                _example("a1", "find graph retrieval papers", gold=[]),
            ),
        ]
    )

    assert registry.partition_counts() == {
        "asta/paper_finder_validation": 1,
        "pasa/auto_train": 1,
    }


def test_audit_rejects_cross_role_query_gold_and_source_component_leakage() -> None:
    training = _partition(
        "pasa",
        "auto_train",
        "training",
        _example(
            "p1",
            "Graph retrieval for scientific papers",
            gold=["paper:1"],
            source_components=["pasa:auto:42"],
        ),
    )
    evaluation = _partition(
        "asta",
        "paper_finder_validation",
        "development",
        _example(
            "a1",
            "graph retrieval for scientific papers",
            gold=["paper:1"],
            source_components=["pasa:auto:42"],
        ),
    )

    report = audit_dataset_isolation(DatasetRoleRegistry(partitions=[training, evaluation]))

    assert report.safe_for_training is False
    assert {issue.code for issue in report.issues} == {
        "exact_query_overlap",
        "gold_paper_overlap",
        "source_component_overlap",
    }
    with pytest.raises(ValueError, match="dataset isolation audit failed"):
        assert_training_safe(report)


def test_audit_detects_near_duplicate_queries_without_folding_partition_counts() -> None:
    registry = DatasetRoleRegistry(
        partitions=[
            _partition(
                "pasa",
                "auto_train",
                "training",
                _example(
                    "p1",
                    "papers about robust graph neural network retrieval",
                    gold=["paper:1"],
                ),
            ),
            _partition(
                "asta",
                "paper_finder_validation",
                "development",
                _example(
                    "a1",
                    "robust graph neural network retrieval papers",
                    gold=["paper:2"],
                ),
            ),
        ]
    )

    report = audit_dataset_isolation(registry, near_duplicate_threshold=0.8)

    assert [issue.code for issue in report.issues] == ["near_duplicate_query"]
    assert report.partition_counts == {
        "asta/paper_finder_validation": 1,
        "pasa/auto_train": 1,
    }


def test_official_test_partition_cannot_be_assigned_a_training_role() -> None:
    with pytest.raises(ValueError, match="official test partitions must be final_test"):
        DatasetPartition(
            dataset="asta",
            split="paper_finder_test",
            role="training",
            revision="v1.0.0",
            examples=[],
        )
