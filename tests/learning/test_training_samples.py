from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.data_isolation import DatasetExample, DatasetPartition
from paper_search.learning.training_samples import (
    ActionObservation,
    build_action_training_examples,
)


def _training_partition(role: str = "training") -> DatasetPartition:
    return DatasetPartition(
        dataset="pasa",
        split="auto_train" if role == "training" else "auto_test",
        role=role,
        revision="rev-1",
        examples=[
            DatasetExample(
                query_id="q1",
                query="graph retrieval papers",
                gold_paper_ids=["paper:gold"],
            )
        ],
    )


def _action(action_id: str, text: str) -> PolicyActionCandidate:
    return PolicyActionCandidate(
        action_id=action_id,
        action_type="text_search",
        text=text,
        origin="deterministic_rule",
        provider_hint="either",
    )


def test_builder_derives_positive_hard_and_empty_labels_without_gold_ids() -> None:
    examples = build_action_training_examples(
        _training_partition(),
        [
            ActionObservation(
                query_id="q1",
                action=_action("a1", "graph retrieval"),
                retrieved_paper_ids=["paper:gold", "paper:x"],
            ),
            ActionObservation(
                query_id="q1",
                action=_action("a2", "neural retrieval"),
                retrieved_paper_ids=["paper:x"],
            ),
            ActionObservation(
                query_id="q1",
                action=_action("a3", "unmatched phrase"),
                retrieved_paper_ids=[],
            ),
        ],
    )

    assert [example.label for example in examples] == [
        "positive",
        "hard_negative",
        "empty_negative",
    ]
    assert [example.gold_hit_count for example in examples] == [1, 0, 0]
    serialized = json.dumps(
        [example.model_dump(mode="json") for example in examples],
        sort_keys=True,
    )
    assert "paper:gold" not in serialized
    assert "paper:x" not in serialized


def test_builder_rejects_final_test_gold_for_label_construction() -> None:
    with pytest.raises(ValueError, match="final_test cannot produce training examples"):
        build_action_training_examples(_training_partition("final_test"), [])


def test_action_candidate_rejects_gold_derived_origin() -> None:
    with pytest.raises(ValidationError, match="origin"):
        PolicyActionCandidate.model_validate(
            {
                "action_id": "a1",
                "action_type": "title_search",
                "text": "A gold paper title",
                "origin": "gold_document",
                "provider_hint": "either",
            }
        )


def test_builder_rejects_observation_for_unknown_query() -> None:
    with pytest.raises(ValueError, match="unknown query ID"):
        build_action_training_examples(
            _training_partition(),
            [
                ActionObservation(
                    query_id="missing",
                    action=_action("a1", "graph retrieval"),
                    retrieved_paper_ids=[],
                )
            ],
        )
