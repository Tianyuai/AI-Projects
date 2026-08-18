"""Gold-labelled action-ranking examples with inference-safe action payloads."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, StrictNonNegativeInt
from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.data_isolation import DatasetPartition


ActionLabel = Literal["positive", "hard_negative", "empty_negative"]


class ActionObservation(DomainModel):
    query_id: NonEmptyStr
    action: PolicyActionCandidate
    retrieved_paper_ids: list[NonEmptyStr] = Field(default_factory=list)


class ActionTrainingExample(DomainModel):
    dataset: NonEmptyStr
    split: NonEmptyStr
    query_id: NonEmptyStr
    query: NonEmptyStr
    action: PolicyActionCandidate
    label: ActionLabel
    retrieved_count: StrictNonNegativeInt
    gold_hit_count: StrictNonNegativeInt


def build_action_training_examples(
    partition: DatasetPartition,
    observations: list[ActionObservation],
) -> list[ActionTrainingExample]:
    """Label bounded actions; never serialize Gold or retrieved paper identifiers."""
    partition = DatasetPartition.model_validate(partition)
    if partition.role == "final_test":
        raise ValueError("final_test cannot produce training examples")
    by_query_id = {example.query_id: example for example in partition.examples}
    result: list[ActionTrainingExample] = []
    for raw_observation in observations:
        observation = ActionObservation.model_validate(raw_observation)
        dataset_example = by_query_id.get(observation.query_id)
        if dataset_example is None:
            raise ValueError(f"unknown query ID: {observation.query_id}")
        gold_ids = set(dataset_example.gold_paper_ids)
        retrieved_ids = set(observation.retrieved_paper_ids)
        hit_count = len(gold_ids.intersection(retrieved_ids))
        label: ActionLabel
        if hit_count:
            label = "positive"
        elif retrieved_ids:
            label = "hard_negative"
        else:
            label = "empty_negative"
        result.append(
            ActionTrainingExample(
                dataset=partition.dataset,
                split=partition.split,
                query_id=dataset_example.query_id,
                query=dataset_example.query,
                action=observation.action,
                label=label,
                retrieved_count=len(retrieved_ids),
                gold_hit_count=hit_count,
            )
        )
    return result


__all__ = [
    "ActionLabel",
    "ActionObservation",
    "ActionTrainingExample",
    "build_action_training_examples",
]
