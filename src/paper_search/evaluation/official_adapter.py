from __future__ import annotations

from pydantic import Field, JsonValue, field_validator, model_validator

from paper_search.domain.models import DomainModel, NonEmptyStr
from paper_search.evaluation.dataset import (
    EvaluationQuery,
    PredictionRecord,
    normalize_paper_id,
)


class PaSaRecord(DomainModel):
    """Strict representation of one PaSa dataset record."""

    qid: NonEmptyStr
    question: NonEmptyStr
    answer: list[str] = Field(default_factory=list)
    answer_arxiv_id: list[str] = Field(default_factory=list)
    source_meta: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("answer", "answer_arxiv_id", mode="before")
    @classmethod
    def coerce_single_string_to_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def validate_answer_alignment(self) -> PaSaRecord:
        if (
            self.answer
            and self.answer_arxiv_id
            and len(self.answer) != len(self.answer_arxiv_id)
        ):
            raise ValueError(
                "answer and answer_arxiv_id must have equal lengths when both are non-empty"
            )
        return self


class InternalPredictionRecord(DomainModel):
    """Fixed prediction-file record emitted by the search pipeline."""

    query_id: NonEmptyStr
    selected_paper_ids: list[NonEmptyStr] = Field(default_factory=list)


def adapt_pasa_record(
    record: PaSaRecord,
    *,
    source: str,
    split: str,
    revision: str,
) -> EvaluationQuery:
    """Convert one PaSa source record to the canonical evaluation contract."""
    relevant_paper_ids: list[str] = []
    seen_paper_ids: set[str] = set()
    pair_count = max(len(record.answer), len(record.answer_arxiv_id))

    for index in range(pair_count):
        title = record.answer[index] if index < len(record.answer) else ""
        arxiv_id = (
            record.answer_arxiv_id[index]
            if index < len(record.answer_arxiv_id)
            else ""
        )
        paper_id: str | None = None
        if arxiv_id.strip():
            paper_id = normalize_paper_id(arxiv_id, kind="arxiv")
        elif title.strip():
            paper_id = normalize_paper_id(title, kind="title")

        if paper_id is not None and paper_id not in seen_paper_ids:
            seen_paper_ids.add(paper_id)
            relevant_paper_ids.append(paper_id)

    return EvaluationQuery(
        query_id=record.qid,
        query=record.question,
        relevant_paper_ids=relevant_paper_ids,
        metadata={
            "dataset_revision": revision,
            "source": source,
            "source_meta": dict(record.source_meta),
            "split": split,
        },
    )


def adapt_prediction_record(record: InternalPredictionRecord) -> PredictionRecord:
    """Convert one fixed prediction record to the canonical ranked contract."""
    return PredictionRecord(
        query_id=record.query_id,
        predicted_paper_ids=record.selected_paper_ids,
    )
