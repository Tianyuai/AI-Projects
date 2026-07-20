from __future__ import annotations

import pytest
from pydantic import ValidationError

import paper_search.evaluation.official_adapter as adapter_module
from paper_search.evaluation.official_adapter import PaSaRecord, adapt_pasa_record


def test_pasa_prefers_arxiv_ids_and_copies_provenance() -> None:
    source = PaSaRecord(
        qid="q1",
        question="Find RAG evaluations",
        answer=["Paper A", "Paper B"],
        answer_arxiv_id=["2501.10120v2", "1706.03762"],
        source_meta={"published_time": "2025-01-01"},
    )

    result = adapt_pasa_record(
        source,
        source="AutoScholarQuery",
        split="dev",
        revision="abc",
    )

    assert result.query_id == "q1"
    assert result.query == "Find RAG evaluations"
    assert result.relevant_paper_ids == ["arxiv:2501.10120", "arxiv:1706.03762"]
    assert result.metadata == {
        "dataset_revision": "abc",
        "source": "AutoScholarQuery",
        "source_meta": {"published_time": "2025-01-01"},
        "split": "dev",
    }


def test_pasa_uses_title_only_when_corresponding_arxiv_id_is_missing() -> None:
    source = PaSaRecord(
        qid="q1",
        question="Find papers",
        answer=["Paper A", "Paper B"],
        answer_arxiv_id=["2501.10120", ""],
        source_meta={},
    )

    result = adapt_pasa_record(
        source,
        source="RealScholarQuery",
        split="test",
        revision="abc",
    )

    assert result.relevant_paper_ids == ["arxiv:2501.10120", "title:paper b"]


def test_pasa_deduplicates_canonical_answer_ids_in_source_order() -> None:
    source = PaSaRecord(
        qid="q1",
        question="Find papers",
        answer=["Paper A v1", "Paper A v2", "Paper B"],
        answer_arxiv_id=["2501.10120v1", "2501.10120v2", "1706.03762"],
        source_meta={},
    )

    result = adapt_pasa_record(
        source,
        source="AutoScholarQuery",
        split="dev",
        revision="abc",
    )

    assert result.relevant_paper_ids == ["arxiv:2501.10120", "arxiv:1706.03762"]


def test_pasa_coerces_single_answer_fields_to_lists() -> None:
    source = PaSaRecord.model_validate(
        {
            "qid": "q1",
            "question": "Find one paper",
            "answer": "Paper A",
            "answer_arxiv_id": "2501.10120v3",
            "source_meta": {},
        }
    )

    assert source.answer == ["Paper A"]
    assert source.answer_arxiv_id == ["2501.10120v3"]


def test_pasa_rejects_unequal_non_empty_answer_lists() -> None:
    with pytest.raises(ValidationError, match="equal lengths"):
        PaSaRecord(
            qid="q1",
            question="Find papers",
            answer=["Paper A", "Paper B"],
            answer_arxiv_id=["2501.10120"],
            source_meta={},
        )


def test_pasa_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PaSaRecord.model_validate(
            {
                "qid": "q1",
                "question": "Find papers",
                "answer": [],
                "answer_arxiv_id": [],
                "source_meta": {},
                "unexpected": True,
            }
        )


def test_fixed_prediction_schema_maps_selected_ids_without_deduplication() -> None:
    source = adapter_module.InternalPredictionRecord(
        query_id="q1",
        selected_paper_ids=["arxiv:2501.10120v2", "arxiv:2501.10120"],
    )

    result = adapter_module.adapt_prediction_record(source)

    assert result.query_id == "q1"
    assert result.predicted_paper_ids == ["arxiv:2501.10120", "arxiv:2501.10120"]


def test_prediction_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        adapter_module.InternalPredictionRecord.model_validate(
            {"query_id": "q1", "selected_paper_ids": [], "scores": []}
        )
