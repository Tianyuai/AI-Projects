import pytest
from pydantic import ValidationError

from paper_search.evaluation.dataset import (
    EvaluationQuery,
    PredictionRecord,
    normalize_paper_id,
    normalize_title,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://doi.org/10.1000/ABC", "doi:10.1000/abc"),
        ("doi:10.5555/Example", "doi:10.5555/example"),
        ("10.1234/Bare", "doi:10.1234/bare"),
    ],
)
def test_normalize_doi(raw: str, expected: str) -> None:
    assert normalize_paper_id(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("arXiv:2501.10120v3", "arxiv:2501.10120"),
        ("https://arxiv.org/pdf/1706.03762v5.pdf", "arxiv:1706.03762"),
        ("https://arxiv.org/abs/hep-th/9901001v2", "arxiv:hep-th/9901001"),
    ],
)
def test_normalize_arxiv_id(raw: str, expected: str) -> None:
    assert normalize_paper_id(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://openalex.org/w123", "openalex:W123"),
        ("openalex:W456", "openalex:W456"),
        ("https://www.semanticscholar.org/paper/example/ABC123", "s2:ABC123"),
        ("s2:deadbeef", "s2:deadbeef"),
    ],
)
def test_normalize_provider_id(raw: str, expected: str) -> None:
    assert normalize_paper_id(raw) == expected


def test_normalize_title_uses_nfkc_casefold_punctuation_and_whitespace() -> None:
    assert normalize_title("  Ａ Study:  On RAG!  ") == "a study on rag"
    assert normalize_paper_id("  Ａ Study:  On RAG!  ", kind="title") == (
        "title:a study on rag"
    )
    assert normalize_paper_id("title:Graph-Based Retrieval") == "title:graph based retrieval"


@pytest.mark.parametrize(
    "raw",
    ["", "ordinary untyped title", "openalex:123", "doi:not-a-doi", "title:!!!"],
)
def test_normalize_paper_id_rejects_empty_ambiguous_or_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_paper_id(raw)


def test_evaluation_query_normalizes_ids_and_is_frozen() -> None:
    query = EvaluationQuery(
        query_id=" q1 ",
        query=" RAG evaluation ",
        relevant_paper_ids=["arXiv:2501.10120v2"],
        metadata={"split": "dev"},
    )

    assert query.query_id == "q1"
    assert query.query == "RAG evaluation"
    assert query.relevant_paper_ids == ["arxiv:2501.10120"]
    with pytest.raises(ValidationError):
        query.query = "changed"


def test_evaluation_query_rejects_duplicates_after_normalization() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        EvaluationQuery(
            query_id="q1",
            query="RAG",
            relevant_paper_ids=["arXiv:2501.10120", "2501.10120v2"],
        )


def test_evaluation_query_rejects_unknown_fields_and_non_json_metadata() -> None:
    with pytest.raises(ValidationError):
        EvaluationQuery(query_id="q1", query="RAG", unexpected=True)
    with pytest.raises(ValidationError):
        EvaluationQuery(query_id="q1", query="RAG", metadata={"bad": object()})


def test_prediction_record_normalizes_but_preserves_ranked_duplicates() -> None:
    record = PredictionRecord(
        query_id="q1",
        predicted_paper_ids=["doi:10.1000/A", "https://doi.org/10.1000/a"],
    )

    assert record.predicted_paper_ids == ["doi:10.1000/a", "doi:10.1000/a"]
