from __future__ import annotations

from paper_search.processing.normalize import normalize_openalex_work


def _work_with_future_year() -> dict[str, object]:
    return {
        "id": "https://openalex.org/W999",
        "doi": None,
        "title": "Synthetic normalization fixture",
        "display_name": "Synthetic normalization fixture",
        "abstract_inverted_index": None,
        "authorships": [],
        "publication_year": 2099,
        "primary_location": None,
        "cited_by_count": 0,
        "is_retracted": False,
    }


def test_future_publication_year_is_downgraded_to_unknown() -> None:
    paper = normalize_openalex_work(_work_with_future_year())

    assert paper.publication_year is None
