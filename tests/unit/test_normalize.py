from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paper_search.processing.normalize import (
    normalize_openalex_work,
    reconstruct_abstract,
)


FIXTURE_ROOT = Path("tests/fixtures/openalex")


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def complete_work() -> dict[str, Any]:
    return dict(load_fixture("works_page_1.json")["results"][0])


def test_normalize_openalex_work_maps_complete_record() -> None:
    raw = complete_work()
    raw["locations"] = [
        {
            "landing_page_url": "https://arxiv.org/abs/2501.10120v2",
            "pdf_url": "https://arxiv.org/pdf/2501.10120",
        }
    ]

    paper = normalize_openalex_work(raw)

    assert paper.canonical_id == "doi:10.1000/example"
    assert paper.openalex_id == "W123"
    assert paper.doi == "10.1000/example"
    assert paper.arxiv_id == "2501.10120"
    assert paper.abstract == "retrieval augmented generation"
    assert paper.authors == ["Ada Lovelace", "Grace Hopper"]
    assert paper.publication_year == 2023
    assert paper.venue == "Journal of Safe Fixtures"
    assert paper.url == "https://example.org/paper"
    assert paper.citation_count == 12
    assert paper.is_retracted is False
    assert paper.sources == ["openalex"]


def test_missing_abstract_is_valid_and_openalex_id_is_fallback() -> None:
    raw = load_fixture("works_missing_abstract.json")["results"][0]

    paper = normalize_openalex_work(raw)

    assert paper.abstract is None
    assert paper.canonical_id == "openalex:W125"
    assert paper.url == "https://openalex.org/W125"


def test_arxiv_datacite_doi_overrides_conflicting_location_identifier() -> None:
    raw = complete_work()
    raw["doi"] = "https://doi.org/10.48550/arxiv.2309.17453"
    raw["locations"] = [
        {"landing_page_url": "https://arxiv.org/abs/2602.06317"}
    ]

    paper = normalize_openalex_work(raw)

    assert paper.canonical_id == "doi:10.48550/arxiv.2309.17453"
    assert paper.arxiv_id == "2309.17453"


def test_top_level_openalex_ids_supply_arxiv_alias() -> None:
    raw = complete_work()
    raw["ids"] = {
        "openalex": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1000/example",
        "arxiv": "https://arxiv.org/abs/2401.01234v3",
    }

    paper = normalize_openalex_work(raw)

    assert paper.canonical_id == "doi:10.1000/example"
    assert paper.arxiv_id == "2401.01234"


def test_reconstruct_abstract_orders_all_positions() -> None:
    assert reconstruct_abstract({"later": [2], "first": [0], "middle": [1]}) == (
        "first middle later"
    )


@pytest.mark.parametrize(
    "index",
    [
        {"token": [-1]},
        {"token": [True]},
        {"first": [0], "duplicate": [0]},
        {"token": "not-a-list"},
    ],
)
def test_reconstruct_abstract_rejects_invalid_positions(index: object) -> None:
    with pytest.raises(ValueError):
        reconstruct_abstract(index)


def test_record_without_title_is_rejected() -> None:
    raw = complete_work()
    raw["title"] = None
    raw["display_name"] = None

    with pytest.raises(ValueError, match="title"):
        normalize_openalex_work(raw)


def test_record_without_stable_id_is_rejected() -> None:
    raw = complete_work()
    raw["id"] = None
    raw["doi"] = None

    with pytest.raises(ValueError, match="id"):
        normalize_openalex_work(raw)


@pytest.mark.parametrize("field", ["publication_year", "cited_by_count"])
def test_numeric_fields_reject_booleans(field: str) -> None:
    raw = complete_work()
    raw[field] = True

    with pytest.raises(ValueError, match=field):
        normalize_openalex_work(raw)
