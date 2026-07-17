from __future__ import annotations

import pytest

from paper_search.domain.models import Paper, QuerySpec
from paper_search.processing.filter import apply_hard_filters


def _paper(
    canonical_id: str = "openalex:W1",
    *,
    title: str = "Paper",
    abstract: str | None = "Abstract",
    publication_year: int | None = 2024,
    venue: str | None = "NeurIPS",
    doi: str | None = None,
    openalex_id: str | None = None,
    semantic_scholar_id: str | None = None,
    is_retracted: bool | None = False,
) -> Paper:
    return Paper(
        canonical_id=canonical_id,
        title=title,
        abstract=abstract,
        publication_year=publication_year,
        venue=venue,
        doi=doi,
        openalex_id=openalex_id,
        semantic_scholar_id=semantic_scholar_id,
        is_retracted=is_retracted,
    )


def _query(
    *,
    year_from: int | None = None,
    year_to: int | None = None,
    venues: list[str] | None = None,
    exclusions: list[str] | None = None,
) -> QuerySpec:
    return QuerySpec(
        original_query="paper retrieval",
        research_goal="find relevant papers",
        year_from=year_from,
        year_to=year_to,
        venues=[] if venues is None else venues,
        exclusions=[] if exclusions is None else exclusions,
    )


@pytest.mark.parametrize(
    ("paper", "query", "code"),
    [
        (_paper(is_retracted=True), _query(), "retracted"),
        (
            _paper(canonical_id="title:unstable", doi=None, openalex_id=None),
            _query(),
            "missing_stable_id",
        ),
        (_paper(publication_year=2019), _query(year_from=2020), "year_out_of_range"),
        (_paper(venue="Other"), _query(venues=["NeurIPS"]), "venue_mismatch"),
        (_paper(title="Survey paper"), _query(exclusions=["survey"]), "excluded_term"),
    ],
)
def test_hard_filter_reason(paper: Paper, query: QuerySpec, code: str) -> None:
    result = apply_hard_filters([paper], query)

    assert result.accepted == []
    assert result.rejected[0].reason_code == code
    assert result.rejected[0].reason


def test_hard_filter_uses_first_matching_rule() -> None:
    paper = _paper(
        canonical_id="title:unstable",
        title="Survey paper",
        publication_year=2019,
        venue="Other",
        is_retracted=True,
    )

    result = apply_hard_filters(
        [paper],
        _query(year_from=2020, venues=["NeurIPS"], exclusions=["survey"]),
    )

    assert result.rejected[0].reason_code == "retracted"


@pytest.mark.parametrize(
    "paper",
    [
        _paper(canonical_id="title:unstable", doi="not-a-doi"),
        _paper(canonical_id="title:unstable", openalex_id="not-openalex"),
        _paper(canonical_id="title:unstable", semantic_scholar_id="bad id"),
        _paper(canonical_id="openalex:not-a-work"),
    ],
)
def test_malformed_nonblank_identifiers_do_not_count_as_stable(paper: Paper) -> None:
    result = apply_hard_filters([paper], _query())

    assert result.accepted == []
    assert result.rejected[0].reason_code == "missing_stable_id"


@pytest.mark.parametrize(
    "paper",
    [
        _paper(canonical_id="title:unstable", doi="10.1000/valid"),
        _paper(canonical_id="title:unstable", openalex_id="W123"),
        _paper(canonical_id="title:unstable", semantic_scholar_id="Corpus_123"),
        _paper(canonical_id="doi:10.1000/canonical"),
        _paper(canonical_id="openalex:W456"),
        _paper(canonical_id="s2:Corpus.456"),
        _paper(canonical_id="arxiv:2401.01234"),
    ],
)
def test_valid_supported_identifiers_count_as_stable(paper: Paper) -> None:
    result = apply_hard_filters([paper], _query())

    assert [item.paper for item in result.accepted] == [paper]
    assert result.rejected == []


def test_missing_constrained_fields_are_downweighted_not_removed() -> None:
    result = apply_hard_filters(
        [
            _paper(
                publication_year=None,
                venue=None,
                abstract=None,
                is_retracted=None,
            )
        ],
        _query(year_from=2020, venues=["NeurIPS"], exclusions=["survey"]),
    )

    accepted = result.accepted[0]
    assert accepted.uncertainty_reasons == [
        "missing_year",
        "missing_venue",
        "unknown_retraction_status",
        "missing_abstract_for_exclusion",
    ]
    assert accepted.score_multiplier == pytest.approx(0.7)


def test_filter_keeps_input_order_with_separate_audit_lists() -> None:
    result = apply_hard_filters(
        [
            _paper("openalex:W1"),
            _paper("openalex:W2", is_retracted=True),
            _paper("openalex:W3"),
            _paper("openalex:W4", publication_year=2019),
        ],
        _query(year_from=2020),
    )

    assert [item.paper.canonical_id for item in result.accepted] == [
        "openalex:W1",
        "openalex:W3",
    ]
    assert [item.paper.canonical_id for item in result.rejected] == [
        "openalex:W2",
        "openalex:W4",
    ]


def test_exclusion_phrase_does_not_match_across_title_and_abstract() -> None:
    result = apply_hard_filters(
        [_paper(title="Paper", abstract="Survey evidence")],
        _query(exclusions=["paper survey"]),
    )

    assert [item.paper.canonical_id for item in result.accepted] == ["openalex:W1"]
    assert result.rejected == []


@pytest.mark.parametrize("venue", ["", "   ", "---"])
def test_ambiguous_venue_is_downweighted_not_removed(venue: str) -> None:
    result = apply_hard_filters(
        [_paper(venue=venue)],
        _query(venues=["NeurIPS"]),
    )

    accepted = result.accepted[0]
    assert result.rejected == []
    assert accepted.uncertainty_reasons == ["missing_venue"]
    assert accepted.score_multiplier == pytest.approx(0.9)


@pytest.mark.parametrize("abstract", ["", "   ", "---"])
def test_ambiguous_abstract_is_downweighted_not_removed(abstract: str) -> None:
    result = apply_hard_filters(
        [_paper(abstract=abstract)],
        _query(exclusions=["survey"]),
    )

    accepted = result.accepted[0]
    assert result.rejected == []
    assert accepted.uncertainty_reasons == ["missing_abstract_for_exclusion"]
    assert accepted.score_multiplier == pytest.approx(0.9)
