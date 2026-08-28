from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import pytest

from paper_search.domain.models import Paper, QuerySpec
from paper_search.evaluation.dataset import IdentifierMap, normalize_title
from paper_search.processing import apply_hard_filters
from paper_search.processing.deduplicate import deduplicate_papers


def _paper(
    canonical_id: str,
    *,
    title: str = "Paper",
    abstract: str | None = None,
    authors: list[str] | None = None,
    publication_year: int | None = None,
    venue: str | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    openalex_id: str | None = None,
    semantic_scholar_id: str | None = None,
    url: str | None = None,
    citation_count: int | None = None,
    is_retracted: bool | None = None,
    sources: list[str] | None = None,
) -> Paper:
    return Paper(
        canonical_id=canonical_id,
        title=title,
        abstract=abstract,
        authors=[] if authors is None else authors,
        publication_year=publication_year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        openalex_id=openalex_id,
        semantic_scholar_id=semantic_scholar_id,
        url=url,
        citation_count=citation_count,
        is_retracted=is_retracted,
        sources=[] if sources is None else sources,
    )


def test_same_doi_merges_and_preserves_first_cluster_position() -> None:
    papers = [
        _paper("openalex:W1", title="First", doi="10.1000/a", sources=["openalex"]),
        _paper(
            "s2:S1",
            title="Richer",
            doi="https://doi.org/10.1000/A",
            abstract="body",
            sources=["semantic_scholar"],
        ),
    ]

    result = deduplicate_papers(papers)

    assert len(result.papers) == 1
    assert result.papers[0].doi == "10.1000/a"
    assert result.papers[0].sources == ["semantic_scholar", "openalex"]
    assert result.decisions[0].match_rule == "doi"
    assert result.decisions[0].match_value == "doi:10.1000/a"
    assert result.decisions[0].member_ids == ["openalex:W1", "s2:S1"]


def test_identifier_map_merges_cross_source_aliases(tmp_path: Path) -> None:
    path = tmp_path / "id-map.json"
    path.write_text('{"openalex:W2":"s2:S2"}', encoding="utf-8")
    result = deduplicate_papers(
        [_paper("openalex:W2"), _paper("s2:S2")],
        id_map=IdentifierMap.from_path(path),
    )
    assert len(result.papers) == 1
    assert result.decisions[0].match_rule == "external_id"
    assert result.decisions[0].match_value == "s2:S2"


def test_shared_arxiv_identifier_merges_different_canonical_records() -> None:
    result = deduplicate_papers(
        [
            _paper(
                "doi:10.1145/example",
                title="Published conference version",
                doi="10.1145/example",
                arxiv_id="2301.01234v2",
            ),
            _paper(
                "openalex:W99",
                title="Preprint version with a different title",
                arxiv_id="https://arxiv.org/abs/2301.01234",
                openalex_id="W99",
            ),
        ]
    )

    assert len(result.papers) == 1
    assert result.decisions[0].match_rule == "external_id"
    assert result.decisions[0].match_value == "arxiv:2301.01234"


def test_normalized_exact_title_merges() -> None:
    result = deduplicate_papers(
        [
            _paper("openalex:W3", title="Graph-Based Retrieval"),
            _paper("s2:S3", title="graph based retrieval"),
        ]
    )
    assert result.decisions[0].match_rule == "exact_title"
    assert result.decisions[0].match_value == "graph based retrieval"


def test_exact_title_merge_preserves_arxiv_identity_from_nonrepresentative() -> None:
    result = deduplicate_papers(
        [
            _paper(
                "doi:10.1093/biomet/asaa062",
                title="Randomized numerical linear algebra",
                doi="10.1093/biomet/asaa062",
                abstract="richer published record",
            ),
            _paper(
                "doi:10.48550/arxiv.1411.4357",
                title="Randomized numerical linear algebra",
                doi="10.48550/arxiv.1411.4357",
                arxiv_id="1411.4357",
            ),
        ]
    )

    assert len(result.papers) == 1
    assert result.papers[0].canonical_id == "doi:10.1093/biomet/asaa062"
    assert result.papers[0].arxiv_id == "1411.4357"


def test_fuzzy_title_requires_same_year_and_author_surname() -> None:
    left = _paper(
        "openalex:W4",
        title="Neural Paper Retrieval for Science",
        authors=["Ada Lovelace"],
        publication_year=2024,
    )
    close = _paper(
        "s2:S4",
        title="Neural Paper Retrieval in Science",
        authors=["A. Lovelace"],
        publication_year=2024,
    )
    wrong_year = _paper(
        "openalex:W5",
        title=close.title,
        authors=close.authors,
        publication_year=2023,
    )

    accepted = deduplicate_papers([left, close], fuzzy_title_threshold=0.80)
    assert len(accepted.papers) == 1
    assert accepted.decisions[0].match_value == (
        f"{SequenceMatcher(None, normalize_title(left.title), normalize_title(close.title)).ratio():.6f}"
    )
    assert len(deduplicate_papers([left, wrong_year], fuzzy_title_threshold=0.80).papers) == 2


def test_fuzzy_title_does_not_merge_when_author_is_missing() -> None:
    left = _paper(
        "openalex:W6",
        title="A Study of Search",
        authors=[],
        publication_year=2024,
    )
    right = _paper(
        "s2:S6",
        title="Study of Search",
        authors=["Ada Lovelace"],
        publication_year=2024,
    )
    assert len(deduplicate_papers([left, right], fuzzy_title_threshold=0.70).papers) == 2


@pytest.mark.parametrize(
    "threshold",
    [True, float("nan"), float("inf"), -0.01, 1.01],
)
def test_fuzzy_title_threshold_rejects_invalid_values(threshold: float) -> None:
    with pytest.raises(ValueError, match="fuzzy_title_threshold"):
        deduplicate_papers([], fuzzy_title_threshold=threshold)


def test_transitive_cluster_uses_highest_priority_match_rule() -> None:
    result = deduplicate_papers(
        [
            _paper("openalex:W10", title="First edge", doi="10.1000/transitive"),
            _paper("s2:S10", title="Shared Title", doi="doi:10.1000/TRANSITIVE"),
            _paper("openalex:W11", title="shared-title"),
        ]
    )

    assert len(result.papers) == 1
    assert result.decisions[0].member_ids == ["openalex:W10", "s2:S10", "openalex:W11"]
    assert result.decisions[0].match_rule == "doi"


def test_representative_ranking_scalar_fallback_and_ordered_unions() -> None:
    result = deduplicate_papers(
        [
            _paper(
                "openalex:W20",
                title="Canonical Match",
                abstract="fallback abstract",
                authors=["Ada Lovelace"],
                publication_year=2024,
                venue="Fallback Venue",
                openalex_id="W20",
                url="https://example.test/fallback",
                citation_count=12,
                is_retracted=False,
                sources=["openalex", "shared"],
            ),
            _paper(
                "s2:S20",
                title="canonical-match",
                doi="https://doi.org/10.1000/RICH",
                semantic_scholar_id="S20",
                authors=["Grace Hopper"],
                sources=["semantic_scholar", "shared"],
            ),
        ]
    )

    paper = result.papers[0]
    assert result.decisions[0].representative_id == "s2:S20"
    assert paper.canonical_id == "s2:S20"
    assert paper.doi == "10.1000/rich"
    assert paper.abstract == "fallback abstract"
    assert paper.publication_year == 2024
    assert paper.venue == "Fallback Venue"
    assert paper.openalex_id == "W20"
    assert paper.semantic_scholar_id == "S20"
    assert paper.url == "https://example.test/fallback"
    assert paper.citation_count == 12
    assert paper.is_retracted is False
    assert paper.authors == ["Grace Hopper", "Ada Lovelace"]
    assert paper.sources == ["semantic_scholar", "shared", "openalex"]


def test_representative_identifiers_win_conflicts() -> None:
    result = deduplicate_papers(
        [
            _paper(
                "openalex:W30",
                title="Identifier Conflict",
                abstract="richer",
                openalex_id="W30",
                semantic_scholar_id="S30-left",
            ),
            _paper(
                "openalex:W31",
                title="identifier conflict",
                openalex_id="W31",
                semantic_scholar_id="S30-right",
            ),
        ]
    )

    assert result.papers[0].canonical_id == "openalex:W30"
    assert result.papers[0].openalex_id == "W30"
    assert result.papers[0].semantic_scholar_id == "S30-left"


def test_representative_ranking_prefers_abstract_in_isolation() -> None:
    result = deduplicate_papers(
        [
            _paper("openalex:W32", title="Ranking tie"),
            _paper("s2:S32", title="ranking tie", abstract="only richer field"),
        ]
    )

    assert result.decisions[0].representative_id == "s2:S32"


def test_representative_exact_rank_tie_uses_first_member() -> None:
    result = deduplicate_papers(
        [
            _paper("openalex:W33", title="Exact rank tie"),
            _paper("s2:S33", title="exact rank tie"),
        ]
    )

    assert result.decisions[0].representative_id == "openalex:W33"


def test_retracted_duplicate_overrides_richer_false_representative() -> None:
    result = deduplicate_papers(
        [
            _paper(
                "openalex:W50",
                doi="10.1000/retracted",
                abstract="rich record",
                authors=["Ada Lovelace"],
                is_retracted=False,
            ),
            _paper(
                "s2:S50",
                doi="10.1000/retracted",
                is_retracted=True,
            ),
        ]
    )

    merged = result.papers[0]
    decision = result.decisions[0]
    assert merged.canonical_id == "openalex:W50"
    assert merged.is_retracted is True
    assert hasattr(decision, "conflict_fields")
    assert decision.conflict_fields == ["is_retracted"]
    filtered = apply_hard_filters(
        [merged],
        QuerySpec(original_query="graph", research_goal="graph"),
    )
    assert filtered.accepted == []
    assert filtered.rejected[0].reason_code == "retracted"


def test_cluster_output_uses_first_member_position() -> None:
    result = deduplicate_papers(
        [
            _paper("openalex:W40", title="Cluster"),
            _paper("openalex:W41", title="Unique"),
            _paper("s2:S40", title="cluster", abstract="representative"),
        ]
    )

    assert [paper.canonical_id for paper in result.papers] == ["s2:S40", "openalex:W41"]
    assert len(result.decisions) == 1


def test_output_is_deterministic_and_singletons_have_no_decision() -> None:
    papers = [_paper("openalex:W7"), _paper("openalex:W8", title="Unique")]
    first = deduplicate_papers(papers).model_dump(mode="json")
    second = deduplicate_papers(papers).model_dump(mode="json")
    assert first == second
    assert first["decisions"] == []


def test_punctuation_only_title_singleton_is_preserved() -> None:
    result = deduplicate_papers([_paper("openalex:W60", title="---")])

    assert [paper.canonical_id for paper in result.papers] == ["openalex:W60"]
    assert result.decisions == []


def test_punctuation_only_titles_can_merge_by_doi() -> None:
    result = deduplicate_papers(
        [
            _paper("openalex:W61", title="---", doi="10.1000/punctuation"),
            _paper("s2:S61", title="???", doi="10.1000/punctuation"),
        ]
    )

    assert len(result.papers) == 1
    assert result.decisions[0].match_rule == "doi"
    assert result.decisions[0].match_value == "doi:10.1000/punctuation"
