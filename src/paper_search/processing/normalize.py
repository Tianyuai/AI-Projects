"""Pure mappings from provider records to canonical domain models."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import normalize_paper_id


def reconstruct_abstract(index: object) -> str | None:
    """Rebuild an abstract from OpenAlex's token-to-position representation."""
    if index is None:
        return None
    if not isinstance(index, Mapping):
        raise ValueError("abstract_inverted_index must be a mapping or null")

    positioned: dict[int, str] = {}
    for token, positions in index.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            raise ValueError("invalid abstract inverted index")
        for position in positions:
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError("abstract positions must be non-negative integers")
            if position in positioned:
                raise ValueError("abstract positions must be unique")
            positioned[position] = token
    return " ".join(positioned[position] for position in sorted(positioned)) or None


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer or null")
    return value


def _optional_publication_year(value: object) -> int | None:
    """Keep malformed provider years from invalidating an otherwise usable work."""
    year = _optional_int(value, "publication_year")
    if year is None:
        return None
    maximum_year = date.today().year + 1
    return year if 1900 <= year <= maximum_year else None


def _optional_bool(value: object, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean or null")
    return value


def _extract_authors(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("authorships must be a list")
    authors: list[str] = []
    for authorship in value:
        if not isinstance(authorship, Mapping):
            raise ValueError("each authorship must be a mapping")
        author = authorship.get("author")
        if not isinstance(author, Mapping):
            continue
        name = author.get("display_name")
        if isinstance(name, str) and name.strip():
            authors.append(name.strip())
    return authors


def _extract_location(value: object, fallback_url: str) -> tuple[str | None, str]:
    if value is None:
        return None, fallback_url
    if not isinstance(value, Mapping):
        raise ValueError("primary_location must be a mapping or null")

    venue: str | None = None
    source = value.get("source")
    if isinstance(source, Mapping):
        display_name = source.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            venue = display_name.strip()

    landing_page = value.get("landing_page_url")
    url = landing_page.strip() if isinstance(landing_page, str) and landing_page.strip() else fallback_url
    return venue, url


def normalize_openalex_work(raw_work: Mapping[str, object]) -> Paper:
    """Convert one OpenAlex Work object to the project's canonical Paper model."""
    title_value = raw_work.get("title") or raw_work.get("display_name")
    if not isinstance(title_value, str) or not title_value.strip():
        raise ValueError("OpenAlex work must have a title")

    openalex_value = raw_work.get("id")
    if not isinstance(openalex_value, str) or not openalex_value.strip():
        raise ValueError("OpenAlex work must have an id")
    try:
        openalex_canonical = normalize_paper_id(openalex_value, kind="openalex")
    except ValueError as error:
        raise ValueError("OpenAlex work must have a valid id") from error
    openalex_id = openalex_canonical.removeprefix("openalex:")

    doi: str | None = None
    doi_value = raw_work.get("doi")
    if isinstance(doi_value, str) and doi_value.strip():
        try:
            doi = normalize_paper_id(doi_value, kind="doi").removeprefix("doi:")
        except ValueError:
            doi = None

    canonical_id = f"doi:{doi}" if doi is not None else openalex_canonical
    venue, url = _extract_location(raw_work.get("primary_location"), openalex_value)

    return Paper(
        canonical_id=canonical_id,
        title=title_value,
        abstract=reconstruct_abstract(raw_work.get("abstract_inverted_index")),
        authors=_extract_authors(raw_work.get("authorships")),
        publication_year=_optional_publication_year(raw_work.get("publication_year")),
        venue=venue,
        doi=doi,
        openalex_id=openalex_id,
        url=url,
        citation_count=_optional_int(raw_work.get("cited_by_count"), "cited_by_count"),
        is_retracted=_optional_bool(raw_work.get("is_retracted"), "is_retracted"),
        sources=["openalex"],
    )
