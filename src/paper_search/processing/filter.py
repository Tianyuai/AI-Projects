"""Auditable hard filters for normalized paper candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, Paper, QuerySpec
from paper_search.evaluation.dataset import normalize_title


class AcceptedPaper(DomainModel):
    """A paper retained for ranking with any uncertainty penalties."""

    paper: Paper
    uncertainty_reasons: list[NonEmptyStr]
    score_multiplier: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class RejectedPaper(DomainModel):
    """A paper removed by the first matching hard-filter rule."""

    paper: Paper
    reason_code: NonEmptyStr
    reason: NonEmptyStr


class FilterResult(DomainModel):
    """Separate ordered audit lists for retained and removed papers."""

    accepted: list[AcceptedPaper]
    rejected: list[RejectedPaper]


def _has_stable_id(paper: Paper) -> bool:
    direct_ids = (paper.doi, paper.openalex_id, paper.semantic_scholar_id)
    if any(identifier is not None and identifier.strip() for identifier in direct_ids):
        return True
    canonical_id = paper.canonical_id.casefold()
    return canonical_id.startswith(("doi:", "openalex:", "s2:", "arxiv:"))


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return normalize_title(value)
    except ValueError:
        return None


def _rejection_reason(paper: Paper, query: QuerySpec) -> tuple[str, str] | None:
    if paper.is_retracted is True:
        return "retracted", "Paper is marked as retracted."
    if not _has_stable_id(paper):
        return "missing_stable_id", "Paper has no stable scholarly identifier."
    if paper.publication_year is not None and (
        (query.year_from is not None and paper.publication_year < query.year_from)
        or (query.year_to is not None and paper.publication_year > query.year_to)
    ):
        return "year_out_of_range", "Paper publication year is outside the requested range."
    if paper.venue is not None and query.venues:
        normalized_venue = _normalize_optional_text(paper.venue)
        if normalized_venue is not None and all(
            normalized_venue != normalize_title(venue) for venue in query.venues
        ):
            return "venue_mismatch", "Paper venue does not match the requested venues."
    if query.exclusions:
        searchable_fields = [normalize_title(paper.title)]
        normalized_abstract = _normalize_optional_text(paper.abstract)
        if normalized_abstract is not None:
            searchable_fields.append(normalized_abstract)
        if any(
            normalize_title(term) in field
            for term in query.exclusions
            for field in searchable_fields
        ):
            return "excluded_term", "Paper title or abstract contains an excluded term."
    return None


def _uncertainty_reasons(paper: Paper, query: QuerySpec) -> list[str]:
    reasons: list[str] = []
    if paper.publication_year is None and (
        query.year_from is not None or query.year_to is not None
    ):
        reasons.append("missing_year")
    if _normalize_optional_text(paper.venue) is None and query.venues:
        reasons.append("missing_venue")
    if paper.is_retracted is None:
        reasons.append("unknown_retraction_status")
    if _normalize_optional_text(paper.abstract) is None and query.exclusions:
        reasons.append("missing_abstract_for_exclusion")
    return reasons


def apply_hard_filters(papers: Sequence[Paper], query: QuerySpec) -> FilterResult:
    """Apply ordered rejection rules while preserving candidate order."""
    accepted: list[AcceptedPaper] = []
    rejected: list[RejectedPaper] = []
    for paper in papers:
        rejection = _rejection_reason(paper, query)
        if rejection is not None:
            reason_code, reason = rejection
            rejected.append(RejectedPaper(paper=paper, reason_code=reason_code, reason=reason))
            continue
        uncertainty_reasons = _uncertainty_reasons(paper, query)
        accepted.append(
            AcceptedPaper(
                paper=paper,
                uncertainty_reasons=uncertainty_reasons,
                score_multiplier=max(0.7, 0.9 ** len(uncertainty_reasons)),
            )
        )
    return FilterResult(accepted=accepted, rejected=rejected)
