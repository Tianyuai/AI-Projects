from __future__ import annotations

import re
import unicodedata

from pydantic import Field, JsonValue, field_validator

from paper_search.domain.models import DomainModel, NonEmptyStr


_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "http://dx.doi.org/",
    "doi:",
)
_ARXIV_PATTERN = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})$",
    re.IGNORECASE,
)
_ARXIV_PREFIXES = (
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
    "https://arxiv.org/pdf/",
    "http://arxiv.org/pdf/",
    "arxiv:",
)
_OPENALEX_PATTERN = re.compile(r"^W\d+$", re.IGNORECASE)
_OPENALEX_PREFIXES = (
    "https://openalex.org/",
    "http://openalex.org/",
    "openalex:",
)
_S2_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_S2_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?semanticscholar\.org/paper/(?:[^/]+/)?([^/?#]+)",
    re.IGNORECASE,
)


def normalize_title(value: str) -> str:
    """Normalize a title for diagnostic fallback matching."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    collapsed = " ".join(without_punctuation.split())
    if not collapsed:
        raise ValueError("title must contain letters or numbers")
    return collapsed


def normalize_paper_id(value: str, *, kind: str | None = None) -> str:
    """Return the canonical namespaced representation of a paper identifier."""
    candidate = value.strip()
    lowered = candidate.casefold()
    for prefix in _DOI_PREFIXES:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break

    if (kind is None or kind.casefold() == "doi") and _DOI_PATTERN.fullmatch(candidate):
        return f"doi:{candidate.casefold()}"

    candidate = value.strip()
    lowered = candidate.casefold()
    for prefix in _ARXIV_PREFIXES:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    if candidate.casefold().endswith(".pdf"):
        candidate = candidate[:-4]
    candidate = re.sub(r"v\d+$", "", candidate, flags=re.IGNORECASE)

    if (kind is None or kind.casefold() == "arxiv") and _ARXIV_PATTERN.fullmatch(candidate):
        return f"arxiv:{candidate.casefold()}"

    candidate = value.strip()
    lowered = candidate.casefold()
    for prefix in _OPENALEX_PREFIXES:
        if lowered.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    if (kind is None or kind.casefold() == "openalex") and _OPENALEX_PATTERN.fullmatch(
        candidate
    ):
        return f"openalex:{candidate.upper()}"

    candidate = value.strip()
    semantic_url_match = _S2_URL_PATTERN.match(candidate)
    if semantic_url_match is not None:
        candidate = semantic_url_match.group(1)
    elif candidate.casefold().startswith("s2:"):
        candidate = candidate[3:]
    if (
        (semantic_url_match is not None or value.strip().casefold().startswith("s2:")
         or kind is not None and kind.casefold() in {"s2", "semantic_scholar"})
        and _S2_PATTERN.fullmatch(candidate)
    ):
        return f"s2:{candidate}"

    candidate = value.strip()
    is_explicit_title = candidate.casefold().startswith("title:")
    if is_explicit_title:
        candidate = candidate[6:]
    if is_explicit_title or kind is not None and kind.casefold() == "title":
        return f"title:{normalize_title(candidate)}"

    raise ValueError(f"unsupported or invalid paper identifier: {value!r}")


class EvaluationQuery(DomainModel):
    """One normalized evaluation query and its relevant paper identifiers."""

    query_id: NonEmptyStr
    query: NonEmptyStr
    relevant_paper_ids: list[NonEmptyStr] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("relevant_paper_ids")
    @classmethod
    def normalize_gold_ids(cls, values: list[str]) -> list[str]:
        normalized = [normalize_paper_id(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("relevant_paper_ids contains duplicate canonical IDs")
        return normalized


class PredictionRecord(DomainModel):
    """One ranked prediction list; duplicates are retained until scoring."""

    query_id: NonEmptyStr
    predicted_paper_ids: list[NonEmptyStr] = Field(default_factory=list)

    @field_validator("predicted_paper_ids")
    @classmethod
    def normalize_prediction_ids(cls, values: list[str]) -> list[str]:
        return [normalize_paper_id(value) for value in values]
