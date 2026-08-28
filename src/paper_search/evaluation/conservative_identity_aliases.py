"""Gold-blind, metadata-verified aliases between PASA and provider records."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TypedDict

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import normalize_paper_id, normalize_title
from paper_search.evaluation.predictions import paper_evaluation_aliases


MIN_ABSTRACT_TOKEN_JACCARD = 0.60
MIN_SHARED_ABSTRACT_TOKENS = 40
MAX_PUBLICATION_YEAR_DISTANCE = 2
MIN_TITLE_TOKEN_COUNT = 3


class ConservativeAliasEvidence(TypedDict):
    target_id: str
    candidate_aliases: list[str]
    normalized_title: str
    abstract_token_jaccard: float
    shared_abstract_token_count: int
    publication_year_distance: int


def _tokens(value: str | None) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", (value or "").casefold()))


def _candidate_aliases(paper: Paper) -> tuple[str, ...]:
    normalized: set[str] = set()
    for alias in paper_evaluation_aliases(paper):
        try:
            normalized.add(normalize_paper_id(alias))
        except ValueError:
            continue
    return tuple(sorted(normalized))


def _evaluate_candidate(
    candidate: Paper,
    references: Sequence[Paper],
) -> tuple[str, ConservativeAliasEvidence | None]:
    if not references:
        return "no_pasa_title_match", None
    if len(references) != 1:
        return "ambiguous_pasa_title", None
    reference = references[0]
    if reference.arxiv_id is None:
        return "pasa_arxiv_identity_missing", None
    if normalize_title(candidate.title) != normalize_title(reference.title):
        return "normalized_title_mismatch", None
    if len(_tokens(candidate.title)) < MIN_TITLE_TOKEN_COUNT:
        return "title_too_short", None

    target_id = normalize_paper_id(reference.arxiv_id, kind="arxiv")
    aliases = _candidate_aliases(candidate)
    if target_id in aliases:
        return "already_linked", None
    if candidate.arxiv_id is not None:
        return "conflicting_candidate_arxiv", None
    if not candidate.abstract or not reference.abstract:
        return "missing_abstract", None

    candidate_tokens = _tokens(candidate.abstract)
    reference_tokens = _tokens(reference.abstract)
    shared = candidate_tokens.intersection(reference_tokens)
    union = candidate_tokens.union(reference_tokens)
    jaccard = len(shared) / len(union) if union else 0.0
    if (
        len(shared) < MIN_SHARED_ABSTRACT_TOKENS
        or jaccard < MIN_ABSTRACT_TOKEN_JACCARD
    ):
        return "insufficient_abstract_overlap", None
    if candidate.publication_year is None or reference.publication_year is None:
        return "missing_publication_year", None
    year_distance = abs(candidate.publication_year - reference.publication_year)
    if year_distance > MAX_PUBLICATION_YEAR_DISTANCE:
        return "publication_year_mismatch", None
    if not aliases:
        return "candidate_identifier_missing", None

    return (
        "accepted",
        {
            "target_id": target_id,
            "candidate_aliases": list(aliases),
            "normalized_title": normalize_title(candidate.title),
            "abstract_token_jaccard": round(jaccard, 6),
            "shared_abstract_token_count": len(shared),
            "publication_year_distance": year_distance,
        },
    )


def build_conservative_pasa_identifier_aliases(
    candidates: Sequence[Paper],
    references_by_title: Mapping[str, Sequence[Paper]],
) -> tuple[
    dict[str, str],
    list[ConservativeAliasEvidence],
    dict[str, int],
]:
    """Build aliases without query labels, Gold ids, or title-only acceptance."""

    aliases: dict[str, str] = {}
    evidence_by_identity: dict[
        tuple[str, tuple[str, ...]], ConservativeAliasEvidence
    ] = {}
    decisions: Counter[str] = Counter()
    for candidate in candidates:
        title_key = normalize_title(candidate.title)
        reason, evidence = _evaluate_candidate(
            candidate,
            references_by_title.get(title_key, ()),
        )
        decisions[reason] += 1
        if evidence is None:
            continue
        candidate_aliases = tuple(evidence["candidate_aliases"])
        target_id = evidence["target_id"]
        for alias in candidate_aliases:
            previous = aliases.get(alias)
            if previous is not None and previous != target_id:
                raise ValueError(f"conservative identity alias conflict for {alias}")
            aliases[alias] = target_id
        evidence_by_identity[(target_id, candidate_aliases)] = evidence
    return (
        dict(sorted(aliases.items())),
        [evidence_by_identity[key] for key in sorted(evidence_by_identity)],
        dict(sorted(decisions.items())),
    )


__all__ = [
    "ConservativeAliasEvidence",
    "MAX_PUBLICATION_YEAR_DISTANCE",
    "MIN_ABSTRACT_TOKEN_JACCARD",
    "MIN_SHARED_ABSTRACT_TOKENS",
    "MIN_TITLE_TOKEN_COUNT",
    "build_conservative_pasa_identifier_aliases",
]
