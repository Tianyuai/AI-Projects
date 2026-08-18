"""Deterministic duplicate clustering for normalized papers."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, Paper
from paper_search.evaluation.dataset import IdentifierMap, normalize_paper_id, normalize_title


MatchRule = Literal["doi", "external_id", "exact_title", "fuzzy_title"]
DEDUPLICATION_VERSION = "week1-dedup-v1"
FUZZY_TITLE_THRESHOLD = 0.98
_RULE_PRIORITY: dict[MatchRule, int] = {
    "doi": 0,
    "external_id": 1,
    "exact_title": 2,
    "fuzzy_title": 3,
}


class MergeDecision(DomainModel):
    """Explain why a multi-paper duplicate cluster was formed."""

    representative_id: NonEmptyStr
    member_ids: list[NonEmptyStr]
    match_rule: MatchRule
    match_value: NonEmptyStr
    conflict_fields: list[NonEmptyStr] = Field(default_factory=list)


class DeduplicationResult(DomainModel):
    """Stable deduplicated papers and decisions for merged clusters."""

    papers: list[Paper]
    decisions: list[MergeDecision]


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self._parents = list(range(size))

    def find(self, item: int) -> int:
        while self._parents[item] != item:
            self._parents[item] = self._parents[self._parents[item]]
            item = self._parents[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parents[right_root] = left_root
        else:
            self._parents[left_root] = right_root


def _try_normalize(value: str | None, *, kind: str | None = None) -> str | None:
    if value is None:
        return None
    try:
        return normalize_paper_id(value, kind=kind)
    except ValueError:
        return None


def _try_normalize_title(value: str) -> str | None:
    try:
        return normalize_title(value)
    except ValueError:
        return None


def _doi_identifier(paper: Paper) -> str | None:
    return _try_normalize(paper.doi, kind="doi")


def _external_identifiers(paper: Paper, id_map: IdentifierMap | None) -> set[str]:
    candidates = (
        _doi_identifier(paper),
        _try_normalize(paper.openalex_id, kind="openalex"),
        _try_normalize(paper.semantic_scholar_id, kind="semantic_scholar"),
        _try_normalize(paper.canonical_id),
    )
    identifiers = {
        candidate
        for candidate in candidates
        if candidate is not None and not candidate.startswith("title:")
    }
    if id_map is None:
        return identifiers
    return {id_map.resolve(identifier) for identifier in identifiers}


def _author_surnames(paper: Paper) -> set[str]:
    surnames: set[str] = set()
    for author in paper.authors:
        normalized = unicodedata.normalize("NFKC", author).casefold()
        without_punctuation = "".join(
            " " if unicodedata.category(character).startswith("P") else character
            for character in normalized
        )
        tokens = without_punctuation.split()
        if tokens:
            surnames.add(tokens[-1])
    return surnames


def _validate_fuzzy_title_threshold(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError("fuzzy_title_threshold must be a finite number between 0.0 and 1.0")
    return float(value)


def _ordered_union(values: Sequence[Sequence[str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in values:
        for value in group:
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result


def _representative_rank(paper: Paper) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(_doi_identifier(paper) is not None),
        len(_external_identifiers(paper, None)),
        int(paper.abstract is not None),
        len(paper.authors),
        int(paper.publication_year is not None),
        int(paper.venue is not None),
        int(paper.citation_count is not None),
    )


def _representative_index(members: Sequence[int], papers: Sequence[Paper]) -> int:
    return max(
        members,
        key=lambda index: (*_representative_rank(papers[index]), -index),
    )


def _merge_cluster(
    members: Sequence[int], papers: Sequence[Paper]
) -> tuple[Paper, int, list[str]]:
    representative_index = _representative_index(members, papers)
    representative = papers[representative_index]
    data = representative.model_dump()
    for field in (
        "abstract",
        "publication_year",
        "venue",
        "doi",
        "arxiv_id",
        "openalex_id",
        "semantic_scholar_id",
        "url",
        "citation_count",
    ):
        if data[field] is not None:
            continue
        for index in members:
            value = getattr(papers[index], field)
            if value is not None:
                data[field] = value
                break

    retraction_values = {papers[index].is_retracted for index in members}
    data["is_retracted"] = (
        True if True in retraction_values else False if False in retraction_values else None
    )
    conflict_fields = (
        ["is_retracted"] if True in retraction_values and False in retraction_values else []
    )

    normalized_doi = _try_normalize(data["doi"], kind="doi")
    if normalized_doi is not None:
        data["doi"] = normalized_doi.removeprefix("doi:")
    ordered_members = [representative_index, *(index for index in members if index != representative_index)]
    data["authors"] = _ordered_union([papers[index].authors for index in ordered_members])
    data["sources"] = _ordered_union([papers[index].sources for index in ordered_members])
    return Paper.model_validate(data), representative_index, conflict_fields


def deduplicate_papers(
    papers: Sequence[Paper],
    *,
    id_map: IdentifierMap | None = None,
    fuzzy_title_threshold: float = FUZZY_TITLE_THRESHOLD,
) -> DeduplicationResult:
    """Merge duplicate papers while preserving stable cluster order."""
    fuzzy_title_threshold = _validate_fuzzy_title_threshold(fuzzy_title_threshold)
    disjoint_set = _DisjointSet(len(papers))
    doi_identifiers = [_doi_identifier(paper) for paper in papers]
    external_identifiers = [_external_identifiers(paper, id_map) for paper in papers]
    normalized_titles = [_try_normalize_title(paper.title) for paper in papers]
    author_surnames = [_author_surnames(paper) for paper in papers]
    edges: list[tuple[int, int, MatchRule, str]] = []

    for left in range(len(papers)):
        for right in range(left + 1, len(papers)):
            left_doi = doi_identifiers[left]
            left_title = normalized_titles[left]
            right_title = normalized_titles[right]
            if left_doi is not None and left_doi == doi_identifiers[right]:
                rule: MatchRule = "doi"
                value = left_doi
            else:
                shared_identifiers = external_identifiers[left] & external_identifiers[right]
                if shared_identifiers:
                    rule = "external_id"
                    value = min(shared_identifiers)
                elif left_title is not None and left_title == right_title:
                    rule = "exact_title"
                    value = left_title
                else:
                    if left_title is None or right_title is None:
                        continue
                    same_known_year = (
                        papers[left].publication_year is not None
                        and papers[left].publication_year == papers[right].publication_year
                    )
                    shared_surnames = author_surnames[left] & author_surnames[right]
                    if not same_known_year or not shared_surnames:
                        continue
                    ratio = SequenceMatcher(
                        None,
                        left_title,
                        right_title,
                    ).ratio()
                    if ratio < fuzzy_title_threshold:
                        continue
                    rule = "fuzzy_title"
                    value = f"{ratio:.6f}"
            disjoint_set.union(left, right)
            edges.append((left, right, rule, value))

    clusters: dict[int, list[int]] = {}
    for index in range(len(papers)):
        clusters.setdefault(disjoint_set.find(index), []).append(index)

    merged_papers: list[Paper] = []
    decisions: list[MergeDecision] = []
    for members in clusters.values():
        merged, representative_index, conflict_fields = _merge_cluster(members, papers)
        merged_papers.append(merged)
        if len(members) == 1:
            continue
        cluster_edges = [edge for edge in edges if edge[0] in members and edge[1] in members]
        best_edge = min(cluster_edges, key=lambda edge: _RULE_PRIORITY[edge[2]])
        decisions.append(
            MergeDecision(
                representative_id=papers[representative_index].canonical_id,
                member_ids=[papers[index].canonical_id for index in members],
                match_rule=best_edge[2],
                match_value=best_edge[3],
                conflict_fields=conflict_fields,
            )
        )

    return DeduplicationResult(papers=merged_papers, decisions=decisions)
