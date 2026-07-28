from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from pydantic import Field

from paper_search.domain.models import (
    CitationEdge,
    CitationExpansion,
    DomainModel,
    Paper,
    ProviderPaperId,
    ResolvedCitationEdge,
)


_UNRESOLVED_EDGE_WARNING = "unresolved_citation_edge"


class CitationExpansionResult(DomainModel):
    papers: list[Paper]
    edges: list[ResolvedCitationEdge]
    skipped_edge_count: int = Field(ge=0)
    truncated: bool
    warnings: list[str] = Field(default_factory=list)


def _validate_limit(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_provider_id(
    provider_id: ProviderPaperId,
    canonical_by_provider_id: Mapping[ProviderPaperId, str],
) -> str | None:
    canonical_id = canonical_by_provider_id.get(provider_id)
    if canonical_id is None:
        return None
    normalized = canonical_id.strip()
    return normalized or None


def _paper_provider_ids(paper: Paper) -> tuple[ProviderPaperId, ...]:
    provider_ids: list[ProviderPaperId] = []
    if paper.openalex_id is not None:
        provider_ids.append(ProviderPaperId(provider="openalex", value=paper.openalex_id))
    if paper.semantic_scholar_id is not None:
        provider_ids.append(
            ProviderPaperId(provider="semantic_scholar", value=paper.semantic_scholar_id)
        )
    return tuple(provider_ids)


def _resolved_paper(paper: Paper, canonical_by_provider_id: Mapping[ProviderPaperId, str]) -> Paper:
    canonical_id = paper.canonical_id
    for provider_id in _paper_provider_ids(paper):
        resolved = _resolve_provider_id(provider_id, canonical_by_provider_id)
        if resolved is not None:
            canonical_id = resolved
            break
    if canonical_id == paper.canonical_id:
        return paper
    return paper.model_copy(update={"canonical_id": canonical_id})


def _source_edge_hash(*, provider: str, citing_canonical_id: str, cited_canonical_id: str) -> str:
    digest = hashlib.sha256(
        f"{provider}|{citing_canonical_id}|{cited_canonical_id}".encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _resolve_edge(
    edge: CitationEdge,
    canonical_by_provider_id: Mapping[ProviderPaperId, str],
) -> tuple[str, str] | None:
    if edge.citing_provider_id.provider != edge.provider:
        return None
    if edge.cited_provider_id.provider != edge.provider:
        return None
    citing_canonical_id = _resolve_provider_id(edge.citing_provider_id, canonical_by_provider_id)
    cited_canonical_id = _resolve_provider_id(edge.cited_provider_id, canonical_by_provider_id)
    if citing_canonical_id is None or cited_canonical_id is None:
        return None
    return citing_canonical_id, cited_canonical_id


def expand_one_hop(
    seeds: Sequence[Paper],
    expansion: CitationExpansion,
    canonical_by_provider_id: Mapping[ProviderPaperId, str],
    *,
    max_seeds: int = 1,
    max_expanded: int = 2,
) -> CitationExpansionResult:
    max_seeds = _validate_limit(max_seeds, name="max_seeds")
    max_expanded = _validate_limit(max_expanded, name="max_expanded")
    if not seeds:
        raise ValueError("seeds must not be empty")

    seed_canonical_ids = [paper.canonical_id for paper in seeds]
    if len(set(seed_canonical_ids)) != len(seed_canonical_ids):
        raise ValueError("seed canonical IDs must be unique")

    active_seeds = list(seeds[:max_seeds])
    active_seed_ids = {paper.canonical_id for paper in active_seeds}

    preliminary_edges: list[ResolvedCitationEdge] = []
    connected_expansion_ids: set[str] = set()
    seen_edge_keys: set[tuple[str, str, str]] = set()
    skipped_edge_count = 0

    for raw_edge in expansion.raw_edges:
        resolved = _resolve_edge(raw_edge, canonical_by_provider_id)
        if resolved is None:
            skipped_edge_count += 1
            continue

        citing_canonical_id, cited_canonical_id = resolved
        if citing_canonical_id == cited_canonical_id:
            continue
        if citing_canonical_id not in active_seed_ids and cited_canonical_id not in active_seed_ids:
            continue

        edge_key = (raw_edge.provider, citing_canonical_id, cited_canonical_id)
        if edge_key in seen_edge_keys:
            continue
        seen_edge_keys.add(edge_key)
        preliminary_edges.append(
            ResolvedCitationEdge(
                provider=raw_edge.provider,
                citing_canonical_id=citing_canonical_id,
                cited_canonical_id=cited_canonical_id,
                source_edge_hash=_source_edge_hash(
                    provider=raw_edge.provider,
                    citing_canonical_id=citing_canonical_id,
                    cited_canonical_id=cited_canonical_id,
                ),
            )
        )
        if citing_canonical_id not in active_seed_ids:
            connected_expansion_ids.add(citing_canonical_id)
        if cited_canonical_id not in active_seed_ids:
            connected_expansion_ids.add(cited_canonical_id)

    papers = list(active_seeds)
    included_expansion_ids: set[str] = set()
    truncated = False

    for raw_paper in expansion.papers:
        paper = _resolved_paper(raw_paper, canonical_by_provider_id)
        canonical_id = paper.canonical_id
        if canonical_id in active_seed_ids:
            continue
        if canonical_id not in connected_expansion_ids:
            continue
        if canonical_id in included_expansion_ids:
            continue
        if len(included_expansion_ids) >= max_expanded:
            truncated = True
            continue
        included_expansion_ids.add(canonical_id)
        papers.append(paper)

    allowed_ids = {paper.canonical_id for paper in papers}
    edges = [
        edge
        for edge in preliminary_edges
        if edge.citing_canonical_id in allowed_ids and edge.cited_canonical_id in allowed_ids
    ]

    warnings: list[str] = []
    if skipped_edge_count > 0:
        warnings.append(_UNRESOLVED_EDGE_WARNING)

    return CitationExpansionResult(
        papers=papers,
        edges=edges,
        skipped_edge_count=skipped_edge_count,
        truncated=truncated,
        warnings=warnings,
    )
