"""Deterministic reciprocal-rank fusion across provider results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, Paper, ProviderResult
from paper_search.processing.deduplicate import deduplicate_papers


FusionMethod = Literal["rrf", "weighted"]


class FusedPaper(DomainModel):
    paper: Paper
    score: float = Field(ge=0, allow_inf_nan=False)
    source_ranks: dict[str, int]


def fuse_provider_results(
    results: Mapping[str, ProviderResult[list[Paper]]],
    *,
    method: FusionMethod | str,
    rrf_k: int = 60,
    provider_weights: Mapping[str, float] | None = None,
) -> list[FusedPaper]:
    """Merge duplicates and combine source ranks with deterministic tie breaks."""
    if method not in {"rrf", "weighted"}:
        raise ValueError("method must be 'rrf' or 'weighted'")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")
    if method == "weighted":
        if not provider_weights:
            raise ValueError("weighted fusion requires provider_weights")
        for provider, weight in provider_weights.items():
            if not provider or not math.isfinite(weight) or weight < 0:
                raise ValueError("provider weights must be finite nonnegative numbers")

    flattened: list[Paper] = []
    ranks_by_original: dict[str, dict[str, int]] = {}
    for provider in sorted(results):
        for rank, paper in enumerate(results[provider].data, start=1):
            flattened.append(paper)
            ranks_by_original.setdefault(paper.canonical_id, {})[provider] = rank
    if not flattened:
        return []

    deduplicated = deduplicate_papers(flattened)
    member_to_representative = {paper.canonical_id: paper.canonical_id for paper in flattened}
    for decision in deduplicated.decisions:
        for member in decision.member_ids:
            member_to_representative[member] = decision.representative_id

    combined_ranks: dict[str, dict[str, int]] = {
        paper.canonical_id: {} for paper in deduplicated.papers
    }
    for original_id, provider_ranks in ranks_by_original.items():
        representative = member_to_representative[original_id]
        target = combined_ranks.setdefault(representative, {})
        for provider, rank in provider_ranks.items():
            target[provider] = min(target.get(provider, rank), rank)

    fused: list[FusedPaper] = []
    for paper in deduplicated.papers:
        source_ranks = combined_ranks[paper.canonical_id]
        if method == "rrf":
            score = sum(1.0 / (rrf_k + rank) for rank in source_ranks.values())
        else:
            assert provider_weights is not None
            score = sum(provider_weights.get(provider, 0.0) / rank for provider, rank in source_ranks.items())
        fused.append(FusedPaper(paper=paper, score=score, source_ranks=source_ranks))
    fused.sort(key=lambda item: (-item.score, item.paper.canonical_id))
    return fused
