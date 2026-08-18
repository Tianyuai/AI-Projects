"""Method-level supervision for deciding whether to launch citation expansion."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, Paper, UnitFloat
from paper_search.evaluation.dataset import normalize_paper_id
from paper_search.learning.data_isolation import DatasetRole
from paper_search.recall_experiments.paper_identity import arxiv_datacite_anchor


class GraphMethodLabel(DomainModel):
    dataset: NonEmptyStr
    split: NonEmptyStr
    role: Literal["training", "development"]
    query_id: NonEmptyStr
    query: NonEmptyStr
    routing_label: Literal["beneficial", "not_beneficial", "unavailable"]
    gold_association_count: int = Field(strict=True, gt=0)
    anchor_gold_hit_ids: tuple[NonEmptyStr, ...] = ()
    pre_graph_gold_hit_ids: tuple[NonEmptyStr, ...] = ()
    graph_gold_hit_ids: tuple[NonEmptyStr, ...] = ()
    graph_marginal_gold_hit_ids: tuple[NonEmptyStr, ...] = ()
    graph_marginal_recall: UnitFloat
    seed_count: int = Field(strict=True, ge=0)
    graph_action_count: int = Field(strict=True, ge=0)
    search_api_calls: int = Field(strict=True, ge=0)


def _identity(value: str) -> str:
    normalized = normalize_paper_id(value)
    if normalized.startswith("arxiv:"):
        return arxiv_datacite_anchor(normalized)
    return normalized


def _paper_identities(paper: Paper) -> set[str]:
    values = [paper.canonical_id]
    if paper.doi is not None:
        values.append(f"doi:{paper.doi}")
    if paper.arxiv_id is not None:
        values.append(f"arxiv:{paper.arxiv_id}")
    if paper.openalex_id is not None:
        values.append(f"openalex:{paper.openalex_id}")
    return {_identity(value) for value in values}


def _gold_hits(papers: list[Paper], gold: set[str]) -> set[str]:
    identities: set[str] = set()
    for paper in papers:
        identities.update(_paper_identities(paper))
    return identities.intersection(gold)


def build_graph_method_label(
    *,
    dataset: str,
    split: str,
    role: DatasetRole,
    query_id: str,
    query: str,
    gold_paper_ids: list[str],
    anchor_hits: list[Paper],
    pre_graph_hits: list[Paper],
    graph_hits: list[Paper],
    seed_count: int,
    graph_action_count: int,
    graph_infrastructure_failure: bool,
    search_api_calls: int,
) -> GraphMethodLabel:
    if role == "final_test":
        raise ValueError("final_test cannot produce graph method labels")
    gold = {_identity(value) for value in gold_paper_ids}
    if not gold:
        raise ValueError("graph method labels require Gold associations")
    anchor_gold = _gold_hits(anchor_hits, gold)
    pre_graph_gold = _gold_hits(pre_graph_hits, gold)
    graph_gold = _gold_hits(graph_hits, gold)
    marginal = graph_gold.difference(pre_graph_gold)
    unavailable = graph_action_count == 0 or graph_infrastructure_failure
    routing_label: Literal["beneficial", "not_beneficial", "unavailable"]
    if unavailable:
        routing_label = "unavailable"
    elif marginal:
        routing_label = "beneficial"
    else:
        routing_label = "not_beneficial"
    return GraphMethodLabel(
        dataset=dataset,
        split=split,
        role=role,
        query_id=query_id,
        query=query,
        routing_label=routing_label,
        gold_association_count=len(gold),
        anchor_gold_hit_ids=tuple(sorted(anchor_gold)),
        pre_graph_gold_hit_ids=tuple(sorted(pre_graph_gold)),
        graph_gold_hit_ids=tuple(sorted(graph_gold)),
        graph_marginal_gold_hit_ids=tuple(sorted(marginal)),
        graph_marginal_recall=len(marginal) / len(gold),
        seed_count=seed_count,
        graph_action_count=graph_action_count,
        search_api_calls=search_api_calls,
    )


__all__ = ["GraphMethodLabel", "build_graph_method_label"]
