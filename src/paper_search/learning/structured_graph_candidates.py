"""Deterministic structured query decomposition followed by grounded graph expansion."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence

from paper_search.learning.candidates import query_content_terms
from paper_search.recall_experiments.contracts import (
    CitationExpandAction,
    CitationExpandPayload,
    RecallActionBatch,
    RecallGenerationContext,
    RetrievalActionResult,
    TextSearchAction,
    TextSearchPayload,
    TitleSearchAction,
    TitleSearchPayload,
)
from paper_search.recall_experiments.generation.base import GenerationResult


_POLICY_VERSION = "structured-graph-candidate-pool-v1"
_POLICY_MATERIAL = json.dumps(
    {
        "decomposition": "relation-target-and-constraint-facets-v1",
        "graph": "grounded-openalex-references-and-citations-v1",
        "selection": "lexical-evidence-v1",
        "version": _POLICY_VERSION,
    },
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
_RELATION_TARGET = re.compile(
    r"\b(introduced|proposed|developed|presented|described|coined)\b"
    r"(?:\s+the\s+(?:concept|method|framework|model|algorithm)\s+of)?\s+(.+?)"
    r"[?.!]*$",
    flags=re.IGNORECASE,
)
_ACRONYM = re.compile(r"\s*\(([^()]{2,20})\)\s*")


def _artifact(batch: RecallActionBatch) -> bytes:
    return json.dumps(
        batch.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _clean_target(value: str) -> str:
    value = value.strip(" \t\r\n?.!")
    return " ".join(_ACRONYM.sub(r" \1 ", value).split())


def _structured_actions(query: str) -> list[TextSearchAction | TitleSearchAction]:
    actions: list[TextSearchAction | TitleSearchAction] = []
    relation = _RELATION_TARGET.search(query)
    if relation:
        verb = relation.group(1).casefold()
        target = _clean_target(relation.group(2))
        actions.extend(
            [
                TitleSearchAction(
                    action_id="structured-title-target",
                    strategy="structured:relation-target",
                    action_type="title_search",
                    payload=TitleSearchPayload(title_text=target),
                ),
                TextSearchAction(
                    action_id="structured-relation-target",
                    strategy="structured:relation-target",
                    action_type="text_search",
                    payload=TextSearchPayload(query_text=f"{verb} {target}"),
                ),
            ]
        )
    terms = query_content_terms(query)
    constraint_parts = re.split(r"\bfor\b", query, maxsplit=1, flags=re.IGNORECASE)
    if len(constraint_parts) == 2:
        left_terms = query_content_terms(constraint_parts[0])
        right_terms = query_content_terms(constraint_parts[1])
        anchor = left_terms[-2:]
        domain = right_terms[-2:]
        modifiers = right_terms[:-2]
    else:
        anchor, domain, modifiers = terms[:2], terms[-2:], terms[2:-2]
    if len(anchor) >= 2 and len(domain) >= 2:
        actions.append(
            TextSearchAction(
                action_id="structured-facet-domain",
                strategy="structured:constraint-facet",
                action_type="text_search",
                payload=TextSearchPayload(query_text=" ".join([*anchor, *domain])),
            )
        )
        if modifiers:
            actions.append(
                TextSearchAction(
                    action_id="structured-facet-modifiers",
                    strategy="structured:constraint-facet",
                    action_type="text_search",
                    payload=TextSearchPayload(
                        query_text=" ".join([*anchor, *modifiers[:3]])
                    ),
                )
            )
    return _deduplicated(actions)


def _action_identity(
    action: TextSearchAction | TitleSearchAction,
) -> tuple[str, str, str]:
    if isinstance(action, TextSearchAction):
        text = action.payload.query_text
        mode = action.payload.search_mode
    else:
        text = action.payload.title_text
        mode = "lexical"
    return action.action_type, mode, " ".join(text.split()).casefold()


def _deduplicated(
    actions: Sequence[TextSearchAction | TitleSearchAction],
) -> list[TextSearchAction | TitleSearchAction]:
    selected: list[TextSearchAction | TitleSearchAction] = []
    seen: set[tuple[str, str, str]] = set()
    for action in actions:
        identity = _action_identity(action)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(action)
    return selected


class StructuredCandidateGenerator:
    """Generate only structured lexical/title actions, without anchors or graph."""

    generator_type = "fixed_actions"
    source_sha256 = "sha256:" + hashlib.sha256(_POLICY_MATERIAL).hexdigest()
    candidate_policy = "structured-candidate-pool-v1"

    def __init__(self, *, max_actions: int = 4) -> None:
        if type(max_actions) is not int or not 1 <= max_actions <= 4:
            raise ValueError("max_actions must be an integer between 1 and 4")
        self.max_actions = max_actions

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        query = " ".join(unicodedata.normalize("NFKC", context.original_query).split())
        selected = _structured_actions(query)[: self.max_actions]
        batch = RecallActionBatch(actions=selected)
        return GenerationResult(
            query_id=context.query_id,
            action_batch=batch,
            artifact_bytes=_artifact(batch),
            provenance={
                "candidate_policy": self.candidate_policy,
                "candidate_pool_size": str(len(selected)),
                "collection_mode": "structured_only",
                "gold_visibility": "blind",
                "graph_status": "disabled",
            },
        )


class FixedBudgetOpenAlexQueryGenerator:
    """Build scheme A: original lexical + original semantic + structured actions."""

    generator_type = "fixed_actions"
    candidate_policy = "fixed-budget-openalex-v1"
    source_sha256 = "sha256:" + hashlib.sha256(
        b"fixed-budget-openalex-v1:1-lexical:1-semantic:4-structured"
    ).hexdigest()

    def __init__(self, *, max_openalex_actions: int = 6) -> None:
        if (
            type(max_openalex_actions) is not int
            or not 2 <= max_openalex_actions <= 6
        ):
            raise ValueError(
                "max_openalex_actions must be an integer between 2 and 6"
            )
        self.max_openalex_actions = max_openalex_actions

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        query = " ".join(unicodedata.normalize("NFKC", context.original_query).split())
        anchors: list[TextSearchAction | TitleSearchAction] = [
            TextSearchAction(
                action_id="openalex-original-lexical",
                strategy="fixed-budget:lexical-original",
                action_type="text_search",
                payload=TextSearchPayload(query_text=query, search_mode="lexical"),
            ),
            TextSearchAction(
                action_id="openalex-original-semantic",
                strategy="fixed-budget:semantic-original",
                action_type="text_search",
                payload=TextSearchPayload(query_text=query, search_mode="semantic"),
            ),
        ]
        structured_limit = min(4, self.max_openalex_actions - 2)
        actions = _deduplicated([*anchors, *_structured_actions(query)])[
            : 2 + structured_limit
        ]
        batch = RecallActionBatch(actions=actions)
        return GenerationResult(
            query_id=context.query_id,
            action_batch=batch,
            artifact_bytes=_artifact(batch),
            provenance={
                "candidate_policy": self.candidate_policy,
                "candidate_pool_size": str(len(actions)),
                "collection_mode": "fixed_budget_openalex",
                "gold_visibility": "blind",
                "graph_status": "disabled",
                "max_openalex_actions": str(self.max_openalex_actions),
            },
        )


class OpenAlexGraphExpansionGenerator:
    """Optional evidence-grounded graph fallback, separate from default retrieval."""

    def __init__(
        self,
        *,
        max_actions: int = 12,
        max_graph_seeds: int = 2,
        graph_limit: int = 50,
    ) -> None:
        if max_actions < 1 or max_graph_seeds < 0 or graph_limit < 1:
            raise ValueError("graph expansion limits must be positive")
        self.max_actions = max_actions
        self.max_graph_seeds = max_graph_seeds
        self.graph_limit = graph_limit

    async def refine(
        self,
        context: RecallGenerationContext,
        anchor_generation: GenerationResult,
        first_round_results: Sequence[RetrievalActionResult],
    ) -> GenerationResult:
        del first_round_results
        query_terms = set(query_content_terms(context.original_query))
        scored: list[tuple[int, str]] = []
        for seed in context.seed_candidates:
            paper = seed.paper
            if not paper.openalex_id:
                continue
            title_terms = set(query_content_terms(paper.title))
            abstract_terms = set(query_content_terms(paper.abstract or ""))
            title_overlap = query_terms.intersection(title_terms)
            total_overlap = query_terms.intersection(title_terms | abstract_terms)
            if len(total_overlap) < 2:
                continue
            score = 2 * len(title_overlap) + len(total_overlap)
            scored.append((score, paper.canonical_id))
        selected_ids = [
            canonical_id
            for _, canonical_id in sorted(scored, key=lambda item: (-item[0], item[1]))[
                : self.max_graph_seeds
            ]
        ]
        if not selected_ids:
            return anchor_generation.model_copy(
                update={
                    "provenance": {
                        **anchor_generation.provenance,
                        "graph_status": "no_grounded_openalex_seed",
                    }
                }
            )
        actions = list(anchor_generation.action_batch.actions)
        for seed_index, canonical_id in enumerate(selected_ids, start=1):
            for direction in ("references", "citations"):
                if len(actions) >= self.max_actions:
                    break
                actions.append(
                    CitationExpandAction(
                        action_id=f"structured-graph-{seed_index}-{direction}",
                        strategy="structured:openalex-citation-graph",
                        action_type="citation_expand",
                        payload=CitationExpandPayload(
                            seed_canonical_id=canonical_id,
                            direction=direction,
                            limit=self.graph_limit,
                        ),
                    )
                )
        batch = RecallActionBatch(actions=actions)
        return anchor_generation.model_copy(
            update={
                "action_batch": batch,
                "artifact_bytes": _artifact(batch),
                "provenance": {
                    **anchor_generation.provenance,
                    "candidate_pool_size": str(len(actions)),
                    "graph_status": "grounded_openalex_expansion",
                    "graph_seed_ids": ",".join(selected_ids),
                },
            }
        )


class StructuredGraphCandidateGenerator:
    """Compatibility wrapper for frozen structured+graph experiment receipts."""

    generator_type = "fixed_actions"
    source_sha256 = "sha256:" + hashlib.sha256(_POLICY_MATERIAL).hexdigest()
    candidate_policy = _POLICY_VERSION

    def __init__(
        self,
        *,
        max_actions: int = 12,
        max_graph_seeds: int = 2,
        graph_limit: int = 50,
    ) -> None:
        if max_actions < 2 or max_graph_seeds < 0 or graph_limit < 1:
            raise ValueError("structured graph limits must be positive")
        self.max_actions = max_actions
        self._graph = OpenAlexGraphExpansionGenerator(
            max_actions=max_actions,
            max_graph_seeds=max_graph_seeds,
            graph_limit=graph_limit,
        )

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        query = " ".join(unicodedata.normalize("NFKC", context.original_query).split())
        actions: list[TextSearchAction | TitleSearchAction] = [
            TextSearchAction(
                action_id="structured-anchor-original",
                strategy="structured:anchor-original",
                action_type="text_search",
                payload=TextSearchPayload(query_text=query, search_mode="lexical"),
            ),
            TextSearchAction(
                action_id="structured-anchor-semantic",
                strategy="structured:anchor-semantic",
                action_type="text_search",
                payload=TextSearchPayload(query_text=query, search_mode="semantic"),
            ),
            *_structured_actions(query),
        ]
        selected = _deduplicated(actions)[: min(self.max_actions, 8)]
        batch = RecallActionBatch(actions=selected)
        return GenerationResult(
            query_id=context.query_id,
            action_batch=batch,
            artifact_bytes=_artifact(batch),
            provenance={
                "candidate_policy": _POLICY_VERSION,
                "candidate_pool_size": str(len(selected)),
                "collection_mode": "structured_graph",
                "gold_visibility": "blind",
                "graph_status": "pending_first_round_evidence",
            },
        )
    async def refine(
        self,
        context: RecallGenerationContext,
        anchor_generation: GenerationResult,
        first_round_results: Sequence[RetrievalActionResult],
    ) -> GenerationResult:
        return await self._graph.refine(
            context,
            anchor_generation,
            first_round_results,
        )


__all__ = [
    "FixedBudgetOpenAlexQueryGenerator",
    "OpenAlexGraphExpansionGenerator",
    "StructuredCandidateGenerator",
    "StructuredGraphCandidateGenerator",
]
