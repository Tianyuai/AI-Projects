"""Gold-blind OpenAlex confidence routing with PASA kept diagnostic-only."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, QuerySpec
from paper_search.evaluation.dataset import normalize_title
from paper_search.evaluation.predictions import paper_matches_evaluation_ids
from paper_search.learning.candidate_ceiling import QueryAdaptiveHighRecallGenerator
from paper_search.learning.candidates import query_content_terms
from paper_search.learning.cpu_document_ranker import (
    DocumentCandidateEvidence,
    DocumentRankingQuery,
)
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    RecallGenerationContext,
    RecallSearchAction,
)


GoldMissCategory = Literal[
    "online_gold_hit",
    "identity_mismatch_suspected",
    "pasa_only_gold",
    "gold_metadata_unavailable",
]
ConfidenceReason = Literal[
    "zero_results",
    "low_yield",
    "low_query_alignment",
    "facet_gap",
    "cross_action_disagreement",
    "single_action_support",
]


class OpenAlexGoldMissAttribution(DomainModel):
    query_id: str
    category: GoldMissCategory
    online_gold_hit_count: int = Field(ge=0)
    pasa_only_gold_count: int = Field(ge=0)
    title_alias_candidate_count: int = Field(ge=0)
    pasa_used_for_action_generation: Literal[False] = False


class OpenAlexRecallDecision(DomainModel):
    low_confidence: bool
    reason_codes: list[ConfidenceReason]
    candidate_count: int = Field(ge=0)
    action_count: int = Field(ge=0)
    cross_action_supported_candidate_count: int = Field(ge=0)
    required_facet_count: int = Field(ge=0)
    covered_facet_count: int = Field(ge=0)
    query_aligned_candidate_count: int = Field(default=0, ge=0)
    minimum_query_aligned_candidate_count: int = Field(default=0, ge=0)
    gold_features_used: Literal[False] = False


def _is_pasa_source(value: str) -> bool:
    return "pasa" in value.casefold()


def _is_pasa_only(candidate: DocumentCandidateEvidence) -> bool:
    source_names = tuple(candidate.source_ranks)
    if any(not _is_pasa_source(source) for source in source_names):
        return False
    return bool(source_names) and all(_is_pasa_source(source) for source in source_names)


def _online_candidates(
    candidates: Sequence[DocumentCandidateEvidence],
) -> list[DocumentCandidateEvidence]:
    return [candidate for candidate in candidates if not _is_pasa_only(candidate)]


def attribute_openalex_gold_miss(
    query: DocumentRankingQuery,
) -> OpenAlexGoldMissAttribution:
    """Classify mixed-pool Gold visibility without using PASA to form actions."""

    online = _online_candidates(query.candidates)
    pasa_only = [candidate for candidate in query.candidates if _is_pasa_only(candidate)]
    online_gold = [
        candidate
        for candidate in online
        if paper_matches_evaluation_ids(candidate.paper, query.gold_paper_ids)
    ]
    pasa_gold = [
        candidate
        for candidate in pasa_only
        if paper_matches_evaluation_ids(candidate.paper, query.gold_paper_ids)
    ]
    title_aliases = {
        normalize_title(candidate.paper.title) for candidate in pasa_gold
    }.intersection(normalize_title(candidate.paper.title) for candidate in online)
    if online_gold:
        category: GoldMissCategory = "online_gold_hit"
    elif title_aliases:
        category = "identity_mismatch_suspected"
    elif pasa_gold:
        category = "pasa_only_gold"
    else:
        category = "gold_metadata_unavailable"
    return OpenAlexGoldMissAttribution(
        query_id=query.query_id,
        category=category,
        online_gold_hit_count=len(online_gold),
        pasa_only_gold_count=len(pasa_gold),
        title_alias_candidate_count=len(title_aliases),
    )


def _facets(spec: QuerySpec) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in [*spec.methods, *spec.datasets, *spec.tasks, *spec.must_have]:
        normalized = normalize_title(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(value)
    return output


def _facet_is_covered(
    facet: str, candidates: Sequence[DocumentCandidateEvidence]
) -> bool:
    normalized_facet = normalize_title(facet)
    facet_terms = set(query_content_terms(facet))
    for candidate in candidates:
        searchable = " ".join(
            part
            for part in (candidate.paper.title, candidate.paper.abstract or "")
            if part
        )
        normalized_searchable = normalize_title(searchable)
        if normalized_facet in normalized_searchable:
            return True
        if facet_terms and facet_terms.issubset(query_content_terms(searchable)):
            return True
    return False


def assess_openalex_recall_confidence(
    spec: QuerySpec,
    candidates: Sequence[DocumentCandidateEvidence],
    *,
    output_limit: int = 50,
) -> OpenAlexRecallDecision:
    """Derive a conservative confidence decision from runtime-observable evidence."""

    if output_limit <= 0:
        raise ValueError("output_limit must be positive")
    online = sorted(
        _online_candidates(candidates),
        key=lambda candidate: candidate.baseline_score,
        reverse=True,
    )
    top = online[:output_limit]
    source_sets = [
        {source for source in candidate.source_ranks if not _is_pasa_source(source)}
        for candidate in online
    ]
    actions = set().union(*source_sets) if source_sets else set()
    cross_supported = sum(len(sources) >= 2 for sources in source_sets)
    facets = _facets(spec)
    covered = sum(_facet_is_covered(facet, top) for facet in facets)
    original_terms = set(query_content_terms(spec.original_query))
    minimum_overlap = min(2, len(original_terms))
    aligned = sum(
        bool(minimum_overlap)
        and len(
            original_terms.intersection(
                query_content_terms(
                    " ".join(
                        part
                        for part in (
                            candidate.paper.title,
                            candidate.paper.abstract or "",
                        )
                        if part
                    )
                )
            )
        )
        >= minimum_overlap
        for candidate in top
    )
    minimum_aligned = max(3, math.ceil(output_limit * 0.25))

    reasons: list[ConfidenceReason] = []
    if not online:
        reasons.append("zero_results")
    elif len(online) < output_limit:
        reasons.append("low_yield")
    if facets and covered < len(facets):
        reasons.append("facet_gap")
    if len(actions) >= 2 and cross_supported == 0:
        reasons.append("cross_action_disagreement")
    elif online and len(actions) < 2:
        reasons.append("single_action_support")
    if not facets and len(top) == output_limit and aligned < minimum_aligned:
        reasons.append("low_query_alignment")

    low_confidence = bool(
        {"zero_results", "low_yield", "low_query_alignment", "facet_gap"}.intersection(
            reasons
        )
    )
    return OpenAlexRecallDecision(
        low_confidence=low_confidence,
        reason_codes=reasons if low_confidence else [],
        candidate_count=len(online),
        action_count=len(actions),
        cross_action_supported_candidate_count=cross_supported,
        required_facet_count=len(facets),
        covered_facet_count=covered,
        query_aligned_candidate_count=aligned,
        minimum_query_aligned_candidate_count=minimum_aligned,
    )


def _action_identity(action: RecallSearchAction) -> tuple[str, str, str] | None:
    if action.action_type != "text_search":
        return None
    return (
        action.action_type,
        action.payload.search_mode,
        " ".join(action.payload.query_text.split()).casefold(),
    )


_PRIORITIES: Mapping[ConfidenceReason, tuple[str, ...]] = {
    "zero_results": (
        "high-recall-original-semantic",
        "high-recall-topic-lexical",
        "high-recall-entity-lexical",
    ),
    "low_yield": (
        "high-recall-original-semantic",
        "high-recall-topic-lexical",
    ),
    "low_query_alignment": (
        "high-recall-topic-lexical",
        "high-recall-original-semantic",
    ),
    "facet_gap": (
        "high-recall-entity-lexical",
        "high-recall-context-semantic",
    ),
    "cross_action_disagreement": (
        "high-recall-context-semantic",
        "high-recall-topic-semantic",
    ),
    "single_action_support": (
        "high-recall-original-semantic",
        "high-recall-topic-lexical",
    ),
}


async def select_openalex_supplement_actions(
    context: RecallGenerationContext,
    *,
    frozen_query_specs: Mapping[str, QuerySpec],
    first_round_actions: Sequence[RecallSearchAction],
    decision: OpenAlexRecallDecision,
    max_total_actions: int = 6,
) -> RecallActionBatch:
    """Select existing v2 templates within unspent budget; never inspect Gold."""

    if context.gold_documents:
        raise ValueError("supplement selection must be Gold-blind")
    if not 1 <= max_total_actions <= 6:
        raise ValueError("max_total_actions must be between one and six")
    if len(first_round_actions) > max_total_actions:
        raise ValueError("first-round actions exceed the total OpenAlex budget")
    remaining = max_total_actions - len(first_round_actions)
    if not decision.low_confidence or remaining == 0:
        return RecallActionBatch(actions=[])

    generated = await QueryAdaptiveHighRecallGenerator(
        frozen_query_specs=frozen_query_specs,
        max_openalex_actions=6,
    ).generate(context)
    by_id = {action.action_id: action for action in generated.action_batch.actions}
    ordered_ids: list[str] = []
    for reason in decision.reason_codes:
        for action_id in _PRIORITIES[reason]:
            if action_id in by_id and action_id not in ordered_ids:
                ordered_ids.append(action_id)
    ordered_ids.extend(
        action.action_id
        for action in generated.action_batch.actions
        if action.action_id not in ordered_ids
    )

    seen = {
        identity
        for action in first_round_actions
        if (identity := _action_identity(action)) is not None
    }
    selected: list[RecallSearchAction] = []
    for action_id in ordered_ids:
        action = by_id[action_id]
        identity = _action_identity(action)
        if identity is None or identity in seen:
            continue
        seen.add(identity)
        selected.append(action)
        if len(selected) == remaining:
            break
    return RecallActionBatch(actions=selected)


__all__ = [
    "OpenAlexGoldMissAttribution",
    "OpenAlexRecallDecision",
    "assess_openalex_recall_confidence",
    "attribute_openalex_gold_miss",
    "select_openalex_supplement_actions",
]
