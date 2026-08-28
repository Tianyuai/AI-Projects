"""Blind, candidate-driven cross-vocabulary OpenAlex recall expansion."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from paper_search.domain.models import QuerySpec
from paper_search.learning.candidates import query_content_terms
from paper_search.learning.cpu_document_ranker import DocumentCandidateEvidence
from paper_search.learning.query_constraint_profile import profile_query_constraints
from paper_search.learning.structured_openalex_actions import (
    build_structured_openalex_action_batch,
    propose_structured_openalex_supplement,
)
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    TextSearchAction,
    TextSearchPayload,
)


_ACTION_ID = "contrastive-bridge-local-idf-v1"
_STRATEGY = "candidate-family:cross-vocabulary-bridge"
_REFINED_ACTION_ID = "contrastive-bridge-anchor-conditioned-v2"
_REFINED_STRATEGY = "candidate-family:cross-vocabulary-bridge-refined"
_PRODUCTION_SUPPRESSED_EXPANSION_TERMS = (
    "generation",
    "image",
    "language",
    "neural",
)
_GENERIC_BRIDGE_TERMS = frozenset(
    {
        "across",
        "absence",
        "absent",
        "against",
        "analysis",
        "among",
        "application",
        "applications",
        "approach",
        "approaches",
        "based",
        "between",
        "beyond",
        "challenge",
        "challenges",
        "data",
        "deep",
        "different",
        "efficient",
        "except",
        "excluding",
        "first",
        "framework",
        "frameworks",
        "future",
        "had",
        "has",
        "have",
        "high",
        "into",
        "its",
        "lack",
        "lacking",
        "large",
        "learning",
        "method",
        "methods",
        "model",
        "models",
        "missing",
        "network",
        "networks",
        "new",
        "no",
        "not",
        "novel",
        "only",
        "other",
        "over",
        "propose",
        "proposed",
        "proposes",
        "proposing",
        "provide",
        "provided",
        "provides",
        "review",
        "reviews",
        "studies",
        "study",
        "survey",
        "surveys",
        "their",
        "through",
        "toward",
        "towards",
        "under",
        "used",
        "using",
        "via",
        "without",
        "within",
        "been",
        "being",
    }
)
BridgeEvidenceProfile = Literal["standard", "negation", "unconstrained"]


@dataclass(frozen=True)
class CrossVocabularyBridgeProposal:
    """A Gold-blind lexical action derived only from online candidate text."""

    query_text: str
    anchors: tuple[str, ...]
    expansion_terms: tuple[str, ...]
    candidate_support: dict[str, int]
    action_support: dict[str, int]
    grounded_candidate_count: int
    online_candidate_count: int
    evidence_profile: BridgeEvidenceProfile = "standard"
    title_support: dict[str, int] = field(default_factory=dict)
    anchor_cooccurrence: dict[str, int] = field(default_factory=dict)
    anchor_support: dict[str, int] = field(default_factory=dict)


def _is_online_candidate(candidate: DocumentCandidateEvidence) -> bool:
    evidence = [*candidate.paper.sources, *candidate.source_ranks]
    return all("pasa" not in value.casefold() for value in evidence)


def _action_family(value: str) -> str:
    return value.split("@", 1)[0].casefold()


def propose_cross_vocabulary_bridge(
    query: str,
    candidates: Sequence[DocumentCandidateEvidence],
    *,
    max_candidates: int = 20,
    max_expansion_terms: int = 2,
    min_candidate_support: int = 3,
    min_action_support: int = 2,
    max_document_frequency_ratio: float = 0.60,
) -> CrossVocabularyBridgeProposal | None:
    """Propose one conservative bridge action without inspecting Gold labels.

    Eligible terms must be absent from the original query, occur in at least
    three grounded candidates, and be supported by at least two independent
    retrieval actions.  Very common local terms are suppressed to avoid broad
    pseudo-relevance-feedback drift.
    """

    return _propose_bridge(
        query,
        candidates,
        profile="standard",
        excluded_terms=frozenset(),
        generic_terms=frozenset(),
        suppressed_expansion_terms=frozenset(),
        max_candidates=max_candidates,
        max_expansion_terms=max_expansion_terms,
        min_candidate_support=min_candidate_support,
        min_action_support=min_action_support,
        max_document_frequency_ratio=max_document_frequency_ratio,
        min_title_support=1,
        min_anchor_cooccurrence=0,
        anchor_count=2,
        min_anchors_per_candidate=1,
    )


def propose_refined_cross_vocabulary_bridge(
    query: str,
    candidates: Sequence[DocumentCandidateEvidence],
    *,
    profile: Literal["negation", "unconstrained"],
    exclusions: Sequence[str] = (),
    suppressed_expansion_terms: Sequence[str] = (),
    max_candidates: int = 20,
    max_expansion_terms: int = 2,
) -> CrossVocabularyBridgeProposal | None:
    """Apply v2 generic-term suppression and anchor-conditioned evidence."""

    if profile == "negation":
        excluded_terms = frozenset(
            term for exclusion in exclusions for term in query_content_terms(exclusion)
        )
        if not excluded_terms:
            raise ValueError("negation bridge requires frozen exclusion terms")
        return _propose_bridge(
            query,
            candidates,
            profile=profile,
            excluded_terms=excluded_terms,
            generic_terms=_GENERIC_BRIDGE_TERMS,
            suppressed_expansion_terms=frozenset(
                term.casefold() for term in suppressed_expansion_terms
            ),
            max_candidates=max_candidates,
            max_expansion_terms=max_expansion_terms,
            min_candidate_support=3,
            min_action_support=2,
            # Negation exclusions already remove the forbidden concept and
            # functional cues.  A positive bridge repeated across every
            # grounded title is useful corroboration, not local drift.
            max_document_frequency_ratio=1.0,
            min_title_support=2,
            min_anchor_cooccurrence=2,
            anchor_count=3,
            min_anchors_per_candidate=2,
        )
    if exclusions:
        raise ValueError("unconstrained bridge must not receive exclusions")
    return _propose_bridge(
        query,
        candidates,
        profile=profile,
        excluded_terms=frozenset(),
        generic_terms=_GENERIC_BRIDGE_TERMS,
        suppressed_expansion_terms=frozenset(
            term.casefold() for term in suppressed_expansion_terms
        ),
        max_candidates=max_candidates,
        max_expansion_terms=max_expansion_terms,
        min_candidate_support=4,
        min_action_support=2,
        # Retain the v1 local-DF guard, while allowing a four-of-seven term
        # that also satisfies the stricter title/action/anchor evidence.
        max_document_frequency_ratio=0.60,
        min_title_support=2,
        min_anchor_cooccurrence=3,
        anchor_count=3,
        min_anchors_per_candidate=2,
    )


def _propose_bridge(
    query: str,
    candidates: Sequence[DocumentCandidateEvidence],
    *,
    profile: BridgeEvidenceProfile,
    excluded_terms: frozenset[str],
    generic_terms: frozenset[str],
    suppressed_expansion_terms: frozenset[str],
    max_candidates: int,
    max_expansion_terms: int,
    min_candidate_support: int,
    min_action_support: int,
    max_document_frequency_ratio: float,
    min_title_support: int,
    min_anchor_cooccurrence: int,
    anchor_count: int,
    min_anchors_per_candidate: int,
) -> CrossVocabularyBridgeProposal | None:
    if not query.strip():
        raise ValueError("query must be non-empty")
    if max_candidates <= 0 or max_expansion_terms <= 0 or anchor_count <= 0:
        raise ValueError("bridge candidate, expansion, and anchor bounds must be positive")
    if min_candidate_support <= 0 or min_action_support <= 0 or min_title_support <= 0:
        raise ValueError("bridge support bounds must be positive")
    if min_anchor_cooccurrence < 0 or min_anchors_per_candidate <= 0:
        raise ValueError("anchor co-occurrence bound must be non-negative")
    if not 0 < max_document_frequency_ratio <= 1:
        raise ValueError("document-frequency ratio must be in (0, 1]")

    query_terms = query_content_terms(query)
    original_query_set = set(query_terms)
    positive_query_terms = [
        term
        for term in query_terms
        if term not in generic_terms and term not in excluded_terms
    ]
    if not positive_query_terms:
        return None
    query_set = set(positive_query_terms)
    online = [candidate for candidate in candidates if _is_online_candidate(candidate)]
    online.sort(
        key=lambda candidate: (
            -candidate.baseline_score,
            candidate.paper.canonical_id.casefold(),
        )
    )
    selected = online[:max_candidates]
    minimum_overlap = min(2, len(query_set))
    grounded: list[
        tuple[DocumentCandidateEvidence, set[str], set[str], set[str]]
    ] = []
    for candidate in selected:
        title_terms = set(query_content_terms(candidate.paper.title))
        abstract_terms = set(query_content_terms(candidate.paper.abstract or ""))
        document_terms = title_terms | abstract_terms
        if len(query_set & document_terms) < minimum_overlap:
            continue
        action_families = {_action_family(value) for value in candidate.source_ranks}
        grounded.append((candidate, title_terms, document_terms, action_families))
    if len(grounded) < min_candidate_support:
        return None

    query_document_frequency = {
        term: sum(
            term in document_terms
            for _candidate, _title, document_terms, _actions in grounded
        )
        for term in positive_query_terms
    }
    if profile == "standard":
        anchored = sorted(
            enumerate(positive_query_terms),
            key=lambda item: (
                query_document_frequency[item[1]] == 0,
                query_document_frequency[item[1]],
                item[0],
            ),
        )
    else:
        anchored = sorted(
            (
                (index, term)
                for index, term in enumerate(positive_query_terms)
                if query_document_frequency[term] >= 2
            ),
            key=lambda item: (-query_document_frequency[item[1]], item[0]),
        )
        required_anchor_count = min(anchor_count, len(positive_query_terms))
        if len(anchored) < required_anchor_count:
            return None
    anchors = tuple(
        term for _index, term in anchored[: min(anchor_count, len(anchored))]
    )
    anchor_set = set(anchors)

    term_candidates: dict[str, set[str]] = defaultdict(set)
    term_actions: dict[str, set[str]] = defaultdict(set)
    term_title_candidates: dict[str, set[str]] = defaultdict(set)
    term_anchor_candidates: dict[str, set[str]] = defaultdict(set)
    for candidate, title_terms, document_terms, action_families in grounded:
        paper_id = candidate.paper.canonical_id.casefold()
        bridge_terms = (
            document_terms
            - original_query_set
            - excluded_terms
            - generic_terms
            - suppressed_expansion_terms
        )
        for term in bridge_terms:
            term_candidates[term].add(paper_id)
            term_actions[term].update(action_families)
            if term in title_terms:
                term_title_candidates[term].add(paper_id)
            if len(anchor_set & document_terms) >= min_anchors_per_candidate:
                term_anchor_candidates[term].add(paper_id)

    grounded_count = len(grounded)
    eligible: list[tuple[float, str]] = []
    for term, paper_ids in term_candidates.items():
        support = len(paper_ids)
        action_support = len(term_actions[term])
        title_support = len(term_title_candidates[term])
        anchor_cooccurrence = len(term_anchor_candidates[term])
        frequency_ratio = support / grounded_count
        if (
            support < min_candidate_support
            or action_support < min_action_support
            or title_support < min_title_support
            or anchor_cooccurrence < min_anchor_cooccurrence
            or frequency_ratio > max_document_frequency_ratio
        ):
            continue
        local_idf = math.log((grounded_count + 1) / (support + 1)) + 1.0
        title_fraction = title_support / support
        diversity = 1.0 + 0.15 * min(action_support - 1, 4)
        score = math.sqrt(support) * local_idf * diversity * (0.75 + 0.25 * title_fraction)
        if profile != "standard":
            anchor_fraction = anchor_cooccurrence / support
            score *= 0.65 + 0.35 * anchor_fraction
        eligible.append((score, term))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    expansion_terms = tuple(term for _score, term in eligible[:max_expansion_terms])
    if not expansion_terms:
        return None

    return CrossVocabularyBridgeProposal(
        query_text=" ".join([*anchors, *expansion_terms]),
        anchors=anchors,
        expansion_terms=expansion_terms,
        candidate_support={term: len(term_candidates[term]) for term in expansion_terms},
        action_support={term: len(term_actions[term]) for term in expansion_terms},
        grounded_candidate_count=grounded_count,
        online_candidate_count=len(selected),
        evidence_profile=profile,
        title_support={
            term: len(term_title_candidates[term]) for term in expansion_terms
        },
        anchor_cooccurrence={
            term: len(term_anchor_candidates[term]) for term in expansion_terms
        },
        anchor_support={
            term: query_document_frequency[term] for term in anchors
        },
    )


def build_cross_vocabulary_action_batch(
    proposal: CrossVocabularyBridgeProposal,
) -> RecallActionBatch:
    """Turn a frozen proposal into exactly one lexical OpenAlex action."""

    return RecallActionBatch(
        actions=[
            TextSearchAction(
                action_id=_ACTION_ID,
                strategy=_STRATEGY,
                action_type="text_search",
                payload=TextSearchPayload(
                    query_text=proposal.query_text,
                    search_mode="lexical",
                ),
            )
        ]
    )


def build_refined_cross_vocabulary_action_batch(
    proposal: CrossVocabularyBridgeProposal,
) -> RecallActionBatch:
    """Turn a refined v2 proposal into exactly one lexical OpenAlex action."""

    if proposal.evidence_profile not in {"negation", "unconstrained"}:
        raise ValueError("refined action requires a refined evidence profile")
    return RecallActionBatch(
        actions=[
            TextSearchAction(
                action_id=_REFINED_ACTION_ID,
                strategy=_REFINED_STRATEGY,
                action_type="text_search",
                payload=TextSearchPayload(
                    query_text=proposal.query_text,
                    search_mode="lexical",
                ),
            )
        ]
    )


def select_production_cross_vocabulary_supplement(
    query_spec: QuerySpec,
    candidates: Sequence[DocumentCandidateEvidence],
) -> RecallActionBatch:
    """Schedule one bounded supplement from the existing LLM query context."""

    local_profile = profile_query_constraints(query_spec)
    if {"negation", "title_like"}.intersection(local_profile.labels):
        return RecallActionBatch(actions=[])
    has_structured_constraint = bool(
        query_spec.datasets
        or query_spec.methods
        or query_spec.tasks
        or query_spec.venues
        or query_spec.must_have
        or query_spec.year_from is not None
        or query_spec.year_to is not None
    )
    if has_structured_constraint:
        structured = propose_structured_openalex_supplement(query_spec)
        if structured is None:
            return RecallActionBatch(actions=[])
        return build_structured_openalex_action_batch(structured)
    proposal = propose_refined_cross_vocabulary_bridge(
        query_spec.original_query,
        candidates,
        profile="unconstrained",
        suppressed_expansion_terms=_PRODUCTION_SUPPRESSED_EXPANSION_TERMS,
    )
    if proposal is None:
        return RecallActionBatch(actions=[])
    return build_refined_cross_vocabulary_action_batch(proposal)


__all__ = [
    "CrossVocabularyBridgeProposal",
    "build_cross_vocabulary_action_batch",
    "build_refined_cross_vocabulary_action_batch",
    "propose_cross_vocabulary_bridge",
    "propose_refined_cross_vocabulary_bridge",
    "select_production_cross_vocabulary_supplement",
]
