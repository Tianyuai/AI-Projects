"""Gold-blind query-native anchors with cross-title phrase evidence."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from paper_search.domain.models import QuerySpec
from paper_search.learning.candidates import query_content_terms
from paper_search.learning.cpu_document_ranker import DocumentCandidateEvidence
from paper_search.learning.query_constraint_profile import profile_query_constraints
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    TextSearchAction,
    TextSearchPayload,
)


_ACTION_ID = "query-native-title-phrase-v3"
_STRATEGY = "candidate-family:query-native-title-phrase-v3"
_QUOTED_PHRASE = re.compile(r'["“”]([^"“”]{2,80})["“”]')
_PARENTHETICAL_ALIAS = re.compile(
    r"([A-Za-z][A-Za-z0-9+_. -]{2,60}?)\s*\(([A-Z][A-Za-z0-9+_.-]{1,15})\)"
)
_TECHNICAL_TOKEN = re.compile(
    r"\b(?:[A-Z][A-Z0-9]{1,}|[A-Za-z][A-Za-z0-9]*[-+_.][A-Za-z0-9+_.-]+|"
    r"[A-Za-z]+\d+[A-Za-z0-9-]*)\b"
)
_SPAN_TOKEN = re.compile(r"[a-z0-9]+")
_PHRASE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "using",
        "via",
        "with",
        "without",
    }
)
_GENERIC_PHRASE_TERMS = frozenset(
    {
        "analysis",
        "approach",
        "approaches",
        "based",
        "data",
        "deep",
        "framework",
        "frameworks",
        "learning",
        "method",
        "methods",
        "model",
        "models",
        "neural",
        "network",
        "networks",
        "paper",
        "papers",
        "study",
        "studies",
        "survey",
        "surveys",
        "using",
    }
)
_FUNCTIONAL_BOUNDARY_TERMS = frozenset(
    {
        "analysis",
        "approach",
        "approaches",
        "based",
        "review",
        "reviews",
        "survey",
        "surveys",
        "using",
        "via",
    }
)


@dataclass(frozen=True)
class QueryNativeTitlePhraseProposal:
    """One bounded action supported by query spans and online candidate titles."""

    query_text: str
    query_anchors: tuple[str, ...]
    supported_phrase: str
    phrase_candidate_support: int
    phrase_action_support: int
    phrase_anchor_cooccurrence: int
    grounded_candidate_count: int
    online_candidate_count: int


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _contains_sequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    width = len(needle)
    return bool(width) and any(
        list(haystack[index : index + width]) == list(needle)
        for index in range(len(haystack) - width + 1)
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalized(value).strip(" ,.;:?!")
        folded = normalized.casefold()
        if not normalized or folded in seen:
            continue
        seen.add(folded)
        result.append(normalized)
    return result


def _query_native_anchor_candidates(spec: QuerySpec) -> list[tuple[str, tuple[str, ...]]]:
    query_terms = query_content_terms(spec.original_query)
    explicit = _ordered_unique(
        [
            *spec.methods,
            *spec.datasets,
            *spec.tasks,
            *spec.must_have,
            *spec.topics,
            *spec.venues,
            *[match.group(1) for match in _QUOTED_PHRASE.finditer(spec.original_query)],
            *[
                value
                for match in _PARENTHETICAL_ALIAS.finditer(spec.original_query)
                for value in (match.group(1), match.group(2))
            ],
            *[match.group(0) for match in _TECHNICAL_TOKEN.finditer(spec.original_query)],
        ]
    )
    candidates: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = set()
    for value in explicit:
        tokens = tuple(query_content_terms(value))
        if not tokens or tokens in seen or not _contains_sequence(query_terms, tokens):
            continue
        seen.add(tokens)
        candidates.append((" ".join(tokens), tokens))
    for width in (4, 3, 2):
        for index in range(len(query_terms) - width + 1):
            tokens = tuple(query_terms[index : index + width])
            if tokens in seen or all(term in _GENERIC_PHRASE_TERMS for term in tokens):
                continue
            seen.add(tokens)
            candidates.append((" ".join(tokens), tokens))
    return candidates


def _is_online_candidate(candidate: DocumentCandidateEvidence) -> bool:
    evidence = [*candidate.paper.sources, *candidate.source_ranks]
    return bool(evidence) and all("pasa" not in value.casefold() for value in evidence)


def _action_families(candidate: DocumentCandidateEvidence) -> set[str]:
    return {value.split("@", 1)[0].casefold() for value in candidate.source_ranks}


def _ngrams(
    tokens: Sequence[str], *, minimum: int = 2, maximum: int = 3
) -> Iterable[tuple[str, ...]]:
    for width in range(maximum, minimum - 1, -1):
        for index in range(len(tokens) - width + 1):
            yield tuple(tokens[index : index + width])


def _contiguous_phrase_ngrams(value: str) -> Iterable[tuple[str, ...]]:
    tokens = _SPAN_TOKEN.findall(unicodedata.normalize("NFKC", value).casefold())
    for phrase in _ngrams(tokens):
        if any(term in _PHRASE_STOPWORDS for term in phrase):
            continue
        yield phrase


def _merge_supported_sequences(
    anchor: Sequence[str], phrase: Sequence[str]
) -> tuple[str, ...]:
    if _contains_sequence(phrase, anchor):
        return tuple(phrase)
    if _contains_sequence(anchor, phrase):
        return tuple(anchor)
    maximum = min(len(anchor), len(phrase))
    for width in range(maximum, 0, -1):
        if list(anchor[-width:]) == list(phrase[:width]):
            return (*anchor, *phrase[width:])
        if list(phrase[-width:]) == list(anchor[:width]):
            return (*phrase, *anchor[width:])
    return (*anchor, *phrase)


def propose_query_native_title_phrase_bridge(
    spec: QuerySpec,
    candidates: Sequence[DocumentCandidateEvidence],
    *,
    max_candidates: int = 20,
    min_phrase_candidate_support: int = 2,
    min_phrase_action_support: int = 2,
    max_phrase_document_frequency_ratio: float = 0.75,
) -> QueryNativeTitlePhraseProposal | None:
    """Build one query-native, title-supported phrase action or abstain."""

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if min_phrase_candidate_support <= 0 or min_phrase_action_support <= 0:
        raise ValueError("phrase support bounds must be positive")
    if not 0 < max_phrase_document_frequency_ratio <= 1:
        raise ValueError("phrase document-frequency ratio must be in (0, 1]")
    local_profile = profile_query_constraints(spec)
    if spec.exclusions or "negation" in local_profile.labels:
        return None

    query_terms = query_content_terms(spec.original_query)
    query_set = set(query_terms)
    anchor_candidates = _query_native_anchor_candidates(spec)
    if not query_terms or not anchor_candidates:
        return None

    online = [candidate for candidate in candidates if _is_online_candidate(candidate)]
    online.sort(
        key=lambda candidate: (
            -candidate.baseline_score,
            candidate.paper.canonical_id.casefold(),
        )
    )
    selected = online[:max_candidates]
    grounded: list[
        tuple[DocumentCandidateEvidence, tuple[str, ...], set[str]]
    ] = []
    for candidate in selected:
        title_terms = tuple(query_content_terms(candidate.paper.title))
        if len(query_set.intersection(title_terms)) < min(2, len(query_set)):
            continue
        grounded.append((candidate, title_terms, _action_families(candidate)))
    if len(grounded) < min_phrase_candidate_support:
        return None

    anchor_support: dict[tuple[str, ...], int] = {}
    anchor_display: dict[tuple[str, ...], str] = {}
    for display, tokens in anchor_candidates:
        support = sum(
            _contains_sequence(title_terms, tokens)
            for _candidate, title_terms, _actions in grounded
        )
        if support >= 2:
            anchor_support[tokens] = support
            anchor_display[tokens] = display
    if not anchor_support:
        return None
    anchor = min(
        anchor_support,
        key=lambda tokens: (-len(tokens), -anchor_support[tokens], anchor_display[tokens].casefold()),
    )

    original_ngrams = set(_contiguous_phrase_ngrams(spec.original_query))
    phrase_candidates: dict[tuple[str, ...], set[str]] = defaultdict(set)
    phrase_actions: dict[tuple[str, ...], set[str]] = defaultdict(set)
    phrase_anchor_candidates: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for candidate, title_terms, action_families in grounded:
        paper_id = candidate.paper.canonical_id.casefold()
        anchor_overlap = len(set(anchor).intersection(title_terms))
        for phrase in set(_contiguous_phrase_ngrams(candidate.paper.title)):
            if (
                phrase in original_ngrams
                or not set(phrase).difference(query_set)
                or all(term in _GENERIC_PHRASE_TERMS for term in phrase)
                or phrase[0] in _FUNCTIONAL_BOUNDARY_TERMS
                or phrase[-1] in _FUNCTIONAL_BOUNDARY_TERMS
                or anchor_overlap < min(2, len(anchor))
            ):
                continue
            phrase_candidates[phrase].add(paper_id)
            phrase_actions[phrase].update(action_families)
            phrase_anchor_candidates[phrase].add(paper_id)

    grounded_count = len(grounded)
    eligible_phrases: list[tuple[str, ...]] = []
    for phrase, paper_ids in phrase_candidates.items():
        support = len(paper_ids)
        action_support = len(phrase_actions[phrase])
        frequency_ratio = support / grounded_count
        if (
            support < min_phrase_candidate_support
            or action_support < min_phrase_action_support
            or frequency_ratio > max_phrase_document_frequency_ratio
        ):
            continue
        eligible_phrases.append(phrase)
    eligible_phrases = [
        phrase
        for phrase in eligible_phrases
        if not any(
            len(extension) > len(phrase)
            and _contains_sequence(extension, phrase)
            and phrase_candidates[extension] == phrase_candidates[phrase]
            and phrase_actions[extension] == phrase_actions[phrase]
            for extension in eligible_phrases
        )
    ]
    eligible: list[tuple[float, tuple[str, ...]]] = []
    for phrase in eligible_phrases:
        support = len(phrase_candidates[phrase])
        action_support = len(phrase_actions[phrase])
        score = support * 10.0 + action_support - 0.1 * len(phrase)
        eligible.append((score, phrase))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-item[0], " ".join(item[1])))
    phrase = eligible[0][1]

    action_terms = _merge_supported_sequences(anchor, phrase)
    return QueryNativeTitlePhraseProposal(
        query_text=" ".join(action_terms),
        query_anchors=(anchor_display[anchor],),
        supported_phrase=" ".join(phrase),
        phrase_candidate_support=len(phrase_candidates[phrase]),
        phrase_action_support=len(phrase_actions[phrase]),
        phrase_anchor_cooccurrence=len(phrase_anchor_candidates[phrase]),
        grounded_candidate_count=grounded_count,
        online_candidate_count=len(selected),
    )


def build_query_native_title_phrase_action_batch(
    proposal: QueryNativeTitlePhraseProposal,
) -> RecallActionBatch:
    """Turn a frozen proposal into exactly one lexical search action."""

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


__all__ = [
    "QueryNativeTitlePhraseProposal",
    "build_query_native_title_phrase_action_batch",
    "propose_query_native_title_phrase_bridge",
]
