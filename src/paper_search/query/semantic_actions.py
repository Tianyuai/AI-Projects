"""Gold-blind validation for LLM-generated retrieval expressions."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from paper_search.domain.models import QuerySpec, SubQuery


SEMANTIC_ACTION_PROMPT_VERSION = "query-analyze-semantic-actions-v2"
PROTECTED_ACTION_PROMPT_VERSION = "query-analyze-protected-actions-v3"
_MAX_ACTION_CHARACTERS = 180
_MAX_CONTENT_TERMS = 16
_MAX_UNSUPPORTED_NOVEL_CONTENT_TERMS = 3
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_FORBIDDEN_IDENTIFIER = re.compile(
    r"(?:https?://|www\.|\b10\.\d{4,9}/\S+|\bW\d{6,}\b|"
    r"\b(?:CorpusId|OpenAlex|S2)\s*[:=]\s*[A-Za-z0-9._-]+)",
    flags=re.IGNORECASE,
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "based",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "paper",
        "papers",
        "study",
        "studies",
        "the",
        "to",
        "using",
        "with",
        "without",
    }
)


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(_normalized(value)))


def _content_terms(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _tokens(value)
        if token not in _STOPWORDS and (len(token) >= 3 or any(ch.isdigit() for ch in token))
    )


def _concept_key(token: str) -> str:
    """Normalize only conservative English inflections, never free synonyms."""

    if len(token) <= 4 or any(ch.isdigit() for ch in token):
        return token
    if token.endswith("ually") and len(token) > 8:
        return token[:-5]
    if token.endswith("ieval") and len(token) > 7:
        return token[:-2]
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        return token[:-1]
    return token


def _concept_terms(value: str) -> tuple[str, ...]:
    return tuple(_concept_key(term) for term in _content_terms(value))


def _contains_sequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    width = len(needle)
    return bool(width) and any(
        tuple(haystack[index : index + width]) == tuple(needle)
        for index in range(len(haystack) - width + 1)
    )


def _supported_constraints(
    candidate: SubQuery,
    supported_terms: set[str],
) -> list[str]:
    return [
        constraint
        for constraint in candidate.target_constraints
        if (
            (terms := set(_concept_terms(constraint)))
            and len(terms.intersection(supported_terms)) * 2 >= len(terms)
        )
    ]


def _looks_like_named_entity(value: str) -> bool:
    return any(ch.isdigit() for ch in value) or sum(
        ch.isupper() for ch in value
    ) >= 2


def _explicit_hard_constraints(spec: QuerySpec) -> tuple[tuple[str, ...], ...]:
    """Return explicit methods, datasets, and conservatively named entities."""

    original = _concept_terms(spec.original_query)
    constraints: list[tuple[str, ...]] = []
    named_entities = [
        value for value in spec.must_have if _looks_like_named_entity(value)
    ]
    for value in [*spec.methods, *spec.datasets, *named_entities]:
        terms = _concept_terms(value)
        if terms and _contains_sequence(original, terms) and terms not in constraints:
            constraints.append(terms)
    return tuple(constraints)


def _contains_exclusion(spec: QuerySpec, text: str) -> bool:
    action_tokens = _tokens(text)
    return any(
        _contains_sequence(action_tokens, _tokens(exclusion))
        for exclusion in spec.exclusions
        if _tokens(exclusion)
    )


def preserves_explicit_hard_constraints(spec: QuerySpec, text: str) -> bool:
    """Require production-extracted hard phrases and reject explicit exclusions."""

    action_sequence = _concept_terms(text)
    return not _contains_exclusion(spec, text) and all(
        _contains_sequence(action_sequence, constraint)
        for constraint in _explicit_hard_constraints(spec)
    )


def _is_safe_semantic_action(
    spec: QuerySpec,
    candidate: SubQuery,
    *,
    soft_concept_evidence: set[str],
) -> bool:
    text = " ".join(candidate.text.split())
    if (
        not text
        or len(text) > _MAX_ACTION_CHARACTERS
        or candidate.action_type != "text_search"
        or candidate.query_type == "exact"
        or _FORBIDDEN_IDENTIFIER.search(text) is not None
        or _contains_exclusion(spec, text)
    ):
        return False
    action_sequence = _concept_terms(text)
    hard_constraints = _explicit_hard_constraints(spec)
    if any(
        not _contains_sequence(action_sequence, constraint)
        for constraint in hard_constraints
    ):
        return False
    action_terms = set(action_sequence)
    original_terms = set(_concept_terms(spec.original_query))
    supported_terms = original_terms.union(soft_concept_evidence)
    grounded = _supported_constraints(candidate, supported_terms)
    if not grounded:
        return False
    minimum_original_support = min(2, len(original_terms))
    if (
        len(action_terms.intersection(original_terms)) < minimum_original_support
        or len(action_terms) > _MAX_CONTENT_TERMS
        or len(action_terms.difference(supported_terms))
        > _MAX_UNSUPPORTED_NOVEL_CONTENT_TERMS
    ):
        return False
    return True


def filter_semantic_action_candidates(
    spec: QuerySpec,
    candidates: Iterable[SubQuery],
    *,
    soft_concept_evidence: Iterable[str] = (),
) -> list[SubQuery]:
    """Keep hard-safe actions whose soft rewrites have local frozen evidence."""

    if spec.exclusions:
        return []
    evidence_terms = {
        term
        for value in soft_concept_evidence
        for term in _concept_terms(value)
    }
    original = _normalized(spec.original_query)
    accepted: list[SubQuery] = []
    for candidate in candidates:
        text = _normalized(candidate.text)
        if text == original:
            if (
                candidate.action_type == "text_search"
                and candidate.search_mode == "semantic"
            ):
                accepted.append(candidate)
            continue
        if _is_safe_semantic_action(
            spec,
            candidate,
            soft_concept_evidence=evidence_terms,
        ):
            accepted.append(candidate)
    return accepted


def filter_lexical_action_candidates(
    spec: QuerySpec,
    candidates: Iterable[SubQuery],
    *,
    soft_concept_evidence: Iterable[str] = (),
) -> list[SubQuery]:
    """Keep bounded lexical rewrites while leaving exclusions as downstream constraints."""

    evidence_terms = {
        term
        for value in soft_concept_evidence
        for term in _concept_terms(value)
    }
    original = _normalized(spec.original_query)
    original_terms = set(_concept_terms(spec.original_query))
    supported_terms = original_terms.union(evidence_terms)
    accepted: list[SubQuery] = []
    for candidate in candidates:
        text = " ".join(candidate.text.split())
        normalized = _normalized(text)
        if (
            not text
            or normalized == original
            or len(text) > _MAX_ACTION_CHARACTERS
            or candidate.action_type != "text_search"
            or candidate.search_mode != "lexical"
            or candidate.query_type == "exact"
            or _FORBIDDEN_IDENTIFIER.search(text) is not None
            or _contains_exclusion(spec, text)
        ):
            continue
        action_sequence = _concept_terms(text)
        if any(
            not _contains_sequence(action_sequence, constraint)
            for constraint in _explicit_hard_constraints(spec)
        ):
            continue
        action_terms = set(action_sequence)
        if (
            not _supported_constraints(candidate, supported_terms)
            or len(action_terms.intersection(original_terms)) < min(2, len(original_terms))
            or len(action_terms) > _MAX_CONTENT_TERMS
            or len(action_terms.difference(supported_terms))
            > _MAX_UNSUPPORTED_NOVEL_CONTENT_TERMS
        ):
            continue
        accepted.append(candidate)
    return accepted


__all__ = [
    "PROTECTED_ACTION_PROMPT_VERSION",
    "SEMANTIC_ACTION_PROMPT_VERSION",
    "filter_lexical_action_candidates",
    "filter_semantic_action_candidates",
    "preserves_explicit_hard_constraints",
]
