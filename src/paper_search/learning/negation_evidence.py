"""Deterministic stance evidence for explicit query exclusions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from paper_search.learning.candidates import query_content_terms


ExclusionStance = Literal["clean", "conflict", "unknown"]
NEGATION_EVIDENCE_SCHEMA_VERSION = "negation-stance-v6-conflict-topic-anchors"
_BOUNDARY_PUNCTUATION = " \t\r\n,.;:!?\"'`()[]{}"
_QUERY_CONTROL_TERMS = frozenset(
    {
        "based",
        "exclude",
        "excluding",
        "learning",
        "studied",
        "use",
        "used",
        "uses",
        "using",
        "without",
        "works",
    }
)
_GENERIC_ACADEMIC_TOPIC_TERMS = frozenset(
    {
        "approach",
        "architecture",
        "bound",
        "classical",
        "context",
        "derive",
        "estimate",
        "example",
        "first",
        "framework",
        "generalization",
        "involve",
        "learning",
        "lower",
        "method",
        "model",
        "neural",
        "prediction",
        "result",
        "setting",
        "system",
        "task",
    }
)
_SHORT_ACRONYM = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}(?![A-Za-z0-9])")


def _phrase_pattern(value: str) -> str:
    return r"\s+".join(re.escape(part) for part in value.split())


def classify_exclusion_stance(text: str, exclusion: str) -> ExclusionStance:
    """Classify explicit use/non-use statements; neutral mentions stay unknown."""

    normalized_text = " ".join(text.casefold().split())
    normalized_exclusion = " ".join(
        exclusion.casefold().strip(_BOUNDARY_PUNCTUATION).split()
    )
    if not normalized_text or not normalized_exclusion:
        return "unknown"
    phrase = _phrase_pattern(normalized_exclusion)
    bridge = (
        r"(?:(?!(?:not|without|no|less|fewer|minimal)\b)"
        r"[\w+_.-]+\s+){0,3}"
    )
    clean_patterns = (
        rf"\bwithout\s+{bridge}{phrase}(?![\w])",
        rf"\b(?:avoiding|avoids?|excluding|excludes?|instead\s+of|rather\s+than)"
        rf"\s+{bridge}{phrase}(?![\w])",
        rf"\bno\s+{bridge}{phrase}(?![\w])"
        rf"(?:\s+(?:is|are|was|were)\s+(?:used|employed|applied|required))?",
        rf"\b(?:do|does|did|is|are|was|were|will)\s+not\s+"
        rf"(?:use|uses|used|employ|employs|employed|include|includes|included|"
        rf"incorporate|incorporates|incorporated|rely\s+on)\s+{bridge}{phrase}"
        rf"(?![\w])",
        rf"(?<![\w]){phrase}(?![\w])\s+(?:is|are|was|were)\s+not\s+"
        rf"(?:used|employed|applied|included|required)",
    )
    if any(re.search(pattern, normalized_text) for pattern in clean_patterns):
        return "clean"
    own_subject = (
        r"(?:we|our\s+(?:(?:new|proposed)\s+)?"
        r"(?:method|model|approach|framework|system|experiments?)|"
        r"this\s+(?:method|model|approach|framework|system|study|work))"
    )
    subject_bridge = (
        r"(?:(?:also|directly|explicitly|successfully|then|can|could|do|does|"
        r"did|will|would|have|has|had|is|are|was|were)\s+){0,3}"
    )
    use_cue = (
        r"(?:using|uses?|used|employing|employs?|employed|with|via|through|"
        r"adopting|adopts?|adopted|incorporating|incorporates?|incorporated|"
        r"leveraging|leverages?|leveraged)"
    )
    title_prefix = (
        r"^(?![^.!?]{0,120}\b(?:prior|previous|existing|earlier|related|"
        r"conventional)\b)[^.!?]{0,120}?\b"
    )
    conflict_patterns = (
        rf"{own_subject}\s+{subject_bridge}{use_cue}\s+{bridge}{phrase}(?![\w])",
        rf"{own_subject}\s+{subject_bridge}(?:based|relying)\s+on\s+{bridge}{phrase}"
        rf"(?![\w])",
        rf"\bwe\s+(?:propose|present|introduce|develop)\s+"
        rf"(?:(?:a|an|the|our)\s+)?(?:method|model|approach|framework|system)\s+"
        rf"{use_cue}\s+{bridge}{phrase}(?![\w])",
        rf"{title_prefix}(?:using|employing|adopting|incorporating|leveraging)"
        rf"\s+{bridge}{phrase}(?![\w])",
        rf"{title_prefix}with\s+{phrase}(?![\w])",
        rf"(?<![\w]){phrase}(?![\w])\s+(?:is|are|was|were)\s+"
        rf"(?:used|employed|applied|adopted|incorporated)\s+"
        rf"(?:in|by)\s+(?:our|this)\b",
    )
    if any(re.search(pattern, normalized_text) for pattern in conflict_patterns):
        return "conflict"
    return "unknown"


def negation_evidence_fractions(
    text: str, exclusions: Sequence[str]
) -> tuple[float, float]:
    """Return explicit conflict and explicit clean fractions over exclusions."""

    normalized = tuple(
        dict.fromkeys(
            " ".join(value.casefold().strip(_BOUNDARY_PUNCTUATION).split())
            for value in exclusions
            if value.strip()
        )
    )
    if not normalized:
        return 0.0, 0.0
    stances = [classify_exclusion_stance(text, value) for value in normalized]
    divisor = len(stances)
    return (
        sum(stance == "conflict" for stance in stances) / divisor,
        sum(stance == "clean" for stance in stances) / divisor,
    )


def negation_topic_relevant(
    query: str, text: str, exclusions: Sequence[str]
) -> bool:
    """Require conflict evidence to overlap the non-exclusion query topic."""

    ordered_topic_terms = negation_topic_terms(query, exclusions)
    topic_terms = set(ordered_topic_terms)
    if not topic_terms:
        return False
    evidence_text = _stance_evidence_text(text, exclusions)
    if not evidence_text:
        return False
    candidate_terms = set(_topic_terms(evidence_text))
    overlap = topic_terms.intersection(candidate_terms)
    distinctive = topic_terms.difference(_GENERIC_ACADEMIC_TOPIC_TERMS)
    distinctive_overlap = distinctive.intersection(candidate_terms)
    if len(distinctive) >= 2:
        return len(distinctive_overlap) >= 2
    return len(distinctive_overlap) == 1 and len(overlap) >= 2


def _topic_key(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def _topic_terms(value: str) -> tuple[str, ...]:
    regular = [_topic_key(term) for term in query_content_terms(value)]
    acronyms = [match.group(0).casefold() for match in _SHORT_ACRONYM.finditer(value)]
    return tuple(dict.fromkeys([*regular, *acronyms]))


def _stance_evidence_text(text: str, exclusions: Sequence[str]) -> str:
    segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        if segment.strip()
    ]
    return " ".join(
        segment
        for segment in segments
        if any(
            classify_exclusion_stance(segment, exclusion) != "unknown"
            for exclusion in exclusions
        )
    )


def negation_topic_terms(query: str, exclusions: Sequence[str]) -> tuple[str, ...]:
    """Return stable non-exclusion terms used by strict topical conflict search."""

    exclusion_terms = {
        term for value in exclusions for term in _topic_terms(value)
    }
    retained = [
        term
        for term in _topic_terms(query)
        if term not in exclusion_terms and term not in _QUERY_CONTROL_TERMS
    ]
    distinctive = [
        term for term in retained if term not in _GENERIC_ACADEMIC_TOPIC_TERMS
    ]
    generic = [term for term in retained if term in _GENERIC_ACADEMIC_TOPIC_TERMS]
    return tuple(dict.fromkeys([*distinctive, *generic]))


__all__ = [
    "ExclusionStance",
    "NEGATION_EVIDENCE_SCHEMA_VERSION",
    "classify_exclusion_stance",
    "negation_evidence_fractions",
    "negation_topic_terms",
    "negation_topic_relevant",
]
