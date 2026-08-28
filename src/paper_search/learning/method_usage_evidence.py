"""Deterministic evidence that a candidate itself uses a requested method."""

from __future__ import annotations

import re
from collections.abc import Sequence


METHOD_USAGE_EVIDENCE_SCHEMA_VERSION = "method-usage-v1-affirmative-own-use"
LEGACY_METHOD_TEXT_MATCH_SCHEMA_VERSION = "method-text-match-v0-exact-mention"
_BOUNDARY_PUNCTUATION = " \t\r\n,.;:!?\"'`()[]{}"


def _phrase_pattern(value: str) -> str:
    return r"\s+".join(re.escape(part) for part in value.split())


def _segments(value: str) -> tuple[str, ...]:
    return tuple(
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|[\r\n]+", value)
        if segment.strip()
    )


def _title_affirms_method_use(title: str, method: str) -> bool:
    normalized = " ".join(title.casefold().split())
    if not normalized:
        return False
    phrase = _phrase_pattern(method)
    if re.search(
        r"\b(?:survey|review|comparison|benchmark|prior|previous|existing|"
        r"earlier|related|conventional)\b",
        normalized,
    ):
        return False
    bridge = r"(?:(?:[\w+_.-]+)\s+){0,4}"
    return bool(
        re.search(
            rf"\b(?:using|employing|adopting|incorporating|leveraging|with|via|"
            rf"through|based\s+on)\s+{bridge}{phrase}(?![\w])",
            normalized,
        )
        or re.search(rf"(?<![\w]){phrase}(?![\w])[- ]based\b", normalized)
    )


def _abstract_affirms_method_use(abstract: str, method: str) -> bool:
    phrase = _phrase_pattern(method)
    bridge = (
        r"(?:(?:also|directly|explicitly|successfully|then|can|could|do|does|"
        r"did|will|would|have|has|had|is|are|was|were)\s+){0,3}"
    )
    use_cue = (
        r"(?:using|uses?|used|employing|employs?|employed|adopting|adopts?|"
        r"adopted|incorporating|incorporates?|incorporated|leveraging|"
        r"leverages?|leveraged|relying\s+on|based\s+on)"
    )
    own_subject = (
        r"(?:we|our\s+(?:(?:new|proposed)\s+)?"
        r"(?:method|model|approach|framework|system|study|work)|"
        r"this\s+(?:method|model|approach|framework|system|study|work))"
    )
    for raw_segment in _segments(abstract):
        segment = " ".join(raw_segment.casefold().split())
        if not re.search(rf"(?<![\w]){phrase}(?![\w])", segment):
            continue
        affirmative_patterns = (
            rf"{own_subject}\s+{bridge}{use_cue}\s+"
            rf"(?:(?:[\w+_.-]+)\s+){{0,4}}{phrase}(?![\w])",
            rf"\bwe\s+(?:propose|present|introduce|develop)\s+[^.!?]{{0,120}}?"
            rf"{use_cue}\s+(?:(?:[\w+_.-]+)\s+){{0,4}}{phrase}(?![\w])",
            rf"(?<![\w]){phrase}(?![\w])\s+(?:is|are|was|were)\s+"
            rf"(?:used|employed|applied|adopted|incorporated)\s+"
            rf"(?:in|by)\s+(?:our|this)\b",
        )
        if any(re.search(pattern, segment) for pattern in affirmative_patterns):
            return True
        background_patterns = (
            rf"\b(?:prior|previous|existing|earlier|related|conventional|baseline)"
            rf"\b[^.!?]{{0,120}}?(?<![\w]){phrase}(?![\w])",
            rf"\b(?:such\s+as|e\.g\.|including)\s+{phrase}(?![\w])",
            rf"\b(?:unlike|versus|vs\.?|rather\s+than|instead\s+of|"
            rf"as\s+opposed\s+to|compared\s+(?:with|to))\s+"
            rf"(?:(?:[\w+_.-]+)\s+){{0,3}}{phrase}(?![\w])",
            rf"\b(?:survey|review|comparison)\b[^.!?]{{0,120}}?"
            rf"(?<![\w]){phrase}(?![\w])",
        )
        if any(re.search(pattern, segment) for pattern in background_patterns):
            continue
    return False


def method_usage_evidence_fraction(
    title: str,
    abstract: str,
    methods: Sequence[str],
) -> float:
    """Return the fraction of requested methods with affirmative own-use evidence."""

    normalized = tuple(
        dict.fromkeys(
            " ".join(method.casefold().strip(_BOUNDARY_PUNCTUATION).split())
            for method in methods
            if method.strip()
        )
    )
    if not normalized:
        return 0.0
    matched = sum(
        _title_affirms_method_use(title, method)
        or _abstract_affirms_method_use(abstract, method)
        for method in normalized
    )
    return matched / len(normalized)


__all__ = [
    "LEGACY_METHOD_TEXT_MATCH_SCHEMA_VERSION",
    "METHOD_USAGE_EVIDENCE_SCHEMA_VERSION",
    "method_usage_evidence_fraction",
]
