"""Paper normalization and deduplication helpers."""

from paper_search.processing.deduplicate import (
    DEDUPLICATION_VERSION,
    DeduplicationResult,
    FUZZY_TITLE_THRESHOLD,
    MergeDecision,
    deduplicate_papers,
)
from paper_search.processing.filter import (
    AcceptedPaper,
    FILTERING_VERSION,
    FilterResult,
    MINIMUM_UNCERTAINTY_MULTIPLIER,
    RejectedPaper,
    UNCERTAINTY_REASON_MULTIPLIER,
    apply_hard_filters,
)
from paper_search.processing.normalize import normalize_openalex_work, reconstruct_abstract


__all__ = [
    "AcceptedPaper",
    "DEDUPLICATION_VERSION",
    "DeduplicationResult",
    "FilterResult",
    "FILTERING_VERSION",
    "FUZZY_TITLE_THRESHOLD",
    "MINIMUM_UNCERTAINTY_MULTIPLIER",
    "MergeDecision",
    "RejectedPaper",
    "UNCERTAINTY_REASON_MULTIPLIER",
    "apply_hard_filters",
    "deduplicate_papers",
    "normalize_openalex_work",
    "reconstruct_abstract",
]
