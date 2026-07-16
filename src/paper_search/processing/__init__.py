"""Paper normalization and deduplication helpers."""

from paper_search.processing.deduplicate import (
    DeduplicationResult,
    MergeDecision,
    deduplicate_papers,
)
from paper_search.processing.filter import (
    AcceptedPaper,
    FilterResult,
    RejectedPaper,
    apply_hard_filters,
)
from paper_search.processing.normalize import normalize_openalex_work, reconstruct_abstract


__all__ = [
    "AcceptedPaper",
    "DeduplicationResult",
    "FilterResult",
    "MergeDecision",
    "RejectedPaper",
    "apply_hard_filters",
    "deduplicate_papers",
    "normalize_openalex_work",
    "reconstruct_abstract",
]
