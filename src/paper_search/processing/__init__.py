"""Paper normalization and deduplication helpers."""

from paper_search.processing.deduplicate import (
    DeduplicationResult,
    MergeDecision,
    deduplicate_papers,
)
from paper_search.processing.normalize import normalize_openalex_work, reconstruct_abstract


__all__ = [
    "DeduplicationResult",
    "MergeDecision",
    "deduplicate_papers",
    "normalize_openalex_work",
    "reconstruct_abstract",
]
