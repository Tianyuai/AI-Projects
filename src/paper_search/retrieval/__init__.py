"""External scholarly search providers."""

from paper_search.retrieval.openalex import OPENALEX_SELECT_FIELDS, OpenAlexProvider
from paper_search.retrieval.base import SearchProvider
from paper_search.retrieval.semantic_scholar import SemanticScholarProvider


__all__ = [
    "OPENALEX_SELECT_FIELDS",
    "OpenAlexProvider",
    "SearchProvider",
    "SemanticScholarProvider",
]
