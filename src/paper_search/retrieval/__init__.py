"""External scholarly search providers."""

from paper_search.retrieval.openalex import OPENALEX_SELECT_FIELDS, OpenAlexProvider
from paper_search.retrieval.base import SearchProvider
from paper_search.retrieval.routing import RoutedSubquery, route_baseline_subqueries
from paper_search.retrieval.semantic_scholar import SemanticScholarProvider
from paper_search.retrieval.snapshot_adapters import (
    LiveCaptureSearchProvider,
    ReplaySearchProvider,
)


__all__ = [
    "OPENALEX_SELECT_FIELDS",
    "OpenAlexProvider",
    "LiveCaptureSearchProvider",
    "ReplaySearchProvider",
    "RoutedSubquery",
    "SearchProvider",
    "SemanticScholarProvider",
    "route_baseline_subqueries",
]
