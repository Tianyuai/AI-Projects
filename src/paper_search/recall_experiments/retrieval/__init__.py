"""Explicit retrieval action registration and provider adapters."""

from paper_search.recall_experiments.retrieval.backends import (
    BackendCitationResult,
    BackendSearchResult,
    BudgetedCitationBackend,
    BudgetedSearchBackend,
    CitationActionHandler,
    CitationBackend,
    SearchActionHandler,
    SearchBackend,
)
from paper_search.recall_experiments.retrieval.registry import RetrievalActionRegistry

__all__ = [
    "BackendCitationResult",
    "BackendSearchResult",
    "BudgetedCitationBackend",
    "BudgetedSearchBackend",
    "CitationActionHandler",
    "CitationBackend",
    "RetrievalActionRegistry",
    "SearchActionHandler",
    "SearchBackend",
]
