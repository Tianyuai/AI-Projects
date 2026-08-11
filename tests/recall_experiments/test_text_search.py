from __future__ import annotations

import asyncio
import inspect

import pytest

from paper_search.domain.models import Paper, UsageActual
from paper_search.recall_experiments.contracts import (
    RetrievalActionResult,
    RetrievalExecutionContext,
    TextSearchAction,
    TextSearchPayload,
    TitleSearchAction,
    TitleSearchPayload,
)
from paper_search.recall_experiments.retrieval.backends import BackendSearchResult
from paper_search.recall_experiments.retrieval.text_search import TextSearchHandler


class FakeSearchBackend:
    def __init__(self, result: BackendSearchResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str, dict[str, object], int]] = []

    async def search(
        self,
        action_id: str,
        query: str,
        filters: dict[str, object],
        limit: int,
    ) -> BackendSearchResult:
        self.calls.append((action_id, query, filters, limit))
        return self.result


def test_text_search_passes_validated_query_filters_and_limit_without_changing_rank() -> None:
    ranked_hits = [
        Paper(canonical_id="paper-2", title="Provider rank two"),
        Paper(canonical_id="paper-1", title="Provider rank one"),
    ]
    backend = FakeSearchBackend(
        BackendSearchResult(
            hits=ranked_hits,
            usage=UsageActual(search_api_calls=1),
            provenance={"provider": "offline-search"},
        )
    )
    handler = TextSearchHandler(backend=backend)
    action = TextSearchAction(
        action_id="text-1",
        strategy="validated query",
        action_type="text_search",
        payload=TextSearchPayload(query_text="normalized query"),
    )
    context = RetrievalExecutionContext(
        query_id="query-1",
        provider_filters={"year_from": 2020},
        max_results_per_action=7,
    )

    observed = asyncio.run(handler.execute(action, context))

    assert backend.calls == [
        ("text-1", "normalized query", {"year_from": 2020}, 7)
    ]
    assert isinstance(observed, RetrievalActionResult)
    assert observed.action_id == "text-1"
    assert observed.action_type == "text_search"
    assert observed.hits == ranked_hits
    assert observed.usage == UsageActual(search_api_calls=1)
    assert observed.provenance == {"provider": "offline-search"}


def test_text_search_module_has_no_recall_filtering_or_handler_dependencies() -> None:
    import paper_search.recall_experiments.retrieval.text_search as module

    source = inspect.getsource(module)

    assert "title_search" not in source
    assert "citation_expand" not in source
    assert "recall(" not in source
    assert ".filter(" not in source


def test_text_search_rejects_a_non_text_action_without_calling_the_backend() -> None:
    backend = FakeSearchBackend(BackendSearchResult())
    handler = TextSearchHandler(backend=backend)
    action = TitleSearchAction(
        action_id="title-1",
        strategy="wrong handler",
        action_type="title_search",
        payload=TitleSearchPayload(title_text="Validated title"),
    )
    context = RetrievalExecutionContext(query_id="query-1", max_results_per_action=3)

    with pytest.raises(TypeError, match="text search"):
        asyncio.run(handler.execute(action, context))

    assert backend.calls == []
