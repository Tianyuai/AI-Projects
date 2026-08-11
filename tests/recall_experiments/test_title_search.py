from __future__ import annotations

import asyncio
import inspect

import pytest

from paper_search.domain.models import Paper
from paper_search.recall_experiments.contracts import (
    RetrievalExecutionContext,
    TextSearchAction,
    TextSearchPayload,
    TitleSearchAction,
    TitleSearchPayload,
)
from paper_search.recall_experiments.retrieval.backends import BackendSearchResult
from paper_search.recall_experiments.retrieval.title_search import TitleSearchHandler


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


def test_title_search_uses_the_complete_validated_title_as_a_normal_query() -> None:
    ranked_hits = [
        Paper(canonical_id="paper-2", title="Provider rank two"),
        Paper(canonical_id="paper-1", title="Provider rank one"),
    ]
    backend = FakeSearchBackend(
        BackendSearchResult(
            hits=ranked_hits,
            provenance={"provider": "offline-search"},
        )
    )
    handler = TitleSearchHandler(backend=backend)
    action = TitleSearchAction(
        action_id="title-1",
        strategy="validated title",
        action_type="title_search",
        payload=TitleSearchPayload(title_text="Complete Validated Paper Title"),
    )
    context = RetrievalExecutionContext(
        query_id="query-1",
        provider_filters={"year_to": 2024},
        max_results_per_action=3,
    )

    observed = asyncio.run(handler.execute(action, context))

    assert backend.calls == [
        ("title-1", "Complete Validated Paper Title", {"year_to": 2024}, 3)
    ]
    assert observed.action_type == "title_search"
    assert observed.hits == ranked_hits
    assert observed.provenance == {"provider": "offline-search"}


def test_title_search_does_not_depend_on_raw_llm_title_extraction_or_other_handlers() -> None:
    import paper_search.recall_experiments.retrieval.title_search as module

    source = inspect.getsource(module)

    assert "extract_title_candidates" not in source
    assert "text_search" not in source
    assert "citation_expand" not in source


def test_title_search_rejects_a_non_title_action_without_calling_the_backend() -> None:
    backend = FakeSearchBackend(BackendSearchResult())
    handler = TitleSearchHandler(backend=backend)
    action = TextSearchAction(
        action_id="text-1",
        strategy="wrong handler",
        action_type="text_search",
        payload=TextSearchPayload(query_text="Validated text"),
    )
    context = RetrievalExecutionContext(query_id="query-1", max_results_per_action=3)

    with pytest.raises(TypeError, match="title search"):
        asyncio.run(handler.execute(action, context))

    assert backend.calls == []
