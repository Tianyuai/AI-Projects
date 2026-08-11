from __future__ import annotations

import asyncio
import inspect

import pytest

from paper_search.domain.models import ErrorDetail, Paper, UsageActual
from paper_search.recall_experiments.contracts import (
    CitationExpandAction,
    CitationExpandPayload,
    RetrievalActionHandler,
    RetrievalExecutionContext,
    SeedCandidate,
    TextSearchAction,
    TextSearchPayload,
)
from paper_search.recall_experiments.retrieval.backends import (
    BackendCitationResult,
    BackendSearchResult,
)
from paper_search.recall_experiments.retrieval.citation_expand import CitationExpandHandler
from paper_search.recall_experiments.retrieval.registry import RetrievalActionRegistry
from paper_search.recall_experiments.retrieval.text_search import TextSearchHandler
from paper_search.recall_experiments.retrieval.title_search import TitleSearchHandler


class FakeCitationBackend:
    def __init__(self, result: BackendCitationResult) -> None:
        self.result = result
        self.calls: list[tuple[str, Paper, str, int]] = []

    async def expand(
        self,
        action_id: str,
        seed: Paper,
        direction: str,
        limit: int,
    ) -> BackendCitationResult:
        self.calls.append((action_id, seed, direction, limit))
        return self.result


class FakeSearchBackend:
    async def search(
        self,
        action_id: str,
        query: str,
        filters: dict[str, object],
        limit: int,
    ) -> BackendSearchResult:
        return BackendSearchResult()


def _seed(*, semantic_scholar_id: str | None = "S2:seed") -> Paper:
    return Paper(
        canonical_id="seed-1",
        title="Frozen seed",
        semantic_scholar_id=semantic_scholar_id,
        sources=["semantic_scholar"],
    )


def _expanded() -> Paper:
    return Paper(
        canonical_id="expanded-1",
        title="Expanded neighbor",
        semantic_scholar_id="S2:expanded",
        sources=["semantic_scholar"],
    )


def _context(seed: Paper) -> RetrievalExecutionContext:
    return RetrievalExecutionContext(
        query_id="query-1",
        max_results_per_action=9,
        seed_candidates=[SeedCandidate(paper=seed)],
    )


def _action(*, direction: str = "references", seed_id: str = "seed-1") -> CitationExpandAction:
    return CitationExpandAction(
        action_id="cite-1",
        strategy="frozen seed expansion",
        action_type="citation_expand",
        payload=CitationExpandPayload(
            seed_canonical_id=seed_id,
            direction=direction,
            limit=4,
        ),
    )


@pytest.mark.parametrize("direction", ["references", "citations", "both"])
def test_citation_expand_delegates_the_selected_frozen_seed_direction_and_limit(
    direction: str,
) -> None:
    seed = _seed()
    neighbor = _expanded()
    backend = FakeCitationBackend(
        BackendCitationResult(
            direction=direction,
            hits=[neighbor],
            usage=UsageActual(search_api_calls=1),
            provenance={"provider": "semantic_scholar"},
        )
    )
    handler = CitationExpandHandler(backend=backend)

    observed = asyncio.run(handler.execute(_action(direction=direction), _context(seed)))

    assert backend.calls == [("cite-1", seed, direction, 4)]
    assert observed.hits == [seed, neighbor]
    assert observed.provenance == {
        "provider": "semantic_scholar",
        "seed": "frozen",
        "expanded": "semantic_scholar",
    }


def test_citation_expand_rejects_an_unknown_non_frozen_seed_without_calling_the_backend() -> None:
    backend = FakeCitationBackend(BackendCitationResult(direction="references"))
    handler = CitationExpandHandler(backend=backend)

    observed = asyncio.run(handler.execute(_action(seed_id="newly-found"), _context(_seed())))

    assert backend.calls == []
    assert observed.hits == []
    assert observed.errors[0].code == "seed_unavailable"


def test_citation_expand_preserves_the_frozen_seed_when_the_backend_reports_missing_semantic_id() -> None:
    seed = _seed(semantic_scholar_id=None)
    backend = FakeCitationBackend(
        BackendCitationResult(
            direction="references",
            errors=[
                ErrorDetail(
                    code="seed_unavailable",
                    message="semantic ID is unavailable",
                    retryable=False,
                    provider="semantic_scholar",
                )
            ],
        )
    )
    handler = CitationExpandHandler(backend=backend)

    observed = asyncio.run(handler.execute(_action(), _context(seed)))

    assert backend.calls == [("cite-1", seed, "references", 4)]
    assert observed.hits == [seed]
    assert observed.errors[0].code == "seed_unavailable"


def test_citation_expand_keeps_partial_neighbors_errors_and_infrastructure_failure() -> None:
    seed = _seed()
    neighbor = _expanded()
    provider_error = ErrorDetail(
        code="network_error",
        message="offline provider error",
        retryable=True,
        provider="semantic_scholar",
    )
    backend = FakeCitationBackend(
        BackendCitationResult(
            direction="both",
            hits=[neighbor],
            errors=[provider_error],
            infrastructure_failure=True,
        )
    )
    handler = CitationExpandHandler(backend=backend)

    observed = asyncio.run(handler.execute(_action(direction="both"), _context(seed)))

    assert observed.hits == [seed, neighbor]
    assert observed.errors == [provider_error]
    assert observed.infrastructure_failure is True


def test_citation_expand_keeps_one_seed_when_the_backend_repeats_it() -> None:
    seed = _seed()
    backend = FakeCitationBackend(
        BackendCitationResult(direction="references", hits=[seed, _expanded()])
    )
    handler = CitationExpandHandler(backend=backend)

    observed = asyncio.run(handler.execute(_action(), _context(seed)))

    assert observed.hits == [seed, _expanded()]


def test_citation_expand_rejects_a_non_citation_action_without_calling_the_backend() -> None:
    backend = FakeCitationBackend(BackendCitationResult(direction="references"))
    handler = CitationExpandHandler(backend=backend)
    action = TextSearchAction(
        action_id="text-1",
        strategy="wrong handler",
        action_type="text_search",
        payload=TextSearchPayload(query_text="Validated text"),
    )

    with pytest.raises(TypeError, match="citation"):
        asyncio.run(handler.execute(action, _context(_seed())))

    assert backend.calls == []


def test_retrieval_handler_modules_do_not_import_each_other() -> None:
    import paper_search.recall_experiments.retrieval.citation_expand as citation_module
    import paper_search.recall_experiments.retrieval.text_search as text_module
    import paper_search.recall_experiments.retrieval.title_search as title_module

    modules = (text_module, title_module, citation_module)
    names = ("text_search", "title_search", "citation_expand")

    for module in modules:
        source = inspect.getsource(module)
        for name in names:
            if name != module.__name__.rsplit(".", maxsplit=1)[-1]:
                assert name not in source


def test_handlers_satisfy_the_registry_protocol() -> None:
    registry = RetrievalActionRegistry()
    handlers: dict[str, RetrievalActionHandler] = {
        "text_search": TextSearchHandler(backend=FakeSearchBackend()),
        "title_search": TitleSearchHandler(backend=FakeSearchBackend()),
        "citation_expand": CitationExpandHandler(
            backend=FakeCitationBackend(BackendCitationResult(direction="references"))
        ),
    }

    registry.register("text_search", handlers["text_search"])
    registry.register("title_search", handlers["title_search"])
    registry.register("citation_expand", handlers["citation_expand"])

    assert tuple(registry) == ("text_search", "title_search", "citation_expand")
