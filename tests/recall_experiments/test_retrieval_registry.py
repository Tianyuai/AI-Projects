"""Contract tests for explicit, offline recall retrieval backends."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import (
    CitationExpansion,
    ErrorDetail,
    Paper,
    ProviderResult,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.recall_experiments.contracts import (
    RetrievalExecutionContext,
    TextSearchAction,
    TextSearchPayload,
)
from paper_search.recall_experiments.retrieval.backends import (
    BackendSearchResult,
    BudgetedCitationBackend,
    BudgetedSearchBackend,
    SearchActionHandler,
)
from paper_search.recall_experiments.retrieval.registry import RetrievalActionRegistry
from paper_search.retrieval.snapshot_adapters import ReplaySearchProvider
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencySnapshotReader,
)


def _budget() -> SearchBudget:
    return SearchBudget(
        max_search_api_calls=10,
        max_llm_calls=10,
        max_total_tokens=10_000,
        max_cost_cny=Decimal("10.00"),
        max_elapsed_seconds=60,
        soft_deadline_seconds=30,
        max_citation_seeds=2,
    )


def _paper() -> Paper:
    return Paper(
        canonical_id="paper-1",
        title="Offline adapter paper",
        sources=["openalex"],
    )


def _result(*, errors: list[ErrorDetail] | None = None) -> ProviderResult[list[Paper]]:
    return ProviderResult(
        data=[_paper()],
        usage=UsageActual(search_api_calls=1, elapsed_ms=7, cost_cny=Decimal("0.01")),
        provenance={
            "provider": "openalex",
            "endpoint": "/works",
            "model_id": "openalex-works-v1",
            "requested_at": datetime.now(UTC).isoformat(),
            "response_hash": "sha256:test",
            "fixture": "offline",
        },
        cache_hit=True,
        latency_ms=7,
        errors=errors or [],
    )


class _OfflineSearchProvider:
    def __init__(self, result: ProviderResult[list[Paper]]) -> None:
        self.result = result
        self.reservations = []

    async def search(self, query: str, filters: dict[str, object], limit: int, reservation: object) -> ProviderResult[list[Paper]]:
        assert query == "offline query"
        assert filters == {"year_from": 2020}
        assert limit == 3
        self.reservations.append(reservation)
        return self.result


class _Handler:
    def __init__(self, name: str) -> None:
        self.name = name

    async def execute(self, action: object, context: object) -> object:
        return (self.name, action, context)


class _FakeBackend:
    async def search(
        self,
        action_id: str,
        query: str,
        filters: dict[str, object],
        limit: int,
    ) -> BackendSearchResult:
        assert (action_id, query, filters, limit) == (
            "action-1",
            "offline query",
            {"year_from": 2020},
            3,
        )
        return BackendSearchResult(hits=[_paper()], provenance={"provider": "fake"})


class _OfflineCitationProvider:
    def __init__(self, result: ProviderResult[CitationExpansion]) -> None:
        self.result = result
        self.reservations = []

    async def references(self, paper_id: object, limit: int, reservation: object) -> ProviderResult[CitationExpansion]:
        assert getattr(paper_id, "value") == "S2:seed"
        assert limit == 2
        self.reservations.append(reservation)
        return self.result

    async def citations(self, paper_id: object, limit: int, reservation: object) -> ProviderResult[CitationExpansion]:
        raise AssertionError("citation direction was not requested")


class _BothDirectionCitationProvider:
    def __init__(self, result: ProviderResult[CitationExpansion]) -> None:
        self.result = result
        self.reservations = []

    async def references(
        self, paper_id: object, limit: int, reservation: object
    ) -> ProviderResult[CitationExpansion]:
        assert getattr(paper_id, "value") == "S2:seed"
        assert limit == 2
        self.reservations.append(reservation)
        return self.result

    async def citations(
        self, paper_id: object, limit: int, reservation: object
    ) -> ProviderResult[CitationExpansion]:
        assert getattr(paper_id, "value") == "S2:seed"
        assert limit == 2
        self.reservations.append(reservation)
        return self.result


class _CancellingSearchProvider:
    def __init__(self) -> None:
        self.reservation: object | None = None

    async def search(
        self, query: str, filters: dict[str, object], limit: int, reservation: object
    ) -> ProviderResult[list[Paper]]:
        self.reservation = reservation
        raise asyncio.CancelledError()


def test_registry_is_explicit_ordered_and_rejects_duplicates() -> None:
    first = _Handler("first")
    second = _Handler("second")
    registry = RetrievalActionRegistry()

    registry.register("text_search", first)
    registry.register("title_search", second)

    assert tuple(registry) == ("text_search", "title_search")
    assert registry.resolve("text_search") is first
    with pytest.raises(ValueError, match="already registered"):
        registry.register("text_search", _Handler("duplicate"))
    with pytest.raises(KeyError, match="unknown retrieval action"):
        registry.resolve("citation_expand")


def test_unregister_removes_only_the_requested_explicit_handler() -> None:
    registry = RetrievalActionRegistry()
    text_handler = _Handler("text")
    registry.register("text_search", text_handler)
    registry.register("title_search", _Handler("title"))

    assert registry.unregister("text_search") is text_handler
    assert tuple(registry) == ("title_search",)
    with pytest.raises(KeyError, match="unknown retrieval action"):
        registry.unregister("text_search")


def test_handler_executes_an_injected_backend_without_provider_imports() -> None:
    handler = SearchActionHandler(backend=_FakeBackend())
    action = TextSearchAction(
        action_id="action-1",
        strategy="offline",
        action_type="text_search",
        payload=TextSearchPayload(query_text="offline query"),
    )
    context = RetrievalExecutionContext(
        query_id="query-1",
        provider_filters={"year_from": 2020},
        max_results_per_action=3,
    )

    observed = asyncio.run(handler.execute(action, context))

    assert observed.action_id == "action-1"
    assert observed.action_type == "text_search"
    assert observed.hits == [_paper()]
    assert observed.provenance == {"provider": "fake"}


def test_budgeted_search_reserves_once_and_preserves_provider_result() -> None:
    controller = HardBudgetController(_budget())
    provider = _OfflineSearchProvider(_result())
    backend = BudgetedSearchBackend(
        provider=provider,
        controller=controller,
        call_estimate=UsageEstimate(
            search_api_calls=1, elapsed_ms=10, cost_cny=Decimal("0.02")
        ),
    )

    observed = asyncio.run(
        backend.search("action-1", "offline query", {"year_from": 2020}, 3)
    )

    assert observed.hits == [_paper()]
    assert observed.usage == provider.result.usage
    assert observed.provenance == provider.result.provenance
    assert observed.errors == []
    assert observed.infrastructure_failure is False
    assert len(provider.reservations) == 1
    terminal = controller.terminal_outcome(provider.reservations[0])
    assert terminal == ("settled", provider.result.usage)
    assert controller.export_state()["reservations"] == []


def test_budgeted_citation_backend_uses_the_supplied_semantic_provider() -> None:
    controller = HardBudgetController(_budget())
    expanded = _paper().model_copy(
        update={"canonical_id": "paper-2", "semantic_scholar_id": "S2:expanded"}
    )
    provider_result = ProviderResult(
        data=CitationExpansion(papers=[expanded], raw_edges=[]),
        usage=UsageActual(search_api_calls=1, elapsed_ms=7, cost_cny=Decimal("0.01")),
        provenance={
            "provider": "semantic_scholar",
            "endpoint": "/paper/S2:seed/references",
            "model_id": "semantic-graph-v1",
            "requested_at": datetime.now(UTC).isoformat(),
            "response_hash": "sha256:test",
        },
        cache_hit=True,
        latency_ms=7,
        errors=[],
    )
    provider = _OfflineCitationProvider(provider_result)
    backend = BudgetedCitationBackend(
        provider=provider,
        controller=controller,
        call_estimate=UsageEstimate(
            search_api_calls=1, elapsed_ms=10, cost_cny=Decimal("0.02")
        ),
    )
    seed = _paper().model_copy(update={"semantic_scholar_id": "S2:seed"})

    observed = asyncio.run(backend.expand("citation-1", seed, "references", 2))

    assert observed.hits == [expanded]
    assert observed.usage == provider_result.usage
    assert observed.provenance == provider_result.provenance
    assert len(provider.reservations) == 1
    assert controller.terminal_outcome(provider.reservations[0]) == (
        "settled",
        provider_result.usage,
    )


def test_both_direction_citation_terminalizes_an_accounting_failure() -> None:
    controller = HardBudgetController(_budget())
    oversized = ProviderResult(
        data=CitationExpansion(papers=[], raw_edges=[]),
        usage=UsageActual(search_api_calls=2, elapsed_ms=7, cost_cny=Decimal("0.01")),
        provenance={
            "provider": "semantic_scholar",
            "endpoint": "/paper/S2:seed/references",
            "model_id": "semantic-graph-v1",
            "requested_at": datetime.now(UTC).isoformat(),
            "response_hash": "sha256:test",
        },
        cache_hit=True,
        latency_ms=7,
        errors=[],
    )
    provider = _OfflineCitationProvider(oversized)
    backend = BudgetedCitationBackend(
        provider=provider,
        controller=controller,
        call_estimate=UsageEstimate(
            search_api_calls=1, elapsed_ms=10, cost_cny=Decimal("0.02")
        ),
    )
    seed = _paper().model_copy(update={"semantic_scholar_id": "S2:seed"})

    observed = asyncio.run(backend.expand("citation-1", seed, "both", 2))

    assert observed.errors[0].code == "accounting_failure"
    assert observed.infrastructure_failure is True
    assert controller.terminal_outcome(provider.reservations[0]) == (
        "failed",
        oversized.usage,
    )


def test_both_direction_citation_uses_action_traceable_receipts_and_aggregates_usage() -> None:
    controller = HardBudgetController(_budget())
    result = ProviderResult(
        data=CitationExpansion(papers=[], raw_edges=[]),
        usage=UsageActual(search_api_calls=1, elapsed_ms=7, cost_cny=Decimal("0.01")),
        provenance={
            "provider": "semantic_scholar",
            "endpoint": "/paper/S2:seed",
            "model_id": "semantic-graph-v1",
            "requested_at": datetime.now(UTC).isoformat(),
            "response_hash": "sha256:test",
        },
        cache_hit=True,
        latency_ms=7,
        errors=[],
    )
    provider = _BothDirectionCitationProvider(result)
    backend = BudgetedCitationBackend(
        provider=provider,
        controller=controller,
        call_estimate=UsageEstimate(
            search_api_calls=1, elapsed_ms=10, cost_cny=Decimal("0.02")
        ),
    )
    seed = _paper().model_copy(update={"semantic_scholar_id": "S2:seed"})

    observed = asyncio.run(backend.expand("citation-42", seed, "both", 2))

    assert observed.usage == UsageActual(
        search_api_calls=2, elapsed_ms=14, cost_cny=Decimal("0.02")
    )
    assert [reservation.action for reservation in provider.reservations] == [
        "citation-42.references:S2:seed",
        "citation-42.citations:S2:seed",
    ]
    assert [controller.terminal_outcome(reservation)[0] for reservation in provider.reservations] == [
        "settled",
        "settled",
    ]


def test_cancelled_search_releases_its_undispatched_reservation() -> None:
    controller = HardBudgetController(_budget())
    backend = BudgetedSearchBackend(
        provider=_CancellingSearchProvider(),
        controller=controller,
        call_estimate=UsageEstimate(
            search_api_calls=1, elapsed_ms=10, cost_cny=Decimal("0.02")
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(backend.search("action-1", "offline query", {}, 3))

    assert controller.export_state()["reservations"] == []


def test_both_direction_citation_preserves_provider_error_and_terminal_receipt() -> None:
    controller = HardBudgetController(_budget())
    provider_error = ErrorDetail(
        code="network_error",
        message="offline citation provider failed",
        retryable=True,
        provider="semantic_scholar",
    )
    result = ProviderResult(
        data=CitationExpansion(papers=[], raw_edges=[]),
        usage=UsageActual(search_api_calls=1, elapsed_ms=7, cost_cny=Decimal("0.01")),
        provenance={
            "provider": "semantic_scholar",
            "endpoint": "/paper/S2:seed/references",
            "model_id": "semantic-graph-v1",
            "requested_at": datetime.now(UTC).isoformat(),
            "response_hash": "sha256:test",
        },
        cache_hit=True,
        latency_ms=7,
        errors=[provider_error],
    )
    provider = _BothDirectionCitationProvider(result)
    backend = BudgetedCitationBackend(
        provider=provider,
        controller=controller,
        call_estimate=UsageEstimate(
            search_api_calls=1, elapsed_ms=10, cost_cny=Decimal("0.02")
        ),
    )
    seed = _paper().model_copy(update={"semantic_scholar_id": "S2:seed"})

    observed = asyncio.run(backend.expand("citation-43", seed, "both", 2))

    assert [error.code for error in observed.errors] == [provider_error.code]
    assert observed.infrastructure_failure is True
    assert len(provider.reservations) == 1
    assert controller.terminal_outcome(provider.reservations[0]) == ("settled", result.usage)


@pytest.mark.parametrize(
    "code",
    ["authentication_error", "rate_limited", "network_error", "snapshot_unavailable"],
)
def test_budgeted_search_classifies_provider_infrastructure_failures(code: str) -> None:
    error = ErrorDetail(
        code=code,
        message="offline failure",
        retryable=code in {"rate_limited", "network_error"},
        provider="openalex",
    )
    controller = HardBudgetController(_budget())
    backend = BudgetedSearchBackend(
        provider=_OfflineSearchProvider(_result(errors=[error])),
        controller=controller,
        call_estimate=UsageEstimate(
            search_api_calls=1, elapsed_ms=10, cost_cny=Decimal("0.02")
        ),
    )

    observed = asyncio.run(
        backend.search("action-1", "offline query", {"year_from": 2020}, 3)
    )

    assert observed.errors == [error]
    assert observed.infrastructure_failure is True


def test_replay_search_returns_snapshot_unavailable_for_a_novel_request(
    tmp_path: Path,
) -> None:
    store = DependencyCaptureStore(tmp_path / "sealed", clock=lambda: datetime.now(UTC))
    manifest = store.seal()
    replay = ReplaySearchProvider(
        dependency="openalex",
        reader=DependencySnapshotReader(
            store.manifest_path,
            snapshot_manifest_sha256=store.manifest_sha256,
            snapshot_set_id=manifest.snapshot_set_id,
        ),
    )

    observed = asyncio.run(replay.search("novel request", {}, 1, object()))

    assert observed.data == []
    assert observed.errors[0].code == "snapshot_unavailable"
    assert not hasattr(replay, "_client")
