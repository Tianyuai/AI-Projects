"""Contract tests for explicit, offline recall retrieval backends."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from paper_search.control.budget import HardBudgetController
from paper_search.control.pricing import ActualCostPricer, load_pricing_policy
from paper_search.domain.models import (
    CitationExpansion,
    ErrorDetail,
    Paper,
    ProviderResult,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.live_identity import LiveDependencyEvidence, LiveProviderDescriptor
from paper_search.recall_experiments.contracts import (
    RetrievalExecutionContext,
    TextSearchAction,
    TextSearchPayload,
)
from paper_search.recall_experiments.identity import (
    LiveDependencyIdentity,
    dependency_identity_from_evidence,
    validate_scheme_b_dependency_identity,
)
from paper_search.recall_experiments.retrieval.backends import (
    BackendSearchResult,
    BudgetedCitationBackend,
    BudgetedSearchBackend,
    SearchActionHandler,
)
from paper_search.recall_experiments.retrieval.registry import RetrievalActionRegistry
from paper_search.retrieval.snapshot_adapters import (
    LiveCaptureSearchProvider,
    ReplaySearchProvider,
)
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


def test_dependency_identity_is_converted_from_strict_generic_evidence() -> None:
    evidence = LiveDependencyEvidence(
        identity_schema_version="live-dependency-evidence-v1",
        provider=LiveProviderDescriptor(
            identity_schema_version="live-provider-descriptor-v1",
            provider="openalex",
            dependency="openalex",
            adapter="openalex-works-v1",
            model=None,
            version="live-capture-search-v1",
            endpoints=("https://api.openalex.org/works",),
            operations=("search",),
        ),
        pricing_policy_sha256="sha256:" + "a" * 64,
        controller_policy_sha256="sha256:" + "b" * 64,
        formal_live=True,
    )

    identity = dependency_identity_from_evidence(evidence)

    assert identity.model_dump(mode="json") == {
        "identity_schema_version": "live-dependency-runtime-identity-v1",
        "provider": "openalex",
        "dependency": "openalex",
        "adapter": "openalex-works-v1",
        "model": None,
        "version": "live-capture-search-v1",
        "endpoints": ["https://api.openalex.org/works"],
        "operations": ["search"],
        "pricing_policy_sha256": "sha256:" + "a" * 64,
        "controller_policy_sha256": "sha256:" + "b" * 64,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "openalex.evil"},
        {"adapter": "openalex-works-v2"},
        {"version": "live-capture-search-v2"},
        {"model": "unexpected-model"},
        {"endpoints": ("https://api.openalex.org/evil",)},
        {"operations": ("search", "exfiltrate")},
    ],
)
def test_scheme_b_search_identity_rejects_every_unadmitted_surface_change(
    changes: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "identity_schema_version": "live-dependency-runtime-identity-v1",
        "provider": "openalex",
        "dependency": "openalex",
        "adapter": "openalex-works-v1",
        "model": None,
        "version": "live-capture-search-v1",
        "endpoints": ("https://api.openalex.org/works",),
        "operations": ("search",),
        "pricing_policy_sha256": "sha256:" + "a" * 64,
        "controller_policy_sha256": "sha256:" + "b" * 64,
    }
    payload.update(changes)
    identity = LiveDependencyIdentity.model_validate(payload)

    with pytest.raises(ValueError, match="search surface"):
        validate_scheme_b_dependency_identity("search", identity)


def test_scheme_b_citation_identity_requires_the_exact_semantic_scholar_surface() -> None:
    identity = LiveDependencyIdentity(
        identity_schema_version="live-dependency-runtime-identity-v1",
        provider="semantic_scholar",
        dependency="semantic_scholar",
        adapter="semantic-graph-v1",
        model=None,
        version="live-capture-search-v1",
        endpoints=(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references",
            "https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations",
        ),
        operations=("search", "batch", "references", "citations"),
        pricing_policy_sha256="sha256:" + "a" * 64,
        controller_policy_sha256="sha256:" + "b" * 64,
    )

    assert validate_scheme_b_dependency_identity("citation", identity) is identity
    with pytest.raises(ValueError, match="citation surface"):
        validate_scheme_b_dependency_identity(
            "citation",
            identity.model_copy(update={"operations": ("references",)}),
        )


def test_live_retrieval_backends_bind_provider_owned_identity_and_exact_objects(
    tmp_path: Path,
) -> None:
    controller = HardBudgetController(_budget(), formal_live=True)
    pricer = ActualCostPricer(
        load_pricing_policy(Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")),
        valued_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
        ) as client:
            store = DependencyCaptureStore(tmp_path / "snapshot")
            search_provider = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=store,
                pricer=pricer,
                controller=controller,
            )
            citation_provider = LiveCaptureSearchProvider(
                dependency="semantic_scholar",
                client=client,
                capture_store=store,
                pricer=pricer,
                controller=controller,
            )
            estimate = UsageEstimate(search_api_calls=1, cost_cny=Decimal("0.01"))
            search = BudgetedSearchBackend(
                provider=search_provider,
                controller=controller,
                call_estimate=estimate,
            )
            citation = BudgetedCitationBackend(
                provider=citation_provider,
                controller=controller,
                call_estimate=estimate,
            )
            assert search.dependency_identity.dependency == "openalex"
            assert citation.dependency_identity.dependency == "semantic_scholar"
            assert {"references", "citations"}.issubset(
                citation.dependency_identity.operations
            )
            assert search.live_pricer is pricer
            assert citation.live_pricer is pricer
            assert search.live_controller is controller
            assert citation.live_controller is controller

    asyncio.run(run())


def test_live_search_backend_can_explicitly_bind_semantic_scholar(
    tmp_path: Path,
) -> None:
    controller = HardBudgetController(_budget(), formal_live=True)
    pricer = ActualCostPricer(
        load_pricing_policy(Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")),
        valued_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
        ) as client:
            provider = LiveCaptureSearchProvider(
                dependency="semantic_scholar",
                client=client,
                capture_store=DependencyCaptureStore(tmp_path / "snapshot-s2-search"),
                pricer=pricer,
                controller=controller,
            )
            backend = BudgetedSearchBackend(
                provider=provider,
                controller=controller,
                call_estimate=UsageEstimate(
                    search_api_calls=1, cost_cny=Decimal("0.01")
                ),
                dependency="semantic_scholar",
            )

            assert backend.dependency_identity.dependency == "semantic_scholar"

    asyncio.run(run())


def test_budgeted_openalex_backend_executes_one_frozen_continuation_page(
    tmp_path: Path,
) -> None:
    controller = HardBudgetController(_budget(), formal_live=True)
    pricer = ActualCostPricer(
        load_pricing_policy(Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")),
        valued_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=Path("tests/fixtures/openalex/works_page_2.json").read_bytes(),
            request=request,
        )

    async def run() -> BackendSearchResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=DependencyCaptureStore(tmp_path / "snapshot-continuation"),
                pricer=pricer,
                controller=controller,
            )
            backend = BudgetedSearchBackend(
                provider=provider,
                controller=controller,
                call_estimate=UsageEstimate(
                    search_api_calls=1, cost_cny=Decimal("0.01")
                ),
            )
            return await backend.search_continuation(
                "depth-page-2",
                "retrieval augmented generation",
                {},
                cursor="cursor-page-2",
                limit=50,
            )

    result = asyncio.run(run())

    assert len(requests) == 1
    assert requests[0].url.params["cursor"] == "cursor-page-2"
    assert [paper.openalex_id for paper in result.hits] == ["W126"]
    assert result.usage.search_api_calls == 1
    assert result.infrastructure_failure is False


def test_live_retrieval_backends_reject_wrong_role_or_controller(
    tmp_path: Path,
) -> None:
    controller = HardBudgetController(_budget(), formal_live=True)
    other_controller = HardBudgetController(_budget(), formal_live=True)
    pricer = ActualCostPricer(
        load_pricing_policy(Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")),
        valued_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
        ) as client:
            provider = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=pricer,
                controller=controller,
            )
            estimate = UsageEstimate(search_api_calls=1, cost_cny=Decimal("0.01"))
            with pytest.raises(ValueError, match="controller"):
                BudgetedSearchBackend(
                    provider=provider,
                    controller=other_controller,
                    call_estimate=estimate,
                )
            with pytest.raises(ValueError, match="citation"):
                BudgetedCitationBackend(
                    provider=provider,
                    controller=controller,
                    call_estimate=estimate,
                )
            unadmitted = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=DependencyCaptureStore(tmp_path / "unadmitted"),
                pricer=pricer,
                controller=controller,
                adapter_version="openalex-works-v2",
            )
            with pytest.raises(ValueError, match="search surface"):
                BudgetedSearchBackend(
                    provider=unadmitted,
                    controller=controller,
                    call_estimate=estimate,
                )

    asyncio.run(run())


def test_offline_backend_has_no_live_identity() -> None:
    backend = BudgetedSearchBackend(
        provider=_OfflineSearchProvider(_result()),
        controller=HardBudgetController(_budget()),
        call_estimate=UsageEstimate(search_api_calls=1),
    )
    with pytest.raises(ValueError, match="live identity unavailable"):
        _ = backend.dependency_identity


def test_budgeted_search_rejects_caller_duck_with_exact_live_evidence(
    tmp_path: Path,
) -> None:
    controller = HardBudgetController(_budget(), formal_live=True)
    pricer = ActualCostPricer(
        load_pricing_policy(Path("tests/fixtures/pricing/pricing-policy-test-v1.yaml")),
        valued_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
        ) as client:
            trusted = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=DependencyCaptureStore(tmp_path / "snapshot"),
                pricer=pricer,
                controller=controller,
            )

            class ExactDuck:
                live_identity_evidence = trusted.live_identity_evidence
                live_controller = controller
                live_pricer = pricer

                async def search(self, *_args: object) -> object:
                    raise AssertionError("duck provider dispatched")

            with pytest.raises(ValueError, match="trusted live search provider"):
                BudgetedSearchBackend(
                    provider=ExactDuck(),  # type: ignore[arg-type]
                    controller=controller,
                    call_estimate=UsageEstimate(search_api_calls=1),
                )

    asyncio.run(run())


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


class _SequencedCitationProvider:
    def __init__(
        self,
        reference_result: ProviderResult[CitationExpansion],
        citation_result: ProviderResult[CitationExpansion],
    ) -> None:
        self.reference_result = reference_result
        self.citation_result = citation_result
        self.reservations = []

    async def references(
        self, paper_id: object, limit: int, reservation: object
    ) -> ProviderResult[CitationExpansion]:
        self.reservations.append(reservation)
        return self.reference_result

    async def citations(
        self, paper_id: object, limit: int, reservation: object
    ) -> ProviderResult[CitationExpansion]:
        self.reservations.append(reservation)
        return self.citation_result

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


def test_both_direction_citation_error_in_second_direction_keeps_prior_usage() -> None:
    controller = HardBudgetController(_budget())
    reference_result = ProviderResult(
        data=CitationExpansion(papers=[], raw_edges=[]),
        usage=UsageActual(search_api_calls=1, elapsed_ms=7, cost_cny=Decimal("0.01")),
        provenance={
            "provider": "semantic_scholar",
            "endpoint": "/paper/S2:seed/references",
            "model_id": "semantic-graph-v1",
            "requested_at": datetime.now(UTC).isoformat(),
            "response_hash": "sha256:reference",
        },
        cache_hit=True,
        latency_ms=7,
        errors=[],
    )
    citation_error = ErrorDetail(
        code="network_error",
        message="offline citation provider failed",
        retryable=True,
        provider="semantic_scholar",
    )
    citation_result = reference_result.model_copy(
        update={
            "provenance": {**reference_result.provenance, "response_hash": "sha256:citation"},
            "errors": [citation_error],
        }
    )
    provider = _SequencedCitationProvider(reference_result, citation_result)
    backend = BudgetedCitationBackend(
        provider=provider,
        controller=controller,
        call_estimate=UsageEstimate(
            search_api_calls=1, elapsed_ms=10, cost_cny=Decimal("0.02")
        ),
    )
    seed = _paper().model_copy(update={"semantic_scholar_id": "S2:seed"})

    observed = asyncio.run(backend.expand("citation-44", seed, "both", 2))

    assert observed.usage == UsageActual(
        search_api_calls=2, elapsed_ms=14, cost_cny=Decimal("0.02")
    )
    assert [error.code for error in observed.errors] == ["network_error"]
    assert [controller.terminal_outcome(reservation)[0] for reservation in provider.reservations] == [
        "settled",
        "settled",
    ]


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
