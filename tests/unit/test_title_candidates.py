from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import (
    ErrorDetail,
    Paper,
    ProviderResult,
    QuerySpec,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.retrieval.title_candidates import (
    LLMTitleCandidateStage,
    extract_title_candidates,
)
from paper_search.retrieval.snapshot_adapters import ProviderAdapterError
from paper_search.llm.snapshot_adapters import LLMAdapterError


def _budget(**updates: object) -> SearchBudget:
    values = {
        "max_search_api_calls": 8,
        "target_search_api_calls": 1,
        "max_llm_calls": 2,
        "target_llm_calls": 1,
        "max_total_tokens": 100,
        "max_cost_cny": 1.0,
        "max_elapsed_seconds": 2,
        "soft_deadline_seconds": 1,
    }
    values.update(updates)
    return SearchBudget.model_validate(values)


def _spec() -> QuerySpec:
    return QuerySpec(
        original_query="graph neural networks for node classification",
        research_goal="find papers on GNN node classification",
        topics=["graph neural networks"],
    )


def _llm_provenance() -> dict[str, str]:
    return {
        "provider": "llm",
        "endpoint": "/chat/completions",
        "model_id": "fixture",
        "requested_at": datetime(2026, 7, 23, tzinfo=UTC).isoformat(),
        "response_hash": "sha256:llm",
        "snapshot_entry_id": "llm-1",
        "snapshot_cache_key": "sha256:" + "b" * 64,
        "snapshot_response_sha256": "sha256:" + "a" * 64,
        "snapshot_path": "snapshots/llm-1.json",
    }


class FakeTitleAnalyzer:
    def __init__(
        self,
        data: dict[str, object],
        *,
        errors: list[str] | None = None,
        usage: UsageActual | None = None,
    ) -> None:
        self.data = data
        self.errors = errors or []
        self.usage = usage or UsageActual(llm_calls=1, cost_cny=0.1)
        self.calls: list[tuple[str, dict[str, object], object]] = []

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: object,
    ) -> ProviderResult[dict[str, Any]]:
        self.calls.append((prompt_name, payload, reservation))
        return ProviderResult[dict[str, Any]](
            data=self.data,
            usage=self.usage,
            provenance=_llm_provenance(),
            cache_hit=False,
            latency_ms=1,
            errors=[
                ErrorDetail(
                    code=code,
                    message="synthetic",
                    retryable=False,
                    provider="llm",
                )
                for code in self.errors
            ],
        )


class CancellingTitleAnalyzer(FakeTitleAnalyzer):
    def __init__(self, data: dict[str, object]) -> None:
        super().__init__(data)
        self.reservation: object | None = None

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: object,
    ) -> ProviderResult[dict[str, Any]]:
        self.reservation = reservation
        raise asyncio.CancelledError()


class RaisingTitleAnalyzer(FakeTitleAnalyzer):
    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: object,
    ) -> ProviderResult[dict[str, Any]]:
        raise LLMAdapterError("LLM live capture failed")


class FakeTitleProvider:
    def __init__(
        self,
        results: dict[str, list[Paper]] | None = None,
        *,
        failed_queries: set[str] | None = None,
    ) -> None:
        self.results = results or {}
        self.failed_queries = failed_queries or set()
        self.calls: list[tuple[str, dict[str, object], int, object]] = []

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: object,
    ) -> ProviderResult[list[Paper]]:
        self.calls.append((query, filters, limit, reservation))
        failed = query in self.failed_queries
        return ProviderResult[list[Paper]](
            data=[] if failed else self.results.get(query, []),
            usage=UsageActual(search_api_calls=1),
            provenance={
                "provider": "openalex",
                "endpoint": "/works",
                "model_id": "fixture",
                "requested_at": datetime(2026, 7, 23, tzinfo=UTC).isoformat(),
                "response_hash": f"sha256:{query}",
            },
            cache_hit=False,
            latency_ms=1,
            errors=(
                [
                    ErrorDetail(
                        code="provider_error",
                        message="synthetic",
                        retryable=False,
                        provider="openalex",
                    )
                ]
                if failed
                else []
            ),
        )


class RaisingTitleProvider(FakeTitleProvider):
    def __init__(
        self,
        results: dict[str, list[Paper]],
        *,
        raising_queries: set[str],
    ) -> None:
        super().__init__(results)
        self.raising_queries = raising_queries

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: object,
    ) -> ProviderResult[list[Paper]]:
        if query in self.raising_queries:
            raise ProviderAdapterError("provider live capture failed")
        return await super().search(query, filters, limit, reservation)


def _stage(
    analyzer: FakeTitleAnalyzer,
    provider: FakeTitleProvider,
    *,
    max_titles: int = 5,
    max_results_per_title: int = 10,
) -> LLMTitleCandidateStage:
    return LLMTitleCandidateStage(
        analyzer=analyzer,
        provider=provider,
        llm_estimate=UsageEstimate(
            llm_calls=1,
            input_tokens=10,
            output_tokens=10,
            cost_cny=0.1,
        ),
        search_estimate=UsageEstimate(search_api_calls=1),
        max_titles=max_titles,
        max_results_per_title=max_results_per_title,
    )


def test_extract_title_candidates_flat_list_dedupes() -> None:
    data = {"titles": ["A Survey of X", "X for Y", "A Survey of X"]}
    assert extract_title_candidates(data, limit=5) == ["A Survey of X", "X for Y"]


def test_extract_title_candidates_accepts_aliases_and_wrappers() -> None:
    data = {
        "QueryAnalysisResult": {
            "candidate_titles": [{"title": "One"}, "Two", "One"]
        }
    }
    assert extract_title_candidates(data, limit=5) == ["One", "Two"]


def test_extract_title_candidates_skips_junk_and_caps() -> None:
    data = {"titles": ["One", 3, None, {"title": "Two"}, "", "Three", "Four"]}
    assert extract_title_candidates(data, limit=2) == ["One", "Two"]


def test_extract_title_candidates_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError):
        extract_title_candidates({"titles": []}, limit=0)
    with pytest.raises(ValueError):
        extract_title_candidates({"titles": []}, limit=101)
    with pytest.raises(ValueError):
        extract_title_candidates({"titles": []}, limit=True)


def test_recall_degrades_on_llm_errors() -> None:
    controller = HardBudgetController(_budget())
    analyzer = FakeTitleAnalyzer({"titles": ["One"]}, errors=["provider_error"])
    provider = FakeTitleProvider()
    stage = _stage(analyzer, provider)

    result = asyncio.run(
        stage.recall(_spec(), controller=controller)
    )

    assert result.status == "degraded"
    assert result.provider_result.data == []
    assert result.titles_generated == 0
    assert result.titles_searched == 0
    assert provider.calls == []
    assert [d.dependency for d in result.diagnostics] == ["llm"]
    assert controller.committed_usage.llm_calls == 1


def test_recall_searches_titles_and_dedupes() -> None:
    controller = HardBudgetController(_budget())
    papers_a = [
        Paper(canonical_id="openalex:W1", title="A", openalex_id="W1"),
        Paper(canonical_id="openalex:W2", title="B", openalex_id="W2"),
    ]
    papers_b = [
        Paper(canonical_id="openalex:W2", title="B", openalex_id="W2"),
        Paper(canonical_id="openalex:W3", title="C", openalex_id="W3"),
    ]
    analyzer = FakeTitleAnalyzer(
        {"titles": ["Title A", "Title B"]}
    )
    provider = FakeTitleProvider(
        {"Title A": papers_a, "Title B": papers_b}
    )
    stage = _stage(analyzer, provider)

    result = asyncio.run(
        stage.recall(_spec(), controller=controller)
    )

    assert result.status == "applied"
    assert [p.canonical_id for p in result.provider_result.data] == [
        "openalex:W1",
        "openalex:W2",
        "openalex:W3",
    ]
    assert [call[0] for call in provider.calls] == ["Title A", "Title B"]
    assert all(call[1] == {} for call in provider.calls)
    assert all(call[2] == 10 for call in provider.calls)
    assert result.titles_generated == 2
    assert result.titles_searched == 2
    assert result.provider_result.usage.llm_calls == 1
    assert result.provider_result.usage.search_api_calls == 2
    assert [d.dependency for d in result.diagnostics] == [
        "llm",
        "openalex",
        "openalex",
    ]
    assert controller.committed_usage.llm_calls == 1
    assert controller.committed_usage.search_api_calls == 2
    prompt_name, payload, _ = analyzer.calls[0]
    assert prompt_name == "title_candidates"
    assert payload["query"] == _spec().original_query
    assert "instructions" in payload


def test_recall_payload_includes_query_spec_context() -> None:
    controller = HardBudgetController(_budget())
    analyzer = FakeTitleAnalyzer({"titles": ["One"]})
    stage = _stage(analyzer, FakeTitleProvider())
    spec = _spec()

    asyncio.run(stage.recall(spec, controller=controller))

    _, payload, _ = analyzer.calls[0]
    assert payload["research_goal"] == spec.research_goal
    assert payload["topics"] == spec.topics
    assert payload["must_have"] == spec.must_have


def test_default_title_candidate_limits_are_twenty() -> None:
    stage = LLMTitleCandidateStage(
        analyzer=FakeTitleAnalyzer({"titles": []}),
        provider=FakeTitleProvider(),
        llm_estimate=UsageEstimate(llm_calls=1, cost_cny=0.1),
        search_estimate=UsageEstimate(search_api_calls=1),
    )

    assert stage._max_titles == 20  # type: ignore[attr-defined]  # noqa: SLF001
    assert "a list of 20" in stage._instructions  # type: ignore[attr-defined]  # noqa: SLF001


def test_recall_continues_past_a_failed_title_search() -> None:
    controller = HardBudgetController(_budget())
    analyzer = FakeTitleAnalyzer({"titles": ["Good", "Bad"]})
    provider = FakeTitleProvider(
        {"Good": [Paper(canonical_id="openalex:W1", title="A")]},
        failed_queries={"Bad"},
    )
    stage = _stage(analyzer, provider)

    result = asyncio.run(
        stage.recall(_spec(), controller=controller)
    )

    assert result.status == "applied"
    assert [p.canonical_id for p in result.provider_result.data] == [
        "openalex:W1"
    ]
    assert result.titles_searched == 2
    assert result.diagnostics[-1].errors
    assert result.diagnostics[-1].dependency == "openalex"


def test_recall_degrades_on_malformed_llm_output() -> None:
    controller = HardBudgetController(_budget())
    analyzer = FakeTitleAnalyzer({"not_titles": []})
    provider = FakeTitleProvider()
    stage = _stage(analyzer, provider)

    result = asyncio.run(
        stage.recall(_spec(), controller=controller)
    )

    assert result.status == "degraded"
    assert result.provider_result.data == []
    assert result.warnings == ["malformed"]
    assert provider.calls == []


def test_recall_respects_max_titles() -> None:
    controller = HardBudgetController(_budget())
    analyzer = FakeTitleAnalyzer(
        {"titles": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]}
    )
    provider = FakeTitleProvider()
    stage = _stage(analyzer, provider, max_titles=5)

    result = asyncio.run(
        stage.recall(_spec(), controller=controller)
    )

    assert result.titles_generated == 5
    assert [call[0] for call in provider.calls] == [
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
    ]


def test_recall_fails_closed_on_cancellation() -> None:
    controller = HardBudgetController(_budget())
    analyzer = CancellingTitleAnalyzer({})
    stage = _stage(analyzer, FakeTitleProvider())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(stage.recall(_spec(), controller=controller))

    assert analyzer.reservation is not None
    terminal = controller.terminal_outcome(analyzer.reservation)
    assert terminal is not None and terminal[0] == "failed"


def test_recall_continues_past_provider_adapter_failure() -> None:
    controller = HardBudgetController(_budget())
    analyzer = FakeTitleAnalyzer({"titles": ["Good", "Bad"]})
    provider = RaisingTitleProvider(
        {"Good": [Paper(canonical_id="openalex:W1", title="A")]},
        raising_queries={"Bad"},
    )
    stage = _stage(analyzer, provider)

    result = asyncio.run(
        stage.recall(_spec(), controller=controller)
    )

    assert result.status == "applied"
    assert [p.canonical_id for p in result.provider_result.data] == [
        "openalex:W1"
    ]
    assert result.titles_searched == 2
    assert result.diagnostics[-1].dependency == "openalex"
    assert result.diagnostics[-1].errors


def test_recall_continues_past_midlist_provider_failure_without_poisoning_controller() -> None:
    controller = HardBudgetController(_budget(max_search_api_calls=8))
    analyzer = FakeTitleAnalyzer({"titles": ["Good1", "Bad", "Good2", "Good3"]})
    provider = RaisingTitleProvider(
        {
            "Good1": [Paper(canonical_id="openalex:W1", title="A")],
            "Good2": [Paper(canonical_id="openalex:W2", title="B")],
            "Good3": [Paper(canonical_id="openalex:W3", title="C")],
        },
        raising_queries={"Bad"},
    )
    stage = _stage(analyzer, provider, max_titles=4)

    result = asyncio.run(stage.recall(_spec(), controller=controller))

    assert result.titles_searched == 4
    assert controller.stop_status() == "continue"
    assert [p.canonical_id for p in result.provider_result.data] == [
        "openalex:W1",
        "openalex:W2",
        "openalex:W3",
    ]


def test_recall_all_title_searches_fail_degrades() -> None:
    controller = HardBudgetController(_budget())
    analyzer = FakeTitleAnalyzer({"titles": ["Only"]})
    provider = RaisingTitleProvider({}, raising_queries={"Only"})
    stage = _stage(analyzer, provider)

    result = asyncio.run(
        stage.recall(_spec(), controller=controller)
    )

    assert result.status == "degraded"
    assert result.provider_result.data == []
    assert result.warnings == ["unavailable"]


def test_recall_degrades_on_llm_adapter_failure() -> None:
    controller = HardBudgetController(_budget())
    stage = _stage(RaisingTitleAnalyzer({}), FakeTitleProvider())

    result = asyncio.run(
        stage.recall(_spec(), controller=controller)
    )

    assert result.status == "degraded"
    assert result.warnings == ["unavailable"]
    assert result.provider_result.data == []
    assert result.diagnostics and result.diagnostics[0].errors
