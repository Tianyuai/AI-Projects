from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

import paper_search.evaluation.embedding_benchmark as benchmark_module
from paper_search.domain.models import Paper
from paper_search.evaluation.embedding_benchmark import (
    benchmark_embedding,
    cuda_peak_allocated_bytes,
    process_peak_rss_bytes,
    reset_cuda_peak_memory,
)
from paper_search.ranking.embedding import EmbeddingRankingResult, EmbeddingScore


class FakeRanker:
    def rank(
        self,
        query: str,
        papers: Sequence[Paper],
    ) -> EmbeddingRankingResult:
        assert query == "synthetic benchmark query"
        return EmbeddingRankingResult(
            ranked=[EmbeddingScore(paper=paper, similarity=0.5) for paper in papers],
            status="applied",
            model_id="fixture-embedding-v1",
            device="cpu",
            fallback_used=False,
            warnings=[],
        )


class UnsafeMetadataRanker:
    def rank(
        self,
        query: str,
        papers: Sequence[Paper],
    ) -> EmbeddingRankingResult:
        del papers
        return EmbeddingRankingResult(
            ranked=[],
            status="degraded",
            model_id=r"D:\private-cache\models\all-MiniLM-L6-v2",
            device="cpu",
            fallback_used=True,
            warnings=[
                "cuda_oom_cpu_fallback",
                f"RuntimeError while ranking query {query!r} from D:\\private-cache",
            ],
        )


class PathMetadataRanker:
    def __init__(self, model_id: str) -> None:
        self._model_id = model_id

    def rank(
        self,
        query: str,
        papers: Sequence[Paper],
    ) -> EmbeddingRankingResult:
        del query, papers
        return EmbeddingRankingResult(
            ranked=[],
            status="applied",
            model_id=self._model_id,
            device="cpu",
            fallback_used=False,
            warnings=[],
        )


class CountingRanker:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def rank(
        self,
        query: str,
        papers: Sequence[Paper],
    ) -> EmbeddingRankingResult:
        del query, papers
        self._events.append("rank")
        return EmbeddingRankingResult(
            ranked=[],
            status="applied",
            model_id="fixture-embedding-v1",
            device="cpu",
            fallback_used=False,
            warnings=[],
        )


def test_benchmark_reports_only_safe_aggregate_fields() -> None:
    times = iter([10.0, 10.125])
    result = benchmark_embedding(
        ranker=FakeRanker(),
        query="synthetic benchmark query",
        papers=[
            Paper(canonical_id="synthetic:1", title="Synthetic paper one"),
            Paper(canonical_id="synthetic:2", title="Synthetic paper two"),
        ],
        batch_size=2,
        clock=lambda: next(times),
        peak_rss=lambda: 123_456,
        cuda_peak=lambda: None,
        cuda_reset=lambda: None,
    )

    assert result.model_dump() == {
        "model_id": "fixture-embedding-v1",
        "device": "cpu",
        "candidate_count": 2,
        "batch_size": 2,
        "latency_ms": 125,
        "process_peak_rss_bytes": 123456,
        "cuda_peak_allocated_bytes": None,
        "status": "applied",
        "fallback_used": False,
        "warnings": [],
    }
    serialized = result.model_dump_json()
    assert "synthetic:1" not in serialized
    assert "Synthetic paper" not in serialized
    assert "synthetic benchmark query" not in serialized


def test_benchmark_sanitizes_unsafe_model_metadata_and_warning_text() -> None:
    result = benchmark_embedding(
        ranker=UnsafeMetadataRanker(),
        query="synthetic benchmark query",
        papers=[],
        batch_size=1,
        clock=lambda: 10.0,
        peak_rss=lambda: 123_456,
        cuda_peak=lambda: None,
        cuda_reset=lambda: None,
    )

    assert result.model_dump() == {
        "model_id": "local_model",
        "device": "cpu",
        "candidate_count": 0,
        "batch_size": 1,
        "latency_ms": 0,
        "process_peak_rss_bytes": 123456,
        "cuda_peak_allocated_bytes": None,
        "status": "degraded",
        "fallback_used": True,
        "warnings": ["cuda_oom_cpu_fallback", "unsanitized_warning"],
    }
    serialized = result.model_dump_json()
    assert "private-cache" not in serialized
    assert "synthetic benchmark query" not in serialized
    assert "RuntimeError" not in serialized


def test_sanitize_warnings_returns_stripped_warning_codes() -> None:
    assert benchmark_module._sanitize_warnings(["  cuda_oom_cpu_fallback  "]) == [
        "cuda_oom_cpu_fallback"
    ]


@pytest.mark.parametrize(
    "model_id",
    [
        r"D:\private-cache\models\all-MiniLM-L6-v2",
        r"\\private-host\share\all-MiniLM-L6-v2",
        "/private-cache/models/all-MiniLM-L6-v2",
        "models/all-MiniLM-L6-v2",
        "~/private-cache/all-MiniLM-L6-v2",
        "./private-cache/all-MiniLM-L6-v2",
        "../private-cache/all-MiniLM-L6-v2",
    ],
)
def test_benchmark_maps_any_detected_local_path_model_id_to_fixed_constant(
    model_id: str,
) -> None:
    result = benchmark_embedding(
        ranker=PathMetadataRanker(model_id),
        query="synthetic benchmark query",
        papers=[],
        batch_size=1,
        clock=lambda: 10.0,
        peak_rss=lambda: 123_456,
        cuda_peak=lambda: None,
        cuda_reset=lambda: None,
    )

    assert result.model_id == "local_model"
    assert "all-MiniLM-L6-v2" not in result.model_dump_json()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_benchmark_rejects_invalid_batch_size_before_side_effects(
    batch_size: int,
) -> None:
    events: list[str] = []

    with pytest.raises(ValueError, match="batch_size"):
        benchmark_embedding(
            ranker=CountingRanker(events),
            query="synthetic benchmark query",
            papers=[],
            batch_size=batch_size,
            clock=lambda: 10.0,
            peak_rss=lambda: 123_456,
            cuda_peak=lambda: None,
            cuda_reset=lambda: events.append("reset"),
        )

    assert events == []


def test_process_peak_rss_is_a_positive_os_measurement() -> None:
    assert process_peak_rss_bytes() > 0


def test_cuda_memory_helpers_use_torch_only_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def reset_peak_memory_stats() -> None:
            events.append("reset")

        @staticmethod
        def max_memory_allocated() -> int:
            return 654_321

    monkeypatch.setattr(
        benchmark_module,
        "import_module",
        lambda _name: SimpleNamespace(cuda=FakeCuda()),
    )

    reset_cuda_peak_memory()

    assert events == ["reset"]
    assert cuda_peak_allocated_bytes() == 654_321
