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
