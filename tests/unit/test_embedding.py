from __future__ import annotations

from collections.abc import Sequence

import pytest

from paper_search.domain.models import Paper
from paper_search.ranking.embedding import EmbeddingRanker


class FakeEncoder:
    def __init__(
        self,
        *,
        device: str,
        vectors: dict[str, list[float]],
        calls: list[tuple[str, list[str], int]],
        closed: list[str],
    ) -> None:
        self.model_id = "fixture-embedding-v1"
        self.device = device
        self._vectors = vectors
        self._calls = calls
        self._closed = closed

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        values = list(texts)
        self._calls.append((self.device, values, batch_size))
        return [self._vectors[value] for value in values]

    def close(self) -> None:
        self._closed.append(self.device)


def _paper(identifier: str, title: str, abstract: str | None) -> Paper:
    return Paper(canonical_id=identifier, title=title, abstract=abstract)


def test_embedding_ranker_uses_original_query_and_title_plus_abstract() -> None:
    calls: list[tuple[str, list[str], int]] = []
    closed: list[str] = []
    vectors = {
        "graph retrieval": [1.0, 0.0],
        "Relevant\nsemantic graph retrieval": [1.0, 0.0],
        "Unrelated\nprotein folding": [0.0, 1.0],
    }
    ranker = EmbeddingRanker(
        encoder_factory=lambda device: FakeEncoder(
            device=device,
            vectors=vectors,
            calls=calls,
            closed=closed,
        ),
        model_id="fixture-embedding-v1",
        preferred_device="cpu",
        batch_size=2,
        fallback_to_cpu=True,
    )

    result = ranker.rank(
        "graph retrieval",
        [
            _paper("paper:unrelated", "Unrelated", "protein folding"),
            _paper("paper:relevant", "Relevant", "semantic graph retrieval"),
        ],
    )

    assert [item.paper.canonical_id for item in result.ranked] == [
        "paper:relevant",
        "paper:unrelated",
    ]
    assert calls == [
        ("cpu", ["graph retrieval"], 1),
        (
            "cpu",
            ["Unrelated\nprotein folding", "Relevant\nsemantic graph retrieval"],
            2,
        ),
    ]
    assert result.status == "applied"
    assert result.device == "cpu"
    assert result.fallback_used is False
    assert result.warnings == []
    assert closed == ["cpu"]


def test_embedding_ranker_uses_title_when_abstract_is_missing_or_blank() -> None:
    calls: list[tuple[str, list[str], int]] = []
    vectors = {
        "query": [1.0, 0.0],
        "Missing": [1.0, 0.0],
        "Blank": [0.0, 1.0],
    }
    ranker = EmbeddingRanker(
        encoder_factory=lambda device: FakeEncoder(
            device=device,
            vectors=vectors,
            calls=calls,
            closed=[],
        ),
        model_id="fixture-embedding-v1",
        preferred_device="cpu",
        batch_size=8,
        fallback_to_cpu=True,
    )

    result = ranker.rank(
        "query",
        [
            _paper("paper:blank", "Blank", "   "),
            _paper("paper:missing", "Missing", None),
        ],
    )

    assert calls[1][1] == ["Blank", "Missing"]
    assert [item.paper.canonical_id for item in result.ranked] == [
        "paper:missing",
        "paper:blank",
    ]


def test_embedding_ranker_batches_candidates_and_preserves_prior_order_on_ties() -> None:
    calls: list[tuple[str, list[str], int]] = []
    vectors = {
        "query": [1.0, 0.0],
        "First": [1.0, 0.0],
        "Second": [1.0, 0.0],
        "Third": [1.0, 0.0],
    }
    ranker = EmbeddingRanker(
        encoder_factory=lambda device: FakeEncoder(
            device=device,
            vectors=vectors,
            calls=calls,
            closed=[],
        ),
        model_id="fixture-embedding-v1",
        preferred_device="cpu",
        batch_size=2,
        fallback_to_cpu=True,
    )
    papers = [
        _paper("paper:z", "First", None),
        _paper("paper:a", "Second", None),
        _paper("paper:m", "Third", None),
    ]

    result = ranker.rank("query", papers)

    assert [call[1] for call in calls] == [
        ["query"],
        ["First", "Second"],
        ["Third"],
    ]
    assert [item.paper.canonical_id for item in result.ranked] == [
        "paper:z",
        "paper:a",
        "paper:m",
    ]


def test_embedding_ranker_returns_empty_without_loading_encoder() -> None:
    loaded: list[str] = []
    ranker = EmbeddingRanker(
        encoder_factory=lambda device: loaded.append(device),  # type: ignore[arg-type,func-returns-value]
        model_id="fixture-embedding-v1",
        preferred_device="cpu",
        batch_size=2,
        fallback_to_cpu=True,
    )

    result = ranker.rank("query", [])

    assert result.ranked == []
    assert result.status == "applied"
    assert loaded == []


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_embedding_ranker_rejects_invalid_batch_size(batch_size: object) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingRanker(
            encoder_factory=lambda _device: object(),  # type: ignore[arg-type,return-value]
            model_id="fixture-embedding-v1",
            preferred_device="cpu",
            batch_size=batch_size,  # type: ignore[arg-type]
            fallback_to_cpu=True,
        )
