from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import paper_search.ranking.sentence_transformer as adapter
from paper_search.ranking.embedding import (
    EmbeddingOutOfMemoryError,
    EmbeddingUnavailableError,
)


class FakeModel:
    def __init__(self, *, raises: RuntimeError | None = None) -> None:
        self.raises = raises
        self.calls: list[dict[str, object]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.calls.append({"texts": texts, **kwargs})
        if self.raises is not None:
            raise self.raises
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_adapter_loads_requested_device_and_normalizes_encode_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    loaded: list[tuple[str, str]] = []
    module = SimpleNamespace(
        SentenceTransformer=lambda model_id, device: (
            loaded.append((model_id, device)),
            model,
        )[1]
    )
    monkeypatch.setattr(adapter, "import_module", lambda _name: module)

    encoder = adapter.SentenceTransformerEncoder(
        model_id="fixture/model",
        device="cpu",
    )
    vectors = encoder.encode(["one", "two"], batch_size=2)

    assert loaded == [("fixture/model", "cpu")]
    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert model.calls == [
        {
            "texts": ["one", "two"],
            "batch_size": 2,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
    ]


def test_adapter_maps_oom_and_other_runtime_errors_to_sanitized_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = "CUDA out of memory at private path"
    monkeypatch.setattr(
        adapter,
        "import_module",
        lambda _name: SimpleNamespace(
            SentenceTransformer=lambda _model_id, device: FakeModel(
                raises=RuntimeError(private)
            )
        ),
    )
    encoder = adapter.SentenceTransformerEncoder(model_id="fixture/model", device="cuda")

    with pytest.raises(EmbeddingOutOfMemoryError, match="device memory"):
        encoder.encode(["one"], batch_size=1)

    monkeypatch.setattr(
        adapter,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("private import path")),
    )
    with pytest.raises(EmbeddingUnavailableError, match="not installed"):
        adapter.SentenceTransformerEncoder(model_id="fixture/model", device="cpu")
