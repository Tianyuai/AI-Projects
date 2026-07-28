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
    def __init__(self, *, raises: RuntimeError | AssertionError | None = None) -> None:
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


def test_adapter_maps_constructor_cuda_oom_and_cleans_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = r"CUDA out of memory while loading D:\private-cache\secret-model"
    events: list[str] = []

    def import_fake(name: str) -> object:
        if name == "sentence_transformers":
            def fail_constructor(_model_id: str, *, device: str) -> None:
                events.append(f"construct:{device}")
                raise RuntimeError(private)

            return SimpleNamespace(SentenceTransformer=fail_constructor)
        if name == "torch":
            return SimpleNamespace(
                cuda=SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))
            )
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(adapter, "import_module", import_fake)

    with pytest.raises(EmbeddingOutOfMemoryError) as captured:
        adapter.SentenceTransformerEncoder(model_id="fixture/model", device="cuda")

    assert str(captured.value) == "embedding encoder exhausted device memory"
    assert private not in str(captured.value)
    assert events == ["construct:cuda", "empty_cache"]


def test_adapter_maps_unsupported_cuda_constructor_to_unavailable_and_cleans_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = r"Torch not compiled with CUDA enabled at D:\private-cache"
    events: list[str] = []

    def import_fake(name: str) -> object:
        if name == "sentence_transformers":
            def fail_constructor(_model_id: str, *, device: str) -> None:
                events.append(f"construct:{device}")
                raise AssertionError(private)

            return SimpleNamespace(SentenceTransformer=fail_constructor)
        if name == "torch":
            return SimpleNamespace(
                cuda=SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))
            )
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(adapter, "import_module", import_fake)

    with pytest.raises(EmbeddingUnavailableError) as captured:
        adapter.SentenceTransformerEncoder(model_id="fixture/model", device="cuda")

    assert str(captured.value) == "embedding encoder failed"
    assert private not in str(captured.value)
    assert events == ["construct:cuda", "empty_cache"]


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_adapter_sanitizes_invalid_local_model_constructor_errors_and_cleans_cache(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[OSError] | type[ValueError],
) -> None:
    private = r"invalid local model path D:\private-cache\secret-model"
    events: list[str] = []

    def import_fake(name: str) -> object:
        if name == "sentence_transformers":
            def fail_constructor(_model_id: str, *, device: str) -> None:
                events.append(f"construct:{device}")
                raise error_type(private)

            return SimpleNamespace(SentenceTransformer=fail_constructor)
        if name == "torch":
            return SimpleNamespace(
                cuda=SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))
            )
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(adapter, "import_module", import_fake)

    with pytest.raises(EmbeddingUnavailableError) as captured:
        adapter.SentenceTransformerEncoder(model_id="fixture/model", device="cuda")

    assert str(captured.value) == "embedding encoder failed"
    assert private not in str(captured.value)
    assert events == ["construct:cuda", "empty_cache"]


def test_adapter_maps_unsupported_cuda_encode_to_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = r"Torch not compiled with CUDA enabled at D:\private-cache"
    model = FakeModel(raises=AssertionError(private))
    monkeypatch.setattr(
        adapter,
        "import_module",
        lambda _name: SimpleNamespace(SentenceTransformer=lambda _model_id, device: model),
    )
    encoder = adapter.SentenceTransformerEncoder(model_id="fixture/model", device="cuda")

    with pytest.raises(EmbeddingUnavailableError) as captured:
        encoder.encode(["one"], batch_size=1)

    assert str(captured.value) == "embedding encoder failed"
    assert private not in str(captured.value)
