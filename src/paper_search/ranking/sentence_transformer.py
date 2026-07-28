"""Lazy sentence-transformers adapter with sanitized failure boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

import numpy as np

from paper_search.ranking.embedding import (
    EmbeddingDevice,
    EmbeddingOutOfMemoryError,
    EmbeddingUnavailableError,
    TextEncoderFactory,
)


class SentenceTransformerEncoder:
    def __init__(self, *, model_id: str, device: EmbeddingDevice) -> None:
        self.model_id = model_id
        self.device = device
        self._model: Any | None = None
        try:
            module = import_module("sentence_transformers")
            self._model = module.SentenceTransformer(model_id, device=device)
        except ImportError:
            raise EmbeddingUnavailableError(
                "sentence-transformers is not installed"
            ) from None
        except RuntimeError as error:
            self._raise_sanitized(error)

    @staticmethod
    def _raise_sanitized(error: RuntimeError) -> None:
        if "out of memory" in str(error).casefold():
            raise EmbeddingOutOfMemoryError(
                "embedding encoder exhausted device memory"
            ) from None
        raise EmbeddingUnavailableError("embedding encoder failed") from None

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        if self._model is None:
            raise EmbeddingUnavailableError("embedding encoder is closed")
        try:
            values = self._model.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except RuntimeError as error:
            self._raise_sanitized(error)
        matrix = np.asarray(values, dtype=np.float32)
        return [[float(value) for value in row] for row in matrix]

    def close(self) -> None:
        self._model = None
        if self.device != "cuda":
            return
        try:
            torch = import_module("torch")
            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            return


def sentence_transformer_factory(model_id: str) -> TextEncoderFactory:
    normalized = model_id.strip()
    if not normalized:
        raise ValueError("model_id must not be empty")

    def factory(device: EmbeddingDevice) -> SentenceTransformerEncoder:
        return SentenceTransformerEncoder(model_id=normalized, device=device)

    return factory
