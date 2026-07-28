"""Deterministic embedding ranking with an injected local encoder."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, Paper

EmbeddingDevice = Literal["cpu", "cuda"]
EmbeddingStatus = Literal["applied", "degraded"]


class EmbeddingUnavailableError(RuntimeError):
    """The requested local encoder cannot be loaded or executed."""


class EmbeddingOutOfMemoryError(RuntimeError):
    """The requested local encoder exhausted device memory."""


class TextEncoder(Protocol):
    model_id: str
    device: EmbeddingDevice

    def encode(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]: ...

    def close(self) -> None: ...


TextEncoderFactory = Callable[[EmbeddingDevice], TextEncoder]

_OUT_OF_MEMORY_HINTS = ("out of memory",)


class EmbeddingScore(DomainModel):
    paper: Paper
    similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)


class EmbeddingRankingResult(DomainModel):
    ranked: list[EmbeddingScore]
    status: EmbeddingStatus
    model_id: NonEmptyStr
    device: EmbeddingDevice
    fallback_used: bool
    warnings: list[NonEmptyStr]


class EmbeddingRankingStage(Protocol):
    def rank(self, query: str, papers: Sequence[Paper]) -> EmbeddingRankingResult: ...


def _paper_text(paper: Paper) -> str:
    abstract = paper.abstract.strip() if paper.abstract is not None else ""
    return f"{paper.title}\n{abstract}" if abstract else paper.title


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise EmbeddingUnavailableError("embedding vectors have incompatible dimensions")
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise EmbeddingUnavailableError("embedding vectors must be finite")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


def _failure_warning_code(device: EmbeddingDevice, exc: Exception) -> str:
    if isinstance(exc, EmbeddingOutOfMemoryError):
        return f"{device}_out_of_memory"
    if isinstance(exc, EmbeddingUnavailableError):
        return f"{device}_unavailable"
    message = str(exc).lower()
    if isinstance(exc, MemoryError) or any(hint in message for hint in _OUT_OF_MEMORY_HINTS):
        return f"{device}_out_of_memory"
    return f"{device}_unavailable"


def _translate_encoder_failure(device: EmbeddingDevice, exc: Exception) -> Exception:
    if isinstance(exc, (EmbeddingUnavailableError, EmbeddingOutOfMemoryError)):
        return exc
    if _failure_warning_code(device, exc).endswith("_out_of_memory"):
        return EmbeddingOutOfMemoryError(_failure_warning_code(device, exc))
    return EmbeddingUnavailableError(_failure_warning_code(device, exc))


class EmbeddingRanker:
    def __init__(
        self,
        *,
        encoder_factory: TextEncoderFactory,
        model_id: str,
        preferred_device: EmbeddingDevice,
        batch_size: int,
        fallback_to_cpu: bool,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self._encoder_factory = encoder_factory
        self._model_id = model_id.strip()
        self._preferred_device = preferred_device
        self._batch_size = batch_size
        self._fallback_to_cpu = fallback_to_cpu

    def _rank_with_encoder(
        self,
        query: str,
        papers: Sequence[Paper],
        encoder: TextEncoder,
    ) -> list[EmbeddingScore]:
        query_vectors = encoder.encode([query], batch_size=1)
        if len(query_vectors) != 1:
            raise EmbeddingUnavailableError("query encoder returned an invalid row count")

        document_vectors: list[list[float]] = []
        texts = [_paper_text(paper) for paper in papers]
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            encoded = encoder.encode(batch, batch_size=self._batch_size)
            if len(encoded) != len(batch):
                raise EmbeddingUnavailableError("document encoder returned an invalid row count")
            document_vectors.extend(encoded)

        scored = [
            (
                index,
                EmbeddingScore(
                    paper=paper,
                    similarity=_cosine_similarity(query_vectors[0], document_vector),
                ),
            )
            for index, (paper, document_vector) in enumerate(
                zip(papers, document_vectors, strict=True)
            )
        ]
        scored.sort(key=lambda item: (-item[1].similarity, item[0]))
        return [score for _, score in scored]

    def _rank_on_device(self, query: str, papers: Sequence[Paper], device: EmbeddingDevice) -> list[EmbeddingScore]:
        encoder: TextEncoder | None = None
        try:
            encoder = self._encoder_factory(device)
            return self._rank_with_encoder(query, papers, encoder)
        except Exception as exc:  # noqa: BLE001
            raise _translate_encoder_failure(device, exc) from exc
        finally:
            if encoder is not None:
                try:
                    encoder.close()
                except Exception:  # noqa: BLE001
                    pass

    def _applied_result(
        self,
        ranked: list[EmbeddingScore],
        *,
        device: EmbeddingDevice,
        fallback_used: bool,
        warnings: list[str],
    ) -> EmbeddingRankingResult:
        return EmbeddingRankingResult(
            ranked=ranked,
            status="applied",
            model_id=self._model_id,
            device=device,
            fallback_used=fallback_used,
            warnings=warnings,
        )

    def _degraded_result(
        self,
        papers: Sequence[Paper],
        *,
        warnings: list[str],
    ) -> EmbeddingRankingResult:
        return EmbeddingRankingResult(
            ranked=[
                EmbeddingScore(paper=paper, similarity=0.0)
                for paper in papers
            ],
            status="degraded",
            model_id=self._model_id,
            device="cpu",
            fallback_used=True,
            warnings=warnings,
        )

    def rank(self, query: str, papers: Sequence[Paper]) -> EmbeddingRankingResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not papers:
            return self._applied_result([], device=self._preferred_device, fallback_used=False, warnings=[])

        if self._preferred_device == "cuda" and self._fallback_to_cpu:
            try:
                ranked = self._rank_on_device(query, papers, "cuda")
            except (EmbeddingUnavailableError, EmbeddingOutOfMemoryError) as first_exc:
                try:
                    ranked = self._rank_on_device(query, papers, "cpu")
                except (EmbeddingUnavailableError, EmbeddingOutOfMemoryError) as second_exc:
                    return self._degraded_result(
                        papers,
                        warnings=[
                            _failure_warning_code("cuda", first_exc),
                            _failure_warning_code("cpu", second_exc),
                        ],
                    )
                return self._applied_result(
                    ranked,
                    device="cpu",
                    fallback_used=True,
                    warnings=[_failure_warning_code("cuda", first_exc)],
                )
            return self._applied_result(
                ranked,
                device="cuda",
                fallback_used=False,
                warnings=[],
            )

        try:
            ranked = self._rank_on_device(query, papers, self._preferred_device)
        except (EmbeddingUnavailableError, EmbeddingOutOfMemoryError) as exc:
            if self._preferred_device == "cpu":
                return self._degraded_result(
                    papers,
                    warnings=[_failure_warning_code("cpu", exc)],
                )
            raise
        return self._applied_result(
            ranked,
            device=self._preferred_device,
            fallback_used=False,
            warnings=[],
        )
