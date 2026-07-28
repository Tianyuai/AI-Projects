"""Deterministic embedding ranking with an injected local encoder."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, Paper

EmbeddingDevice = Literal["cpu", "cuda"]
EmbeddingStatus = Literal["applied", "degraded"]
_SAFE_MODEL_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?$"
)
_PUBLIC_WARNING_CODES = frozenset(
    {
        "cpu_encoder_unavailable",
        "cuda_oom_cpu_fallback",
        "cuda_unavailable_cpu_fallback",
        "encoder_out_of_memory",
        "encoder_unavailable",
        "unsanitized_warning",
    }
)


def sanitize_embedding_model_id(model_id: str) -> str:
    """Return a public-safe embedding model identifier."""
    candidate = model_id.strip()
    if "/" in candidate or "\\" in candidate:
        return "local_model"
    return candidate if _SAFE_MODEL_ID_RE.fullmatch(candidate) else "local_model"


def sanitize_embedding_warnings(warnings: Sequence[str]) -> list[str]:
    """Return bounded public warning codes."""
    sanitized: list[str] = []
    for warning in warnings:
        candidate = warning.strip()
        sanitized.append(candidate if candidate in _PUBLIC_WARNING_CODES else "unsanitized_warning")
    return sanitized


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

    def _rank_on_device(
        self,
        query: str,
        papers: Sequence[Paper],
        device: EmbeddingDevice,
    ) -> list[EmbeddingScore]:
        encoder: TextEncoder | None = None
        try:
            encoder = self._encoder_factory(device)
            return self._rank_with_encoder(query, papers, encoder)
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

    def _degraded(
        self,
        papers: Sequence[Paper],
        *,
        device: EmbeddingDevice,
        warnings: list[str],
        fallback_used: bool,
    ) -> EmbeddingRankingResult:
        return EmbeddingRankingResult(
            ranked=[EmbeddingScore(paper=paper, similarity=0.0) for paper in papers],
            status="degraded",
            model_id=self._model_id,
            device=device,
            fallback_used=fallback_used,
            warnings=warnings,
        )

    def rank(self, query: str, papers: Sequence[Paper]) -> EmbeddingRankingResult:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("query must not be empty")
        if not papers:
            return EmbeddingRankingResult(
                ranked=[],
                status="applied",
                model_id=self._model_id,
                device=self._preferred_device,
                fallback_used=False,
                warnings=[],
            )
        try:
            ranked = self._rank_on_device(
                normalized_query,
                papers,
                self._preferred_device,
            )
        except (EmbeddingOutOfMemoryError, EmbeddingUnavailableError) as error:
            if self._preferred_device != "cuda" or not self._fallback_to_cpu:
                warning = (
                    "encoder_out_of_memory"
                    if isinstance(error, EmbeddingOutOfMemoryError)
                    else "encoder_unavailable"
                )
                return self._degraded(
                    papers,
                    device=self._preferred_device,
                    warnings=[warning],
                    fallback_used=False,
                )
            warning = (
                "cuda_oom_cpu_fallback"
                if isinstance(error, EmbeddingOutOfMemoryError)
                else "cuda_unavailable_cpu_fallback"
            )
            try:
                ranked = self._rank_on_device(normalized_query, papers, "cpu")
            except (EmbeddingOutOfMemoryError, EmbeddingUnavailableError):
                return self._degraded(
                    papers,
                    device="cpu",
                    warnings=[warning, "cpu_encoder_unavailable"],
                    fallback_used=True,
                )
            return EmbeddingRankingResult(
                ranked=ranked,
                status="applied",
                model_id=self._model_id,
                device="cpu",
                fallback_used=True,
                warnings=[warning],
            )
        return EmbeddingRankingResult(
            ranked=ranked,
            status="applied",
            model_id=self._model_id,
            device=self._preferred_device,
            fallback_used=False,
            warnings=[],
        )
