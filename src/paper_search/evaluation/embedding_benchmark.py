"""Safe aggregate measurement for local embedding ranking."""

from __future__ import annotations

import ctypes
import re
import sys
import time
from collections.abc import Callable, Sequence
from importlib import import_module

from pydantic import Field

from paper_search.domain.models import DomainModel, NonEmptyStr, Paper
from paper_search.ranking.embedding import (
    EmbeddingDevice,
    EmbeddingRankingStage,
    EmbeddingStatus,
)

_SAFE_MODEL_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?$"
)
_SAFE_WARNING_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_LOCAL_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2}|~[\\/]|\.{1,2}[\\/])")


class EmbeddingBenchmarkResult(DomainModel):
    model_id: NonEmptyStr
    device: EmbeddingDevice
    candidate_count: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    latency_ms: int = Field(ge=0)
    process_peak_rss_bytes: int = Field(ge=0)
    cuda_peak_allocated_bytes: int | None = Field(default=None, ge=0)
    status: EmbeddingStatus
    fallback_used: bool
    warnings: list[NonEmptyStr]


class _WindowsProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def _validate_batch_size(batch_size: object) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return batch_size


def _sanitize_model_id(model_id: str) -> str:
    candidate = model_id.strip()
    if _LOCAL_PATH_RE.match(candidate) or "/" in candidate or "\\" in candidate:
        return "local_model"
    if not _SAFE_MODEL_ID_RE.fullmatch(candidate):
        raise ValueError("model_id must be a safe identifier")
    return candidate


def _sanitize_warnings(warnings: Sequence[str]) -> list[str]:
    return [
        warning if _SAFE_WARNING_RE.fullmatch(warning.strip()) else "unsanitized_warning"
        for warning in warnings
    ]


def process_peak_rss_bytes() -> int:
    """Return process peak resident memory using only the standard library."""
    if sys.platform == "win32":
        counters = _WindowsProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_WindowsProcessMemoryCounters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        succeeded = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not succeeded:
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return int(counters.peak_working_set_size)

    resource = import_module("resource")
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def reset_cuda_peak_memory() -> None:
    """Reset CUDA peak allocation when CUDA is available."""
    try:
        torch = import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_allocated_bytes() -> int | None:
    """Return CUDA peak allocation, or None when CUDA is unavailable."""
    try:
        torch = import_module("torch")
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def benchmark_embedding(
    *,
    ranker: EmbeddingRankingStage,
    query: str,
    papers: Sequence[Paper],
    batch_size: int,
    clock: Callable[[], float] = time.perf_counter,
    peak_rss: Callable[[], int] = process_peak_rss_bytes,
    cuda_peak: Callable[[], int | None] = cuda_peak_allocated_bytes,
    cuda_reset: Callable[[], None] = reset_cuda_peak_memory,
) -> EmbeddingBenchmarkResult:
    validated_batch_size = _validate_batch_size(batch_size)
    cuda_reset()
    started = clock()
    ranking = ranker.rank(query, papers)
    latency_ms = max(0, round((clock() - started) * 1000))
    return EmbeddingBenchmarkResult(
        model_id=_sanitize_model_id(ranking.model_id),
        device=ranking.device,
        candidate_count=len(papers),
        batch_size=validated_batch_size,
        latency_ms=latency_ms,
        process_peak_rss_bytes=peak_rss(),
        cuda_peak_allocated_bytes=cuda_peak(),
        status=ranking.status,
        fallback_used=ranking.fallback_used,
        warnings=_sanitize_warnings(ranking.warnings),
    )
