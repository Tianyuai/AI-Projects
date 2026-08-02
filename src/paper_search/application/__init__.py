"""Stable application contracts shared by replay and live execution paths."""

from importlib import import_module
from typing import Any

from paper_search.application.contracts import (
    DependencyDiagnostic,
    ReadyHealthResponse,
    SearchErrorResponse,
    SearchErrorCode,
    SearchExecutionResult,
    SearchFailure,
    SearchOutcome,
    SearchRequest,
    SearchSuccess,
    SnapshotRef,
)
from paper_search.application.locks import (
    ArtifactBinding,
    CandidateLock,
    InputLock,
    ReplayLock,
    ValidationLock,
    canonical_lock_bytes,
    load_input_lock,
    lock_sha256,
)
from paper_search.application.modes import ModeBinding
from paper_search.domain.models import (
    DependencyErrorCode,
    DependencyName,
    DependencyState,
    DependencyStatus,
    MoneyCny,
    PlannerStatus,
    SafeRelativePath,
    SearchMode,
    Sha256,
    StructuredSearchResponse,
)

_LAZY_EXPORTS = {
    "ApplicationBundle": ("paper_search.application.composition", "ApplicationBundle"),
    "ArtifactFactory": ("paper_search.application.artifacts", "ArtifactFactory"),
    "CaptureSession": ("paper_search.application.artifacts", "CaptureSession"),
    "CompositionRoot": ("paper_search.application.composition", "CompositionRoot"),
    "SearchApplicationError": (
        "paper_search.application.service",
        "SearchApplicationError",
    ),
    "SearchApplicationService": (
        "paper_search.application.service",
        "SearchApplicationService",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve runtime-heavy public boundaries without import cycles."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "ApplicationBundle",
    "ArtifactFactory",
    "DependencyDiagnostic",
    "DependencyErrorCode",
    "DependencyName",
    "DependencyState",
    "DependencyStatus",
    "ArtifactBinding",
    "CandidateLock",
    "CaptureSession",
    "CompositionRoot",
    "InputLock",
    "MoneyCny",
    "ModeBinding",
    "PlannerStatus",
    "ReadyHealthResponse",
    "SafeRelativePath",
    "SearchErrorResponse",
    "SearchErrorCode",
    "SearchApplicationError",
    "SearchApplicationService",
    "SearchExecutionResult",
    "SearchFailure",
    "SearchMode",
    "SearchOutcome",
    "SearchRequest",
    "SearchSuccess",
    "Sha256",
    "SnapshotRef",
    "StructuredSearchResponse",
    "ReplayLock",
    "ValidationLock",
    "canonical_lock_bytes",
    "load_input_lock",
    "lock_sha256",
]
