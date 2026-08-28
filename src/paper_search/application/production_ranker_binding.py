"""Bind the frozen F5 -> F4 -> B0 selection into an input lock."""

from __future__ import annotations

from copy import deepcopy
from pathlib import PurePosixPath
from collections.abc import Mapping
from typing import Any, cast


_ARTIFACT_FIELDS = (
    ("manifest", "default_manifest", "default_manifest_sha256"),
    ("weights", "default_weights", "default_weights_sha256"),
    ("fallback_manifest", "fallback_manifest", "fallback_manifest_sha256"),
    ("fallback_weights", "fallback_weights", "fallback_weights_sha256"),
    ("emergency_manifest", "emergency_manifest", "emergency_manifest_sha256"),
    ("emergency_weights", "emergency_weights", "emergency_weights_sha256"),
)


def _relative_path(root: str, value: object) -> str:
    root_path = PurePosixPath(root)
    path = PurePosixPath(str(value))
    combined = root_path / path
    if combined.is_absolute() or ".." in combined.parts:
        raise ValueError("production ranker artifact path must stay relative")
    return combined.as_posix()


def bind_production_ranker_selection(
    lock_payload: Mapping[str, Any],
    selection: Mapping[str, object],
    *,
    selection_root: str,
) -> dict[str, Any]:
    """Return a copied lock with the immutable three-level ranker chain bound."""

    if selection.get("schema_version") != "production-document-ranker-selection-v2":
        raise ValueError("unsupported production document ranker selection")
    expected_order = ["F5-gated-fusion", "F4-reliability", "B0"]
    if selection.get("runtime_failover_order") != expected_order:
        raise ValueError("production ranker failover order is not F5, F4, B0")
    if selection.get("per_query_model_switching") is not False:
        raise ValueError("production ranker selection forbids per-query switching")
    if selection.get("test_partition_touched") is not False:
        raise ValueError("production ranker selection touched the test partition")

    binding: dict[str, object] = {"enabled": True}
    for output_name, path_name, hash_name in _ARTIFACT_FIELDS:
        path = selection.get(path_name)
        digest = selection.get(hash_name)
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError(f"production ranker selection is missing {path_name}")
        binding[output_name] = {
            "path": _relative_path(selection_root, path),
            "sha256": digest,
        }

    output = deepcopy(dict(lock_payload))
    baseline = output.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("input lock is missing baseline configuration")
    cast(dict[str, object], baseline)["document_ranker"] = binding
    return output


__all__ = ["bind_production_ranker_selection"]
