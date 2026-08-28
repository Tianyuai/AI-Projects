"""Hash-bound persistence for the promoted supervised lexical bridge."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.lexical_bridge import SupervisedLexicalBridge


_SCHEMA_VERSION = "supervised-lexical-bridge-deployment-v2"
_MODEL_ID = "supervised-lexical-bridge-openalex-v2"
_CONFIGURATION = {
    "learning_objective": "neighbor_idf",
    "max_expansion_terms": 6,
    "min_neighbor_support": 2,
    "neighbors": 12,
    "representation": "word_char",
}


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class LoadedLexicalBridge:
    bridge: SupervisedLexicalBridge
    source_sha256: str
    max_expansion_terms: int
    neighbors: int
    min_neighbor_support: int
    manifest: dict[str, Any]


def freeze_lexical_bridge_model(
    bridge: SupervisedLexicalBridge,
    *,
    model_path: Path,
    manifest_path: Path,
    training_query_count: int,
    raw_train_sha256: str,
    train_partition_sha256: str,
    training_oof_sha256: str,
    independent_dev_sha256: str,
) -> dict[str, Any]:
    if training_query_count <= 0:
        raise ValueError("training query count must be positive")
    if (
        bridge.representation != _CONFIGURATION["representation"]
        or bridge.learning_objective != _CONFIGURATION["learning_objective"]
    ):
        raise ValueError("lexical bridge does not match promoted configuration")
    buffer = io.BytesIO()
    joblib.dump(bridge, buffer, compress=3, protocol=5)
    model_bytes = buffer.getvalue()
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "model_id": _MODEL_ID,
        "target_provider": "openalex",
        "training_query_count": training_query_count,
        "configuration": dict(_CONFIGURATION),
        "model_sha256": _sha256(model_bytes),
        "inputs": {
            "raw_train_sha256": raw_train_sha256,
            "train_partition_sha256": train_partition_sha256,
        },
        "promotion_evidence": {
            "training_oof_sha256": training_oof_sha256,
            "independent_dev_sha256": independent_dev_sha256,
        },
        "test_partition_touched": False,
    }
    write_frozen_bytes(model_path, model_bytes)
    write_frozen_bytes(manifest_path, _canonical_bytes(manifest))
    return manifest


def load_lexical_bridge_model(
    *,
    model_path: Path,
    manifest_path: Path,
) -> LoadedLexicalBridge:
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError:
        raise ValueError("invalid lexical bridge deployment manifest") from None
    try:
        model_bytes = model_path.read_bytes()
    except OSError:
        raise ValueError("lexical bridge artifact is unavailable") from None
    return load_lexical_bridge_model_bytes(
        model_bytes=model_bytes,
        manifest_bytes=manifest_bytes,
    )


def load_lexical_bridge_model_bytes(
    *,
    model_bytes: bytes,
    manifest_bytes: bytes,
) -> LoadedLexicalBridge:
    """Restore a bridge from the exact bytes retained by an input lock."""

    try:
        manifest = json.loads(manifest_bytes)
        if manifest["schema_version"] != _SCHEMA_VERSION:
            raise ValueError
        if manifest["model_id"] != _MODEL_ID:
            raise ValueError
        if manifest["target_provider"] != "openalex":
            raise ValueError
        if manifest["configuration"] != _CONFIGURATION:
            raise ValueError
        if manifest["test_partition_touched"] is not False:
            raise ValueError
        expected_hash = str(manifest["model_sha256"])
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise ValueError("invalid lexical bridge deployment manifest") from None
    if _sha256(model_bytes) != expected_hash:
        raise ValueError("lexical bridge artifact hash mismatch")
    try:
        bridge = joblib.load(io.BytesIO(model_bytes))
    except Exception:
        raise ValueError("lexical bridge artifact is invalid") from None
    if not isinstance(bridge, SupervisedLexicalBridge):
        raise ValueError("lexical bridge artifact has an invalid type")
    if (
        bridge.representation != _CONFIGURATION["representation"]
        or bridge.learning_objective != _CONFIGURATION["learning_objective"]
    ):
        raise ValueError("lexical bridge artifact configuration mismatch")
    return LoadedLexicalBridge(
        bridge=bridge,
        source_sha256=expected_hash,
        max_expansion_terms=cast(int, _CONFIGURATION["max_expansion_terms"]),
        neighbors=cast(int, _CONFIGURATION["neighbors"]),
        min_neighbor_support=cast(int, _CONFIGURATION["min_neighbor_support"]),
        manifest=manifest,
    )


__all__ = [
    "LoadedLexicalBridge",
    "freeze_lexical_bridge_model",
    "load_lexical_bridge_model",
    "load_lexical_bridge_model_bytes",
]
