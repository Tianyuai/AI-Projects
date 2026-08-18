from __future__ import annotations

import hashlib

import pytest

from paper_search.learning.model_bundle import ModelBundleManifest, verify_model_bundle


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def test_model_bundle_verifies_every_declared_artifact(tmp_path) -> None:
    weights = b"weights"
    tokenizer = b"tokenizer"
    (tmp_path / "weights.bin").write_bytes(weights)
    (tmp_path / "tokenizer.json").write_bytes(tokenizer)
    manifest = ModelBundleManifest(
        model_id="action-ranker-v1",
        base_model="sentence-transformers/example",
        framework="transformers",
        training_data_manifest_sha256="sha256:" + "1" * 64,
        confidence_threshold=0.6,
        device_priority=["cuda", "cpu"],
        license="apache-2.0",
        artifacts={
            "weights.bin": _sha256(weights),
            "tokenizer.json": _sha256(tokenizer),
        },
    )

    verified = verify_model_bundle(tmp_path, manifest)

    assert verified.model_id == "action-ranker-v1"
    assert verified.verified_artifacts == ["tokenizer.json", "weights.bin"]
    assert verified.bundle_sha256.startswith("sha256:")


def test_model_bundle_fails_closed_on_artifact_hash_mismatch(tmp_path) -> None:
    (tmp_path / "weights.bin").write_bytes(b"changed")
    manifest = ModelBundleManifest(
        model_id="action-ranker-v1",
        base_model="sentence-transformers/example",
        framework="transformers",
        training_data_manifest_sha256="sha256:" + "1" * 64,
        confidence_threshold=0.6,
        device_priority=["cpu"],
        license="apache-2.0",
        artifacts={"weights.bin": _sha256(b"expected")},
    )

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_model_bundle(tmp_path, manifest)
