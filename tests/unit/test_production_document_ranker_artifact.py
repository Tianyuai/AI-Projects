from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paper_search.learning.gated_feature_fusion_ranker import (
    UnifiedFusionContextResolver,
)
from paper_search.ranking.cpu_document import load_cpu_document_ranking_stage


PROJECT_ROOT = Path(__file__).parents[2]
FALLBACK_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "cpu-pairwise-document-ranker-expanded2385-v1"
)


def test_selected_production_artifact_is_promoted_f5_and_deployable() -> None:
    selection_path = (
        PROJECT_ROOT / "artifacts/models/production-document-ranker-selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    models_root = selection_path.parent
    manifest_path = models_root / selection["default_manifest"]
    weights_path = models_root / selection["default_weights"]
    fallback_manifest_path = models_root / selection["fallback_manifest"]
    fallback_weights_path = models_root / selection["fallback_weights"]
    emergency_manifest_path = models_root / selection["emergency_manifest"]
    emergency_weights_path = models_root / selection["emergency_weights"]

    stage = load_cpu_document_ranking_stage(manifest_path, weights_path)
    fallback_stage = load_cpu_document_ranking_stage(
        fallback_manifest_path, fallback_weights_path
    )
    emergency_stage = load_cpu_document_ranking_stage(
        emergency_manifest_path, emergency_weights_path
    )

    assert selection["production_default"] == "F5-gated-fusion"
    assert selection["production_fallback"] == "F4-reliability"
    assert selection["emergency_fallback"] == "B0"
    assert selection["runtime_failover_order"] == [
        "F5-gated-fusion",
        "F4-reliability",
        "B0",
    ]
    assert selection["per_query_model_switching"] is False
    assert selection["test_partition_touched"] is False
    assert selection["default_manifest_sha256"] == (
        "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert selection["default_weights_sha256"] == (
        "sha256:" + hashlib.sha256(weights_path.read_bytes()).hexdigest()
    )
    assert selection["fallback_manifest_sha256"] == (
        "sha256:" + hashlib.sha256(fallback_manifest_path.read_bytes()).hexdigest()
    )
    assert selection["fallback_weights_sha256"] == (
        "sha256:" + hashlib.sha256(fallback_weights_path.read_bytes()).hexdigest()
    )
    assert selection["emergency_manifest_sha256"] == (
        "sha256:" + hashlib.sha256(emergency_manifest_path.read_bytes()).hexdigest()
    )
    assert selection["emergency_weights_sha256"] == (
        "sha256:" + hashlib.sha256(emergency_weights_path.read_bytes()).hexdigest()
    )
    assert stage.model_id == "gated-feature-fusion-document-ranker-research-v1"
    assert isinstance(stage.ranker.context_store, UnifiedFusionContextResolver)
    assert stage.ranker.runtime_context_scoring is True
    assert json.loads(manifest_path.read_text(encoding="utf-8"))[
        "training_query_count"
    ] == 18_314
    assert fallback_stage.ranker.feature_families == frozenset({"reliability"})
    assert emergency_stage.model_id == "cpu-pairwise-document-ranker-v1"


def test_expanded2385_fallback_artifact_is_deployable_and_promotion_bound() -> None:
    manifest_path = FALLBACK_ARTIFACT_ROOT / "manifest.json"
    weights_path = FALLBACK_ARTIFACT_ROOT / "weights.f64"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    stage = load_cpu_document_ranking_stage(manifest_path, weights_path)
    weights_sha256 = "sha256:" + hashlib.sha256(weights_path.read_bytes()).hexdigest()

    assert stage.model_id == "cpu-pairwise-document-ranker-v1"
    assert manifest["model_sha256"] == weights_sha256
    assert manifest["replacement_authorized"] is True
    assert manifest["test_partition_touched"] is False
    assert manifest["training_query_count"] == 2385
