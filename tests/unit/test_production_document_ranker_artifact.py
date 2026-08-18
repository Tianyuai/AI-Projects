from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paper_search.ranking.cpu_document import load_cpu_document_ranking_stage


PROJECT_ROOT = Path(__file__).parents[2]
ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "cpu-pairwise-document-ranker-expanded2385-v1"
)


def test_expanded2385_production_artifact_is_deployable_and_promotion_bound() -> None:
    manifest_path = ARTIFACT_ROOT / "manifest.json"
    weights_path = ARTIFACT_ROOT / "weights.f64"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    stage = load_cpu_document_ranking_stage(manifest_path, weights_path)
    weights_sha256 = "sha256:" + hashlib.sha256(weights_path.read_bytes()).hexdigest()

    assert stage.model_id == "cpu-pairwise-document-ranker-v1"
    assert manifest["model_sha256"] == weights_sha256
    assert manifest["replacement_authorized"] is True
    assert manifest["test_partition_touched"] is False
    assert manifest["training_query_count"] == 2385
