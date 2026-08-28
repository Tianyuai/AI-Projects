from __future__ import annotations

from copy import deepcopy

from paper_search.application.production_ranker_binding import (
    bind_production_ranker_selection,
)


def test_selection_binds_f5_f4_b0_without_mutating_source_lock() -> None:
    source = {"baseline": {"document_ranker": None}}
    original = deepcopy(source)
    selection = {
        "schema_version": "production-document-ranker-selection-v2",
        "production_default": "F5-gated-fusion",
        "production_fallback": "F4-reliability",
        "emergency_fallback": "B0",
        "runtime_failover_order": ["F5-gated-fusion", "F4-reliability", "B0"],
        "per_query_model_switching": False,
        "default_manifest": "gated/manifest.json",
        "default_manifest_sha256": "sha256:" + "1" * 64,
        "default_weights": "gated/weights.bundle",
        "default_weights_sha256": "sha256:" + "2" * 64,
        "fallback_manifest": "f4/manifest.json",
        "fallback_manifest_sha256": "sha256:" + "3" * 64,
        "fallback_weights": "f4/weights.bundle",
        "fallback_weights_sha256": "sha256:" + "4" * 64,
        "emergency_manifest": "b0/manifest.json",
        "emergency_manifest_sha256": "sha256:" + "5" * 64,
        "emergency_weights": "b0/weights.f64",
        "emergency_weights_sha256": "sha256:" + "6" * 64,
        "test_partition_touched": False,
    }

    bound = bind_production_ranker_selection(
        source, selection, selection_root="artifacts/models"
    )

    assert source == original
    assert bound["baseline"]["document_ranker"] == {
        "enabled": True,
        "manifest": {
            "path": "artifacts/models/gated/manifest.json",
            "sha256": "sha256:" + "1" * 64,
        },
        "weights": {
            "path": "artifacts/models/gated/weights.bundle",
            "sha256": "sha256:" + "2" * 64,
        },
        "fallback_manifest": {
            "path": "artifacts/models/f4/manifest.json",
            "sha256": "sha256:" + "3" * 64,
        },
        "fallback_weights": {
            "path": "artifacts/models/f4/weights.bundle",
            "sha256": "sha256:" + "4" * 64,
        },
        "emergency_manifest": {
            "path": "artifacts/models/b0/manifest.json",
            "sha256": "sha256:" + "5" * 64,
        },
        "emergency_weights": {
            "path": "artifacts/models/b0/weights.f64",
            "sha256": "sha256:" + "6" * 64,
        },
    }
