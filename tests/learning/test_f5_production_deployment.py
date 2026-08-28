from __future__ import annotations

import json
import struct

from paper_search.learning.f5_production_deployment import (
    _extract_reliability_artifact,
)


def test_derived_f4_experiment_id_uses_source_training_query_count() -> None:
    manifest = {
        "dimension_per_family": 2,
        "family_caps": {"reliability": 0.08},
        "training_query_count": 18_314,
        "training_query_count_by_family": {"reliability": 6_361},
        "preference_pair_count_by_family": {"reliability": 761_665},
    }
    family = b"reliability"
    weights = struct.pack("<I", len(family)) + family + (b"\0" * 16)

    derived, _ = _extract_reliability_artifact(
        (json.dumps(manifest) + "\n").encode(), weights
    )

    assert derived["experiment_id"] == (
        "S4-F4-reliability-18314-production-fallback"
    )
