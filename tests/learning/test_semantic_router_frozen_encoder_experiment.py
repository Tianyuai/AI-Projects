from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import compare_semantic_router_frozen_encoder_oof as experiment  # noqa: E402


def test_new_embedding_cache_returns_the_same_values_as_reload(
    monkeypatch: object, tmp_path: Path
) -> None:
    rows = [SimpleNamespace(query_id="query-1")]
    encoded = (
        np.asarray([[1.0 / 3.0, 2.0 / 3.0]], dtype=np.float64),
        np.asarray([[1.0 / 7.0]], dtype=np.float64),
        np.asarray([[1.0 / 11.0]], dtype=np.float64),
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        experiment, "_encode_features", lambda **_: encoded
    )
    cache_path = tmp_path / "features.npz"

    created = experiment._load_or_create_cache(
        cache_path=cache_path,
        rows=rows,
        action_hits_by_query={},
    )
    loaded = experiment._load_or_create_cache(
        cache_path=cache_path,
        rows=rows,
        action_hits_by_query={},
    )

    for created_values, loaded_values in zip(created[:3], loaded[:3], strict=True):
        assert np.array_equal(created_values, loaded_values)
    assert created[3] == loaded[3]
