from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paper_search.learning.cpu_method_router import CpuMethodRouter
from paper_search.learning.method_route_labels import MethodRouteLabel


def _label(
    query_id: str,
    query: str,
    routing_label: str,
    *,
    method: str = "semantic",
    role: str = "training",
    seed_count: int = 0,
) -> MethodRouteLabel:
    return MethodRouteLabel(
        dataset="pasa",
        split="auto_train" if role == "training" else "auto_dev",
        role=role,
        method=method,
        query_id=query_id,
        query=query,
        routing_label=routing_label,
        gold_association_count=2,
        marginal_gold_hit_count=1 if routing_label == "beneficial" else 0,
        marginal_recall=0.5 if routing_label == "beneficial" else 0.0,
        seed_count=seed_count,
        method_action_count=1,
        search_api_calls=1,
    )


def test_router_trains_only_matching_available_training_labels() -> None:
    rows = [
        _label("a", "graph neural retrieval", "beneficial"),
        _label("b", "classical database indexing", "not_beneficial"),
        _label("c", "unreachable", "unavailable"),
        _label("d", "graph expansion", "beneficial", method="graph"),
    ]
    router = CpuMethodRouter(method="semantic", dimension=256, epochs=3)

    assert router.fit(rows) == 2
    assert 0.0 <= router.predict_proba("graph neural retrieval") <= 1.0


def test_router_rejects_development_labels_during_fit() -> None:
    router = CpuMethodRouter(method="semantic")
    with pytest.raises(ValueError, match="development"):
        router.fit([_label("a", "query", "beneficial", role="development")])


def test_graph_router_uses_seed_context() -> None:
    rows = [
        _label("a", "same query", "not_beneficial", method="graph", seed_count=0),
        _label("b", "same query", "beneficial", method="graph", seed_count=8),
    ]
    router = CpuMethodRouter(method="graph", dimension=256, epochs=12, seed=5)
    router.fit(rows)

    assert router.predict_proba("same query", seed_count=8) > router.predict_proba(
        "same query", seed_count=0
    )


def test_router_save_load_is_byte_deterministic(tmp_path: Path) -> None:
    rows = [
        _label("a", "semantic scholarly search", "beneficial"),
        _label("b", "exact title lookup", "not_beneficial"),
    ]
    first = CpuMethodRouter(method="semantic", dimension=256, epochs=4, seed=7)
    second = CpuMethodRouter(method="semantic", dimension=256, epochs=4, seed=7)
    first.fit(rows)
    second.fit(rows)
    first_path = tmp_path / "first.f64"
    second_path = tmp_path / "second.f64"
    first.save(first_path)
    second.save(second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    loaded = CpuMethodRouter.load(
        first_path,
        method="semantic",
        dimension=256,
        epochs=4,
        seed=7,
    )
    assert np.array_equal(first.weights, loaded.weights)
    assert loaded.predict_proba("semantic scholarly search") == first.predict_proba(
        "semantic scholarly search"
    )


def test_router_load_fails_closed_on_wrong_dimension(tmp_path: Path) -> None:
    path = tmp_path / "weights.f64"
    CpuMethodRouter(method="semantic", dimension=32).save(path)
    with pytest.raises(ValueError, match="weight size mismatch"):
        CpuMethodRouter.load(path, method="semantic", dimension=64)
