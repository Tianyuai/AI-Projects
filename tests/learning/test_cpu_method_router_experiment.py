from __future__ import annotations

import json
from pathlib import Path

from paper_search.learning.cpu_method_router_experiment import (
    run_cpu_method_router_experiment,
)
from paper_search.learning.graph_method_labels import GraphMethodLabel
from paper_search.learning.method_route_labels import MethodRouteLabel


def _write(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(
            json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _route_label(
    query_id: str,
    label: str,
    *,
    role: str,
    method: str = "semantic",
    seed_count: int = 0,
) -> MethodRouteLabel:
    return MethodRouteLabel(
        dataset="pasa",
        split="auto_train" if role == "training" else "auto_dev",
        role=role,
        method=method,
        query_id=query_id,
        query=("neural semantic retrieval " if label == "beneficial" else "exact title ")
        + query_id,
        routing_label=label,
        gold_association_count=1,
        marginal_gold_hit_count=1 if label == "beneficial" else 0,
        marginal_recall=1.0 if label == "beneficial" else 0.0,
        seed_count=seed_count,
        method_action_count=1,
        search_api_calls=1,
    )


def _graph_source(row: MethodRouteLabel) -> GraphMethodLabel:
    return GraphMethodLabel(
        dataset=row.dataset,
        split=row.split,
        role=row.role,
        query_id=row.query_id,
        query=row.query,
        routing_label=row.routing_label,
        gold_association_count=1,
        graph_marginal_gold_hit_ids=("doi:x",)
        if row.routing_label == "beneficial"
        else (),
        graph_marginal_recall=row.marginal_recall,
        seed_count=row.seed_count,
        graph_action_count=1,
        search_api_calls=1,
    )


def test_experiment_trains_two_heads_and_writes_hash_manifest(tmp_path: Path) -> None:
    semantic_train = [
        _route_label(str(index), "beneficial" if index % 2 else "not_beneficial", role="training")
        for index in range(8)
    ]
    calibration = [
        _route_label(f"c{index}", "beneficial" if index % 2 else "not_beneficial", role="development")
        for index in range(4)
    ]
    evaluation = [
        _route_label(f"e{index}", "beneficial" if index % 2 else "not_beneficial", role="development")
        for index in range(4)
    ]
    graph = [
        _graph_source(
            _route_label(
                f"g{index}",
                "beneficial" if index % 2 else "not_beneficial",
                role="training",
                method="graph",
                seed_count=8 if index % 2 else 0,
            )
        )
        for index in range(8)
    ]
    paths = {
        "train": tmp_path / "semantic-train.jsonl",
        "calibration": tmp_path / "semantic-calibration.jsonl",
        "evaluation": tmp_path / "semantic-evaluation.jsonl",
        "graph": tmp_path / "graph.jsonl",
    }
    _write(paths["train"], semantic_train)
    _write(paths["calibration"], calibration)
    _write(paths["evaluation"], evaluation)
    _write(paths["graph"], graph)

    manifest = run_cpu_method_router_experiment(
        semantic_train_path=paths["train"],
        semantic_calibration_path=paths["calibration"],
        semantic_evaluation_path=paths["evaluation"],
        graph_train_path=paths["graph"],
        semantic_model_path=tmp_path / "semantic.f64",
        graph_model_path=tmp_path / "graph.f64",
        manifest_path=tmp_path / "manifest.json",
        dimension=256,
        epochs=4,
        graph_folds=2,
    )

    assert manifest.semantic.train_count == 8
    assert manifest.semantic.evaluation.evaluated_count == 4
    assert manifest.graph.train_count == 8
    assert manifest.graph.validation_scheme == "stratified_out_of_fold_calibration_only"
    assert manifest.graph.deployable is False
    assert manifest.semantic.deployable is False
    assert manifest.semantic.model_sha256.startswith("sha256:")
    assert manifest.graph.model_sha256.startswith("sha256:")
