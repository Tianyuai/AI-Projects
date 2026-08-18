from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_search.learning.cpu_semantic_router_promotion import (
    run_cpu_semantic_router_promotion,
)
from paper_search.learning.method_route_labels import MethodRouteLabel


def _label(index: int, *, role: str, beneficial: bool) -> MethodRouteLabel:
    return MethodRouteLabel(
        dataset="pasa",
        split="auto_train" if role == "training" else "auto_dev",
        role=role,
        method="semantic",
        query_id=f"{role}-{index}",
        query=(
            f"semantic neural evidence query {index}"
            if beneficial
            else f"exact title lookup query {index}"
        ),
        routing_label="beneficial" if beneficial else "not_beneficial",
        gold_association_count=1,
        marginal_gold_hit_count=1 if beneficial else 0,
        marginal_recall=1.0 if beneficial else 0.0,
        method_action_count=1,
        search_api_calls=1,
    )


def _write_jsonl(path: Path, rows: list[MethodRouteLabel]) -> None:
    path.write_text(
        "".join(
            json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train = [
        _label(index, role="training", beneficial=index % 2 == 0)
        for index in range(20)
    ]
    development = [
        _label(index, role="development", beneficial=index < 4)
        for index in range(10)
    ]
    train_path = tmp_path / "train.jsonl"
    development_path = tmp_path / "development.jsonl"
    manifest_path = tmp_path / "sample.json"
    gate_path = tmp_path / "gates.json"
    _write_jsonl(train_path, train)
    _write_jsonl(development_path, development)
    manifest_path.write_text(
        json.dumps(
            {
                "role": "development",
                "sample_query_count": len(development),
                "sample": [{"query_id": row.query_id} for row in development],
            }
        ),
        encoding="utf-8",
    )
    gate_path.write_text(
        json.dumps(
            {
                "schema_version": "method-router-enablement-gates-v1",
                "semantic": {
                    "minimum_evaluated_queries": 10,
                    "minimum_beneficial_queries": 2,
                    "minimum_availability_rate": 0.95,
                    "minimum_beneficial_recall": 0.0,
                    "minimum_call_reduction": 0.0,
                    "minimum_f1_lift": 0.0,
                    "minimum_marginal_gold_capture": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return train_path, development_path, manifest_path, gate_path


def test_paired_semantic_router_run_is_deterministic_and_seals_gate_evidence(
    tmp_path: Path,
) -> None:
    train, development, sample, gates = _inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    result1 = run_cpu_semantic_router_promotion(
        train_labels_path=train,
        development_labels_path=development,
        development_manifest_path=sample,
        gate_config_path=gates,
        model_path=first / "model.f64",
        model_manifest_path=first / "model.json",
        evaluation_path=first / "evaluation.json",
        dimension=256,
        epochs=4,
        fold_count=5,
    )
    result2 = run_cpu_semantic_router_promotion(
        train_labels_path=train,
        development_labels_path=development,
        development_manifest_path=sample,
        gate_config_path=gates,
        model_path=second / "model.f64",
        model_manifest_path=second / "model.json",
        evaluation_path=second / "evaluation.json",
        dimension=256,
        epochs=4,
        fold_count=5,
    )

    assert result1.model_sha256 == result2.model_sha256
    assert result1.threshold == result2.threshold
    assert (first / "model.f64").read_bytes() == (second / "model.f64").read_bytes()
    assert result1.train_query_count == 20
    assert result1.development_query_count == 10
    assert result1.threshold_selected_on == "training_stratified_5_fold_oof"
    assert result1.final_test_consumed is False


def test_paired_semantic_router_rejects_development_manifest_mismatch(
    tmp_path: Path,
) -> None:
    train, development, sample, gates = _inputs(tmp_path)
    raw = json.loads(sample.read_text(encoding="utf-8"))
    raw["sample"] = raw["sample"][:-1]
    raw["sample_query_count"] -= 1
    sample.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="development labels do not match"):
        run_cpu_semantic_router_promotion(
            train_labels_path=train,
            development_labels_path=development,
            development_manifest_path=sample,
            gate_config_path=gates,
            model_path=tmp_path / "model.f64",
            model_manifest_path=tmp_path / "model.json",
            evaluation_path=tmp_path / "evaluation.json",
            dimension=256,
            epochs=4,
            fold_count=5,
        )
