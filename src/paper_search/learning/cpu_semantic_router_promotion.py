"""Deterministic CPU training and one-shot promotion for the semantic router."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field

from paper_search.domain.models import DomainModel, Sha256, UnitFloat
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.cpu_baseline import (
    BinaryMetrics,
    evaluate_probabilities,
    select_f1_threshold,
)
from paper_search.learning.cpu_method_router import CpuMethodRouter
from paper_search.learning.method_route_labels import MethodRouteLabel
from paper_search.learning.method_router_gate import (
    MethodRouterGate,
    MethodRouterGateDecision,
    assess_method_router,
)


class SemanticRouterPromotionResult(DomainModel):
    schema_version: Literal["cpu-semantic-router-promotion-v1"] = (
        "cpu-semantic-router-promotion-v1"
    )
    model_id: str = CpuMethodRouter.model_id
    dimension: int = Field(strict=True, gt=0)
    epochs: int = Field(strict=True, gt=0)
    learning_rate: float = Field(gt=0)
    l2: float = Field(ge=0)
    seed: int
    fold_count: int = Field(strict=True, ge=2)
    train_query_count: int = Field(strict=True, gt=0)
    beneficial_train_query_count: int = Field(strict=True, gt=0)
    unavailable_train_query_count: int = Field(strict=True, ge=0)
    development_query_count: int = Field(strict=True, gt=0)
    threshold: UnitFloat
    threshold_selected_on: str
    oof_calibration: BinaryMetrics
    promotion: MethodRouterGateDecision
    model_sha256: Sha256
    model_manifest_sha256: Sha256
    train_labels_sha256: Sha256
    development_labels_sha256: Sha256
    development_manifest_sha256: Sha256
    gate_config_sha256: Sha256
    final_test_consumed: Literal[False] = False


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


def _load_labels(path: Path) -> tuple[list[MethodRouteLabel], bytes]:
    content = path.read_bytes()
    rows = [
        MethodRouteLabel.model_validate_json(line)
        for line in content.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"method route label file is empty: {path}")
    if any(row.method != "semantic" for row in rows):
        raise ValueError("semantic router labels contain another method")
    if len({row.query_id for row in rows}) != len(rows):
        raise ValueError("semantic router labels contain duplicate queries")
    return rows, content


def _folds(
    rows: list[MethodRouteLabel], *, fold_count: int, seed: int
) -> list[list[MethodRouteLabel]]:
    grouped: dict[bool, list[MethodRouteLabel]] = defaultdict(list)
    for row in rows:
        grouped[row.routing_label == "beneficial"].append(row)
    if set(grouped) != {False, True} or any(
        len(group) < fold_count for group in grouped.values()
    ):
        raise ValueError("each semantic class must contain one row per fold")
    folds: list[list[MethodRouteLabel]] = [[] for _ in range(fold_count)]
    for label, group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda row: hashlib.sha256(
                f"{seed}:{label}:{row.query_id}".encode("utf-8")
            ).digest(),
        )
        for index, row in enumerate(ordered):
            folds[index % fold_count].append(row)
    return folds


def _probabilities(
    router: CpuMethodRouter, rows: list[MethodRouteLabel]
) -> list[float]:
    return [router.predict_proba(row.query) for row in rows]


def run_cpu_semantic_router_promotion(
    *,
    train_labels_path: Path,
    development_labels_path: Path,
    development_manifest_path: Path,
    gate_config_path: Path,
    model_path: Path,
    model_manifest_path: Path,
    evaluation_path: Path,
    dimension: int = 16384,
    epochs: int = 16,
    learning_rate: float = 0.08,
    l2: float = 1e-6,
    seed: int = 17,
    fold_count: int = 5,
) -> SemanticRouterPromotionResult:
    train, train_content = _load_labels(train_labels_path)
    development, development_content = _load_labels(development_labels_path)
    if any(row.role != "training" for row in train):
        raise ValueError("semantic training labels contain a non-training role")
    if any(row.role != "development" for row in development):
        raise ValueError("semantic development labels contain a non-development role")
    train_ids = {row.query_id for row in train}
    development_ids = {row.query_id for row in development}
    if train_ids.intersection(development_ids):
        raise ValueError("semantic train and development queries overlap")

    development_manifest_content = development_manifest_path.read_bytes()
    development_manifest = json.loads(development_manifest_content)
    manifest_ids = {
        str(row["query_id"]) for row in development_manifest.get("sample", [])
    }
    if (
        development_manifest.get("role") != "development"
        or development_manifest.get("sample_query_count") != len(manifest_ids)
        or development_ids != manifest_ids
    ):
        raise ValueError("development labels do not match frozen manifest")

    gate_content = gate_config_path.read_bytes()
    gate_raw = json.loads(gate_content)
    if gate_raw.get("schema_version") != "method-router-enablement-gates-v1":
        raise ValueError("unsupported method router gate configuration")
    gate = MethodRouterGate.model_validate(gate_raw.get("semantic"))

    usable_train = [row for row in train if row.routing_label != "unavailable"]
    folds = _folds(usable_train, fold_count=fold_count, seed=seed)
    oof_rows: list[MethodRouteLabel] = []
    oof_probabilities: list[float] = []
    for held_out_index, held_out in enumerate(folds):
        training = [
            row
            for fold_index, fold in enumerate(folds)
            if fold_index != held_out_index
            for row in fold
        ]
        fold_router = CpuMethodRouter(
            method="semantic",
            dimension=dimension,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
        fold_router.fit(training)
        oof_rows.extend(held_out)
        oof_probabilities.extend(_probabilities(fold_router, held_out))
    oof_labels = [row.routing_label == "beneficial" for row in oof_rows]
    threshold = select_f1_threshold(oof_labels, oof_probabilities)
    oof_metrics = evaluate_probabilities(
        oof_labels, oof_probabilities, threshold=threshold
    )

    router = CpuMethodRouter(
        method="semantic",
        dimension=dimension,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
    router.fit(train)
    router.save(model_path)
    model_content = model_path.read_bytes()
    model_sha256 = _sha256(model_content)
    train_sha256 = _sha256(train_content)
    development_sha256 = _sha256(development_content)
    development_manifest_sha256 = _sha256(development_manifest_content)
    gate_sha256 = _sha256(gate_content)

    model_manifest = {
        "schema_version": "cpu-semantic-router-paired-model-v1",
        "model_id": CpuMethodRouter.model_id,
        "method": "semantic",
        "dimension": dimension,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "seed": seed,
        "fold_count": fold_count,
        "train_query_count": len(usable_train),
        "beneficial_train_query_count": sum(oof_labels),
        "unavailable_train_query_count": len(train) - len(usable_train),
        "threshold": threshold,
        "threshold_selected_on": f"training_stratified_{fold_count}_fold_oof",
        "oof_calibration": oof_metrics.model_dump(mode="json"),
        "model_sha256": model_sha256,
        "train_labels_sha256": train_sha256,
    }
    model_manifest_content = _canonical_bytes(model_manifest)
    write_frozen_bytes(model_manifest_path, model_manifest_content)
    model_manifest_sha256 = _sha256(model_manifest_content)

    development_probabilities = _probabilities(router, development)
    promotion = assess_method_router(
        development,
        development_probabilities,
        threshold=threshold,
        gate=gate,
    )
    result = SemanticRouterPromotionResult(
        dimension=dimension,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
        fold_count=fold_count,
        train_query_count=len(usable_train),
        beneficial_train_query_count=sum(oof_labels),
        unavailable_train_query_count=len(train) - len(usable_train),
        development_query_count=len(development),
        threshold=threshold,
        threshold_selected_on=f"training_stratified_{fold_count}_fold_oof",
        oof_calibration=oof_metrics,
        promotion=promotion,
        model_sha256=model_sha256,
        model_manifest_sha256=model_manifest_sha256,
        train_labels_sha256=train_sha256,
        development_labels_sha256=development_sha256,
        development_manifest_sha256=development_manifest_sha256,
        gate_config_sha256=gate_sha256,
    )
    write_frozen_bytes(
        evaluation_path,
        _canonical_bytes(result.model_dump(mode="json")),
    )
    return result


__all__ = [
    "SemanticRouterPromotionResult",
    "run_cpu_semantic_router_promotion",
]
