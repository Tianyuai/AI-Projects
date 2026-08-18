"""Train and evaluate independent CPU routers for semantic and graph methods."""

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
from paper_search.learning.graph_method_labels import GraphMethodLabel
from paper_search.learning.method_route_labels import (
    MethodRouteLabel,
    graph_method_route_labels,
    semantic_method_labels,
)
from paper_search.learning.provider_action_labels import ProviderActionLabel


class MethodHeadSummary(DomainModel):
    method: Literal["semantic", "graph"]
    train_count: int = Field(strict=True, gt=0)
    beneficial_train_count: int = Field(strict=True, gt=0)
    unavailable_train_count: int = Field(strict=True, ge=0)
    confidence_threshold: UnitFloat
    threshold_selected_on: str
    validation_scheme: str
    calibration: BinaryMetrics
    evaluation: BinaryMetrics | None = None
    deployable: bool
    model_sha256: Sha256
    label_sha256: list[Sha256]


class CpuMethodRouterExperimentManifest(DomainModel):
    schema_version: Literal["cpu-method-router-experiment-v1"] = (
        "cpu-method-router-experiment-v1"
    )
    model_id: str = CpuMethodRouter.model_id
    dimension: int = Field(strict=True, gt=0)
    epochs: int = Field(strict=True, gt=0)
    learning_rate: float = Field(gt=0)
    l2: float = Field(ge=0)
    seed: int
    semantic: MethodHeadSummary
    graph: MethodHeadSummary


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _read_jsonl(path: Path) -> tuple[list[dict[str, object]], bytes]:
    content = path.read_bytes()
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines()]
    if not rows:
        raise ValueError(f"label file is empty: {path}")
    return rows, content


def _semantic_labels(path: Path) -> tuple[list[MethodRouteLabel], bytes]:
    rows, content = _read_jsonl(path)
    if "method" in rows[0]:
        labels = [MethodRouteLabel.model_validate(row) for row in rows]
        if any(row.method != "semantic" for row in labels):
            raise ValueError("semantic label file contains another method")
        return labels, content
    provider_rows = [ProviderActionLabel.model_validate(row) for row in rows]
    return semantic_method_labels(provider_rows), content


def _graph_labels(path: Path) -> tuple[list[MethodRouteLabel], bytes]:
    rows, content = _read_jsonl(path)
    return graph_method_route_labels(
        [GraphMethodLabel.model_validate(row) for row in rows]
    ), content


def _usable(rows: list[MethodRouteLabel]) -> list[MethodRouteLabel]:
    return [row for row in rows if row.routing_label != "unavailable"]


def _labels(rows: list[MethodRouteLabel]) -> list[bool]:
    return [row.routing_label == "beneficial" for row in rows]


def _probabilities(
    router: CpuMethodRouter, rows: list[MethodRouteLabel]
) -> list[float]:
    return [
        router.predict_proba(row.query, seed_count=row.seed_count) for row in rows
    ]


def _stratified_folds(
    rows: list[MethodRouteLabel], *, fold_count: int, seed: int
) -> list[list[MethodRouteLabel]]:
    grouped: dict[bool, list[MethodRouteLabel]] = defaultdict(list)
    for row in rows:
        grouped[row.routing_label == "beneficial"].append(row)
    if fold_count < 2 or any(len(group) < fold_count for group in grouped.values()):
        raise ValueError("each graph class must contain at least one row per fold")
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


def _graph_oof_probabilities(
    rows: list[MethodRouteLabel],
    *,
    dimension: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
    fold_count: int,
) -> tuple[list[MethodRouteLabel], list[float]]:
    folds = _stratified_folds(rows, fold_count=fold_count, seed=seed)
    ordered_rows: list[MethodRouteLabel] = []
    probabilities: list[float] = []
    for held_out_index, held_out in enumerate(folds):
        training = [
            row
            for fold_index, fold in enumerate(folds)
            if fold_index != held_out_index
            for row in fold
        ]
        router = CpuMethodRouter(
            method="graph",
            dimension=dimension,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
        router.fit(training)
        ordered_rows.extend(held_out)
        probabilities.extend(_probabilities(router, held_out))
    return ordered_rows, probabilities


def run_cpu_method_router_experiment(
    *,
    semantic_train_path: Path,
    semantic_calibration_path: Path,
    semantic_evaluation_path: Path,
    graph_train_path: Path,
    semantic_model_path: Path,
    graph_model_path: Path,
    manifest_path: Path,
    dimension: int = 16384,
    epochs: int = 16,
    learning_rate: float = 0.08,
    l2: float = 1e-6,
    seed: int = 17,
    graph_folds: int = 5,
) -> CpuMethodRouterExperimentManifest:
    semantic_train, semantic_train_content = _semantic_labels(semantic_train_path)
    semantic_calibration, semantic_calibration_content = _semantic_labels(
        semantic_calibration_path
    )
    semantic_evaluation, semantic_evaluation_content = _semantic_labels(
        semantic_evaluation_path
    )
    graph_train, graph_train_content = _graph_labels(graph_train_path)
    if any(row.role != "training" for row in semantic_train + graph_train):
        raise ValueError("training router labels contain a non-training role")
    if any(
        row.role != "development"
        for row in semantic_calibration + semantic_evaluation
    ):
        raise ValueError("semantic development labels contain a non-development role")

    semantic_train_usable = _usable(semantic_train)
    semantic_calibration_usable = _usable(semantic_calibration)
    semantic_evaluation_usable = _usable(semantic_evaluation)
    graph_train_usable = _usable(graph_train)

    semantic_router = CpuMethodRouter(
        method="semantic",
        dimension=dimension,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
    semantic_router.fit(semantic_train)
    semantic_calibration_probabilities = _probabilities(
        semantic_router, semantic_calibration_usable
    )
    semantic_threshold = select_f1_threshold(
        _labels(semantic_calibration_usable), semantic_calibration_probabilities
    )
    semantic_evaluation_probabilities = _probabilities(
        semantic_router, semantic_evaluation_usable
    )

    graph_oof_rows, graph_oof_probabilities = _graph_oof_probabilities(
        graph_train_usable,
        dimension=dimension,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
        fold_count=graph_folds,
    )
    graph_threshold = select_f1_threshold(
        _labels(graph_oof_rows), graph_oof_probabilities
    )
    graph_router = CpuMethodRouter(
        method="graph",
        dimension=dimension,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
    graph_router.fit(graph_train)

    semantic_router.save(semantic_model_path)
    graph_router.save(graph_model_path)
    semantic_model_content = semantic_model_path.read_bytes()
    graph_model_content = graph_model_path.read_bytes()
    manifest = CpuMethodRouterExperimentManifest(
        dimension=dimension,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
        semantic=MethodHeadSummary(
            method="semantic",
            train_count=len(semantic_train_usable),
            beneficial_train_count=sum(_labels(semantic_train_usable)),
            unavailable_train_count=len(semantic_train) - len(semantic_train_usable),
            confidence_threshold=semantic_threshold,
            threshold_selected_on="isolated_pasa_auto_dev_calibration",
            validation_scheme="held_out_calibration_and_evaluation",
            calibration=evaluate_probabilities(
                _labels(semantic_calibration_usable),
                semantic_calibration_probabilities,
                threshold=semantic_threshold,
            ),
            evaluation=evaluate_probabilities(
                _labels(semantic_evaluation_usable),
                semantic_evaluation_probabilities,
                threshold=semantic_threshold,
            ),
            deployable=False,
            model_sha256=_sha256(semantic_model_content),
            label_sha256=[
                _sha256(semantic_train_content),
                _sha256(semantic_calibration_content),
                _sha256(semantic_evaluation_content),
            ],
        ),
        graph=MethodHeadSummary(
            method="graph",
            train_count=len(graph_train_usable),
            beneficial_train_count=sum(_labels(graph_train_usable)),
            unavailable_train_count=len(graph_train) - len(graph_train_usable),
            confidence_threshold=graph_threshold,
            threshold_selected_on="training_out_of_fold_predictions",
            validation_scheme="stratified_out_of_fold_calibration_only",
            calibration=evaluate_probabilities(
                _labels(graph_oof_rows),
                graph_oof_probabilities,
                threshold=graph_threshold,
            ),
            evaluation=None,
            deployable=False,
            model_sha256=_sha256(graph_model_content),
            label_sha256=[_sha256(graph_train_content)],
        ),
    )
    content = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    write_frozen_bytes(manifest_path, content)
    return manifest


__all__ = [
    "CpuMethodRouterExperimentManifest",
    "MethodHeadSummary",
    "run_cpu_method_router_experiment",
]
