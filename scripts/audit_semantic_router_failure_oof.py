from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.cpu_baseline import evaluate_probabilities, select_f1_threshold
from paper_search.learning.cpu_method_router import CpuMethodRouter
from paper_search.learning.cpu_semantic_router_promotion import _folds, _load_labels
from paper_search.learning.method_route_labels import MethodRouteLabel
from paper_search.learning.method_router_gate import MethodRouterGate


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


def _probabilities(
    router: CpuMethodRouter, rows: list[MethodRouteLabel]
) -> list[float]:
    return [router.predict_proba(row.query) for row in rows]


def _rank_metrics(labels: list[bool], scores: list[float]) -> dict[str, float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    ordered = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    auc = (
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
        if positives and negatives
        else 0.5
    )
    descending = sorted(
        zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True
    )
    seen_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(descending, start=1):
        if label:
            seen_positive += 1
            precision_sum += seen_positive / rank
    return {
        "roc_auc": auc,
        "average_precision": precision_sum / positives if positives else 0.0,
    }


def _threshold_row(
    rows: list[MethodRouteLabel], scores: list[float], threshold: float
) -> dict[str, float | bool]:
    labels = [row.routing_label == "beneficial" for row in rows]
    selected = [score >= threshold for score in scores]
    metrics = evaluate_probabilities(labels, scores, threshold=threshold)
    always = evaluate_probabilities(labels, [1.0] * len(labels), threshold=0.5)
    all_marginal = sum(row.marginal_gold_hit_count for row in rows)
    selected_marginal = sum(
        row.marginal_gold_hit_count
        for row, use in zip(rows, selected, strict=True)
        if use
    )
    marginal_capture = selected_marginal / all_marginal if all_marginal else 0.0
    call_reduction = 1.0 - sum(selected) / len(selected)
    f1_lift = float(metrics.f1) - float(always.f1)
    feasible = (
        float(metrics.recall) >= 0.90
        and call_reduction >= 0.25
        and f1_lift >= 0.05
        and marginal_capture >= 0.90
    )
    return {
        "threshold": threshold,
        "precision": float(metrics.precision),
        "beneficial_recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "f1_lift": f1_lift,
        "positive_prediction_rate": float(metrics.positive_prediction_rate),
        "call_reduction": call_reduction,
        "marginal_gold_capture": marginal_capture,
        "fixed_benefit_cost_feasible": feasible,
    }


def _stratum_metrics(
    rows: list[MethodRouteLabel],
    scores: list[float],
    metadata: dict[str, dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(metadata[row.query_id][field])].append(index)
    output: list[dict[str, Any]] = []
    for value, indexes in sorted(grouped.items()):
        group_labels = [rows[index].routing_label == "beneficial" for index in indexes]
        group_scores = [scores[index] for index in indexes]
        positive_scores = [
            score
            for score, label in zip(group_scores, group_labels, strict=True)
            if label
        ]
        negative_scores = [
            score
            for score, label in zip(group_scores, group_labels, strict=True)
            if not label
        ]
        output.append(
            {
                "value": value,
                "query_count": len(indexes),
                "beneficial_query_count": sum(group_labels),
                "beneficial_rate": sum(group_labels) / len(indexes),
                "mean_positive_score": (
                    sum(positive_scores) / len(positive_scores)
                    if positive_scores
                    else None
                ),
                "mean_negative_score": (
                    sum(negative_scores) / len(negative_scores)
                    if negative_scores
                    else None
                ),
                **_rank_metrics(group_labels, group_scores),
            }
        )
    return output


def _nested_training_sample(
    rows: list[MethodRouteLabel], *, target: int, seed: int
) -> list[MethodRouteLabel]:
    grouped: dict[bool, list[MethodRouteLabel]] = defaultdict(list)
    for row in rows:
        grouped[row.routing_label == "beneficial"].append(row)
    positive_target = max(1, round(target * len(grouped[True]) / len(rows)))
    positive_target = min(positive_target, len(grouped[True]))
    negative_target = min(target - positive_target, len(grouped[False]))
    if positive_target + negative_target < target:
        positive_target = min(target - negative_target, len(grouped[True]))
    selected: list[MethodRouteLabel] = []
    for label, count in ((True, positive_target), (False, negative_target)):
        ordered = sorted(
            grouped[label],
            key=lambda row: hashlib.sha256(
                f"learning-curve:{seed}:{label}:{row.query_id}".encode("utf-8")
            ).digest(),
        )
        selected.extend(ordered[:count])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows, labels_content = _load_labels(args.train_labels)
    if any(row.role != "training" for row in rows):
        raise ValueError("failure attribution accepts training labels only")
    manifest_content = args.train_manifest.read_bytes()
    manifest = json.loads(manifest_content)
    metadata = {str(row["query_id"]): row for row in manifest["sample"]}
    if set(metadata) != {row.query_id for row in rows}:
        raise ValueError("training labels do not match frozen manifest")
    model_manifest_content = args.model_manifest.read_bytes()
    model_manifest = json.loads(model_manifest_content)
    gate_content = args.gate_config.read_bytes()
    gate_raw = json.loads(gate_content)
    gate = MethodRouterGate.model_validate(gate_raw["semantic"])

    folds = _folds(
        rows,
        fold_count=int(model_manifest["fold_count"]),
        seed=int(model_manifest["seed"]),
    )
    oof_rows: list[MethodRouteLabel] = []
    oof_scores: list[float] = []
    for held_out_index, held_out in enumerate(folds):
        training = [
            row
            for fold_index, fold in enumerate(folds)
            if fold_index != held_out_index
            for row in fold
        ]
        router = CpuMethodRouter(
            method="semantic",
            dimension=int(model_manifest["dimension"]),
            epochs=int(model_manifest["epochs"]),
            learning_rate=float(model_manifest["learning_rate"]),
            l2=float(model_manifest["l2"]),
            seed=int(model_manifest["seed"]),
        )
        router.fit(training)
        oof_rows.extend(held_out)
        oof_scores.extend(_probabilities(router, held_out))
    oof_labels = [row.routing_label == "beneficial" for row in oof_rows]
    reproduced_threshold = select_f1_threshold(oof_labels, oof_scores)
    if abs(reproduced_threshold - float(model_manifest["threshold"])) > 1e-15:
        raise ValueError("OOF threshold does not reproduce frozen model manifest")

    threshold_rows = [
        _threshold_row(oof_rows, oof_scores, threshold)
        for threshold in sorted(set(oof_scores), reverse=True)
    ]
    feasible = [row for row in threshold_rows if row["fixed_benefit_cost_feasible"]]
    high_capture = [
        row
        for row in threshold_rows
        if row["beneficial_recall"] >= float(gate.minimum_beneficial_recall)
        and row["marginal_gold_capture"]
        >= float(gate.minimum_marginal_gold_capture)
    ]

    final_router = CpuMethodRouter.load(
        args.model,
        method="semantic",
        dimension=int(model_manifest["dimension"]),
        epochs=int(model_manifest["epochs"]),
        learning_rate=float(model_manifest["learning_rate"]),
        l2=float(model_manifest["l2"]),
        seed=int(model_manifest["seed"]),
    )
    in_sample_scores = _probabilities(final_router, rows)

    learning_curve: list[dict[str, Any]] = []
    for target in (200, 500, 1000, 1600):
        curve_labels: list[bool] = []
        curve_scores: list[float] = []
        for held_out_index, held_out in enumerate(folds):
            pool = [
                row
                for fold_index, fold in enumerate(folds)
                if fold_index != held_out_index
                for row in fold
            ]
            training = _nested_training_sample(
                pool,
                target=min(target, len(pool)),
                seed=int(model_manifest["seed"]),
            )
            router = CpuMethodRouter(
                method="semantic",
                dimension=int(model_manifest["dimension"]),
                epochs=int(model_manifest["epochs"]),
                learning_rate=float(model_manifest["learning_rate"]),
                l2=float(model_manifest["l2"]),
                seed=int(model_manifest["seed"]),
            )
            router.fit(training)
            curve_labels.extend(
                row.routing_label == "beneficial" for row in held_out
            )
            curve_scores.extend(_probabilities(router, held_out))
        learning_curve.append(
            {
                "training_queries_per_fold": target,
                "held_out_prediction_count": len(curve_labels),
                **_rank_metrics(curve_labels, curve_scores),
            }
        )

    current_threshold = _threshold_row(
        oof_rows, oof_scores, float(model_manifest["threshold"])
    )
    best_f1 = max(threshold_rows, key=lambda row: (row["f1"], row["threshold"]))
    best_high_capture = (
        max(high_capture, key=lambda row: row["call_reduction"])
        if high_capture
        else None
    )
    report = {
        "schema_version": "semantic-router-oof-failure-attribution-v1",
        "scope": "training_only_stratified_oof_no_development_or_test_labels",
        "query_count": len(rows),
        "beneficial_query_count": sum(oof_labels),
        "beneficial_rate": sum(oof_labels) / len(oof_labels),
        "oof_ranking": _rank_metrics(oof_labels, oof_scores),
        "in_sample_ranking": _rank_metrics(
            [row.routing_label == "beneficial" for row in rows],
            in_sample_scores,
        ),
        "generalization_gap": {
            "roc_auc": _rank_metrics(
                [row.routing_label == "beneficial" for row in rows],
                in_sample_scores,
            )["roc_auc"]
            - _rank_metrics(oof_labels, oof_scores)["roc_auc"],
            "average_precision": _rank_metrics(
                [row.routing_label == "beneficial" for row in rows],
                in_sample_scores,
            )["average_precision"]
            - _rank_metrics(oof_labels, oof_scores)["average_precision"],
        },
        "threshold_audit": {
            "current_f1_threshold": current_threshold,
            "best_f1_threshold": best_f1,
            "fixed_benefit_cost_feasible_threshold_count": len(feasible),
            "best_call_reduction_with_required_recall_and_marginal_capture": best_high_capture,
        },
        "strata": {
            field: _stratum_metrics(oof_rows, oof_scores, metadata, field)
            for field in (
                "intent_family",
                "length_bucket",
                "gold_count_bucket",
                "fold",
            )
        },
        "learning_curve": learning_curve,
        "inputs": {
            "train_labels_sha256": _sha256(labels_content),
            "train_manifest_sha256": _sha256(manifest_content),
            "model_sha256": _sha256(args.model.read_bytes()),
            "model_manifest_sha256": _sha256(model_manifest_content),
            "gate_config_sha256": _sha256(gate_content),
        },
        "development_labels_read": False,
        "final_test_consumed": False,
    }
    content = _canonical_bytes(report)
    write_frozen_bytes(args.output, content)
    print(json.dumps({**report, "report_sha256": _sha256(content)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
