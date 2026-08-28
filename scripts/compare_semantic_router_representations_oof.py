from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from scipy.sparse import csr_matrix, hstack  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]
from sklearn.pipeline import FeatureUnion  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.cpu_baseline import select_f1_threshold
from paper_search.learning.cpu_method_router import CpuMethodRouter
from paper_search.learning.cpu_semantic_router_promotion import _folds, _load_labels
from paper_search.learning.method_route_labels import MethodRouteLabel
from paper_search.learning.method_router_gate import (
    MethodRouterGate,
    assess_method_router,
)
from paper_search.learning.provider_action_dataset import (
    _restore_redacted_usage,
    _successful_or_failed_receipt,
)
from paper_search.learning.semantic_router_features import (
    LEXICAL_ROUTE_FEATURE_NAMES,
    extract_lexical_route_features,
)
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    RetrievalActionResult,
)


Variant = Literal["current_hashed", "tfidf_query", "retrieval_state", "combined"]


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _load_receipt_features(
    *,
    query_by_id: dict[str, str],
    run_roots: list[Path],
    repair_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    action_hits_by_query, source_by_query = _load_receipt_action_hits(
        query_by_id=query_by_id,
        run_roots=run_roots,
        repair_root=repair_root,
    )
    features = {
        query_id: extract_lexical_route_features(
            query_by_id[query_id], action_hits_by_query[query_id]
        )
        for query_id in sorted(query_by_id)
    }
    counts = {
        "query_count": len(features),
        "repair_override_query_count": sum(
            source == str(repair_root) for source in source_by_query.values()
        ),
        "feature_count": len(LEXICAL_ROUTE_FEATURE_NAMES),
    }
    return features, counts


def _load_receipt_action_hits(
    *,
    query_by_id: dict[str, str],
    run_roots: list[Path],
    repair_root: Path,
) -> tuple[dict[str, list[list[Paper]]], dict[str, str]]:
    action_hits_by_query: dict[str, list[list[Paper]]] = {}
    source_by_query: dict[str, str] = {}

    def consume(root: Path, *, override: bool) -> None:
        for report_path in sorted((root / "openalex").glob("*/canary-report.json")):
            batch = report_path.parent
            report = json.loads(report_path.read_text(encoding="utf-8"))
            raw_map = report.get("actions_by_query")
            if not isinstance(raw_map, dict):
                raise ValueError(f"invalid action map: {report_path}")
            for query_id, raw_actions in raw_map.items():
                if query_id not in query_by_id:
                    raise ValueError(f"receipt query is absent from labels: {query_id}")
                if query_id in action_hits_by_query and not override:
                    raise ValueError(f"duplicate lexical receipt: {query_id}")
                actions = RecallActionBatch.model_validate(
                    {"actions": raw_actions}
                ).actions
                if any(
                    action.action_type != "text_search"
                    or action.payload.search_mode != "lexical"
                    for action in actions
                ):
                    if override:
                        raise ValueError(f"repair receipt is not lexical: {query_id}")
                    continue
                receipt = _successful_or_failed_receipt(batch, query_id)
                results = [
                    RetrievalActionResult.model_validate(
                        _restore_redacted_usage(item)
                    )
                    for item in receipt.get("results", [])
                ]
                results_by_id = {result.action_id: result for result in results}
                if set(results_by_id) != {action.action_id for action in actions}:
                    raise ValueError(f"action/result mismatch: {query_id}")
                action_hits_by_query[query_id] = [
                    results_by_id[action.action_id].hits for action in actions
                ]
                source_by_query[query_id] = str(root)

    for root in run_roots:
        consume(root, override=False)
    consume(repair_root, override=True)
    if set(action_hits_by_query) != set(query_by_id):
        missing = set(query_by_id).difference(action_hits_by_query)
        raise ValueError(f"lexical receipt coverage mismatch: missing={len(missing)}")
    return action_hits_by_query, source_by_query


def _tfidf() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def _logistic() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=17,
        solver="liblinear",
    )


def _sklearn_fold_probabilities(
    *,
    variant: Variant,
    training: list[MethodRouteLabel],
    held_out: list[MethodRouteLabel],
    state_by_id: dict[str, np.ndarray],
) -> list[float]:
    train_queries = [row.query for row in training]
    held_queries = [row.query for row in held_out]
    train_state = np.vstack([state_by_id[row.query_id] for row in training])
    held_state = np.vstack([state_by_id[row.query_id] for row in held_out])
    train_labels = np.asarray(
        [row.routing_label == "beneficial" for row in training], dtype=np.int64
    )
    query_train = query_held = None
    if variant in {"tfidf_query", "combined"}:
        vectorizer = _tfidf()
        query_train = vectorizer.fit_transform(train_queries)
        query_held = vectorizer.transform(held_queries)
    state_train = state_held = None
    if variant in {"retrieval_state", "combined"}:
        scaler = StandardScaler()
        state_train = csr_matrix(scaler.fit_transform(train_state))
        state_held = csr_matrix(scaler.transform(held_state))
    if variant == "tfidf_query":
        train_matrix, held_matrix = query_train, query_held
    elif variant == "retrieval_state":
        train_matrix, held_matrix = state_train, state_held
    else:
        if query_train is None or query_held is None:
            raise AssertionError("combined representation requires query features")
        if state_train is None or state_held is None:
            raise AssertionError("combined representation requires state features")
        train_matrix = hstack([query_train, state_train], format="csr")
        held_matrix = hstack([query_held, state_held], format="csr")
    classifier = _logistic()
    classifier.fit(train_matrix, train_labels)
    return cast(list[float], classifier.predict_proba(held_matrix)[:, 1].tolist())


def _current_fold_probabilities(
    *,
    training: list[MethodRouteLabel],
    held_out: list[MethodRouteLabel],
    model_manifest: dict[str, Any],
) -> list[float]:
    router = CpuMethodRouter(
        method="semantic",
        dimension=int(model_manifest["dimension"]),
        epochs=int(model_manifest["epochs"]),
        learning_rate=float(model_manifest["learning_rate"]),
        l2=float(model_manifest["l2"]),
        seed=int(model_manifest["seed"]),
    )
    router.fit(training)
    return [router.predict_proba(row.query) for row in held_out]


def _variant_summary(
    *,
    rows: list[MethodRouteLabel],
    scores: list[float],
    fold_slices: list[tuple[int, int]],
    gate: MethodRouterGate,
) -> dict[str, Any]:
    labels = [row.routing_label == "beneficial" for row in rows]
    threshold = select_f1_threshold(labels, scores)
    decision = assess_method_router(rows, scores, threshold=threshold, gate=gate)
    feasible: list[tuple[float, Any]] = []
    for candidate in sorted(set(scores), reverse=True):
        candidate_decision = assess_method_router(
            rows, scores, threshold=candidate, gate=gate
        )
        if candidate_decision.enable:
            feasible.append((candidate, candidate_decision))
    best_feasible = (
        max(
            feasible,
            key=lambda item: (
                item[1].call_reduction,
                item[1].routed.f1,
                item[0],
            ),
        )
        if feasible
        else None
    )
    fold_metrics = []
    for start, end in fold_slices:
        fold_labels = labels[start:end]
        fold_scores = scores[start:end]
        fold_metrics.append(
            {
                "query_count": end - start,
                "roc_auc": roc_auc_score(fold_labels, fold_scores),
                "average_precision": average_precision_score(
                    fold_labels, fold_scores
                ),
            }
        )
    return {
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
        "f1_threshold": threshold,
        "f1_threshold_decision": decision.model_dump(mode="json"),
        "feasible_threshold_count": len(feasible),
        "best_feasible_decision": (
            {
                "threshold": best_feasible[0],
                "decision": best_feasible[1].model_dump(mode="json"),
            }
            if best_feasible is not None
            else None
        ),
        "fold_metrics": fold_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows, label_content = _load_labels(args.labels)
    if any(row.role != "training" for row in rows):
        raise ValueError("representation comparison accepts training labels only")
    model_manifest_content = args.model_manifest.read_bytes()
    model_manifest = json.loads(model_manifest_content)
    gate_content = args.gate_config.read_bytes()
    gate = MethodRouterGate.model_validate(json.loads(gate_content)["semantic"])
    query_by_id = {row.query_id: row.query for row in rows}
    state_by_id, receipt_counts = _load_receipt_features(
        query_by_id=query_by_id,
        run_roots=args.run_root,
        repair_root=args.repair_root,
    )
    folds = _folds(
        rows,
        fold_count=int(model_manifest["fold_count"]),
        seed=int(model_manifest["seed"]),
    )
    variants: dict[Variant, tuple[list[MethodRouteLabel], list[float]]] = {}
    fold_slices: list[tuple[int, int]] = []
    variants_to_run: tuple[Variant, ...] = (
        "current_hashed",
        "tfidf_query",
        "retrieval_state",
        "combined",
    )
    for variant in variants_to_run:
        ordered_rows: list[MethodRouteLabel] = []
        probabilities: list[float] = []
        fold_slices = []
        for held_out_index, held_out in enumerate(folds):
            training = [
                row
                for fold_index, fold in enumerate(folds)
                if fold_index != held_out_index
                for row in fold
            ]
            start = len(ordered_rows)
            ordered_rows.extend(held_out)
            fold_slices.append((start, len(ordered_rows)))
            if variant == "current_hashed":
                probabilities.extend(
                    _current_fold_probabilities(
                        training=training,
                        held_out=held_out,
                        model_manifest=model_manifest,
                    )
                )
            else:
                probabilities.extend(
                    _sklearn_fold_probabilities(
                        variant=variant,
                        training=training,
                        held_out=held_out,
                        state_by_id=state_by_id,
                    )
                )
        variants[variant] = (ordered_rows, probabilities)

    summaries = {
        variant: _variant_summary(
            rows=variant_rows,
            scores=scores,
            fold_slices=fold_slices,
            gate=gate,
        )
        for variant, (variant_rows, scores) in variants.items()
    }
    baseline = summaries["current_hashed"]
    criteria = {
        "minimum_roc_auc_delta": 0.05,
        "minimum_average_precision_delta": 0.05,
        "minimum_improved_fold_count": 4,
        "require_fixed_gate_feasible_threshold": True,
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for variant in ("tfidf_query", "retrieval_state", "combined"):
        summary = summaries[variant]
        improved_folds = sum(
            candidate["roc_auc"] > control["roc_auc"]
            for candidate, control in zip(
                summary["fold_metrics"], baseline["fold_metrics"], strict=True
            )
        )
        auc_delta = summary["roc_auc"] - baseline["roc_auc"]
        ap_delta = summary["average_precision"] - baseline["average_precision"]
        pass_predeployment = (
            auc_delta >= criteria["minimum_roc_auc_delta"]
            and ap_delta >= criteria["minimum_average_precision_delta"]
            and improved_folds >= criteria["minimum_improved_fold_count"]
            and summary["feasible_threshold_count"] > 0
        )
        comparisons[variant] = {
            "roc_auc_delta": auc_delta,
            "average_precision_delta": ap_delta,
            "improved_fold_count": improved_folds,
            "pass_predeployment_gate": pass_predeployment,
        }
    report = {
        "schema_version": "semantic-router-representation-ablation-oof-v1",
        "scope": "training_only_fixed_5_fold_oof_no_development_or_test",
        "query_count": len(rows),
        "receipt_counts": receipt_counts,
        "route_feature_names": list(LEXICAL_ROUTE_FEATURE_NAMES),
        "fixed_predeployment_criteria": criteria,
        "variants": summaries,
        "comparisons": comparisons,
        "inputs": {
            "labels_sha256": _sha256(label_content),
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
