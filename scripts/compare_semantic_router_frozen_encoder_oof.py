from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from scipy.sparse import csr_matrix, hstack  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from compare_semantic_router_representations_oof import (  # type: ignore[import-not-found]
    _canonical_bytes,
    _current_fold_probabilities,
    _load_receipt_action_hits,
    _sha256,
    _tfidf,
    _variant_summary,
)
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.cpu_semantic_router_promotion import _folds, _load_labels
from paper_search.learning.method_route_labels import MethodRouteLabel
from paper_search.learning.method_router_gate import MethodRouterGate
from paper_search.learning.semantic_router_features import (
    LEXICAL_ROUTE_FEATURE_NAMES,
    SEMANTIC_MATCH_FEATURE_NAMES,
    extract_lexical_route_features,
    extract_semantic_match_features,
)


Variant = Literal["current_hashed", "prior_combined", "frozen_encoder", "hybrid"]
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
TITLE_LIMIT_PER_ACTION = 10


def _classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=17,
        solver="liblinear",
    )


def _encode_features(
    *,
    rows: list[MethodRouteLabel],
    action_hits_by_query: dict[str, list[list[Any]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_files_only=True,
        device="cpu",
    )
    queries = [row.query for row in rows]
    query_embeddings = model.encode(
        queries,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float64)
    titles: list[str] = []
    slices_by_query: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        action_slices = []
        for hits in action_hits_by_query[row.query_id]:
            start = len(titles)
            titles.extend(paper.title for paper in hits[:TITLE_LIMIT_PER_ACTION])
            action_slices.append((start, len(titles)))
        slices_by_query[row.query_id] = action_slices
    title_embeddings = model.encode(
        titles,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float64)

    lexical_rows = []
    match_rows = []
    for index, row in enumerate(rows):
        action_hits = action_hits_by_query[row.query_id]
        lexical_rows.append(extract_lexical_route_features(row.query, action_hits))
        action_embeddings = [
            title_embeddings[start:end]
            for start, end in slices_by_query[row.query_id]
        ]
        match_rows.append(
            extract_semantic_match_features(query_embeddings[index], action_embeddings)
        )
    return query_embeddings, np.vstack(lexical_rows), np.vstack(match_rows)


def _load_or_create_cache(
    *,
    cache_path: Path,
    rows: list[MethodRouteLabel],
    action_hits_by_query: dict[str, list[list[Any]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes]:
    expected_ids = np.asarray([row.query_id for row in rows])
    if cache_path.exists():
        content = cache_path.read_bytes()
        with np.load(io.BytesIO(content), allow_pickle=False) as cache:
            query_ids = cache["query_ids"]
            if not np.array_equal(query_ids, expected_ids):
                raise ValueError("embedding cache query order does not match labels")
            return (
                cache["query_embeddings"].astype(np.float64),
                cache["lexical_features"].astype(np.float64),
                cache["semantic_match_features"].astype(np.float64),
                content,
            )
    query_embeddings, lexical_features, match_features = _encode_features(
        rows=rows,
        action_hits_by_query=action_hits_by_query,
    )
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        query_ids=expected_ids,
        query_embeddings=query_embeddings.astype(np.float32),
        lexical_features=lexical_features.astype(np.float32),
        semantic_match_features=match_features.astype(np.float32),
    )
    content = buffer.getvalue()
    write_frozen_bytes(cache_path, content)
    with np.load(io.BytesIO(content), allow_pickle=False) as cache:
        return (
            cache["query_embeddings"].astype(np.float64),
            cache["lexical_features"].astype(np.float64),
            cache["semantic_match_features"].astype(np.float64),
            content,
        )


def _fold_probabilities(
    *,
    variant: Variant,
    training: list[MethodRouteLabel],
    held_out: list[MethodRouteLabel],
    index_by_id: dict[str, int],
    query_embeddings: np.ndarray,
    lexical_features: np.ndarray,
    match_features: np.ndarray,
) -> list[float]:
    train_indices = [index_by_id[row.query_id] for row in training]
    held_indices = [index_by_id[row.query_id] for row in held_out]
    train_labels = np.asarray(
        [row.routing_label == "beneficial" for row in training], dtype=np.int64
    )
    numeric = np.hstack([lexical_features, match_features])
    if variant == "prior_combined":
        dense = lexical_features
    else:
        dense = np.hstack([query_embeddings, numeric])
    scaler = StandardScaler()
    train_dense = csr_matrix(scaler.fit_transform(dense[train_indices]))
    held_dense = csr_matrix(scaler.transform(dense[held_indices]))

    if variant in {"prior_combined", "hybrid"}:
        vectorizer = _tfidf()
        train_text = vectorizer.fit_transform([row.query for row in training])
        held_text = vectorizer.transform([row.query for row in held_out])
        train_matrix = hstack([train_text, train_dense], format="csr")
        held_matrix = hstack([held_text, held_dense], format="csr")
    else:
        train_matrix, held_matrix = train_dense, held_dense
    classifier = _classifier()
    classifier.fit(train_matrix, train_labels)
    return cast(list[float], classifier.predict_proba(held_matrix)[:, 1].tolist())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows, label_content = _load_labels(args.labels)
    if any(row.role != "training" for row in rows):
        raise ValueError("frozen encoder comparison accepts training labels only")
    model_manifest_content = args.model_manifest.read_bytes()
    model_manifest = json.loads(model_manifest_content)
    gate_content = args.gate_config.read_bytes()
    gate = MethodRouterGate.model_validate(json.loads(gate_content)["semantic"])
    query_by_id = {row.query_id: row.query for row in rows}
    action_hits_by_query, source_by_query = _load_receipt_action_hits(
        query_by_id=query_by_id,
        run_roots=args.run_root,
        repair_root=args.repair_root,
    )
    query_embeddings, lexical_features, match_features, cache_content = (
        _load_or_create_cache(
            cache_path=args.embedding_cache,
            rows=rows,
            action_hits_by_query=action_hits_by_query,
        )
    )
    index_by_id = {row.query_id: index for index, row in enumerate(rows)}
    folds = _folds(
        rows,
        fold_count=int(model_manifest["fold_count"]),
        seed=int(model_manifest["seed"]),
    )
    variants_to_run: tuple[Variant, ...] = (
        "current_hashed",
        "prior_combined",
        "frozen_encoder",
        "hybrid",
    )
    variants: dict[Variant, tuple[list[MethodRouteLabel], list[float]]] = {}
    fold_slices: list[tuple[int, int]] = []
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
                    _fold_probabilities(
                        variant=variant,
                        training=training,
                        held_out=held_out,
                        index_by_id=index_by_id,
                        query_embeddings=query_embeddings,
                        lexical_features=lexical_features,
                        match_features=match_features,
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
    criteria = {
        "minimum_roc_auc_delta": 0.05,
        "minimum_average_precision_delta": 0.05,
        "minimum_improved_fold_count": 4,
        "require_fixed_gate_feasible_threshold": True,
    }
    baseline = summaries["current_hashed"]
    comparisons: dict[str, dict[str, Any]] = {}
    for variant in ("prior_combined", "frozen_encoder", "hybrid"):
        summary = summaries[variant]
        improved_folds = sum(
            candidate["roc_auc"] > control["roc_auc"]
            for candidate, control in zip(
                summary["fold_metrics"], baseline["fold_metrics"], strict=True
            )
        )
        auc_delta = summary["roc_auc"] - baseline["roc_auc"]
        ap_delta = summary["average_precision"] - baseline["average_precision"]
        comparisons[variant] = {
            "roc_auc_delta": auc_delta,
            "average_precision_delta": ap_delta,
            "improved_fold_count": improved_folds,
            "pass_predeployment_gate": (
                auc_delta >= criteria["minimum_roc_auc_delta"]
                and ap_delta >= criteria["minimum_average_precision_delta"]
                and improved_folds >= criteria["minimum_improved_fold_count"]
                and summary["feasible_threshold_count"] > 0
            ),
        }
    report = {
        "schema_version": "semantic-router-frozen-encoder-ablation-oof-v1",
        "scope": "training_only_fixed_5_fold_oof_no_development_or_test",
        "query_count": len(rows),
        "encoder": {
            "model": MODEL_NAME,
            "revision": MODEL_REVISION,
            "frozen": True,
            "device": "cpu",
            "title_limit_per_action": TITLE_LIMIT_PER_ACTION,
            "embedding_dimension": int(query_embeddings.shape[1]),
        },
        "repair_override_query_count": sum(
            source == str(args.repair_root) for source in source_by_query.values()
        ),
        "feature_names": {
            "lexical": list(LEXICAL_ROUTE_FEATURE_NAMES),
            "semantic_match": list(SEMANTIC_MATCH_FEATURE_NAMES),
        },
        "fixed_predeployment_criteria": criteria,
        "variants": summaries,
        "comparisons": comparisons,
        "inputs": {
            "labels_sha256": _sha256(label_content),
            "model_manifest_sha256": _sha256(model_manifest_content),
            "gate_config_sha256": _sha256(gate_content),
            "embedding_cache_sha256": _sha256(cache_content),
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
