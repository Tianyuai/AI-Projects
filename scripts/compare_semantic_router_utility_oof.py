from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from scipy.sparse import csr_matrix, hstack  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from compare_semantic_router_frozen_encoder_oof import (  # type: ignore[import-not-found]
    _fold_probabilities,
)
from compare_semantic_router_representations_oof import (  # type: ignore[import-not-found]
    _canonical_bytes,
    _sha256,
    _tfidf,
    _variant_summary,
)
from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.cpu_semantic_router_promotion import _folds, _load_labels
from paper_search.learning.method_route_labels import MethodRouteLabel
from paper_search.learning.method_router_gate import MethodRouterGate
from paper_search.learning.semantic_router_utility import (
    marginal_capture_at_call_reduction,
    semantic_utility_labels,
    utility_rank_scores,
)


FIXED_CALL_REDUCTIONS = (0.10, 0.20, 0.25, 0.30)
MINIMUM_POSITIVE_OVERALL_BUDGET_COUNT = 3
MINIMUM_IMPROVED_FOLD_COUNT = 4


class BudgetComparison(TypedDict):
    call_reduction: float
    marginal_capture_delta: float
    improved_fold_count: int
    non_degraded_fold_count: int


def _utility_fold_scores(
    *,
    training: list[MethodRouteLabel],
    held_out: list[MethodRouteLabel],
    index_by_id: dict[str, int],
    target_by_id: dict[str, float],
    query_embeddings: np.ndarray,
    lexical_features: np.ndarray,
    match_features: np.ndarray,
) -> list[float]:
    train_indices = [index_by_id[row.query_id] for row in training]
    held_indices = [index_by_id[row.query_id] for row in held_out]
    dense = np.hstack([query_embeddings, lexical_features, match_features])
    scaler = StandardScaler()
    train_dense = csr_matrix(scaler.fit_transform(dense[train_indices]))
    held_dense = csr_matrix(scaler.transform(dense[held_indices]))
    vectorizer = _tfidf()
    train_text = vectorizer.fit_transform([row.query for row in training])
    held_text = vectorizer.transform([row.query for row in held_out])
    train_matrix = hstack([train_text, train_dense], format="csr")
    held_matrix = hstack([held_text, held_dense], format="csr")
    targets = np.asarray(
        [target_by_id[row.query_id] for row in training], dtype=np.float64
    )
    regressor = Ridge(alpha=1.0, solver="lsqr")
    regressor.fit(train_matrix, targets)
    predictions = cast(list[float], regressor.predict(held_matrix).tolist())
    return utility_rank_scores(predictions)


def _capture_grid(
    rows: list[MethodRouteLabel],
    scores: list[float],
    fold_slices: list[tuple[int, int]],
) -> list[dict[str, object]]:
    output = []
    for reduction in FIXED_CALL_REDUCTIONS:
        overall = marginal_capture_at_call_reduction(
            rows,
            scores,
            target_call_reduction=reduction,
        )
        folds = [
            marginal_capture_at_call_reduction(
                rows[start:end],
                scores[start:end],
                target_call_reduction=reduction,
            )
            for start, end in fold_slices
        ]
        output.append({"call_reduction": reduction, "overall": overall, "folds": folds})
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-labels", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--utility-label-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows, label_content = _load_labels(args.method_labels)
    if any(row.role != "training" for row in rows):
        raise ValueError("utility comparison accepts training labels only")
    utility_labels = semantic_utility_labels(rows)
    utility_content = b"".join(
        (
            json.dumps(
                label.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for label in utility_labels
    )
    write_frozen_bytes(args.utility_label_output, utility_content)
    target_by_id = {
        label.query_id: label.marginal_hits_per_call for label in utility_labels
    }
    if set(target_by_id) != {row.query_id for row in rows}:
        raise ValueError("utility target coverage must match route labels")

    cache_content = args.embedding_cache.read_bytes()
    with np.load(io.BytesIO(cache_content), allow_pickle=False) as cache:
        query_ids = cache["query_ids"]
        query_embeddings = cache["query_embeddings"].astype(np.float64)
        lexical_features = cache["lexical_features"].astype(np.float64)
        match_features = cache["semantic_match_features"].astype(np.float64)
    if not np.array_equal(query_ids, np.asarray([row.query_id for row in rows])):
        raise ValueError("embedding cache order does not match labels")
    index_by_id = {row.query_id: index for index, row in enumerate(rows)}

    model_manifest_content = args.model_manifest.read_bytes()
    model_manifest = json.loads(model_manifest_content)
    gate_content = args.gate_config.read_bytes()
    gate = MethodRouterGate.model_validate(json.loads(gate_content)["semantic"])
    folds = _folds(
        rows,
        fold_count=int(model_manifest["fold_count"]),
        seed=int(model_manifest["seed"]),
    )
    ordered_rows: list[MethodRouteLabel] = []
    binary_scores: list[float] = []
    utility_scores: list[float] = []
    fold_slices: list[tuple[int, int]] = []
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
        binary_scores.extend(
            _fold_probabilities(
                variant="hybrid",
                training=training,
                held_out=held_out,
                index_by_id=index_by_id,
                query_embeddings=query_embeddings,
                lexical_features=lexical_features,
                match_features=match_features,
            )
        )
        utility_scores.extend(
            _utility_fold_scores(
                training=training,
                held_out=held_out,
                index_by_id=index_by_id,
                target_by_id=target_by_id,
                query_embeddings=query_embeddings,
                lexical_features=lexical_features,
                match_features=match_features,
            )
        )

    variant_scores = {"binary_hybrid": binary_scores, "expected_marginal_hits": utility_scores}
    summaries = {
        name: _variant_summary(
            rows=ordered_rows,
            scores=scores,
            fold_slices=fold_slices,
            gate=gate,
        )
        for name, scores in variant_scores.items()
    }
    capture_grids = {
        name: _capture_grid(ordered_rows, scores, fold_slices)
        for name, scores in variant_scores.items()
    }
    budget_comparisons: list[BudgetComparison] = []
    for binary, utility in zip(
        capture_grids["binary_hybrid"],
        capture_grids["expected_marginal_hits"],
        strict=True,
    ):
        binary_overall = cast(dict[str, float], binary["overall"])
        utility_overall = cast(dict[str, float], utility["overall"])
        binary_folds = cast(list[dict[str, float]], binary["folds"])
        utility_folds = cast(list[dict[str, float]], utility["folds"])
        budget_comparisons.append(
            {
                "call_reduction": cast(float, binary["call_reduction"]),
                "marginal_capture_delta": (
                    utility_overall["marginal_gold_capture"]
                    - binary_overall["marginal_gold_capture"]
                ),
                "improved_fold_count": sum(
                    candidate["marginal_gold_capture"]
                    > control["marginal_gold_capture"]
                    for candidate, control in zip(
                        utility_folds, binary_folds, strict=True
                    )
                ),
                "non_degraded_fold_count": sum(
                    candidate["marginal_gold_capture"]
                    >= control["marginal_gold_capture"]
                    for candidate, control in zip(
                        utility_folds, binary_folds, strict=True
                    )
                ),
            }
        )
    criteria = {
        "minimum_positive_overall_budget_count": MINIMUM_POSITIVE_OVERALL_BUDGET_COUNT,
        "minimum_improved_fold_count_per_qualifying_budget": MINIMUM_IMPROVED_FOLD_COUNT,
        "require_no_overall_budget_regression": True,
    }
    qualifying_budgets = sum(
        row["marginal_capture_delta"] > 0
        and row["improved_fold_count"]
        >= MINIMUM_IMPROVED_FOLD_COUNT
        for row in budget_comparisons
    )
    stable_improvement = (
        qualifying_budgets >= MINIMUM_POSITIVE_OVERALL_BUDGET_COUNT
        and all(row["marginal_capture_delta"] >= 0 for row in budget_comparisons)
    )
    report = {
        "schema_version": "semantic-router-continuous-utility-oof-v1",
        "scope": "training_only_fixed_5_fold_oof_no_development_or_test",
        "query_count": len(rows),
        "target": "observed_marginal_gold_hits_per_semantic_api_call",
        "cost_observation": "all examples have one semantic API call; cost remains a fixed deployment gate rather than a learned varying label",
        "model": {
            "type": "ridge_regression",
            "alpha": 1.0,
            "solver": "lsqr",
            "features": "tfidf_plus_frozen_bge_query_plus_lexical_state_plus_query_title_match",
            "device": "cpu",
        },
        "fixed_stability_criteria": criteria,
        "variants": summaries,
        "capture_at_fixed_call_reduction": capture_grids,
        "budget_comparisons": budget_comparisons,
        "stable_improvement": stable_improvement,
        "inputs": {
            "method_labels_sha256": _sha256(label_content),
            "utility_labels_sha256": _sha256(utility_content),
            "embedding_cache_sha256": _sha256(cache_content),
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
