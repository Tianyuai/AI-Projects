"""Replay production-anchored F5 deltas on queries unseen by production."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.evaluation.predictions import paper_evaluation_id  # noqa: E402
from paper_search.learning.anchored_fusion import (  # noqa: E402
    PRIMARY_METRICS,
    blend_anchored_family_weights,
    new_only_batch_indexes,
    new_only_query_ids,
    scale_anchored_family_weights,
    select_conservative_alpha,
    select_conservative_scale,
)
from paper_search.learning.f5_production_deployment import (  # noqa: E402
    load_f5_production_ranker_bytes,
)
from paper_search.learning.cpu_document_ranker import (  # noqa: E402
    DocumentRankingQuery,
)
from paper_search.learning.large_scale_fusion_oof import (  # noqa: E402
    fold_for_query_id,
)
from paper_search.learning.large_scale_fusion_training import (  # noqa: E402
    apply_frozen_candidate_overlay,
    load_frozen_fusion_activation_inputs,
    load_training_package,
    read_fusion_checkpoint,
    read_query_shard,
    with_context_label_files,
)
from scripts.evaluate_large_scale_fusion_oof import _ranker  # noqa: E402
from paper_search.retrieval.pasa_paper_database import (  # noqa: E402
    PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE,
    PASA_TRAINING_GOLD_INJECTED_SOURCE,
)


_TRAINABLE_FAMILIES = frozenset({"entity", "task_provenance"})


def _online_retrieval_candidate_view(
    query: DocumentRankingQuery,
) -> tuple[DocumentRankingQuery, int]:
    """Exclude PASA-only training supervision while retaining online-supported rows."""

    retained = []
    for candidate in query.candidates:
        source_names = tuple(candidate.source_ranks)
        has_online_support = any(
            "pasa" not in source.casefold() for source in source_names
        )
        has_pasa_support = any("pasa" in source.casefold() for source in source_names)
        has_training_marker = bool(
            {
                PASA_TRAINING_GOLD_INJECTED_SOURCE,
                PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE,
            }.intersection(candidate.paper.sources)
        )
        if not has_online_support and (has_pasa_support or has_training_marker):
            continue
        retained.append(candidate)
    removed_count = len(query.candidates) - len(retained)
    return query.model_copy(update={"candidates": retained}), removed_count


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _model_name(
    value: float,
    *,
    reliability_scale: bool,
    task_provenance_scale: float = 1.0,
) -> str:
    if reliability_scale:
        return (
            f"reliability-scale-{value:g}-"
            f"task-provenance-scale-{task_provenance_scale:g}"
        )
    return f"anchored-alpha-{value:g}"


def _log(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}, sort_keys=True), flush=True)


def _query_ids_from_shards(shard_dir: Path) -> frozenset[str]:
    manifest_path = shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("test_partition_touched") is not False:
        raise ValueError("production shard manifest does not prove test isolation")
    batch_count = int(manifest.get("batch_count", 0))
    completed = manifest.get("completed_shards")
    if not isinstance(completed, list) or len(completed) != batch_count:
        raise ValueError("production shard manifest is incomplete")

    output: list[str] = []
    for index in range(batch_count):
        path = shard_dir / f"shard-{index:05d}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    output.append(str(row["query_id"]))
    if len(output) != int(manifest.get("query_count", -1)):
        raise ValueError("production shard query count mismatch")
    if len(output) != len(set(output)):
        raise ValueError("production shard query ids are duplicated")
    return frozenset(output)


def _gold_ranks(query: Any, ranked: Sequence[Any]) -> list[int]:
    gold = set(query.gold_paper_ids)
    return [
        rank
        for rank, candidate in enumerate(ranked, start=1)
        if paper_evaluation_id(candidate.paper) in gold
    ]


def _metrics(
    rows: Sequence[Mapping[str, object]], model: str
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("anchored fusion metrics require rows")
    recall = {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}
    reciprocal_rank = ndcg = 0.0
    for row in rows:
        gold_count = int(row["gold_count"])
        all_ranks = row.get("gold_ranks")
        if gold_count <= 0 or not isinstance(all_ranks, Mapping):
            raise ValueError("anchored fusion row is invalid")
        raw_ranks = all_ranks.get(model)
        if not isinstance(raw_ranks, list):
            raise ValueError(f"anchored fusion row is missing model: {model}")
        ranks = sorted(int(value) for value in raw_ranks)
        for cutoff in recall:
            recall[cutoff] += sum(rank <= cutoff for rank in ranks) / gold_count
        if ranks:
            reciprocal_rank += 1.0 / ranks[0]
        dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 10)
        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(gold_count, 10) + 1)
        )
        ndcg += dcg / ideal
    count = len(rows)
    return {
        "query_count": count,
        "mrr": reciprocal_rank / count,
        "ndcg_at_10": ndcg / count,
        **{
            f"recall_at_{cutoff}": total / count
            for cutoff, total in recall.items()
        },
    }


def _deltas(
    candidate: Mapping[str, float | int], baseline: Mapping[str, float | int]
) -> dict[str, float]:
    return {
        metric: float(candidate[metric]) - float(baseline[metric])
        for metric in PRIMARY_METRICS
    }


def _write_predictions(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, mtime=0, fileobj=raw
        ) as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, separators=(",", ":"), sort_keys=True).encode()
                    + b"\n"
                )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--production-manifest", type=Path, required=True)
    parser.add_argument("--production-bundle", type=Path, required=True)
    parser.add_argument("--production-shard-dir", type=Path, required=True)
    parser.add_argument("--context-freeze-manifest", type=Path, required=True)
    parser.add_argument("--oof-run-dir", type=Path, required=True)
    parser.add_argument("--oof-report", type=Path, required=True)
    candidates = parser.add_mutually_exclusive_group()
    candidates.add_argument(
        "--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75]
    )
    candidates.add_argument("--reliability-scales", type=float, nargs="+")
    parser.add_argument("--task-provenance-scale", type=float, default=1.0)
    parser.add_argument(
        "--candidate-view",
        choices=("mixed", "online-only"),
        default="mixed",
    )
    parser.add_argument(
        "--trainable-families",
        choices=("entity", "hard_constraint", "reliability", "task_provenance"),
        nargs="+",
        default=sorted(_TRAINABLE_FAMILIES),
    )
    parser.add_argument("--selection-fold", type=int, default=1)
    parser.add_argument("--max-pairs-per-query-family", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reliability_scale_mode = args.reliability_scales is not None
    trainable_families = frozenset(args.trainable_families)
    values = sorted(
        set(args.reliability_scales if reliability_scale_mode else args.alphas)
    )
    invalid_value = (
        any(not 0.0 <= value < 1.0 for value in values)
        if reliability_scale_mode
        else any(not 0.0 < value <= 1.0 for value in values)
    )
    if (
        not values
        or invalid_value
        or not 0.0 <= args.task_provenance_scale <= 1.0
        or (not reliability_scale_mode and args.task_provenance_scale != 1.0)
        or not trainable_families
        or args.selection_fold not in {1, 2, 3}
    ):
        raise ValueError("anchored fusion gate configuration is invalid")

    production_manifest_bytes = args.production_manifest.read_bytes()
    production_bundle_bytes = args.production_bundle.read_bytes()
    package = load_training_package(
        handoff_path=args.handoff,
        partition_path=args.partition,
        production_bundle_path=args.production_bundle,
    )
    activation = load_frozen_fusion_activation_inputs(
        args.context_freeze_manifest,
        expected_query_ids=package.query_ids,
        production_manifest_bytes=production_manifest_bytes,
        production_bundle_bytes=production_bundle_bytes,
    )
    package = with_context_label_files(
        package,
        task_labels_path=activation.task_labels_path,
        constraint_labels_path=activation.constraint_labels_path,
    )

    oof_report = json.loads(args.oof_report.read_text(encoding="utf-8"))
    if (
        oof_report.get("query_count") != len(package.query_ids)
        or oof_report.get("frozen_activation_package_sha256")
        != activation.manifest_sha256
        or oof_report.get("test_partition_touched") is not False
    ):
        raise ValueError("OOF report does not match the frozen activation package")
    epochs = int(oof_report.get("epochs", 0))
    if epochs <= 0:
        raise ValueError("OOF report has no completed epochs")

    production_query_ids = _query_ids_from_shards(args.production_shard_dir)
    expanded_query_ids = frozenset(package.query_ids)
    new_query_ids = new_only_query_ids(production_query_ids, expanded_query_ids)
    expanded_shard_manifest = json.loads(
        (activation.shard_dir / "manifest.json").read_text(encoding="utf-8")
    )
    completed_shards = sorted(
        expanded_shard_manifest["completed_shards"],
        key=lambda row: int(row["batch_index"]),
    )
    active_batch_indexes = new_only_batch_indexes(
        package.query_ids,
        tuple(int(row["query_count"]) for row in completed_shards),
        new_query_ids,
    )
    expected_fold_counts = {
        str(fold): sum(fold_for_query_id(query_id) == fold for query_id in new_query_ids)
        for fold in (1, 2, 3)
    }

    production_artifact = load_f5_production_ranker_bytes(
        production_manifest_bytes, production_bundle_bytes
    )
    exact_ranker = _ranker(
        package,
        args.production_manifest,
        max_pairs_per_query_family=args.max_pairs_per_query_family,
    )
    if (
        production_artifact.feature_families != exact_ranker.feature_families
        or production_artifact.dimension != exact_ranker.dimension
        or production_artifact.family_caps != exact_ranker.family_caps
    ):
        raise ValueError("production and exact fusion configurations differ")
    production_exact = copy.copy(exact_ranker)
    production_exact.weights = {
        family: values.copy()
        for family, values in production_artifact.weights.items()
    }

    anchored: dict[int, dict[float, Any]] = {}
    checkpoint_inputs: dict[str, object] = {}
    for fold in (1, 2, 3):
        checkpoint_dir = args.oof_run_dir / f"fold-{fold}" / "fit-checkpoint"
        checkpoint = read_fusion_checkpoint(checkpoint_dir)
        if (
            checkpoint.epoch_index != epochs
            or checkpoint.next_batch_index != 0
            or checkpoint.batch_count != activation.batch_count
        ):
            raise ValueError(f"OOF fold checkpoint is incomplete: {fold}")
        checkpoint_inputs[str(fold)] = {
            "checkpoint_path": str(checkpoint_dir / "checkpoint.json"),
            "checkpoint_sha256": _sha256(
                (checkpoint_dir / "checkpoint.json").read_bytes()
            ),
            "weights_path": str(checkpoint_dir / "weights.npz"),
            "weights_sha256": _sha256((checkpoint_dir / "weights.npz").read_bytes()),
        }
        anchored[fold] = {}
        for value in values:
            ranker = copy.copy(exact_ranker)
            if reliability_scale_mode:
                base_weights = scale_anchored_family_weights(
                    production_artifact.weights,
                    family="task_provenance",
                    scale=args.task_provenance_scale,
                )
                ranker.weights = scale_anchored_family_weights(
                    base_weights,
                    family="reliability",
                    scale=value,
                )
            else:
                ranker.weights = blend_anchored_family_weights(
                    production_artifact.weights,
                    checkpoint.weights,
                    alpha=value,
                    trainable_families=trainable_families,
                )
            anchored[fold][value] = ranker

    rows: list[dict[str, object]] = []
    removed_candidate_count = 0
    query_count_with_removed_candidates = 0
    _log(
        "anchored_delta_scope",
        batch_count=activation.batch_count,
        evaluated_batch_count=len(active_batch_indexes),
        skipped_batch_count=activation.batch_count - len(active_batch_indexes),
        new_only_query_count=len(new_query_ids),
    )
    for batch_index in range(activation.batch_count):
        if batch_index not in active_batch_indexes:
            continue
        for query in read_query_shard(
            activation.shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
        ):
            if query.query_id not in new_query_ids:
                continue
            if (
                args.candidate_view == "mixed"
                and query.query_id in activation.overlay_by_query_id
            ):
                query = apply_frozen_candidate_overlay(
                    query, activation.overlay_by_query_id[query.query_id]
                )
            elif args.candidate_view == "online-only":
                query, removed_count = _online_retrieval_candidate_view(query)
                removed_candidate_count += removed_count
                query_count_with_removed_candidates += int(removed_count > 0)
            fold = fold_for_query_id(query.query_id)
            b0 = list(
                exact_ranker.baseline_ranker.rank(query.query, query.candidates)
            )
            production = production_exact.rank(query.query, query.candidates)
            variants = {
                _model_name(
                    value,
                    reliability_scale=reliability_scale_mode,
                    task_provenance_scale=args.task_provenance_scale,
                ): anchored[fold][value].rank(
                    query.query, query.candidates
                )
                for value in values
            }
            expected = {paper_evaluation_id(row.paper) for row in query.candidates}
            if any(
                {paper_evaluation_id(row.paper) for row in ranked} != expected
                for ranked in [b0, production, *variants.values()]
            ):
                raise ValueError("anchored fusion ranker changed candidate identity")
            rows.append(
                {
                    "query_id": query.query_id,
                    "fold": fold,
                    "gold_count": len(query.gold_paper_ids),
                    "gold_ranks": {
                        "B0": _gold_ranks(query, b0),
                        "production": _gold_ranks(query, production),
                        **{
                            name: _gold_ranks(query, ranked)
                            for name, ranked in variants.items()
                        },
                    },
                }
            )
        _log(
            "anchored_delta_rank",
            completed_batch=batch_index + 1,
            batch_count=activation.batch_count,
            ranked_query_count=len(rows),
        )

    if len(rows) != len(new_query_ids):
        raise ValueError("anchored fusion did not rank every new-only query")
    models = [
        "B0",
        "production",
        *[
            _model_name(
                value,
                reliability_scale=reliability_scale_mode,
                task_provenance_scale=args.task_provenance_scale,
            )
            for value in values
        ],
    ]
    metrics_by_fold = {
        str(fold): {
            model: _metrics(
                [row for row in rows if int(row["fold"]) == fold], model
            )
            for model in models
        }
        for fold in (1, 2, 3)
    }
    selection_metrics = metrics_by_fold[str(args.selection_fold)]
    candidate_selection_metrics = {
        value: selection_metrics[
            _model_name(
                value,
                reliability_scale=reliability_scale_mode,
                task_provenance_scale=args.task_provenance_scale,
            )
        ]
        for value in values
    }
    selected_value = (
        select_conservative_scale(
            production_metrics=selection_metrics["production"],
            b0_metrics=selection_metrics["B0"],
            candidate_metrics_by_scale=candidate_selection_metrics,
        )
        if reliability_scale_mode
        else select_conservative_alpha(
            production_metrics=selection_metrics["production"],
            b0_metrics=selection_metrics["B0"],
            candidate_metrics_by_alpha=candidate_selection_metrics,
        )
    )
    confirmation_folds = [fold for fold in (1, 2, 3) if fold != args.selection_fold]
    confirmation_rows = [
        row for row in rows if int(row["fold"]) in confirmation_folds
    ]
    confirmation_passed = False
    confirmation: dict[str, object] | None = None
    if selected_value is not None:
        candidate_name = _model_name(
            selected_value,
            reliability_scale=reliability_scale_mode,
            task_provenance_scale=args.task_provenance_scale,
        )
        confirmation_metrics = {
            model: _metrics(confirmation_rows, model)
            for model in ("B0", "production", candidate_name)
        }
        confirmation_passed = (
            (
                select_conservative_scale(
                    production_metrics=confirmation_metrics["production"],
                    b0_metrics=confirmation_metrics["B0"],
                    candidate_metrics_by_scale={
                        selected_value: confirmation_metrics[candidate_name]
                    },
                )
                if reliability_scale_mode
                else select_conservative_alpha(
                    production_metrics=confirmation_metrics["production"],
                    b0_metrics=confirmation_metrics["B0"],
                    candidate_metrics_by_alpha={
                        selected_value: confirmation_metrics[candidate_name]
                    },
                )
            )
            is not None
        )
        confirmation = {
            "folds": confirmation_folds,
            "metrics": confirmation_metrics,
            "deltas_vs_production": _deltas(
                confirmation_metrics[candidate_name],
                confirmation_metrics["production"],
            ),
            "passed": confirmation_passed,
        }

    prediction_path = args.output.with_name(
        args.output.stem + "-gold-ranks.jsonl.gz"
    )
    _write_predictions(prediction_path, rows)
    report = {
        "schema_version": "production-anchored-fusion-delta-gate-v1",
        "query_count": len(rows),
        "production_training_query_count": len(production_query_ids),
        "expanded_training_query_count": len(expanded_query_ids),
        "new_only_query_count": len(new_query_ids),
        "evaluated_batch_count": len(active_batch_indexes),
        "new_only_fold_counts": expected_fold_counts,
        "selection_fold": args.selection_fold,
        "candidate_mode": (
            "production-reliability-scale"
            if reliability_scale_mode
            else "production-anchored-selected-family-blend"
        ),
        "candidate_view": args.candidate_view,
        "removed_training_only_candidate_count": removed_candidate_count,
        "query_count_with_removed_training_only_candidates": (
            query_count_with_removed_candidates
        ),
        "candidate_values": values,
        "base_family_scales": {
            "task_provenance": args.task_provenance_scale,
        },
        "trainable_families": (
            [] if reliability_scale_mode else sorted(trainable_families)
        ),
        "frozen_families": sorted(
            set(production_artifact.feature_families)
            - (set() if reliability_scale_mode else trainable_families)
        ),
        "calibrated_family": "reliability" if reliability_scale_mode else None,
        "metrics_by_fold": metrics_by_fold,
        "selection": {
            "selected_value": selected_value,
            "metrics": selection_metrics,
            "passed": selected_value is not None,
        },
        "confirmation": confirmation,
        "passed": selected_value is not None and confirmation_passed,
        "policy": (
            "largest-scale-nondecreasing-vs-production-and-b0-on-all-primary-"
            "metrics-with-strict-production-gain"
            if reliability_scale_mode
            else "smallest-alpha-nondecreasing-vs-production-and-b0-on-all-"
            "primary-metrics-with-strict-production-gain"
        ),
        "candidate_pool_identity_unchanged": True,
        "production_query_ids_sha256": _sha256(
            "\n".join(sorted(production_query_ids)).encode()
        ),
        "new_only_query_ids_sha256": _sha256(
            "\n".join(sorted(new_query_ids)).encode()
        ),
        "frozen_activation_package_sha256": activation.manifest_sha256,
        "prediction_shard": {
            "path": str(prediction_path),
            "sha256": _sha256(prediction_path.read_bytes()),
        },
        "inputs": {
            "production_manifest": {
                "path": str(args.production_manifest),
                "sha256": _sha256(production_manifest_bytes),
            },
            "production_bundle": {
                "path": str(args.production_bundle),
                "sha256": _sha256(production_bundle_bytes),
            },
            "context_freeze_manifest": {
                "path": str(args.context_freeze_manifest),
                "sha256": _sha256(args.context_freeze_manifest.read_bytes()),
            },
            "oof_report": {
                "path": str(args.oof_report),
                "sha256": _sha256(args.oof_report.read_bytes()),
            },
            "fold_checkpoints": checkpoint_inputs,
        },
        "development_labels_used_for_training": False,
        "online_requests_made": 0,
        "llm_requests_made": 0,
        "test_partition_touched": False,
        "production_lock_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    _log(
        "anchored_delta_complete",
        output=str(args.output),
        query_count=len(rows),
        selected_value=selected_value,
        passed=report["passed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
