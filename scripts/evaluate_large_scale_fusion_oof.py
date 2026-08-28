"""Run resumable three-fold OOF evaluation over frozen F4/F5 query shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.evaluation.predictions import paper_evaluation_id  # noqa: E402
from paper_search.learning.f5_production_deployment import _jsonl_rows  # noqa: E402
from paper_search.learning.gated_feature_fusion_ranker import (  # noqa: E402
    FUSION_FAMILIES,
    FrozenFusionContextStore,
    GatedFeatureFusionRanker,
    load_gated_feature_fusion_ranker_bytes,
)
from paper_search.learning.large_scale_fusion_oof import (  # noqa: E402
    _comparison as _oof_metric_comparison,
    _metrics as _oof_metrics,
    build_oof_comparison,
    fold_for_query_id,
)
from paper_search.learning.large_scale_fusion_training import (  # noqa: E402
    FusionTrainingCheckpoint,
    FrozenFusionActivationInputs,
    apply_frozen_candidate_overlay,
    load_frozen_fusion_activation_inputs,
    load_training_package,
    read_fusion_checkpoint,
    read_query_shard,
    unpack_production_bundle_components,
    with_context_label_files,
    write_fusion_checkpoint,
)
from paper_search.retrieval.pasa_paper_database import (  # noqa: E402
    ARXIV_MISSING_YEAR_EVIDENCE_POLICY,
)
from scripts.train_large_scale_fusion import (  # noqa: E402
    DEFAULT_PAIR_BUDGET_BY_FAMILY,
    _fit_input_sha256,
    _load_production_replay_manifest,
    _production_replay_batch_index,
    _read_production_replay_shard,
    _validated_pair_budgets,
)
from paper_search.learning.query_constraint_annotations import (  # noqa: E402
    FrozenConstraintAnnotation,
    FrozenConstraintProfileStore,
)
from paper_search.learning.task_slot_document_ranker import (  # noqa: E402
    FrozenTaskSlotLabelStore,
)
from paper_search.ranking.cpu_document import (  # noqa: E402
    load_cpu_document_ranking_stage_bytes,
)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _log(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}, sort_keys=True), flush=True)


@dataclass(frozen=True)
class IncrementalOOFSplit:
    """A leakage-safe fold over only the queries added after production fit."""

    fold: int
    fold_by_query_id: dict[str, int]
    incremental_query_ids: frozenset[str]
    held_out_query_ids: frozenset[str]
    training_query_ids: frozenset[str]


def _incremental_oof_split(
    *,
    package_query_ids: Sequence[str],
    base_query_ids: set[str] | frozenset[str],
    fold: int,
) -> IncrementalOOFSplit:
    if fold not in {1, 2, 3}:
        raise ValueError("incremental OOF fold must be 1, 2, or 3")
    package_ids = list(package_query_ids)
    package_set = set(package_ids)
    if len(package_ids) != len(package_set):
        raise ValueError("expanded package query IDs must be unique")
    missing = set(base_query_ids) - package_set
    if missing:
        raise ValueError("base query IDs must be contained in the expanded package")
    incremental = package_set - set(base_query_ids)
    if not incremental:
        raise ValueError("incremental OOF requires queries beyond the production base")
    fold_by_query_id = {
        query_id: fold_for_query_id(query_id) for query_id in incremental
    }
    held_out = {
        query_id
        for query_id, assigned_fold in fold_by_query_id.items()
        if assigned_fold == fold
    }
    if not held_out:
        raise ValueError(f"incremental OOF fold {fold} is empty")
    return IncrementalOOFSplit(
        fold=fold,
        fold_by_query_id=fold_by_query_id,
        incremental_query_ids=frozenset(incremental),
        held_out_query_ids=frozenset(held_out),
        training_query_ids=frozenset(package_set - held_out),
    )


def _ranker(
    package: Any,
    production_manifest_path: Path,
    *,
    max_pairs_per_query_family: int,
) -> GatedFeatureFusionRanker:
    production = json.loads(production_manifest_path.read_text(encoding="utf-8"))
    components = unpack_production_bundle_components(
        package.production_bundle_path.read_bytes()
    )
    baseline = load_cpu_document_ranking_stage_bytes(
        (json.dumps(production["baseline_manifest"], sort_keys=True) + "\n").encode(),
        components["baseline_weights"],
    ).ranker
    task_store = FrozenTaskSlotLabelStore.from_jsonl_bytes(package.task_labels_bytes)
    constraints = FrozenConstraintProfileStore(
        [
            FrozenConstraintAnnotation.model_validate(row)
            for row in _jsonl_rows(
                package.constraint_labels_bytes, label="constraint labels"
            )
        ]
    )
    config = production["f5_manifest"]
    return GatedFeatureFusionRanker(
        baseline_ranker=baseline,
        context_store=FrozenFusionContextStore(
            task_store=task_store, constraint_store=constraints
        ),
        feature_families=set(FUSION_FAMILIES),
        gated=True,
        dimension=int(config["dimension_per_family"]),
        epochs=1,
        family_caps={str(k): float(v) for k, v in config["family_caps"].items()},
        maximum_total_residual=float(config["maximum_total_residual"]),
        constraint_text_evidence=True,
        runtime_context_scoring=True,
        max_pairs_per_query_family=max_pairs_per_query_family,
        publication_year_evidence_policy=ARXIV_MISSING_YEAR_EVIDENCE_POLICY,
    )


def _warm_started_ranker(
    package: Any,
    production_manifest_path: Path,
    *,
    pair_budget_by_family: Mapping[str, int],
) -> tuple[GatedFeatureFusionRanker, dict[str, Any], dict[str, Any]]:
    """Build the same production-warm-start ranker used by the full fit."""

    production = json.loads(production_manifest_path.read_text(encoding="utf-8"))
    components = unpack_production_bundle_components(
        package.production_bundle_path.read_bytes()
    )
    baseline = load_cpu_document_ranking_stage_bytes(
        (json.dumps(production["baseline_manifest"], sort_keys=True) + "\n").encode(),
        components["baseline_weights"],
    ).ranker
    task_store = FrozenTaskSlotLabelStore.from_jsonl_bytes(package.task_labels_bytes)
    constraints = FrozenConstraintProfileStore(
        [
            FrozenConstraintAnnotation.model_validate(row)
            for row in _jsonl_rows(
                package.constraint_labels_bytes, label="constraint labels"
            )
        ]
    )
    context_store = FrozenFusionContextStore(
        task_store=task_store,
        constraint_store=constraints,
    )
    config = production["f5_manifest"]
    budgets = _validated_pair_budgets(pair_budget_by_family)
    ranker = GatedFeatureFusionRanker(
        baseline_ranker=baseline,
        context_store=context_store,
        feature_families=set(FUSION_FAMILIES),
        gated=True,
        dimension=int(config["dimension_per_family"]),
        epochs=1,
        family_caps={str(k): float(v) for k, v in config["family_caps"].items()},
        maximum_total_residual=float(config["maximum_total_residual"]),
        constraint_text_evidence=True,
        runtime_context_scoring=True,
        pair_budget_by_family=budgets,
        publication_year_evidence_policy=ARXIV_MISSING_YEAR_EVIDENCE_POLICY,
    )
    production_weights = components["fusion_weights"]
    loaded = load_gated_feature_fusion_ranker_bytes(
        (json.dumps(config, sort_keys=True) + "\n").encode("utf-8"),
        production_weights,
        baseline_ranker=baseline,
        context_store=context_store,
    )
    if (
        loaded.feature_families != ranker.feature_families
        or loaded.dimension != ranker.dimension
        or loaded.family_caps != ranker.family_caps
        or loaded.maximum_total_residual != ranker.maximum_total_residual
    ):
        raise ValueError("production F5 is incompatible with incremental OOF")
    production_family_weights = {
        family: weights.copy() for family, weights in loaded.weights.items()
    }
    ranker.weights = {
        family: weights.copy() for family, weights in production_family_weights.items()
    }
    warm_start = {
        "strategy": "production-f5-exact-weight-initialization-v1",
        "model_id": loaded.model_id,
        "weights_sha256": _sha256(production_weights),
        "family_weight_sha256": dict(config["family_weight_sha256"]),
        "training_query_count": int(config.get("training_query_count", 0)),
    }
    return ranker, production_family_weights, warm_start


def _query_ids_from_replay_shards(
    *, shard_dir: Path, manifest: Mapping[str, Any]
) -> set[str]:
    query_ids: set[str] = set()
    for batch_index, record in enumerate(manifest["completed_shards"]):
        if int(record.get("batch_index", -1)) != batch_index:
            raise ValueError("production replay shard order mismatch")
        path = shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
        if record.get("sha256") != _sha256(path.read_bytes()):
            raise ValueError(f"production replay shard hash mismatch: {path}")
        for query in read_query_shard(path):
            if query.query_id in query_ids:
                raise ValueError("production replay query IDs must be unique")
            query_ids.add(query.query_id)
    expected = int(manifest.get("query_count", 0))
    if expected <= 0 or len(query_ids) != expected:
        raise ValueError("production replay query count does not match its manifest")
    return query_ids


def _anchor_families_to_production(
    ranker: GatedFeatureFusionRanker,
    *,
    production_family_weights: Mapping[str, Any],
    anchored_families: set[str] | frozenset[str],
) -> None:
    unknown = set(anchored_families) - set(ranker.feature_families)
    if unknown:
        raise ValueError(f"cannot anchor unknown feature families: {sorted(unknown)}")
    for family in anchored_families:
        ranker.weights[family] = production_family_weights[family].copy()


def _legacy_checkpoint_identity(
    input_sha256: str,
    fold: int,
    epochs: int,
    max_pairs_per_query_family: int,
) -> str:
    return _sha256(
        (
            f"{input_sha256}|oof-fold={fold}|epochs={epochs}|"
            "oof-checkpoint-schema=v2-frozen-activation-inputs|"
            f"max-pairs-per-query-family={max_pairs_per_query_family}"
        ).encode()
    )


def _query_id_set_sha256(query_ids: Sequence[str] | set[str] | frozenset[str]) -> str:
    return _sha256(("\n".join(sorted(query_ids)) + "\n").encode("utf-8"))


def _checkpoint_identity(
    input_sha256: str,
    *,
    fold: int,
    epochs: int,
    held_out_query_ids: set[str] | frozenset[str],
    anchored_families: set[str] | frozenset[str],
) -> str:
    return _sha256(
        (
            f"{input_sha256}|oof-fold={fold}|epochs={epochs}|"
            "oof-checkpoint-schema=v3-incremental-production-warmstart|"
            f"held-out-query-ids={_query_id_set_sha256(held_out_query_ids)}|"
            "anchored-families="
            f"{','.join(sorted(anchored_families)) or 'none'}"
        ).encode("utf-8")
    )


def _fit_fold(
    *,
    ranker: GatedFeatureFusionRanker,
    fold: int,
    epochs: int,
    shard_dir: Path,
    checkpoint_dir: Path,
    batch_count: int,
    input_sha256: str,
    activation_inputs: FrozenFusionActivationInputs | None,
) -> None:
    identity = _legacy_checkpoint_identity(
        input_sha256,
        fold,
        epochs,
        ranker.max_pairs_per_query_family or 0,
    )
    pair_counts = {family: 0 for family in ranker.feature_families}
    query_counts = {family: 0 for family in ranker.feature_families}
    epoch_index = next_batch_index = 0
    if (checkpoint_dir / "checkpoint.json").is_file():
        checkpoint = read_fusion_checkpoint(checkpoint_dir)
        if checkpoint.input_sha256 != identity or checkpoint.batch_count != batch_count:
            raise ValueError("OOF checkpoint does not match fold inputs")
        ranker.weights = {name: value.copy() for name, value in checkpoint.weights.items()}
        pair_counts = dict(checkpoint.pair_counts)
        query_counts = dict(checkpoint.query_counts)
        epoch_index = checkpoint.epoch_index
        next_batch_index = checkpoint.next_batch_index
        _log("oof_resume", fold=fold, epoch=epoch_index, batch=next_batch_index)
    started = time.monotonic()
    for epoch in range(epoch_index, epochs):
        first_batch = next_batch_index if epoch == epoch_index else 0
        for batch_index in range(first_batch, batch_count):
            train = [
                query
                for query in read_query_shard(
                    shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
                )
                if fold_for_query_id(query.query_id) != fold
            ]
            if activation_inputs is not None:
                train = [
                    apply_frozen_candidate_overlay(
                        query, activation_inputs.overlay_by_query_id[query.query_id]
                    )
                    if query.query_id in activation_inputs.overlay_by_query_id
                    else query
                    for query in train
                ]
            batch_pairs = ranker.fit(train) if train else pair_counts.fromkeys(pair_counts, 0)
            if epoch == 0:
                for family in ranker.feature_families:
                    pair_counts[family] += int(batch_pairs[family])
                    query_counts[family] += int(ranker.last_fit_query_count[family])
            saved_epoch, saved_batch = (
                (epoch + 1, 0)
                if batch_index + 1 == batch_count
                else (epoch, batch_index + 1)
            )
            write_fusion_checkpoint(
                checkpoint_dir,
                FusionTrainingCheckpoint(
                    input_sha256=identity,
                    epoch_index=saved_epoch,
                    next_batch_index=saved_batch,
                    batch_count=batch_count,
                    pair_counts=pair_counts,
                    query_counts=query_counts,
                    weights={name: value.copy() for name, value in ranker.weights.items()},
                ),
            )
            _log(
                "oof_fit",
                fold=fold,
                epoch=epoch + 1,
                epochs=epochs,
                completed_batch=batch_index + 1,
                batch_count=batch_count,
                elapsed_seconds=round(time.monotonic() - started, 1),
            )
        next_batch_index = 0


def _gold_ranks(query: Any, ranked: Sequence[Any]) -> list[int]:
    gold = set(query.gold_paper_ids)
    return [
        rank
        for rank, candidate in enumerate(ranked, 1)
        if paper_evaluation_id(candidate.paper) in gold
    ]


def _evaluate_fold(
    *, ranker: GatedFeatureFusionRanker, fold: int, shard_dir: Path, batch_count: int,
    output_path: Path, activation_inputs: FrozenFusionActivationInputs | None,
) -> None:
    temporary = output_path.with_name(output_path.name + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=6, mtime=0, fileobj=raw) as stream:
            for batch_index in range(batch_count):
                queries = [
                    query
                    for query in read_query_shard(
                        shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
                    )
                    if fold_for_query_id(query.query_id) == fold
                ]
                if activation_inputs is not None:
                    queries = [
                        apply_frozen_candidate_overlay(
                            query,
                            activation_inputs.overlay_by_query_id[query.query_id],
                        )
                        if query.query_id in activation_inputs.overlay_by_query_id
                        else query
                        for query in queries
                    ]
                for query in queries:
                    b0 = list(ranker.baseline_ranker.rank(query.query, query.candidates))
                    f4 = ranker.rank_variant(
                        query.query, query.candidates,
                        families=frozenset({"reliability"}), gated=True,
                    )
                    f5 = ranker.rank(query.query, query.candidates)
                    expected = {paper_evaluation_id(row.paper) for row in query.candidates}
                    if any(
                        {paper_evaluation_id(row.paper) for row in ranked} != expected
                        for ranked in (b0, f4, f5)
                    ):
                        raise ValueError("OOF ranker changed candidate identity")
                    row = {
                        "query_id": query.query_id,
                        "fold": fold,
                        "gold_count": len(query.gold_paper_ids),
                        "gold_ranks": {
                            "B0": _gold_ranks(query, b0),
                            "F4": _gold_ranks(query, f4),
                            "F5": _gold_ranks(query, f5),
                        },
                    }
                    stream.write(json.dumps(row, separators=(",", ":")).encode() + b"\n")
                    count += 1
                _log(
                    "oof_rank",
                    fold=fold,
                    completed_batch=batch_index + 1,
                    batch_count=batch_count,
                    ranked_query_count=count,
                )
    temporary.replace(output_path)


def _fit_incremental_fold(
    *,
    package: Any,
    ranker: GatedFeatureFusionRanker,
    fold: int,
    held_out_query_ids: frozenset[str],
    anchored_families: frozenset[str],
    epochs: int,
    shard_dir: Path,
    checkpoint_dir: Path,
    batch_count: int,
    input_sha256: str,
    activation_inputs: FrozenFusionActivationInputs | None,
    production_replay_shard_dir: Path,
    replay_manifest: Mapping[str, Any],
    replay_every_batches: int,
) -> None:
    identity = _checkpoint_identity(
        input_sha256,
        fold=fold,
        epochs=epochs,
        held_out_query_ids=held_out_query_ids,
        anchored_families=anchored_families,
    )
    replay_batch_count = int(replay_manifest["batch_count"])
    pair_counts = {family: 0 for family in ranker.feature_families}
    query_counts = {family: 0 for family in ranker.feature_families}
    replay_pair_counts = {family: 0 for family in ranker.feature_families}
    replay_query_counts = {family: 0 for family in ranker.feature_families}
    epoch_index = next_batch_index = 0
    if (checkpoint_dir / "checkpoint.json").is_file():
        checkpoint = read_fusion_checkpoint(checkpoint_dir)
        if (
            checkpoint.input_sha256 != identity
            or checkpoint.batch_count != batch_count
            or set(checkpoint.weights) != set(ranker.feature_families)
        ):
            raise ValueError("incremental OOF checkpoint does not match fold inputs")
        ranker.weights = {
            name: value.copy() for name, value in checkpoint.weights.items()
        }
        pair_counts = dict(checkpoint.pair_counts)
        query_counts = dict(checkpoint.query_counts)
        replay_pair_counts = dict(checkpoint.replay_pair_counts)
        replay_query_counts = dict(checkpoint.replay_query_counts)
        epoch_index = checkpoint.epoch_index
        next_batch_index = checkpoint.next_batch_index
        _log("incremental_oof_resume", fold=fold, epoch=epoch_index, batch=next_batch_index)
    started = time.monotonic()
    for epoch in range(epoch_index, epochs):
        first_batch = next_batch_index if epoch == epoch_index else 0
        for batch_index in range(first_batch, batch_count):
            train = [
                query
                for query in read_query_shard(
                    shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
                )
                if query.query_id not in held_out_query_ids
            ]
            if activation_inputs is not None:
                train = [
                    apply_frozen_candidate_overlay(
                        query, activation_inputs.overlay_by_query_id[query.query_id]
                    )
                    if query.query_id in activation_inputs.overlay_by_query_id
                    else query
                    for query in train
                ]
            batch_pairs = (
                ranker.fit(train)
                if train
                else {family: 0 for family in ranker.feature_families}
            )
            if epoch == 0:
                for family in ranker.feature_families:
                    pair_counts[family] += int(batch_pairs[family])
                    query_counts[family] += int(ranker.last_fit_query_count[family])

            replay_batch_index = _production_replay_batch_index(
                batch_index,
                training_batch_count=batch_count,
                replay_batch_count=replay_batch_count,
                replay_every_batches=replay_every_batches,
            )
            if replay_batch_index is not None:
                replay_queries = _read_production_replay_shard(
                    shard_dir=production_replay_shard_dir,
                    manifest=replay_manifest,
                    batch_index=replay_batch_index,
                    package=package,
                )
                if any(
                    query.query_id in held_out_query_ids for query in replay_queries
                ):
                    raise ValueError("production replay leaked an incremental holdout")
                if replay_queries:
                    replay_pairs = ranker.fit(replay_queries)
                    if epoch == 0:
                        for family in ranker.feature_families:
                            replay_pair_counts[family] += int(replay_pairs[family])
                            replay_query_counts[family] += int(
                                ranker.last_fit_query_count[family]
                            )

            saved_epoch, saved_batch = (
                (epoch + 1, 0)
                if batch_index + 1 == batch_count
                else (epoch, batch_index + 1)
            )
            write_fusion_checkpoint(
                checkpoint_dir,
                FusionTrainingCheckpoint(
                    input_sha256=identity,
                    epoch_index=saved_epoch,
                    next_batch_index=saved_batch,
                    batch_count=batch_count,
                    pair_counts=pair_counts,
                    query_counts=query_counts,
                    replay_pair_counts=replay_pair_counts,
                    replay_query_counts=replay_query_counts,
                    weights={
                        name: value.copy() for name, value in ranker.weights.items()
                    },
                ),
            )
            if (batch_index + 1) % 25 == 0 or batch_index + 1 == batch_count:
                _log(
                    "incremental_oof_fit",
                    fold=fold,
                    epoch=epoch + 1,
                    epochs=epochs,
                    completed_batch=batch_index + 1,
                    batch_count=batch_count,
                    elapsed_seconds=round(time.monotonic() - started, 1),
                )
        next_batch_index = 0


def _evaluate_incremental_fold(
    *,
    ranker: GatedFeatureFusionRanker,
    production_ranker: GatedFeatureFusionRanker,
    fold: int,
    held_out_query_ids: frozenset[str],
    shard_dir: Path,
    batch_count: int,
    output_path: Path,
    activation_inputs: FrozenFusionActivationInputs | None,
) -> None:
    temporary = output_path.with_name(output_path.name + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[str] = set()
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, mtime=0, fileobj=raw
        ) as stream:
            for batch_index in range(batch_count):
                queries = [
                    query
                    for query in read_query_shard(
                        shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
                    )
                    if query.query_id in held_out_query_ids
                ]
                if activation_inputs is not None:
                    queries = [
                        apply_frozen_candidate_overlay(
                            query,
                            activation_inputs.overlay_by_query_id[query.query_id],
                        )
                        if query.query_id in activation_inputs.overlay_by_query_id
                        else query
                        for query in queries
                    ]
                for query in queries:
                    if query.query_id in seen:
                        raise ValueError("incremental OOF evaluated a query twice")
                    seen.add(query.query_id)
                    b0 = list(ranker.baseline_ranker.rank(query.query, query.candidates))
                    f4 = ranker.rank_variant(
                        query.query,
                        query.candidates,
                        families=frozenset({"reliability"}),
                        gated=True,
                    )
                    f5 = ranker.rank(query.query, query.candidates)
                    production_f4 = production_ranker.rank_variant(
                        query.query,
                        query.candidates,
                        families=frozenset({"reliability"}),
                        gated=True,
                    )
                    production_f5 = production_ranker.rank(
                        query.query, query.candidates
                    )
                    expected = {
                        paper_evaluation_id(row.paper) for row in query.candidates
                    }
                    if any(
                        {paper_evaluation_id(row.paper) for row in ranked} != expected
                        for ranked in (
                            b0,
                            f4,
                            f5,
                            production_f4,
                            production_f5,
                        )
                    ):
                        raise ValueError("incremental OOF ranker changed candidate identity")
                    stream.write(
                        json.dumps(
                            {
                                "query_id": query.query_id,
                                "fold": fold,
                                "gold_count": len(query.gold_paper_ids),
                                "gold_ranks": {
                                    "B0": _gold_ranks(query, b0),
                                    "F4": _gold_ranks(query, f4),
                                    "F5": _gold_ranks(query, f5),
                                    "PRODUCTION_F4": _gold_ranks(
                                        query, production_f4
                                    ),
                                    "PRODUCTION_F5": _gold_ranks(
                                        query, production_f5
                                    ),
                                },
                            },
                            separators=(",", ":"),
                        ).encode()
                        + b"\n"
                    )
                    count += 1
    if seen != set(held_out_query_ids):
        raise ValueError("incremental OOF did not evaluate the exact held-out fold")
    temporary.replace(output_path)
    _log("incremental_oof_rank", fold=fold, ranked_query_count=count)


def _build_incremental_oof_comparison(
    rows: Sequence[Mapping[str, object]], *, incremental_query_count: int
) -> dict[str, object]:
    report = build_oof_comparison(
        rows, training_query_count=incremental_query_count
    )
    validated = list(rows)
    folds = (1, 2, 3)
    production_models = ("PRODUCTION_F4", "PRODUCTION_F5")
    production_overall = {
        model: _oof_metrics(validated, model) for model in production_models
    }
    production_by_fold = {
        str(fold): {
            model: _oof_metrics(
                [row for row in validated if int(row["fold"]) == fold], model
            )
            for model in production_models
        }
        for fold in folds
    }
    metrics = report["metrics"]
    metrics_by_fold = report["metrics_by_fold"]
    if not isinstance(metrics, dict) or not isinstance(metrics_by_fold, dict):
        raise ValueError("OOF report metrics have an unexpected schema")
    metrics.update(production_overall)
    for fold in folds:
        fold_metrics = metrics_by_fold[str(fold)]
        if not isinstance(fold_metrics, dict):
            raise ValueError("OOF fold metrics have an unexpected schema")
        fold_metrics.update(production_by_fold[str(fold)])

    def comparison(candidate: str, baseline: str) -> dict[str, object]:
        return {
            "overall": _oof_metric_comparison(
                production_overall[baseline], metrics[candidate]
            ),
            "folds": {
                str(fold): _oof_metric_comparison(
                    production_by_fold[str(fold)][baseline],
                    metrics_by_fold[str(fold)][candidate],
                )
                for fold in folds
            },
        }

    report["candidate_vs_production"] = {
        "F4_vs_PRODUCTION_F4": comparison("F4", "PRODUCTION_F4"),
        "F5_vs_PRODUCTION_F5": comparison("F5", "PRODUCTION_F5"),
    }
    return report


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_incremental_oof(
    *,
    args: Any,
    package: Any,
    shard_dir: Path,
    batch_count: int,
    activation_inputs: FrozenFusionActivationInputs | None,
) -> int:
    if args.production_replay_shard_dir is None:
        raise ValueError(
            "incremental production-warm-start OOF requires production replay shards"
        )
    if args.production_replay_every_batches <= 0:
        raise ValueError("production replay interval must be positive")
    folds = sorted(set(args.folds))
    if not folds or any(fold not in {1, 2, 3} for fold in folds):
        raise ValueError("OOF folds must be selected from 1, 2, and 3")
    anchored_families = frozenset(args.anchor_family)
    pair_budgets = _validated_pair_budgets(
        {
            "entity": args.entity_pair_budget,
            "hard_constraint": args.hard_constraint_pair_budget,
            "reliability": args.reliability_pair_budget,
            "task_provenance": args.task_provenance_pair_budget,
        }
    )
    replay_manifest, replay_manifest_sha256 = _load_production_replay_manifest(
        args.production_replay_shard_dir
    )
    base_query_ids = _query_ids_from_replay_shards(
        shard_dir=args.production_replay_shard_dir,
        manifest=replay_manifest,
    )
    splits = {
        fold: _incremental_oof_split(
            package_query_ids=package.query_ids,
            base_query_ids=base_query_ids,
            fold=fold,
        )
        for fold in (1, 2, 3)
    }
    incremental_query_ids = splits[1].incremental_query_ids
    if any(
        split.incremental_query_ids != incremental_query_ids
        for split in splits.values()
    ):
        raise ValueError("incremental OOF folds disagree on the added query set")
    if set().union(
        *(set(split.held_out_query_ids) for split in splits.values())
    ) != set(incremental_query_ids):
        raise ValueError("incremental OOF folds do not cover every added query")

    prediction_paths: dict[int, Path] = {}
    training_input_sha256: str | None = None
    warm_start: dict[str, Any] | None = None
    for fold in folds:
        split = splits[fold]
        ranker, production_family_weights, current_warm_start = _warm_started_ranker(
            package,
            args.production_manifest,
            pair_budget_by_family=pair_budgets,
        )
        current_input_sha256 = _fit_input_sha256(
            package.input_sha256,
            pair_budget_by_family=pair_budgets,
            activation_manifest_sha256=(
                activation_inputs.manifest_sha256 if activation_inputs else None
            ),
            warm_start_weights_sha256=str(current_warm_start["weights_sha256"]),
            replay_manifest_sha256=replay_manifest_sha256,
            replay_every_batches=args.production_replay_every_batches,
        )
        if training_input_sha256 is None:
            training_input_sha256 = current_input_sha256
            warm_start = current_warm_start
        elif (
            current_input_sha256 != training_input_sha256
            or current_warm_start != warm_start
        ):
            raise ValueError("incremental OOF warm-start inputs changed between folds")
        _fit_incremental_fold(
            package=package,
            ranker=ranker,
            fold=fold,
            held_out_query_ids=split.held_out_query_ids,
            anchored_families=anchored_families,
            epochs=args.epochs,
            shard_dir=shard_dir,
            checkpoint_dir=args.run_dir / f"fold-{fold}" / "fit-checkpoint",
            batch_count=batch_count,
            input_sha256=current_input_sha256,
            activation_inputs=activation_inputs,
            production_replay_shard_dir=args.production_replay_shard_dir,
            replay_manifest=replay_manifest,
            replay_every_batches=args.production_replay_every_batches,
        )
        _anchor_families_to_production(
            ranker,
            production_family_weights=production_family_weights,
            anchored_families=anchored_families,
        )
        production_ranker, _, production_warm_start = _warm_started_ranker(
            package,
            args.production_manifest,
            pair_budget_by_family=pair_budgets,
        )
        if production_warm_start != current_warm_start:
            raise ValueError("production comparison ranker changed within a fold")
        prediction_path = (
            args.run_dir
            / f"fold-{fold}"
            / "gold-ranks-with-production-v2.jsonl.gz"
        )
        if not prediction_path.is_file():
            _evaluate_incremental_fold(
                ranker=ranker,
                production_ranker=production_ranker,
                fold=fold,
                held_out_query_ids=split.held_out_query_ids,
                shard_dir=shard_dir,
                batch_count=batch_count,
                output_path=prediction_path,
                activation_inputs=activation_inputs,
            )
        prediction_paths[fold] = prediction_path

    complete_paths = {
        fold: (
            args.run_dir
            / f"fold-{fold}"
            / "gold-ranks-with-production-v2.jsonl.gz"
        )
        for fold in (1, 2, 3)
    }
    if not all(path.is_file() for path in complete_paths.values()):
        _write_json(
            args.output,
            {
                "schema_version": "incremental-production-warmstart-oof-partial-v1",
                "oof_complete": False,
                "completed_folds": sorted(
                    fold for fold, path in complete_paths.items() if path.is_file()
                ),
                "expanded_training_query_count": len(package.query_ids),
                "production_base_query_count": len(base_query_ids),
                "incremental_query_count": len(incremental_query_ids),
                "production_promotion_authorized": False,
                "online_requests_made": 0,
                "test_partition_touched": False,
            },
        )
        _log("incremental_oof_partial", completed_folds=folds)
        return 0

    rows: list[dict[str, object]] = []
    for path in complete_paths.values():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    report = _build_incremental_oof_comparison(
        rows, incremental_query_count=len(incremental_query_ids)
    )
    report.update(
        {
            "schema_version": (
                "incremental-production-warmstart-gated-fusion-oof-v1"
            ),
            "oof_scope": "new-only-incremental-production-warmstart-v1",
            "oof_complete": True,
            "expanded_training_query_count": len(package.query_ids),
            "production_base_query_count": len(base_query_ids),
            "incremental_query_count": len(incremental_query_ids),
            "training_input_sha256": training_input_sha256,
            "training_package_sha256": package.input_sha256,
            "frozen_activation_package_sha256": (
                activation_inputs.manifest_sha256 if activation_inputs else None
            ),
            "pair_budget_by_family": pair_budgets,
            "production_warm_start": warm_start,
            "production_pair_replay": {
                "enabled": True,
                "source_manifest_sha256": replay_manifest_sha256,
                "every_training_batches": args.production_replay_every_batches,
                "current_exact_gold_rebound": True,
                "training_injected_candidates_removed": True,
            },
            "anchored_families": sorted(anchored_families),
            "epochs": args.epochs,
            "prediction_shards": [
                {
                    "fold": fold,
                    "path": str(path),
                    "sha256": _sha256(path.read_bytes()),
                }
                for fold, path in complete_paths.items()
            ],
            "production_promotion_authorized": False,
        }
    )
    _write_json(args.output, report)
    _log("incremental_oof_complete", output=str(args.output), query_count=len(rows))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--production-manifest", type=Path, required=True)
    parser.add_argument("--production-bundle", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-pairs-per-query-family", type=int, default=32)
    parser.add_argument(
        "--reliability-pair-budget",
        type=int,
        default=DEFAULT_PAIR_BUDGET_BY_FAMILY["reliability"],
    )
    parser.add_argument(
        "--task-provenance-pair-budget",
        type=int,
        default=DEFAULT_PAIR_BUDGET_BY_FAMILY["task_provenance"],
    )
    parser.add_argument(
        "--entity-pair-budget",
        type=int,
        default=DEFAULT_PAIR_BUDGET_BY_FAMILY["entity"],
    )
    parser.add_argument(
        "--hard-constraint-pair-budget",
        type=int,
        default=DEFAULT_PAIR_BUDGET_BY_FAMILY["hard_constraint"],
    )
    parser.add_argument("--production-replay-shard-dir", type=Path)
    parser.add_argument("--production-replay-every-batches", type=int, default=1)
    parser.add_argument("--incremental-new-only", action="store_true")
    parser.add_argument(
        "--anchor-family",
        action="append",
        choices=sorted(FUSION_FAMILIES),
        default=[],
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--task-labels", type=Path)
    parser.add_argument("--constraint-labels", type=Path)
    parser.add_argument("--context-freeze-manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0 or args.max_pairs_per_query_family <= 0:
        raise ValueError("OOF epochs must be positive")
    if args.incremental_new_only:
        _validated_pair_budgets(
            {
                "entity": args.entity_pair_budget,
                "hard_constraint": args.hard_constraint_pair_budget,
                "reliability": args.reliability_pair_budget,
                "task_provenance": args.task_provenance_pair_budget,
            }
        )
    elif (
        args.production_replay_shard_dir is not None
        or args.anchor_family
        or args.folds != [1, 2, 3]
    ):
        raise ValueError(
            "production replay, family anchors, and fold selection require "
            "--incremental-new-only"
        )
    package = load_training_package(
        handoff_path=args.handoff,
        partition_path=args.partition,
        production_bundle_path=args.production_bundle,
    )
    if args.context_freeze_manifest is not None and (
        args.task_labels is not None
        or args.constraint_labels is not None
        or args.shard_dir is not None
    ):
        raise ValueError(
            "context freeze manifest cannot be mixed with shard or label overrides"
        )
    if (args.task_labels is None) != (args.constraint_labels is None):
        raise ValueError("task and constraint label files must be supplied together")
    activation_inputs: FrozenFusionActivationInputs | None = None
    if args.context_freeze_manifest is not None:
        activation_inputs = load_frozen_fusion_activation_inputs(
            args.context_freeze_manifest,
            expected_query_ids=package.query_ids,
            production_manifest_bytes=args.production_manifest.read_bytes(),
            production_bundle_bytes=args.production_bundle.read_bytes(),
        )
        package = with_context_label_files(
            package,
            task_labels_path=activation_inputs.task_labels_path,
            constraint_labels_path=activation_inputs.constraint_labels_path,
        )
    if args.task_labels is not None and args.constraint_labels is not None:
        package = with_context_label_files(
            package,
            task_labels_path=args.task_labels,
            constraint_labels_path=args.constraint_labels,
        )
    if activation_inputs is not None:
        shard_dir = activation_inputs.shard_dir
        batch_count = activation_inputs.batch_count
    else:
        if args.shard_dir is None:
            raise ValueError("OOF requires shard-dir or context-freeze-manifest")
        shard_dir = args.shard_dir
        shard_manifest = json.loads(
            (shard_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            shard_manifest.get("input_sha256") != package.input_sha256
            or shard_manifest.get("query_count") != len(package.query_ids)
            or shard_manifest.get("test_partition_touched") is not False
        ):
            raise ValueError("OOF shards do not match the frozen training package")
        batch_count = int(shard_manifest["batch_count"])
    if args.incremental_new_only:
        return _run_incremental_oof(
            args=args,
            package=package,
            shard_dir=shard_dir,
            batch_count=batch_count,
            activation_inputs=activation_inputs,
        )
    training_input_sha256 = _fit_input_sha256(
        package.input_sha256,
        max_pairs_per_query_family=args.max_pairs_per_query_family,
        activation_manifest_sha256=(
            activation_inputs.manifest_sha256 if activation_inputs else None
        ),
    )
    prediction_paths: list[Path] = []
    for fold in (1, 2, 3):
        ranker = _ranker(
            package,
            args.production_manifest,
            max_pairs_per_query_family=args.max_pairs_per_query_family,
        )
        _fit_fold(
            ranker=ranker,
            fold=fold,
            epochs=args.epochs,
            shard_dir=shard_dir,
            checkpoint_dir=args.run_dir / f"fold-{fold}" / "fit-checkpoint",
            batch_count=batch_count,
            input_sha256=training_input_sha256,
            activation_inputs=activation_inputs,
        )
        prediction_path = args.run_dir / f"fold-{fold}" / "gold-ranks.jsonl.gz"
        if not prediction_path.is_file():
            _evaluate_fold(
                ranker=ranker,
                fold=fold,
                shard_dir=shard_dir,
                batch_count=batch_count,
                output_path=prediction_path,
                activation_inputs=activation_inputs,
            )
        prediction_paths.append(prediction_path)
    rows: list[dict[str, object]] = []
    for path in prediction_paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    report = build_oof_comparison(rows, training_query_count=len(package.query_ids))
    report["training_input_sha256"] = training_input_sha256
    report["frozen_activation_package_sha256"] = (
        activation_inputs.manifest_sha256 if activation_inputs else None
    )
    report["training_package_sha256"] = package.input_sha256
    report["constraint_text_evidence"] = True
    report["epochs"] = args.epochs
    report["prediction_shards"] = [
        {"path": str(path), "sha256": _sha256(path.read_bytes())}
        for path in prediction_paths
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    _log("oof_complete", output=str(args.output), query_count=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
