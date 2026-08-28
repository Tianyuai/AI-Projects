"""Validate, shard, and resumably fit the full frozen F4/F5 training package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from paper_search.learning.f5_production_deployment import _jsonl_rows
from paper_search.learning.gated_feature_fusion_ranker import (
    FUSION_FAMILIES,
    FrozenFusionContextStore,
    GatedFeatureFusionRanker,
    load_gated_feature_fusion_ranker_bytes,
)
from paper_search.retrieval.pasa_paper_database import (
    ARXIV_MISSING_YEAR_EVIDENCE_POLICY,
    PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE,
    PASA_TRAINING_GOLD_INJECTED_SOURCE,
)
from paper_search.learning.large_scale_fusion_training import (
    FusionTrainingCheckpoint,
    FrozenFusionActivationInputs,
    apply_frozen_candidate_overlay,
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
    load_frozen_fusion_activation_inputs,
    query_has_gold_candidate,
    read_fusion_checkpoint,
    read_query_shard,
    unpack_production_bundle_components,
    with_context_label_files,
    write_fusion_checkpoint,
    write_query_shard,
)
from paper_search.learning.method_usage_evidence import (
    METHOD_USAGE_EVIDENCE_SCHEMA_VERSION,
)
from paper_search.learning.negation_evidence import NEGATION_EVIDENCE_SCHEMA_VERSION
from paper_search.learning.query_constraint_annotations import (
    FrozenConstraintAnnotation,
    FrozenConstraintProfileStore,
)
from paper_search.learning.task_slot_document_ranker import FrozenTaskSlotLabelStore
from paper_search.ranking.cpu_document import load_cpu_document_ranking_stage_bytes


DEFAULT_PAIR_BUDGET_BY_FAMILY = {
    "entity": 48,
    "hard_constraint": 128,
    "reliability": 96,
    "task_provenance": 112,
}
_PAIR_REPLAY_POLICY = "production-pair-replay-current-gold-no-injection-v1"


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _fit_input_sha256(
    package_input_sha256: str,
    *,
    pair_budget_by_family: Mapping[str, int] | None = None,
    max_pairs_per_query_family: int | None = None,
    activation_manifest_sha256: str | None = None,
    warm_start_weights_sha256: str | None = None,
    replay_manifest_sha256: str | None = None,
    replay_every_batches: int | None = None,
    method_usage_evidence_schema_version: str = (
        METHOD_USAGE_EVIDENCE_SCHEMA_VERSION
    ),
) -> str:
    if pair_budget_by_family is not None and max_pairs_per_query_family is not None:
        raise ValueError("uniform and family-specific pair budgets cannot be mixed")
    if pair_budget_by_family is None:
        if max_pairs_per_query_family is None or max_pairs_per_query_family <= 0:
            raise ValueError("a positive fusion pair budget is required")
        budgets = {
            family: max_pairs_per_query_family for family in sorted(FUSION_FAMILIES)
        }
    else:
        budgets = _validated_pair_budgets(pair_budget_by_family)
    replay_interval = replay_every_batches or 0
    if replay_interval < 0:
        raise ValueError("production replay interval cannot be negative")
    return _sha256(
        (
            f"{package_input_sha256}|"
            "fusion-feature-schema=v6-affirmative-method-usage|"
            f"method-usage-evidence={method_usage_evidence_schema_version}|"
            f"negation-evidence={NEGATION_EVIDENCE_SCHEMA_VERSION}|"
            f"publication-year-evidence={ARXIV_MISSING_YEAR_EVIDENCE_POLICY}|"
            f"activation-package={activation_manifest_sha256 or 'none'}|"
            "pair-budget-by-family="
            f"{json.dumps(budgets, sort_keys=True, separators=(',', ':'))}|"
            f"warm-start-weights={warm_start_weights_sha256 or 'none'}|"
            f"pair-replay-policy={_PAIR_REPLAY_POLICY}|"
            f"pair-replay-manifest={replay_manifest_sha256 or 'none'}|"
            f"pair-replay-every-batches={replay_interval}"
        ).encode()
    )


def _validated_pair_budgets(values: Mapping[str, int]) -> dict[str, int]:
    budgets = {str(family): int(value) for family, value in values.items()}
    if set(budgets) != set(FUSION_FAMILIES) or any(
        isinstance(value, bool) or value <= 0 for value in values.values()
    ):
        raise ValueError("pair budgets must positively cover every F4/F5 family")
    return dict(sorted(budgets.items()))


def _revalidated_production_replay_queries(
    queries: Sequence[Any], package: Any
) -> list[Any]:
    """Rebind old candidates to current exact labels and remove injected evidence."""

    output = []
    injection_sources = {
        PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE,
        PASA_TRAINING_GOLD_INJECTED_SOURCE,
    }
    for query in queries:
        row = package.rows_by_query_id.get(query.query_id)
        if row is None:
            continue
        current_query = str(row.get("query", ""))
        if current_query != query.query:
            raise ValueError(f"production replay query text mismatch: {query.query_id}")
        raw_gold = row.get("gold_paper_ids")
        if not isinstance(raw_gold, list) or not raw_gold or any(
            not isinstance(value, str) or not value for value in raw_gold
        ):
            raise ValueError(f"production replay Gold is invalid: {query.query_id}")
        candidates = [
            candidate
            for candidate in query.candidates
            if not injection_sources.intersection(candidate.paper.sources)
        ]
        rebound = query.model_copy(
            update={"gold_paper_ids": raw_gold, "candidates": candidates}
        )
        if len(candidates) >= 2 and query_has_gold_candidate(rebound):
            output.append(rebound)
    return output


def _load_production_replay_manifest(shard_dir: Path) -> tuple[dict[str, Any], str]:
    path = shard_dir / "manifest.json"
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in {
        "large-scale-fusion-query-shards-v1",
        "large-scale-fusion-query-shards-v2",
    }:
        raise ValueError("unsupported production replay shard manifest")
    if manifest.get("test_partition_touched") is not False:
        raise ValueError("production replay shards do not prove test isolation")
    batch_count = int(manifest.get("batch_count", 0))
    completed = manifest.get("completed_shards")
    if not isinstance(completed, list) or batch_count <= 0 or len(completed) != batch_count:
        raise ValueError("production replay shard manifest is incomplete")
    indexes = {int(row.get("batch_index", -1)) for row in completed if isinstance(row, dict)}
    if indexes != set(range(batch_count)):
        raise ValueError("production replay shard indexes are incomplete")
    return manifest, _sha256(payload)


def _read_production_replay_shard(
    *,
    shard_dir: Path,
    manifest: Mapping[str, Any],
    batch_index: int,
    package: Any,
) -> list[Any]:
    rows = manifest["completed_shards"]
    record = rows[batch_index]
    if int(record.get("batch_index", -1)) != batch_index:
        raise ValueError("production replay shard order mismatch")
    path = shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
    if record.get("sha256") != _sha256(path.read_bytes()):
        raise ValueError(f"production replay shard hash mismatch: {path}")
    return _revalidated_production_replay_queries(read_query_shard(path), package)


def _production_replay_batch_index(
    batch_index: int,
    *,
    training_batch_count: int,
    replay_batch_count: int,
    replay_every_batches: int,
) -> int | None:
    """Spread a deterministic replay subset, or every shard when interval is one."""

    if (
        batch_index < 0
        or batch_index >= training_batch_count
        or training_batch_count <= 0
        or replay_batch_count <= 0
        or replay_every_batches <= 0
    ):
        raise ValueError("production replay schedule is invalid")
    if replay_every_batches == 1:
        if replay_batch_count > training_batch_count:
            raise ValueError("full production replay requires no more replay batches")
        before = batch_index * replay_batch_count // training_batch_count
        after = (batch_index + 1) * replay_batch_count // training_batch_count
        return after - 1 if after > before else None
    if (batch_index + 1) % replay_every_batches != 0:
        return None
    slot_count = training_batch_count // replay_every_batches
    slot = (batch_index + 1) // replay_every_batches - 1
    return slot * replay_batch_count // slot_count


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _log(stage: str, **values: object) -> None:
    print(
        json.dumps(
            {"stage": stage, **values},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _prepare_shards(
    *, package: Any, shard_dir: Path, batch_size: int, workers: int = 1
) -> dict[str, Any]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_dir / "manifest.json"
    expected_batch_count = math.ceil(len(package.query_ids) / batch_size)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != "large-scale-fusion-query-shards-v2"
            or manifest.get("input_sha256") != package.input_sha256
            or manifest.get("batch_size") != batch_size
            or manifest.get("batch_count") != expected_batch_count
        ):
            raise ValueError("existing query shard manifest does not match inputs")
    else:
        manifest = {
            "schema_version": "large-scale-fusion-query-shards-v2",
            "input_sha256": package.input_sha256,
            "candidate_hydration_policy": package.candidate_hydration_policy,
            "query_count": len(package.query_ids),
            "batch_size": batch_size,
            "batch_count": expected_batch_count,
            "completed_shards": [],
            "candidate_count": 0,
            "gold_hit_query_count": 0,
            "test_partition_touched": False,
        }
        _atomic_json(manifest_path, manifest)

    indexed = index_training_receipts(package)
    completed = {
        int(row["batch_index"]): row for row in manifest["completed_shards"]
    }
    def build(query_id: str):  # type: ignore[no-untyped-def]
        return build_document_ranking_query(
            package,
            query_id,
            indexed[query_id],
            additive_receipt_roots=package.additive_receipt_roots,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch_index in range(expected_batch_count):
            start = batch_index * batch_size
            query_ids = package.query_ids[start : start + batch_size]
            shard_path = shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
            prior = completed.get(batch_index)
            if prior is not None and shard_path.is_file():
                payload = shard_path.read_bytes()
                if prior.get("sha256") == _sha256(payload):
                    continue
            queries = list(executor.map(build, query_ids))
            write_query_shard(shard_path, queries)
            candidate_count = sum(len(query.candidates) for query in queries)
            gold_hit_count = sum(query_has_gold_candidate(query) for query in queries)
            row = {
                "batch_index": batch_index,
                "query_count": len(queries),
                "candidate_count": candidate_count,
                "gold_hit_query_count": gold_hit_count,
                "sha256": _sha256(shard_path.read_bytes()),
            }
            completed[batch_index] = row
            manifest["completed_shards"] = [
                completed[index] for index in sorted(completed)
            ]
            manifest["candidate_count"] = sum(
                int(item["candidate_count"]) for item in completed.values()
            )
            manifest["gold_hit_query_count"] = sum(
                int(item["gold_hit_query_count"]) for item in completed.values()
            )
            _atomic_json(manifest_path, manifest)
            _log(
                "prepare_shards",
                completed=batch_index + 1,
                total=expected_batch_count,
                query_count=start + len(queries),
                candidate_count=manifest["candidate_count"],
                gold_hit_query_count=manifest["gold_hit_query_count"],
            )
    if len(completed) != expected_batch_count:
        raise ValueError("query shard preparation is incomplete")
    if int(manifest["gold_hit_query_count"]) <= 0:
        raise ValueError("query shards contain no scorer-aligned gold candidate")
    return dict(manifest)


def _weights_bytes(ranker: GatedFeatureFusionRanker) -> tuple[bytes, dict[str, str]]:
    records: list[bytes] = []
    hashes: dict[str, str] = {}
    for family in sorted(ranker.feature_families):
        name = family.encode("utf-8")
        vector = ranker.weights[family].astype("<f8", copy=False).tobytes(order="C")
        hashes[family] = _sha256(vector)
        records.append(struct.pack("<I", len(name)) + name + vector)
    return b"".join(records), hashes


def _write_final_artifacts(
    *,
    output_dir: Path,
    package: Any,
    ranker: GatedFeatureFusionRanker,
    pair_counts: dict[str, int],
    query_counts: dict[str, int],
    replay_pair_counts: dict[str, int],
    replay_query_counts: dict[str, int],
    activation_inputs: FrozenFusionActivationInputs | None,
    warm_start: Mapping[str, object],
    replay_manifest_sha256: str | None,
    replay_every_batches: int,
) -> None:
    weights, family_hashes = _weights_bytes(ranker)
    if ranker.pair_budget_by_family is None:
        raise ValueError("final F4/F5 artifact requires family-specific pair budgets")
    pair_budgets = ranker.pair_budget_by_family
    total_pair_counts = {
        family: pair_counts[family] + replay_pair_counts[family]
        for family in ranker.feature_families
    }
    total_query_counts = {
        family: query_counts[family] + replay_query_counts[family]
        for family in ranker.feature_families
    }
    manifest = {
        **ranker.manifest_fields(),
        "experiment_id": f"S4-F4-F5-{len(package.query_ids)}-frozen-full-fit",
        "training_query_count": len(package.query_ids),
        "training_input_sha256": _fit_input_sha256(
            package.input_sha256,
            pair_budget_by_family=pair_budgets,
            activation_manifest_sha256=(
                activation_inputs.manifest_sha256 if activation_inputs else None
            ),
            warm_start_weights_sha256=str(warm_start["weights_sha256"]),
            replay_manifest_sha256=replay_manifest_sha256,
            replay_every_batches=replay_every_batches,
        ),
        "training_package_sha256": package.input_sha256,
        "frozen_activation_package_sha256": (
            activation_inputs.manifest_sha256 if activation_inputs else None
        ),
        "candidate_shard_manifest_sha256": (
            activation_inputs.shard_manifest_sha256 if activation_inputs else None
        ),
        "preference_pair_count_by_family": dict(sorted(total_pair_counts.items())),
        "training_query_count_by_family": dict(sorted(total_query_counts.items())),
        "exact_preference_pair_count_by_family": dict(sorted(pair_counts.items())),
        "exact_training_query_count_by_family": dict(sorted(query_counts.items())),
        "production_replay_preference_pair_count_by_family": dict(
            sorted(replay_pair_counts.items())
        ),
        "production_replay_query_count_by_family": dict(
            sorted(replay_query_counts.items())
        ),
        "production_warm_start": dict(warm_start),
        "production_pair_replay": {
            "enabled": replay_manifest_sha256 is not None,
            "policy": _PAIR_REPLAY_POLICY,
            "source_manifest_sha256": replay_manifest_sha256,
            "every_training_batches": replay_every_batches,
            "current_exact_gold_rebound": True,
            "training_injected_candidates_removed": True,
        },
        "family_weight_sha256": family_hashes,
        "weights_sha256": _sha256(weights),
        "candidate_pool_identity_unchanged": True,
        "candidate_source_search_repeated": False,
        "title_diagnostic_only": False,
        "abstract_route_isolated": False,
        "online_requests_made": 0,
        "development_labels_used_for_training": False,
        "test_partition_touched": False,
        "production_promotion_authorized": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_tmp = output_dir / "weights.bin.tmp"
    weights_tmp.write_bytes(weights)
    weights_tmp.replace(output_dir / "weights.bin")
    _atomic_json(output_dir / "manifest.json", manifest)

    reliability = ranker.weights["reliability"].astype("<f8", copy=False)
    name = b"reliability"
    f4_weights = struct.pack("<I", len(name)) + name + reliability.tobytes(order="C")
    f4_manifest = dict(manifest)
    f4_manifest.update(
        {
            "experiment_id": f"S4-F4-reliability-{len(package.query_ids)}-frozen-full-fit",
            "feature_families": ["reliability"],
            "family_caps": {"reliability": ranker.family_caps["reliability"]},
            "pair_budget_by_family": {
                "reliability": pair_budgets["reliability"]
            },
            "family_weight_sha256": {"reliability": _sha256(reliability.tobytes(order="C"))},
            "weights_sha256": _sha256(f4_weights),
            "preference_pair_count_by_family": {
                "reliability": total_pair_counts["reliability"]
            },
            "training_query_count_by_family": {
                "reliability": total_query_counts["reliability"]
            },
        }
    )
    f4_dir = output_dir.parent / (output_dir.name + "-f4")
    f4_dir.mkdir(parents=True, exist_ok=True)
    f4_tmp = f4_dir / "weights.bin.tmp"
    f4_tmp.write_bytes(f4_weights)
    f4_tmp.replace(f4_dir / "weights.bin")
    _atomic_json(f4_dir / "manifest.json", f4_manifest)


def _fit(
    *,
    package: Any,
    production_manifest_path: Path,
    shard_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    epochs: int,
    pair_budget_by_family: Mapping[str, int],
    activation_inputs: FrozenFusionActivationInputs | None = None,
    production_replay_shard_dir: Path | None = None,
    replay_every_batches: int = 1,
) -> None:
    production_manifest = json.loads(
        production_manifest_path.read_text(encoding="utf-8")
    )
    components = unpack_production_bundle_components(
        package.production_bundle_path.read_bytes()
    )
    baseline_manifest = production_manifest["baseline_manifest"]
    baseline_stage = load_cpu_document_ranking_stage_bytes(
        (json.dumps(baseline_manifest, sort_keys=True) + "\n").encode("utf-8"),
        components["baseline_weights"],
    )
    task_store = FrozenTaskSlotLabelStore.from_jsonl_bytes(
        package.task_labels_bytes
    )
    constraint_rows = _jsonl_rows(
        package.constraint_labels_bytes, label="constraint labels"
    )
    constraint_store = FrozenConstraintProfileStore(
        [FrozenConstraintAnnotation.model_validate(row) for row in constraint_rows]
    )
    prior_config = production_manifest["f5_manifest"]
    context_store = FrozenFusionContextStore(
        task_store=task_store,
        constraint_store=constraint_store,
    )
    budgets = _validated_pair_budgets(pair_budget_by_family)
    ranker = GatedFeatureFusionRanker(
        baseline_ranker=baseline_stage.ranker,
        context_store=context_store,
        feature_families=set(FUSION_FAMILIES),
        gated=True,
        dimension=int(prior_config["dimension_per_family"]),
        epochs=1,
        family_caps={
            str(name): float(value)
            for name, value in prior_config["family_caps"].items()
        },
        maximum_total_residual=float(prior_config["maximum_total_residual"]),
        constraint_text_evidence=True,
        runtime_context_scoring=True,
        pair_budget_by_family=budgets,
        publication_year_evidence_policy=ARXIV_MISSING_YEAR_EVIDENCE_POLICY,
    )
    production_weights = components["fusion_weights"]
    loaded_production = load_gated_feature_fusion_ranker_bytes(
        (json.dumps(prior_config, sort_keys=True) + "\n").encode("utf-8"),
        production_weights,
        baseline_ranker=baseline_stage.ranker,
        context_store=context_store,
    )
    if (
        loaded_production.feature_families != ranker.feature_families
        or loaded_production.dimension != ranker.dimension
        or loaded_production.family_caps != ranker.family_caps
        or loaded_production.maximum_total_residual
        != ranker.maximum_total_residual
    ):
        raise ValueError("production F5 is incompatible with the fine-tuning ranker")
    ranker.weights = {
        family: weights.copy()
        for family, weights in loaded_production.weights.items()
    }
    warm_start = {
        "strategy": "production-f5-exact-weight-initialization-v1",
        "model_id": loaded_production.model_id,
        "weights_sha256": _sha256(production_weights),
        "family_weight_sha256": dict(prior_config["family_weight_sha256"]),
        "training_query_count": int(prior_config.get("training_query_count", 0)),
    }
    shard_manifest = json.loads(
        (shard_dir / "manifest.json").read_text(encoding="utf-8")
    )
    batch_count = int(shard_manifest["batch_count"])
    replay_manifest: dict[str, Any] | None = None
    replay_manifest_sha256: str | None = None
    replay_batch_count = 0
    if production_replay_shard_dir is not None:
        replay_manifest, replay_manifest_sha256 = _load_production_replay_manifest(
            production_replay_shard_dir
        )
        replay_batch_count = int(replay_manifest["batch_count"])
        if replay_every_batches <= 0 or replay_every_batches > batch_count:
            raise ValueError("production replay interval is invalid")
    else:
        replay_every_batches = 0
    pair_counts = {family: 0 for family in ranker.feature_families}
    query_counts = {family: 0 for family in ranker.feature_families}
    replay_pair_counts = {family: 0 for family in ranker.feature_families}
    replay_query_counts = {family: 0 for family in ranker.feature_families}
    epoch_index = next_batch_index = 0
    fit_input_sha256 = _fit_input_sha256(
        package.input_sha256,
        pair_budget_by_family=budgets,
        activation_manifest_sha256=(
            activation_inputs.manifest_sha256 if activation_inputs else None
        ),
        warm_start_weights_sha256=str(warm_start["weights_sha256"]),
        replay_manifest_sha256=replay_manifest_sha256,
        replay_every_batches=replay_every_batches,
    )
    if (checkpoint_dir / "checkpoint.json").is_file():
        checkpoint = read_fusion_checkpoint(checkpoint_dir)
        if (
            checkpoint.input_sha256 != fit_input_sha256
            or checkpoint.batch_count != batch_count
            or set(checkpoint.weights) != set(ranker.feature_families)
        ):
            raise ValueError("existing fit checkpoint does not match inputs")
        ranker.weights = {
            family: weights.copy() for family, weights in checkpoint.weights.items()
        }
        epoch_index = checkpoint.epoch_index
        next_batch_index = checkpoint.next_batch_index
        pair_counts = dict(checkpoint.pair_counts)
        query_counts = dict(checkpoint.query_counts)
        replay_pair_counts = dict(checkpoint.replay_pair_counts)
        replay_query_counts = dict(checkpoint.replay_query_counts)
        _log(
            "resume_fit",
            epoch_index=epoch_index,
            next_batch_index=next_batch_index,
            batch_count=batch_count,
        )

    started = time.monotonic()
    for epoch in range(epoch_index, epochs):
        first_batch = next_batch_index if epoch == epoch_index else 0
        for batch_index in range(first_batch, batch_count):
            queries = read_query_shard(
                shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
            )
            if activation_inputs is not None:
                queries = [
                    apply_frozen_candidate_overlay(
                        query, activation_inputs.overlay_by_query_id[query.query_id]
                    )
                    if query.query_id in activation_inputs.overlay_by_query_id
                    else query
                    for query in queries
                ]
            batch_pairs = ranker.fit(queries)
            if epoch == 0:
                for family in ranker.feature_families:
                    pair_counts[family] += int(batch_pairs[family])
                    query_counts[family] += int(ranker.last_fit_query_count[family])
            replay_batch_index = (
                _production_replay_batch_index(
                    batch_index,
                    training_batch_count=batch_count,
                    replay_batch_count=replay_batch_count,
                    replay_every_batches=replay_every_batches,
                )
                if replay_manifest is not None
                and production_replay_shard_dir is not None
                else None
            )
            if (
                replay_batch_index is not None
                and replay_manifest is not None
                and production_replay_shard_dir is not None
            ):
                replay_queries = _read_production_replay_shard(
                    shard_dir=production_replay_shard_dir,
                    manifest=replay_manifest,
                    batch_index=replay_batch_index,
                    package=package,
                )
                if replay_queries:
                    replay_pairs = ranker.fit(replay_queries)
                    if epoch == 0:
                        for family in ranker.feature_families:
                            replay_pair_counts[family] += int(replay_pairs[family])
                            replay_query_counts[family] += int(
                                ranker.last_fit_query_count[family]
                            )
            if batch_index + 1 == batch_count:
                saved_epoch = epoch + 1
                saved_batch = 0
            else:
                saved_epoch = epoch
                saved_batch = batch_index + 1
            write_fusion_checkpoint(
                checkpoint_dir,
                FusionTrainingCheckpoint(
                    input_sha256=fit_input_sha256,
                    epoch_index=saved_epoch,
                    next_batch_index=saved_batch,
                    batch_count=batch_count,
                    pair_counts=pair_counts,
                    query_counts=query_counts,
                    replay_pair_counts=replay_pair_counts,
                    replay_query_counts=replay_query_counts,
                    weights={
                        family: weights.copy()
                        for family, weights in ranker.weights.items()
                    },
                ),
            )
            _log(
                "fit",
                epoch=epoch + 1,
                epochs=epochs,
                completed_batch=batch_index + 1,
                batch_count=batch_count,
                elapsed_seconds=round(time.monotonic() - started, 1),
                preference_pair_count_by_family=pair_counts,
                production_replay_pair_count_by_family=replay_pair_counts,
            )
        next_batch_index = 0
    _write_final_artifacts(
        output_dir=output_dir,
        package=package,
        ranker=ranker,
        pair_counts=pair_counts,
        query_counts=query_counts,
        replay_pair_counts=replay_pair_counts,
        replay_query_counts=replay_query_counts,
        activation_inputs=activation_inputs,
        warm_start=warm_start,
        replay_manifest_sha256=replay_manifest_sha256,
        replay_every_batches=replay_every_batches,
    )
    _log("fit_complete", output_dir=str(output_dir), training_query_count=len(package.query_ids))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--production-manifest", type=Path, required=True)
    parser.add_argument("--production-bundle", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-pairs-per-query-family", type=int)
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
    parser.add_argument("--prepare-workers", type=int, default=8)
    parser.add_argument("--task-labels", type=Path)
    parser.add_argument("--constraint-labels", type=Path)
    parser.add_argument("--context-freeze-manifest", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.batch_size <= 0
        or args.epochs <= 0
        or args.prepare_workers <= 0
    ):
        raise ValueError("batch size and epochs must be positive")
    if args.max_pairs_per_query_family is not None:
        if args.max_pairs_per_query_family <= 0:
            raise ValueError("uniform pair budget must be positive")
        pair_budget_by_family = {
            family: args.max_pairs_per_query_family
            for family in FUSION_FAMILIES
        }
    else:
        pair_budget_by_family = _validated_pair_budgets(
            {
                "entity": args.entity_pair_budget,
                "hard_constraint": args.hard_constraint_pair_budget,
                "reliability": args.reliability_pair_budget,
                "task_provenance": args.task_provenance_pair_budget,
            }
        )
    if (
        args.production_replay_shard_dir is not None
        and args.production_replay_every_batches <= 0
    ):
        raise ValueError("production replay interval must be positive")

    package = load_training_package(
        handoff_path=args.handoff,
        partition_path=args.partition,
        production_bundle_path=args.production_bundle,
    )
    if args.context_freeze_manifest is not None and (
        args.task_labels is not None or args.constraint_labels is not None
    ):
        raise ValueError("context freeze manifest cannot be mixed with label overrides")
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
    _log(
        "package_validated",
        query_count=len(package.query_ids),
        task_label_count=package.task_label_count,
        constraint_label_count=package.constraint_label_count,
        context_label_override=(
            args.task_labels is not None or activation_inputs is not None
        ),
        conflicting_ready_query_count=len(package.conflicting_query_ids),
        online_llm_requests=0,
        test_partition_touched=False,
    )
    shard_dir = (
        activation_inputs.shard_dir
        if activation_inputs is not None
        else args.run_dir / "query-shards"
    )
    if activation_inputs is None:
        _prepare_shards(
            package=package,
            shard_dir=shard_dir,
            batch_size=args.batch_size,
            workers=args.prepare_workers,
        )
    if args.prepare_only:
        _log(
            "prepare_complete",
            shard_dir=str(shard_dir),
            query_count=len(package.query_ids),
        )
        return 0
    _fit(
        package=package,
        production_manifest_path=args.production_manifest,
        shard_dir=shard_dir,
        checkpoint_dir=args.run_dir / "fit-checkpoint",
        output_dir=args.output_dir,
        epochs=args.epochs,
        pair_budget_by_family=pair_budget_by_family,
        activation_inputs=activation_inputs,
        production_replay_shard_dir=args.production_replay_shard_dir,
        replay_every_batches=args.production_replay_every_batches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
