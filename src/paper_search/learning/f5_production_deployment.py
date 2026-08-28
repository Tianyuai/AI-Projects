"""Content-addressed production packaging for the promoted F5 ranker."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from typing import cast

from paper_search.learning.gated_feature_fusion_ranker import (
    GatedFeatureFusionRanker,
    UnifiedFusionContextResolver,
    load_gated_feature_fusion_ranker_bytes,
)
from paper_search.learning.query_constraint_annotations import (
    FrozenConstraintAnnotation,
    FrozenConstraintProfileStore,
)
from paper_search.learning.task_slot_document_ranker import FrozenTaskSlotLabelStore


_BUNDLE_MAGIC = b"F5PROD1\0"
_COMPONENT_NAMES = (
    "baseline_weights",
    "f5_weights",
    "task_labels",
    "constraint_labels",
)
_HEADER = struct.Struct("<8sQQQQ")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _object(payload: bytes, *, label: str) -> dict[str, object]:
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], raw)


def _jsonl_rows(payload: bytes, *, label: str) -> list[dict[str, object]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 JSONL") from error
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{label} row {line_number} must be an object")
        rows.append(cast(dict[str, object], raw))
    if not rows:
        raise ValueError(f"{label} cannot be empty")
    return rows


def _validate_training_rows(rows: list[dict[str, object]], *, label: str) -> None:
    if any(
        row.get("role") != "training" or row.get("split") != "auto_train"
        for row in rows
    ):
        raise ValueError(f"{label} must contain only auto_train rows")


def build_f5_production_bundle(
    *,
    baseline_manifest_bytes: bytes,
    baseline_weights_bytes: bytes,
    f5_manifest_bytes: bytes,
    f5_weights_bytes: bytes,
    task_labels_bytes: bytes,
    constraint_labels_bytes: bytes,
    promotion_evidence_bytes: bytes,
    large_oof_report_bytes: bytes,
) -> tuple[bytes, bytes]:
    """Freeze F5 and all inference context behind the existing two-file lock."""

    baseline_manifest = _object(baseline_manifest_bytes, label="baseline manifest")
    f5_manifest = _object(f5_manifest_bytes, label="F5 manifest")
    evidence = _object(promotion_evidence_bytes, label="promotion evidence")
    large_oof = _object(large_oof_report_bytes, label="large OOF report")
    decision = evidence.get("promotion_decision")
    replay = evidence.get("live_replay_consistency")
    if not isinstance(decision, Mapping) or not isinstance(replay, Mapping):
        raise ValueError("promotion evidence is missing gate results")
    if decision.get("passed") is not True or decision.get(
        "authorizes_production_default_switch"
    ) is not True:
        raise ValueError("promotion evidence does not authorize F5 production")
    if replay.get("all_identical") is not True:
        raise ValueError("promotion evidence failed live/replay consistency")
    if evidence.get("test_partition_touched") is not False:
        raise ValueError("promotion evidence touched the test partition")
    training_query_count = f5_manifest.get("training_query_count")
    if training_query_count != evidence.get("training_query_count"):
        raise ValueError("F5 training count disagrees with promotion evidence")
    if f5_manifest.get("test_partition_touched") is not False:
        raise ValueError("F5 training touched the test partition")
    if large_oof.get("training_query_count") != training_query_count:
        raise ValueError("large OOF count disagrees with the F5 manifest")
    if large_oof.get("test_partition_touched") is not False:
        raise ValueError("large OOF evaluation touched the test partition")
    experiments = large_oof.get("experiments")
    if not isinstance(experiments, Mapping):
        raise ValueError("large OOF report is missing experiments")
    f5_experiment = experiments.get(f"S4-F5-gated-fusion-{training_query_count}")
    if not isinstance(f5_experiment, Mapping):
        raise ValueError("large OOF report is missing the F5 experiment")
    overall_deltas: dict[str, dict[str, float]] = {}
    for baseline_name in ("metrics_vs_b0", "metrics_vs_f4"):
        comparison = f5_experiment.get(baseline_name)
        if not isinstance(comparison, Mapping) or not isinstance(
            comparison.get("overall"), Mapping
        ):
            raise ValueError(f"large OOF report is missing {baseline_name}")
        metrics = cast(Mapping[str, object], comparison["overall"])
        deltas = {
            metric: float(cast(Mapping[str, object], values)["delta"])
            for metric, values in metrics.items()
            if metric != "query_count" and isinstance(values, Mapping)
        }
        if not deltas or any(delta <= 0.0 for delta in deltas.values()):
            raise ValueError(f"F5 large OOF does not improve every {baseline_name} metric")
        overall_deltas[baseline_name] = deltas

    task_rows = _jsonl_rows(task_labels_bytes, label="task labels")
    constraint_rows = _jsonl_rows(constraint_labels_bytes, label="constraint labels")
    _validate_training_rows(task_rows, label="task labels")
    _validate_training_rows(constraint_rows, label="constraint labels")
    FrozenTaskSlotLabelStore.from_jsonl_bytes(task_labels_bytes)
    for row in constraint_rows:
        FrozenConstraintAnnotation.model_validate(row)

    components = (
        baseline_weights_bytes,
        f5_weights_bytes,
        task_labels_bytes,
        constraint_labels_bytes,
    )
    weights_bytes = _HEADER.pack(
        _BUNDLE_MAGIC, *(len(component) for component in components)
    ) + b"".join(components)
    manifest = {
        "schema_version": "gated-feature-fusion-production-manifest-v1",
        "model_id": GatedFeatureFusionRanker.model_id,
        "production_default": "F5-gated-fusion",
        "production_fallback": "F4-reliability",
        "emergency_fallback": "B0",
        "training_query_count": training_query_count,
        "test_partition_touched": False,
        "model_sha256": _sha256(weights_bytes),
        "baseline_manifest": baseline_manifest,
        "f5_manifest": f5_manifest,
        "components": {
            name: {"length": len(payload), "sha256": _sha256(payload)}
            for name, payload in zip(_COMPONENT_NAMES, components, strict=True)
        },
        "promotion": {
            "evidence_sha256": _sha256(promotion_evidence_bytes),
            "large_oof_report_sha256": _sha256(large_oof_report_bytes),
            "policy_id": decision.get("promotion_policy_id"),
            "passed": True,
            "auto_dev_query_count": replay.get("query_count"),
            "live_replay_all_identical": True,
            "large_oof_query_count": training_query_count,
            "large_oof_all_overall_deltas_positive": True,
            "large_oof_overall_deltas": overall_deltas,
        },
    }
    return (
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
        weights_bytes,
    )


def _extract_reliability_artifact(
    f5_manifest_bytes: bytes, f5_weights_bytes: bytes
) -> tuple[dict[str, object], bytes]:
    manifest = _object(f5_manifest_bytes, label="F5 manifest")
    training_query_count = manifest.get("training_query_count")
    if (
        isinstance(training_query_count, bool)
        or not isinstance(training_query_count, int)
        or training_query_count <= 0
    ):
        raise ValueError("F5 manifest has an invalid training query count")
    dimension = manifest.get("dimension_per_family")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("F5 manifest has an invalid family dimension")
    vector_size = dimension * 8
    offset = 0
    reliability_record: bytes | None = None
    reliability_vector: bytes | None = None
    while offset < len(f5_weights_bytes):
        start = offset
        if len(f5_weights_bytes) - offset < 4:
            raise ValueError("truncated F5 family weights")
        name_size = struct.unpack_from("<I", f5_weights_bytes, offset)[0]
        offset += 4
        if name_size <= 0 or len(f5_weights_bytes) - offset < name_size + vector_size:
            raise ValueError("truncated F5 family weight record")
        family = f5_weights_bytes[offset : offset + name_size].decode("utf-8")
        offset += name_size
        vector = f5_weights_bytes[offset : offset + vector_size]
        offset += vector_size
        if family == "reliability":
            reliability_record = f5_weights_bytes[start:offset]
            reliability_vector = vector
    if reliability_record is None or reliability_vector is None:
        raise ValueError("F5 weights are missing the reliability family")

    derived = dict(manifest)
    derived.update(
        {
            "experiment_id": (
                f"S4-F4-reliability-{training_query_count}-production-fallback"
            ),
            "feature_families": ["reliability"],
            "family_caps": {
                "reliability": cast(Mapping[str, object], manifest["family_caps"])[
                    "reliability"
                ]
            },
            "family_weight_sha256": {"reliability": _sha256(reliability_vector)},
            "weights_sha256": _sha256(reliability_record),
            "training_query_count_by_family": {
                "reliability": cast(
                    Mapping[str, object], manifest["training_query_count_by_family"]
                )["reliability"]
            },
            "preference_pair_count_by_family": {
                "reliability": cast(
                    Mapping[str, object], manifest["preference_pair_count_by_family"]
                )["reliability"]
            },
        }
    )
    return derived, reliability_record


def build_f4_production_bundle(
    *,
    baseline_manifest_bytes: bytes,
    baseline_weights_bytes: bytes,
    f5_manifest_bytes: bytes,
    f5_weights_bytes: bytes,
    large_oof_report_bytes: bytes,
) -> tuple[bytes, bytes]:
    """Derive and freeze the independently deployable F4 reliability fallback."""

    baseline_manifest = _object(baseline_manifest_bytes, label="baseline manifest")
    f4_manifest, f4_weights = _extract_reliability_artifact(
        f5_manifest_bytes, f5_weights_bytes
    )
    large_oof = _object(large_oof_report_bytes, label="large OOF report")
    training_query_count = f4_manifest.get("training_query_count")
    experiments = large_oof.get("experiments")
    if (
        large_oof.get("training_query_count") != training_query_count
        or large_oof.get("test_partition_touched") is not False
        or not isinstance(experiments, Mapping)
    ):
        raise ValueError("invalid large OOF evidence for F4")
    experiment = experiments.get(f"S4-F4-reliability-{training_query_count}")
    if not isinstance(experiment, Mapping):
        raise ValueError("large OOF report is missing the F4 experiment")
    comparison = experiment.get("metrics_vs_b0")
    if not isinstance(comparison, Mapping) or not isinstance(
        comparison.get("overall"), Mapping
    ):
        raise ValueError("large OOF report is missing F4 vs B0 metrics")
    metrics = cast(Mapping[str, object], comparison["overall"])
    deltas = {
        metric: float(cast(Mapping[str, object], values)["delta"])
        for metric, values in metrics.items()
        if metric != "query_count" and isinstance(values, Mapping)
    }
    if not deltas or any(delta <= 0.0 for delta in deltas.values()):
        raise ValueError("F4 large OOF does not improve every B0 overall metric")

    components = (baseline_weights_bytes, f4_weights, b"", b"")
    weights_bytes = _HEADER.pack(
        _BUNDLE_MAGIC, *(len(component) for component in components)
    ) + b"".join(components)
    manifest = {
        "schema_version": "gated-feature-fusion-production-manifest-v1",
        "model_id": GatedFeatureFusionRanker.model_id,
        "method": "F4-reliability",
        "production_role": "performance_fallback",
        "production_default": "F5-gated-fusion",
        "production_fallback": "F4-reliability",
        "emergency_fallback": "B0",
        "feature_families": ["reliability"],
        "training_query_count": training_query_count,
        "test_partition_touched": False,
        "baseline_manifest": baseline_manifest,
        "f5_manifest": f4_manifest,
        "components": {
            name: {"length": len(payload), "sha256": _sha256(payload)}
            for name, payload in zip(_COMPONENT_NAMES, components, strict=True)
        },
        "promotion": {
            "large_oof_report_sha256": _sha256(large_oof_report_bytes),
            "large_oof_query_count": training_query_count,
            "large_oof_all_overall_deltas_positive_vs_b0": True,
            "large_oof_overall_deltas_vs_b0": deltas,
        },
    }
    manifest["model_sha256"] = _sha256(weights_bytes)
    return (
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
        weights_bytes,
    )


def _unpack_components(
    manifest: Mapping[str, object], weights_bytes: bytes
) -> dict[str, bytes]:
    if len(weights_bytes) < _HEADER.size:
        raise ValueError("truncated F5 production bundle")
    magic, *lengths = _HEADER.unpack_from(weights_bytes)
    if magic != _BUNDLE_MAGIC:
        raise ValueError("invalid F5 production bundle magic")
    if _HEADER.size + sum(lengths) != len(weights_bytes):
        raise ValueError("F5 production bundle length mismatch")
    offset = _HEADER.size
    decoded: dict[str, bytes] = {}
    raw_components = manifest.get("components")
    if not isinstance(raw_components, Mapping):
        raise ValueError("F5 production manifest is missing components")
    for name, length in zip(_COMPONENT_NAMES, lengths, strict=True):
        payload = weights_bytes[offset : offset + length]
        offset += length
        metadata = raw_components.get(name)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"F5 production manifest is missing {name}")
        if metadata.get("length") != length or metadata.get("sha256") != _sha256(
            payload
        ):
            raise ValueError(f"F5 production component mismatch: {name}")
        decoded[name] = payload
    return decoded


def load_f5_production_ranker_bytes(
    manifest_bytes: bytes, weights_bytes: bytes
) -> GatedFeatureFusionRanker:
    """Restore the promoted F5 ranker from immutable deployment bytes."""

    manifest = _object(manifest_bytes, label="F5 production manifest")
    if manifest.get("schema_version") != "gated-feature-fusion-production-manifest-v1":
        raise ValueError("unsupported F5 production manifest schema")
    if manifest.get("model_sha256") != _sha256(weights_bytes):
        raise ValueError("F5 production model hash mismatch")
    if manifest.get("test_partition_touched") is not False:
        raise ValueError("F5 production manifest touched the test partition")
    components = _unpack_components(manifest, weights_bytes)

    from paper_search.ranking.cpu_document import (  # avoid import cycle
        load_cpu_document_ranking_stage_bytes,
    )

    baseline_manifest = manifest.get("baseline_manifest")
    f5_manifest = manifest.get("f5_manifest")
    baseline_stage = load_cpu_document_ranking_stage_bytes(
        (json.dumps(baseline_manifest, sort_keys=True) + "\n").encode("utf-8"),
        components["baseline_weights"],
    )
    task_store = FrozenTaskSlotLabelStore.from_jsonl_bytes(components["task_labels"])
    constraint_rows = (
        _jsonl_rows(components["constraint_labels"], label="constraint labels")
        if components["constraint_labels"]
        else []
    )
    constraint_store = FrozenConstraintProfileStore(
        [FrozenConstraintAnnotation.model_validate(row) for row in constraint_rows]
    )
    task_terms = [
        str(task.get("normalized_value", ""))
        for row in _jsonl_rows(components["task_labels"], label="task labels")
        if isinstance(row.get("tasks"), list)
        for task in cast(list[object], row["tasks"])
        if isinstance(task, Mapping)
    ] if components["task_labels"] else []
    method_terms = [
        str(value)
        for row in constraint_rows
        for value in cast(list[object], row.get("methods", []))
    ]
    dataset_terms = [
        str(value)
        for row in constraint_rows
        for value in cast(list[object], row.get("datasets", []))
    ]
    context_store = UnifiedFusionContextResolver(
        task_store=task_store,
        constraint_store=constraint_store,
        task_terms=task_terms,
        method_terms=method_terms,
        dataset_terms=dataset_terms,
    )
    return load_gated_feature_fusion_ranker_bytes(
        (json.dumps(f5_manifest, sort_keys=True) + "\n").encode("utf-8"),
        components["f5_weights"],
        baseline_ranker=baseline_stage.ranker,
        context_store=context_store,
    )


__all__ = [
    "build_f4_production_bundle",
    "build_f5_production_bundle",
    "load_f5_production_ranker_bytes",
]
