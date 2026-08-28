"""Freeze zero-request local contexts only when they yield effective fusion pairs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.learning.f5_production_deployment import (  # noqa: E402
    _jsonl_rows,
    load_f5_production_ranker_bytes,
)
from paper_search.learning.fusion_activation import (  # noqa: E402
    audit_fusion_query_activation,
)
from paper_search.learning.gated_feature_fusion_ranker import (  # noqa: E402
    FUSION_FAMILIES,
    FrozenFusionContextStore,
    FusionQueryContext,
    TASK_PROVENANCE_ALLOWED_STATUSES,
    UnifiedFusionContextResolver,
)
from paper_search.learning.large_scale_fusion_training import (  # noqa: E402
    FrozenCandidateOverlayEntry,
    apply_frozen_candidate_overlay,
    unpack_production_bundle_components,
)
from paper_search.learning.cpu_document_ranker import (  # noqa: E402
    DocumentCandidateEvidence,
    DocumentRankingQuery,
)
from paper_search.learning.negation_evidence import (  # noqa: E402
    NEGATION_EVIDENCE_SCHEMA_VERSION,
)
from paper_search.learning.query_constraint_annotations import (  # noqa: E402
    FrozenConstraintAnnotation,
    FrozenConstraintProfileStore,
)
from paper_search.learning.task_slot_document_ranker import (  # noqa: E402
    FrozenTaskSlotLabelStore,
)
from paper_search.retrieval.pasa_paper_database import (  # noqa: E402
    ARXIV_MISSING_YEAR_EVIDENCE_POLICY,
)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite immutable freeze output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _effective(audit: Mapping[str, object], family: str) -> bool:
    families = cast(Mapping[str, Mapping[str, object]], audit["families"])
    values = families.get(family)
    return bool(values and int(values["effective_pair_count"]) > 0)


_QUERY_HEADER_RE = re.compile(
    rb'^\{"query_id":("(?:\\.|[^"\\])*"),'
    rb'"query":("(?:\\.|[^"\\])*")'
)


def _query_header(raw_line: bytes) -> tuple[str, str]:
    match = _QUERY_HEADER_RE.match(raw_line)
    if match is None:
        raise ValueError("query shard row lacks the stable id/query prefix")
    values = json.loads(b"[" + match.group(1) + b"," + match.group(2) + b"]")
    if (
        not isinstance(values, list)
        or len(values) != 2
        or not all(isinstance(value, str) for value in values)
    ):
        raise ValueError("query shard row has invalid id/query values")
    return values[0], values[1]


def _merge_task_row(
    row: Mapping[str, object], *, tasks: Sequence[str]
) -> dict[str, object]:
    output = dict(row)
    tasks_value = cast(Sequence[Mapping[str, object]], output.get("tasks", []))
    ambiguous = {
        str(value).casefold() for value in cast(Sequence[object], output.get("ambiguous_fields", []))
    }
    if (
        tasks_value
        and output.get("task_label_status") in TASK_PROVENANCE_ALLOWED_STATUSES
        and not {"task", "tasks"}.intersection(ambiguous)
        and any(float(value.get("confidence", 0.0)) > 0.0 for value in tasks_value)
    ):
        return output
    output.update(
        {
            "tasks": [
                {
                    "normalized_value": task,
                    "confidence": 0.9,
                    "evidence_span": task,
                    "strength": "must",
                }
                for task in tasks
            ],
            "ambiguous_fields": [],
            "annotator_id": UnifiedFusionContextResolver.resolver_id,
            "task_label_status": "runtime_deterministic",
        }
    )
    return output


def _merge_constraint_row(
    row: Mapping[str, object],
    *,
    profile: Any,
    effective_signals: set[str],
) -> dict[str, object]:
    output = dict(row)
    labels = set(cast(list[str], output.get("labels", [])))
    sources = dict(cast(Mapping[str, str], output.get("label_sources", {})))
    confidence = dict(cast(Mapping[str, float], output.get("label_confidence", {})))
    evidence = dict(cast(Mapping[str, list[str]], output.get("evidence", {})))

    def usable(label: str, field: str) -> bool:
        return (
            label in labels
            and bool(output.get(field))
            and float(confidence.get(label, 0.0)) >= 0.8
        )

    for label, field in (("method", "methods"), ("dataset", "datasets")):
        if label in effective_signals:
            inferred = list(getattr(profile, field))
            if inferred and not usable(label, field):
                output[field] = inferred
                labels.add(label)
                sources[label] = "local_deterministic"
                confidence[label] = 0.9
                evidence[label] = inferred
    if "year" in effective_signals:
        if "year" in profile.labels and not (
            "year" in labels
            and float(confidence.get("year", 0.0)) >= 0.8
            and (
                output.get("year_from") is not None
                or output.get("year_to") is not None
            )
        ):
            output["year_from"] = profile.year_from
            output["year_to"] = profile.year_to
            labels.add("year")
            sources["year"] = "local_deterministic"
            confidence["year"] = 1.0
            evidence["year"] = [
                str(value)
                for value in (profile.year_from, profile.year_to)
                if value is not None
            ]
    if "negation" in effective_signals:
        inferred_exclusions = list(profile.exclusions)
        if (
            "negation" in profile.labels
            and inferred_exclusions
            and not usable("negation", "exclusions")
        ):
            output["exclusions"] = inferred_exclusions
            labels.add("negation")
            sources["negation"] = "local_deterministic"
            confidence["negation"] = 1.0
            evidence["negation"] = inferred_exclusions
    def structurally_usable(label: str) -> bool:
        if float(confidence.get(label, 0.0)) < 0.8:
            return False
        if label == "method":
            return bool(output.get("methods"))
        if label == "dataset":
            return bool(output.get("datasets"))
        if label == "task":
            return bool(output.get("tasks"))
        if label == "year":
            return output.get("year_from") is not None or output.get("year_to") is not None
        if label == "negation":
            return bool(output.get("exclusions"))
        return True

    labels = {label for label in labels if structurally_usable(label)}
    for label, field in (
        ("method", "methods"),
        ("dataset", "datasets"),
        ("task", "tasks"),
        ("negation", "exclusions"),
    ):
        if label not in labels:
            output[field] = []
    if "year" not in labels:
        output["year_from"] = None
        output["year_to"] = None
    output["labels"] = sorted(labels)
    output["label_sources"] = {
        label: value for label, value in sources.items() if label in labels
    }
    output["label_confidence"] = {
        label: value for label, value in confidence.items() if label in labels
    }
    output["evidence"] = {
        label: value for label, value in evidence.items() if label in labels
    }
    if labels:
        output["status"] = "accepted"
    else:
        output["status"] = "partial"
    return output


def _replace_deterministic_constraint_signals(
    row: Mapping[str, object], *, profile: Any, signals: set[str]
) -> dict[str, object]:
    """Make rule-derived year/negation fields canonical to the current parser."""

    unsupported = signals.difference({"year", "negation"})
    if unsupported:
        raise ValueError(f"unsupported deterministic signals: {sorted(unsupported)}")
    output = dict(row)
    labels = set(cast(Sequence[str], output.get("labels", [])))
    sources = dict(cast(Mapping[str, str], output.get("label_sources", {})))
    confidence = dict(cast(Mapping[str, float], output.get("label_confidence", {})))
    evidence = dict(cast(Mapping[str, list[str]], output.get("evidence", {})))
    if "year" in signals:
        has_year = bool(
            profile is not None
            and "year" in profile.labels
            and (profile.year_from is not None or profile.year_to is not None)
        )
        output["year_from"] = profile.year_from if has_year else None
        output["year_to"] = profile.year_to if has_year else None
        if has_year:
            labels.add("year")
            sources["year"] = "local_deterministic"
            confidence["year"] = 1.0
            evidence["year"] = [
                str(value)
                for value in (profile.year_from, profile.year_to)
                if value is not None
            ]
        else:
            labels.discard("year")
            sources.pop("year", None)
            confidence.pop("year", None)
            evidence.pop("year", None)
    if "negation" in signals:
        exclusions = (
            list(profile.exclusions)
            if profile is not None and "negation" in profile.labels
            else []
        )
        output["exclusions"] = exclusions
        if exclusions:
            labels.add("negation")
            sources["negation"] = "local_deterministic"
            confidence["negation"] = 1.0
            evidence["negation"] = exclusions
        else:
            labels.discard("negation")
            sources.pop("negation", None)
            confidence.pop("negation", None)
            evidence.pop("negation", None)
    output["labels"] = sorted(labels)
    output["label_sources"] = sources
    output["label_confidence"] = confidence
    output["evidence"] = evidence
    return _merge_constraint_row(output, profile=None, effective_signals=set())


def _constraint_signal_ready(profile: Any, signal: str) -> bool:
    if profile is None or profile.confidence < 0.8 or signal not in profile.labels:
        return False
    if signal == "method":
        return bool(profile.methods)
    if signal == "dataset":
        return bool(profile.datasets)
    if signal == "year":
        return profile.year_from is not None or profile.year_to is not None
    if signal == "negation":
        return bool(profile.exclusions)
    raise ValueError(f"unsupported constraint signal: {signal}")


_SIGNAL_FAMILY = {
    "task_provenance": "task_provenance",
    "method": "entity",
    "dataset": "entity",
    "year": "hard_constraint",
    "negation": "hard_constraint",
}


def _signal_ready(context: FusionQueryContext, signal: str) -> bool:
    if signal == "task_provenance":
        label = context.task_label
        return bool(
            label
            and label.reliability_weight > 0.0
            and label.task_label_status in TASK_PROVENANCE_ALLOWED_STATUSES
        )
    return _constraint_signal_ready(context.constraint_profile, signal)


def _project_context(context: FusionQueryContext, signal: str) -> FusionQueryContext:
    """Remove sibling signals so one target cannot activate another's pair."""

    if signal == "task_provenance":
        return FusionQueryContext(task_label=context.task_label)
    profile = context.constraint_profile
    if profile is None:
        return FusionQueryContext()
    projected = profile.model_copy(
        update={
            "labels": [signal],
            "methods": list(profile.methods) if signal == "method" else [],
            "datasets": list(profile.datasets) if signal == "dataset" else [],
            "tasks": [],
            "exclusions": list(profile.exclusions) if signal == "negation" else [],
            "year_from": profile.year_from if signal == "year" else None,
            "year_to": profile.year_to if signal == "year" else None,
            "has_negation": signal == "negation",
            "constraint_count": 1,
        }
    )
    return FusionQueryContext(constraint_profile=projected)


def _audit_signal(
    query: DocumentRankingQuery,
    ranker: Any,
    context: FusionQueryContext,
    signal: str,
) -> tuple[dict[str, object], int]:
    family = _SIGNAL_FAMILY[signal]
    original_families = ranker.feature_families
    ranker.feature_families = frozenset({family})
    try:
        audit = audit_fusion_query_activation(
            query, ranker, context=_project_context(context, signal)
        )
    finally:
        ranker.feature_families = original_families
    values = cast(Mapping[str, Mapping[str, object]], audit["families"])[family]
    count = int(values["effective_pair_count"])
    if signal != "task_provenance" and count:
        signal_counts = cast(Mapping[str, int], values["signal_effective_pair_count"])
        if int(signal_counts.get(signal, 0)) != count:
            raise ValueError(f"isolated {signal} audit returned inconsistent pair counts")
    return cast(dict[str, object], audit), count


def _validate_training_rows(
    rows: Sequence[Mapping[str, object]], *, label: str
) -> None:
    query_ids = [str(row.get("query_id", "")) for row in rows]
    if not all(query_ids) or len(query_ids) != len(set(query_ids)):
        raise ValueError(f"{label} query ids must be non-empty and unique")
    invalid = [
        query_id
        for query_id, row in zip(query_ids, rows, strict=True)
        if row.get("role") != "training" or row.get("split") != "auto_train"
    ]
    if invalid:
        raise ValueError(f"{label} contains non-training rows: {invalid[:10]}")


def _apply_candidate_overlay(
    query: DocumentRankingQuery,
    candidates: Sequence[Any],
) -> DocumentRankingQuery:
    return apply_frozen_candidate_overlay(
        query,
        FrozenCandidateOverlayEntry(
            query_sha256=_sha256(query.query.encode("utf-8")),
            candidates=tuple(candidates),
        ),
    )


def freeze_context(
    *,
    shard_dir: Path,
    manifest_path: Path,
    bundle_path: Path,
    output_dir: Path,
    max_pairs_per_query_family: int = 32,
    target_signals: set[str] | None = None,
    candidate_overlay_path: Path | None = None,
) -> dict[str, object]:
    if max_pairs_per_query_family <= 0:
        raise ValueError("pair limit must be positive")
    selected_target_signals = target_signals or set(_SIGNAL_FAMILY)
    unsupported_signals = selected_target_signals.difference(_SIGNAL_FAMILY)
    if unsupported_signals:
        raise ValueError(f"unsupported target signals: {sorted(unsupported_signals)}")

    production_manifest_bytes = manifest_path.read_bytes()
    production_bundle_bytes = bundle_path.read_bytes()
    ranker = load_f5_production_ranker_bytes(
        production_manifest_bytes, production_bundle_bytes
    )
    ranker.max_pairs_per_query_family = max_pairs_per_query_family
    ranker.publication_year_evidence_policy = ARXIV_MISSING_YEAR_EVIDENCE_POLICY
    resolver = ranker.context_store
    if not isinstance(resolver, UnifiedFusionContextResolver):
        raise ValueError("F5 artifact did not load the unified context resolver")

    components = unpack_production_bundle_components(production_bundle_bytes)
    task_rows = _jsonl_rows(components["task_labels"], label="task labels")
    constraint_rows = _jsonl_rows(
        components["constraint_labels"], label="constraint labels"
    )
    _validate_training_rows(task_rows, label="task labels")
    _validate_training_rows(constraint_rows, label="constraint labels")
    task_by_id = {str(row["query_id"]): dict(row) for row in task_rows}
    constraint_by_id = {
        str(row["query_id"]): _merge_constraint_row(
            row, profile=None, effective_signals=set()
        )
        for row in constraint_rows
    }
    base_task_store = FrozenTaskSlotLabelStore.from_jsonl_bytes(
        _jsonl_bytes([task_by_id[key] for key in sorted(task_by_id)])
    )
    base_constraint_store = FrozenConstraintProfileStore(
        [
            FrozenConstraintAnnotation.model_validate(constraint_by_id[key])
            for key in sorted(constraint_by_id)
        ]
    )
    base_context_store = FrozenFusionContextStore(
        task_store=base_task_store, constraint_store=base_constraint_store
    )

    shard_manifest_path = shard_dir / "manifest.json"
    shard_manifest_bytes = shard_manifest_path.read_bytes()
    shard_manifest = json.loads(shard_manifest_bytes)
    if shard_manifest.get("schema_version") not in {
        "large-scale-fusion-query-shards-v1",
        "large-scale-fusion-query-shards-v2",
    }:
        raise ValueError("unsupported query shard schema")
    if shard_manifest.get("test_partition_touched") is not False:
        raise ValueError("query shard manifest does not prove test isolation")
    batch_count = int(shard_manifest["batch_count"])
    completed = cast(Sequence[Mapping[str, object]], shard_manifest["completed_shards"])
    completed_by_index = {int(row["batch_index"]): row for row in completed}
    if len(completed) != batch_count or set(completed_by_index) != set(
        range(batch_count)
    ):
        raise ValueError("query shard manifest is incomplete or has duplicate batches")

    overlay_by_id: dict[str, list[DocumentCandidateEvidence]] = {}
    overlay_query_hash_by_id: dict[str, str] = {}
    overlay_bytes: bytes | None = None
    overlay_manifest_bytes: bytes | None = None
    overlay_manifest_path: Path | None = None
    overlay_manifest: Mapping[str, object] | None = None
    if candidate_overlay_path is not None:
        overlay_bytes = candidate_overlay_path.read_bytes()
        overlay_manifest_path = candidate_overlay_path.with_name("manifest.json")
        overlay_manifest_bytes = overlay_manifest_path.read_bytes()
        overlay_manifest = cast(
            Mapping[str, object], json.loads(overlay_manifest_bytes)
        )
        if overlay_manifest.get("schema_version") != (
            "pasa-negation-hard-constraint-overlay-v1"
        ):
            raise ValueError("unsupported candidate overlay schema")
        if overlay_manifest.get("test_partition_touched") is not False:
            raise ValueError("candidate overlay does not prove test isolation")
        overlay_outputs = cast(Mapping[str, object], overlay_manifest["outputs"])
        if overlay_outputs.get("overlay_rows_sha256") != _sha256(overlay_bytes):
            raise ValueError("candidate overlay hash does not match its manifest")
        for raw_line in overlay_bytes.splitlines():
            if not raw_line.strip():
                continue
            row = cast(Mapping[str, object], json.loads(raw_line))
            query_id = str(row["query_id"])
            if query_id in overlay_by_id:
                raise ValueError("candidate overlay query ids must be unique")
            overlay_query_hash_by_id[query_id] = str(row["query_sha256"])
            overlay_by_id[query_id] = [
                DocumentCandidateEvidence.model_validate(value)
                for value in cast(Sequence[object], row["appended_candidates"])
            ]

    signal_eligible_counts: Counter[str] = Counter()
    eligible_signals_by_query: dict[str, set[str]] = {}
    preliminary_effective_by_query: dict[str, set[str]] = {}
    deterministic_target_signals = selected_target_signals.intersection(
        {"year", "negation"}
    )
    seen_query_ids: set[str] = set()
    ordered_query_ids: list[str] = []
    overlay_seen_first_pass: set[str] = set()
    processed = 0
    for batch_index in range(batch_count):
        shard_path = shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
        shard_bytes = shard_path.read_bytes()
        metadata = completed_by_index[batch_index]
        if _sha256(shard_bytes) != metadata.get("sha256"):
            raise ValueError(f"query shard hash mismatch: {shard_path}")
        row_count = 0
        with gzip.GzipFile(fileobj=io.BytesIO(shard_bytes), mode="rb") as stream:
            for raw_line in stream:
                if not raw_line.strip():
                    continue
                row_count += 1
                processed += 1
                query_id, query_text = _query_header(raw_line)
                if query_id in seen_query_ids:
                    raise ValueError(f"duplicate query id across shards: {query_id}")
                seen_query_ids.add(query_id)
                ordered_query_ids.append(query_id)
                if query_id not in task_by_id or query_id not in constraint_by_id:
                    raise ValueError(f"query lacks frozen context row: {query_id}")
                expected_query_hash = _sha256(query_text.encode("utf-8"))
                if task_by_id[query_id].get("query_sha256") != expected_query_hash:
                    raise ValueError(f"task label query hash mismatch: {query_id}")
                if constraint_by_id[query_id].get("query_sha256") != expected_query_hash:
                    raise ValueError(f"constraint label query hash mismatch: {query_id}")

                base_context = base_context_store.for_training_query(query_text)
                local_context = resolver.for_local_query(query_text)
                if deterministic_target_signals:
                    constraint_by_id[query_id] = (
                        _replace_deterministic_constraint_signals(
                            constraint_by_id[query_id],
                            profile=local_context.constraint_profile,
                            signals=deterministic_target_signals,
                        )
                    )
                audit_contexts: dict[str, tuple[FusionQueryContext, bool]] = {}
                for signal in sorted(selected_target_signals):
                    if signal in deterministic_target_signals:
                        if _signal_ready(local_context, signal):
                            audit_contexts[signal] = (local_context, True)
                    elif _signal_ready(base_context, signal):
                        audit_contexts[signal] = (base_context, False)
                    elif _signal_ready(local_context, signal):
                        audit_contexts[signal] = (local_context, True)
                if not audit_contexts:
                    continue
                query = DocumentRankingQuery.model_validate_json(raw_line)
                if query_id in overlay_by_id:
                    if overlay_query_hash_by_id[query_id] != expected_query_hash:
                        raise ValueError(f"candidate overlay query hash mismatch: {query_id}")
                    query = _apply_candidate_overlay(query, overlay_by_id[query_id])
                    overlay_seen_first_pass.add(query_id)
                eligible_signals = set(audit_contexts)
                eligible_signals_by_query[query_id] = eligible_signals
                for signal in eligible_signals:
                    signal_eligible_counts[signal] += 1

                preliminary_effective: set[str] = set()
                local_effective: set[str] = set()
                for signal, (audit_context, is_local) in audit_contexts.items():
                    _audit, pair_count = _audit_signal(
                        query, ranker, audit_context, signal
                    )
                    if pair_count <= 0:
                        continue
                    preliminary_effective.add(signal)
                    if is_local:
                        local_effective.add(signal)
                preliminary_effective_by_query[query_id] = preliminary_effective
                if "task_provenance" in local_effective:
                    local_task = local_context.task_label
                    if local_task is None:
                        raise ValueError("effective local task context is missing")
                    task_by_id[query_id] = _merge_task_row(
                        task_by_id[query_id],
                        tasks=[value.normalized_value for value in local_task.tasks],
                    )
                local_constraints = local_effective.intersection(
                    {"method", "dataset", "year", "negation"}
                )
                if local_constraints:
                    constraint_by_id[query_id] = _merge_constraint_row(
                        constraint_by_id[query_id],
                        profile=local_context.constraint_profile,
                        effective_signals=local_constraints,
                    )
        if row_count != int(metadata["query_count"]):
            raise ValueError(f"query shard row count mismatch: {shard_path}")
        if (batch_index + 1) % 25 == 0 or batch_index + 1 == batch_count:
            print(
                json.dumps(
                    {
                        "event": "freeze_context_scan_progress",
                        "batches_complete": batch_index + 1,
                        "batch_count": batch_count,
                        "queries_scanned": processed,
                        "eligible_queries": len(eligible_signals_by_query),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    if processed != int(shard_manifest["query_count"]):
        raise ValueError("query shard aggregate query count mismatch")
    if overlay_seen_first_pass != set(overlay_by_id):
        raise ValueError("not every candidate overlay query was eligible in the first pass")

    merged_tasks = [task_by_id[key] for key in sorted(task_by_id)]
    merged_constraints = [constraint_by_id[key] for key in sorted(constraint_by_id)]
    _validate_training_rows(merged_tasks, label="merged task labels")
    _validate_training_rows(merged_constraints, label="merged constraint labels")
    task_bytes = _jsonl_bytes(merged_tasks)
    constraint_bytes = _jsonl_bytes(merged_constraints)
    final_task_store = FrozenTaskSlotLabelStore.from_jsonl_bytes(task_bytes)
    final_constraint_store = FrozenConstraintProfileStore(
        [FrozenConstraintAnnotation.model_validate(row) for row in merged_constraints]
    )
    final_context_store = FrozenFusionContextStore(
        task_store=final_task_store, constraint_store=final_constraint_store
    )

    selected: dict[str, list[str]] = {family: [] for family in FUSION_FAMILIES}
    pair_counts: Counter[str] = Counter()
    selected_signals: dict[str, list[str]] = {
        signal: [] for signal in _SIGNAL_FAMILY
    }
    signal_pair_counts: Counter[str] = Counter()
    backfill_counts: Counter[str] = Counter()
    candidate_backfill: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []
    training_pair_failures: list[dict[str, str]] = []
    final_audited_ids: set[str] = set()
    overlay_seen_second_pass: set[str] = set()
    for batch_index in range(batch_count):
        shard_path = shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
        shard_bytes = shard_path.read_bytes()
        if _sha256(shard_bytes) != completed_by_index[batch_index].get("sha256"):
            raise ValueError(f"query shard changed during freeze: {shard_path}")
        with gzip.GzipFile(fileobj=io.BytesIO(shard_bytes), mode="rb") as stream:
            for raw_line in stream:
                if not raw_line.strip():
                    continue
                query_id, query_text = _query_header(raw_line)
                eligible_signals = eligible_signals_by_query.get(query_id)
                if not eligible_signals:
                    continue
                final_audited_ids.add(query_id)
                query = DocumentRankingQuery.model_validate_json(raw_line)
                if query_id in overlay_by_id:
                    query = _apply_candidate_overlay(query, overlay_by_id[query_id])
                    overlay_seen_second_pass.add(query_id)
                final_context = final_context_store.for_training_query(query_text)

                for signal in preliminary_effective_by_query[query_id]:
                    if not _signal_ready(final_context, signal):
                        training_pair_failures.append(
                            {"query_id": query_id, "signal": signal, "reason": "gate"}
                        )
                        continue
                    _audit, isolated_count = _audit_signal(
                        query, ranker, final_context, signal
                    )
                    if isolated_count <= 0:
                        training_pair_failures.append(
                            {"query_id": query_id, "signal": signal, "reason": "pair"}
                        )

                final_families = {
                    _SIGNAL_FAMILY[signal]
                    for signal in eligible_signals
                    if _signal_ready(final_context, signal)
                }
                if final_families:
                    original_families = ranker.feature_families
                    ranker.feature_families = frozenset(final_families)
                    try:
                        audit = audit_fusion_query_activation(
                            query, ranker, context=final_context
                        )
                    finally:
                        ranker.feature_families = original_families
                else:
                    audit = {"query_sha256": _sha256(query_text.encode()), "families": {}}
                family_values = cast(
                    Mapping[str, Mapping[str, object]], audit["families"]
                )
                effective_signals: set[str] = set()
                query_signal_counts: dict[str, int] = {}
                if "task_provenance" in final_families:
                    count = int(
                        family_values["task_provenance"]["effective_pair_count"]
                    )
                    if count > 0 and "task_provenance" in eligible_signals:
                        effective_signals.add("task_provenance")
                        signal_pair_counts["task_provenance"] += count
                        query_signal_counts["task_provenance"] = count
                for family in ("entity", "hard_constraint"):
                    if family not in final_families:
                        continue
                    counts = cast(
                        Mapping[str, int],
                        family_values[family]["signal_effective_pair_count"],
                    )
                    for signal, count in counts.items():
                        if (
                            signal in eligible_signals
                            and signal in selected_target_signals
                            and int(count) > 0
                        ):
                            effective_signals.add(signal)
                            signal_pair_counts[signal] += int(count)
                            query_signal_counts[signal] = int(count)
                missing_signals = sorted(eligible_signals - effective_signals)
                for signal in missing_signals:
                    backfill_counts[signal] += 1
                if missing_signals:
                    candidate_backfill.append(
                        {
                            "query_id": query_id,
                            "query": query_text,
                            "signals": missing_signals,
                            "families": sorted(
                                {_SIGNAL_FAMILY[signal] for signal in missing_signals}
                            ),
                            "reason": "no_gold_hard_negative_feature_contrast",
                        }
                    )
                for signal in effective_signals:
                    selected_signals[signal].append(query_id)
                effective_families = sorted(
                    {_SIGNAL_FAMILY[signal] for signal in effective_signals}
                )
                for family in effective_families:
                    selected[family].append(query_id)
                    pair_counts[family] += int(
                        family_values[family]["effective_pair_count"]
                    )
                if not preliminary_effective_by_query[query_id].issubset(
                    effective_signals
                ):
                    for signal in sorted(
                        preliminary_effective_by_query[query_id] - effective_signals
                    ):
                        training_pair_failures.append(
                            {
                                "query_id": query_id,
                                "signal": signal,
                                "reason": "combined_pair_cap",
                            }
                        )
                if effective_families:
                    snapshots.append(
                        {
                            "query_id": query_id,
                            "query_sha256": audit["query_sha256"],
                            "effective_families": effective_families,
                            "effective_signals": sorted(effective_signals),
                            "effective_pair_count_by_signal": {
                                signal: query_signal_counts[signal]
                                for signal in sorted(effective_signals)
                            },
                            "candidate_evidence_sha256": {
                                family: family_values[family][
                                    "candidate_evidence_sha256"
                                ]
                                for family in effective_families
                            },
                            "selected_pair_evidence_sha256": {
                                family: family_values[family][
                                    "selected_pair_evidence_sha256"
                                ]
                                for family in effective_families
                            },
                        }
                    )
        if (batch_index + 1) % 25 == 0 or batch_index + 1 == batch_count:
            print(
                json.dumps(
                    {
                        "event": "freeze_context_revalidation_progress",
                        "batches_complete": batch_index + 1,
                        "batch_count": batch_count,
                        "queries_revalidated": len(final_audited_ids),
                        "effective_queries": len(snapshots),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    if final_audited_ids != set(eligible_signals_by_query):
        raise ValueError("not every eligible query was revalidated")
    if overlay_seen_second_pass != set(overlay_by_id):
        raise ValueError("not every candidate overlay query was revalidated")
    if training_pair_failures:
        raise ValueError(
            "frozen context did not reproduce selected training pairs: "
            + json.dumps(training_pair_failures[:10], ensure_ascii=False, sort_keys=True)
        )

    snapshot_bytes = _jsonl_bytes(snapshots)
    queue_bytes = _jsonl_bytes(candidate_backfill)
    base_gold_hit_query_count = int(shard_manifest["gold_hit_query_count"])
    overlay_gold_hit_delta = 0
    if overlay_manifest is not None:
        overlay_gold_hit_delta = int(
            overlay_manifest.get("expanded_gold_hit_query_count", 0)
        ) - int(overlay_manifest.get("base_gold_hit_query_count", 0))
        if overlay_gold_hit_delta < 0:
            raise ValueError("candidate overlay Gold-hit coverage regressed")
    report: dict[str, object] = {
        "schema_version": "directed-fusion-context-freeze-v3",
        "target_signals": sorted(selected_target_signals),
        "query_count": processed,
        "effective_query_count_by_family": {
            family: len(ids) for family, ids in sorted(selected.items())
        },
        "effective_pair_count_by_family": dict(sorted(pair_counts.items())),
        "eligible_query_count_by_signal": dict(sorted(signal_eligible_counts.items())),
        "effective_query_count_by_signal": {
            signal: len(ids) for signal, ids in sorted(selected_signals.items())
        },
        "effective_pair_count_by_signal": dict(sorted(signal_pair_counts.items())),
        "candidate_backfill_query_count": len(candidate_backfill),
        "candidate_backfill_query_count_by_signal": dict(sorted(backfill_counts.items())),
        "training_pair_revalidation_failure_count": len(training_pair_failures),
        "selected_query_ids_by_family": {
            family: sorted(ids) for family, ids in sorted(selected.items())
        },
        "selected_query_ids_by_signal": {
            signal: sorted(ids) for signal, ids in sorted(selected_signals.items())
        },
        "outputs": {
            "task_labels_sha256": _sha256(task_bytes),
            "constraint_labels_sha256": _sha256(constraint_bytes),
            "context_snapshot_sha256": _sha256(snapshot_bytes),
            "candidate_backfill_sha256": _sha256(queue_bytes),
        },
        "resolver_id": resolver.resolver_id,
        "resolver_sha256": resolver.resolver_sha256,
        "negation_evidence_schema_version": NEGATION_EVIDENCE_SCHEMA_VERSION,
        "publication_year_evidence_policy": (
            ranker.publication_year_evidence_policy
        ),
        "max_pairs_per_query_family": max_pairs_per_query_family,
        "source_candidate_package": {
            "mode": "immutable-shards-referenced",
            "manifest_path": str(shard_manifest_path.resolve()),
            "manifest_sha256": _sha256(shard_manifest_bytes),
            "schema_version": shard_manifest.get("schema_version"),
            "batch_count": batch_count,
            "query_count": int(shard_manifest["query_count"]),
            "query_id_order_sha256": _sha256(
                "".join(f"{query_id}\n" for query_id in ordered_query_ids).encode(
                    "utf-8"
                )
            ),
            "base_candidate_count": int(shard_manifest["candidate_count"]),
            "overlay_candidate_count": sum(
                len(candidates) for candidates in overlay_by_id.values()
            ),
            "expanded_candidate_count": int(shard_manifest["candidate_count"])
            + sum(len(candidates) for candidates in overlay_by_id.values()),
            "base_gold_hit_query_count": base_gold_hit_query_count,
            "expanded_gold_hit_query_count": (
                base_gold_hit_query_count + overlay_gold_hit_delta
            ),
            "base_candidate_membership_and_order_unchanged": True,
            "overlay_append_order_deterministic": True,
            "all_shard_hashes_verified_twice": True,
            "candidate_overlay": (
                {
                    "mode": "append-unique-hard-constraint-only",
                    "path": str(candidate_overlay_path.resolve()),
                    "sha256": _sha256(overlay_bytes),
                    "manifest_path": str(overlay_manifest_path.resolve()),
                    "manifest_sha256": _sha256(overlay_manifest_bytes),
                }
                if candidate_overlay_path is not None
                and overlay_bytes is not None
                and overlay_manifest_bytes is not None
                and overlay_manifest_path is not None
                else {"mode": "none"}
            ),
        },
        "training_evidence_policy": {
            "source_candidate_hydration_policy": shard_manifest.get(
                "candidate_hydration_policy", "legacy-v1-unspecified"
            ),
            "publication_year_evidence_policy": (
                ranker.publication_year_evidence_policy
            ),
            "candidate_rewrite_performed": False,
            "candidate_overlay_policy": (
                "append-unique-hard-constraint-only-pasa-candidates-v1"
                if overlay_by_id
                else "none"
            ),
            "pasa_gold_source_suppression": (
                "gold-role-pasa-only-source-features-suppressed-v1"
            ),
            "pair_direction": "gold-to-valid-contrast-only-v3",
            "query_pair_balance": max_pairs_per_query_family,
        },
        "partition_validation": {
            "task_rows_all_training_auto_train": True,
            "constraint_rows_all_training_auto_train": True,
            "shard_manifest_test_partition_touched": False,
            "unique_query_ids_verified": True,
        },
        "input_artifacts": {
            "production_manifest_sha256": _sha256(production_manifest_bytes),
            "production_bundle_sha256": _sha256(production_bundle_bytes),
            "candidate_overlay_sha256": (
                _sha256(overlay_bytes) if overlay_bytes is not None else None
            ),
            "candidate_overlay_manifest_sha256": (
                _sha256(overlay_manifest_bytes)
                if overlay_manifest_bytes is not None
                else None
            ),
        },
        "online_requests_made": 0,
        "llm_requests_made": 0,
        "development_labels_used_for_training": False,
        "test_partition_touched": False,
        "training_started": False,
        "production_lock_modified": False,
    }
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_bytes(output_dir / "task-labels.merged.jsonl", task_bytes)
    _atomic_bytes(output_dir / "constraint-labels.merged.jsonl", constraint_bytes)
    _atomic_bytes(output_dir / "effective-context-snapshot.jsonl", snapshot_bytes)
    _atomic_bytes(output_dir / "candidate-backfill-queue.jsonl", queue_bytes)
    _atomic_bytes(output_dir / "manifest.json", report_bytes)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=Path("artifacts/models/gated-feature-fusion-13300-v1/manifest.json"),
    )
    parser.add_argument(
        "--production-bundle",
        type=Path,
        default=Path("artifacts/models/gated-feature-fusion-13300-v1/weights.bundle"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-overlay", type=Path)
    parser.add_argument("--max-pairs-per-query-family", type=int, default=32)
    parser.add_argument(
        "--target-signal",
        action="append",
        choices=("task_provenance", "method", "dataset", "year", "negation"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = freeze_context(
        shard_dir=args.shard_dir,
        manifest_path=args.production_manifest,
        bundle_path=args.production_bundle,
        output_dir=args.output_dir,
        max_pairs_per_query_family=args.max_pairs_per_query_family,
        target_signals=set(args.target_signal) if args.target_signal else None,
        candidate_overlay_path=args.candidate_overlay,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key
                not in {
                    "selected_query_ids_by_family",
                    "selected_query_ids_by_signal",
                }
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
