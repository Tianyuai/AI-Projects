"""Validated, resumable inputs and checkpoints for large-scale F4/F5 fitting."""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from paper_search.domain.models import Paper
from paper_search.evaluation.predictions import (
    paper_evaluation_aliases,
    paper_matches_evaluation_ids,
)
from paper_search.learning.cpu_document_ranker import (
    DocumentCandidateEvidence,
    DocumentRankingQuery,
    build_production_document_candidates,
)
from paper_search.learning.openalex_daily_schedule import search_action_identity
from paper_search.learning.query_constraint_annotations import (
    FrozenConstraintAnnotation,
    query_sha256,
)
from paper_search.learning.task_slot_document_ranker import FrozenTaskSlotLabelStore
from paper_search.processing.deduplicate import deduplicate_papers
from paper_search.retrieval.pasa_paper_database import (
    PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE,
    mark_pasa_training_gold_injected,
)


_BUNDLE_HEADER = struct.Struct("<8sQQQQ")
_BUNDLE_MAGIC = b"F5PROD1\0"
_CANDIDATE_HYDRATION_POLICY = "receipt-candidates-v2-conservative-pasa-gold-marker"


@dataclass(frozen=True)
class FusionTrainingPackage:
    handoff_path: Path
    partition_path: Path
    production_bundle_path: Path
    query_ids: tuple[str, ...]
    rows_by_query_id: dict[str, dict[str, Any]]
    ordered_receipt_roots: tuple[Path, ...]
    conflicting_query_ids: tuple[str, ...]
    task_labels_bytes: bytes
    constraint_labels_bytes: bytes
    task_label_count: int
    constraint_label_count: int
    input_sha256: str
    additive_receipt_roots: tuple[Path, ...] = ()
    candidate_hydration_policy: str = _CANDIDATE_HYDRATION_POLICY


@dataclass(frozen=True)
class FusionTrainingCheckpoint:
    input_sha256: str
    epoch_index: int
    next_batch_index: int
    batch_count: int
    pair_counts: dict[str, int]
    query_counts: dict[str, int]
    weights: dict[str, np.ndarray]
    replay_pair_counts: dict[str, int] = field(default_factory=dict)
    replay_query_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class FrozenCandidateOverlayEntry:
    query_sha256: str
    candidates: tuple[DocumentCandidateEvidence, ...]


@dataclass(frozen=True)
class FrozenFusionActivationInputs:
    manifest_path: Path
    manifest_sha256: str
    shard_dir: Path
    shard_manifest_sha256: str
    batch_count: int
    task_labels_path: Path
    constraint_labels_path: Path
    overlay_by_query_id: dict[str, FrozenCandidateOverlayEntry]


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object")
    return raw


def _jsonl_objects(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{label} row {line_number} must be an object")
        output.append(raw)
    return output


def unpack_production_bundle_components(payload: bytes) -> dict[str, bytes]:
    if len(payload) < _BUNDLE_HEADER.size:
        raise ValueError("production bundle is truncated")
    magic, baseline_size, fusion_size, task_size, constraint_size = (
        _BUNDLE_HEADER.unpack_from(payload)
    )
    if magic != _BUNDLE_MAGIC:
        raise ValueError("production bundle has invalid magic")
    expected = (
        _BUNDLE_HEADER.size
        + baseline_size
        + fusion_size
        + task_size
        + constraint_size
    )
    if expected != len(payload):
        raise ValueError("production bundle length mismatch")
    task_start = _BUNDLE_HEADER.size + baseline_size + fusion_size
    constraint_start = task_start + task_size
    return {
        "baseline_weights": payload[_BUNDLE_HEADER.size : _BUNDLE_HEADER.size + baseline_size],
        "fusion_weights": payload[
            _BUNDLE_HEADER.size + baseline_size : task_start
        ],
        "task_labels": payload[task_start:constraint_start],
        "constraint_labels": payload[
            constraint_start : constraint_start + constraint_size
        ],
    }


def unpack_production_bundle_labels(payload: bytes) -> tuple[bytes, bytes]:
    components = unpack_production_bundle_components(payload)
    return components["task_labels"], components["constraint_labels"]


def load_training_package(
    *,
    handoff_path: Path,
    partition_path: Path,
    production_bundle_path: Path,
) -> FusionTrainingPackage:
    """Validate the sealed handoff and join its ready rows to frozen LLM labels."""

    handoff = _json_object(handoff_path, label="training handoff")
    if handoff.get("schema_version") != "openalex-ranking-training-handoff-v1":
        raise ValueError("unsupported training handoff schema")
    if handoff.get("test_partition_touched") is not False:
        raise ValueError("training handoff touched the test partition")
    if handoff.get("online_llm_requests") != 0:
        raise ValueError("training handoff used online LLM requests")

    raw_query_ids = handoff.get("cumulative_unique_ready_query_ids")
    if not isinstance(raw_query_ids, list) or not raw_query_ids or any(
        not isinstance(value, str) or not value for value in raw_query_ids
    ):
        raise ValueError("training handoff ready query ids are invalid")
    query_ids = tuple(raw_query_ids)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("training handoff ready query ids are duplicated")

    raw_roots = handoff.get("ordered_receipt_roots")
    if not isinstance(raw_roots, list) or not raw_roots or any(
        not isinstance(value, str) or not value for value in raw_roots
    ):
        raise ValueError("training handoff receipt roots are invalid")
    roots = tuple(Path(value).resolve() for value in raw_roots)
    if len(set(roots)) != len(roots):
        raise ValueError("training handoff receipt roots are duplicated")
    missing_roots = [root for root in roots if not root.is_dir()]
    if missing_roots:
        raise ValueError(f"training handoff receipt root is unavailable: {missing_roots[0]}")

    supplement = handoff.get("high_recall_candidate_supplement")
    additive_roots: tuple[Path, ...] = ()
    if supplement is not None:
        if not isinstance(supplement, dict):
            raise ValueError("training handoff high-recall supplement is invalid")
        if supplement.get("llm_request_count") != 0:
            raise ValueError("training handoff high-recall supplement used LLM requests")
        if supplement.get("test_partition_touched") is not False:
            raise ValueError("training handoff high-recall supplement touched test data")
        raw_additive_roots = supplement.get("receipt_roots")
        if not isinstance(raw_additive_roots, list) or not raw_additive_roots or any(
            not isinstance(value, str) or not value for value in raw_additive_roots
        ):
            raise ValueError("training handoff additive receipt roots are invalid")
        additive_roots = tuple(Path(value).resolve() for value in raw_additive_roots)
        if len(set(additive_roots)) != len(additive_roots):
            raise ValueError("training handoff additive receipt roots are duplicated")
        if not set(additive_roots).issubset(roots):
            raise ValueError("training handoff additive receipt root is not sealed")

    rows: dict[str, dict[str, Any]] = {}
    partition_bytes = partition_path.read_bytes()
    for row in _jsonl_objects(partition_bytes, label="training partition"):
        if row.get("role") != "training" or row.get("split") != "auto_train":
            raise ValueError("training partition contains a non-auto_train row")
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in rows:
            raise ValueError("training partition query ids must be unique")
        rows[query_id] = row
    missing_partition = set(query_ids) - set(rows)
    if missing_partition:
        raise ValueError(
            "training partition coverage is incomplete: "
            f"{sorted(missing_partition)[0]}"
        )

    task_bytes, constraint_bytes = unpack_production_bundle_labels(
        production_bundle_path.read_bytes()
    )
    task_rows = _jsonl_objects(task_bytes, label="task labels")
    constraint_rows = _jsonl_objects(constraint_bytes, label="constraint labels")
    FrozenTaskSlotLabelStore.from_jsonl_bytes(task_bytes)
    for row in constraint_rows:
        FrozenConstraintAnnotation.model_validate(row)
    task_by_id = {str(row.get("query_id", "")): row for row in task_rows}
    constraint_by_id = {str(row.get("query_id", "")): row for row in constraint_rows}
    if len(task_by_id) != len(task_rows):
        raise ValueError("task label query ids must be unique")
    if len(constraint_by_id) != len(constraint_rows):
        raise ValueError("constraint label query ids must be unique")
    missing_task = set(query_ids) - set(task_by_id)
    if missing_task:
        raise ValueError(
            f"task label coverage is incomplete: {sorted(missing_task)[0]}"
        )
    missing_constraint = set(query_ids) - set(constraint_by_id)
    if missing_constraint:
        raise ValueError(
            "constraint label coverage is incomplete: "
            f"{sorted(missing_constraint)[0]}"
        )
    for query_id in query_ids:
        query = str(rows[query_id].get("query", ""))
        digest = query_sha256(query)
        if task_by_id[query_id].get("query_sha256") != digest:
            raise ValueError(f"task label query hash mismatch: {query_id}")
        if constraint_by_id[query_id].get("query_sha256") != digest:
            raise ValueError(f"constraint label query hash mismatch: {query_id}")

    raw_conflicts = handoff.get("conflicts", [])
    if not isinstance(raw_conflicts, list) or any(
        not isinstance(value, str) for value in raw_conflicts
    ):
        raise ValueError("training handoff conflicts are invalid")
    conflicts = tuple(value for value in raw_conflicts if value in set(query_ids))
    identity = json.dumps(
        {
            "handoff_sha256": _sha256(handoff_path.read_bytes()),
            "partition_sha256": _sha256(partition_bytes),
            "bundle_sha256": _sha256(production_bundle_path.read_bytes()),
            "query_ids": query_ids,
            "receipt_roots": [str(root) for root in roots],
            "additive_receipt_roots": [str(root) for root in additive_roots],
            "candidate_hydration_policy": _CANDIDATE_HYDRATION_POLICY,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return FusionTrainingPackage(
        handoff_path=handoff_path.resolve(),
        partition_path=partition_path.resolve(),
        production_bundle_path=production_bundle_path.resolve(),
        query_ids=query_ids,
        rows_by_query_id={query_id: rows[query_id] for query_id in query_ids},
        ordered_receipt_roots=roots,
        conflicting_query_ids=conflicts,
        task_labels_bytes=task_bytes,
        constraint_labels_bytes=constraint_bytes,
        task_label_count=len(task_rows),
        constraint_label_count=len(constraint_rows),
        input_sha256=_sha256(identity),
        additive_receipt_roots=additive_roots,
        candidate_hydration_policy=_CANDIDATE_HYDRATION_POLICY,
    )


def with_context_label_files(
    package: FusionTrainingPackage,
    *,
    task_labels_path: Path,
    constraint_labels_path: Path,
) -> FusionTrainingPackage:
    """Replace context labels with a validated, content-addressed frozen supplement."""

    task_bytes = task_labels_path.read_bytes()
    constraint_bytes = constraint_labels_path.read_bytes()
    task_rows = _jsonl_objects(task_bytes, label="task labels")
    constraint_rows = _jsonl_objects(constraint_bytes, label="constraint labels")
    FrozenTaskSlotLabelStore.from_jsonl_bytes(task_bytes)
    for row in constraint_rows:
        FrozenConstraintAnnotation.model_validate(row)
    task_by_id = {str(row.get("query_id", "")): row for row in task_rows}
    constraint_by_id = {
        str(row.get("query_id", "")): row for row in constraint_rows
    }
    if len(task_by_id) != len(task_rows) or len(constraint_by_id) != len(
        constraint_rows
    ):
        raise ValueError("context label query ids must be unique")
    for query_id in package.query_ids:
        if query_id not in task_by_id or query_id not in constraint_by_id:
            raise ValueError(f"context label coverage is incomplete: {query_id}")
        row = package.rows_by_query_id[query_id]
        digest = query_sha256(str(row["query"]))
        if task_by_id[query_id].get("query_sha256") != digest:
            raise ValueError(f"task label query hash mismatch: {query_id}")
        if constraint_by_id[query_id].get("query_sha256") != digest:
            raise ValueError(f"constraint label query hash mismatch: {query_id}")
        if (
            task_by_id[query_id].get("role") != "training"
            or task_by_id[query_id].get("split") != "auto_train"
            or constraint_by_id[query_id].get("role") != "training"
            or constraint_by_id[query_id].get("split") != "auto_train"
        ):
            raise ValueError("context labels must remain auto_train-only")
    identity = json.dumps(
        {
            "base_input_sha256": package.input_sha256,
            "task_labels_sha256": _sha256(task_bytes),
            "constraint_labels_sha256": _sha256(constraint_bytes),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return replace(
        package,
        task_labels_bytes=task_bytes,
        constraint_labels_bytes=constraint_bytes,
        task_label_count=len(task_rows),
        constraint_label_count=len(constraint_rows),
        input_sha256=_sha256(identity),
    )


def index_training_receipts(
    package: FusionTrainingPackage,
) -> dict[str, tuple[Path, ...]]:
    """Index ready-query retrieval files in the exact sealed root order."""

    selected = set(package.query_ids)
    indexed: dict[str, list[Path]] = {query_id: [] for query_id in package.query_ids}
    for root in package.ordered_receipt_roots:
        for path in sorted(root.rglob("retrieval/attempt-01/*.json")):
            query_id = path.stem
            if query_id in selected:
                indexed[query_id].append(path.resolve())
    missing = [query_id for query_id, paths in indexed.items() if not paths]
    if missing:
        raise ValueError(f"ready query has no receipt: {missing[0]}")
    return {query_id: tuple(paths) for query_id, paths in indexed.items()}


def _generation_path(retrieval_path: Path) -> Path:
    parts = list(retrieval_path.parts)
    try:
        index = parts.index("retrieval")
    except ValueError as error:
        raise ValueError(f"invalid retrieval path: {retrieval_path}") from error
    parts[index] = "generation"
    return Path(*parts)


def _fair_merge_non_reinforcing_candidates(
    baseline: list[DocumentCandidateEvidence],
    augmented: list[DocumentCandidateEvidence],
) -> list[DocumentCandidateEvidence]:
    """Keep baseline evidence immutable and fairly merge supplemental members."""

    baseline_aliases = {
        alias
        for candidate in baseline
        for alias in paper_evaluation_aliases(candidate.paper)
    }
    baseline_source_positions = {
        item for candidate in baseline for item in candidate.source_ranks.items()
    }
    baseline_papers = [candidate.paper for candidate in baseline]
    output = list(baseline)
    for candidate in augmented:
        if baseline_aliases.intersection(paper_evaluation_aliases(candidate.paper)):
            continue
        if baseline_source_positions.intersection(candidate.source_ranks.items()):
            continue
        if len(deduplicate_papers([*baseline_papers, candidate.paper]).papers) == len(
            baseline_papers
        ):
            continue
        output.append(candidate)
    return sorted(
        output,
        key=lambda candidate: candidate.baseline_score,
        reverse=True,
    )


def build_document_ranking_query(
    package: FusionTrainingPackage,
    query_id: str,
    receipt_paths: tuple[Path, ...],
    *,
    additive_receipt_roots: tuple[Path, ...] = (),
    non_reinforcing_additive: bool = False,
) -> DocumentRankingQuery:
    """Merge receipts, optionally preserving base evidence for additive roots."""

    if query_id not in package.rows_by_query_id:
        raise ValueError(f"query is outside the sealed training package: {query_id}")
    row = package.rows_by_query_id[query_id]
    gold_paper_ids = list(row["gold_paper_ids"])
    selected_actions: dict[tuple[str, str, str, str], tuple[str, list[Paper]]] = {}
    baseline_actions: dict[tuple[str, str, str, str], tuple[str, list[Paper]]] = {}
    resolved_additive_roots = tuple(root.resolve() for root in additive_receipt_roots)
    for retrieval_path in receipt_paths:
        additive = any(
            retrieval_path.is_relative_to(root) for root in resolved_additive_roots
        )
        retrieval = _json_object(retrieval_path, label="retrieval receipt")
        if retrieval.get("query_id") != query_id:
            raise ValueError(f"retrieval query id mismatch: {retrieval_path}")
        generation_path = _generation_path(retrieval_path)
        generation = _json_object(generation_path, label="generation receipt")
        if generation.get("query_id") != query_id:
            raise ValueError(f"generation query id mismatch: {generation_path}")
        provenance = generation.get("generation_provenance")
        namespace = "default"
        if isinstance(provenance, dict):
            raw_namespace = provenance.get("candidate_source_namespace")
            if isinstance(raw_namespace, str) and raw_namespace.strip():
                namespace = raw_namespace.strip()
        raw_actions = generation.get("actions")
        raw_results = retrieval.get("results")
        if not isinstance(raw_actions, list) or not isinstance(raw_results, list):
            raise ValueError(f"invalid receipt rows for {query_id}")
        identities: dict[str, tuple[str, str, str]] = {}
        for action in raw_actions:
            if not isinstance(action, dict) or not isinstance(
                action.get("action_id"), str
            ):
                continue
            action_identity = search_action_identity(action)
            if action_identity is not None:
                identities[action["action_id"]] = (
                    action_identity.action_type,
                    action_identity.search_mode,
                    action_identity.normalized_text,
                )
        for result in raw_results:
            if not isinstance(result, dict):
                continue
            action_id = result.get("action_id")
            if not isinstance(action_id, str) or action_id not in identities:
                continue
            hits = result.get("hits")
            if not isinstance(hits, list):
                continue
            identity_tuple = identities[action_id]
            selected_identity = (namespace, *identity_tuple)
            digest_material = (
                identity_tuple if namespace == "default" else selected_identity
            )
            identity_digest = hashlib.sha256(
                json.dumps(digest_material, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:12]
            source_id = f"{action_id}@{identity_digest}"
            validated_hits = [Paper.model_validate(hit) for hit in hits]
            retrieval_provenance = retrieval.get("retrieval_provenance")
            mixed_training_candidates = (
                isinstance(retrieval_provenance, dict)
                and retrieval_provenance.get("candidate_policy")
                == "mixed_lexical_plus_gold_training"
            )
            if mixed_training_candidates:
                # Legacy immutable PASA receipts predate per-paper injection markers.
                # Conservatively suppress source-only evidence for every PASA Gold
                # candidate; lexical Gold remains usable through its text evidence.
                validated_hits = [
                    mark_pasa_training_gold_injected(paper)
                    if paper_matches_evaluation_ids(paper, gold_paper_ids)
                    else paper
                    for paper in validated_hits
                ]
            previous = selected_actions.get(selected_identity)
            if additive and previous is not None:
                selected_actions[selected_identity] = (
                    previous[0],
                    [*previous[1], *validated_hits],
                )
            else:
                selected_actions[selected_identity] = (source_id, validated_hits)
            if not additive:
                baseline_actions[selected_identity] = (source_id, validated_hits)
    query = str(row["query"])
    candidates = build_production_document_candidates(
        query,
        list(selected_actions.values()),
    )
    if non_reinforcing_additive:
        baseline_candidates = build_production_document_candidates(
            query,
            list(baseline_actions.values()),
        )
        candidates = _fair_merge_non_reinforcing_candidates(
            baseline_candidates,
            candidates,
        )
    if not candidates:
        raise ValueError(f"ready query has no valid candidates: {query_id}")
    compact_candidates = [
        DocumentCandidateEvidence(
            paper=Paper(
                canonical_id=candidate.paper.canonical_id,
                title=candidate.paper.title,
                abstract=candidate.paper.abstract,
                publication_year=candidate.paper.publication_year,
                citation_count=candidate.paper.citation_count,
                doi=candidate.paper.doi,
                arxiv_id=candidate.paper.arxiv_id,
                openalex_id=candidate.paper.openalex_id,
                semantic_scholar_id=candidate.paper.semantic_scholar_id,
                sources=list(candidate.paper.sources),
            ),
            baseline_score=candidate.baseline_score,
            source_ranks=candidate.source_ranks,
        )
        for candidate in candidates
    ]
    return DocumentRankingQuery(
        query_id=query_id,
        query=query,
        gold_paper_ids=gold_paper_ids,
        candidates=compact_candidates,
    )


def write_query_shard(path: Path, queries: list[DocumentRankingQuery]) -> None:
    if not queries:
        raise ValueError("training query shard cannot be empty")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("training query shard ids must be unique")
    payload = b"".join(
        query.model_dump_json(
            exclude_none=True, exclude_computed_fields=True
        ).encode("utf-8")
        + b"\n"
        for query in queries
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=6, mtime=0, fileobj=raw_stream
        ) as stream:
            stream.write(payload)
    temporary.replace(path)


def query_has_gold_candidate(query: DocumentRankingQuery) -> bool:
    return any(
        paper_matches_evaluation_ids(candidate.paper, query.gold_paper_ids)
        for candidate in query.candidates
    )


def apply_frozen_candidate_overlay(
    query: DocumentRankingQuery,
    entry: FrozenCandidateOverlayEntry,
) -> DocumentRankingQuery:
    """Append a hash-bound, hard-constraint-only overlay without changing base order."""

    if query_sha256(query.query) != entry.query_sha256:
        raise ValueError("candidate overlay query hash mismatch")
    merged = list(query.candidates)
    aliases = {
        alias
        for candidate in query.candidates
        for alias in paper_evaluation_aliases(candidate.paper)
    }
    for candidate in entry.candidates:
        candidate_aliases = paper_evaluation_aliases(candidate.paper)
        if aliases.intersection(candidate_aliases):
            raise ValueError("candidate overlay overlaps immutable base membership")
        if PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE not in candidate.paper.sources:
            raise ValueError("candidate overlay lacks the hard-constraint-only marker")
        merged.append(candidate)
        aliases.update(candidate_aliases)
    return query.model_copy(update={"candidates": merged})


def ordered_query_ids_sha256(query_ids: Sequence[str]) -> str:
    return _sha256("".join(f"{query_id}\n" for query_id in query_ids).encode("utf-8"))


def load_frozen_fusion_activation_inputs(
    manifest_path: Path,
    *,
    expected_query_ids: Sequence[str],
    production_manifest_bytes: bytes,
    production_bundle_bytes: bytes,
) -> FrozenFusionActivationInputs:
    """Validate and resolve the exact shards, context, and overlay audited by freeze."""

    manifest_bytes = manifest_path.read_bytes()
    manifest = _json_object(manifest_path, label="fusion activation manifest")
    if manifest.get("schema_version") != "directed-fusion-context-freeze-v3":
        raise ValueError("unsupported fusion activation manifest schema")
    for flag in (
        "development_labels_used_for_training",
        "test_partition_touched",
        "training_started",
        "production_lock_modified",
    ):
        if manifest.get(flag) is not False:
            raise ValueError(f"fusion activation manifest safety flag failed: {flag}")
    if manifest.get("online_requests_made") != 0 or manifest.get("llm_requests_made") != 0:
        raise ValueError("fusion activation package is not zero-request")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("fusion activation outputs are invalid")
    task_path = manifest_path.with_name("task-labels.merged.jsonl")
    constraint_path = manifest_path.with_name("constraint-labels.merged.jsonl")
    if outputs.get("task_labels_sha256") != _sha256(task_path.read_bytes()):
        raise ValueError("frozen task label hash mismatch")
    if outputs.get("constraint_labels_sha256") != _sha256(
        constraint_path.read_bytes()
    ):
        raise ValueError("frozen constraint label hash mismatch")

    artifacts = manifest.get("input_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("fusion activation input artifacts are invalid")
    if artifacts.get("production_manifest_sha256") != _sha256(
        production_manifest_bytes
    ):
        raise ValueError("production manifest differs from frozen activation input")
    if artifacts.get("production_bundle_sha256") != _sha256(production_bundle_bytes):
        raise ValueError("production bundle differs from frozen activation input")

    source = manifest.get("source_candidate_package")
    if not isinstance(source, dict):
        raise ValueError("fusion activation source candidate package is invalid")
    if source.get("query_count") != len(expected_query_ids):
        raise ValueError("frozen candidate query count mismatch")
    if source.get("query_id_order_sha256") != ordered_query_ids_sha256(
        expected_query_ids
    ):
        raise ValueError("frozen candidate query order mismatch")
    source_manifest_path = Path(str(source.get("manifest_path", "")))
    source_manifest_bytes = source_manifest_path.read_bytes()
    if source.get("manifest_sha256") != _sha256(source_manifest_bytes):
        raise ValueError("frozen source shard manifest hash mismatch")
    source_manifest = json.loads(source_manifest_bytes)
    if not isinstance(source_manifest, dict) or source_manifest.get(
        "schema_version"
    ) not in {
        "large-scale-fusion-query-shards-v1",
        "large-scale-fusion-query-shards-v2",
    }:
        raise ValueError("unsupported frozen source shard schema")
    if source_manifest.get("test_partition_touched") is not False:
        raise ValueError("frozen source shards do not prove test isolation")
    batch_count = int(source_manifest.get("batch_count", 0))
    completed = source_manifest.get("completed_shards")
    if not isinstance(completed, list) or batch_count <= 0 or len(completed) != batch_count:
        raise ValueError("frozen source shard manifest is incomplete")
    indexes: set[int] = set()
    query_count = candidate_count = gold_hit_count = 0
    shard_dir = source_manifest_path.parent
    for raw in completed:
        if not isinstance(raw, dict):
            raise ValueError("frozen source shard record is invalid")
        index = int(raw.get("batch_index", -1))
        if index in indexes or index < 0 or index >= batch_count:
            raise ValueError("frozen source shard indexes are invalid")
        indexes.add(index)
        shard_path = shard_dir / f"shard-{index:05d}.jsonl.gz"
        if raw.get("sha256") != _sha256(shard_path.read_bytes()):
            raise ValueError(f"frozen source shard hash mismatch: {shard_path}")
        query_count += int(raw.get("query_count", 0))
        candidate_count += int(raw.get("candidate_count", 0))
        gold_hit_count += int(raw.get("gold_hit_query_count", 0))
    if indexes != set(range(batch_count)):
        raise ValueError("frozen source shard coverage is incomplete")
    if (
        query_count != int(source_manifest.get("query_count", -1))
        or candidate_count != int(source_manifest.get("candidate_count", -1))
        or gold_hit_count != int(source_manifest.get("gold_hit_query_count", -1))
    ):
        raise ValueError("frozen source shard aggregate counts mismatch")

    overlay_by_query_id: dict[str, FrozenCandidateOverlayEntry] = {}
    overlay = source.get("candidate_overlay")
    if not isinstance(overlay, dict) or overlay.get("mode") not in {
        "none",
        "append-unique-hard-constraint-only",
    }:
        raise ValueError("frozen candidate overlay descriptor is invalid")
    if overlay.get("mode") != "none":
        overlay_path = Path(str(overlay.get("path", "")))
        overlay_bytes = overlay_path.read_bytes()
        overlay_manifest_path = Path(str(overlay.get("manifest_path", "")))
        overlay_manifest_bytes = overlay_manifest_path.read_bytes()
        if overlay.get("sha256") != _sha256(overlay_bytes):
            raise ValueError("frozen candidate overlay hash mismatch")
        if overlay.get("manifest_sha256") != _sha256(overlay_manifest_bytes):
            raise ValueError("frozen candidate overlay manifest hash mismatch")
        overlay_manifest = json.loads(overlay_manifest_bytes)
        if not isinstance(overlay_manifest, dict) or overlay_manifest.get(
            "schema_version"
        ) != "pasa-negation-hard-constraint-overlay-v1":
            raise ValueError("unsupported frozen candidate overlay schema")
        if overlay_manifest.get("test_partition_touched") is not False:
            raise ValueError("frozen candidate overlay does not prove test isolation")
        overlay_outputs = overlay_manifest.get("outputs")
        if not isinstance(overlay_outputs, dict) or overlay_outputs.get(
            "overlay_rows_sha256"
        ) != _sha256(overlay_bytes):
            raise ValueError("frozen candidate overlay receipt hash mismatch")
        expected_ids = set(expected_query_ids)
        for line in overlay_bytes.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("frozen candidate overlay row is invalid")
            query_id = str(row.get("query_id", ""))
            if query_id not in expected_ids or query_id in overlay_by_query_id:
                raise ValueError("frozen candidate overlay query id is invalid")
            candidates = tuple(
                DocumentCandidateEvidence.model_validate(value)
                for value in row.get("appended_candidates", [])
            )
            if any(
                PASA_TRAINING_CONSTRAINT_INJECTED_SOURCE
                not in candidate.paper.sources
                for candidate in candidates
            ):
                raise ValueError("candidate overlay lacks the hard-constraint-only marker")
            overlay_by_query_id[query_id] = FrozenCandidateOverlayEntry(
                query_sha256=str(row.get("query_sha256", "")),
                candidates=candidates,
            )

    return FrozenFusionActivationInputs(
        manifest_path=manifest_path.resolve(),
        manifest_sha256=_sha256(manifest_bytes),
        shard_dir=shard_dir.resolve(),
        shard_manifest_sha256=_sha256(source_manifest_bytes),
        batch_count=batch_count,
        task_labels_path=task_path.resolve(),
        constraint_labels_path=constraint_path.resolve(),
        overlay_by_query_id=overlay_by_query_id,
    )


def read_query_shard(path: Path) -> list[DocumentRankingQuery]:
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    queries = [
        DocumentRankingQuery.model_validate_json(line)
        for line in payload.splitlines()
        if line.strip()
    ]
    if not queries:
        raise ValueError("training query shard is empty")
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("training query shard ids must be unique")
    return queries


def write_fusion_checkpoint(
    directory: Path, checkpoint: FusionTrainingCheckpoint
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        **{  # type: ignore[arg-type]
            family: np.asarray(weights, dtype="<f8")
            for family, weights in sorted(checkpoint.weights.items())
        },
    )
    weights_payload = buffer.getvalue()
    manifest = {
        "schema_version": "large-scale-fusion-checkpoint-v2",
        "input_sha256": checkpoint.input_sha256,
        "epoch_index": checkpoint.epoch_index,
        "next_batch_index": checkpoint.next_batch_index,
        "batch_count": checkpoint.batch_count,
        "pair_counts": dict(sorted(checkpoint.pair_counts.items())),
        "query_counts": dict(sorted(checkpoint.query_counts.items())),
        "replay_pair_counts": dict(sorted(checkpoint.replay_pair_counts.items())),
        "replay_query_counts": dict(sorted(checkpoint.replay_query_counts.items())),
        "families": sorted(checkpoint.weights),
        "dimension_by_family": {
            family: int(np.asarray(weights).size)
            for family, weights in sorted(checkpoint.weights.items())
        },
        "weights_sha256": _sha256(weights_payload),
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    weights_tmp = directory / "weights.npz.tmp"
    manifest_tmp = directory / "checkpoint.json.tmp"
    weights_tmp.write_bytes(weights_payload)
    manifest_tmp.write_bytes(manifest_payload)
    weights_tmp.replace(directory / "weights.npz")
    manifest_tmp.replace(directory / "checkpoint.json")


def read_fusion_checkpoint(directory: Path) -> FusionTrainingCheckpoint:
    manifest = _json_object(directory / "checkpoint.json", label="checkpoint")
    if manifest.get("schema_version") not in {
        "large-scale-fusion-checkpoint-v1",
        "large-scale-fusion-checkpoint-v2",
    }:
        raise ValueError("unsupported fusion checkpoint schema")
    weights_payload = (directory / "weights.npz").read_bytes()
    if manifest.get("weights_sha256") != _sha256(weights_payload):
        raise ValueError("fusion checkpoint weights hash mismatch")
    with np.load(io.BytesIO(weights_payload), allow_pickle=False) as archive:
        weights = {
            family: np.asarray(archive[family], dtype=np.float64).copy()
            for family in archive.files
        }
    if sorted(weights) != manifest.get("families"):
        raise ValueError("fusion checkpoint family mismatch")
    dimensions = manifest.get("dimension_by_family")
    if not isinstance(dimensions, dict) or any(
        int(dimensions.get(family, -1)) != vector.size
        for family, vector in weights.items()
    ):
        raise ValueError("fusion checkpoint dimension mismatch")
    return FusionTrainingCheckpoint(
        input_sha256=str(manifest["input_sha256"]),
        epoch_index=int(manifest["epoch_index"]),
        next_batch_index=int(manifest["next_batch_index"]),
        batch_count=int(manifest["batch_count"]),
        pair_counts={
            str(name): int(value)
            for name, value in dict(manifest["pair_counts"]).items()
        },
        query_counts={
            str(name): int(value)
            for name, value in dict(manifest["query_counts"]).items()
        },
        weights=weights,
        replay_pair_counts={
            str(name): int(value)
            for name, value in dict(manifest.get("replay_pair_counts", {})).items()
        },
        replay_query_counts={
            str(name): int(value)
            for name, value in dict(manifest.get("replay_query_counts", {})).items()
        },
    )


__all__ = [
    "build_document_ranking_query",
    "FusionTrainingCheckpoint",
    "FusionTrainingPackage",
    "FrozenCandidateOverlayEntry",
    "FrozenFusionActivationInputs",
    "apply_frozen_candidate_overlay",
    "load_frozen_fusion_activation_inputs",
    "ordered_query_ids_sha256",
    "index_training_receipts",
    "load_training_package",
    "query_has_gold_candidate",
    "read_query_shard",
    "read_fusion_checkpoint",
    "unpack_production_bundle_labels",
    "unpack_production_bundle_components",
    "with_context_label_files",
    "write_query_shard",
    "write_fusion_checkpoint",
]
