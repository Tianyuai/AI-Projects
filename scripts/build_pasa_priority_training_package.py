"""Build a checkpointed leakage-safe PASA package for dataset/task targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from paper_search.learning.candidates import query_content_terms
from paper_search.learning.f5_production_deployment import (
    load_f5_production_ranker_bytes,
)
from paper_search.learning.gated_feature_fusion_ranker import (
    UnifiedFusionContextResolver,
)
from paper_search.learning.pasa_priority_training_package import (
    build_mixed_pasa_candidates,
    build_unified_context_freeze_rows,
    validate_priority_queue_rows,
)
from paper_search.learning.pasa_training_augmentation import (
    write_pasa_augmented_handoff,
    write_pasa_supplement_receipt,
)
from paper_search.learning.query_constraint_annotations import (
    FrozenConstraintAnnotation,
    FrozenConstraintProfileStore,
)
from paper_search.learning.task_slot_document_ranker import FrozenTaskSlotLabelStore
from paper_search.retrieval.pasa_paper_database import PasaPaperDatabase


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_or_verify(path: Path, payload: bytes, *, label: str) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"immutable {label} differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _checkpoint_valid(path: Path, output_dir: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    receipts = payload.get("receipt_sha256")
    if not isinstance(receipts, dict):
        raise ValueError(f"invalid PASA checkpoint: {path}")
    for relative, expected in receipts.items():
        receipt = output_dir / str(relative)
        if not receipt.is_file() or _sha256_file(receipt) != expected:
            raise ValueError(f"PASA checkpoint receipt mismatch: {receipt}")
    return payload


def _freeze_context(
    *,
    output_dir: Path,
    strict_ids: Sequence[str],
    partition: Mapping[str, Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    constraint_rows: Sequence[Mapping[str, Any]],
    resolver: UnifiedFusionContextResolver,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    context_dir = output_dir / "unified-context"
    manifest_path = context_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest["outputs"].items():
            if name.endswith("_sha256"):
                filename = {
                    "task_labels_sha256": "task-labels.jsonl",
                    "constraint_labels_sha256": "constraint-labels.jsonl",
                }[name]
                if _sha256_file(context_dir / filename) != expected:
                    raise ValueError("unified context checkpoint hash mismatch")
        return manifest

    task_by_id = {str(row["query_id"]): row for row in task_rows}
    constraint_by_id = {str(row["query_id"]): row for row in constraint_rows}
    local_profiles = {
        query_id: resolver.for_local_query(str(partition[query_id]["query"])).constraint_profile
        for query_id in strict_ids
    }
    if any(profile is None for profile in local_profiles.values()):
        raise ValueError("production local resolver returned an empty constraint profile")
    frozen_tasks, frozen_constraints, summary = build_unified_context_freeze_rows(
        strict_ready_query_ids=strict_ids,
        partition_rows=partition,
        task_rows_by_query=task_by_id,
        constraint_rows_by_query=constraint_by_id,
        local_profiles_by_query={
            query_id: profile
            for query_id, profile in local_profiles.items()
            if profile is not None
        },
    )
    task_bytes = _jsonl_bytes(frozen_tasks)
    constraint_bytes = _jsonl_bytes(frozen_constraints)
    FrozenTaskSlotLabelStore.from_jsonl_bytes(task_bytes)
    annotations = [
        FrozenConstraintAnnotation.model_validate(row) for row in frozen_constraints
    ]
    FrozenConstraintProfileStore(annotations)
    _write_or_verify(
        context_dir / "task-labels.jsonl", task_bytes, label="unified task labels"
    )
    _write_or_verify(
        context_dir / "constraint-labels.jsonl",
        constraint_bytes,
        label="unified constraint labels",
    )
    manifest = {
        "schema_version": "pasa-priority-unified-training-context-v1",
        **summary,
        "resolver_id": resolver.resolver_id,
        "resolver_sha256": resolver.resolver_sha256,
        "inputs": dict(sorted(input_hashes.items())),
        "outputs": {
            "task_labels_sha256": _sha256_bytes(task_bytes),
            "constraint_labels_sha256": _sha256_bytes(constraint_bytes),
        },
        "online_requests_made": 0,
        "llm_requests_made": 0,
        "production_lock_modified": False,
    }
    _write_or_verify(
        manifest_path, _canonical_json(manifest), label="unified context manifest"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-queue", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--task-labels", type=Path, required=True)
    parser.add_argument("--constraint-labels", type=Path, required=True)
    parser.add_argument("--production-manifest", type=Path, required=True)
    parser.add_argument("--production-bundle", type=Path, required=True)
    parser.add_argument("--pasa-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=6242)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--search-limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if min(
        args.expected_count,
        args.batch_size,
        args.search_limit,
        args.workers,
    ) <= 0:
        raise ValueError("package sizes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partition_rows = _read_jsonl(args.partition)
    partition = {str(row["query_id"]): row for row in partition_rows}
    handoff = json.loads(args.handoff.read_text(encoding="utf-8"))
    strict_ids = [str(value) for value in handoff["cumulative_unique_ready_query_ids"]]
    if len(strict_ids) != len(set(strict_ids)):
        raise ValueError("strict-ready handoff contains duplicate query ids")
    priority_rows = validate_priority_queue_rows(
        _read_jsonl(args.priority_queue),
        strict_ready_query_ids=set(strict_ids),
        expected_count=args.expected_count,
    )
    if any(
        partition[query_id].get("role") != "training"
        or partition[query_id].get("split") != "auto_train"
        for query_id in strict_ids
    ):
        raise ValueError("PASA package permits auto_train rows only")

    input_hashes = {
        "priority_queue_sha256": _sha256_file(args.priority_queue),
        "partition_sha256": _sha256_file(args.partition),
        "handoff_sha256": _sha256_file(args.handoff),
        "task_labels_sha256": _sha256_file(args.task_labels),
        "constraint_labels_sha256": _sha256_file(args.constraint_labels),
        "production_manifest_sha256": _sha256_file(args.production_manifest),
        "production_bundle_sha256": _sha256_file(args.production_bundle),
    }
    ranker = load_f5_production_ranker_bytes(
        args.production_manifest.read_bytes(), args.production_bundle.read_bytes()
    )
    resolver = ranker.context_store
    if not isinstance(resolver, UnifiedFusionContextResolver):
        raise ValueError("production ranker lacks the unified local resolver")
    context_manifest = _freeze_context(
        output_dir=args.output_dir,
        strict_ids=strict_ids,
        partition=partition,
        task_rows=_read_jsonl(args.task_labels),
        constraint_rows=_read_jsonl(args.constraint_labels),
        resolver=resolver,
        input_hashes=input_hashes,
    )

    pasa = PasaPaperDatabase(args.pasa_index)
    input_hashes["pasa_index_sha256"] = pasa.index_sha256
    gold_ids = [
        str(gold_id)
        for target in priority_rows
        for gold_id in partition[str(target["query_id"])]["gold_paper_ids"]
    ]
    gold_lookup = pasa.lookup_arxiv_many(gold_ids)
    receipts_root = args.output_dir / "pasa-mixed-receipts"
    checkpoints_dir = args.output_dir / "checkpoints"
    batch_count = (len(priority_rows) + args.batch_size - 1) // args.batch_size
    checkpoint_rows: list[dict[str, Any]] = []

    def build_one(target: Mapping[str, Any]) -> dict[str, Any]:
        query_id = str(target["query_id"])
        row = partition[query_id]
        query = str(row["query"])
        terms = query_content_terms(query)[:8]
        lexical_query = " ".join(terms) if terms else query
        lexical = pasa.search(lexical_query, args.search_limit)
        papers, audit = build_mixed_pasa_candidates(
            lexical_papers=lexical,
            gold_paper_ids=[str(value) for value in row["gold_paper_ids"]],
            gold_lookup=gold_lookup,
        )
        return {
            "query_id": query_id,
            "query": query,
            "papers": papers,
            "audit": audit,
            "eligible_signals": list(target["eligible_signals"]),
        }

    for batch_index in range(batch_count):
        checkpoint_path = checkpoints_dir / f"batch-{batch_index:05d}.json"
        completed = _checkpoint_valid(checkpoint_path, args.output_dir)
        if completed is not None:
            checkpoint_rows.append(completed)
            print(
                json.dumps(
                    {
                        "batch": batch_index + 1,
                        "batch_count": batch_count,
                        "status": "reused_checkpoint",
                        "processed_query_count": min(
                            len(priority_rows), (batch_index + 1) * args.batch_size
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        start = batch_index * args.batch_size
        targets = priority_rows[start : start + args.batch_size]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(build_one, targets))
        receipt_hashes: dict[str, str] = {}
        audits: list[dict[str, Any]] = []
        for result in results:
            paths = write_pasa_supplement_receipt(
                output_root=receipts_root,
                query_id=str(result["query_id"]),
                query=str(result["query"]),
                papers=result["papers"],
                index_sha256=pasa.index_sha256,
                mixed_candidate_audit=result["audit"],
            )
            for receipt_path in paths.values():
                relative = receipt_path.relative_to(args.output_dir).as_posix()
                receipt_hashes[relative] = _sha256_file(receipt_path)
            audits.append(
                {
                    "query_id": result["query_id"],
                    "eligible_signals": result["eligible_signals"],
                    **result["audit"],
                }
            )
        checkpoint = {
            "schema_version": "pasa-priority-batch-checkpoint-v1",
            "batch_index": batch_index,
            "query_ids": [str(target["query_id"]) for target in targets],
            "query_count": len(targets),
            "receipt_sha256": dict(sorted(receipt_hashes.items())),
            "candidate_audit": audits,
            "mixed_positive_negative_query_count": sum(
                bool(row["mixed_positive_negative"]) for row in audits
            ),
            "online_requests_made": 0,
            "llm_requests_made": 0,
            "test_partition_touched": False,
        }
        _write_or_verify(
            checkpoint_path,
            _canonical_json(checkpoint),
            label="PASA batch checkpoint",
        )
        checkpoint_rows.append(checkpoint)
        print(
            json.dumps(
                {
                    "batch": batch_index + 1,
                    "batch_count": batch_count,
                    "status": "completed",
                    "processed_query_count": start + len(targets),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    augmented_handoff = args.output_dir / "ranking-training-handoff-openalex-pasa.json"
    write_pasa_augmented_handoff(
        base_handoff_path=args.handoff,
        supplement_root=receipts_root,
        output_path=augmented_handoff,
        supplement_query_count=len(priority_rows),
        index_sha256=pasa.index_sha256,
    )
    audits = [
        audit
        for checkpoint in checkpoint_rows
        for audit in checkpoint["candidate_audit"]
    ]
    signal_counts: Counter[str] = Counter()
    for audit in audits:
        signal_counts.update(str(value) for value in audit["eligible_signals"])
    checkpoint_digest = _sha256_bytes(
        _canonical_json(
            {
                "checkpoints": {
                    f"batch-{int(row['batch_index']):05d}.json": _sha256_file(
                        checkpoints_dir / f"batch-{int(row['batch_index']):05d}.json"
                    )
                    for row in checkpoint_rows
                }
            }
        )
    )
    manifest = {
        "schema_version": "pasa-priority-training-candidate-package-v1",
        "strict_ready_query_count": len(strict_ids),
        "priority_query_count": len(priority_rows),
        "batch_size": args.batch_size,
        "batch_count": batch_count,
        "search_limit": args.search_limit,
        "workers": args.workers,
        "signal_query_count": dict(sorted(signal_counts.items())),
        "candidate_count": sum(int(row["supplement_candidate_count"]) for row in audits),
        "positive_candidate_count": sum(
            int(row["positive_candidate_count"]) for row in audits
        ),
        "lexical_negative_candidate_count": sum(
            int(row["lexical_negative_candidate_count"]) for row in audits
        ),
        "mixed_positive_negative_query_count": sum(
            bool(row["mixed_positive_negative"]) for row in audits
        ),
        "source_label_leakage_guard": {
            "candidate_policy": "mixed_lexical_plus_gold_training",
            "positive_and_negative_share_action": True,
            "gold_only_receipt_count": 0,
        },
        "inputs": dict(sorted(input_hashes.items())),
        "outputs": {
            "augmented_handoff": str(augmented_handoff.resolve()),
            "augmented_handoff_sha256": _sha256_file(augmented_handoff),
            "receipt_root": str(receipts_root.resolve()),
            "checkpoint_digest": checkpoint_digest,
            "unified_context_manifest": str(
                (args.output_dir / "unified-context" / "manifest.json").resolve()
            ),
            "unified_context_manifest_sha256": _sha256_bytes(
                _canonical_json(context_manifest)
            ),
        },
        "online_requests_made": 0,
        "llm_requests_made": 0,
        "training_started": False,
        "production_lock_modified": False,
        "test_partition_touched": False,
    }
    _write_or_verify(
        args.output_dir / "manifest.json",
        _canonical_json(manifest),
        label="PASA priority package manifest",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
