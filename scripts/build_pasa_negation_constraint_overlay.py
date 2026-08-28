"""Build and audit an offline PASA overlay used only by hard-constraint training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.evaluation.predictions import (  # noqa: E402
    paper_evaluation_aliases,
    paper_matches_evaluation_ids,
)
from paper_search.learning.cpu_document_ranker import (  # noqa: E402
    DocumentCandidateEvidence,
    DocumentRankingQuery,
    build_document_candidates,
)
from paper_search.learning.f5_production_deployment import (  # noqa: E402
    load_f5_production_ranker_bytes,
)
from paper_search.learning.fusion_activation import (  # noqa: E402
    audit_fusion_query_activation,
)
from paper_search.learning.gated_feature_fusion_ranker import (  # noqa: E402
    FrozenFusionContextStore,
    UnifiedFusionContextResolver,
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
    PasaPaperDatabase,
    build_pasa_negation_training_supplement,
)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite immutable overlay: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _append_unique_candidates(
    base: Sequence[DocumentCandidateEvidence],
    supplement: Sequence[DocumentCandidateEvidence],
) -> tuple[list[DocumentCandidateEvidence], list[DocumentCandidateEvidence]]:
    merged = list(base)
    appended: list[DocumentCandidateEvidence] = []
    aliases = {
        alias for candidate in base for alias in paper_evaluation_aliases(candidate.paper)
    }
    for candidate in supplement:
        candidate_aliases = paper_evaluation_aliases(candidate.paper)
        if aliases.intersection(candidate_aliases):
            continue
        merged.append(candidate)
        appended.append(candidate)
        aliases.update(candidate_aliases)
    return merged, appended


def _query_gold_hit(query: DocumentRankingQuery) -> bool:
    gold = set(query.gold_paper_ids)
    return any(
        paper_matches_evaluation_ids(candidate.paper, gold)
        for candidate in query.candidates
    )


def build_overlay(
    *,
    queue_path: Path,
    shard_dir: Path,
    task_labels_path: Path,
    constraint_labels_path: Path,
    pasa_index_path: Path,
    production_manifest_path: Path,
    production_bundle_path: Path,
    output_dir: Path,
    constraint_negative_limit: int,
) -> dict[str, object]:
    if constraint_negative_limit <= 0:
        raise ValueError("constraint negative limit must be positive")
    queue_bytes = queue_path.read_bytes()
    queue_rows = [
        row for row in _read_jsonl(queue_path) if "negation" in row.get("signals", [])
    ]
    targets = {str(row["query_id"]): row for row in queue_rows}
    if len(targets) != len(queue_rows):
        raise ValueError("negation overlay target ids must be unique")

    task_bytes = task_labels_path.read_bytes()
    constraint_bytes = constraint_labels_path.read_bytes()
    task_store = FrozenTaskSlotLabelStore.from_jsonl_bytes(task_bytes)
    constraint_store = FrozenConstraintProfileStore(
        [
            FrozenConstraintAnnotation.model_validate_json(line)
            for line in constraint_bytes.splitlines()
            if line.strip()
        ]
    )
    context_store = FrozenFusionContextStore(
        task_store=task_store, constraint_store=constraint_store
    )
    production_manifest_bytes = production_manifest_path.read_bytes()
    production_bundle_bytes = production_bundle_path.read_bytes()
    ranker = load_f5_production_ranker_bytes(
        production_manifest_bytes, production_bundle_bytes
    )
    local_resolver = ranker.context_store
    if not isinstance(local_resolver, UnifiedFusionContextResolver):
        raise ValueError("production ranker lacks the unified local resolver")
    ranker.context_store = context_store
    ranker.feature_families = frozenset({"hard_constraint"})
    ranker.max_pairs_per_query_family = 32
    ranker.constraint_text_evidence = True
    pasa = PasaPaperDatabase(pasa_index_path)

    shard_manifest_path = shard_dir / "manifest.json"
    shard_manifest_bytes = shard_manifest_path.read_bytes()
    shard_manifest = json.loads(shard_manifest_bytes)
    completed = {
        int(row["batch_index"]): row
        for row in cast(
            Sequence[Mapping[str, object]], shard_manifest["completed_shards"]
        )
    }
    rows: list[dict[str, object]] = []
    found_targets: set[str] = set()
    for batch_index in range(int(shard_manifest["batch_count"])):
        shard_path = shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
        shard_bytes = shard_path.read_bytes()
        if _sha256(shard_bytes) != completed[batch_index].get("sha256"):
            raise ValueError(f"query shard hash mismatch: {shard_path}")
        with gzip.GzipFile(fileobj=io.BytesIO(shard_bytes), mode="rb") as stream:
            for raw_line in stream:
                if not raw_line.strip():
                    continue
                raw = json.loads(raw_line)
                query_id = str(raw["query_id"])
                if query_id not in targets:
                    continue
                found_targets.add(query_id)
                query = DocumentRankingQuery.model_validate(raw)
                frozen_context = context_store.for_training_query(query.query)
                frozen_profile = frozen_context.constraint_profile
                context = (
                    frozen_context
                    if frozen_profile is not None
                    and "negation" in frozen_profile.labels
                    and frozen_profile.exclusions
                    else local_resolver.for_local_query(query.query)
                )
                profile = context.constraint_profile
                if (
                    profile is None
                    or "negation" not in profile.labels
                    or not profile.exclusions
                ):
                    raise ValueError(f"negation target lacks frozen exclusions: {query_id}")
                papers, supplement_audit = build_pasa_negation_training_supplement(
                    database=pasa,
                    query=query.query,
                    gold_paper_ids=query.gold_paper_ids,
                    negative_exclusions=profile.exclusions,
                    constraint_negative_limit=constraint_negative_limit,
                )
                supplement_candidates = build_document_candidates(
                    [("pasa-negation-constraint-overlay", papers)]
                )
                merged_candidates, appended = _append_unique_candidates(
                    query.candidates, supplement_candidates
                )
                expanded = query.model_copy(update={"candidates": merged_candidates})
                activation = audit_fusion_query_activation(
                    expanded, ranker, context=context
                )
                hard = cast(
                    Mapping[str, Mapping[str, object]], activation["families"]
                )["hard_constraint"]
                signal_counts = cast(
                    Mapping[str, int], hard["signal_effective_pair_count"]
                )
                rows.append(
                    {
                        "query_id": query_id,
                        "query_sha256": activation["query_sha256"],
                        "base_candidate_count": len(query.candidates),
                        "base_gold_hit": _query_gold_hit(query),
                        "overlay_candidate_count": len(appended),
                        "expanded_candidate_count": len(merged_candidates),
                        "expanded_gold_hit": _query_gold_hit(expanded),
                        "constraint_negative_candidate_count": int(
                            supplement_audit["constraint_negative_candidate_count"]
                        ),
                        "direct_gold_candidate_count": int(
                            supplement_audit["direct_gold_candidate_count"]
                        ),
                        "negation_effective_pair_count": int(
                            signal_counts.get("negation", 0)
                        ),
                        "selected_pair_evidence_sha256": hard[
                            "selected_pair_evidence_sha256"
                        ],
                        "appended_candidates": [
                            candidate.model_dump(
                                mode="json",
                                exclude_none=True,
                                exclude_computed_fields=True,
                            )
                            for candidate in appended
                        ],
                    }
                )
        if (batch_index + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "event": "pasa_negation_overlay_progress",
                        "batches_complete": batch_index + 1,
                        "targets_complete": len(found_targets),
                        "target_count": len(targets),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    if found_targets != set(targets):
        raise ValueError("not every negation target was found in query shards")
    rows.sort(key=lambda row: str(row["query_id"]))
    rows_bytes = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )
    effective_rows = [
        row for row in rows if int(row["negation_effective_pair_count"]) > 0
    ]
    report: dict[str, object] = {
        "schema_version": "pasa-negation-hard-constraint-overlay-v1",
        "negation_evidence_schema_version": NEGATION_EVIDENCE_SCHEMA_VERSION,
        "target_query_count": len(rows),
        "base_gold_hit_query_count": sum(bool(row["base_gold_hit"]) for row in rows),
        "expanded_gold_hit_query_count": sum(
            bool(row["expanded_gold_hit"]) for row in rows
        ),
        "overlay_candidate_count": sum(
            int(row["overlay_candidate_count"]) for row in rows
        ),
        "constraint_negative_candidate_count": sum(
            int(row["constraint_negative_candidate_count"]) for row in rows
        ),
        "direct_gold_candidate_count": sum(
            int(row["direct_gold_candidate_count"]) for row in rows
        ),
        "effective_query_count": len(effective_rows),
        "effective_pair_count": sum(
            int(row["negation_effective_pair_count"]) for row in effective_rows
        ),
        "selected_query_ids": [str(row["query_id"]) for row in effective_rows],
        "source_feature_policy": "hard-constraint-only-candidates-v1",
        "inputs": {
            "queue_sha256": _sha256(queue_bytes),
            "task_labels_sha256": _sha256(task_bytes),
            "constraint_labels_sha256": _sha256(constraint_bytes),
            "query_shard_manifest_sha256": _sha256(shard_manifest_bytes),
            "production_manifest_sha256": _sha256(production_manifest_bytes),
            "production_bundle_sha256": _sha256(production_bundle_bytes),
            "pasa_index_sha256": pasa.index_sha256,
        },
        "outputs": {"overlay_rows_sha256": _sha256(rows_bytes)},
        "network_request_count": 0,
        "llm_request_count": 0,
        "training_started": False,
        "production_lock_modified": False,
        "development_labels_used_for_training": False,
        "test_partition_touched": False,
    }
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_immutable(output_dir / "overlay-candidates.jsonl", rows_bytes)
    _write_immutable(output_dir / "manifest.json", report_bytes)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--task-labels", type=Path, required=True)
    parser.add_argument("--constraint-labels", type=Path, required=True)
    parser.add_argument("--pasa-index", type=Path, required=True)
    parser.add_argument("--production-manifest", type=Path, required=True)
    parser.add_argument("--production-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--constraint-negative-limit", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_overlay(
        queue_path=args.queue,
        shard_dir=args.shard_dir,
        task_labels_path=args.task_labels,
        constraint_labels_path=args.constraint_labels,
        pasa_index_path=args.pasa_index,
        production_manifest_path=args.production_manifest,
        production_bundle_path=args.production_bundle,
        output_dir=args.output_dir,
        constraint_negative_limit=args.constraint_negative_limit,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
