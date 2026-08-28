"""Compare strict-ready OpenAlex pairs with prospective offline PASA supplements."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from paper_search.learning.gated_feature_fusion_ranker import (
    FusionQueryContext,
    UnifiedFusionContextResolver,
    gated_family_eligibility,
)
from paper_search.learning.f5_production_deployment import (
    load_f5_production_ranker_bytes,
)
from paper_search.learning.pasa_effective_sample_comparison import (
    compare_effective_sample_coverage,
)
from paper_search.learning.query_constraint_annotations import (
    FrozenConstraintAnnotation,
    FrozenConstraintProfileStore,
)
from paper_search.learning.task_slot_document_ranker import FrozenTaskSlotLabelStore
from paper_search.retrieval.pasa_paper_database import PasaPaperDatabase


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _signal_eligibility(
    *,
    partition: Mapping[str, Mapping[str, Any]],
    strict_ready_query_ids: Sequence[str],
    task_store: FrozenTaskSlotLabelStore,
    constraint_store: FrozenConstraintProfileStore,
    local_constraint_resolver: UnifiedFusionContextResolver | None,
) -> dict[str, dict[str, bool]]:
    output: dict[str, dict[str, bool]] = {}
    for query_id in strict_ready_query_ids:
        row = partition[query_id]
        if row.get("role") != "training" or row.get("split") != "auto_train":
            raise ValueError("strict-ready comparison permits auto_train rows only")
        query = str(row["query"])
        frozen_context = FusionQueryContext(
            task_label=task_store.for_training_query(query),
            constraint_profile=constraint_store.profile_for_query(query),
        )
        constraint_profile = frozen_context.constraint_profile
        if local_constraint_resolver is not None:
            constraint_profile = local_constraint_resolver.for_local_query(
                query
            ).constraint_profile
        context = FusionQueryContext(
            task_label=frozen_context.task_label,
            constraint_profile=constraint_profile,
        )
        profile = context.constraint_profile
        labels = set(profile.labels) if profile is not None else set()
        entity_gate = gated_family_eligibility(context, "entity", gated=True)
        hard_gate = gated_family_eligibility(context, "hard_constraint", gated=True)
        output[query_id] = {
            "task_provenance": gated_family_eligibility(
                context, "task_provenance", gated=True
            ),
            "method": entity_gate and "method" in labels,
            "dataset": entity_gate and "dataset" in labels,
            "year": hard_gate and "year" in labels,
            "negation": hard_gate and "negation" in labels,
        }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-audit", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--task-labels", type=Path, required=True)
    parser.add_argument("--constraint-labels", type=Path, required=True)
    parser.add_argument("--pasa-index", type=Path, required=True)
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--production-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--priority-queue-output", type=Path)
    parser.add_argument("--shallow-candidate-threshold", type=int, default=50)
    parser.add_argument("--minimum-hard-negatives", type=int, default=20)
    parser.add_argument("--hard-negative-limit", type=int, default=100)
    args = parser.parse_args(argv)

    query_rows = _read_jsonl(args.query_audit)
    partition_rows = _read_jsonl(args.partition)
    partition = {str(row["query_id"]): row for row in partition_rows}
    handoff = json.loads(args.handoff.read_text(encoding="utf-8"))
    strict_ids = [str(value) for value in handoff["cumulative_unique_ready_query_ids"]]
    if len(strict_ids) != len(set(strict_ids)):
        raise ValueError("strict-ready handoff contains duplicate query ids")
    missing_partition = set(strict_ids) - set(partition)
    if missing_partition:
        raise ValueError(
            f"strict-ready queries missing from partition: {len(missing_partition)}"
        )

    task_store = FrozenTaskSlotLabelStore.from_jsonl(args.task_labels)
    annotations = [
        FrozenConstraintAnnotation.model_validate(row)
        for row in _read_jsonl(args.constraint_labels)
    ]
    constraint_store = FrozenConstraintProfileStore(annotations)
    if (args.production_manifest is None) != (args.production_bundle is None):
        raise ValueError("production manifest and bundle must be supplied together")
    local_constraint_resolver = None
    if args.production_manifest is not None and args.production_bundle is not None:
        production_ranker = load_f5_production_ranker_bytes(
            args.production_manifest.read_bytes(), args.production_bundle.read_bytes()
        )
        resolver = production_ranker.context_store
        if not isinstance(resolver, UnifiedFusionContextResolver):
            raise ValueError("production ranker lacks the unified local resolver")
        local_constraint_resolver = resolver
    eligibility = _signal_eligibility(
        partition=partition,
        strict_ready_query_ids=strict_ids,
        task_store=task_store,
        constraint_store=constraint_store,
        local_constraint_resolver=local_constraint_resolver,
    )
    gold_ids_by_query = {
        query_id: [str(value) for value in partition[query_id]["gold_paper_ids"]]
        for query_id in strict_ids
    }
    pasa = PasaPaperDatabase(args.pasa_index)
    pasa_gold = pasa.lookup_arxiv_many(
        [gold_id for values in gold_ids_by_query.values() for gold_id in values]
    )
    summary, queue = compare_effective_sample_coverage(
        query_rows=query_rows,
        gold_ids_by_query=gold_ids_by_query,
        strict_ready_query_ids=strict_ids,
        pasa_available_gold_ids=set(pasa_gold),
        signal_eligibility_by_query=eligibility,
        shallow_candidate_threshold=args.shallow_candidate_threshold,
        minimum_hard_negatives=args.minimum_hard_negatives,
        hard_negative_limit=args.hard_negative_limit,
    )
    summary["inputs"] = {
        "query_audit_sha256": _sha256(args.query_audit),
        "partition_sha256": _sha256(args.partition),
        "handoff_sha256": _sha256(args.handoff),
        "task_labels_sha256": _sha256(args.task_labels),
        "constraint_labels_sha256": _sha256(args.constraint_labels),
        "pasa_index_sha256": pasa.index_sha256,
    }
    if args.production_manifest is not None and args.production_bundle is not None:
        summary["inputs"]["production_manifest_sha256"] = _sha256(
            args.production_manifest
        )
        summary["inputs"]["production_bundle_sha256"] = _sha256(args.production_bundle)
    summary["constraint_context_basis"] = (
        "production_unified_local_resolver"
        if local_constraint_resolver is not None
        else "frozen_constraint_labels"
    )
    summary["queue"] = {
        "row_count": len(queue),
        "recommended_action_counts": {
            action: sum(row["recommended_action"] == action for row in queue)
            for action in sorted({str(row["recommended_action"]) for row in queue})
        },
    }
    priority_queue = [
        row
        for row in queue
        if row["recommended_action"] == "pasa_mixed_lexical_gold_supplement"
        and {"dataset", "task_provenance"}.intersection(row["eligible_signals"])
    ]
    summary["recommended_scope"] = {
        "signals": ["dataset", "task_provenance"],
        "query_count": len(priority_queue),
        "full_gold_availability_rescue_count": summary["pasa"][
            "rescued_pair_feasible_query_count"
        ],
        "nonpriority_rescue_deferred_count": (
            int(summary["pasa"]["rescued_pair_feasible_query_count"])
            - len(priority_queue)
        ),
    }
    _write_atomic(
        args.output,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    _write_atomic(
        args.queue_output,
        b"".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
            + b"\n"
            for row in queue
        ),
    )
    if args.priority_queue_output is not None:
        _write_atomic(
            args.priority_queue_output,
            b"".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
                for row in priority_queue
            ),
        )
    print(
        json.dumps(
            {
                "strict_ready_query_count": summary["strict_ready_query_count"],
                "base_pair_feasible": summary["base"][
                    "positive_and_hard_negative_query_count"
                ],
                "openalex_pasa_pair_feasible": summary["openalex_pasa"][
                    "positive_and_hard_negative_query_count"
                ],
                "rescued_pair_feasible": summary["pasa"][
                    "rescued_pair_feasible_query_count"
                ],
                "queue_row_count": len(queue),
                "network_request_count": 0,
                "training_started": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
