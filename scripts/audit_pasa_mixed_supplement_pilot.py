"""Run a bounded offline PASA lexical+Gold supplement pilot without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from paper_search.learning.cpu_document_ranker import (
    DocumentRankingQuery,
    build_document_candidates,
)
from paper_search.learning.f5_production_deployment import (
    load_f5_production_ranker_bytes,
)
from paper_search.learning.fusion_activation import audit_fusion_query_activation
from paper_search.learning.gated_feature_fusion_ranker import (
    UnifiedFusionContextResolver,
)
from paper_search.learning.negation_evidence import NEGATION_EVIDENCE_SCHEMA_VERSION
from paper_search.retrieval.pasa_paper_database import (
    PasaPaperDatabase,
    build_pasa_training_supplement,
)


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


def _sample_key(row: Mapping[str, Any]) -> tuple[int, str]:
    signals = set(row.get("eligible_signals", []))
    if signals.intersection({"method", "dataset", "year", "negation"}):
        priority = 0
    elif "task_provenance" in signals:
        priority = 1
    else:
        priority = 2
    digest = hashlib.sha256(str(row["query_id"]).encode("utf-8")).hexdigest()
    return priority, digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--pasa-index", type=Path, required=True)
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--production-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--search-limit", type=int, default=100)
    parser.add_argument("--required-signal")
    args = parser.parse_args(argv)
    if min(args.sample_size, args.search_limit) <= 0:
        raise ValueError("pilot sizes must be positive")

    partition = {
        str(row["query_id"]): row for row in _read_jsonl(args.partition)
    }
    targets = [
        row
        for row in _read_jsonl(args.queue)
        if row.get("recommended_action")
        == "pasa_mixed_lexical_gold_supplement"
        and (
            args.required_signal is None
            or args.required_signal in row.get("eligible_signals", [])
        )
    ]
    selected = sorted(targets, key=_sample_key)[: args.sample_size]
    pasa = PasaPaperDatabase(args.pasa_index)
    if (args.production_manifest is None) != (args.production_bundle is None):
        raise ValueError("production manifest and bundle must be supplied together")
    ranker = None
    local_resolver = None
    if args.production_manifest is not None and args.production_bundle is not None:
        ranker = load_f5_production_ranker_bytes(
            args.production_manifest.read_bytes(), args.production_bundle.read_bytes()
        )
        if not isinstance(ranker.context_store, UnifiedFusionContextResolver):
            raise ValueError("production ranker lacks the unified local resolver")
        local_resolver = ranker.context_store
    rows: list[dict[str, Any]] = []
    for target in selected:
        query_id = str(target["query_id"])
        partition_row = partition[query_id]
        if (
            partition_row.get("role") != "training"
            or partition_row.get("split") != "auto_train"
        ):
            raise ValueError("PASA pilot permits auto_train rows only")
        local_context = (
            local_resolver.for_local_query(str(partition_row["query"]))
            if local_resolver is not None
            else None
        )
        profile = local_context.constraint_profile if local_context is not None else None
        papers, supplement = build_pasa_training_supplement(
            database=pasa,
            query=str(partition_row["query"]),
            gold_paper_ids=[str(value) for value in partition_row["gold_paper_ids"]],
            search_limit=args.search_limit,
            negative_exclusions=profile.exclusions if profile is not None else (),
        )
        available_gold_count = int(target["pasa_available_gold_count"])
        direct_gold_count = int(supplement["direct_gold_candidate_count"])
        lexical_gold_count = max(0, available_gold_count - direct_gold_count)
        lexical_count = int(supplement["lexical_candidate_count"])
        lexical_negative_count = max(0, lexical_count - lexical_gold_count)
        output_row: dict[str, Any] = {
                "query_id": query_id,
                "eligible_signals": list(target.get("eligible_signals", [])),
                "base_candidate_count": int(target["base_candidate_count"]),
                "base_hard_negative_candidate_count": int(
                    target["base_hard_negative_candidate_count"]
                ),
                "pasa_available_gold_count": available_gold_count,
                "pasa_lexical_candidate_count": lexical_count,
                "pasa_lexical_gold_count": lexical_gold_count,
                "pasa_lexical_negative_count": lexical_negative_count,
                "pasa_direct_appended_gold_count": direct_gold_count,
                "mixed_action_has_positive_and_negative": (
                    available_gold_count > 0 and lexical_negative_count > 0
                ),
            }
        if ranker is not None and local_resolver is not None:
            activation = audit_fusion_query_activation(
                DocumentRankingQuery(
                    query_id=query_id,
                    query=str(partition_row["query"]),
                    gold_paper_ids=[
                        str(value) for value in partition_row["gold_paper_ids"]
                    ],
                    candidates=build_document_candidates(
                        [("pasa-mixed-training-supplement", papers)]
                    ),
                ),
                ranker,
                context=local_context,
            )
            output_row["family_activation"] = {
                family: {
                    "gate_eligible": bool(values["gate_eligible"]),
                    "effective_pair_count": int(values["effective_pair_count"]),
                    "signal_effective_pair_count": dict(
                        values.get("signal_effective_pair_count", {})
                    ),
                    "reason": str(values["reason"]),
                }
                for family, values in activation["families"].items()
            }
        rows.append(output_row)

    signal_counts: Counter[str] = Counter()
    mixed_signal_counts: Counter[str] = Counter()
    for row in rows:
        for signal in row["eligible_signals"]:
            signal_counts[str(signal)] += 1
            if row["mixed_action_has_positive_and_negative"]:
                mixed_signal_counts[str(signal)] += 1
    activation_coverage: dict[str, dict[str, int]] = {}
    signal_activation: Counter[str] = Counter()
    if ranker is not None:
        for family in sorted(ranker.feature_families):
            family_rows = [
                row["family_activation"][family]
                for row in rows
                if "family_activation" in row
            ]
            activation_coverage[family] = {
                "eligible_query_count": sum(
                    bool(values["gate_eligible"]) for values in family_rows
                ),
                "effective_query_count": sum(
                    int(values["effective_pair_count"]) > 0 for values in family_rows
                ),
                "effective_pair_count": sum(
                    int(values["effective_pair_count"]) for values in family_rows
                ),
            }
            for values in family_rows:
                signal_activation.update(
                    {
                        str(signal): int(count)
                        for signal, count in values[
                            "signal_effective_pair_count"
                        ].items()
                    }
                )
    summary = {
        "schema_version": "pasa-mixed-training-supplement-pilot-v2",
        "negation_evidence_schema_version": NEGATION_EVIDENCE_SCHEMA_VERSION,
        "target_query_count": len(targets),
        "sample_query_count": len(rows),
        "search_limit": args.search_limit,
        "mixed_positive_negative_action_query_count": sum(
            bool(row["mixed_action_has_positive_and_negative"]) for row in rows
        ),
        "natural_lexical_gold_hit_query_count": sum(
            int(row["pasa_lexical_gold_count"]) > 0 for row in rows
        ),
        "direct_gold_append_query_count": sum(
            int(row["pasa_direct_appended_gold_count"]) > 0 for row in rows
        ),
        "pasa_lexical_candidate_count": sum(
            int(row["pasa_lexical_candidate_count"]) for row in rows
        ),
        "pasa_lexical_negative_count": sum(
            int(row["pasa_lexical_negative_count"]) for row in rows
        ),
        "pasa_available_gold_count": sum(
            int(row["pasa_available_gold_count"]) for row in rows
        ),
        "signal_sample_query_count": dict(sorted(signal_counts.items())),
        "signal_mixed_action_query_count": dict(sorted(mixed_signal_counts.items())),
        "family_activation_on_pasa_mixed_candidates": activation_coverage,
        "signal_effective_pair_count": dict(sorted(signal_activation.items())),
        "source_label_leakage_guard": {
            "gold_only_action_used": False,
            "positive_and_negative_share_pasa_action": True,
        },
        "pasa_index_sha256": pasa.index_sha256,
        "network_request_count": 0,
        "llm_request_count": 0,
        "training_started": False,
        "production_lock_modified": False,
        "test_partition_touched": False,
    }
    _write_atomic(
        args.output,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    _write_atomic(
        args.rows_output,
        b"".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
            + b"\n"
            for row in rows
        ),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
