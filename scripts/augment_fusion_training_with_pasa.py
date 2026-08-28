"""Build checkpointed OpenAlex+PASA fusion shards and compare effective pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from paper_search.evaluation.predictions import paper_matches_evaluation_ids
from paper_search.learning.f5_production_deployment import (
    load_f5_production_ranker_bytes,
)
from paper_search.learning.fusion_activation import audit_fusion_query_activation
from paper_search.learning.gated_feature_fusion_ranker import (
    FUSION_FAMILIES,
    FusionQueryContext,
    GatedFeatureFusionRanker,
    gated_family_eligibility,
)
from paper_search.learning.large_scale_fusion_training import (
    FusionTrainingPackage,
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
    write_query_shard,
)
from paper_search.learning.pasa_training_augmentation import (
    write_pasa_augmented_handoff,
    write_pasa_supplement_receipt,
)
from paper_search.retrieval.pasa_paper_database import (
    PasaPaperDatabase,
    build_pasa_training_supplement,
    mark_pasa_training_gold_injected,
)


_SIGNAL_FAMILY = {
    "task_provenance": "task_provenance",
    "method": "entity",
    "dataset": "entity",
    "year": "hard_constraint",
    "negation": "hard_constraint",
}


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    payload = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )
    _write_atomic(path, payload)


def _candidate_profile(query: Any) -> dict[str, int]:
    positive = sum(
        paper_matches_evaluation_ids(candidate.paper, query.gold_paper_ids)
        for candidate in query.candidates
    )
    negative = len(query.candidates) - positive
    return {
        "candidate_count": len(query.candidates),
        "positive_candidate_count": positive,
        "hard_negative_candidate_count": min(negative, 100),
    }


def _signal_rows(
    context: FusionQueryContext,
    activation: Mapping[str, object],
) -> dict[str, dict[str, int | bool]]:
    families = cast(Mapping[str, Mapping[str, object]], activation["families"])
    profile = context.constraint_profile
    labels = set(profile.labels) if profile is not None else set()
    task = context.task_label
    output: dict[str, dict[str, int | bool]] = {}
    for signal, family in _SIGNAL_FAMILY.items():
        if signal == "task_provenance":
            present = task is not None and task.reliability_weight > 0.0
        else:
            present = signal in labels
        family_row = families[family]
        eligible = present and bool(family_row["gate_eligible"])
        signal_pair_counts = cast(
            Mapping[str, int], family_row.get("signal_effective_pair_count", {})
        )
        pair_count = (
            int(family_row["effective_pair_count"])
            if signal == "task_provenance" and eligible
            else int(signal_pair_counts.get(signal, 0))
            if eligible
            else 0
        )
        output[signal] = {
            "eligible": eligible,
            "effective": pair_count > 0,
            "effective_pair_count": pair_count,
        }
    return output


def _query_audit(
    query: Any,
    ranker: GatedFeatureFusionRanker,
) -> dict[str, object]:
    profile = _candidate_profile(query)
    context = ranker.context_store.for_training_query(query.query)
    pair_feasible = (
        profile["positive_candidate_count"] > 0
        and profile["hard_negative_candidate_count"] > 0
    )
    eligibility = {
        family: gated_family_eligibility(context, family, gated=ranker.gated)
        for family in ranker.feature_families
    }
    requires_constraint_features = any(
        eligibility.get(family, False) for family in ("entity", "hard_constraint")
    )
    if pair_feasible and requires_constraint_features:
        activation = audit_fusion_query_activation(query, ranker, context=context)
    else:
        activation = {
            "families": {
                family: {
                    "gate_eligible": eligibility[family],
                    "effective_pair_count": 0,
                }
                for family in ranker.feature_families
            }
        }
    if pair_feasible:
        simple_pair_count = profile["positive_candidate_count"] * min(
            profile["hard_negative_candidate_count"], ranker.hard_negative_limit
        )
        if ranker.max_pairs_per_query_family is not None:
            simple_pair_count = min(
                simple_pair_count, ranker.max_pairs_per_query_family
            )
        families = cast(dict[str, dict[str, object]], activation["families"])
        for family in ("reliability", "task_provenance"):
            if eligibility.get(family, False):
                families[family]["effective_pair_count"] = simple_pair_count
    signals = _signal_rows(context, activation)
    families = cast(Mapping[str, Mapping[str, object]], activation["families"])
    return {
        "query_id": query.query_id,
        **profile,
        "has_positive_and_hard_negative": (
            profile["positive_candidate_count"] > 0
            and profile["hard_negative_candidate_count"] > 0
        ),
        "family_gate_eligible": {
            family: bool(families[family]["gate_eligible"])
            for family in sorted(FUSION_FAMILIES)
        },
        "family_effective_pair_count": {
            family: int(families[family]["effective_pair_count"])
            for family in sorted(FUSION_FAMILIES)
        },
        "signals": signals,
    }


def _supplement_decision(
    audit: Mapping[str, object],
    *,
    shallow_candidate_threshold: int,
    minimum_hard_negatives: int,
) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    if int(audit["positive_candidate_count"]) == 0:
        reasons.append("missing_gold_positive")
        signals = cast(Mapping[str, Mapping[str, object]], audit["signals"])
        for signal in ("method", "dataset", "year", "negation"):
            if bool(signals[signal]["eligible"]):
                reasons.append(f"ineffective_gate:{signal}")
        return ("lexical" if len(reasons) > 1 else "gold_only"), reasons
    if int(audit["hard_negative_candidate_count"]) < minimum_hard_negatives:
        reasons.append("missing_hard_negative")
    if int(audit["candidate_count"]) < shallow_candidate_threshold:
        reasons.append("shallow_candidate_pool")
    signals = cast(Mapping[str, Mapping[str, object]], audit["signals"])
    for signal in ("method", "dataset", "year", "negation"):
        if bool(signals[signal]["eligible"]) and not bool(
            signals[signal]["effective"]
        ):
            reasons.append(f"ineffective_gate:{signal}")
    return ("lexical", reasons) if reasons else (None, [])


def _completed_batch(shard: Path, audit: Path) -> bool:
    return shard.is_file() and audit.is_file()


def _process_phase(
    *,
    phase: str,
    package: FusionTrainingPackage,
    ranker: GatedFeatureFusionRanker,
    output_dir: Path,
    batch_size: int,
    pasa_database: PasaPaperDatabase | None = None,
    pasa_gold_lookup: Mapping[str, Any] | None = None,
    pasa_receipt_root: Path | None = None,
    search_limit: int = 100,
    shallow_candidate_threshold: int = 50,
    minimum_hard_negatives: int = 20,
) -> None:
    indexed = index_training_receipts(package)
    batch_count = (len(package.query_ids) + batch_size - 1) // batch_size
    shard_dir = output_dir / f"{phase}-shards"
    audit_dir = output_dir / f"{phase}-audit-batches"
    for batch_index in range(batch_count):
        start = batch_index * batch_size
        query_ids = package.query_ids[start : start + batch_size]
        shard_path = shard_dir / f"shard-{batch_index:05d}.jsonl.gz"
        audit_path = audit_dir / f"batch-{batch_index:05d}.jsonl"
        if _completed_batch(shard_path, audit_path):
            continue
        queries = []
        audit_rows: list[dict[str, object]] = []
        for query_id in query_ids:
            query = build_document_ranking_query(package, query_id, indexed[query_id])
            row = _query_audit(query, ranker)
            if pasa_database is not None:
                if pasa_receipt_root is None:
                    raise ValueError("PASA receipt root is required for base phase")
                mode, reasons = _supplement_decision(
                    row,
                    shallow_candidate_threshold=shallow_candidate_threshold,
                    minimum_hard_negatives=minimum_hard_negatives,
                )
                row["pasa_supplement_mode"] = mode
                row["pasa_supplement_reasons"] = reasons
                if mode is not None:
                    if mode == "gold_only":
                        lookup = pasa_gold_lookup or {}
                        papers = []
                        seen_paper_ids: set[str] = set()
                        for gold_id in query.gold_paper_ids:
                            paper = lookup.get(gold_id)
                            if paper is None or paper.canonical_id in seen_paper_ids:
                                continue
                            seen_paper_ids.add(paper.canonical_id)
                            papers.append(mark_pasa_training_gold_injected(paper))
                        supplement = {
                            "lexical_candidate_count": 0,
                            "direct_gold_candidate_count": len(papers),
                            "supplement_candidate_count": len(papers),
                        }
                    else:
                        context = ranker.context_store.for_training_query(query.query)
                        exclusions = (
                            context.constraint_profile.exclusions
                            if context.constraint_profile is not None
                            else []
                        )
                        papers, supplement = build_pasa_training_supplement(
                            database=pasa_database,
                            query=query.query,
                            gold_paper_ids=query.gold_paper_ids,
                            search_limit=search_limit,
                            negative_exclusions=exclusions,
                        )
                    if papers:
                        write_pasa_supplement_receipt(
                            output_root=pasa_receipt_root,
                            query_id=query.query_id,
                            query=query.query,
                            papers=papers,
                            index_sha256=pasa_database.index_sha256,
                        )
                    row["pasa_supplement"] = supplement
                else:
                    row["pasa_supplement"] = {
                        "lexical_candidate_count": 0,
                        "direct_gold_candidate_count": 0,
                        "supplement_candidate_count": 0,
                    }
            queries.append(query)
            audit_rows.append(row)
        write_query_shard(shard_path, queries)
        _write_jsonl_atomic(audit_path, audit_rows)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "completed_batch": batch_index + 1,
                    "batch_count": batch_count,
                    "processed_query_count": start + len(query_ids),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    manifest = {
        "schema_version": "pasa-augmented-query-shards-v1",
        "phase": phase,
        "query_count": len(package.query_ids),
        "batch_size": batch_size,
        "batch_count": batch_count,
        "package_input_sha256": package.input_sha256,
        "shard_sha256": {
            path.name: _sha256(path.read_bytes())
            for path in sorted(shard_dir.glob("*.jsonl.gz"))
        },
        "audit_sha256": {
            path.name: _sha256(path.read_bytes())
            for path in sorted(audit_dir.glob("*.jsonl"))
        },
        "online_requests_made": 0,
        "llm_requests_made": 0,
        "test_partition_touched": False,
    }
    _write_atomic(output_dir / f"{phase}-shard-manifest.json", _canonical_json(manifest))


def _audit_rows(output_dir: Path, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / f"{phase}-audit-batches").glob("*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _coverage(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    family_eligible: Counter[str] = Counter()
    family_effective: Counter[str] = Counter()
    family_pairs: Counter[str] = Counter()
    signal_eligible: Counter[str] = Counter()
    signal_effective: Counter[str] = Counter()
    signal_pairs: Counter[str] = Counter()
    for row in rows:
        eligible = cast(Mapping[str, bool], row["family_gate_eligible"])
        pairs = cast(Mapping[str, int], row["family_effective_pair_count"])
        for family in FUSION_FAMILIES:
            family_eligible[family] += bool(eligible[family])
            family_effective[family] += int(pairs[family]) > 0
            family_pairs[family] += int(pairs[family])
        signals = cast(Mapping[str, Mapping[str, object]], row["signals"])
        for signal in _SIGNAL_FAMILY:
            signal_eligible[signal] += bool(signals[signal]["eligible"])
            signal_effective[signal] += bool(signals[signal]["effective"])
            signal_pairs[signal] += int(signals[signal]["effective_pair_count"])
    return {
        "query_count": len(rows),
        "candidate_count": sum(int(row["candidate_count"]) for row in rows),
        "gold_hit_query_count": sum(
            int(row["positive_candidate_count"]) > 0 for row in rows
        ),
        "positive_candidate_count": sum(
            int(row["positive_candidate_count"]) for row in rows
        ),
        "hard_negative_candidate_count_capped100": sum(
            int(row["hard_negative_candidate_count"]) for row in rows
        ),
        "positive_and_hard_negative_query_count": sum(
            bool(row["has_positive_and_hard_negative"]) for row in rows
        ),
        "shallow_candidate_query_count": sum(
            int(row["candidate_count"]) < 50 for row in rows
        ),
        "family": {
            family: {
                "eligible_query_count": family_eligible[family],
                "effective_query_count": family_effective[family],
                "effective_pair_count": family_pairs[family],
            }
            for family in sorted(FUSION_FAMILIES)
        },
        "signals": {
            signal: {
                "eligible_query_count": signal_eligible[signal],
                "effective_query_count": signal_effective[signal],
                "effective_pair_count": signal_pairs[signal],
            }
            for signal in _SIGNAL_FAMILY
        },
    }


def _delta(before: object, after: object) -> object:
    if isinstance(before, dict) and isinstance(after, dict):
        return {key: _delta(before[key], after[key]) for key in before if key in after}
    if isinstance(before, int) and isinstance(after, int):
        return after - before
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--production-manifest", type=Path, required=True)
    parser.add_argument("--production-bundle", type=Path, required=True)
    parser.add_argument("--pasa-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--search-limit", type=int, default=100)
    parser.add_argument("--shallow-candidate-threshold", type=int, default=50)
    parser.add_argument("--minimum-hard-negatives", type=int, default=20)
    parser.add_argument("--max-pairs-per-query-family", type=int, default=32)
    args = parser.parse_args(argv)
    if min(
        args.batch_size,
        args.search_limit,
        args.shallow_candidate_threshold,
        args.minimum_hard_negatives,
        args.max_pairs_per_query_family,
    ) <= 0:
        raise ValueError("augmentation sizes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_training_package(
        handoff_path=args.handoff,
        partition_path=args.partition,
        production_bundle_path=args.production_bundle,
    )
    ranker = load_f5_production_ranker_bytes(
        args.production_manifest.read_bytes(), args.production_bundle.read_bytes()
    )
    ranker.max_pairs_per_query_family = args.max_pairs_per_query_family
    pasa = PasaPaperDatabase(args.pasa_index)
    pasa_gold_lookup = pasa.lookup_arxiv_many(
        [
            str(gold_id)
            for query_id in base.query_ids
            for gold_id in base.rows_by_query_id[query_id]["gold_paper_ids"]
        ]
    )
    receipt_root = args.output_dir / "pasa-receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    _process_phase(
        phase="openalex-base",
        package=base,
        ranker=ranker,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        pasa_database=pasa,
        pasa_gold_lookup=pasa_gold_lookup,
        pasa_receipt_root=receipt_root,
        search_limit=args.search_limit,
        shallow_candidate_threshold=args.shallow_candidate_threshold,
        minimum_hard_negatives=args.minimum_hard_negatives,
    )
    base_rows = _audit_rows(args.output_dir, "openalex-base")
    supplement_query_count = sum(
        bool(row.get("pasa_supplement_reasons"))
        and int(cast(Mapping[str, int], row["pasa_supplement"])["supplement_candidate_count"]) > 0
        for row in base_rows
    )
    augmented_handoff_path = args.output_dir / "ranking-training-handoff-openalex-pasa.json"
    write_pasa_augmented_handoff(
        base_handoff_path=args.handoff,
        supplement_root=receipt_root,
        output_path=augmented_handoff_path,
        supplement_query_count=supplement_query_count,
        index_sha256=pasa.index_sha256,
    )
    augmented = load_training_package(
        handoff_path=augmented_handoff_path,
        partition_path=args.partition,
        production_bundle_path=args.production_bundle,
    )
    _process_phase(
        phase="openalex-pasa",
        package=augmented,
        ranker=ranker,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    augmented_rows = _audit_rows(args.output_dir, "openalex-pasa")
    before = _coverage(base_rows)
    after = _coverage(augmented_rows)
    summary = {
        "schema_version": "openalex-pasa-training-coverage-comparison-v1",
        "strict_ready_query_count": len(base.query_ids),
        "strict_ready_ceiling_unchanged": True,
        "base": before,
        "openalex_pasa": after,
        "delta": _delta(before, after),
        "pasa": {
            "index_sha256": pasa.index_sha256,
            "target_query_count": sum(
                bool(row.get("pasa_supplement_reasons")) for row in base_rows
            ),
            "supplemented_query_count": supplement_query_count,
            "direct_gold_candidate_count": sum(
                int(cast(Mapping[str, int], row["pasa_supplement"])["direct_gold_candidate_count"])
                for row in base_rows
            ),
            "lexical_candidate_count": sum(
                int(cast(Mapping[str, int], row["pasa_supplement"])["lexical_candidate_count"])
                for row in base_rows
            ),
            "network_request_count": 0,
            "llm_request_count": 0,
        },
        "production_lock_modified": False,
        "training_started": False,
        "test_partition_touched": False,
    }
    _write_atomic(args.output_dir / "coverage-comparison.json", _canonical_json(summary))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
