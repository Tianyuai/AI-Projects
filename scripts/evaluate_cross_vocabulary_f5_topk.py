"""Compare frozen original and supplemental OpenAlex pools under one production F5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from paper_search.evaluation.predictions import (
    paper_evaluation_aliases,
    paper_evaluation_id,
    paper_id_aliases,
)
from paper_search.evaluation.dataset import IdentifierMap
from paper_search.learning.cpu_document_ranker import (
    DocumentCandidateEvidence,
)
from paper_search.learning.f5_production_deployment import (
    load_f5_production_ranker_bytes,
)
from paper_search.learning.large_scale_fusion_training import (
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
)
from paper_search.learning.openalex_daily_schedule import (
    load_settled_search_action_identities,
    search_action_identity,
)
from paper_search.processing.deduplicate import deduplicate_papers
from paper_search.recall_experiments.contracts import RecallActionBatch
try:
    from scripts.run_cross_vocabulary_openalex_validation import (
        DEFAULT_HANDOFF,
        DEFAULT_PARTITION,
        _load_jsonl,
        _online_only_package,
        _verify_frozen_package,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from run_cross_vocabulary_openalex_validation import (
        DEFAULT_HANDOFF,
        DEFAULT_PARTITION,
        _load_jsonl,
        _online_only_package,
        _verify_frozen_package,
    )


DEFAULT_VALIDATION_ROOT = Path(
    "data/training_private/recall_policy/contrastive-openalex-bridge-nu128-v2"
)
DEFAULT_SELECTION = Path("artifacts/models/production-document-ranker-selection.json")
DEFAULT_OUTPUT_NAME = "f5-topk-candidate-ab-v1.json"
DEFAULT_CUTOFFS = (5, 10, 20, 50)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _number(value: object, *, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be numeric")
    return float(cast(int | float, value))


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def ranking_metrics(
    gold_paper_ids: Sequence[str],
    ranked: Sequence[DocumentCandidateEvidence],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    identifier_map: IdentifierMap | None = None,
) -> dict[str, object]:
    """Compute binary paper-ranking metrics using all scorer-facing aliases."""

    if not gold_paper_ids:
        raise ValueError("ranking metrics require at least one Gold paper")
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("ranking cutoffs must be positive")
    def resolved(values: Sequence[str] | frozenset[str]) -> set[str]:
        if identifier_map is None:
            return set(values)
        return {identifier_map.resolve(value) for value in values}

    gold_groups: list[set[str]] = []
    for identifier in gold_paper_ids:
        aliases = resolved(paper_id_aliases(identifier))
        overlapping = [
            index
            for index, group in enumerate(gold_groups)
            if group.intersection(aliases)
        ]
        if not overlapping:
            gold_groups.append(aliases)
            continue
        first = overlapping[0]
        gold_groups[first].update(aliases)
        for index in reversed(overlapping[1:]):
            gold_groups[first].update(gold_groups.pop(index))

    first_rank_by_gold: dict[int, int] = {}
    for rank, candidate in enumerate(ranked, start=1):
        candidate_aliases = resolved(paper_evaluation_aliases(candidate.paper))
        for index, group in enumerate(gold_groups):
            if index in first_rank_by_gold or not group.intersection(candidate_aliases):
                continue
            first_rank_by_gold[index] = rank
    gold_ranks = sorted(set(first_rank_by_gold.values()))
    gold_count = len(gold_groups)
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank in gold_ranks if rank <= 10
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(gold_count, 10) + 1)
    )
    output: dict[str, object] = {
        "gold_ranks": gold_ranks,
        "mrr": 1.0 / gold_ranks[0] if gold_ranks else 0.0,
        "ndcg_at_10": dcg / ideal,
    }
    for cutoff in cutoffs:
        hits = sum(rank <= cutoff for rank in first_rank_by_gold.values())
        output[f"hit_at_{cutoff}"] = float(hits > 0)
        output[f"recall_at_{cutoff}"] = min(hits, gold_count) / gold_count
    return output


def audit_candidate_pool_monotonicity(
    baseline: Sequence[DocumentCandidateEvidence],
    augmented: Sequence[DocumentCandidateEvidence],
) -> dict[str, int]:
    """Prove that adding the sealed action did not remove a baseline paper."""

    augmented_aliases = [paper_evaluation_aliases(row.paper) for row in augmented]
    missing: list[str] = []
    for row in baseline:
        aliases = paper_evaluation_aliases(row.paper)
        if any(aliases.intersection(other) for other in augmented_aliases):
            continue
        if any(
            set(row.source_ranks.items()).intersection(other.source_ranks.items())
            for other in augmented
        ):
            continue
        if any(
            len(deduplicate_papers([row.paper, other.paper]).papers) == 1
            for other in augmented
        ):
            continue
        missing.append(row.paper.canonical_id)
    if missing:
        raise ValueError(f"augmented pool lost baseline candidate: {missing[0]}")
    if len(augmented) < len(baseline):
        raise ValueError("augmented pool is smaller than the baseline pool")
    return {
        "baseline_candidate_count": len(baseline),
        "missing_member_count": 0,
    }


def _arm_metrics(
    rows: Sequence[Mapping[str, object]], arm: str, metric_names: Sequence[str]
) -> dict[str, float]:
    return {
        metric: sum(
            _number(
                cast(Mapping[str, object], row[arm]).get(metric),
                label=f"{arm} {metric}",
            )
            for row in rows
        )
        / len(rows)
        for metric in metric_names
    }


def aggregate_comparison(
    rows: Sequence[Mapping[str, object]],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, object]:
    """Macro-aggregate one candidate-pool A/B slice."""

    if not rows:
        raise ValueError("candidate-pool comparison slice cannot be empty")
    metric_names = [
        "mrr",
        "ndcg_at_10",
        *(f"hit_at_{cutoff}" for cutoff in cutoffs),
        *(f"recall_at_{cutoff}" for cutoff in cutoffs),
    ]
    baseline = _arm_metrics(rows, "baseline", metric_names)
    augmented = _arm_metrics(rows, "augmented", metric_names)
    baseline_hits = [
        bool(cast(list[object], cast(Mapping[str, object], row["baseline"])["gold_ranks"]))
        for row in rows
    ]
    augmented_hits = [
        bool(cast(list[object], cast(Mapping[str, object], row["augmented"])["gold_ranks"]))
        for row in rows
    ]
    both_first_rank_shifts = [
        _integer(
            cast(list[object], cast(Mapping[str, object], row["baseline"])["gold_ranks"])[0],
            label="baseline first Gold rank",
        )
        - _integer(
            cast(list[object], cast(Mapping[str, object], row["augmented"])["gold_ranks"])[0],
            label="augmented first Gold rank",
        )
        for row, baseline_hit, augmented_hit in zip(
            rows, baseline_hits, augmented_hits, strict=True
        )
        if baseline_hit and augmented_hit
    ]
    direction: dict[str, object] = {}
    for cutoff in cutoffs:
        metric = f"recall_at_{cutoff}"
        deltas = [
            _number(
                cast(Mapping[str, object], row["augmented"]).get(metric),
                label=f"augmented {metric}",
            )
            - _number(
                cast(Mapping[str, object], row["baseline"]).get(metric),
                label=f"baseline {metric}",
            )
            for row in rows
        ]
        direction[f"direction_at_{cutoff}"] = {
            "improved_query_count": sum(delta > 0.0 for delta in deltas),
            "worsened_query_count": sum(delta < 0.0 for delta in deltas),
            "unchanged_query_count": sum(delta == 0.0 for delta in deltas),
        }
    added_counts = [
        _integer(
            row.get("augmented_candidate_count"),
            label="augmented candidate count",
        )
        - _integer(
            row.get("baseline_candidate_count"),
            label="baseline candidate count",
        )
        for row in rows
    ]
    return {
        "query_count": len(rows),
        "candidate_pool": {
            "baseline_candidate_count": sum(
                _integer(
                    row.get("baseline_candidate_count"),
                    label="baseline candidate count",
                )
                for row in rows
            ),
            "augmented_candidate_count": sum(
                _integer(
                    row.get("augmented_candidate_count"),
                    label="augmented candidate count",
                )
                for row in rows
            ),
            "added_candidate_count": sum(added_counts),
            "mean_added_candidate_count": sum(added_counts) / len(rows),
            "baseline_gold_hit_query_count": sum(baseline_hits),
            "augmented_gold_hit_query_count": sum(augmented_hits),
            "gold_hit_promotions": sum(
                not before and after
                for before, after in zip(baseline_hits, augmented_hits, strict=True)
            ),
            "gold_hit_regressions": sum(
                before and not after
                for before, after in zip(baseline_hits, augmented_hits, strict=True)
            ),
        },
        "baseline": baseline,
        "augmented": augmented,
        "delta": {
            metric: augmented[metric] - baseline[metric] for metric in metric_names
        },
        "first_gold_rank_when_both_hit": {
            "query_count": len(both_first_rank_shifts),
            "improved_query_count": sum(value > 0 for value in both_first_rank_shifts),
            "worsened_query_count": sum(value < 0 for value in both_first_rank_shifts),
            "unchanged_query_count": sum(value == 0 for value in both_first_rank_shifts),
            "mean_rank_gain": (
                sum(both_first_rank_shifts) / len(both_first_rank_shifts)
                if both_first_rank_shifts
                else 0.0
            ),
            "median_rank_gain": (
                float(statistics.median(both_first_rank_shifts))
                if both_first_rank_shifts
                else 0.0
            ),
        },
        **direction,
    }


def _load_production_paths(selection_path: Path) -> tuple[Path, Path, dict[str, object]]:
    selection_raw = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection_raw, dict):
        raise ValueError("production selection must be a JSON object")
    selection = cast(dict[str, object], selection_raw)
    if (
        selection.get("production_default") != "F5-gated-fusion"
        or selection.get("per_query_model_switching") is not False
        or selection.get("test_partition_touched") is not False
    ):
        raise ValueError("production selection does not bind the expected F5 policy")
    manifest_value = selection.get("default_manifest")
    weights_value = selection.get("default_weights")
    if not isinstance(manifest_value, str) or not isinstance(weights_value, str):
        raise ValueError("production selection paths are invalid")
    manifest_path = (selection_path.parent / manifest_value).resolve()
    weights_path = (selection_path.parent / weights_value).resolve()
    if _sha256(manifest_path.read_bytes()) != selection.get("default_manifest_sha256"):
        raise ValueError("production F5 manifest hash mismatch")
    if _sha256(weights_path.read_bytes()) != selection.get("default_weights_sha256"):
        raise ValueError("production F5 weights hash mismatch")
    return manifest_path, weights_path, selection


def _supplemental_receipts(
    validation_root: Path,
    query_ids: Sequence[str],
    actions: Mapping[str, object],
) -> dict[str, tuple[Path, ...]]:
    receipts_root = (validation_root / "receipts").resolve()
    selected = set(query_ids)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(receipts_root.rglob("retrieval/attempt-01/*.json")):
        if path.stem in selected:
            grouped[path.stem].append(path.resolve())
    completed = load_settled_search_action_identities([receipts_root])
    for query_id in query_ids:
        raw_batch = actions.get(query_id)
        batch = RecallActionBatch.model_validate(raw_batch)
        if len(batch.actions) != 1:
            raise ValueError(f"supplemental action count changed: {query_id}")
        identity = search_action_identity(batch.actions[0].model_dump(mode="json"))
        if identity is None or identity not in completed.get(query_id, frozenset()):
            raise ValueError(f"supplemental action is not settled: {query_id}")
        if query_id not in grouped:
            raise ValueError(f"supplemental receipt is missing: {query_id}")
    return {query_id: tuple(grouped[query_id]) for query_id in query_ids}


def _baseline_snapshot_rows(validation_root: Path) -> dict[str, dict[str, object]]:
    rows = _load_jsonl(validation_root / "baseline-candidates.jsonl")
    indexed = {str(row["query_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("frozen baseline candidate rows are duplicated")
    return indexed


def _signal_rows(validation_root: Path) -> dict[str, str]:
    rows = _load_jsonl(validation_root / "proposal-diagnostics.jsonl")
    indexed = {str(row["query_id"]): str(row["signal"]) for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("frozen proposal diagnostics are duplicated")
    return indexed


def _verify_baseline_snapshot(
    query_id: str,
    candidates: Sequence[DocumentCandidateEvidence],
    snapshot: Mapping[str, object],
) -> None:
    aliases = sorted(
        {
            alias
            for candidate in candidates
            for alias in paper_evaluation_aliases(candidate.paper)
        }
    )
    expected_aliases = snapshot.get("candidate_aliases")
    if snapshot.get("candidate_count") != len(candidates) or expected_aliases != aliases:
        raise ValueError(f"reconstructed baseline candidate pool drifted: {query_id}")


def run(args: argparse.Namespace) -> dict[str, object]:
    workspace_root = Path(args.workspace_root).resolve()
    validation_root = _resolve(workspace_root, args.validation_root)
    handoff_path = _resolve(workspace_root, args.handoff)
    partition_path = _resolve(workspace_root, args.partition)
    selection_path = _resolve(workspace_root, args.production_selection)
    identifier_map_path = (
        _resolve(workspace_root, args.identifier_map)
        if args.identifier_map is not None
        else None
    )
    identifier_map = (
        IdentifierMap.from_path(identifier_map_path)
        if identifier_map_path is not None
        else None
    )
    output_path = (
        _resolve(workspace_root, args.output)
        if args.output is not None
        else validation_root / DEFAULT_OUTPUT_NAME
    )
    non_reinforcing_supplement = bool(
        getattr(args, "non_reinforcing_supplement", False)
    )

    validation_manifest, partition_rows, actions = _verify_frozen_package(
        validation_root
    )
    query_ids = tuple(str(row["query_id"]) for row in partition_rows)
    if len(query_ids) != 128:
        raise ValueError("this comparison requires the frozen 128-query partition")
    manifest_path, weights_path, selection = _load_production_paths(selection_path)
    inputs = cast(Mapping[str, object], validation_manifest.get("inputs"))
    for label, path in (
        ("handoff_sha256", handoff_path),
        ("partition_sha256", partition_path),
        ("production_bundle_sha256", weights_path),
    ):
        if inputs.get(label) != _sha256(path.read_bytes()):
            raise ValueError(f"frozen validation input hash mismatch: {label}")

    package = load_training_package(
        handoff_path=handoff_path,
        partition_path=partition_path,
        production_bundle_path=weights_path,
    )
    package = _online_only_package(package)
    selected_package = replace(
        package,
        query_ids=query_ids,
        rows_by_query_id={query_id: package.rows_by_query_id[query_id] for query_id in query_ids},
    )
    baseline_paths = index_training_receipts(selected_package)
    supplemental_paths = _supplemental_receipts(
        validation_root,
        query_ids,
        actions,
    )
    baseline_snapshots = _baseline_snapshot_rows(validation_root)
    signals = _signal_rows(validation_root)
    if set(baseline_snapshots) != set(query_ids) or set(signals) != set(query_ids):
        raise ValueError("frozen candidate or signal coverage is incomplete")

    ranker = load_f5_production_ranker_bytes(
        manifest_path.read_bytes(),
        weights_path.read_bytes(),
    )
    supplemental_root = (validation_root / "receipts").resolve()
    per_query: list[dict[str, object]] = []
    for index, query_id in enumerate(query_ids, start=1):
        baseline_query = build_document_ranking_query(
            selected_package,
            query_id,
            baseline_paths[query_id],
        )
        _verify_baseline_snapshot(
            query_id,
            baseline_query.candidates,
            baseline_snapshots[query_id],
        )
        augmented_query = build_document_ranking_query(
            selected_package,
            query_id,
            baseline_paths[query_id] + supplemental_paths[query_id],
            additive_receipt_roots=(supplemental_root,),
            non_reinforcing_additive=non_reinforcing_supplement,
        )
        audit_candidate_pool_monotonicity(
            baseline_query.candidates,
            augmented_query.candidates,
        )
        ranked_baseline = ranker.rank(
            baseline_query.query,
            baseline_query.candidates,
        )
        ranked_augmented = ranker.rank(
            augmented_query.query,
            augmented_query.candidates,
        )
        baseline_metrics = ranking_metrics(
            baseline_query.gold_paper_ids,
            ranked_baseline,
            identifier_map=identifier_map,
        )
        augmented_metrics = ranking_metrics(
            augmented_query.gold_paper_ids,
            ranked_augmented,
            identifier_map=identifier_map,
        )
        per_query.append(
            {
                "query_index": index,
                "query_id": query_id,
                "signal": signals[query_id],
                "gold_count": len(baseline_query.gold_paper_ids),
                "baseline_candidate_count": len(baseline_query.candidates),
                "augmented_candidate_count": len(augmented_query.candidates),
                "added_candidate_count": (
                    len(augmented_query.candidates) - len(baseline_query.candidates)
                ),
                "baseline": baseline_metrics,
                "augmented": augmented_metrics,
                "baseline_top_50": [
                    paper_evaluation_id(candidate.paper)
                    for candidate in ranked_baseline[:50]
                ],
                "augmented_top_50": [
                    paper_evaluation_id(candidate.paper)
                    for candidate in ranked_augmented[:50]
                ],
                "top_50_sequence_changed": [
                    paper_evaluation_id(candidate.paper)
                    for candidate in ranked_baseline[:50]
                ]
                != [
                    paper_evaluation_id(candidate.paper)
                    for candidate in ranked_augmented[:50]
                ],
            }
        )

    by_signal = {
        signal: aggregate_comparison(
            [row for row in per_query if row["signal"] == signal]
        )
        for signal in sorted(set(signals.values()))
    }
    result: dict[str, object] = {
        "schema_version": (
            "production-f5-cross-vocabulary-candidate-ab-identity-nonreinforcing-v3"
            if identifier_map is not None and non_reinforcing_supplement
            else (
                "production-f5-cross-vocabulary-candidate-ab-identity-v2"
                if identifier_map is not None
                else (
                    "production-f5-cross-vocabulary-candidate-ab-nonreinforcing-v2"
                    if non_reinforcing_supplement
                    else "production-f5-cross-vocabulary-candidate-ab-v1"
                )
            )
        ),
        "comparison": {
            "baseline": "original-openalex-only-candidate-pool+production-F5",
            "augmented": "original-plus-sealed-cross-vocabulary-pool+same-production-F5",
            "candidate_pool_only_intervention": True,
            "same_ranker_both_arms": True,
            "verified_identifier_map_applied": identifier_map is not None,
            "supplemental_overlap_policy": (
                "append-new-members-preserve-baseline-evidence"
                if non_reinforcing_supplement
                else "same-provider-action-reinforcement"
            ),
        },
        "query_count": len(per_query),
        "cutoffs": list(DEFAULT_CUTOFFS),
        "overall": aggregate_comparison(per_query),
        "by_signal": by_signal,
        "top_50_sequence_changed_query_count": sum(
            bool(row["top_50_sequence_changed"]) for row in per_query
        ),
        "inputs": {
            "frozen_validation_manifest_sha256": _sha256(
                (validation_root / "manifest.json").read_bytes()
            ),
            "frozen_partition_sha256": _sha256(
                (validation_root / "partition.jsonl").read_bytes()
            ),
            "frozen_actions_sha256": _sha256(
                (validation_root / "actions.json").read_bytes()
            ),
            "collection_status_sha256": _sha256(
                (validation_root / "collection-status.json").read_bytes()
            ),
            "paired_candidate_evaluation_sha256": _sha256(
                (validation_root / "paired-evaluation-result-v1.json").read_bytes()
            ),
            "training_handoff_sha256": _sha256(handoff_path.read_bytes()),
            "training_partition_sha256": _sha256(partition_path.read_bytes()),
            "production_selection_sha256": _sha256(selection_path.read_bytes()),
            "production_manifest_sha256": _sha256(manifest_path.read_bytes()),
            "production_weights_sha256": _sha256(weights_path.read_bytes()),
            "production_default": selection["production_default"],
            "production_training_query_count": selection["training_query_count"],
            **(
                {"verified_identifier_map_sha256": _sha256(identifier_map_path.read_bytes())}
                if identifier_map_path is not None
                else {}
            ),
        },
        "safety": {
            "online_requests_made": 0,
            "llm_requests_made": 0,
            "training_started": False,
            "test_partition_touched": False,
            "production_lock_modified": False,
            "candidate_membership_monotonic": True,
            "baseline_evidence_immutable": non_reinforcing_supplement,
        },
        "per_query": per_query,
    }
    payload = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_immutable(output_path, payload)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument(
        "--production-selection",
        type=Path,
        default=DEFAULT_SELECTION,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--identifier-map", type=Path)
    parser.add_argument(
        "--non-reinforcing-supplement",
        action="store_true",
        help="append supplemental-only papers without changing baseline evidence",
    )
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    summary = {
        "query_count": result["query_count"],
        "overall": result["overall"],
        "by_signal": result["by_signal"],
        "top_50_sequence_changed_query_count": result[
            "top_50_sequence_changed_query_count"
        ],
        "safety": result["safety"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
