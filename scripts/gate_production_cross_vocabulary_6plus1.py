"""Gate bounded production cross-vocabulary recall using sealed local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from paper_search.application.locks import load_verified_input_lock
from paper_search.domain.models import QuerySpec
from paper_search.learning.cross_vocabulary_bridge import (
    select_production_cross_vocabulary_supplement,
)
from paper_search.learning.large_scale_fusion_training import (
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
)
from paper_search.recall_experiments.contracts import RecallActionBatch

try:
    from scripts.evaluate_cross_vocabulary_f5_topk import (
        DEFAULT_HANDOFF,
        DEFAULT_PARTITION,
        DEFAULT_VALIDATION_ROOT,
        _baseline_snapshot_rows,
        _online_only_package,
        _verify_baseline_snapshot,
        _verify_frozen_package,
    )
    from scripts.run_cross_vocabulary_openalex_validation import (
        DEFAULT_CONTEXT_MANIFEST,
        _load_context_signals,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from evaluate_cross_vocabulary_f5_topk import (  # type: ignore[no-redef]
        DEFAULT_HANDOFF,
        DEFAULT_PARTITION,
        DEFAULT_VALIDATION_ROOT,
        _baseline_snapshot_rows,
        _online_only_package,
        _verify_baseline_snapshot,
        _verify_frozen_package,
    )
    from run_cross_vocabulary_openalex_validation import (  # type: ignore[no-redef]
        DEFAULT_CONTEXT_MANIFEST,
        _load_context_signals,
    )


DEFAULT_F5_AB = "f5-topk-candidate-ab-v1.json"
DEFAULT_PRODUCTION_LOCK = Path("deliverables/evaluator/live-evaluator.lock.yaml")
DEFAULT_CANDIDATE_LOCK = Path(
    "deliverables/evaluator/"
    "live-evaluator-unconstrained-6plus1.candidate.lock.yaml"
)
DEFAULT_OUTPUT = "production-6plus1-offline-gate-v1.json"


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def evaluate_f5_evidence(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the already sealed same-F5 candidate-pool comparison."""

    comparison = _object(payload.get("comparison"), label="comparison")
    if comparison.get("candidate_pool_only_intervention") is not True:
        raise ValueError("F5 evidence changed more than the candidate pool")
    if comparison.get("same_ranker_both_arms") is not True:
        raise ValueError("F5 evidence did not use the same ranker")
    by_signal = _object(payload.get("by_signal"), label="by_signal")
    unconstrained = _object(
        by_signal.get("unconstrained"),
        label="unconstrained evidence",
    )
    if _integer(unconstrained.get("query_count"), label="unconstrained query count") != 121:
        raise ValueError("unconstrained evidence must contain exactly 121 queries")
    candidate_pool = _object(
        unconstrained.get("candidate_pool"),
        label="unconstrained candidate pool",
    )
    baseline_hits = _integer(
        candidate_pool.get("baseline_gold_hit_query_count"),
        label="baseline Gold-hit query count",
    )
    augmented_hits = _integer(
        candidate_pool.get("augmented_gold_hit_query_count"),
        label="augmented Gold-hit query count",
    )
    if baseline_hits != 0 or augmented_hits != 18:
        raise ValueError("sealed unconstrained Gold-hit evidence changed")
    if _integer(
        candidate_pool.get("gold_hit_regressions"),
        label="Gold-hit regressions",
    ) != 0:
        raise ValueError("candidate Gold-hit regression detected")

    improved: dict[str, int] = {}
    for cutoff in (5, 10, 20, 50):
        direction = _object(
            unconstrained.get(f"direction_at_{cutoff}"),
            label=f"Top-{cutoff} direction",
        )
        if _integer(
            direction.get("worsened_query_count"),
            label=f"Top-{cutoff} worsened query count",
        ) != 0:
            raise ValueError("Top-K regression detected")
        if cutoff != 5:
            improved[str(cutoff)] = _integer(
                direction.get("improved_query_count"),
                label=f"Top-{cutoff} improved query count",
            )
    if improved != {"10": 3, "20": 9, "50": 13}:
        raise ValueError("sealed unconstrained Top-K evidence changed")

    safety = _object(payload.get("safety"), label="safety")
    expected_safety = {
        "candidate_membership_monotonic": True,
        "llm_requests_made": 0,
        "online_requests_made": 0,
        "production_lock_modified": False,
        "test_partition_touched": False,
        "training_started": False,
    }
    for key, expected in expected_safety.items():
        if safety.get(key) != expected:
            raise ValueError(f"sealed F5 safety field changed: {key}")
    return {
        "passed": True,
        "unconstrained_query_count": 121,
        "unconstrained_gold_hit_query_count": augmented_hits,
        "top_k_improved_query_counts": improved,
        "top_k_worsened_query_count": 0,
        "candidate_membership_monotonic": True,
        "same_production_f5": True,
    }


def _strings(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _query_spec(query: str, raw: Mapping[str, object]) -> QuerySpec:
    return QuerySpec(
        original_query=query,
        research_goal=query,
        methods=_strings(raw.get("methods")),
        tasks=_strings(raw.get("tasks")),
        datasets=_strings(raw.get("datasets")),
        exclusions=_strings(raw.get("exclusions")),
        year_from=cast(int | None, raw.get("year_from")),
        year_to=cast(int | None, raw.get("year_to")),
    )


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


def run(args: argparse.Namespace) -> dict[str, object]:
    workspace_root = Path(args.workspace_root).resolve()
    validation_root = _resolve(workspace_root, args.validation_root)
    handoff_path = _resolve(workspace_root, args.handoff)
    partition_path = _resolve(workspace_root, args.partition)
    context_manifest_path = _resolve(workspace_root, args.context_manifest)
    production_lock_path = _resolve(workspace_root, args.production_lock)
    candidate_lock_path = _resolve(workspace_root, args.candidate_lock)
    output_path = validation_root / DEFAULT_OUTPUT
    ab_path = validation_root / DEFAULT_F5_AB

    production_lock = load_verified_input_lock(
        production_lock_path,
        artifact_root=workspace_root,
    ).lock
    if (
        production_lock.baseline.strategy != "fixed-one-round"
        or production_lock.baseline.cross_vocabulary_supplement is not None
    ):
        raise ValueError("current production lock was unexpectedly modified")
    candidate_lock = load_verified_input_lock(
        candidate_lock_path,
        artifact_root=workspace_root,
    ).lock
    supplement_binding = candidate_lock.baseline.cross_vocabulary_supplement
    if (
        candidate_lock.baseline.strategy != "bounded-two-stage-unconstrained"
        or supplement_binding is None
        or supplement_binding.max_total_openalex_actions != 7
    ):
        raise ValueError("candidate lock does not bind the bounded 6+1 strategy")

    _manifest, partition_rows, frozen_actions = _verify_frozen_package(
        validation_root
    )
    query_ids = tuple(str(row["query_id"]) for row in partition_rows)
    if len(query_ids) != 128 or len(set(query_ids)) != 128:
        raise ValueError("frozen bridge gate requires exactly 128 unique queries")
    package = _online_only_package(
        load_training_package(
            handoff_path=handoff_path,
            partition_path=partition_path,
            production_bundle_path=(
                workspace_root
                / "artifacts/models/"
                "gated-feature-fusion-18314-unified-context-v3-v1/weights.bundle"
            ),
        )
    )
    selected_package = replace(
        package,
        query_ids=query_ids,
        rows_by_query_id={query_id: package.rows_by_query_id[query_id] for query_id in query_ids},
    )
    baseline_paths = index_training_receipts(selected_package)
    baseline_snapshots = _baseline_snapshot_rows(validation_root)
    context_signals = _load_context_signals(package, context_manifest_path)
    diagnostics = {
        str(row["query_id"]): row
        for row in (
            json.loads(line)
            for line in (validation_root / "proposal-diagnostics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        if isinstance(row, dict)
    }

    matched_unconstrained = 0
    strict_abstentions = 0
    for query_id in query_ids:
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
        signal = str(diagnostics[query_id]["signal"])
        actual = select_production_cross_vocabulary_supplement(
            _query_spec(baseline_query.query, context_signals[query_id]),
            baseline_query.candidates,
        )
        expected = RecallActionBatch.model_validate(frozen_actions[query_id])
        if signal == "unconstrained":
            if actual != expected:
                raise ValueError(
                    f"production selector changed frozen action identity: {query_id}"
                )
            matched_unconstrained += 1
        else:
            if actual.actions:
                raise ValueError(
                    f"strict constrained abstention failed: {query_id}"
                )
            strict_abstentions += 1
    if matched_unconstrained != 121 or strict_abstentions != 7:
        raise ValueError("production selector partition changed")

    raw_ab = json.loads(ab_path.read_text(encoding="utf-8"))
    if not isinstance(raw_ab, dict):
        raise ValueError("sealed F5 A/B evidence is invalid")
    f5_gate = evaluate_f5_evidence(cast(dict[str, object], raw_ab))
    result: dict[str, object] = {
        "schema_version": "production-cross-vocabulary-6plus1-offline-gate-v1",
        "passed": True,
        "query_count": 128,
        "selector_action_identity_match_query_count": matched_unconstrained,
        "strict_abstention_query_count": strict_abstentions,
        "f5_evidence": f5_gate,
        "bounds": {
            "baseline_openalex_actions_max": 6,
            "supplement_openalex_actions_max": 1,
            "total_openalex_actions_max": 7,
            "baseline_raw_candidates_max": 300,
            "supplement_raw_candidates_max": 50,
            "total_raw_candidates_max": 350,
        },
        "inputs": {
            "production_lock_sha256": _sha256(production_lock_path.read_bytes()),
            "candidate_lock_sha256": _sha256(candidate_lock_path.read_bytes()),
            "f5_ab_sha256": _sha256(ab_path.read_bytes()),
            "validation_manifest_sha256": _sha256(
                (validation_root / "manifest.json").read_bytes()
            ),
            "context_manifest_sha256": _sha256(context_manifest_path.read_bytes()),
        },
        "decision": {
            "candidate_lock_offline_gate_passed": True,
            "current_production_lock_replaced": False,
            "online_smoke_authorized": False,
        },
        "safety": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "training_started": False,
            "test_partition_touched": False,
            "production_lock_modified": False,
        },
    }
    payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_immutable(output_path, payload)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument(
        "--context-manifest",
        type=Path,
        default=DEFAULT_CONTEXT_MANIFEST,
    )
    parser.add_argument(
        "--production-lock",
        type=Path,
        default=DEFAULT_PRODUCTION_LOCK,
    )
    parser.add_argument(
        "--candidate-lock",
        type=Path,
        default=DEFAULT_CANDIDATE_LOCK,
    )
    return parser


def main() -> None:
    print(
        json.dumps(
            run(build_parser().parse_args()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
