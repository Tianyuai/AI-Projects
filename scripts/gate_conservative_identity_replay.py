"""Gate conservative PASA identity aliases against two sealed offline replays."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from paper_search.evaluation.dataset import IdentifierMap, sha256_file


DEFAULT_ROOT = Path(
    "data/training_private/recall_policy/contrastive-openalex-bridge-nu128-v2"
)
DEFAULT_BEFORE = "f5-topk-candidate-ab-v1.json"
DEFAULT_AFTER = "f5-topk-candidate-ab-identity-v2.json"
DEFAULT_MAP = "conservative-pasa-identity-alias-map-v1.json"
DEFAULT_EVIDENCE = "conservative-pasa-identity-alias-evidence-v1.json"
DEFAULT_ATTRIBUTION = "pasa-miss-attribution-v1.json"
DEFAULT_OUTPUT = "conservative-identity-replay-gate-v2.json"
DEFAULT_PRODUCTION_LOCK = Path("deliverables/evaluator/live-evaluator.lock.yaml")
DEFAULT_EXISTING_MAP = Path("data/identifier-map.json")


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    raw = payload.get("per_query")
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("replay per-query rows are invalid")
    return cast(list[dict[str, object]], raw)


def _gold_ranks(row: Mapping[str, object], arm: str) -> list[int]:
    metrics = _object(row.get(arm), label=f"{arm} metrics")
    ranks = metrics.get("gold_ranks")
    if not isinstance(ranks, list) or not all(type(rank) is int for rank in ranks):
        raise ValueError(f"{arm} Gold ranks are invalid")
    return cast(list[int], ranks)


def compare_replays(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, int]:
    """Prove identity evaluation only adds matches to unchanged ranked pools."""

    before_rows = _rows(before)
    after_rows = _rows(after)
    before_ids = [str(row.get("query_id")) for row in before_rows]
    after_ids = [str(row.get("query_id")) for row in after_rows]
    if before_ids != after_ids or len(set(before_ids)) != len(before_ids):
        raise ValueError("replay query identity or order drift")

    count_drift = 0
    sequence_drift = 0
    removed = 0
    new_baseline = 0
    new_augmented = 0
    changed = 0
    for previous, current in zip(before_rows, after_rows, strict=True):
        if any(
            previous.get(key) != current.get(key)
            for key in ("baseline_candidate_count", "augmented_candidate_count")
        ):
            count_drift += 1
        if any(
            previous.get(key) != current.get(key)
            for key in ("baseline_top_50", "augmented_top_50")
        ):
            sequence_drift += 1
        previous_baseline = _gold_ranks(previous, "baseline")
        current_baseline = _gold_ranks(current, "baseline")
        previous_augmented = _gold_ranks(previous, "augmented")
        current_augmented = _gold_ranks(current, "augmented")
        if not set(previous_baseline).issubset(current_baseline) or not set(
            previous_augmented
        ).issubset(current_augmented):
            removed += 1
        new_baseline += not previous_baseline and bool(current_baseline)
        new_augmented += not previous_augmented and bool(current_augmented)
        changed += (
            previous_baseline != current_baseline
            or previous_augmented != current_augmented
        )
    if count_drift:
        raise ValueError("candidate count drift detected")
    if sequence_drift:
        raise ValueError("ranked candidate sequence drift detected")
    if removed:
        raise ValueError("identity replay removed a previously matched Gold rank")
    return {
        "query_count": len(before_rows),
        "candidate_count_drift_query_count": count_drift,
        "ranked_sequence_drift_query_count": sequence_drift,
        "removed_gold_rank_query_count": removed,
        "newly_resolved_baseline_query_count": new_baseline,
        "newly_resolved_augmented_query_count": new_augmented,
        "gold_rank_changed_query_count": changed,
    }


def _load(path: Path) -> dict[str, object]:
    return _object(json.loads(path.read_text(encoding="utf-8")), label=str(path))


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def run(args: argparse.Namespace) -> dict[str, object]:
    workspace_root = Path(args.workspace_root).resolve()
    root = (workspace_root / args.validation_root).resolve()
    before_path = root / DEFAULT_BEFORE
    after_path = root / DEFAULT_AFTER
    alias_map_path = root / DEFAULT_MAP
    evidence_path = root / DEFAULT_EVIDENCE
    attribution_path = root / DEFAULT_ATTRIBUTION
    output_path = root / DEFAULT_OUTPUT
    production_lock_path = (workspace_root / args.production_lock).resolve()
    existing_map_path = (workspace_root / args.existing_identifier_map).resolve()

    before = _load(before_path)
    after = _load(after_path)
    evidence = _load(evidence_path)
    attribution = _load(attribution_path)
    if before.get("schema_version") != "production-f5-cross-vocabulary-candidate-ab-v1":
        raise ValueError("baseline replay schema changed")
    if after.get("schema_version") != "production-f5-cross-vocabulary-candidate-ab-identity-v2":
        raise ValueError("identity replay schema changed")
    comparison = _object(after.get("comparison"), label="identity comparison")
    if comparison.get("verified_identifier_map_applied") is not True:
        raise ValueError("identity replay did not bind the verified alias map")
    after_inputs = _object(after.get("inputs"), label="identity replay inputs")
    if after_inputs.get("verified_identifier_map_sha256") != sha256_file(alias_map_path):
        raise ValueError("identity replay alias-map hash mismatch")
    evidence_inputs = _object(evidence.get("inputs"), label="identity evidence inputs")
    if evidence_inputs.get("identifier_map_sha256") != sha256_file(alias_map_path):
        raise ValueError("identity evidence alias-map hash mismatch")
    policy = _object(evidence.get("policy"), label="identity policy")
    if policy.get("title_only_alias_allowed") is not False:
        raise ValueError("title-only identity aliases were not rejected")
    safety = _object(evidence.get("safety"), label="identity safety")
    if any(
        safety.get(key) != expected
        for key, expected in {
            "query_gold_associations_used_for_alias_derivation": False,
            "gold_identifier_membership_used_for_alias_derivation": False,
            "online_requests_made": 0,
            "llm_requests_made": 0,
            "training_started": False,
            "test_partition_touched": False,
            "production_lock_modified": False,
        }.items()
    ):
        raise ValueError("identity evidence safety flags changed")

    replay_comparison = compare_replays(before, after)
    if replay_comparison["query_count"] != 128:
        raise ValueError("identity gate requires exactly 128 replay queries")

    alias_map = IdentifierMap.from_path(alias_map_path)
    existing_map = IdentifierMap.from_path(existing_map_path)
    conflicts = [
        alias
        for alias, target in alias_map.resolved_pairs()
        if existing_map.covers(alias) and existing_map.resolve(alias) != target
    ]
    if conflicts:
        raise ValueError(f"identity alias conflicts with existing map: {conflicts[0]}")

    attribution_rows = _rows(attribution)
    diagnosed_ids = {
        str(row["query_id"])
        for row in attribution_rows
        if row.get("category") == "openalex_metadata_identity_gap"
    }
    if len(diagnosed_ids) != 12:
        raise ValueError("diagnosed identity-gap partition changed")
    after_by_id = {str(row["query_id"]): row for row in _rows(after)}
    resolved_diagnosed = sorted(
        query_id
        for query_id in diagnosed_ids
        if _gold_ranks(after_by_id[query_id], "augmented")
    )
    abstained_diagnosed = sorted(diagnosed_ids.difference(resolved_diagnosed))
    if len(resolved_diagnosed) != 11 or len(abstained_diagnosed) != 1:
        raise ValueError("conservative identity-gap resolution count changed")

    before_pool = _object(
        _object(before.get("overall"), label="baseline overall").get("candidate_pool"),
        label="baseline candidate pool",
    )
    after_pool = _object(
        _object(after.get("overall"), label="identity overall").get("candidate_pool"),
        label="identity candidate pool",
    )
    after_by_signal = _object(after.get("by_signal"), label="identity signals")
    unconstrained = _object(
        after_by_signal.get("unconstrained"),
        label="identity-corrected unconstrained evidence",
    )
    top_k_directions: dict[str, dict[str, int]] = {}
    for cutoff in (10, 20, 50):
        direction = _object(
            unconstrained.get(f"direction_at_{cutoff}"),
            label=f"identity-corrected Top-{cutoff} direction",
        )
        top_k_directions[str(cutoff)] = {
            key: _integer(direction.get(key), label=f"Top-{cutoff} {key}")
            for key in (
                "improved_query_count",
                "worsened_query_count",
                "unchanged_query_count",
            )
        }
    zero_top_k_regression = all(
        direction["worsened_query_count"] == 0
        for direction in top_k_directions.values()
    )
    result: dict[str, object] = {
        "schema_version": "conservative-identity-replay-gate-v2",
        "scope": "identity-alias-repair-only",
        "passed": True,
        "replay_comparison": replay_comparison,
        "candidate_gold_hit_query_counts": {
            "before": {
                "baseline": before_pool["baseline_gold_hit_query_count"],
                "augmented": before_pool["augmented_gold_hit_query_count"],
            },
            "after": {
                "baseline": after_pool["baseline_gold_hit_query_count"],
                "augmented": after_pool["augmented_gold_hit_query_count"],
            },
        },
        "diagnosed_identity_gaps": {
            "query_count": len(diagnosed_ids),
            "resolved_query_count": len(resolved_diagnosed),
            "strict_abstention_query_count": len(abstained_diagnosed),
            "strict_abstention_query_ids": abstained_diagnosed,
        },
        "alias_evidence": {
            "accepted_identity_record_count": evidence["accepted_identity_record_count"],
            "identifier_alias_count": evidence["identifier_alias_count"],
            "existing_identifier_map_overlap_count": sum(
                existing_map.covers(alias) for alias, _target in alias_map.resolved_pairs()
            ),
            "existing_identifier_map_conflict_count": 0,
        },
        "identity_corrected_candidate_policy": {
            "unconstrained_query_count": unconstrained["query_count"],
            "top_k_directions": top_k_directions,
            "zero_top_k_regression_gate_passed": zero_top_k_regression,
            "previous_v1_identity_metrics_superseded": True,
        },
        "decision": {
            "identity_repair_offline_gate_passed": True,
            "candidate_lock_promotion_recommended": zero_top_k_regression,
            "current_production_lock_replaced": False,
        },
        "inputs": {
            "before_replay_sha256": sha256_file(before_path),
            "after_replay_sha256": sha256_file(after_path),
            "alias_map_sha256": sha256_file(alias_map_path),
            "alias_evidence_sha256": sha256_file(evidence_path),
            "attribution_sha256": sha256_file(attribution_path),
            "existing_identifier_map_sha256": sha256_file(existing_map_path),
            "production_lock_sha256": sha256_file(production_lock_path),
        },
        "safety": {
            "online_request_count": 0,
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
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--production-lock", type=Path, default=DEFAULT_PRODUCTION_LOCK)
    parser.add_argument(
        "--existing-identifier-map",
        type=Path,
        default=DEFAULT_EXISTING_MAP,
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
