"""Gate the non-reinforcing OpenAlex supplement without touching live locks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from paper_search.evaluation.dataset import sha256_file


DEFAULT_DEVELOPMENT_ROOT = Path(
    "data/training_private/recall_policy/contrastive-openalex-bridge-nu128-v2"
)
DEFAULT_CONFIRMATION_ROOT = Path(
    "data/training_private/recall_policy/"
    "contrastive-openalex-bridge-confirmation128-v2-nonreinforcing-v1"
)
DEFAULT_REINFORCING = "f5-topk-candidate-ab-identity-v2.json"
DEFAULT_FIXED = "f5-topk-candidate-ab-identity-nonreinforcing-fairmerge-v1.json"
DEFAULT_CONFIRMATION = (
    "f5-topk-candidate-ab-identity-nonreinforcing-fairmerge-confirmation-v1.json"
)
DEFAULT_OUTPUT = "nonreinforcing-supplement-fairmerge-promotion-gate-v5.json"
GATE_SCHEMA_VERSION = "nonreinforcing-supplement-fairmerge-promotion-gate-v5"
GATE_SCOPE = "same-provider-supplement-fair-initial-order"
DEFAULT_PRODUCTION_LOCK = Path("deliverables/evaluator/live-evaluator.lock.yaml")
DEFAULT_CANDIDATE_LOCK = Path(
    "deliverables/evaluator/live-evaluator-unconstrained-6plus1.candidate.lock.yaml"
)


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    raw = payload.get("per_query")
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("replay per-query rows are invalid")
    return cast(list[dict[str, object]], raw)


def _load(path: Path) -> dict[str, object]:
    return _object(json.loads(path.read_text(encoding="utf-8")), label=str(path))


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _direction_summary(payload: Mapping[str, object]) -> dict[str, dict[str, int]]:
    overall = _object(payload.get("overall"), label="overall metrics")
    output: dict[str, dict[str, int]] = {}
    for cutoff in (5, 10, 20, 50):
        direction = _object(
            overall.get(f"direction_at_{cutoff}"),
            label=f"Top-{cutoff} direction",
        )
        output[str(cutoff)] = {
            key: _integer(direction.get(key), label=f"Top-{cutoff} {key}")
            for key in (
                "improved_query_count",
                "worsened_query_count",
                "unchanged_query_count",
            )
        }
    return output


def _verify_safety(payload: Mapping[str, object]) -> None:
    safety = _object(payload.get("safety"), label="replay safety")
    expected = {
        "baseline_evidence_immutable": True,
        "candidate_membership_monotonic": True,
        "llm_requests_made": 0,
        "online_requests_made": 0,
        "production_lock_modified": False,
        "test_partition_touched": False,
        "training_started": False,
    }
    if any(safety.get(key) != value for key, value in expected.items()):
        raise ValueError("non-reinforcing replay safety flags changed")


def _query_ids(payload: Mapping[str, object]) -> list[str]:
    query_ids = [str(row.get("query_id")) for row in _rows(payload)]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("replay query ids are not unique")
    return query_ids


def _verify_development_replay(
    reinforcing: Mapping[str, object],
    fixed: Mapping[str, object],
) -> dict[str, int]:
    old_rows = _rows(reinforcing)
    new_rows = _rows(fixed)
    if _query_ids(reinforcing) != _query_ids(fixed):
        raise ValueError("development replay query identity or order drift")
    count_drift = 0
    baseline_drift = 0
    for old, new in zip(old_rows, new_rows, strict=True):
        if any(
            old.get(key) != new.get(key)
            for key in ("baseline_candidate_count", "augmented_candidate_count")
        ):
            count_drift += 1
        if any(
            old.get(key) != new.get(key)
            for key in ("baseline", "baseline_top_50")
        ):
            baseline_drift += 1
    if count_drift or baseline_drift:
        raise ValueError("non-reinforcing replay changed frozen pool or baseline arm")
    return {
        "query_count": len(new_rows),
        "candidate_count_drift_query_count": count_drift,
        "baseline_arm_drift_query_count": baseline_drift,
    }


def _gold_ranks(row: Mapping[str, object], arm: str) -> list[int]:
    metrics = _object(row.get(arm), label=f"{arm} metrics")
    ranks = metrics.get("gold_ranks")
    if not isinstance(ranks, list) or not all(type(rank) is int for rank in ranks):
        raise ValueError(f"{arm} Gold ranks are invalid")
    return cast(list[int], ranks)


def _find_query(payload: Mapping[str, object], query_id: str) -> dict[str, object]:
    matches = [row for row in _rows(payload) if row.get("query_id") == query_id]
    if len(matches) != 1:
        raise ValueError(f"expected one replay row for {query_id}")
    return matches[0]


def _zero_regressions(directions: Mapping[str, Mapping[str, int]]) -> bool:
    return all(row["worsened_query_count"] == 0 for row in directions.values())


def _promotion_recommended(
    *,
    merge_fix_passed: bool,
    development_directions: Mapping[str, Mapping[str, int]],
    confirmation_directions: Mapping[str, Mapping[str, int]],
    independent_current_policy_confirmation: bool,
) -> bool:
    return (
        merge_fix_passed
        and independent_current_policy_confirmation
        and development_directions["50"]["improved_query_count"] > 0
        and confirmation_directions["50"]["improved_query_count"] > 0
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    workspace_root = Path(args.workspace_root).resolve()
    development_root = (workspace_root / args.development_root).resolve()
    confirmation_root = (workspace_root / args.confirmation_root).resolve()
    reinforcing_path = development_root / DEFAULT_REINFORCING
    fixed_path = development_root / DEFAULT_FIXED
    confirmation_path = confirmation_root / DEFAULT_CONFIRMATION
    development_manifest_path = development_root / "manifest.json"
    confirmation_manifest_path = confirmation_root / "manifest.json"
    output_path = development_root / DEFAULT_OUTPUT
    production_lock_path = (workspace_root / args.production_lock).resolve()
    candidate_lock_path = (workspace_root / args.candidate_lock).resolve()

    reinforcing = _load(reinforcing_path)
    fixed = _load(fixed_path)
    confirmation = _load(confirmation_path)
    development_manifest = _load(development_manifest_path)
    confirmation_manifest = _load(confirmation_manifest_path)
    expected_schema = (
        "production-f5-cross-vocabulary-candidate-ab-identity-nonreinforcing-v3"
    )
    if fixed.get("schema_version") != expected_schema:
        raise ValueError("fixed replay schema changed")
    if confirmation.get("schema_version") != expected_schema:
        raise ValueError("confirmation replay schema changed")
    for replay in (fixed, confirmation):
        _verify_safety(replay)
        if replay.get("query_count") != 128:
            raise ValueError("promotion replay must contain exactly 128 queries")

    replay_invariance = _verify_development_replay(reinforcing, fixed)
    development_ids = set(_query_ids(fixed))
    confirmation_ids = set(_query_ids(confirmation))
    overlap_count = len(development_ids.intersection(confirmation_ids))
    if overlap_count:
        raise ValueError("development and confirmation queries overlap")

    development_directions = _direction_summary(fixed)
    confirmation_directions = _direction_summary(confirmation)
    merge_fix_passed = _zero_regressions(
        development_directions
    ) and _zero_regressions(confirmation_directions)

    regression_query_id = "AutoScholarQuery_train_5701"
    old_regression = _find_query(reinforcing, regression_query_id)
    fixed_regression = _find_query(fixed, regression_query_id)
    old_augmented_ranks = _gold_ranks(old_regression, "augmented")
    fixed_baseline_ranks = _gold_ranks(fixed_regression, "baseline")
    fixed_augmented_ranks = _gold_ranks(fixed_regression, "augmented")
    regression_boundary_repaired = bool(
        fixed_baseline_ranks
        and fixed_augmented_ranks
        and fixed_baseline_ranks[0] <= 20
        and fixed_augmented_ranks[0] <= 20
        and old_augmented_ranks
        and old_augmented_ranks[0] > 20
    )
    if not regression_boundary_repaired:
        raise ValueError("known Top-20 regression was not repaired")

    current_policy = str(development_manifest.get("action_policy"))
    confirmation_policy = str(confirmation_manifest.get("action_policy"))
    independent_current_policy_confirmation = current_policy == confirmation_policy
    promotion_recommended = _promotion_recommended(
        merge_fix_passed=merge_fix_passed,
        development_directions=development_directions,
        confirmation_directions=confirmation_directions,
        independent_current_policy_confirmation=independent_current_policy_confirmation,
    )

    result: dict[str, object] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "scope": GATE_SCOPE,
        "passed": merge_fix_passed,
        "replay_invariance": replay_invariance,
        "query_partition": {
            "development_query_count": len(development_ids),
            "confirmation_query_count": len(confirmation_ids),
            "overlap_query_count": overlap_count,
        },
        "top_k_directions": {
            "current_v2_development": development_directions,
            "current_v2_disjoint_confirmation": confirmation_directions,
        },
        "known_regression_repair": {
            "query_id": regression_query_id,
            "reinforcing_augmented_first_gold_rank": old_augmented_ranks[0],
            "fixed_baseline_first_gold_rank": fixed_baseline_ranks[0],
            "fixed_augmented_first_gold_rank": fixed_augmented_ranks[0],
            "top_20_boundary_regression_repaired": regression_boundary_repaired,
            "remaining_rank_shift_is_new_candidate_competition": True,
        },
        "evidence_limit": {
            "development_action_policy": current_policy,
            "confirmation_action_policy": confirmation_policy,
            "independent_current_policy_confirmation": (
                independent_current_policy_confirmation
            ),
            "confirmation_only_validates_merge_guardrail": (
                not independent_current_policy_confirmation
            ),
        },
        "decision": {
            "nonreinforcing_merge_fix_gate_passed": merge_fix_passed,
            "candidate_lock_promotion_recommended": promotion_recommended,
            "candidate_lock_status": (
                "promotable"
                if promotion_recommended
                else (
                    "rejected-no-independent-top50-gain"
                    if independent_current_policy_confirmation
                    else "hold-pending-disjoint-current-v2-confirmation"
                )
            ),
            "current_production_lock_replaced": False,
            "current_candidate_lock_replaced": False,
        },
        "inputs": {
            "reinforcing_development_replay_sha256": sha256_file(reinforcing_path),
            "fixed_development_replay_sha256": sha256_file(fixed_path),
            "fixed_confirmation_replay_sha256": sha256_file(confirmation_path),
            "development_manifest_sha256": sha256_file(development_manifest_path),
            "confirmation_manifest_sha256": sha256_file(confirmation_manifest_path),
            "production_lock_sha256": sha256_file(production_lock_path),
            "candidate_lock_sha256": sha256_file(candidate_lock_path),
        },
        "safety": {
            "online_request_count": 0,
            "llm_request_count": 0,
            "training_started": False,
            "test_partition_touched": False,
            "production_lock_modified": False,
            "candidate_lock_modified": False,
        },
    }
    artifact_bytes = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_immutable(output_path, artifact_bytes)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--development-root", type=Path, default=DEFAULT_DEVELOPMENT_ROOT
    )
    parser.add_argument(
        "--confirmation-root", type=Path, default=DEFAULT_CONFIRMATION_ROOT
    )
    parser.add_argument("--production-lock", type=Path, default=DEFAULT_PRODUCTION_LOCK)
    parser.add_argument("--candidate-lock", type=Path, default=DEFAULT_CANDIDATE_LOCK)
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
