"""Freeze an offline audit of one query-adaptive OpenAlex recall pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from paper_search.evaluation.predictions import paper_id_aliases
from paper_search.learning.openalex_daily_schedule import (
    SearchActionIdentity,
    estimate_max_openalex_search_api_calls,
    load_settled_search_action_identities,
    search_action_identity,
)
from paper_search.learning.unified_recall_context import (
    load_frozen_recall_query_specs,
)


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], raw)


def _constraint_family(spec: object) -> str:
    if getattr(spec, "exclusions"):
        return "negation"
    if getattr(spec, "datasets"):
        return "dataset"
    if getattr(spec, "methods"):
        return "method"
    if getattr(spec, "year_from") is not None or getattr(spec, "year_to") is not None:
        return "year"
    if getattr(spec, "tasks"):
        return "task"
    return "unstructured"


def _planned_actions(plan: dict[str, Any]) -> dict[str, tuple[SearchActionIdentity, ...]]:
    result: dict[str, tuple[SearchActionIdentity, ...]] = {}
    for shard in plan["schedule"]["shards"]:
        for row in shard["queries"]:
            query_id = str(row["query_id"])
            if query_id in result:
                raise ValueError(f"duplicate planned query: {query_id}")
            result[query_id] = tuple(
                SearchActionIdentity.model_validate(item)
                for item in row["missing_actions"]
            )
    return result


def _paper_aliases(raw: dict[str, Any]) -> set[str]:
    aliases = set(paper_id_aliases(raw.get("canonical_id")))
    for value, kind in (
        (raw.get("doi"), "doi"),
        (raw.get("arxiv_id"), "arxiv"),
        (raw.get("openalex_id"), "openalex"),
        (raw.get("semantic_scholar_id"), "semantic_scholar"),
    ):
        aliases.update(paper_id_aliases(value, kind=kind))
    return aliases


def _gold_hit_ids(
    gold_ids: tuple[str, ...], candidates: list[dict[str, Any]]
) -> list[str]:
    candidate_aliases: set[str] = set()
    for candidate in candidates:
        candidate_aliases.update(_paper_aliases(candidate))
    return [
        gold_id
        for gold_id in gold_ids
        if candidate_aliases.intersection(paper_id_aliases(gold_id))
    ]


def _reconciled_authorization_calls(
    *, observed_receipt_calls: int, reconciliation: dict[str, Any]
) -> int:
    reconciled = (
        observed_receipt_calls
        - int(reconciliation["failed_action_result_count"])
        + int(reconciliation.get("conservative_in_flight_call_count", 0))
    )
    if reconciled < 0:
        raise ValueError("reconciled authorization call count is negative")
    return reconciled


def _result_is_settled(result: dict[str, Any]) -> bool:
    errors = result.get("errors", [])
    if result.get("infrastructure_failure") is True or not isinstance(errors, list):
        return False
    return not errors or all(
        isinstance(error, dict)
        and error.get("code") == "invalid_work"
        and error.get("retryable") is False
        for error in errors
    )


def _evaluate_validation_gate(
    *,
    gate: dict[str, Any],
    metrics: dict[str, Any],
    strata: dict[str, dict[str, int]],
    missing_action_count: int,
    final_failed_query_count: int,
    llm_call_count: int,
) -> dict[str, Any]:
    thresholds = cast(dict[str, Any], gate["thresholds"])
    checks: list[dict[str, Any]] = []

    def minimum(name: str, observed: int | float, threshold: int | float) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "operator": ">=",
                "threshold": threshold,
                "passed": observed >= threshold,
            }
        )

    def maximum(name: str, observed: int | float, threshold: int | float) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "operator": "<=",
                "threshold": threshold,
                "passed": observed <= threshold,
            }
        )

    for suffix in (
        "gold_hit_query_count",
        "gold_hit_query_rate",
        "gold_hit_count",
        "macro_candidate_recall",
    ):
        minimum(
            f"minimum_{suffix}",
            metrics[suffix],
            thresholds[f"minimum_{suffix}"],
        )
    required_strata = cast(
        dict[str, int], thresholds["minimum_hit_query_count_by_required_stratum"]
    )
    for family, threshold in required_strata.items():
        minimum(
            f"minimum_hit_query_count_by_required_stratum.{family}",
            int(strata.get(family, {}).get("hit_query_count", 0)),
            threshold,
        )
    maximum(
        "maximum_missing_action_identity_count",
        missing_action_count,
        int(thresholds["maximum_missing_action_identity_count"]),
    )
    maximum(
        "maximum_final_failed_query_count",
        final_failed_query_count,
        int(thresholds["maximum_final_failed_query_count"]),
    )
    maximum(
        "maximum_llm_call_count",
        llm_call_count,
        int(thresholds["maximum_llm_call_count"]),
    )
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def _index_canary_reports(
    paths: list[Path],
) -> dict[str, tuple[dict[str, Any], Path]]:
    indexed: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in paths:
        report = _read_json(path)
        rows = report.get("result", {}).get("per_query", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"canary report contains no queries: {path}")
        for row in rows:
            query_id = row.get("query_id") if isinstance(row, dict) else None
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"canary report query identity is invalid: {path}")
            if query_id in indexed:
                raise ValueError(f"duplicate successful canary query: {query_id}")
            indexed[query_id] = (report, path)
    return indexed


def _main(args: argparse.Namespace) -> None:
    root = Path(args.workspace_root).resolve()
    pilot = (root / args.pilot).resolve()
    receipts = pilot / "receipts" / "openalex"
    manifest_path = pilot / "manifest.json"
    plan_path = pilot / "plan.json"
    partition_path = pilot / "pilot-partition.jsonl"
    output_path = pilot / args.output_name
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen result: {output_path}")

    manifest = _read_json(manifest_path)
    sample_role = str(manifest.get("sample_role", "recall_policy_discovery"))
    plan_bytes = plan_path.read_bytes()
    plan = cast(dict[str, Any], json.loads(plan_bytes))
    if manifest.get("plan_sha256") != _sha256(plan_bytes):
        raise ValueError("pilot plan hash mismatch")
    planned = _planned_actions(plan)
    planned_action_count = sum(len(items) for items in planned.values())
    if planned_action_count != int(manifest["new_openalex_action_count"]):
        raise ValueError("pilot action count mismatch")

    settled = load_settled_search_action_identities([receipts])
    missing = {
        query_id: sorted(
            set(actions).difference(set(settled.get(query_id, frozenset()))),
            key=lambda item: (
                item.action_type,
                item.search_mode,
                item.normalized_text,
            ),
        )
        for query_id, actions in planned.items()
    }
    missing = {query_id: items for query_id, items in missing.items() if items}
    if missing:
        raise ValueError(f"pilot has {sum(map(len, missing.values()))} missing actions")

    canary_paths = sorted(receipts.rglob("canary-report.json"))
    canary_by_query = _index_canary_reports(canary_paths)
    if set(canary_by_query) != set(planned):
        raise ValueError("successful canary coverage does not match the pilot plan")

    recall_paths = sorted(receipts.rglob("recall-report.json"))
    retrieval_paths = sorted(receipts.rglob("retrieval/attempt-01/*.json"))
    raw_search_calls = 0
    action_result_count = 0
    failed_retrieval_count = 0
    result_error_codes: Counter[str] = Counter()
    result_with_errors = 0
    for path in retrieval_paths:
        receipt = _read_json(path)
        if receipt.get("attempt_status") == "failed":
            failed_retrieval_count += 1
        for item in receipt.get("results", []):
            action_result_count += 1
            raw_search_calls += int(item["usage"]["search_api_calls"])
            errors = item.get("errors", [])
            if errors:
                result_with_errors += 1
            result_error_codes.update(str(error["code"]) for error in errors)

    llm_calls = sum(
        int(_read_json(path)["usage"]["llm_calls"]) for path in canary_paths
    )
    partition_rows = [
        json.loads(line)
        for line in partition_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    partition_by_query = {str(row["query_id"]): row for row in partition_rows}
    if set(partition_by_query) != set(planned):
        raise ValueError("pilot partition coverage does not match the pilot plan")

    candidates_by_query_action: dict[
        str, dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    planned_sets = {query_id: set(actions) for query_id, actions in planned.items()}
    for generation_path in sorted(receipts.rglob("generation/attempt-01/*.json")):
        retrieval_path = (
            generation_path.parents[2]
            / "retrieval"
            / "attempt-01"
            / generation_path.name
        )
        if not retrieval_path.is_file():
            continue
        generation = _read_json(generation_path)
        retrieval = _read_json(retrieval_path)
        query_id = str(generation.get("query_id"))
        if query_id not in planned_sets or retrieval.get("query_id") != query_id:
            continue
        identity_by_action_id: dict[str, SearchActionIdentity] = {}
        for action in generation.get("actions", []):
            identity = search_action_identity(action)
            if identity is not None:
                identity_by_action_id[str(action["action_id"])] = identity
        for result_row in retrieval.get("results", []):
            action_id = str(result_row.get("action_id"))
            identity = identity_by_action_id.get(action_id)
            if (
                identity not in planned_sets[query_id]
                or not _result_is_settled(result_row)
            ):
                continue
            candidates_by_query_action[query_id][action_id].extend(
                cast(list[dict[str, Any]], result_row.get("hits", []))
            )

    hit_query_ids: set[str] = set()
    gold_hit_count = 0
    gold_association_count = 0
    macro_recall = 0.0
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    action_attribution: Counter[str] = Counter()
    attributed_hits: set[tuple[str, str]] = set()

    specs = load_frozen_recall_query_specs(
        partition_path=(root / args.training_partition).resolve(),
        manifest_path=(root / args.context_manifest).resolve(),
    )
    for query_id in planned:
        family = _constraint_family(specs[query_id])
        gold_ids = tuple(
            str(item) for item in partition_by_query[query_id]["gold_paper_ids"]
        )
        candidates = [
            paper
            for papers in candidates_by_query_action[query_id].values()
            for paper in papers
        ]
        hits = _gold_hit_ids(gold_ids, candidates)
        strata[family]["query_count"] += 1
        strata[family]["gold_association_count"] += len(gold_ids)
        strata[family]["gold_hit_count"] += len(hits)
        if hits:
            hit_query_ids.add(query_id)
            strata[family]["hit_query_count"] += 1
        gold_hit_count += len(hits)
        gold_association_count += len(gold_ids)
        macro_recall += len(hits) / len(gold_ids)
        for hit_id in hits:
            gold_aliases = set(paper_id_aliases(hit_id))
            matched_actions: set[str] = set()
            for action_id, papers in candidates_by_query_action[query_id].items():
                for paper in papers:
                    if gold_aliases.intersection(_paper_aliases(paper)):
                        matched_actions.add(action_id)
            for action_id in matched_actions:
                action_attribution[action_id] += 1
                attributed_hits.add((query_id, hit_id))

    expected_max_raw_calls = sum(
        estimate_max_openalex_search_api_calls(
            actions,
            max_results_per_action=int(manifest["max_results_per_action"]),
        )
        for actions in planned.values()
    )
    lexical_action_count = sum(
        action.search_mode == "lexical"
        for actions in planned.values()
        for action in actions
    )
    semantic_action_count = planned_action_count - lexical_action_count
    authorization_cap = int(args.authorized_raw_search_api_calls)

    if any(
        row.get("role") != "training" or row.get("split") != "auto_train"
        for row in partition_rows
    ):
        raise ValueError("pilot partition is not isolated auto_train data")
    for path in sorted(receipts.rglob("generation/attempt-01/*.json")):
        receipt = _read_json(path)
        for action in receipt.get("actions", []):
            payload_keys = set(action.get("payload", {}))
            if (
                action.get("action_type") != "text_search"
                or "query_text" not in payload_keys
                or not payload_keys.issubset({"query_text", "search_mode"})
            ):
                raise ValueError(f"outbound action is outside query-text search: {path}")

    production_path = (root / args.production_selection).resolve()
    evaluator_path = (root / args.evaluator_lock).resolve()
    production_sha = _sha256(production_path.read_bytes())
    evaluator_sha = _sha256(evaluator_path.read_bytes())
    if production_sha != args.expected_production_sha256:
        raise ValueError("production selection changed during recall discovery")
    if evaluator_sha != args.expected_evaluator_sha256:
        raise ValueError("evaluator lock changed during recall discovery")

    metrics = {
        "gold_hit_query_count": len(hit_query_ids),
        "gold_hit_query_rate": len(hit_query_ids) / len(planned),
        "gold_hit_count": gold_hit_count,
        "gold_association_count": gold_association_count,
        "macro_candidate_recall": macro_recall / len(planned),
        "micro_candidate_recall": gold_hit_count / gold_association_count,
        "candidate_recall_scope": "new query-adaptive actions only",
        "evidence_role": sample_role,
    }
    strata_result = {
        family: dict(sorted(counts.items()))
        for family, counts in sorted(strata.items())
    }
    gate_result: dict[str, Any] | None = None
    gate_path: Path | None = None
    if sample_role == "independent_frozen_validation":
        if args.validation_gate is None:
            raise ValueError("independent validation requires --validation-gate")
        gate_path = (root / args.validation_gate).resolve()
        gate = _read_json(gate_path)
        if manifest.get("validation_gate_sha256") != _sha256(gate_path.read_bytes()):
            raise ValueError("validation gate hash mismatch")
        if int(gate["query_count"]) != len(planned):
            raise ValueError("validation gate query count mismatch")
        gate_result = _evaluate_validation_gate(
            gate=gate,
            metrics=metrics,
            strata=strata_result,
            missing_action_count=0,
            final_failed_query_count=0,
            llm_call_count=llm_calls,
        )

    is_validation = sample_role == "independent_frozen_validation"
    reconciliation: dict[str, Any] | None = None
    observed_authorization_calls = raw_search_calls
    if args.reconciliation is not None:
        reconciliation_path = (root / args.reconciliation).resolve()
        reconciliation = _read_json(reconciliation_path)
        observed_authorization_calls = _reconciled_authorization_calls(
            observed_receipt_calls=raw_search_calls,
            reconciliation=reconciliation,
        )
    else:
        reconciliation_path = None
    continuation_path = (
        (root / args.continuation_manifest).resolve()
        if args.continuation_manifest is not None
        else None
    )

    result = {
        "schema_version": (
            "query-adaptive-recall-validation-result-v2"
            if is_validation
            else "query-adaptive-recall-discovery-result-v1"
        ),
        "bindings": {
            "pilot_manifest_sha256": _sha256(manifest_path.read_bytes()),
            "plan_sha256": _sha256(plan_bytes),
            "pilot_partition_sha256": _sha256(partition_path.read_bytes()),
            "production_selection_sha256": production_sha,
            "evaluator_lock_sha256": evaluator_sha,
            **(
                {"validation_gate_sha256": _sha256(gate_path.read_bytes())}
                if gate_path is not None
                else {}
            ),
            **(
                {"network_reconciliation_sha256": _sha256(reconciliation_path.read_bytes())}
                if reconciliation_path is not None
                else {}
            ),
            **(
                {"continuation_manifest_sha256": _sha256(continuation_path.read_bytes())}
                if continuation_path is not None
                else {}
            ),
        },
        "scope": {
            "query_count": len(planned),
            "role": sample_role,
            "split": "auto_train",
            "test_partition_touched": False,
            "gold_used_for_action_generation": False,
            "outbound_payload": "query_text_and_search_mode_only",
            "gold_labels_or_identifiers_sent": False,
            "llm_calls": llm_calls,
        },
        "completion": {
            "planned_unique_action_identity_count": planned_action_count,
            "lexical_action_identity_count": lexical_action_count,
            "semantic_action_identity_count": semantic_action_count,
            "settled_unique_action_identity_count": planned_action_count,
            "missing_action_identity_count": 0,
            "successful_query_report_count": len(canary_by_query),
            "retrieval_receipt_count": len(retrieval_paths),
            "recall_report_count": len(recall_paths),
            "failed_initial_attempt_count": failed_retrieval_count,
            "action_result_count_including_retries": action_result_count,
            "result_with_errors_count": result_with_errors,
            "result_error_code_counts": dict(sorted(result_error_codes.items())),
        },
        "raw_request_accounting": {
            "authorized_raw_search_api_call_cap": authorization_cap,
            "planned_unique_action_identity_count": planned_action_count,
            "corrected_preflight_max_raw_search_api_calls": expected_max_raw_calls,
            "observed_raw_search_api_calls": raw_search_calls,
            "reconciled_non_dispatched_receipt_calls": (
                int(reconciliation["failed_action_result_count"])
                if reconciliation is not None
                else 0
            ),
            "conservative_in_flight_call_count": (
                int(reconciliation.get("conservative_in_flight_call_count", 0))
                if reconciliation is not None
                else 0
            ),
            "observed_authorization_calls": observed_authorization_calls,
            "authorization_overrun": max(
                0, observed_authorization_calls - authorization_cap
            ),
            "max_pagination_calls_beyond_unique_actions": (
                expected_max_raw_calls - planned_action_count
            ),
            "observed_pagination_calls_beyond_action_attempts": (
                raw_search_calls - action_result_count
            ),
            "retry_action_attempt_count": action_result_count
            - planned_action_count,
            "cause": "lexical depth 100 may use two pages; sandbox-blocked non-dispatched receipts are reconciled separately",
            "further_online_calls_blocked_pending_new_authorization": True,
        },
        ("validation_metrics" if is_validation else "discovery_metrics"): metrics,
        "strata": strata_result,
        "gold_hit_action_attribution": {
            "attributed_gold_hit_count": len(attributed_hits),
            "unattributed_gold_hit_count": gold_hit_count - len(attributed_hits),
            "source_evidence_counts": dict(sorted(action_attribution.items())),
            "counting_note": "one Gold hit may be supported by multiple actions",
        },
        "decision": (
            {
                "validation_gate_passed": bool(gate_result["passed"]),
                "candidate_pack_freeze_authorized": bool(gate_result["passed"]),
                "f4_f5_retraining_started": False,
                "production_lock_modified": False,
                "next_step": (
                    "freeze expanded candidate pack and generate effective pairs"
                    if gate_result["passed"]
                    else "hold candidate-pack expansion and inspect failed gate checks"
                ),
            }
            if gate_result is not None
            else {
                "recall_policy_promoted": False,
                "f4_f5_retraining_started": False,
                "next_gate": "disjoint frozen OpenAlex-only recall validation",
                "new_online_authorization_required": True,
            }
        ),
        **({"validation_gate": gate_result} if gate_result is not None else {}),
    }
    output_path.write_bytes(_canonical_bytes(result) + b"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--training-partition", required=True)
    parser.add_argument("--context-manifest", required=True)
    parser.add_argument("--production-selection", required=True)
    parser.add_argument("--evaluator-lock", required=True)
    parser.add_argument("--expected-production-sha256", required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument("--authorized-raw-search-api-calls", type=int, required=True)
    parser.add_argument("--validation-gate")
    parser.add_argument("--reconciliation")
    parser.add_argument("--continuation-manifest")
    parser.add_argument("--output-name", default="discovery-result-v1.json")
    _main(parser.parse_args())


if __name__ == "__main__":
    main()
