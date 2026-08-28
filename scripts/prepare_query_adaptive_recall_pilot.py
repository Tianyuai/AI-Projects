"""Freeze a Gold-blind, receipt-deduplicated OpenAlex recall pilot."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

from paper_search.learning.candidate_ceiling import QueryAdaptiveHighRecallGenerator
from paper_search.learning.openalex_daily_schedule import (
    OpenAlexDailyShard,
    OpenAlexDailyTrainingPlan,
    OpenAlexDailyTrainingSchedule,
    ScheduledQueryActions,
    SearchActionIdentity,
    build_missing_action_work,
    estimate_max_openalex_search_api_calls,
    load_completed_search_action_identities,
    search_action_identity,
)
from paper_search.learning.unified_recall_context import (
    load_frozen_recall_query_specs,
)
from paper_search.recall_experiments.contracts import RecallGenerationContext


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _load_quota_usage_from_ledger(
    path: Path,
    *,
    window: date,
) -> dict[str, object]:
    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        raw_rows = connection.execute(
            """
            SELECT key_slot, max_search_calls, used_search_calls
            FROM quota_usage
            WHERE window = ?
            ORDER BY key_slot
            """,
            (window.isoformat(),),
        ).fetchall()
    if not raw_rows:
        raise ValueError("quota ledger has no rows for the requested window")
    rows = [
        {
            "window": window.isoformat(),
            "key_slot": int(key_slot),
            "max_search_calls": int(maximum),
            "used_search_calls": int(used),
        }
        for key_slot, maximum, used in raw_rows
    ]
    capacity = sum(int(row["max_search_calls"]) for row in rows)
    used = sum(int(row["used_search_calls"]) for row in rows)
    return {
        "window": window.isoformat(),
        "capacity": capacity,
        "used": used,
        "remaining": capacity - used,
        "rows": rows,
    }


def _sampling_digest(query_id: str, *, candidate_policy: str) -> str:
    return hashlib.sha256(f"{candidate_policy}:{query_id}".encode()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _recursive_query_ids(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        query_id = value.get("query_id")
        if isinstance(query_id, str):
            result.add(query_id)
        for nested in value.values():
            result.update(_recursive_query_ids(nested))
    elif isinstance(value, list):
        for nested in value:
            result.update(_recursive_query_ids(nested))
    return result


def _load_excluded_query_ids(
    *,
    manifest_paths: list[Path],
    receipt_roots: list[Path],
) -> set[str]:
    excluded: set[str] = set()
    for path in manifest_paths:
        excluded.update(_recursive_query_ids(json.loads(path.read_text(encoding="utf-8"))))
    for root in receipt_roots:
        if not root.is_dir():
            raise ValueError(f"exclusion receipt root is unavailable: {root}")
        for path in sorted(root.rglob("generation/attempt-01/*.json")):
            excluded.update(
                _recursive_query_ids(json.loads(path.read_text(encoding="utf-8")))
            )
    return excluded


def _load_no_hit_ids(audit_dir: Path) -> set[str]:
    result: set[str] = set()
    for path in sorted(audit_dir.glob("batch-*.jsonl")):
        for row in _jsonl(path):
            if row.get("positive_candidate_count") == 0:
                query_id = row.get("query_id")
                if not isinstance(query_id, str):
                    raise ValueError("candidate audit query identity is invalid")
                result.add(query_id)
    if not result:
        raise ValueError("candidate audit contains no missing-Gold queries")
    return result


async def _required_actions(
    rows: list[dict[str, object]],
    generator: QueryAdaptiveHighRecallGenerator,
    specs: dict[str, object],
) -> dict[str, tuple[SearchActionIdentity, ...]]:
    required: dict[str, tuple[SearchActionIdentity, ...]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        query = str(row["query"])
        generation = await generator.generate(
            RecallGenerationContext(
                query_id=query_id,
                original_query=query,
                query_spec=cast(Any, specs[query_id]),
            )
        )
        identities = tuple(
            identity
            for action in generation.action_batch.actions
            if (
                identity := search_action_identity(action.model_dump(mode="json"))
            )
            is not None
        )
        if not identities or len(identities) != len(set(identities)):
            raise ValueError(f"query-adaptive actions are invalid: {query_id}")
        required[query_id] = identities
    return required


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


def _sampling_bucket(
    row: dict[str, object],
    *,
    spec: object,
    length_cuts: list[int],
) -> tuple[str, str]:
    length = len(str(row["query"]).split())
    length_bucket = str(1 + sum(length > cut for cut in length_cuts))
    return _constraint_family(spec), length_bucket


def _select_stratified(
    rows: list[dict[str, object]],
    *,
    specs: dict[str, object],
    work_by_query: dict[str, ScheduledQueryActions],
    sample_size: int,
    max_missing_actions: int,
    max_raw_search_api_calls: int,
    max_results_per_action: int,
    candidate_policy: str,
) -> list[ScheduledQueryActions]:
    lengths = sorted(len(str(row["query"]).split()) for row in rows)
    cuts = [lengths[len(lengths) * index // 4] for index in (1, 2, 3)]

    buckets: dict[tuple[str, str], list[ScheduledQueryActions]] = defaultdict(list)
    row_by_id = {str(row["query_id"]): row for row in rows}
    for query_id, work in work_by_query.items():
        row = row_by_id[query_id]
        key = _sampling_bucket(
            row,
            spec=specs[query_id],
            length_cuts=cuts,
        )
        buckets[key].append(work)
    for key in buckets:
        buckets[key].sort(
            key=lambda item: _sampling_digest(
                item.query_id,
                candidate_policy=candidate_policy,
            )
        )

    selected: list[ScheduledQueryActions] = []
    actions = 0
    raw_calls = 0
    ordered_keys = sorted(buckets)
    while len(selected) < sample_size:
        progressed = False
        for key in ordered_keys:
            if not buckets[key] or len(selected) == sample_size:
                continue
            item = buckets[key].pop(0)
            action_count = len(item.missing_actions)
            raw_call_count = estimate_max_openalex_search_api_calls(
                item.missing_actions,
                max_results_per_action=max_results_per_action,
            )
            if (
                actions + action_count > max_missing_actions
                or raw_calls + raw_call_count > max_raw_search_api_calls
            ):
                continue
            selected.append(item)
            actions += action_count
            raw_calls += raw_call_count
            progressed = True
        if not progressed:
            break
    if len(selected) != sample_size:
        raise ValueError(
            "pilot sample cannot satisfy the requested size within the call cap"
        )
    return selected


def _schedule_today(
    work: list[ScheduledQueryActions],
    *,
    quota: dict[str, object],
    window: date,
    max_results_per_action: int,
) -> OpenAlexDailyTrainingSchedule:
    rows = cast(list[dict[str, object]], quota["rows"])
    capacity = {
        int(row["key_slot"]): int(row["max_search_calls"])
        - int(row["used_search_calls"])
        for row in rows
        if str(row["window"]) == window.isoformat()
    }
    hard_cap = {int(row["max_search_calls"]) for row in rows}
    if len(hard_cap) != 1:
        raise ValueError("OpenAlex key hard caps are inconsistent")
    assignments: dict[int, list[ScheduledQueryActions]] = defaultdict(list)
    used_actions = {slot: 0 for slot in capacity}
    used_raw_calls = {slot: 0 for slot in capacity}
    for item in work:
        action_count = len(item.missing_actions)
        raw_call_count = estimate_max_openalex_search_api_calls(
            item.missing_actions,
            max_results_per_action=max_results_per_action,
        )
        eligible = [
            slot
            for slot in sorted(capacity)
            if used_raw_calls[slot] + raw_call_count <= capacity[slot]
        ]
        if not eligible:
            raise ValueError("today's remaining OpenAlex quota cannot fit the pilot")
        slot = min(eligible, key=lambda value: (used_raw_calls[value], value))
        assignments[slot].append(item)
        used_actions[slot] += action_count
        used_raw_calls[slot] += raw_call_count
    shards = tuple(
        OpenAlexDailyShard(
            window=window,
            key_slot=slot,
            max_search_calls=next(iter(hard_cap)),
            planned_search_calls=used_actions[slot],
            queries=tuple(assignments[slot]),
        )
        for slot in sorted(assignments)
    )
    return OpenAlexDailyTrainingSchedule(
        schema_version="openalex-daily-training-schedule-v1",
        first_window=window,
        last_training_window=window,
        final_test_window=window + timedelta(days=1),
        key_count=len(capacity),
        max_search_calls_per_key=next(iter(hard_cap)),
        planned_search_calls=sum(used_actions.values()),
        available_training_search_calls=sum(capacity.values()),
        shards=shards,
    )


async def _main(args: argparse.Namespace) -> None:
    root = Path(args.workspace_root).resolve()
    resolve = lambda value: (root / value).resolve()  # noqa: E731
    partition_path = resolve(args.partition)
    context_manifest = resolve(args.context_manifest)
    audit_dir = resolve(args.audit_batches)
    handoff_path = resolve(args.handoff)
    handoff = cast(
        dict[str, object], json.loads(handoff_path.read_text(encoding="utf-8"))
    )
    if handoff.get("test_partition_touched") is not False:
        raise ValueError("training handoff does not prove final-test isolation")
    if args.quota_ledger is not None:
        if args.window is None:
            raise ValueError("current quota ledger usage requires an explicit window")
        window = args.window
        quota = _load_quota_usage_from_ledger(
            resolve(args.quota_ledger),
            window=window,
        )
    else:
        quota = cast(dict[str, object], handoff["quota_usage"])
        window = date.fromisoformat(str(quota["window"]))
        if args.window is not None and args.window != window:
            raise ValueError(
                "requested window does not match the frozen quota inventory"
            )

    all_rows = _jsonl(partition_path)
    specs = load_frozen_recall_query_specs(
        partition_path=partition_path,
        manifest_path=context_manifest,
    )
    no_hit_ids = _load_no_hit_ids(audit_dir)
    excluded = _load_excluded_query_ids(
        manifest_paths=[resolve(value) for value in args.exclude_manifest],
        receipt_roots=[resolve(value) for value in args.exclude_receipt_root],
    )
    eligible_rows = [
        row
        for row in all_rows
        if str(row["query_id"]) in no_hit_ids
        and str(row["query_id"]) not in excluded
    ]
    generator = QueryAdaptiveHighRecallGenerator(frozen_query_specs=specs)
    required = await _required_actions(
        eligible_rows, generator, cast(dict[str, object], specs)
    )
    receipt_roots = [Path(value) for value in cast(list[str], handoff["ordered_receipt_roots"])]
    completed = load_completed_search_action_identities(receipt_roots)
    work = build_missing_action_work(required, completed)
    work_by_query = {item.query_id: item for item in work}
    selectable_rows = [
        row for row in eligible_rows if str(row["query_id"]) in work_by_query
    ]
    selected = _select_stratified(
        selectable_rows,
        specs=cast(dict[str, object], specs),
        work_by_query=work_by_query,
        sample_size=args.sample_size,
        max_missing_actions=args.max_missing_calls,
        max_raw_search_api_calls=args.max_raw_search_api_calls,
        max_results_per_action=args.max_results_per_action,
        candidate_policy=generator.candidate_policy,
    )
    schedule = _schedule_today(
        selected,
        quota=quota,
        window=window,
        max_results_per_action=args.max_results_per_action,
    )
    selected_ids = {item.query_id for item in selected}
    reused = sum(
        len(set(required[query_id]) & set(completed.get(query_id, frozenset())))
        for query_id in selected_ids
    )
    required_count = sum(len(required[query_id]) for query_id in selected_ids)
    estimated_max_raw_search_api_calls = sum(
        estimate_max_openalex_search_api_calls(
            item.missing_actions,
            max_results_per_action=args.max_results_per_action,
        )
        for item in selected
    )
    inventory = {
        query_id: [
            item.model_dump(mode="json")
            for item in sorted(
                set(required[query_id]) & set(completed.get(query_id, frozenset())),
                key=lambda value: (
                    value.action_type,
                    value.search_mode,
                    value.normalized_text,
                ),
            )
        ]
        for query_id in sorted(selected_ids)
    }
    plan = OpenAlexDailyTrainingPlan(
        schema_version="openalex-daily-training-plan-v1",
        partition_sha256=_sha256(partition_path.read_bytes()),
        receipt_inventory_sha256=_sha256(_canonical_bytes(inventory)),
        required_query_count=len(selected),
        required_search_actions=required_count,
        reused_search_actions=reused,
        missing_search_actions=schedule.planned_search_calls,
        schedule=schedule,
    )
    row_by_id = {str(row["query_id"]): row for row in all_rows}
    stratum_counts: dict[str, int] = defaultdict(int)
    for item in selected:
        stratum_counts[_constraint_family(specs[item.query_id])] += 1
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    plan_bytes = _canonical_bytes(plan.model_dump(mode="json")) + b"\n"
    (output / "plan.json").write_bytes(plan_bytes)
    selected_rows = [row_by_id[item.query_id] for item in selected]
    (output / "pilot-partition.jsonl").write_bytes(
        b"".join(_canonical_bytes(row) + b"\n" for row in selected_rows)
    )
    manifest: dict[str, Any] = {
        "schema_version": "query-adaptive-recall-pilot-v1",
        "candidate_policy": generator.candidate_policy,
        "gold_visibility": "blind",
        "selection_uses_gold_content": False,
        "source_population": "OpenAlex+PASA mixed candidate Gold-miss auto_train",
        "source_no_hit_query_count": len(no_hit_ids),
        "eligible_after_prior_audit_exclusion": len(eligible_rows),
        "sample_role": args.sample_role,
        "sample_query_count": len(selected),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "required_action_count": required_count,
        "reused_completed_action_count": reused,
        "new_openalex_action_count": schedule.planned_search_calls,
        "estimated_max_raw_search_api_calls": estimated_max_raw_search_api_calls,
        "authorized_raw_search_api_call_cap": args.max_raw_search_api_calls,
        "max_results_per_action": args.max_results_per_action,
        "llm_request_count": 0,
        "test_partition_touched": False,
        "partition_sha256": _sha256(partition_path.read_bytes()),
        "context_manifest_sha256": _sha256(context_manifest.read_bytes()),
        "handoff_sha256": _sha256(handoff_path.read_bytes()),
        "quota_inventory_sha256": _sha256(_canonical_bytes(quota)),
        "excluded_prior_audit_query_count": len(excluded),
        "plan_sha256": _sha256(plan_bytes),
        "query_ids_sha256": _sha256(
            ("\n".join(sorted(selected_ids)) + "\n").encode()
        ),
    }
    if args.validation_gate is not None:
        gate_path = resolve(args.validation_gate)
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not isinstance(gate, dict) or gate.get("query_count") != len(selected):
            raise ValueError("validation gate does not match the frozen sample")
        manifest["validation_gate_sha256"] = _sha256(gate_path.read_bytes())
    elif args.sample_role == "independent_frozen_validation":
        raise ValueError("independent validation requires a frozen validation gate")
    (output / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--partition",
        default="data/training_private/partitions/pasa_auto_train.jsonl",
    )
    parser.add_argument("--context-manifest", required=True)
    parser.add_argument("--audit-batches", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--quota-ledger")
    parser.add_argument("--exclude-manifest", action="append", default=[])
    parser.add_argument("--exclude-receipt-root", action="append", default=[])
    parser.add_argument(
        "--sample-role",
        choices=("recall_policy_discovery", "independent_frozen_validation"),
        default="recall_policy_discovery",
    )
    parser.add_argument("--validation-gate")
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--max-missing-calls", type=int, default=1024)
    parser.add_argument("--max-raw-search-api-calls", type=int, required=True)
    parser.add_argument("--max-results-per-action", type=int, default=100)
    parser.add_argument("--window", type=date.fromisoformat)
    parser.add_argument("--output", required=True)
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
