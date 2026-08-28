"""Run hash-locked OpenAlex smoke checks and daily auto_train shards."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

from paper_search.learning.openalex_daily_runner import (
    batch_has_provider_quota_exhaustion,
    is_recoverable_canary_failure,
    is_scheduled_work_complete,
    load_scheduled_training_shard,
    select_smoke_work,
)
from paper_search.learning.openalex_daily_schedule import (
    OpenAlexDailyTrainingPlan,
    SQLiteOpenAlexDailyQuotaLedger,
    ScheduledMissingActionGenerator,
    ScheduledQueryActions,
    build_missing_action_work,
    current_openalex_quota_window,
    load_completed_search_action_identities,
)
from paper_search.learning.candidate_ceiling import QueryAdaptiveHighRecallGenerator
from paper_search.learning.unified_recall_context import load_frozen_recall_query_specs
from paper_search.recall_experiments.canary_inputs import (
    LoadedCanaryInput,
    RecallCase,
)
from paper_search.recall_experiments.canary_runtime import (
    build_live_runtime_bundle,
    load_runtime_profile,
    resolve_runtime_secrets,
)
from paper_search.recall_experiments.canary_service import RecallCanaryService
from paper_search.recall_experiments.generation.base import QueryGenerator
from paper_search.recall_experiments.recipes import load_recall_recipe


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def quota_is_exhausted(used_search_calls: int, max_search_calls: int) -> bool:
    return used_search_calls >= max_search_calls


def _reported_hard_cap_status(
    *,
    stopped_before_batch: bool,
    used_search_calls: int,
    max_search_calls: int,
) -> bool:
    return stopped_before_batch or quota_is_exhausted(
        used_search_calls,
        max_search_calls,
    )


def _load_plan(path: Path, expected_sha256: str) -> OpenAlexDailyTrainingPlan:
    content = path.read_bytes()
    if _sha256(content) != expected_sha256:
        raise ValueError("OpenAlex training plan hash does not match")
    return OpenAlexDailyTrainingPlan.model_validate_json(content)


def _identifier_map_bytes(rows: tuple[dict[str, object], ...]) -> bytes:
    gold_lists = [cast(list[object], row["gold_paper_ids"]) for row in rows]
    identifiers = {
        str(paper_id).casefold(): (
            "doi:10.48550/arxiv."
            + str(paper_id).casefold().removeprefix("arxiv:")
        )
        for gold_ids in gold_lists
        for paper_id in gold_ids
    }
    return _canonical_bytes(identifiers)


def _loaded_input(
    rows: tuple[dict[str, object], ...], identifier_map_bytes: bytes
) -> LoadedCanaryInput:
    cases = tuple(
        RecallCase(
            query_id=str(row["query_id"]),
            query=str(row["query"]),
            gold_paper_ids=tuple(
                str(item) for item in cast(list[object], row["gold_paper_ids"])
            ),
        )
        for row in rows
    )
    input_bytes = _canonical_bytes([case.model_dump(mode="json") for case in cases])
    return LoadedCanaryInput(
        input_kind="jsonl",
        cases=cases,
        evaluation_status="available",
        input_sha256=_sha256(input_bytes),
        identifier_map_bytes=identifier_map_bytes,
        identifier_map_sha256=_sha256(identifier_map_bytes),
    )


def _next_batch_paths(
    output_root: Path,
    *,
    batch_number: int,
) -> tuple[Path, Path]:
    stem = f"batch-{batch_number:04d}"
    retry = 0
    while True:
        suffix = "" if retry == 0 else f"-retry-{retry:03d}"
        run_path = output_root / "openalex" / f"{stem}{suffix}"
        capture_path = output_root / "captures" / "openalex" / f"{stem}{suffix}"
        if not run_path.exists() and not capture_path.exists():
            return run_path, capture_path
        retry += 1


async def _execute_work(
    *,
    workspace_root: Path,
    rows: tuple[dict[str, object], ...],
    work: tuple[ScheduledQueryActions, ...],
    output_root: Path,
    ledger_path: Path,
    window: date,
    key_slot: int,
    max_search_calls: int,
    chunk_size: int,
    profile_path: Path,
    recipe_path: Path,
    source_generator: QueryGenerator | None = None,
) -> dict[str, object]:
    if len(rows) != len(work):
        raise ValueError("scheduled rows and work must have identical query coverage")
    if [str(row.get("query_id")) for row in rows] != [
        item.query_id for item in work
    ]:
        raise ValueError("scheduled rows and work are not aligned")
    now = datetime.now(UTC)
    if current_openalex_quota_window(now) != window:
        raise ValueError("requested shard is not in the current OpenAlex quota window")
    profile = load_runtime_profile(profile_path)
    secrets = resolve_runtime_secrets(profile, openalex_key_slot=key_slot)
    recipe = load_recall_recipe(recipe_path)
    ledger = SQLiteOpenAlexDailyQuotaLedger(
        ledger_path,
        window=window,
        key_slot=key_slot,
        max_search_calls=max_search_calls,
        clock=lambda: datetime.now(UTC),
    )
    identifier_bytes = _identifier_map_bytes(rows)
    completed_queries = 0
    completed_actions = 0
    failed_batches = 0
    stopped_at_hard_cap = False
    external_quota_exhausted = False
    for offset in range(0, len(rows), chunk_size):
        if quota_is_exhausted(ledger.used_search_calls, max_search_calls):
            stopped_at_hard_cap = True
            break
        batch_rows = rows[offset : offset + chunk_size]
        batch_work = work[offset : offset + chunk_size]
        run_path, capture_path = _next_batch_paths(
            output_root,
            batch_number=offset // chunk_size + 1,
        )
        bundle = await build_live_runtime_bundle(
            profile=profile,
            secrets=secrets,
            loaded_recipe=recipe,
            capture_root=capture_path,
            search_dependency="openalex",
            openalex_attempt_gate=ledger,
        )
        try:
            try:
                await RecallCanaryService(workspace_root=workspace_root).run(
                    loaded_recipe=recipe,
                    loaded_input=_loaded_input(batch_rows, identifier_bytes),
                    runtime_bundle=bundle,
                    output_path=run_path,
                    generator_override=ScheduledMissingActionGenerator(
                        batch_work,
                        source=source_generator,
                    ),
                )
            except RuntimeError as error:
                if not is_recoverable_canary_failure(error):
                    raise
                failed_batches += 1
        finally:
            await bundle.aclose()
        completed_queries += len(batch_rows)
        completed_actions += sum(len(item.missing_actions) for item in batch_work)
        if batch_has_provider_quota_exhaustion(run_path):
            external_quota_exhausted = True
            break
    return {
        "window": window.isoformat(),
        "key_slot": key_slot,
        "completed_queries": completed_queries,
        "completed_actions": completed_actions,
        "failed_batches": failed_batches,
        "ledger_used_search_calls": ledger.used_search_calls,
        "stopped_at_hard_cap": _reported_hard_cap_status(
            stopped_before_batch=stopped_at_hard_cap,
            used_search_calls=ledger.used_search_calls,
            max_search_calls=max_search_calls,
        ),
        "external_quota_exhausted": external_quota_exhausted,
    }


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--partition", required=True)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    inspect = commands.add_parser("inspect", parents=[common])
    inspect.add_argument("--window", type=date.fromisoformat, required=True)
    inspect.add_argument("--key-slot", type=int, required=True)
    collect = commands.add_parser("collect", parents=[common])
    collect.add_argument("--window", type=date.fromisoformat, required=True)
    collect.add_argument("--key-slot", type=int, required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--ledger", required=True)
    collect.add_argument("--chunk-size", type=int, choices=(1, 2), default=2)
    collect.add_argument(
        "--profile",
        default="configs/recall_experiments/runtime/fixed-budget-openalex-live.yaml",
    )
    collect.add_argument(
        "--recipe",
    )
    collect.add_argument(
        "--policy",
        choices=("core4_semantic_boolean", "query_adaptive_high_recall"),
        default="core4_semantic_boolean",
    )
    collect.add_argument("--context-manifest")
    smoke = commands.add_parser("smoke", parents=[common])
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--ledger", required=True)
    smoke.add_argument("--key-count", type=int, default=11)
    smoke.add_argument(
        "--profile",
        default="configs/recall_experiments/runtime/fixed-budget-openalex-live.yaml",
    )
    smoke.add_argument(
        "--recipe",
        default="configs/recall_experiments/methods/core4-semantic-boolean-live.yaml",
    )
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


async def _run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.workspace_root).resolve()
    plan_path = _resolve(root, args.plan)
    partition_path = _resolve(root, args.partition)
    if args.command == "inspect":
        prepared = load_scheduled_training_shard(
            plan_path=plan_path,
            expected_plan_sha256=args.plan_sha256,
            partition_path=partition_path,
            window=args.window,
            key_slot=args.key_slot,
        )
        return prepared.model_dump(mode="json", exclude={"rows", "work"})
    if args.command == "collect":
        output = _resolve(root, args.output)
        receipt_roots = [output] if output.is_dir() else []
        prepared = load_scheduled_training_shard(
            plan_path=plan_path,
            expected_plan_sha256=args.plan_sha256,
            partition_path=partition_path,
            window=args.window,
            key_slot=args.key_slot,
            completed_receipt_roots=receipt_roots,
        )
        if not prepared.work:
            return {
                "window": args.window.isoformat(),
                "key_slot": args.key_slot,
                "completed_queries": 0,
                "completed_actions": 0,
                "status": "already_complete",
            }
        source_generator: QueryGenerator | None = None
        if args.policy == "query_adaptive_high_recall":
            if args.context_manifest is None:
                raise ValueError(
                    "query-adaptive collection requires --context-manifest"
                )
            specs = load_frozen_recall_query_specs(
                partition_path=partition_path,
                manifest_path=_resolve(root, args.context_manifest),
            )
            source_generator = QueryAdaptiveHighRecallGenerator(
                frozen_query_specs=specs
            )
        recipe_value = args.recipe or (
            "configs/recall_experiments/methods/"
            "query-adaptive-high-recall-live.yaml"
            if args.policy == "query_adaptive_high_recall"
            else "configs/recall_experiments/methods/"
            "core4-semantic-boolean-live.yaml"
        )
        return await _execute_work(
            workspace_root=root,
            rows=prepared.rows,
            work=prepared.work,
            output_root=output,
            ledger_path=_resolve(root, args.ledger),
            window=args.window,
            key_slot=args.key_slot,
            max_search_calls=prepared.max_search_calls,
            chunk_size=args.chunk_size,
            profile_path=_resolve(root, args.profile),
            recipe_path=_resolve(root, recipe_value),
            source_generator=source_generator,
        )
    plan = _load_plan(plan_path, args.plan_sha256)
    smoke_window = current_openalex_quota_window(datetime.now(UTC))
    output = _resolve(root, args.output)
    results: list[dict[str, object]] = []
    for key_slot in range(1, args.key_count + 1):
        smoke_work = select_smoke_work(plan, key_slot=key_slot)
        source_shard = next(
            shard
            for shard in plan.schedule.shards
            if shard.key_slot == key_slot
            and any(item.query_id == smoke_work.query_id for item in shard.queries)
        )
        key_output = output / f"key-{key_slot:02d}"
        completed = (
            load_completed_search_action_identities([key_output])
            if key_output.is_dir()
            else {}
        )
        remaining = build_missing_action_work(
            {smoke_work.query_id: smoke_work.missing_actions},
            completed,
        )
        if not remaining:
            results.append({"key_slot": key_slot, "status": "already_complete"})
            continue
        prepared = load_scheduled_training_shard(
            plan_path=plan_path,
            expected_plan_sha256=args.plan_sha256,
            partition_path=partition_path,
            window=source_shard.window,
            key_slot=key_slot,
        )
        row_by_query = {str(row["query_id"]): row for row in prepared.rows}
        result = await _execute_work(
            workspace_root=root,
            rows=(row_by_query[smoke_work.query_id],),
            work=(remaining[0],),
            output_root=key_output,
            ledger_path=_resolve(root, args.ledger),
            window=smoke_window,
            key_slot=key_slot,
            max_search_calls=plan.schedule.max_search_calls_per_key,
            chunk_size=1,
            profile_path=_resolve(root, args.profile),
            recipe_path=_resolve(root, args.recipe),
        )
        result["status"] = (
            "succeeded"
            if is_scheduled_work_complete([remaining[0]], [key_output])
            else "failed"
        )
        results.append(result)
    return {
        "schema_version": "openalex-key-smoke-v1",
        "window": smoke_window.isoformat(),
        "key_count": args.key_count,
        "results": results,
    }


def main(argv: list[str] | None = None) -> None:
    result = asyncio.run(_run(build_parser().parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
