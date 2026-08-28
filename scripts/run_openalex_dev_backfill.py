"""Freeze, collect, and finalize exact-config auto_dev OpenAlex backfills."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, cast

from paper_search.domain.models import QuerySpec
from paper_search.learning.candidate_ceiling import Core4SemanticBooleanQueryGenerator
from paper_search.learning.openalex_daily_schedule import (
    RecallGenerationContext,
    ScheduledQueryActions,
    SearchActionIdentity,
    search_action_identity,
)
from paper_search.learning.openalex_dev_backfill import (
    CORE4_POLICY,
    SCHEMA_VERSION,
    assign_work_to_keys,
    build_work,
    canonical_bytes,
    inventory_core4_receipts,
    load_auto_dev_rows,
    load_manifest_query_ids,
    manifest_digest,
    select_query_ids,
    sha256_file,
)


def _json_write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload) + b"\n")


async def _required(rows: tuple[dict[str, object], ...]) -> dict[str, tuple[SearchActionIdentity, ...]]:
    generator = Core4SemanticBooleanQueryGenerator()
    output: dict[str, tuple[SearchActionIdentity, ...]] = {}
    for row in rows:
        query_id = str(row["query_id"])
        result = await generator.generate(
            RecallGenerationContext(
                query_id=query_id,
                original_query=str(row["query"]),
                query_spec=QuerySpec(
                    original_query=str(row["query"]), research_goal=str(row["query"])
                ),
            )
        )
        identities = tuple(
            identity
            for action in result.action_batch.actions
            if (identity := search_action_identity(action.model_dump(mode="json")))
            is not None
        )
        if not identities or len(identities) > 6 or len(set(identities)) != len(identities):
            raise ValueError(f"unexpected core4 action contract for {query_id}")
        output[query_id] = identities
    return output


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _work_json(work: tuple[ScheduledQueryActions, ...]) -> list[dict[str, object]]:
    return [
        {
            "query_id": item.query_id,
            "actions": [action.model_dump(mode="json") for action in item.missing_actions],
        }
        for item in work
    ]


def audit(args: argparse.Namespace) -> int:
    root = Path(args.workspace_root).resolve()
    partition = (root / args.partition).resolve()
    latest = (root / args.latest_manifest).resolve()
    rows = load_auto_dev_rows(partition)
    if sha256_file(latest) != args.latest_manifest_sha256:
        raise ValueError("latest manifest hash does not match")
    latest_ids = load_manifest_query_ids(latest)
    row_ids = {str(row["query_id"]) for row in rows}
    latest_overlap = latest_ids & row_ids
    required = asyncio.run(_required(rows))
    receipt_roots = [(root / value).resolve() for value in args.receipt_root]
    inventory = inventory_core4_receipts(
        receipt_roots=receipt_roots, required=required
    )
    complete = set(inventory["complete_query_ids"])
    conflicts = set(inventory["conflict_query_ids"])
    required_ids = sorted(required)
    if args.selection == "repair":
        selected = tuple(sorted(conflicts))
    else:
        selected = select_query_ids(
            required_query_ids=required_ids,
            complete_query_ids=complete,
            conflict_query_ids=conflicts,
            target_count=args.target_count,
        )
    new_query_ids = tuple(
        query_id for query_id in selected if query_id not in complete
    )
    if args.selection == "repair":
        new_query_ids = selected
    work = build_work(new_query_ids, required)
    by_key = assign_work_to_keys(
        work, key_count=args.key_count, max_calls=args.max_calls_per_key
    )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_digest": "pending",
        "selection": args.selection,
        "window": args.window,
        "action_policy": CORE4_POLICY,
        "partition_path": _relative(root, partition),
        "partition_sha256": sha256_file(partition),
        "latest_batch_manifest_path": _relative(root, latest),
        "latest_batch_manifest_sha256": sha256_file(latest),
        "latest_batch_query_count": len(latest_ids),
        "latest_batch_manifest_overlap_count": len(latest_overlap),
        "latest_batch_manifest_outside_current_partition_count": len(
            latest_ids - row_ids
        ),
        "receipt_roots": [_relative(root, value) for value in receipt_roots],
        "files_scanned": inventory["files_scanned"],
        "parse_error_count": inventory["parse_error_count"],
        "required_query_count": len(required),
        "existing_complete_query_count": len(complete),
        "existing_conflict_query_count": len(conflicts),
        "existing_missing_query_count": len(inventory["missing_query_ids"]),
        "target_complete_query_count": args.target_count if args.selection != "repair" else len(complete),
        "selected_query_count": len(selected),
        "selected_query_ids": list(selected),
        "new_query_count": len(new_query_ids),
        "new_query_ids": list(new_query_ids),
        "conflict_query_ids": sorted(conflicts),
        "white_list_by_key": {
            str(key): _work_json(values) for key, values in sorted(by_key.items())
        },
        "planned_openalex_calls": sum(len(item.missing_actions) for item in work),
        "max_calls_per_key": args.max_calls_per_key,
        "key_count": args.key_count,
        "source_inventory_digest": manifest_digest(
            {
                "complete": sorted(complete),
                "conflict": sorted(conflicts),
                "missing": inventory["missing_query_ids"],
            }
        ),
    }
    manifest["manifest_digest"] = manifest_digest(manifest)
    output = (root / args.output).resolve()
    _json_write(output, manifest)
    _json_write(
        output.with_name("next-whitelist.json"),
        {
            "schema_version": SCHEMA_VERSION,
            "manifest_path": _relative(root, output),
            "manifest_sha256": sha256_file(output),
            "query_ids": list(selected),
            "planned_openalex_calls": manifest["planned_openalex_calls"],
        },
    )
    print(json.dumps({"manifest": str(output), **{key: manifest[key] for key in (
        "selection", "existing_complete_query_count", "existing_conflict_query_count",
        "existing_missing_query_count", "selected_query_count", "planned_openalex_calls",
    )}}, ensure_ascii=False, sort_keys=True))
    return 0


def _load_manifest(root: Path, path: str) -> dict[str, Any]:
    payload = json.loads((root / path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("backfill manifest schema mismatch")
    return cast(dict[str, Any], payload)


async def collect(args: argparse.Namespace) -> int:
    root = Path(args.workspace_root).resolve()
    manifest = _load_manifest(root, args.manifest)
    if manifest["window"] != args.window:
        raise ValueError("collection window differs from manifest")
    by_key = manifest["white_list_by_key"]
    entries = by_key.get(str(args.key_slot), [])
    if not entries:
        raise ValueError(f"manifest has no work for key slot {args.key_slot}")
    partition = (root / manifest["partition_path"]).resolve()
    rows = {str(row["query_id"]): row for row in load_auto_dev_rows(partition)}
    work = tuple(
        ScheduledQueryActions(
            query_id=str(item["query_id"]),
            missing_actions=tuple(
                SearchActionIdentity.model_validate(action)
                for action in item["actions"]
            ),
        )
        for item in entries
    )
    selected_rows = tuple(rows[item.query_id] for item in work)
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError(f"immutable output root already exists: {output}")
    sys.path.insert(0, str(root / "scripts"))
    from run_openalex_daily_training import _execute_work  # noqa: PLC0415

    result = await _execute_work(
        workspace_root=root,
        rows=selected_rows,
        work=work,
        output_root=output,
        ledger_path=(root / args.ledger).resolve(),
        window=date.fromisoformat(args.window),
        key_slot=args.key_slot,
        max_search_calls=int(manifest["max_calls_per_key"]),
        chunk_size=args.chunk_size,
        profile_path=(root / args.profile).resolve(),
        recipe_path=(root / args.recipe).resolve(),
    )
    result.update(
        {
            "manifest": args.manifest,
            "manifest_sha256": sha256_file(root / args.manifest),
            "key_slot": args.key_slot,
        }
    )
    _json_write(output / "key-result.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def finalize(args: argparse.Namespace) -> int:
    root = Path(args.workspace_root).resolve()
    manifest = _load_manifest(root, args.manifest)
    partition = (root / manifest["partition_path"]).resolve()
    load_auto_dev_rows(partition)
    selected_ids = manifest.get("new_query_ids", manifest["selected_query_ids"])
    required = {
        query_id: tuple(
            SearchActionIdentity.model_validate(action)
            for action in next(
                item["actions"]
                for key_entries in manifest["white_list_by_key"].values()
                for item in key_entries
                if item["query_id"] == query_id
            )
        )
        for query_id in selected_ids
    }
    output_roots = [
        (root / value).resolve()
        for value in args.output_root
    ]
    inventory = inventory_core4_receipts(
        receipt_roots=output_roots, required=required
    )
    selected = set(selected_ids)
    success = sorted(selected & set(inventory["complete_query_ids"]))
    failure = sorted(selected - set(success))
    destination = (root / args.output).resolve()
    _json_write(destination / "success-query-ids.json", success)
    _json_write(destination / "failure-query-ids.json", failure)
    _json_write(
        destination / "next-whitelist.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_manifest": args.manifest,
            "source_manifest_sha256": sha256_file(root / args.manifest),
            "query_ids": failure,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest": args.manifest,
        "selected_query_count": len(selected),
        "success_query_count": len(success),
        "failure_query_count": len(failure),
        "files_scanned": inventory["files_scanned"],
        "parse_error_count": inventory["parse_error_count"],
        "openalex_calls": sum(
            json.loads((root / args.manifest).read_text(encoding="utf-8"))["planned_openalex_calls"]
            for _ in [0]
        ),
    }
    _json_write(destination / "finalize-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--workspace-root", default=".")
    a.add_argument("--partition", required=True)
    a.add_argument("--latest-manifest", required=True)
    a.add_argument("--latest-manifest-sha256", required=True)
    a.add_argument("--receipt-root", action="append", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--window", required=True)
    a.add_argument("--selection", choices=("phase", "repair"), default="phase")
    a.add_argument("--target-count", type=int, default=300)
    a.add_argument("--key-count", type=int, default=11)
    a.add_argument("--max-calls-per-key", type=int, default=900)
    c = sub.add_parser("collect")
    c.add_argument("--workspace-root", default=".")
    c.add_argument("--manifest", required=True)
    c.add_argument("--window", required=True)
    c.add_argument("--key-slot", type=int, required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--ledger", required=True)
    c.add_argument("--chunk-size", type=int, choices=(1, 2), default=2)
    c.add_argument("--profile", default="configs/recall_experiments/runtime/fixed-budget-openalex-live.yaml")
    c.add_argument("--recipe", default="configs/recall_experiments/methods/core4-semantic-boolean-live.yaml")
    f = sub.add_parser("finalize")
    f.add_argument("--workspace-root", default=".")
    f.add_argument("--manifest", required=True)
    f.add_argument("--output-root", action="append", required=True)
    f.add_argument("--output", required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "audit":
        raise SystemExit(audit(args))
    if args.command == "collect":
        raise SystemExit(asyncio.run(collect(args)))
    raise SystemExit(finalize(args))


if __name__ == "__main__":
    main()
