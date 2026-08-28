"""Freeze, collect, and evaluate a Gold-blind OpenAlex cursor-depth validation."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.evaluation.dataset import IdentifierMap  # noqa: E402
from paper_search.evaluation.predictions import (  # noqa: E402
    paper_evaluation_aliases,
    paper_matches_evaluation_ids,
)
from paper_search.learning.f5_production_deployment import (  # noqa: E402
    load_f5_production_ranker_bytes,
)
from paper_search.learning.large_scale_fusion_training import (  # noqa: E402
    FusionTrainingPackage,
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
)
from paper_search.learning.openalex_daily_schedule import (  # noqa: E402
    SQLiteOpenAlexDailyQuotaLedger,
    current_openalex_quota_window,
    search_action_identity,
)
from paper_search.recall_experiments.canary_runtime import (  # noqa: E402
    build_live_runtime_bundle,
    load_runtime_profile,
    resolve_runtime_secrets,
)
from paper_search.recall_experiments.contracts import (  # noqa: E402
    RecallActionBatch,
    assert_no_forbidden_identifier_keys_or_patterns,
)
from paper_search.recall_experiments.recipes import load_recall_recipe  # noqa: E402
from scripts.evaluate_cross_vocabulary_f5_topk import (  # noqa: E402
    _load_production_paths,
    aggregate_comparison,
    ranking_metrics,
)
from scripts.prepare_online_miss_provider_validation import (  # noqa: E402
    collect_prior_query_ids,
    primary_stratum,
)
from scripts.run_cross_vocabulary_openalex_validation import (  # noqa: E402
    _online_only_package,
)
from scripts.run_provider_recall_comparison import (  # noqa: E402
    _load_production_identifier_context,
)


DEFAULT_OUTPUT = Path(
    "data/training_private/recall_policy/openalex-depth-continuation64-v1"
)
DEFAULT_AUDIT_ROOT = Path(
    "data/training_private/evaluations/"
    "online-recall-ceiling-production-f5-21429-v1"
)
DEFAULT_HANDOFF = Path(
    "data/training_private/training_runs/"
    "openalex-pasa-high-recall-expanded-21429-v1/"
    "ranking-training-handoff-expanded.json"
)
DEFAULT_PARTITION = Path("data/training_private/partitions/pasa_auto_train.jsonl")
DEFAULT_BUNDLE = Path(
    "artifacts/models/gated-feature-fusion-18314-unified-context-v3-v1/weights.bundle"
)
DEFAULT_SELECTION = Path("artifacts/models/production-document-ranker-selection.json")
DEFAULT_LOCK = Path("deliverables/evaluator/live-evaluator.lock.yaml")
DEFAULT_PROFILE = Path(
    "configs/recall_experiments/runtime/fixed-budget-openalex-live.yaml"
)
DEFAULT_RECIPE = Path(
    "configs/recall_experiments/methods/scheme-b-semantic-backfill-live.yaml"
)
DEFAULT_QUERY_COUNT = 64
DEFAULT_RAW_REQUEST_CAP = 64
DEFAULT_CANDIDATE_CAP = 200
_SELECTION_SEED = "openalex-depth-continuation64-v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _jsonl(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"JSONL row is not an object: {path}")
        rows.append(raw)
    return rows


def _forbidden_request_key(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if "gold" in normalized or "final_test" in normalized:
                return str(key)
            found = _forbidden_request_key(nested)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            found = _forbidden_request_key(nested)
            if found is not None:
                return found
    return None


def build_gold_blind_request_row(proposal: Mapping[str, object]) -> dict[str, object]:
    """Project a local proposal into the only payload allowed for live collection."""

    action = proposal.get("source_action")
    if not isinstance(action, Mapping):
        raise ValueError("depth proposal has no source action")
    batch = RecallActionBatch(actions=[dict(action)])
    dumped_action = batch.actions[0].model_dump(mode="json")
    identity = search_action_identity(dumped_action)
    if identity is None or identity.search_mode != "lexical":
        raise ValueError("depth continuation requires one lexical text action")
    cursor = proposal.get("cursor")
    if not isinstance(cursor, str) or not cursor.strip() or cursor == "*":
        raise ValueError("depth proposal has no frozen continuation cursor")
    source_hit_count = proposal.get("source_hit_count")
    if source_hit_count != 50:
        raise ValueError("depth proposal must bind an exact 50-result first page")
    row: dict[str, object] = {
        "schema_version": "openalex-depth-request-v1",
        "query_id": str(proposal["query_id"]),
        "derived_query_text": identity.normalized_text,
        "filters": {},
        "cursor": cursor,
        "limit": 50,
        "rank_offset": 50,
        "source_action": dumped_action,
        "source_candidate_namespace": str(
            proposal.get("source_candidate_namespace", "default")
        ),
        "source_snapshot_sha256": str(proposal["source_snapshot_sha256"]),
        "source_retrieval_sha256": str(proposal["source_retrieval_sha256"]),
    }
    verify_gold_blind_request_plan([row])
    return row


def verify_gold_blind_request_plan(rows: Sequence[Mapping[str, object]]) -> None:
    """Reject labels, identifiers, or partition material outside the live contract."""

    if not rows:
        raise ValueError("Gold-blind request plan is empty")
    expected_keys = {
        "schema_version",
        "query_id",
        "derived_query_text",
        "filters",
        "cursor",
        "limit",
        "rank_offset",
        "source_action",
        "source_candidate_namespace",
        "source_snapshot_sha256",
        "source_retrieval_sha256",
    }
    identities: set[tuple[str, str]] = set()
    query_ids: set[str] = set()
    for row in rows:
        forbidden = _forbidden_request_key(row)
        if forbidden is not None:
            raise ValueError(f"Gold-blind request contains forbidden key: {forbidden}")
        if set(row) != expected_keys:
            raise ValueError("Gold-blind request shape changed")
        query_id = row.get("query_id")
        cursor = row.get("cursor")
        text = row.get("derived_query_text")
        if (
            row.get("schema_version") != "openalex-depth-request-v1"
            or not isinstance(query_id, str)
            or not query_id.startswith("AutoScholarQuery_train_")
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(cursor, str)
            or not cursor.strip()
            or cursor == "*"
            or row.get("limit") != 50
            or row.get("rank_offset") != 50
            or row.get("filters") != {}
        ):
            raise ValueError("Gold-blind request values are invalid")
        if query_id in query_ids or (text, cursor) in identities:
            raise ValueError("Gold-blind request identities are duplicated")
        query_ids.add(query_id)
        identities.add((text, cursor))
        action = row.get("source_action")
        if not isinstance(action, Mapping):
            raise ValueError("Gold-blind request source action is invalid")
        assert_no_forbidden_identifier_keys_or_patterns(dict(action))


def select_disjoint_depth_rows(
    rows: Sequence[dict[str, object]],
    *,
    prior_query_ids: set[str],
    limit: int,
    seed: str,
) -> list[dict[str, object]]:
    """Select a stable, disjoint continuation sample with exact page-one depth."""

    if limit <= 0 or not seed:
        raise ValueError("depth selection limit and seed are required")
    eligible: dict[str, dict[str, object]] = {}
    for row in rows:
        query_id = str(row.get("query_id", ""))
        cursor = row.get("cursor")
        if (
            not query_id
            or query_id in prior_query_ids
            or row.get("source_hit_count") != 50
            or not isinstance(cursor, str)
            or not cursor.strip()
            or cursor == "*"
        ):
            continue
        if query_id in eligible:
            raise ValueError(f"duplicate depth proposal: {query_id}")
        eligible[query_id] = row
    ordered = sorted(
        eligible.values(),
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['query_id']}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ordered) < limit:
        raise ValueError(
            f"insufficient exact depth proposals: required={limit} available={len(ordered)}"
        )
    return ordered[:limit]


def _rank_misses(audit_root: Path) -> dict[str, list[str]]:
    misses: dict[str, list[str]] = {}
    for path in sorted(audit_root.glob("rank-shard-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("gold_ranks"):
                    continue
                query_id = str(row.get("query_id", ""))
                labels = row.get("labels")
                if not query_id or not isinstance(labels, list):
                    raise ValueError("online recall rank row is invalid")
                misses[query_id] = [str(value) for value in labels]
    if not misses:
        raise ValueError("online recall audit has no no-hit rows")
    return misses


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{label} is invalid: {path}")
    return raw


def _generation_path(retrieval_path: Path) -> Path:
    parts = list(retrieval_path.parts)
    try:
        index = parts.index("retrieval")
    except ValueError as error:
        raise ValueError(f"invalid retrieval path: {retrieval_path}") from error
    parts[index] = "generation"
    return Path(*parts)


def _snapshot_path(retrieval_path: Path, relative: str) -> Path:
    for parent in retrieval_path.parents:
        if parent.name.startswith("batch-") and parent.parent.name == "openalex":
            return (
                parent.parent.parent
                / "captures"
                / "openalex"
                / parent.name
                / relative
            ).resolve()
    raise ValueError(f"cannot resolve capture root for {retrieval_path}")


def _usable_source_actions(
    receipt_paths: Sequence[Path],
) -> list[dict[str, object]]:
    final_by_identity: dict[object, dict[str, object]] = {}
    for retrieval_path in receipt_paths:
        generation_path = _generation_path(retrieval_path)
        generation = _json_object(generation_path, label="generation receipt")
        retrieval = _json_object(retrieval_path, label="retrieval receipt")
        raw_actions = generation.get("actions")
        raw_results = retrieval.get("results")
        if not isinstance(raw_actions, list) or not isinstance(raw_results, list):
            continue
        actions: dict[str, dict[str, object]] = {}
        identities: dict[str, object] = {}
        for raw_action in raw_actions:
            if not isinstance(raw_action, dict) or not isinstance(
                raw_action.get("action_id"), str
            ):
                continue
            identity = search_action_identity(raw_action)
            if identity is None or identity.search_mode != "lexical":
                continue
            action_id = str(raw_action["action_id"])
            actions[action_id] = cast(dict[str, object], raw_action)
            identities[action_id] = identity
        provenance = generation.get("generation_provenance")
        namespace = "default"
        if isinstance(provenance, Mapping):
            value = provenance.get("candidate_source_namespace")
            if isinstance(value, str) and value.strip():
                namespace = value.strip()
        for result in raw_results:
            if not isinstance(result, dict):
                continue
            action_id = result.get("action_id")
            hits = result.get("hits")
            errors = result.get("errors")
            result_provenance = result.get("provenance")
            if (
                not isinstance(action_id, str)
                or action_id not in identities
                or not isinstance(hits, list)
                or len(hits) != 50
                or errors != []
                or result.get("infrastructure_failure") is True
                or not isinstance(result_provenance, Mapping)
                or result_provenance.get("provider") != "openalex"
            ):
                continue
            refs_raw = result_provenance.get("snapshot_refs")
            try:
                refs = json.loads(refs_raw) if isinstance(refs_raw, str) else None
            except json.JSONDecodeError:
                refs = None
            if not isinstance(refs, list) or len(refs) != 1 or not isinstance(refs[0], dict):
                continue
            relative = refs[0].get("snapshot_path")
            expected_sha = refs[0].get("response_sha256")
            if not isinstance(relative, str) or not isinstance(expected_sha, str):
                continue
            snapshot_path = _snapshot_path(retrieval_path, relative)
            if not snapshot_path.is_file() or _sha256_file(snapshot_path) != expected_sha:
                continue
            snapshot = _json_object(snapshot_path, label="OpenAlex response snapshot")
            meta = snapshot.get("meta")
            raw_works = snapshot.get("results")
            cursor = meta.get("next_cursor") if isinstance(meta, Mapping) else None
            if (
                not isinstance(raw_works, list)
                or len(raw_works) != 50
                or not isinstance(cursor, str)
                or not cursor.strip()
                or cursor == "*"
            ):
                continue
            identity = identities[action_id]
            final_by_identity[(namespace, identity)] = {
                "source_action": actions[action_id],
                "source_candidate_namespace": namespace,
                "cursor": cursor,
                "source_snapshot_sha256": expected_sha,
                "source_retrieval_sha256": _sha256_file(retrieval_path),
                "source_retrieval_path": str(retrieval_path),
                "source_hit_count": 50,
            }
    def priority(row: Mapping[str, object]) -> tuple[int, str]:
        action = cast(Mapping[str, object], row["source_action"])
        action_id = str(action.get("action_id", ""))
        return (
            0 if action_id == "policy-1" else 1 if action_id.startswith("policy-") else 2,
            action_id,
        )

    return sorted(final_by_identity.values(), key=priority)


def _source_proposal(
    package: FusionTrainingPackage,
    query_id: str,
    labels: Sequence[str],
    receipt_paths: Sequence[Path],
) -> dict[str, object] | None:
    actions = _usable_source_actions(receipt_paths)
    if not actions:
        return None
    row = package.rows_by_query_id[query_id]
    return {
        "query_id": query_id,
        "query": str(row["query"]),
        "gold_paper_ids": [str(value) for value in row["gold_paper_ids"]],
        "labels": list(labels),
        "signal": primary_stratum(labels),
        **actions[0],
    }


def _baseline_snapshot(query: object) -> dict[str, object]:
    candidates = getattr(query, "candidates")
    return {
        "query_id": getattr(query, "query_id"),
        "candidate_count": len(candidates),
        "candidate_aliases": sorted(
            {
                alias
                for candidate in candidates
                for alias in paper_evaluation_aliases(candidate.paper)
            }
        ),
    }


def prepare(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.workspace_root).resolve()
    resolve = lambda value: (root / value).resolve()  # noqa: E731
    output = resolve(args.output)
    if output.exists() and (output / "manifest.json").exists():
        raise ValueError("frozen depth validation package already exists")
    audit_root = resolve(args.audit_root)
    handoff_path = resolve(args.handoff)
    partition_path = resolve(args.partition)
    bundle_path = resolve(args.production_bundle)
    misses = _rank_misses(audit_root)
    package = _online_only_package(
        load_training_package(
            handoff_path=handoff_path,
            partition_path=partition_path,
            production_bundle_path=bundle_path,
        )
    )
    receipt_index = index_training_receipts(package)
    prior_query_ids = collect_prior_query_ids(
        root / "data/training_private/recall_policy",
        ignored_roots=(output,),
    )
    prior_query_ids.update(
        collect_prior_query_ids(
            root / "data/training_private/online_recall",
            ignored_roots=(output,),
        )
    )
    ordered_ids = sorted(
        (
            query_id
            for query_id in package.query_ids
            if query_id in misses and query_id not in prior_query_ids
        ),
        key=lambda query_id: hashlib.sha256(
            f"{_SELECTION_SEED}\0{query_id}".encode("utf-8")
        ).hexdigest(),
    )
    proposals: list[dict[str, object]] = []
    scanned = 0
    for query_id in ordered_ids:
        scanned += 1
        proposal = _source_proposal(
            package,
            query_id,
            misses[query_id],
            receipt_index[query_id],
        )
        if proposal is not None:
            proposals.append(proposal)
        if len(proposals) >= args.query_count:
            break
    selected = select_disjoint_depth_rows(
        proposals,
        prior_query_ids=prior_query_ids,
        limit=args.query_count,
        seed=_SELECTION_SEED,
    )
    query_ids = tuple(str(row["query_id"]) for row in selected)
    selected_package = replace(
        package,
        query_ids=query_ids,
        rows_by_query_id={
            query_id: package.rows_by_query_id[query_id] for query_id in query_ids
        },
    )
    selected_receipts = index_training_receipts(selected_package)
    baseline_rows: list[dict[str, object]] = []
    for query_id in query_ids:
        query = build_document_ranking_query(
            selected_package,
            query_id,
            selected_receipts[query_id],
        )
        if any(
            paper_matches_evaluation_ids(candidate.paper, query.gold_paper_ids)
            for candidate in query.candidates
        ):
            raise ValueError(f"selected query is no longer an online-only miss: {query_id}")
        baseline_rows.append(_baseline_snapshot(query))
    partition_rows = [
        {
            "dataset": "pasa",
            "query_id": row["query_id"],
            "query": row["query"],
            "gold_paper_ids": row["gold_paper_ids"],
            "role": "training",
            "split": "auto_train",
        }
        for row in selected
    ]
    request_rows = [build_gold_blind_request_row(row) for row in selected]
    verify_gold_blind_request_plan(request_rows)
    actions = {
        str(row["query_id"]): {"actions": [row["source_action"]]}
        for row in selected
    }
    diagnostics = [
        {
            "query_id": row["query_id"],
            "signal": row["signal"],
            "labels": row["labels"],
            "source_action_id": cast(Mapping[str, object], row["source_action"])[
                "action_id"
            ],
            "source_candidate_namespace": row["source_candidate_namespace"],
            "source_snapshot_sha256": row["source_snapshot_sha256"],
            "source_retrieval_sha256": row["source_retrieval_sha256"],
            "source_hit_count": row["source_hit_count"],
            "rank_offset": 50,
        }
        for row in selected
    ]
    partition_bytes = _jsonl(partition_rows)
    request_bytes = _jsonl(request_rows)
    actions_bytes = _canonical_bytes(actions)
    diagnostics_bytes = _jsonl(diagnostics)
    baseline_bytes = _jsonl(baseline_rows)
    signal_counts = Counter(str(row["signal"]) for row in selected)
    prior_bytes = _canonical_bytes(sorted(prior_query_ids))
    manifest: dict[str, object] = {
        "schema_version": "openalex-depth-continuation-validation-v1",
        "purpose": "disjoint-current-online-miss-second-page-recall-test",
        "query_count": len(selected),
        "selection_policy": "sha256-disjoint-current-online-miss-exact-page1-v1",
        "selection_seed": _SELECTION_SEED,
        "scanned_query_count": scanned,
        "eligible_source_query_count": len(proposals),
        "excluded_prior_query_count": len(prior_query_ids),
        "prior_query_inventory_sha256": _sha256_bytes(prior_bytes),
        "signal_counts": dict(sorted(signal_counts.items())),
        "source_page_size": 50,
        "continuation_page_size": 50,
        "provider_rank_offset": 50,
        "max_raw_openalex_requests": args.max_raw_requests,
        "request_retry_policy": "no-retry-one-attempt-per-query",
        "inputs": {
            "audit_summary_sha256": _sha256_file(audit_root / "summary.json"),
            "audit_progress_sha256": _sha256_file(audit_root / "progress.json"),
            "handoff_sha256": _sha256_file(handoff_path),
            "partition_sha256": _sha256_file(partition_path),
            "production_bundle_sha256": _sha256_file(bundle_path),
        },
        "outputs": {
            "partition_sha256": _sha256_bytes(partition_bytes),
            "request_plan_sha256": _sha256_bytes(request_bytes),
            "actions_sha256": _sha256_bytes(actions_bytes),
            "diagnostics_sha256": _sha256_bytes(diagnostics_bytes),
            "baseline_candidates_sha256": _sha256_bytes(baseline_bytes),
        },
        "request_plan_gold_blind": True,
        "pasa_used_as_online_candidate_source": False,
        "llm_requests_made": 0,
        "test_partition_touched": False,
        "production_lock_modified": False,
        "training_started": False,
    }
    _write_immutable(output / "partition.jsonl", partition_bytes)
    _write_immutable(output / "request-plan.jsonl", request_bytes)
    _write_immutable(output / "actions.json", actions_bytes)
    _write_immutable(output / "diagnostics.jsonl", diagnostics_bytes)
    _write_immutable(output / "baseline-candidates.jsonl", baseline_bytes)
    _write_immutable(
        output / "manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return manifest


def _verify_package(
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = _json_object(output / "manifest.json", label="depth manifest")
    if manifest.get("schema_version") != "openalex-depth-continuation-validation-v1":
        raise ValueError("depth manifest schema is invalid")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("depth manifest output hashes are missing")
    paths = {
        "partition_sha256": output / "partition.jsonl",
        "request_plan_sha256": output / "request-plan.jsonl",
        "actions_sha256": output / "actions.json",
        "diagnostics_sha256": output / "diagnostics.jsonl",
        "baseline_candidates_sha256": output / "baseline-candidates.jsonl",
    }
    for key, path in paths.items():
        if outputs.get(key) != _sha256_file(path):
            raise ValueError(f"frozen depth artifact hash mismatch: {path}")
    partition_rows = _load_jsonl(output / "partition.jsonl")
    request_rows = _load_jsonl(output / "request-plan.jsonl")
    actions = _json_object(output / "actions.json", label="depth actions")
    verify_gold_blind_request_plan(request_rows)
    query_ids = [str(row.get("query_id", "")) for row in partition_rows]
    if (
        len(query_ids) != manifest.get("query_count")
        or len(query_ids) != len(set(query_ids))
        or set(query_ids) != {str(row.get("query_id", "")) for row in request_rows}
        or set(query_ids) != set(actions)
        or any(
            row.get("role") != "training" or row.get("split") != "auto_train"
            for row in partition_rows
        )
        or manifest.get("llm_requests_made") != 0
        or manifest.get("test_partition_touched") is not False
    ):
        raise ValueError("frozen depth partition isolation failed")
    return manifest, partition_rows, request_rows, actions


def _next_batch_paths(receipts_root: Path) -> tuple[Path, Path]:
    index = 1
    while True:
        run = receipts_root / "openalex" / f"batch-{index:04d}"
        capture = receipts_root / "captures" / "openalex" / f"batch-{index:04d}"
        if not run.exists() and not capture.exists():
            return run, capture
        index += 1


def _completed_query_ids(receipts_root: Path) -> set[str]:
    return {
        path.stem
        for path in receipts_root.rglob("retrieval/attempt-01/*.json")
        if path.is_file()
    }


def _error_rows(errors: Sequence[object]) -> list[object]:
    return [
        error.model_dump(mode="json") if hasattr(error, "model_dump") else error
        for error in errors
    ]


async def _collect(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.workspace_root).resolve()
    output = (root / args.output).resolve()
    manifest, _partition_rows, request_rows, _actions = _verify_package(output)
    if manifest.get("max_raw_openalex_requests") != args.max_raw_requests:
        raise ValueError("collection cap differs from frozen authorization")
    receipts_root = output / "receipts"
    receipts_root.mkdir(parents=True, exist_ok=True)
    completed = _completed_query_ids(receipts_root)
    pending = [row for row in request_rows if str(row["query_id"]) not in completed]
    window = current_openalex_quota_window(datetime.now(UTC))
    ledger = SQLiteOpenAlexDailyQuotaLedger(
        output / "quota-ledger.sqlite3",
        window=window,
        key_slot=args.key_slot,
        max_search_calls=args.max_raw_requests,
        clock=lambda: datetime.now(UTC),
    )
    profile = load_runtime_profile((root / args.profile).resolve())
    secrets = resolve_runtime_secrets(profile, openalex_key_slot=args.key_slot)
    recipe = load_recall_recipe((root / args.recipe).resolve())
    print(
        f"preflight queries={len(request_rows)} pending={len(pending)} "
        f"used_raw={ledger.used_search_calls} cap={args.max_raw_requests} llm=0",
        flush=True,
    )
    for offset in range(0, len(pending), args.chunk_size):
        if ledger.used_search_calls >= args.max_raw_requests:
            break
        batch = pending[offset : offset + args.chunk_size]
        run_path, capture_path = _next_batch_paths(receipts_root)
        bundle = await build_live_runtime_bundle(
            profile=profile,
            secrets=secrets,
            loaded_recipe=recipe,
            capture_root=capture_path,
            search_dependency="openalex",
            openalex_attempt_gate=ledger,
        )
        batch_results: list[dict[str, object]] = []
        try:
            backend = bundle.runtime.search_backend
            for row in batch:
                query_id = str(row["query_id"])
                action = cast(dict[str, object], row["source_action"])
                action_id = str(action["action_id"])
                result = await backend.search_continuation(
                    f"depth-continuation:{query_id}",
                    str(row["derived_query_text"]),
                    cast(dict[str, object], row["filters"]),
                    cursor=str(row["cursor"]),
                    limit=int(row["limit"]),
                )
                generation = {
                    "query_id": query_id,
                    "attempt_id": "attempt-01",
                    "attempt_status": "succeeded",
                    "valid_repeat_ordinal": None,
                    "actions": [action],
                    "generation_provenance": {
                        "generator": "frozen_openalex_cursor_continuation",
                        "candidate_source_namespace": row[
                            "source_candidate_namespace"
                        ],
                        "source_snapshot_sha256": row["source_snapshot_sha256"],
                        "source_retrieval_sha256": row["source_retrieval_sha256"],
                        "provider_rank_offset": "50",
                        "llm_calls": "0",
                    },
                    "llm_call_receipts": [],
                    "repair_count": 0,
                }
                retrieval_result = {
                    "action_id": action_id,
                    "action_type": action["action_type"],
                    "hits": [paper.model_dump(mode="json") for paper in result.hits],
                    "usage": result.usage.model_dump(mode="json"),
                    "provenance": dict(result.provenance),
                    "errors": _error_rows(result.errors),
                    "infrastructure_failure": result.infrastructure_failure,
                }
                retrieval = {
                    "query_id": query_id,
                    "attempt_id": "attempt-01",
                    "attempt_status": "succeeded",
                    "valid_repeat_ordinal": None,
                    "results": [retrieval_result],
                    "retrieval_provenance": {
                        "candidate_policy": "openalex_cursor_continuation_page2",
                        "provider_rank_offset": 50,
                        "source_page_size": 50,
                    },
                }
                _write_immutable(
                    run_path / "generation/attempt-01" / f"{query_id}.json",
                    (
                        json.dumps(generation, ensure_ascii=False, indent=2) + "\n"
                    ).encode("utf-8"),
                )
                _write_immutable(
                    run_path / "retrieval/attempt-01" / f"{query_id}.json",
                    (
                        json.dumps(retrieval, ensure_ascii=False, indent=2) + "\n"
                    ).encode("utf-8"),
                )
                batch_results.append(
                    {
                        "query_id": query_id,
                        "search_api_calls": result.usage.search_api_calls,
                        "hit_count": len(result.hits),
                        "error_codes": [error.code for error in result.errors],
                        "infrastructure_failure": result.infrastructure_failure,
                    }
                )
        finally:
            await bundle.aclose()
        _write_immutable(
            run_path / "batch-report.json",
            (
                json.dumps(
                    {
                        "schema_version": "openalex-depth-batch-report-v1",
                        "results": batch_results,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )
        print(
            f"attempted={min(offset + len(batch), len(pending))}/{len(pending)} "
            f"used_raw={ledger.used_search_calls}",
            flush=True,
        )
    completed = _completed_query_ids(receipts_root)
    result_rows = [
        row
        for path in receipts_root.rglob("batch-report.json")
        for row in cast(list[dict[str, object]], _json_object(path, label="batch report")["results"])
    ]
    successful = sum(
        row.get("search_api_calls") == 1 and row.get("infrastructure_failure") is False
        for row in result_rows
    )
    status: dict[str, object] = {
        "schema_version": "openalex-depth-collection-status-v1",
        "window": window.isoformat(),
        "key_slot": args.key_slot,
        "query_count": len(request_rows),
        "attempted_query_count": len(completed),
        "successful_query_count": successful,
        "missing_query_count": len(request_rows) - len(completed),
        "ledger_used_raw_openalex_requests": ledger.used_search_calls,
        "max_raw_openalex_requests": args.max_raw_requests,
        "llm_requests_made": 0,
        "test_partition_touched": False,
        "production_lock_modified": False,
    }
    status_path = output / "collection-status.json"
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def collect(args: argparse.Namespace) -> dict[str, object]:
    return asyncio.run(_collect(args))


def _verify_baseline_snapshot(
    query_id: str,
    candidates: Sequence[object],
    snapshot: Mapping[str, object],
) -> None:
    aliases = sorted(
        {
            alias
            for candidate in candidates
            for alias in paper_evaluation_aliases(getattr(candidate, "paper"))
        }
    )
    if snapshot.get("candidate_count") != len(candidates) or snapshot.get(
        "candidate_aliases"
    ) != aliases:
        raise ValueError(f"reconstructed baseline candidate pool drifted: {query_id}")


def _has_gold(
    candidates: Sequence[object],
    gold_ids: Sequence[str],
    identifier_map: IdentifierMap,
) -> bool:
    resolved_gold = {identifier_map.resolve(value) for value in gold_ids}
    for candidate in candidates:
        aliases = {
            identifier_map.resolve(value)
            for value in paper_evaluation_aliases(getattr(candidate, "paper"))
        }
        if aliases.intersection(resolved_gold):
            return True
    return False


def comparison_candidate_counts(
    *,
    baseline_pre_cap: int,
    augmented_pre_cap: int,
    baseline_after_cap: int,
    augmented_after_cap: int,
) -> dict[str, int]:
    """Expose both the aggregate contract and pre-cap recall diagnostics."""

    values = (
        baseline_pre_cap,
        augmented_pre_cap,
        baseline_after_cap,
        augmented_after_cap,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("candidate counts must be non-negative integers")
    return {
        "baseline_candidate_count": baseline_after_cap,
        "augmented_candidate_count": augmented_after_cap,
        "added_candidate_count": augmented_after_cap - baseline_after_cap,
        "baseline_candidate_count_pre_cap": baseline_pre_cap,
        "augmented_candidate_count_pre_cap": augmented_pre_cap,
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.workspace_root).resolve()
    output = (root / args.output).resolve()
    manifest, partition_rows, _request_rows, _actions = _verify_package(output)
    status = _json_object(output / "collection-status.json", label="collection status")
    if status.get("attempted_query_count") != manifest.get("query_count"):
        raise ValueError("depth collection is incomplete")
    handoff_path = (root / args.handoff).resolve()
    partition_path = (root / args.partition).resolve()
    bundle_path = (root / args.production_bundle).resolve()
    selection_path = (root / args.production_selection).resolve()
    lock_path = (root / args.production_lock).resolve()
    package = _online_only_package(
        load_training_package(
            handoff_path=handoff_path,
            partition_path=partition_path,
            production_bundle_path=bundle_path,
        )
    )
    query_ids = tuple(str(row["query_id"]) for row in partition_rows)
    selected_package = replace(
        package,
        query_ids=query_ids,
        rows_by_query_id={
            query_id: package.rows_by_query_id[query_id] for query_id in query_ids
        },
    )
    baseline_paths = index_training_receipts(selected_package)
    supplemental_root = (output / "receipts").resolve()
    supplemental_paths = {
        query_id: tuple(
            sorted(supplemental_root.rglob(f"retrieval/attempt-01/{query_id}.json"))
        )
        for query_id in query_ids
    }
    if any(len(paths) != 1 for paths in supplemental_paths.values()):
        raise ValueError("depth supplemental receipt coverage is invalid")
    baseline_snapshots = {
        str(row["query_id"]): row
        for row in _load_jsonl(output / "baseline-candidates.jsonl")
    }
    signals = {
        str(row["query_id"]): str(row["signal"])
        for row in _load_jsonl(output / "diagnostics.jsonl")
    }
    manifest_path, weights_path, selection = _load_production_paths(selection_path)
    ranker = load_f5_production_ranker_bytes(
        manifest_path.read_bytes(),
        weights_path.read_bytes(),
    )
    identity_context = _load_production_identifier_context(
        workspace_root=root,
        lock_path=lock_path,
    )
    identifier_map = IdentifierMap.from_bytes(
        identity_context.identifier_map_bytes,
        source="production combined identifier aliases",
    )
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
            non_reinforcing_additive=True,
        )
        baseline_full = baseline_query.candidates
        augmented_full = augmented_query.candidates
        baseline_capped = baseline_full[: args.candidate_cap]
        augmented_capped = augmented_full[: args.candidate_cap]
        ranked_baseline = ranker.rank(baseline_query.query, baseline_capped)
        ranked_augmented = ranker.rank(augmented_query.query, augmented_capped)
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
        baseline_aliases = {
            alias
            for candidate in baseline_full
            for alias in paper_evaluation_aliases(candidate.paper)
        }
        new_candidates = [
            candidate
            for candidate in augmented_full
            if not baseline_aliases.intersection(
                paper_evaluation_aliases(candidate.paper)
            )
        ]
        source_ranks = [
            rank
            for candidate in new_candidates
            for rank in candidate.source_ranks.values()
        ]
        if source_ranks and min(source_ranks) <= 50:
            raise ValueError(f"depth candidates lost provider rank offset: {query_id}")
        gold_ids = baseline_query.gold_paper_ids
        pre_cap_hit = _has_gold(augmented_full, gold_ids, identifier_map)
        post_cap_hit = _has_gold(augmented_capped, gold_ids, identifier_map)
        per_query.append(
            {
                "query_index": index,
                "query_id": query_id,
                "signal": signals[query_id],
                "gold_count": len(gold_ids),
                **comparison_candidate_counts(
                    baseline_pre_cap=len(baseline_full),
                    augmented_pre_cap=len(augmented_full),
                    baseline_after_cap=len(baseline_capped),
                    augmented_after_cap=len(augmented_capped),
                ),
                "added_unique_candidate_count": len(new_candidates),
                "minimum_added_provider_rank": min(source_ranks) if source_ranks else None,
                "depth_gold_hit_pre_cap": pre_cap_hit,
                "depth_gold_hit_after_candidate_cap": post_cap_hit,
                "baseline": baseline_metrics,
                "augmented": augmented_metrics,
            }
        )
    overall = aggregate_comparison(per_query)
    by_signal = {
        signal: aggregate_comparison(
            [row for row in per_query if row["signal"] == signal]
        )
        for signal in sorted(set(signals.values()))
    }
    result: dict[str, object] = {
        "schema_version": "production-f5-openalex-depth-candidate-ab-v1",
        "comparison": {
            "baseline": "current-openalex-only-pool+production-F5",
            "augmented": "same-pool+one-frozen-OpenAlex-page2+same-production-F5",
            "same_ranker_both_arms": True,
            "nonreinforcing_fair_merge": True,
            "provider_rank_offset": 50,
            "candidate_cap": args.candidate_cap,
        },
        "query_count": len(per_query),
        "successful_live_query_count": status["successful_query_count"],
        "raw_openalex_request_count": status[
            "ledger_used_raw_openalex_requests"
        ],
        "depth_gold_hit_query_count_pre_cap": sum(
            bool(row["depth_gold_hit_pre_cap"]) for row in per_query
        ),
        "depth_gold_hit_query_count_after_candidate_cap": sum(
            bool(row["depth_gold_hit_after_candidate_cap"]) for row in per_query
        ),
        "added_unique_candidate_count": sum(
            int(row["added_unique_candidate_count"]) for row in per_query
        ),
        "overall": overall,
        "by_signal": by_signal,
        "inputs": {
            "frozen_manifest_sha256": _sha256_file(output / "manifest.json"),
            "collection_status_sha256": _sha256_file(
                output / "collection-status.json"
            ),
            "production_selection_sha256": _sha256_file(selection_path),
            "production_manifest_sha256": _sha256_file(manifest_path),
            "production_weights_sha256": _sha256_file(weights_path),
            "production_default": selection["production_default"],
            "production_training_query_count": selection["training_query_count"],
            **identity_context.evidence,
        },
        "safety": {
            "evaluation_online_requests_made": 0,
            "llm_requests_made": 0,
            "training_started": False,
            "test_partition_touched": False,
            "production_lock_modified": False,
            "request_plan_gold_blind": True,
        },
        "per_query": per_query,
    }
    result_path = output / "production-f5-depth-ab-v1.json"
    _write_immutable(
        result_path,
        (
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "evaluate"))
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--production-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--production-selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--production-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--max-raw-requests", type=int, default=DEFAULT_RAW_REQUEST_CAP)
    parser.add_argument("--candidate-cap", type=int, default=DEFAULT_CANDIDATE_CAP)
    parser.add_argument("--key-slot", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=4, choices=(1, 2, 4, 8))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.query_count != DEFAULT_QUERY_COUNT:
        raise ValueError("this authorization is frozen to exactly 64 queries")
    if args.max_raw_requests != DEFAULT_RAW_REQUEST_CAP:
        raise ValueError("this authorization is frozen to at most 64 raw requests")
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "collect":
        result = collect(args)
    else:
        result = evaluate(args)
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"per_query", "outputs", "inputs"}
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
