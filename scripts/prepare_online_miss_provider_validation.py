"""Freeze a disjoint OpenAlex-miss sample for paired OpenAlex/S2 recall testing."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.domain.models import QuerySpec  # noqa: E402
from paper_search.learning.openalex_daily_schedule import (  # noqa: E402
    SearchActionIdentity,
    search_action_identity,
)
from paper_search.recall_experiments.contracts import (  # noqa: E402
    RecallActionBatch,
    RecallGenerationContext,
    assert_no_forbidden_identifier_keys_or_patterns,
)
from scripts.run_provider_recall_comparison import (  # noqa: E402
    _build_generator_override,
)


DEFAULT_AUDIT_ROOT = Path(
    "data/training_private/evaluations/"
    "online-recall-ceiling-production-f5-21429-v1"
)
DEFAULT_PARTITION = Path("data/training_private/partitions/pasa_auto_train.jsonl")
DEFAULT_RECALL_ROOT = Path("data/training_private/recall_policy")
DEFAULT_ONLINE_RECALL_ROOT = Path("data/training_private/online_recall")
DEFAULT_HANDOFF = Path(
    "data/training_private/training_runs/"
    "openalex-pasa-high-recall-expanded-21429-v1/"
    "ranking-training-handoff-expanded.json"
)
DEFAULT_OUTPUT = Path(
    "data/training_private/recall_policy/"
    "online-miss-openalex-s2-paired128-v1"
)
_QUERY_FILE = re.compile(r"^(AutoScholarQuery_train_\d+)\.json$")
_TARGET_STRATA = ("method", "negation", "unconstrained")


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


def _jsonl(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _is_below(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root.resolve() or root.resolve() in resolved.parents for root in roots)


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def collect_prior_query_ids(
    root: Path,
    *,
    ignored_roots: Sequence[Path] = (),
) -> set[str]:
    """Inventory query identities already used by frozen or executed recall work."""

    if not root.exists():
        return set()
    ignored = tuple(path.resolve() for path in ignored_roots)
    query_ids: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or _is_below(path, ignored):
            continue
        match = _QUERY_FILE.match(path.name)
        if match:
            query_ids.add(match.group(1))
        if path.suffix == ".jsonl" and "partition" in path.name:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("query_id"), str):
                    query_ids.add(str(row["query_id"]))
        if path.name in {"query-selection.json", "sample-manifest.json"}:
            raw = _json_object(path)
            values = raw.get("query_ids") if raw is not None else None
            if isinstance(values, list):
                query_ids.update(value for value in values if isinstance(value, str))
    return query_ids


def primary_stratum(labels: Sequence[str]) -> str:
    normalized = {str(value) for value in labels}
    for value in ("negation", "method", "dataset", "year"):
        if value in normalized:
            return value
    return "unconstrained"


def select_stratified(
    rows: Sequence[dict[str, Any]],
    *,
    quotas: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Take exact per-stratum quotas without borrowing across diagnostic strata."""

    if not quotas or any(value < 0 for value in quotas.values()):
        raise ValueError("stratum quotas are invalid")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for stratum, target in quotas.items():
        pool = [row for row in rows if row.get("stratum") == stratum]
        if len(pool) < target:
            raise ValueError(
                f"insufficient {stratum} proposals: required={target} available={len(pool)}"
            )
        for row in pool[:target]:
            query_id = str(row.get("query_id", ""))
            if not query_id or query_id in seen:
                raise ValueError("proposal query identities are invalid")
            seen.add(query_id)
            selected.append(row)
    return selected


def audit_authorizes_recall(summary: Mapping[str, object]) -> bool:
    decision = summary.get("decision")
    metrics = summary.get("summary")
    overall = metrics.get("overall") if isinstance(metrics, Mapping) else None
    return bool(
        isinstance(decision, Mapping)
        and decision.get("recommended_next_branch")
        == "openalex-s2-supplemental-recall"
        and isinstance(overall, Mapping)
        and overall.get("dominant_bottleneck_at_20") == "recall"
    )


def _rank_rows(audit_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(audit_root.glob("rank-shard-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    if not rows:
        raise ValueError("online recall audit rank shards are unavailable")
    return rows


def _partition_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("training partition row is invalid")
        query_id = str(row.get("query_id", ""))
        if (
            not query_id
            or query_id in rows
            or row.get("role") != "training"
            or row.get("split") != "auto_train"
        ):
            raise ValueError("training partition identity or isolation is invalid")
        rows[query_id] = row
    return rows


def _actions_from_object(value: object) -> Iterable[Mapping[str, object]]:
    if isinstance(value, dict):
        actions = value.get("actions")
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict):
                    yield action
        for nested in value.values():
            yield from _actions_from_object(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _actions_from_object(nested)


def _action_inventory_paths(
    *,
    handoff_path: Path,
    recall_root: Path,
    ignored_roots: Sequence[Path],
) -> list[Path]:
    handoff = _json_object(handoff_path)
    if handoff is None:
        raise ValueError("training handoff is invalid")
    raw_roots = handoff.get("ordered_receipt_roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise ValueError("training handoff has no receipt roots")
    paths: set[Path] = set()
    for raw_root in raw_roots:
        if not isinstance(raw_root, str):
            raise ValueError("training handoff receipt root is invalid")
        root = Path(raw_root)
        if not root.is_dir():
            raise ValueError(f"training receipt root is unavailable: {root}")
        paths.update(root.rglob("generation/**/*.json"))
    for path in recall_root.rglob("*.json"):
        if _is_below(path, ignored_roots):
            continue
        if path.name == "actions.json" or "generation" in path.parts:
            paths.add(path)
    return sorted(paths)


def collect_prior_action_identities(
    *,
    handoff_path: Path,
    recall_root: Path,
    ignored_roots: Sequence[Path] = (),
) -> set[SearchActionIdentity]:
    identities: set[SearchActionIdentity] = set()
    for path in _action_inventory_paths(
        handoff_path=handoff_path,
        recall_root=recall_root,
        ignored_roots=ignored_roots,
    ):
        raw = _json_object(path)
        if raw is None:
            continue
        for action in _actions_from_object(raw):
            identity = search_action_identity(action)
            if identity is not None:
                identities.add(identity)
    return identities


def _identity_payload(identity: SearchActionIdentity) -> dict[str, str]:
    return identity.model_dump(mode="json")


async def _generate_proposals(
    *,
    root: Path,
    candidates: Sequence[dict[str, Any]],
    prior_identities: set[SearchActionIdentity],
    quotas: Mapping[str, int],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    generator = _build_generator_override(
        collection_mode="lexical_bridge",
        role="training",
        max_actions=4,
        workspace_root=root,
    )
    proposals: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    selected_identities: set[SearchActionIdentity] = set()
    available: Counter[str] = Counter()
    for row in candidates:
        stratum = str(row["stratum"])
        if stratum not in quotas or available[stratum] >= quotas[stratum]:
            continue
        counts["scanned"] += 1
        query_id = str(row["query_id"])
        query = str(row["query"])
        generation = await generator.generate(
            RecallGenerationContext(
                query_id=query_id,
                original_query=query,
                query_spec=QuerySpec(original_query=query, research_goal=query),
            )
        )
        if generation.call_receipts:
            raise ValueError("local lexical action generation unexpectedly used an LLM")
        dumped = generation.action_batch.model_dump(mode="json")
        assert_no_forbidden_identifier_keys_or_patterns(dumped)
        raw_actions = dumped.get("actions")
        if not isinstance(raw_actions, list):
            raise ValueError("generated action batch is invalid")
        ordered_actions = sorted(
            (action for action in raw_actions if isinstance(action, dict)),
            key=lambda action: 0 if action.get("action_id") == "lexical-bridge-1" else 1,
        )
        chosen_action: dict[str, Any] | None = None
        chosen_identity: SearchActionIdentity | None = None
        for action in ordered_actions:
            identity = search_action_identity(action)
            if (
                identity is None
                or identity.search_mode != "lexical"
                or identity in prior_identities
                or identity in selected_identities
            ):
                continue
            chosen_action = dict(action)
            chosen_identity = identity
            break
        if chosen_action is None or chosen_identity is None:
            counts[f"no_new_action_{stratum}"] += 1
            continue
        action_batch = RecallActionBatch(actions=[chosen_action]).model_dump(mode="json")
        assert_no_forbidden_identifier_keys_or_patterns(action_batch)
        selected_identities.add(chosen_identity)
        available[stratum] += 1
        counts[f"eligible_{stratum}"] += 1
        proposals.append(
            {
                **row,
                "action_batch": action_batch,
                "action_identity": _identity_payload(chosen_identity),
                "action_source": (
                    "supervised_lexical_bridge"
                    if chosen_action.get("action_id") == "lexical-bridge-1"
                    else "production_pairwise_lexical"
                ),
                "generation_provenance": generation.provenance,
            }
        )
        if all(available[key] >= value for key, value in quotas.items()):
            break
    return proposals, counts


async def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.workspace_root).resolve()
    resolve = lambda value: (root / value).resolve()  # noqa: E731
    audit_root = resolve(args.audit_root)
    partition_path = resolve(args.partition)
    recall_root = resolve(args.recall_root)
    online_recall_root = resolve(args.online_recall_root)
    handoff_path = resolve(args.handoff)
    output = resolve(args.output)
    quotas = {
        "method": args.method_count,
        "negation": args.negation_count,
        "unconstrained": args.unconstrained_count,
    }
    if any(value <= 0 for value in quotas.values()):
        raise ValueError("every target stratum requires a positive sample size")
    if sum(quotas.values()) != args.query_count:
        raise ValueError("stratum quotas must sum to query count")

    summary_path = audit_root / "summary.json"
    progress_path = audit_root / "progress.json"
    summary = _json_object(summary_path)
    if summary is None:
        raise ValueError("online recall audit summary is invalid")
    if not audit_authorizes_recall(summary):
        raise ValueError("online recall audit does not authorize the recall branch")

    ignored = (output,)
    prior_query_ids = collect_prior_query_ids(recall_root, ignored_roots=ignored)
    prior_query_ids.update(
        collect_prior_query_ids(online_recall_root, ignored_roots=ignored)
    )
    prior_identities = collect_prior_action_identities(
        handoff_path=handoff_path,
        recall_root=recall_root,
        ignored_roots=ignored,
    )
    partition = _partition_rows(partition_path)
    candidates: list[dict[str, Any]] = []
    miss_count_by_stratum: Counter[str] = Counter()
    excluded_prior_by_stratum: Counter[str] = Counter()
    for audit_row in _rank_rows(audit_root):
        if audit_row.get("gold_ranks"):
            continue
        query_id = str(audit_row.get("query_id", ""))
        if query_id not in partition:
            raise ValueError(f"audit query is missing from auto_train: {query_id}")
        labels = [str(value) for value in audit_row.get("labels", [])]
        stratum = primary_stratum(labels)
        miss_count_by_stratum[stratum] += 1
        if query_id in prior_query_ids:
            excluded_prior_by_stratum[stratum] += 1
            continue
        row = partition[query_id]
        candidates.append(
            {
                "query_id": query_id,
                "query": str(row["query"]),
                "gold_paper_ids": list(row["gold_paper_ids"]),
                "labels": labels,
                "stratum": stratum,
                "online_candidate_count": int(audit_row["online_candidate_count"]),
            }
        )
    candidates.sort(
        key=lambda row: hashlib.sha256(
            ("online-miss-openalex-s2-paired128-v1\0" + str(row["query_id"])).encode()
        ).hexdigest()
    )
    proposals, scan_counts = await _generate_proposals(
        root=root,
        candidates=candidates,
        prior_identities=prior_identities,
        quotas=quotas,
    )
    selected = select_stratified(proposals, quotas=quotas)

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
    actions = {str(row["query_id"]): row["action_batch"] for row in selected}
    request_plan = [
        {
            "query_id": row["query_id"],
            "action_batch": row["action_batch"],
            "providers": ["openalex", "semantic_scholar"],
            "max_results_per_action": 50,
        }
        for row in selected
    ]
    diagnostics = [
        {
            "query_id": row["query_id"],
            "stratum": row["stratum"],
            "labels": row["labels"],
            "online_candidate_count": row["online_candidate_count"],
            "action_identity": row["action_identity"],
            "action_source": row["action_source"],
            "generation_provenance": row["generation_provenance"],
        }
        for row in selected
    ]
    partition_bytes = _jsonl(partition_rows)
    actions_bytes = _canonical_bytes(actions)
    request_bytes = _jsonl(request_plan)
    diagnostics_bytes = _jsonl(diagnostics)
    source_counts = Counter(str(row["action_source"]) for row in selected)
    selected_counts = Counter(str(row["stratum"]) for row in selected)
    prior_identity_bytes = _canonical_bytes(
        sorted(
            (_identity_payload(identity) for identity in prior_identities),
            key=lambda row: (row["action_type"], row["search_mode"], row["normalized_text"]),
        )
    )
    manifest = {
        "schema_version": "online-miss-provider-paired-validation-package-v1",
        "purpose": "new-disjoint-openalex-only-miss-recall-confirmation",
        "baseline_population": "current-online-supported-candidate-pool-has-no-gold-hit",
        "query_count": len(selected),
        "selection_policy": "sha256-order-disjoint-openalex-only-miss-stratified-v1",
        "selected_stratum_counts": dict(sorted(selected_counts.items())),
        "eligible_miss_count_by_stratum": dict(sorted(miss_count_by_stratum.items())),
        "excluded_prior_query_count": len(prior_query_ids),
        "excluded_prior_query_count_by_stratum": dict(
            sorted(excluded_prior_by_stratum.items())
        ),
        "prior_action_identity_count": len(prior_identities),
        "prior_action_identity_inventory_sha256": _sha256_bytes(prior_identity_bytes),
        "action_policy": "first-new-action-from-existing-supervised-lexical-bridge-v2-or-production-pairwise-lexical-v1",
        "action_source_counts": dict(sorted(source_counts.items())),
        "scan_counts": dict(sorted(scan_counts.items())),
        "paired_provider_contract": {
            "providers": ["openalex", "semantic_scholar"],
            "identical_action_per_query_across_providers": True,
            "max_actions_per_query_per_provider": 1,
            "max_results_per_action": 50,
            "planned_search_requests": {
                "openalex": len(selected),
                "semantic_scholar": len(selected),
                "total": len(selected) * 2,
            },
        },
        "inputs": {
            "audit_summary_path": str(summary_path.relative_to(root)),
            "audit_summary_sha256": _sha256_file(summary_path),
            "audit_progress_sha256": _sha256_file(progress_path),
            "partition_path": str(partition_path.relative_to(root)),
            "partition_sha256": _sha256_file(partition_path),
            "handoff_sha256": _sha256_file(handoff_path),
            "production_selection_sha256": _sha256_file(
                root / "artifacts/models/production-document-ranker-selection.json"
            ),
            "production_pairwise_action_manifest_sha256": _sha256_file(
                root / "data/training/cpu-pairwise-action-ranker-openalex-v1.json"
            ),
            "supervised_lexical_bridge_manifest_sha256": _sha256_file(
                root / "data/training/supervised-lexical-bridge-openalex-v2.json"
            ),
        },
        "outputs": {
            "partition_sha256": _sha256_bytes(partition_bytes),
            "actions_sha256": _sha256_bytes(actions_bytes),
            "request_plan_sha256": _sha256_bytes(request_bytes),
            "diagnostics_sha256": _sha256_bytes(diagnostics_bytes),
        },
        "safety": {
            "query_ids_disjoint_from_prior_recall_samples": True,
            "action_identities_disjoint_from_prior_inventory": True,
            "generation_gold_blind": True,
            "gold_labels_or_identifiers_in_request_plan": False,
            "final_test_query_count": 0,
            "auto_dev_538_used_for_selection": False,
            "llm_requests_made": 0,
            "online_requests_made": 0,
            "training_started": False,
            "production_lock_modified": False,
        },
        "decision": {
            "offline_input_gate_passed": True,
            "online_collection_authorized": False,
            "method_pair_retraining_started": False,
            "next_step": "request bounded paired OpenAlex/S2 collection authorization",
        },
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _write_immutable(output / "partition.jsonl", partition_bytes)
    _write_immutable(output / "actions.json", actions_bytes)
    _write_immutable(output / "provider-request-plan.jsonl", request_bytes)
    _write_immutable(output / "selection-diagnostics.jsonl", diagnostics_bytes)
    _write_immutable(output / "manifest.json", manifest_bytes)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--recall-root", type=Path, default=DEFAULT_RECALL_ROOT)
    parser.add_argument(
        "--online-recall-root", type=Path, default=DEFAULT_ONLINE_RECALL_ROOT
    )
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--method-count", type=int, default=24)
    parser.add_argument("--negation-count", type=int, default=24)
    parser.add_argument("--unconstrained-count", type=int, default=80)
    return parser


def main() -> None:
    manifest = asyncio.run(_prepare(build_parser().parse_args()))
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
