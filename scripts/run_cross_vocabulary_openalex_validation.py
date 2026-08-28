"""Prepare, collect, and evaluate a blind OpenAlex cross-vocabulary canary."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from paper_search.evaluation.predictions import (
    paper_evaluation_aliases,
    paper_matches_evaluation_ids,
)
from paper_search.learning.candidates import query_content_terms
from paper_search.learning.cross_vocabulary_bridge import (
    CrossVocabularyBridgeProposal,
    build_cross_vocabulary_action_batch,
    build_refined_cross_vocabulary_action_batch,
    propose_cross_vocabulary_bridge,
    propose_refined_cross_vocabulary_bridge,
)
from paper_search.learning.large_scale_fusion_training import (
    FusionTrainingPackage,
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
    query_has_gold_candidate,
)
from paper_search.learning.openalex_daily_schedule import (
    SQLiteOpenAlexDailyQuotaLedger,
    SearchActionIdentity,
    current_openalex_quota_window,
    load_settled_search_action_identities,
    search_action_identity,
)
from paper_search.learning.query_constraint_annotations import query_sha256
from paper_search.processing.deduplicate import deduplicate_papers
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
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    assert_no_forbidden_identifier_keys_or_patterns,
)
from paper_search.recall_experiments.generation.fixed import FixedActionGenerator
from paper_search.recall_experiments.recipes import load_recall_recipe


DEFAULT_OUTPUT = Path(
    "data/training_private/recall_policy/"
    "contrastive-openalex-bridge-validation128-v1"
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
DEFAULT_CONTEXT_MANIFEST = Path(
    "data/training_private/training_runs/"
    "openalex-pasa-high-recall-v3-no-leakage-context-21429-v2/manifest.json"
)
DEFAULT_PROFILE = Path(
    "configs/recall_experiments/runtime/fixed-budget-openalex-live.yaml"
)
DEFAULT_RECIPE = Path(
    "configs/recall_experiments/methods/scheme-b-semantic-backfill-live.yaml"
)
_PASA_ROOT_NAMES = frozenset(
    {
        "pasa-mixed-receipts",
        "pasa-targeted-repair-candidate-receipts",
        "pasa-year-mixed-repair-candidate-receipts",
    }
)


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _jsonl(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_bytes().splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}")
            rows.append(cast(dict[str, object], value))
    return rows


def _online_only_package(package: FusionTrainingPackage) -> FusionTrainingPackage:
    excluded = tuple(
        root for root in package.ordered_receipt_roots if root.name in _PASA_ROOT_NAMES
    )
    if {root.name for root in excluded} != _PASA_ROOT_NAMES:
        raise ValueError("sealed PASA receipt roots do not match the expected isolation set")
    online_roots = tuple(
        root for root in package.ordered_receipt_roots if root not in set(excluded)
    )
    if not online_roots or any(root.name in _PASA_ROOT_NAMES for root in online_roots):
        raise ValueError("OpenAlex-only receipt root isolation failed")
    return replace(
        package,
        ordered_receipt_roots=online_roots,
        additive_receipt_roots=(),
    )


def _excluded_query_ids(workspace_root: Path) -> set[str]:
    excluded: set[str] = set()
    relative_paths = [
        Path(
            "data/training_private/recall_policy/"
            "query-adaptive-high-recall-discovery128-v1/pilot-partition.jsonl"
        ),
        Path(
            "data/training_private/recall_policy/"
            "query-adaptive-high-recall-validation352-v1/pilot-partition.jsonl"
        ),
        Path(
            "data/training_private/recall_policy/"
            "contrastive-openalex-bridge-validation128-v1/partition.jsonl"
        ),
    ]
    recall_policy_root = (
        workspace_root / "data" / "training_private" / "recall_policy"
    )
    relative_paths.extend(
        path.relative_to(workspace_root)
        for path in sorted(
            recall_policy_root.glob(
                "contrastive-openalex-bridge-*/partition.jsonl"
            )
        )
    )
    for relative in dict.fromkeys(relative_paths):
        path = workspace_root / relative
        if not path.exists():
            continue
        excluded.update(str(row["query_id"]) for row in _load_jsonl(path))
    for root_name in (
        "lexical-bridge-paired15-production-lexical-v1",
        "lexical-bridge-paired15-augmented-v1",
    ):
        root = workspace_root / "data/training_private/online_recall" / root_name
        for path in root.glob("openalex/*/sample-manifest.json"):
            raw = json.loads(path.read_text(encoding="utf-8"))
            values = raw.get("query_ids") if isinstance(raw, dict) else None
            if isinstance(values, list):
                excluded.update(value for value in values if isinstance(value, str))
    return excluded


def _completed_identities(receipt_paths: Sequence[Path]) -> set[SearchActionIdentity]:
    completed: set[SearchActionIdentity] = set()
    for retrieval_path in receipt_paths:
        generation_path = Path(
            *[
                "generation" if part == "retrieval" else part
                for part in retrieval_path.parts
            ]
        )
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        raw_actions = generation.get("actions") if isinstance(generation, dict) else None
        raw_results = retrieval.get("results") if isinstance(retrieval, dict) else None
        if not isinstance(raw_actions, list) or not isinstance(raw_results, list):
            continue
        identities: dict[str, SearchActionIdentity] = {}
        for action in raw_actions:
            if not isinstance(action, dict) or not isinstance(action.get("action_id"), str):
                continue
            identity = search_action_identity(action)
            if identity is not None:
                identities[str(action["action_id"])] = identity
        for result in raw_results:
            if not isinstance(result, dict):
                continue
            action_id = result.get("action_id")
            errors = result.get("errors")
            if (
                isinstance(action_id, str)
                and action_id in identities
                and isinstance(errors, list)
                and not errors
                and result.get("infrastructure_failure") is not True
            ):
                completed.add(identities[action_id])
    return completed


def _load_context_signals(
    package: FusionTrainingPackage,
    manifest_path: Path,
) -> dict[str, dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("exact context manifest is invalid")
    if (
        manifest.get("schema_version") != "directed-fusion-context-freeze-v3"
        or manifest.get("test_partition_touched") is not False
        or manifest.get("development_labels_used_for_training") is not False
        or manifest.get("llm_requests_made") != 0
    ):
        raise ValueError("exact context safety flags failed")
    task_path = manifest_path.with_name("task-labels.merged.jsonl")
    constraint_path = manifest_path.with_name("constraint-labels.merged.jsonl")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("exact context output hashes are missing")
    if outputs.get("task_labels_sha256") != _sha256(task_path.read_bytes()):
        raise ValueError("exact task context hash mismatch")
    if outputs.get("constraint_labels_sha256") != _sha256(constraint_path.read_bytes()):
        raise ValueError("exact constraint context hash mismatch")
    task_by_id = {str(row["query_id"]): row for row in _load_jsonl(task_path)}
    constraint_by_id = {
        str(row["query_id"]): row for row in _load_jsonl(constraint_path)
    }
    if not _context_ids_cover_package(
        package.query_ids,
        task_query_ids=set(task_by_id),
        constraint_query_ids=set(constraint_by_id),
    ):
        raise ValueError("exact context does not cover the sealed training package")
    signals: dict[str, dict[str, object]] = {}
    for query_id in package.query_ids:
        query = str(package.rows_by_query_id[query_id]["query"])
        task = task_by_id[query_id]
        constraint = constraint_by_id[query_id]
        if (
            task.get("query_sha256") != query_sha256(query)
            or constraint.get("query_sha256") != query_sha256(query)
            or task.get("role") != "training"
            or task.get("split") != "auto_train"
            or constraint.get("role") != "training"
            or constraint.get("split") != "auto_train"
        ):
            raise ValueError(f"exact context isolation failed: {query_id}")
        raw_tasks = task.get("tasks")
        task_values = [
            item.get("normalized_value")
            for item in raw_tasks
            if isinstance(raw_tasks, list)
            and isinstance(item, dict)
            and isinstance(item.get("normalized_value"), str)
        ] if isinstance(raw_tasks, list) else []
        signals[query_id] = {
            "tasks": [*task_values, *cast(list[object], constraint.get("tasks", []))],
            "methods": constraint.get("methods", []),
            "datasets": constraint.get("datasets", []),
            "exclusions": constraint.get("exclusions", []),
            "year_from": constraint.get("year_from"),
            "year_to": constraint.get("year_to"),
        }
    return signals


def _context_ids_cover_package(
    package_query_ids: Sequence[str],
    *,
    task_query_ids: set[str],
    constraint_query_ids: set[str],
) -> bool:
    """Allow extra isolated auto_train labels while requiring full package coverage."""

    required = set(package_query_ids)
    return required.issubset(task_query_ids) and required.issubset(constraint_query_ids)


def _proposal_signal(spec: Mapping[str, object]) -> str:
    exclusions = spec.get("exclusions", ())
    datasets = spec.get("datasets", ())
    methods = spec.get("methods", ())
    tasks = spec.get("tasks", ())
    if exclusions:
        return "negation"
    if datasets:
        return "dataset"
    if methods:
        return "method"
    if spec.get("year_from") is not None or spec.get("year_to") is not None:
        return "year"
    if tasks:
        return "task_provenance"
    return "unconstrained"


def _length_bucket(query: str) -> str:
    count = len(query_content_terms(query))
    if count <= 8:
        return "short"
    if count <= 16:
        return "medium"
    return "long"


def _balanced_sample(rows: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["signal"]), str(row["length_bucket"]))].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: hashlib.sha256(str(row["query_id"]).encode()).hexdigest())
    output: list[dict[str, Any]] = []
    ordered_keys = sorted(buckets)
    while len(output) < limit:
        progress = False
        for key in ordered_keys:
            if buckets[key] and len(output) < limit:
                output.append(buckets[key].pop(0))
                progress = True
        if not progress:
            break
    return output


def _refined_strata_sample(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
    negation_target: int,
) -> list[dict[str, Any]]:
    """Prioritize negation, then fill the frozen sample with unconstrained rows."""

    if limit <= 0 or not 0 <= negation_target <= limit:
        raise ValueError("refined sample bounds are invalid")
    negation = [row for row in rows if row.get("signal") == "negation"]
    unconstrained = [row for row in rows if row.get("signal") == "unconstrained"]
    selected = _balanced_sample(
        negation,
        limit=min(negation_target, len(negation)),
    )
    selected_ids = {str(row["query_id"]) for row in selected}
    selected.extend(
        _balanced_sample(unconstrained, limit=limit - len(selected))
    )
    selected_ids.update(str(row["query_id"]) for row in selected)
    if len(selected) < limit:
        remaining_negation = [
            row for row in negation if str(row["query_id"]) not in selected_ids
        ]
        selected.extend(
            _balanced_sample(remaining_negation, limit=limit - len(selected))
        )
    return selected


def _cross_query_common_expansion_terms(
    rows: Sequence[Mapping[str, object]],
    *,
    min_query_count: int = 6,
    min_query_ratio: float = 0.04,
) -> frozenset[str]:
    """Find expansion terms that recur across unrelated proposal queries."""

    if min_query_count <= 0 or not 0 < min_query_ratio <= 1:
        raise ValueError("cross-query suppression bounds are invalid")
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        proposal = row.get("proposal")
        if not isinstance(proposal, Mapping):
            raise ValueError("proposal row lacks frozen diagnostics")
        terms = proposal.get("expansion_terms")
        if not isinstance(terms, list):
            raise ValueError("proposal diagnostics lack expansion terms")
        for term in {str(value).casefold() for value in terms}:
            counts[term] += 1
    threshold = max(min_query_count, math.ceil(len(rows) * min_query_ratio))
    return frozenset(term for term, count in counts.items() if count >= threshold)


def _proposal_payload(proposal: CrossVocabularyBridgeProposal) -> dict[str, object]:
    return {
        "query_text": proposal.query_text,
        "anchors": list(proposal.anchors),
        "expansion_terms": list(proposal.expansion_terms),
        "candidate_support": proposal.candidate_support,
        "action_support": proposal.action_support,
        "grounded_candidate_count": proposal.grounded_candidate_count,
        "online_candidate_count": proposal.online_candidate_count,
        "evidence_profile": proposal.evidence_profile,
        "title_support": proposal.title_support,
        "anchor_cooccurrence": proposal.anchor_cooccurrence,
        "anchor_support": proposal.anchor_support,
    }


def prepare(args: argparse.Namespace) -> None:
    workspace_root = Path(args.workspace_root).resolve()
    output = (workspace_root / args.output).resolve()
    if output.exists() and (output / "manifest.json").exists():
        raise ValueError("frozen validation package already exists")
    handoff = (workspace_root / args.handoff).resolve()
    partition = (workspace_root / args.partition).resolve()
    bundle = (workspace_root / args.production_bundle).resolve()
    package = load_training_package(
        handoff_path=handoff,
        partition_path=partition,
        production_bundle_path=bundle,
    )
    online_package = _online_only_package(package)
    print(f"indexed-package-roots={len(online_package.ordered_receipt_roots)}", flush=True)
    receipt_index = index_training_receipts(online_package)
    context_manifest = (workspace_root / args.context_manifest).resolve()
    specs = _load_context_signals(package, context_manifest)
    excluded_ids = _excluded_query_ids(workspace_root)
    proposals: list[dict[str, Any]] = []
    seen_action_identities: set[SearchActionIdentity] = set()
    considered_queries = 0
    online_no_hit_queries = 0
    eligible_by_signal: dict[str, int] = defaultdict(int)
    signal_order: tuple[str, ...]
    if args.refined_strata:
        refined_pool_target = max(args.query_count, math.ceil(args.query_count * 1.5))
        if args.unconstrained_only:
            desired_by_signal = {"unconstrained": refined_pool_target}
            signal_order = ("unconstrained",)
        else:
            desired_by_signal = {
                "negation": refined_pool_target,
                "unconstrained": refined_pool_target,
            }
            signal_order = ("negation", "unconstrained")
    else:
        desired_by_signal = {
            "year": 2,
            "negation": 32,
            "dataset": 32,
            "method": 48,
            "task_provenance": 80,
            "unconstrained": 80,
        }
        signal_order = (
            "year",
            "negation",
            "dataset",
            "method",
            "task_provenance",
            "unconstrained",
        )
    query_ids_by_signal: dict[str, list[str]] = defaultdict(list)
    for query_id in package.query_ids:
        if query_id not in excluded_ids:
            query_ids_by_signal[_proposal_signal(specs[query_id])].append(query_id)
    for signal in signal_order:
        query_ids = sorted(
            query_ids_by_signal[signal],
            key=lambda query_id: hashlib.sha256(query_id.encode()).hexdigest(),
        )
        for query_id in query_ids:
            if eligible_by_signal[signal] >= desired_by_signal[signal]:
                break
            considered_queries += 1
            online_query = build_document_ranking_query(
                online_package,
                query_id,
                receipt_index[query_id],
            )
            if query_has_gold_candidate(online_query):
                continue
            online_no_hit_queries += 1
            if any(
                "pasa" in value.casefold()
                for candidate in online_query.candidates
                for value in [*candidate.paper.sources, *candidate.source_ranks]
            ):
                raise ValueError(f"PASA evidence entered OpenAlex-only proposal input: {query_id}")
            spec = specs[query_id]
            if args.refined_strata:
                exclusions = tuple(
                    value
                    for value in cast(Sequence[object], spec.get("exclusions", ()))
                    if isinstance(value, str)
                )
                proposal = propose_refined_cross_vocabulary_bridge(
                    online_query.query,
                    online_query.candidates,
                    profile=cast(Any, signal),
                    exclusions=exclusions,
                )
            else:
                proposal = propose_cross_vocabulary_bridge(
                    online_query.query,
                    online_query.candidates,
                )
            if proposal is None:
                continue
            if args.refined_strata:
                action_payload: dict[str, object] = {}
            else:
                action_payload = build_cross_vocabulary_action_batch(
                    proposal
                ).model_dump(mode="json")
                assert_no_forbidden_identifier_keys_or_patterns(action_payload)
                identity = search_action_identity(action_payload["actions"][0])
                if identity is None:
                    raise ValueError("bridge action has no searchable identity")
                if identity in _completed_identities(receipt_index[query_id]):
                    continue
                if identity in seen_action_identities:
                    continue
                seen_action_identities.add(identity)
            baseline_aliases = sorted(
                {
                    alias
                    for candidate in online_query.candidates
                    for alias in paper_evaluation_aliases(candidate.paper)
                }
            )
            proposals.append(
                {
                    "query_id": query_id,
                    "query": online_query.query,
                    "gold_paper_ids": list(online_query.gold_paper_ids),
                    "action_batch": action_payload,
                    "proposal": _proposal_payload(proposal),
                    "signal": _proposal_signal(spec),
                    "length_bucket": _length_bucket(online_query.query),
                    "baseline_aliases": baseline_aliases,
                    "baseline_candidate_count": len(online_query.candidates),
                    "_online_query": online_query,
                    "_exclusions": exclusions if args.refined_strata else (),
                }
            )
            eligible_by_signal[signal] += 1
        print(
            f"prepare-signal={signal} considered={considered_queries} "
            f"no-hit={online_no_hit_queries} eligible={eligible_by_signal[signal]}",
            flush=True,
        )
    initial_eligible_by_signal = dict(sorted(eligible_by_signal.items()))
    suppressed_cross_query_terms: frozenset[str] = frozenset()
    if args.refined_strata:
        suppressed_cross_query_terms = _cross_query_common_expansion_terms(proposals)
        seen_action_identities.clear()
        refined_proposals: list[dict[str, Any]] = []
        eligible_by_signal = defaultdict(int)
        for row in proposals:
            signal = str(row["signal"])
            online_query = row["_online_query"]
            proposal = propose_refined_cross_vocabulary_bridge(
                online_query.query,
                online_query.candidates,
                profile=cast(Any, signal),
                exclusions=cast(Sequence[str], row["_exclusions"]),
                suppressed_expansion_terms=tuple(suppressed_cross_query_terms),
            )
            if proposal is None:
                continue
            action_payload = build_refined_cross_vocabulary_action_batch(
                proposal
            ).model_dump(mode="json")
            assert_no_forbidden_identifier_keys_or_patterns(action_payload)
            identity = search_action_identity(action_payload["actions"][0])
            if identity is None:
                raise ValueError("refined bridge action has no searchable identity")
            query_id = str(row["query_id"])
            if identity in _completed_identities(receipt_index[query_id]):
                continue
            if identity in seen_action_identities:
                continue
            seen_action_identities.add(identity)
            refined_proposals.append(
                {
                    key: value
                    for key, value in {
                        **row,
                        "action_batch": action_payload,
                        "proposal": _proposal_payload(proposal),
                    }.items()
                    if not key.startswith("_")
                }
            )
            eligible_by_signal[signal] += 1
        proposals = refined_proposals
        print(
            "refined-cross-query-suppression="
            f"{','.join(sorted(suppressed_cross_query_terms)) or 'none'} "
            f"eligible={len(proposals)}",
            flush=True,
        )
    else:
        proposals = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in proposals
        ]
    selected = (
        _refined_strata_sample(
            proposals,
            limit=args.query_count,
            negation_target=(0 if args.unconstrained_only else args.query_count // 2),
        )
        if args.refined_strata
        else _balanced_sample(proposals, limit=args.query_count)
    )
    if len(selected) != args.query_count:
        raise ValueError(
            f"insufficient eligible non-overlapping proposals: {len(selected)}"
        )
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
    diagnostics = [
        {
            "query_id": row["query_id"],
            "signal": row["signal"],
            "length_bucket": row["length_bucket"],
            **cast(dict[str, object], row["proposal"]),
        }
        for row in selected
    ]
    baseline_rows = [
        {
            "query_id": row["query_id"],
            "candidate_aliases": row["baseline_aliases"],
            "candidate_count": row["baseline_candidate_count"],
        }
        for row in selected
    ]
    partition_bytes = _jsonl(partition_rows)
    actions_bytes = _canonical_bytes(actions)
    diagnostics_bytes = _jsonl(diagnostics)
    baseline_bytes = _jsonl(baseline_rows)
    action_texts = [
        str(row["proposal"]["query_text"])
        for row in selected
    ]
    if len(set(action_texts)) != args.query_count:
        raise ValueError("frozen bridge actions are not globally unique")
    signal_counts: dict[str, int] = defaultdict(int)
    length_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        signal_counts[str(row["signal"])] += 1
        length_counts[str(row["length_bucket"])] += 1
    manifest = {
        "schema_version": (
            "contrastive-openalex-bridge-validation-v2"
            if args.refined_strata
            else "contrastive-openalex-bridge-validation-v1"
        ),
        "query_count": args.query_count,
        "action_count": args.query_count,
        "max_raw_openalex_requests": args.max_raw_requests,
        "max_results_per_action": 50,
        "action_policy": (
            "contrastive-bridge-anchor-conditioned-v2"
            if args.refined_strata
            else "contrastive-bridge-local-idf-v1"
        ),
        "sample_policy": (
            "online-only-no-hit-unconstrained-refined-v2"
            if args.refined_strata and args.unconstrained_only
            else "online-only-no-hit-negation-unconstrained-refined-v2"
            if args.refined_strata
            else "online-only-no-hit-exact-context-balanced-v2"
        ),
        "focus_signals": list(signal_order),
        "proposal_pool_size": len(proposals),
        "considered_query_count": considered_queries,
        "online_no_hit_query_count": online_no_hit_queries,
        "eligible_proposal_count_by_signal": dict(sorted(eligible_by_signal.items())),
        "initial_eligible_proposal_count_by_signal": initial_eligible_by_signal,
        "suppressed_cross_query_expansion_terms": sorted(
            suppressed_cross_query_terms
        ),
        "excluded_prior_query_count": len(excluded_ids),
        "signal_counts": dict(sorted(signal_counts.items())),
        "length_bucket_counts": dict(sorted(length_counts.items())),
        "inputs": {
            "handoff_sha256": _sha256(handoff.read_bytes()),
            "partition_sha256": _sha256(partition.read_bytes()),
            "production_bundle_sha256": _sha256(bundle.read_bytes()),
            "context_manifest_sha256": _sha256(context_manifest.read_bytes()),
        },
        "outputs": {
            "partition_sha256": _sha256(partition_bytes),
            "actions_sha256": _sha256(actions_bytes),
            "diagnostics_sha256": _sha256(diagnostics_bytes),
            "baseline_candidates_sha256": _sha256(baseline_bytes),
        },
        "generation_gold_blind": True,
        "pasa_used_for_action_generation": False,
        "llm_requests_made": 0,
        "test_partition_touched": False,
        "production_lock_modified": False,
        "training_started": False,
    }
    _write_immutable(output / "partition.jsonl", partition_bytes)
    _write_immutable(output / "actions.json", actions_bytes)
    _write_immutable(output / "proposal-diagnostics.jsonl", diagnostics_bytes)
    _write_immutable(output / "baseline-candidates.jsonl", baseline_bytes)
    _write_immutable(output / "manifest.json", json.dumps(manifest, indent=2).encode())
    print(json.dumps(manifest, indent=2), flush=True)


def _verify_frozen_package(output: Path) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("validation manifest is invalid")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("validation manifest output hashes are missing")
    paths = {
        "partition_sha256": output / "partition.jsonl",
        "actions_sha256": output / "actions.json",
        "diagnostics_sha256": output / "proposal-diagnostics.jsonl",
        "baseline_candidates_sha256": output / "baseline-candidates.jsonl",
    }
    for key, path in paths.items():
        if outputs.get(key) != _sha256(path.read_bytes()):
            raise ValueError(f"frozen validation artifact hash mismatch: {path}")
    rows = _load_jsonl(output / "partition.jsonl")
    actions = json.loads((output / "actions.json").read_text(encoding="utf-8"))
    if not isinstance(actions, dict):
        raise ValueError("frozen action plan is invalid")
    query_ids = [str(row.get("query_id", "")) for row in rows]
    if (
        len(rows) != manifest.get("query_count")
        or len(set(query_ids)) != len(query_ids)
        or set(actions) != set(query_ids)
        or any(row.get("role") != "training" or row.get("split") != "auto_train" for row in rows)
    ):
        raise ValueError("frozen validation partition isolation failed")
    identities: list[SearchActionIdentity] = []
    for query_id in query_ids:
        batch = RecallActionBatch.model_validate(actions[query_id])
        payload = batch.model_dump(mode="json")
        assert_no_forbidden_identifier_keys_or_patterns(payload)
        if len(batch.actions) != 1 or batch.actions[0].action_type != "text_search":
            raise ValueError("frozen action plan must contain one text action per query")
        identity = search_action_identity(payload["actions"][0])
        if identity is None or identity.search_mode != "lexical":
            raise ValueError("frozen action plan must remain lexical")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError("frozen action identities are not globally unique")
    if manifest.get("llm_requests_made") != 0 or manifest.get("test_partition_touched") is not False:
        raise ValueError("frozen validation safety flags failed")
    return cast(dict[str, object], manifest), rows, cast(dict[str, object], actions)


def _identifier_map_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    identifiers: dict[str, str] = {}
    for row in rows:
        values = row.get("gold_paper_ids")
        if not isinstance(values, list):
            raise ValueError("local evaluation row lacks Gold IDs")
        for value in values:
            folded = str(value).casefold()
            if folded.startswith("arxiv:"):
                identifiers[folded] = "doi:10.48550/arxiv." + folded.removeprefix("arxiv:")
    return _canonical_bytes(identifiers)


def _loaded_input(rows: Sequence[Mapping[str, object]]) -> LoadedCanaryInput:
    cases = tuple(
        RecallCase(
            query_id=str(row["query_id"]),
            query=str(row["query"]),
            gold_paper_ids=tuple(str(value) for value in cast(list[object], row["gold_paper_ids"])),
        )
        for row in rows
    )
    input_bytes = _canonical_bytes([case.model_dump(mode="json") for case in cases])
    identifier_bytes = _identifier_map_bytes(rows)
    return LoadedCanaryInput(
        input_kind="jsonl",
        cases=cases,
        evaluation_status="available",
        input_sha256=_sha256(input_bytes),
        identifier_map_bytes=identifier_bytes,
        identifier_map_sha256=_sha256(identifier_bytes),
    )


def _next_batch_paths(receipts_root: Path) -> tuple[Path, Path]:
    index = 1
    while True:
        run_path = receipts_root / "openalex" / f"batch-{index:04d}"
        capture_path = receipts_root / "captures" / "openalex" / f"batch-{index:04d}"
        if not run_path.exists() and not capture_path.exists():
            return run_path, capture_path
        index += 1


async def _collect(args: argparse.Namespace) -> None:
    workspace_root = Path(args.workspace_root).resolve()
    output = (workspace_root / args.output).resolve()
    manifest, rows, actions = _verify_frozen_package(output)
    if int(manifest["max_raw_openalex_requests"]) != args.max_raw_requests:
        raise ValueError("requested collection cap differs from frozen authorization")
    receipts_root = output / "receipts"
    receipts_root.mkdir(parents=True, exist_ok=True)
    completed = load_settled_search_action_identities([receipts_root])
    expected: dict[str, SearchActionIdentity] = {}
    for query_id, raw in actions.items():
        batch = RecallActionBatch.model_validate(raw)
        identity = search_action_identity(batch.actions[0].model_dump(mode="json"))
        assert identity is not None
        expected[query_id] = identity
    pending_rows = [
        row
        for row in rows
        if expected[str(row["query_id"])] not in completed.get(str(row["query_id"]), frozenset())
    ]
    window = current_openalex_quota_window(datetime.now(UTC))
    ledger = SQLiteOpenAlexDailyQuotaLedger(
        output / "quota-ledger.sqlite3",
        window=window,
        key_slot=args.key_slot,
        max_search_calls=args.max_raw_requests,
        clock=lambda: datetime.now(UTC),
    )
    profile = load_runtime_profile((workspace_root / args.profile).resolve())
    secrets = resolve_runtime_secrets(profile, openalex_key_slot=args.key_slot)
    recipe = load_recall_recipe((workspace_root / args.recipe).resolve())
    print(
        f"preflight queries={len(rows)} pending={len(pending_rows)} "
        f"used_raw={ledger.used_search_calls} cap={args.max_raw_requests} llm=0",
        flush=True,
    )
    for offset in range(0, len(pending_rows), args.chunk_size):
        if ledger.used_search_calls >= args.max_raw_requests:
            break
        batch_rows = pending_rows[offset : offset + args.chunk_size]
        query_ids = [str(row["query_id"]) for row in batch_rows]
        run_path, capture_path = _next_batch_paths(receipts_root)
        bundle = await build_live_runtime_bundle(
            profile=profile,
            secrets=secrets,
            loaded_recipe=recipe,
            capture_root=capture_path,
            search_dependency="openalex",
            openalex_attempt_gate=ledger,
        )
        try:
            await RecallCanaryService(workspace_root=workspace_root).run(
                loaded_recipe=recipe,
                loaded_input=_loaded_input(batch_rows),
                runtime_bundle=bundle,
                output_path=run_path,
                generator_override=FixedActionGenerator(
                    {query_id: cast(dict[str, object], actions[query_id]) for query_id in query_ids},
                    expected_query_ids=query_ids,
                    allowed_actions=["text_search"],
                    max_actions=1,
                    source_sha256=str(cast(dict[str, object], manifest["outputs"])["actions_sha256"]),
                ),
            )
        finally:
            await bundle.aclose()
        print(
            f"collected={min(offset + len(batch_rows), len(pending_rows))}/"
            f"{len(pending_rows)} used_raw={ledger.used_search_calls}",
            flush=True,
        )
    completed = load_settled_search_action_identities([receipts_root])
    completed_count = sum(
        expected[str(row["query_id"])]
        in completed.get(str(row["query_id"]), frozenset())
        for row in rows
    )
    status = {
        "schema_version": "contrastive-openalex-bridge-collection-status-v1",
        "window": window.isoformat(),
        "key_slot": args.key_slot,
        "query_count": len(rows),
        "completed_query_count": completed_count,
        "missing_query_count": len(rows) - completed_count,
        "ledger_used_raw_openalex_requests": ledger.used_search_calls,
        "max_raw_openalex_requests": args.max_raw_requests,
        "llm_requests_made": 0,
        "test_partition_touched": False,
    }
    (output / "collection-status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2), flush=True)


def collect(args: argparse.Namespace) -> None:
    asyncio.run(_collect(args))


def _retrieved_papers(
    receipts_root: Path,
    query_id: str,
    expected: SearchActionIdentity,
) -> list[object]:
    papers: list[object] = []
    for retrieval_path in receipts_root.rglob(f"retrieval/attempt-01/{query_id}.json"):
        generation_path = Path(
            *[
                "generation" if part == "retrieval" else part
                for part in retrieval_path.parts
            ]
        )
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        raw_actions = generation.get("actions") if isinstance(generation, dict) else None
        raw_results = retrieval.get("results") if isinstance(retrieval, dict) else None
        if not isinstance(raw_actions, list) or not isinstance(raw_results, list):
            continue
        action_ids = {
            str(action["action_id"])
            for action in raw_actions
            if isinstance(action, dict)
            and isinstance(action.get("action_id"), str)
            and search_action_identity(action) == expected
        }
        for result in raw_results:
            if (
                isinstance(result, dict)
                and result.get("action_id") in action_ids
                and result.get("infrastructure_failure") is not True
                and _terminal_openalex_result_is_usable(result.get("errors"))
                and isinstance(result.get("hits"), list)
            ):
                from paper_search.domain.models import Paper

                papers.extend(Paper.model_validate(hit) for hit in result["hits"])
    return papers


def _terminal_openalex_result_is_usable(errors: object) -> bool:
    """Accept saved hits when every warning is the settled invalid-work type."""

    if not isinstance(errors, list):
        return False
    return not errors or all(
        isinstance(error, Mapping)
        and error.get("code") == "invalid_work"
        and error.get("retryable") is False
        for error in errors
    )


def evaluate(args: argparse.Namespace) -> None:
    workspace_root = Path(args.workspace_root).resolve()
    output = (workspace_root / args.output).resolve()
    manifest, rows, actions = _verify_frozen_package(output)
    baseline_by_query = {
        str(row["query_id"]): row for row in _load_jsonl(output / "baseline-candidates.jsonl")
    }
    completed = load_settled_search_action_identities([output / "receipts"])
    per_query: list[dict[str, object]] = []
    total_gold = 0
    total_hit_gold = 0
    retrieved_candidate_count = 0
    overlapping_candidate_count = 0
    grounded_candidate_count = 0
    complete_count = 0
    for row in rows:
        query_id = str(row["query_id"])
        batch = RecallActionBatch.model_validate(actions[query_id])
        expected = search_action_identity(batch.actions[0].model_dump(mode="json"))
        assert expected is not None
        is_complete = expected in completed.get(query_id, frozenset())
        complete_count += int(is_complete)
        raw_papers = _retrieved_papers(output / "receipts", query_id, expected)
        papers = deduplicate_papers(cast(Sequence[Any], raw_papers)).papers if raw_papers else []
        gold_ids = [str(value) for value in cast(list[object], row["gold_paper_ids"])]
        hit_gold = [
            gold_id
            for gold_id in gold_ids
            if any(paper_matches_evaluation_ids(paper, [gold_id]) for paper in papers)
        ]
        baseline_aliases = set(cast(list[str], baseline_by_query[query_id]["candidate_aliases"]))
        overlap = sum(
            bool(paper_evaluation_aliases(paper) & baseline_aliases) for paper in papers
        )
        query_terms = set(query_content_terms(str(row["query"])))
        grounded = sum(
            bool(
                query_terms
                & set(query_content_terms(f"{paper.title} {paper.abstract or ''}"))
            )
            for paper in papers
        )
        total_gold += len(gold_ids)
        total_hit_gold += len(hit_gold)
        retrieved_candidate_count += len(papers)
        overlapping_candidate_count += overlap
        grounded_candidate_count += grounded
        per_query.append(
            {
                "query_id": query_id,
                "action_complete": is_complete,
                "baseline_gold_hit": False,
                "bridge_gold_hit": bool(hit_gold),
                "bridge_gold_association_count": len(hit_gold),
                "retrieved_candidate_count": len(papers),
                "baseline_overlap_count": overlap,
                "new_unique_candidate_count": len(papers) - overlap,
                "query_grounded_candidate_count": grounded,
            }
        )
    if complete_count != len(rows):
        progress = {
            "query_count": len(rows),
            "completed_query_count": complete_count,
            "missing_query_count": len(rows) - complete_count,
        }
        (output / "evaluation-progress.json").write_text(
            json.dumps(progress, indent=2), encoding="utf-8"
        )
        raise ValueError(f"collection incomplete: {complete_count}/{len(rows)}")
    hit_queries = sum(bool(row["bridge_gold_hit"]) for row in per_query)
    result = {
        "schema_version": "contrastive-openalex-bridge-paired-evaluation-v1",
        "query_count": len(rows),
        "completed_action_count": complete_count,
        "baseline_gold_hit_query_count": 0,
        "augmented_gold_hit_query_count": hit_queries,
        "incremental_gold_hit_query_count": hit_queries,
        "incremental_gold_association_count": total_hit_gold,
        "baseline_macro_candidate_recall": 0.0,
        "augmented_macro_candidate_recall": sum(
            cast(int, row["bridge_gold_association_count"])
            / len(cast(list[object], source["gold_paper_ids"]))
            for row, source in zip(per_query, rows, strict=True)
        )
        / len(rows),
        "augmented_micro_candidate_recall": total_hit_gold / total_gold,
        "retrieved_candidate_count": retrieved_candidate_count,
        "baseline_overlap_candidate_count": overlapping_candidate_count,
        "new_unique_candidate_count": retrieved_candidate_count - overlapping_candidate_count,
        "duplicate_overlap_rate": (
            overlapping_candidate_count / retrieved_candidate_count
            if retrieved_candidate_count
            else 0.0
        ),
        "query_grounded_candidate_rate": (
            grounded_candidate_count / retrieved_candidate_count
            if retrieved_candidate_count
            else 0.0
        ),
        "candidate_membership_monotonic": True,
        "generation_gold_blind": True,
        "pasa_used_for_action_generation": False,
        "llm_requests_made": 0,
        "test_partition_touched": False,
        "production_lock_modified": False,
        "training_started": False,
        "frozen_manifest_sha256": _sha256((output / "manifest.json").read_bytes()),
        "per_query": per_query,
    }
    result_bytes = json.dumps(result, indent=2).encode("utf-8")
    _write_immutable(output / "paired-evaluation-result-v1.json", result_bytes)
    print(json.dumps({key: value for key, value in result.items() if key != "per_query"}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "collect", "evaluate"))
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--production-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--context-manifest", type=Path, default=DEFAULT_CONTEXT_MANIFEST)
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--proposal-pool-size", type=int, default=256)
    parser.add_argument("--max-raw-requests", type=int, default=160)
    parser.add_argument("--key-slot", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--refined-strata", action="store_true")
    parser.add_argument("--unconstrained-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.query_count != 128 or args.max_raw_requests != 160:
        raise ValueError("this frozen validation is authorized for 128 queries and 160 raw requests")
    if args.unconstrained_only and not args.refined_strata:
        raise ValueError("unconstrained-only requires the refined v2 action policy")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "collect":
        collect(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
