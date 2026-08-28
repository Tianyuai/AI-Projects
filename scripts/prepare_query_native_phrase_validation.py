"""Freeze a bounded, Gold-blind query-native title-phrase validation package."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from paper_search.domain.models import QuerySpec
from paper_search.evaluation.predictions import paper_evaluation_aliases
from paper_search.learning.large_scale_fusion_training import (
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
    query_has_gold_candidate,
)
from paper_search.learning.openalex_daily_schedule import search_action_identity
from paper_search.learning.query_native_phrase_bridge import (
    QueryNativeTitlePhraseProposal,
    build_query_native_title_phrase_action_batch,
    propose_query_native_title_phrase_bridge,
)
from paper_search.learning.unified_recall_context import (
    load_frozen_recall_query_specs,
)
from paper_search.recall_experiments.contracts import (
    assert_no_forbidden_identifier_keys_or_patterns,
)
from scripts.run_cross_vocabulary_openalex_validation import (
    DEFAULT_BUNDLE,
    DEFAULT_HANDOFF,
    DEFAULT_PARTITION,
    _canonical_bytes,
    _completed_identities,
    _excluded_query_ids,
    _jsonl,
    _online_only_package,
    _sha256,
    _write_immutable,
)


DEFAULT_CONTEXT_MANIFEST = Path(
    "data/training_private/training_runs/"
    "openalex-pasa-high-recall-expanded-21429-v1/"
    "unified-context/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "data/training_private/recall_policy/"
    "query-native-title-phrase-validation24-v3"
)


def _signal(spec: QuerySpec) -> str:
    if spec.exclusions:
        return "negation"
    if spec.datasets:
        return "dataset"
    if spec.methods:
        return "method"
    if spec.tasks:
        return "task"
    if spec.year_from is not None or spec.year_to is not None:
        return "year"
    return "unconstrained"


def _balanced_sample(
    rows: Sequence[dict[str, object]],
    *,
    limit: int,
    unconstrained_target: int,
) -> list[dict[str, object]]:
    if limit <= 0 or not 0 <= unconstrained_target <= limit:
        raise ValueError("sample bounds are invalid")
    unconstrained = [row for row in rows if row.get("signal") == "unconstrained"]
    structured = [row for row in rows if row.get("signal") != "unconstrained"]
    selected = [
        *unconstrained[:unconstrained_target],
        *structured[: limit - unconstrained_target],
    ]
    selected_ids = {str(row["query_id"]) for row in selected}
    for row in rows:
        query_id = str(row["query_id"])
        if len(selected) == limit:
            break
        if query_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(query_id)
    ordered_ids = {str(row["query_id"]) for row in selected}
    return [row for row in rows if str(row["query_id"]) in ordered_ids][:limit]


def _proposal_payload(
    proposal: QueryNativeTitlePhraseProposal,
) -> dict[str, object]:
    return {
        "query_text": proposal.query_text,
        "query_anchors": list(proposal.query_anchors),
        "supported_phrase": proposal.supported_phrase,
        "phrase_candidate_support": proposal.phrase_candidate_support,
        "phrase_action_support": proposal.phrase_action_support,
        "phrase_anchor_cooccurrence": proposal.phrase_anchor_cooccurrence,
        "grounded_candidate_count": proposal.grounded_candidate_count,
        "online_candidate_count": proposal.online_candidate_count,
    }


def _prepare(args: argparse.Namespace) -> None:
    root = Path(args.workspace_root).resolve()
    resolve = lambda value: (root / value).resolve()  # noqa: E731
    handoff_path = resolve(args.handoff)
    partition_path = resolve(args.partition)
    bundle_path = resolve(args.production_bundle)
    context_manifest_path = resolve(args.context_manifest)
    output = resolve(args.output)

    package = load_training_package(
        handoff_path=handoff_path,
        partition_path=partition_path,
        production_bundle_path=bundle_path,
    )
    online_package = _online_only_package(package)
    receipt_index = index_training_receipts(online_package)
    specs = load_frozen_recall_query_specs(
        partition_path=partition_path,
        manifest_path=context_manifest_path,
    )
    excluded_ids = _excluded_query_ids(root)
    ordered_ids = sorted(
        (query_id for query_id in package.query_ids if query_id not in excluded_ids),
        key=lambda query_id: hashlib.sha256(query_id.encode()).hexdigest(),
    )

    counters: Counter[str] = Counter()
    proposals: list[dict[str, Any]] = []
    seen_action_identities: set[object] = set()
    seen_action_texts: set[str] = set()
    for query_id in ordered_ids:
        if counters["scanned"] >= args.scan_cap:
            break
        counters["scanned"] += 1
        online_query = build_document_ranking_query(
            online_package,
            query_id,
            receipt_index[query_id],
        )
        if query_has_gold_candidate(online_query):
            counters["online_gold_hit"] += 1
            continue
        counters["online_no_hit"] += 1
        if any(
            "pasa" in value.casefold()
            for candidate in online_query.candidates
            for value in [*candidate.paper.sources, *candidate.source_ranks]
        ):
            raise ValueError(f"PASA entered online-only phrase evidence: {query_id}")
        spec = specs[query_id]
        proposal = propose_query_native_title_phrase_bridge(
            spec,
            online_query.candidates,
        )
        if proposal is None:
            counters["abstained"] += 1
            continue
        action_batch = build_query_native_title_phrase_action_batch(proposal)
        action_payload = action_batch.model_dump(mode="json")
        assert_no_forbidden_identifier_keys_or_patterns(action_payload)
        identity = search_action_identity(action_payload["actions"][0])
        if identity is None:
            raise ValueError("query-native phrase action has no searchable identity")
        if identity in _completed_identities(receipt_index[query_id]):
            counters["completed_action_identity"] += 1
            continue
        action_text = proposal.query_text.casefold()
        if identity in seen_action_identities or action_text in seen_action_texts:
            counters["duplicate_action_identity"] += 1
            continue
        seen_action_identities.add(identity)
        seen_action_texts.add(action_text)
        signal = _signal(spec)
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
                "signal": signal,
                "action_batch": action_payload,
                "proposal": _proposal_payload(proposal),
                "baseline_aliases": baseline_aliases,
                "baseline_candidate_count": len(online_query.candidates),
            }
        )
        counters["proposed"] += 1
        counters[f"proposed_{signal}"] += 1
        structured_count = counters["proposed"] - counters["proposed_unconstrained"]
        if (
            counters["proposed"] >= args.proposal_pool_target
            and counters["proposed_unconstrained"] >= args.unconstrained_target
            and structured_count >= args.query_count - args.unconstrained_target
        ):
            break

    selected = _balanced_sample(
        proposals,
        limit=args.query_count,
        unconstrained_target=args.unconstrained_target,
    )
    if len(selected) != args.query_count:
        raise ValueError(
            f"insufficient unique query-native phrase proposals: {len(selected)}"
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
    signal_counts = Counter(str(row["signal"]) for row in selected)
    module_path = root / "src/paper_search/learning/query_native_phrase_bridge.py"
    manifest = {
        "schema_version": "query-native-title-phrase-validation-package-v3",
        "query_count": args.query_count,
        "action_count": args.query_count,
        "action_policy": "query-native-title-phrase-v3",
        "selection_policy": "sha256-order-online-no-hit-balanced-structured-unconstrained-v1",
        "proposal_pool_target": args.proposal_pool_target,
        "proposal_pool_size": len(proposals),
        "scan_cap": args.scan_cap,
        "scan_counts": dict(sorted(counters.items())),
        "selected_signal_counts": dict(sorted(signal_counts.items())),
        "excluded_prior_query_count": len(excluded_ids),
        "max_actions_per_query": 1,
        "max_results_per_action": 50,
        "inputs": {
            "handoff_sha256": _sha256(handoff_path.read_bytes()),
            "partition_sha256": _sha256(partition_path.read_bytes()),
            "production_bundle_sha256": _sha256(bundle_path.read_bytes()),
            "unified_context_manifest_sha256": _sha256(
                context_manifest_path.read_bytes()
            ),
            "query_native_phrase_module_sha256": _sha256(module_path.read_bytes()),
        },
        "outputs": {
            "partition_sha256": _sha256(partition_bytes),
            "actions_sha256": _sha256(actions_bytes),
            "diagnostics_sha256": _sha256(diagnostics_bytes),
            "baseline_candidates_sha256": _sha256(baseline_bytes),
        },
        "safety": {
            "generation_gold_blind": True,
            "pasa_used_for_action_generation": False,
            "llm_requests_made": 0,
            "online_requests_made": 0,
            "test_partition_touched": False,
            "production_lock_modified": False,
            "training_started": False,
        },
        "decision": {
            "offline_action_gate_passed": True,
            "online_collection_authorized": False,
            "next_step": "request bounded OpenAlex/S2 paired collection authorization",
        },
    }
    _write_immutable(output / "partition.jsonl", partition_bytes)
    _write_immutable(output / "actions.json", actions_bytes)
    _write_immutable(output / "proposal-diagnostics.jsonl", diagnostics_bytes)
    _write_immutable(output / "baseline-candidates.jsonl", baseline_bytes)
    _write_immutable(
        output / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--production-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--context-manifest", type=Path, default=DEFAULT_CONTEXT_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-count", type=int, default=24)
    parser.add_argument("--unconstrained-target", type=int, default=12)
    parser.add_argument("--proposal-pool-target", type=int, default=48)
    parser.add_argument("--scan-cap", type=int, default=2500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.query_count <= 0 or not 0 <= args.unconstrained_target <= args.query_count:
        raise ValueError("query sample bounds are invalid")
    if args.proposal_pool_target < args.query_count or args.scan_cap <= 0:
        raise ValueError("proposal pool or scan cap is invalid")
    _prepare(args)


if __name__ == "__main__":
    main()
