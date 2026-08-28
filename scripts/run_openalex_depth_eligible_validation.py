"""Freeze, collect, and evaluate a prerequisite-controlled page-two sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.domain.models import Paper  # noqa: E402
from paper_search.evaluation.dataset import IdentifierMap  # noqa: E402
from paper_search.learning.candidates import query_content_terms  # noqa: E402
from paper_search.learning.large_scale_fusion_training import (  # noqa: E402
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
)
from paper_search.learning.openalex_daily_schedule import (  # noqa: E402
    search_action_identity,
)
from paper_search.retrieval.pasa_paper_database import (  # noqa: E402
    PasaPaperDatabase,
)
from scripts import run_openalex_depth_validation as depth_v1  # noqa: E402
from scripts.attribute_cross_vocabulary_misses_with_pasa import (  # noqa: E402
    DEFAULT_PASA_INDEX,
)
from scripts.prepare_online_miss_provider_validation import (  # noqa: E402
    collect_prior_query_ids,
)
from scripts.run_cross_vocabulary_openalex_validation import (  # noqa: E402
    _online_only_package,
)
from scripts.run_provider_recall_comparison import (  # noqa: E402
    _load_production_identifier_context,
)


DEFAULT_OUTPUT = Path(
    "data/training_private/recall_policy/openalex-depth-eligible32-v2"
)
DEFAULT_REFERENCE_ROOT = Path(
    "data/training_private/recall_policy/openalex-depth-continuation64-v1"
)
DEFAULT_QUERY_COUNT = 32
DEFAULT_RAW_REQUEST_CAP = 32
_SELECTION_SEED = "openalex-depth-eligible32-v2"
_SELECTION_POLICY = (
    "sha256-disjoint-openalex-verified-query-action-aligned-exact-page1-v2"
)


def _normalized_id(value: object) -> str:
    return str(value).strip().casefold()


def _content_bigrams(value: str) -> set[str]:
    terms = query_content_terms(value)
    return {
        f"{terms[index]} {terms[index + 1]}"
        for index in range(len(terms) - 1)
    }


def _alignment_record(
    *,
    query: str,
    action_text: str,
    gold_id: str,
    paper: Paper,
    verified_openalex_gold_ids: set[str],
) -> dict[str, object]:
    query_terms = set(query_content_terms(query))
    action_terms = set(query_content_terms(action_text))
    title_terms = set(query_content_terms(paper.title))
    gold_text = f"{paper.title} {paper.abstract or ''}"
    gold_terms = set(query_content_terms(gold_text))
    query_title_overlap = len(query_terms.intersection(title_terms))
    query_text_overlap = len(query_terms.intersection(gold_terms))
    action_title_overlap = len(action_terms.intersection(title_terms))
    action_text_overlap = len(action_terms.intersection(gold_terms))
    query_phrase_overlap = len(
        _content_bigrams(query).intersection(_content_bigrams(gold_text))
    )
    action_phrase_overlap = len(
        _content_bigrams(action_text).intersection(_content_bigrams(gold_text))
    )
    metadata_complete = bool(
        paper.title.strip() and (paper.abstract or "").strip()
    )
    query_aligned = (
        query_phrase_overlap > 0
        or query_title_overlap >= 2
        or query_text_overlap >= 3
    )
    action_aligned = (
        action_phrase_overlap > 0
        or action_title_overlap >= 2
        or action_text_overlap >= 3
    )
    return {
        "gold_id": gold_id,
        "metadata_complete": metadata_complete,
        "openalex_identity_metadata_verified": (
            gold_id in verified_openalex_gold_ids
        ),
        "query_gold_title_term_overlap": query_title_overlap,
        "query_gold_text_term_overlap": query_text_overlap,
        "query_gold_phrase_overlap": query_phrase_overlap,
        "action_gold_title_term_overlap": action_title_overlap,
        "action_gold_text_term_overlap": action_text_overlap,
        "action_gold_phrase_overlap": action_phrase_overlap,
        "query_aligned": query_aligned,
        "action_aligned": action_aligned,
    }


def classify_depth_eligibility(
    *,
    query: str,
    action_text: str,
    gold_paper_ids: Sequence[str],
    pasa_gold_papers: Mapping[str, Paper],
    verified_openalex_gold_ids: set[str],
) -> dict[str, object]:
    """Classify whether page depth can be isolated from three known confounders."""

    normalized_papers = {
        _normalized_id(key): paper for key, paper in pasa_gold_papers.items()
    }
    verified = {_normalized_id(value) for value in verified_openalex_gold_ids}
    records = [
        _alignment_record(
            query=query,
            action_text=action_text,
            gold_id=gold_id,
            paper=normalized_papers[gold_id],
            verified_openalex_gold_ids=verified,
        )
        for gold_id in (_normalized_id(value) for value in gold_paper_ids)
        if gold_id in normalized_papers
    ]
    complete = [row for row in records if row["metadata_complete"]]
    verified_rows = [
        row
        for row in complete
        if row["openalex_identity_metadata_verified"]
    ]
    query_aligned = [row for row in verified_rows if row["query_aligned"]]
    eligible_rows = [row for row in query_aligned if row["action_aligned"]]
    if not records:
        category = "pasa_gold_metadata_unavailable"
    elif not complete:
        category = "gold_metadata_incomplete"
    elif not verified_rows:
        category = "openalex_identity_metadata_unverified"
    elif not query_aligned:
        category = "query_gold_vocabulary_mismatch"
    elif not eligible_rows:
        category = "action_gold_technical_expression_mismatch"
    else:
        category = "eligible"
    candidates = eligible_rows or query_aligned or verified_rows or complete or records
    selected = max(
        candidates,
        key=lambda row: (
            int(row["action_gold_phrase_overlap"]),
            int(row["action_gold_title_term_overlap"]),
            int(row["action_gold_text_term_overlap"]),
            int(row["query_gold_phrase_overlap"]),
            int(row["query_gold_title_term_overlap"]),
            str(row["gold_id"]),
        ),
        default=None,
    )
    overlaps = {
        key: (0 if selected is None else selected[key])
        for key in (
            "query_gold_title_term_overlap",
            "query_gold_text_term_overlap",
            "query_gold_phrase_overlap",
            "action_gold_title_term_overlap",
            "action_gold_text_term_overlap",
            "action_gold_phrase_overlap",
        )
    }
    return {
        "category": category,
        "eligible": category == "eligible",
        "expected_gold_count": len(gold_paper_ids),
        "pasa_gold_count": len(records),
        "metadata_complete_gold_count": len(complete),
        "openalex_verified_gold_count": len(verified_rows),
        "eligible_gold_count": len(eligible_rows),
        "selected_gold_id": None if selected is None else selected["gold_id"],
        **overlaps,
    }


def verified_openalex_gold_ids_from_map_bytes(content: bytes) -> set[str]:
    """Return arXiv targets with a production-bound OpenAlex or DOI alias."""

    raw: object = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("production identifier map must be a JSON object")
    return {
        target
        for raw_source, raw_target in raw.items()
        if _normalized_id(raw_source).startswith(("openalex:", "doi:"))
        and (target := _normalized_id(raw_target)).startswith("arxiv:")
    }


def select_eligible_depth_rows(
    rows: Sequence[dict[str, object]],
    *,
    prior_query_ids: set[str],
    limit: int,
    seed: str,
) -> list[dict[str, object]]:
    """Select only rows whose local evidence isolates page depth."""

    eligible = [
        row
        for row in rows
        if isinstance(row.get("local_eligibility"), Mapping)
        and cast(Mapping[str, object], row["local_eligibility"]).get("category")
        == "eligible"
        and row.get("production_identity_baseline_gold_hit") is False
    ]
    return depth_v1.select_disjoint_depth_rows(
        eligible,
        prior_query_ids=prior_query_ids,
        limit=limit,
        seed=seed,
    )


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _action_text(proposal: Mapping[str, object]) -> str:
    action = proposal.get("source_action")
    if not isinstance(action, Mapping):
        raise ValueError("depth proposal source action is invalid")
    identity = search_action_identity(dict(action))
    if identity is None or identity.search_mode != "lexical":
        raise ValueError("depth proposal must retain one lexical source action")
    return identity.normalized_text


def _reference_attribution(
    *,
    reference_root: Path,
    pasa_database: PasaPaperDatabase,
    verified_openalex_gold_ids: set[str],
) -> tuple[dict[str, int], str | None]:
    partition_path = reference_root / "partition.jsonl"
    request_path = reference_root / "request-plan.jsonl"
    manifest_path = reference_root / "manifest.json"
    if not partition_path.is_file() or not request_path.is_file():
        return {}, None
    partition_rows = depth_v1._load_jsonl(partition_path)
    requests = {
        str(row["query_id"]): row for row in depth_v1._load_jsonl(request_path)
    }
    gold_ids = [
        str(value)
        for row in partition_rows
        for value in cast(list[object], row["gold_paper_ids"])
    ]
    pasa_by_id = pasa_database.lookup_arxiv_many(gold_ids)
    categories = Counter(
        str(
            classify_depth_eligibility(
                query=str(row["query"]),
                action_text=str(requests[str(row["query_id"])]["derived_query_text"]),
                gold_paper_ids=[
                    str(value)
                    for value in cast(list[object], row["gold_paper_ids"])
                ],
                pasa_gold_papers=pasa_by_id,
                verified_openalex_gold_ids=verified_openalex_gold_ids,
            )["category"]
        )
        for row in partition_rows
    )
    manifest_sha = (
        depth_v1._sha256_file(manifest_path) if manifest_path.is_file() else None
    )
    return dict(sorted(categories.items())), manifest_sha


def prepare(args: argparse.Namespace) -> dict[str, object]:
    """Freeze a Gold-blind request package after local prerequisite filtering."""

    root = Path(args.workspace_root).resolve()
    output = _resolve(root, args.output)
    if output.exists() and (output / "manifest.json").exists():
        raise ValueError("frozen eligible depth validation package already exists")
    audit_root = _resolve(root, args.audit_root)
    handoff_path = _resolve(root, args.handoff)
    partition_path = _resolve(root, args.partition)
    bundle_path = _resolve(root, args.production_bundle)
    lock_path = _resolve(root, args.production_lock)
    pasa_index_path = _resolve(root, args.pasa_index)
    reference_root = _resolve(root, args.reference_root)

    misses = depth_v1._rank_misses(audit_root)
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
    identifier_context = _load_production_identifier_context(
        workspace_root=root,
        lock_path=lock_path,
    )
    verified_openalex_gold_ids = verified_openalex_gold_ids_from_map_bytes(
        identifier_context.identifier_map_bytes
    )
    identifier_map = IdentifierMap.from_bytes(
        identifier_context.identifier_map_bytes,
        source="production combined identifier aliases",
    )
    pasa_database = PasaPaperDatabase(pasa_index_path)
    ordered_ids = sorted(
        (
            query_id
            for query_id in package.query_ids
            if query_id in misses and query_id not in prior_query_ids
        ),
        key=lambda query_id: hashlib.sha256(
            f"{_SELECTION_SEED}\0{query_id}".encode()
        ).hexdigest(),
    )
    all_gold_ids = [
        str(value)
        for query_id in ordered_ids
        for value in cast(
            list[object], package.rows_by_query_id[query_id]["gold_paper_ids"]
        )
    ]
    pasa_by_id = pasa_database.lookup_arxiv_many(all_gold_ids)

    category_counts: Counter[str] = Counter()
    eligible_proposals: list[dict[str, object]] = []
    for query_id in ordered_ids:
        source_row = package.rows_by_query_id[query_id]
        query = str(source_row["query"])
        gold_ids = [
            str(value)
            for value in cast(list[object], source_row["gold_paper_ids"])
        ]
        query_check = classify_depth_eligibility(
            query=query,
            action_text=query,
            gold_paper_ids=gold_ids,
            pasa_gold_papers=pasa_by_id,
            verified_openalex_gold_ids=verified_openalex_gold_ids,
        )
        if query_check["category"] != "eligible":
            category_counts[str(query_check["category"])] += 1
            continue
        proposal = depth_v1._source_proposal(
            package,
            query_id,
            misses[query_id],
            receipt_index[query_id],
        )
        if proposal is None:
            category_counts["no_exact_page1_continuation"] += 1
            continue
        eligibility = classify_depth_eligibility(
            query=query,
            action_text=_action_text(proposal),
            gold_paper_ids=gold_ids,
            pasa_gold_papers=pasa_by_id,
            verified_openalex_gold_ids=verified_openalex_gold_ids,
        )
        category_counts[str(eligibility["category"])] += 1
        if eligibility["category"] == "eligible":
            proposal["local_eligibility"] = eligibility
            eligible_proposals.append(proposal)

    production_miss_proposals: list[dict[str, object]] = []
    baseline_snapshots: dict[str, dict[str, object]] = {}
    for proposal in eligible_proposals:
        query_id = str(proposal["query_id"])
        query = build_document_ranking_query(
            package,
            query_id,
            receipt_index[query_id],
        )
        baseline_hit = depth_v1._has_gold(
            query.candidates,
            query.gold_paper_ids,
            identifier_map,
        )
        proposal["production_identity_baseline_gold_hit"] = baseline_hit
        if baseline_hit:
            category_counts["eligible"] -= 1
            category_counts["production_identity_baseline_hit"] += 1
            continue
        production_miss_proposals.append(proposal)
        baseline_snapshots[query_id] = depth_v1._baseline_snapshot(query)

    selected = select_eligible_depth_rows(
        production_miss_proposals,
        prior_query_ids=prior_query_ids,
        limit=args.query_count,
        seed=_SELECTION_SEED,
    )
    query_ids = tuple(str(row["query_id"]) for row in selected)
    baseline_rows = [baseline_snapshots[query_id] for query_id in query_ids]

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
    request_rows = [depth_v1.build_gold_blind_request_row(row) for row in selected]
    depth_v1.verify_gold_blind_request_plan(request_rows)
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
            "eligibility_category": cast(
                Mapping[str, object], row["local_eligibility"]
            )["category"],
        }
        for row in selected
    ]
    reference_counts, reference_manifest_sha = _reference_attribution(
        reference_root=reference_root,
        pasa_database=pasa_database,
        verified_openalex_gold_ids=verified_openalex_gold_ids,
    )
    selected_evidence = [
        {
            "query_id": row["query_id"],
            "signal": row["signal"],
            "source_action_id": cast(Mapping[str, object], row["source_action"])[
                "action_id"
            ],
            **cast(dict[str, object], row["local_eligibility"]),
        }
        for row in selected
    ]
    eligibility_audit: dict[str, object] = {
        "schema_version": "openalex-depth-eligibility-audit-v2",
        "purpose": "isolate-cursor-depth-from-metadata-and-vocabulary-confounders",
        "source_no_hit_disjoint_query_count": len(ordered_ids),
        "category_counts": dict(sorted(category_counts.items())),
        "pre_identity_eligible_population_count": len(eligible_proposals),
        "eligible_population_count": len(production_miss_proposals),
        "selected_query_count": len(selected),
        "reference_v1_query_count": sum(reference_counts.values()),
        "reference_v1_category_counts": reference_counts,
        "eligibility_policy": {
            "same_gold_must_pass_all_checks": True,
            "pasa_title_and_abstract_required": True,
            "production_openalex_or_doi_alias_required": True,
            "query_alignment": (
                "shared_content_bigram-or-two-title-terms-or-three-title-abstract-terms"
            ),
            "source_action_alignment": (
                "shared_content_bigram-or-two-title-terms-or-three-title-abstract-terms"
            ),
            "gold_used_to_rewrite_outbound_action": False,
        },
        "selected_evidence": selected_evidence,
        "safety": {
            "analysis_only_gold_conditioning": True,
            "outbound_actions_are_unchanged_frozen_source_actions": True,
            "network_request_count": 0,
            "llm_request_count": 0,
            "test_partition_touched": False,
            "production_lock_modified": False,
            "training_started": False,
        },
    }

    partition_bytes = depth_v1._jsonl(partition_rows)
    request_bytes = depth_v1._jsonl(request_rows)
    actions_bytes = depth_v1._canonical_bytes(actions)
    diagnostics_bytes = depth_v1._jsonl(diagnostics)
    baseline_bytes = depth_v1._jsonl(baseline_rows)
    eligibility_bytes = (
        json.dumps(
            eligibility_audit,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    signal_counts = Counter(str(row["signal"]) for row in selected)
    prior_bytes = depth_v1._canonical_bytes(sorted(prior_query_ids))
    manifest: dict[str, object] = {
        "schema_version": "openalex-depth-continuation-validation-v1",
        "purpose": "prerequisite-controlled-disjoint-online-miss-second-page-test",
        "query_count": len(selected),
        "selection_policy": _SELECTION_POLICY,
        "selection_seed": _SELECTION_SEED,
        "source_no_hit_disjoint_query_count": len(ordered_ids),
        "pre_identity_eligible_source_query_count": len(eligible_proposals),
        "eligible_source_query_count": len(production_miss_proposals),
        "excluded_prior_query_count": len(prior_query_ids),
        "prior_query_inventory_sha256": depth_v1._sha256_bytes(prior_bytes),
        "signal_counts": dict(sorted(signal_counts.items())),
        "source_page_size": 50,
        "continuation_page_size": 50,
        "provider_rank_offset": 50,
        "max_raw_openalex_requests": args.max_raw_requests,
        "request_retry_policy": "no-retry-one-attempt-per-query",
        "inputs": {
            "audit_summary_sha256": depth_v1._sha256_file(
                audit_root / "summary.json"
            ),
            "audit_progress_sha256": depth_v1._sha256_file(
                audit_root / "progress.json"
            ),
            "handoff_sha256": depth_v1._sha256_file(handoff_path),
            "partition_sha256": depth_v1._sha256_file(partition_path),
            "production_bundle_sha256": depth_v1._sha256_file(bundle_path),
            "pasa_index_sha256": pasa_database.index_sha256,
            "reference_v1_manifest_sha256": reference_manifest_sha,
            **identifier_context.evidence,
        },
        "outputs": {
            "partition_sha256": depth_v1._sha256_bytes(partition_bytes),
            "request_plan_sha256": depth_v1._sha256_bytes(request_bytes),
            "actions_sha256": depth_v1._sha256_bytes(actions_bytes),
            "diagnostics_sha256": depth_v1._sha256_bytes(diagnostics_bytes),
            "baseline_candidates_sha256": depth_v1._sha256_bytes(baseline_bytes),
            "eligibility_audit_sha256": depth_v1._sha256_bytes(
                eligibility_bytes
            ),
        },
        "request_plan_gold_blind": True,
        "gold_used_for_local_sample_eligibility_only": True,
        "gold_used_to_generate_or_rewrite_actions": False,
        "pasa_used_as_online_candidate_source": False,
        "llm_requests_made": 0,
        "test_partition_touched": False,
        "production_lock_modified": False,
        "training_started": False,
    }
    depth_v1._write_immutable(output / "partition.jsonl", partition_bytes)
    depth_v1._write_immutable(output / "request-plan.jsonl", request_bytes)
    depth_v1._write_immutable(output / "actions.json", actions_bytes)
    depth_v1._write_immutable(output / "diagnostics.jsonl", diagnostics_bytes)
    depth_v1._write_immutable(
        output / "baseline-candidates.jsonl", baseline_bytes
    )
    depth_v1._write_immutable(
        output / "eligibility-audit-v2.json", eligibility_bytes
    )
    depth_v1._write_immutable(
        output / "manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    _verify_eligible_package(output)
    return manifest


def _verify_eligible_package(output: Path) -> None:
    manifest, partition_rows, _request_rows, _actions = depth_v1._verify_package(
        output
    )
    outputs = cast(Mapping[str, object], manifest["outputs"])
    audit_path = output / "eligibility-audit-v2.json"
    audit = json.loads(audit_path.read_bytes())
    if (
        manifest.get("selection_policy") != _SELECTION_POLICY
        or manifest.get("query_count") != DEFAULT_QUERY_COUNT
        or manifest.get("max_raw_openalex_requests") != DEFAULT_RAW_REQUEST_CAP
        or outputs.get("eligibility_audit_sha256")
        != depth_v1._sha256_file(audit_path)
        or not isinstance(audit, dict)
        or audit.get("selected_query_count") != len(partition_rows)
        or any(
            row.get("category") != "eligible"
            for row in cast(list[dict[str, object]], audit.get("selected_evidence"))
        )
    ):
        raise ValueError("eligible depth package verification failed")


def collect(args: argparse.Namespace) -> dict[str, object]:
    output = _resolve(Path(args.workspace_root).resolve(), args.output)
    _verify_eligible_package(output)
    return depth_v1.collect(args)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    output = _resolve(Path(args.workspace_root).resolve(), args.output)
    _verify_eligible_package(output)
    return depth_v1.evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "evaluate"))
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-root", type=Path, default=depth_v1.DEFAULT_AUDIT_ROOT)
    parser.add_argument("--handoff", type=Path, default=depth_v1.DEFAULT_HANDOFF)
    parser.add_argument("--partition", type=Path, default=depth_v1.DEFAULT_PARTITION)
    parser.add_argument(
        "--production-bundle", type=Path, default=depth_v1.DEFAULT_BUNDLE
    )
    parser.add_argument(
        "--production-selection", type=Path, default=depth_v1.DEFAULT_SELECTION
    )
    parser.add_argument(
        "--production-lock", type=Path, default=depth_v1.DEFAULT_LOCK
    )
    parser.add_argument("--pasa-index", type=Path, default=DEFAULT_PASA_INDEX)
    parser.add_argument(
        "--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT
    )
    parser.add_argument("--profile", type=Path, default=depth_v1.DEFAULT_PROFILE)
    parser.add_argument("--recipe", type=Path, default=depth_v1.DEFAULT_RECIPE)
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument(
        "--max-raw-requests", type=int, default=DEFAULT_RAW_REQUEST_CAP
    )
    parser.add_argument(
        "--candidate-cap", type=int, default=depth_v1.DEFAULT_CANDIDATE_CAP
    )
    parser.add_argument("--key-slot", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=4, choices=(1, 2, 4, 8))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.query_count != DEFAULT_QUERY_COUNT:
        raise ValueError("eligible depth package is frozen to exactly 32 queries")
    if args.max_raw_requests != DEFAULT_RAW_REQUEST_CAP:
        raise ValueError("eligible depth collection is capped at 32 raw requests")
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
