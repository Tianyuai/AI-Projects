"""Attribute sealed cross-vocabulary misses with PASA as analysis-only evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from paper_search.domain.models import Paper
from paper_search.evaluation.dataset import normalize_title
from paper_search.learning.candidates import query_content_terms
from paper_search.learning.cpu_document_ranker import DocumentCandidateEvidence
from paper_search.learning.large_scale_fusion_training import (
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
)
from paper_search.recall_experiments.contracts import RecallActionBatch
from paper_search.retrieval.pasa_paper_database import PasaPaperDatabase

try:
    from scripts.evaluate_cross_vocabulary_f5_topk import (
        DEFAULT_HANDOFF,
        DEFAULT_PARTITION,
        DEFAULT_SELECTION,
        DEFAULT_VALIDATION_ROOT,
        _baseline_snapshot_rows,
        _load_production_paths,
        _online_only_package,
        _supplemental_receipts,
        _verify_baseline_snapshot,
        _verify_frozen_package,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from evaluate_cross_vocabulary_f5_topk import (  # type: ignore[no-redef]
        DEFAULT_HANDOFF,
        DEFAULT_PARTITION,
        DEFAULT_SELECTION,
        DEFAULT_VALIDATION_ROOT,
        _baseline_snapshot_rows,
        _load_production_paths,
        _online_only_package,
        _supplemental_receipts,
        _verify_baseline_snapshot,
        _verify_frozen_package,
    )


DEFAULT_PASA_INDEX = Path(
    "data/training_private/pasa_paper_database/index/"
    "232428b0c867268c3b8ded90db4d98c1b30501d6/"
    "pasa-paper-database.sqlite3"
)
DEFAULT_AB_RESULT = "f5-topk-candidate-ab-v1.json"
DEFAULT_OUTPUT = "pasa-miss-attribution-v1.json"
_PHRASE_QUERY_MINIMUM = 10
_PHRASE_QUERY_RATIO_MINIMUM = 0.10


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _action_family(value: str) -> str:
    return value.split("@", 1)[0].casefold()


def _bigrams(value: str) -> tuple[str, ...]:
    terms = query_content_terms(value)
    return tuple(" ".join(terms[index : index + 2]) for index in range(len(terms) - 1))


def _candidate_supported_gold_phrases(
    candidates: Sequence[DocumentCandidateEvidence],
    pasa_gold_papers: Sequence[Paper],
    *,
    anchors: Sequence[str],
    expansion_terms: Sequence[str],
) -> list[dict[str, object]]:
    anchor_terms = {term.casefold() for term in anchors}
    expansion = {term.casefold() for term in expansion_terms}
    if not anchor_terms or not expansion:
        return []
    phrase_candidates: dict[str, set[str]] = defaultdict(set)
    phrase_actions: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        candidate_id = candidate.paper.canonical_id.casefold()
        action_families = {
            _action_family(source) for source in candidate.source_ranks
        }
        for phrase in set(_bigrams(candidate.paper.title)):
            terms = set(phrase.split())
            if not terms.intersection(anchor_terms) or not terms.intersection(expansion):
                continue
            phrase_candidates[phrase].add(candidate_id)
            phrase_actions[phrase].update(action_families)
    gold_phrases = {
        phrase
        for paper in pasa_gold_papers
        for phrase in _bigrams(f"{paper.title} {paper.abstract or ''}")
    }
    output = [
        {
            "phrase": phrase,
            "candidate_support": len(paper_ids),
            "action_support": len(phrase_actions[phrase]),
        }
        for phrase, paper_ids in phrase_candidates.items()
        if phrase in gold_phrases
        and len(paper_ids) >= 2
        and len(phrase_actions[phrase]) >= 2
    ]
    return sorted(
        output,
        key=lambda row: (
            -int(row["candidate_support"]),
            -int(row["action_support"]),
            str(row["phrase"]),
        ),
    )


def attribute_query_miss(
    *,
    query: str,
    action_text: str,
    candidates: Sequence[DocumentCandidateEvidence],
    pasa_gold_papers: Sequence[Paper],
    expected_gold_count: int,
    anchors: Sequence[str],
    expansion_terms: Sequence[str],
) -> dict[str, object]:
    """Classify one miss using local metadata without changing any search action."""

    if expected_gold_count <= 0:
        raise ValueError("PASA miss attribution requires expected Gold papers")
    if not pasa_gold_papers:
        return {
            "category": "pasa_gold_metadata_unavailable",
            "expected_gold_count": expected_gold_count,
            "pasa_gold_count": 0,
            "pasa_gold_coverage_complete": False,
            "exact_title_alias_candidate_count": 0,
            "query_gold_title_overlap": 0,
            "action_gold_title_overlap": 0,
            "action_gold_text_overlap": 0,
            "candidate_supported_gold_phrases": [],
        }

    gold_titles = {normalize_title(paper.title) for paper in pasa_gold_papers}
    exact_title_count = sum(
        normalize_title(candidate.paper.title) in gold_titles
        for candidate in candidates
    )
    query_terms = set(query_content_terms(query))
    action_terms = set(query_content_terms(action_text))
    query_title_overlap = 0
    action_title_overlap = 0
    action_text_overlap = 0
    for paper in pasa_gold_papers:
        title_terms = set(query_content_terms(paper.title))
        text_terms = set(query_content_terms(f"{paper.title} {paper.abstract or ''}"))
        query_title_overlap = max(query_title_overlap, len(query_terms & title_terms))
        action_title_overlap = max(action_title_overlap, len(action_terms & title_terms))
        action_text_overlap = max(action_text_overlap, len(action_terms & text_terms))
    phrase_evidence = _candidate_supported_gold_phrases(
        candidates,
        pasa_gold_papers,
        anchors=anchors,
        expansion_terms=expansion_terms,
    )
    if exact_title_count:
        category = "openalex_metadata_identity_gap"
    elif (
        phrase_evidence
        or action_title_overlap >= 2
        or action_text_overlap >= 3
        or query_title_overlap >= 2
    ):
        category = "action_construction_insufficient"
    else:
        category = "cross_vocabulary_mismatch"
    return {
        "category": category,
        "expected_gold_count": expected_gold_count,
        "pasa_gold_count": len(pasa_gold_papers),
        "pasa_gold_coverage_complete": len(pasa_gold_papers) == expected_gold_count,
        "exact_title_alias_candidate_count": exact_title_count,
        "query_gold_title_overlap": query_title_overlap,
        "action_gold_title_overlap": action_title_overlap,
        "action_gold_text_overlap": action_text_overlap,
        "candidate_supported_gold_phrases": phrase_evidence,
    }


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"JSONL row must be an object: {path}")
            rows.append(cast(dict[str, object], raw))
    return rows


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def run(args: argparse.Namespace) -> dict[str, object]:
    workspace_root = Path(args.workspace_root).resolve()
    validation_root = _resolve(workspace_root, args.validation_root)
    handoff_path = _resolve(workspace_root, args.handoff)
    partition_path = _resolve(workspace_root, args.partition)
    selection_path = _resolve(workspace_root, args.production_selection)
    pasa_index_path = _resolve(workspace_root, args.pasa_index)
    ab_path = validation_root / DEFAULT_AB_RESULT
    output_path = validation_root / DEFAULT_OUTPUT

    _manifest, partition_rows, actions = _verify_frozen_package(validation_root)
    ab = json.loads(ab_path.read_text(encoding="utf-8"))
    if not isinstance(ab, dict) or ab.get("query_count") != 128:
        raise ValueError("F5 candidate A/B result is invalid")
    per_query = ab.get("per_query")
    if not isinstance(per_query, list):
        raise ValueError("F5 candidate A/B rows are missing")
    miss_ids = tuple(
        str(row["query_id"])
        for row in per_query
        if isinstance(row, dict)
        and isinstance(row.get("augmented"), dict)
        and not cast(dict[str, object], row["augmented"]).get("gold_ranks")
    )
    if len(miss_ids) != 109 or len(set(miss_ids)) != 109:
        raise ValueError("remaining miss partition must contain exactly 109 queries")

    manifest_path, weights_path, _selection = _load_production_paths(selection_path)
    package = _online_only_package(
        load_training_package(
            handoff_path=handoff_path,
            partition_path=partition_path,
            production_bundle_path=weights_path,
        )
    )
    selected_package = replace(
        package,
        query_ids=miss_ids,
        rows_by_query_id={query_id: package.rows_by_query_id[query_id] for query_id in miss_ids},
    )
    baseline_paths = index_training_receipts(selected_package)
    supplemental_paths = _supplemental_receipts(
        validation_root,
        miss_ids,
        actions,
    )
    snapshots = _baseline_snapshot_rows(validation_root)
    diagnostics = {
        str(row["query_id"]): row
        for row in _load_jsonl(validation_root / "proposal-diagnostics.jsonl")
    }
    partition_by_id = {str(row["query_id"]): row for row in partition_rows}
    database = PasaPaperDatabase(pasa_index_path)
    all_gold_ids = [
        str(gold_id)
        for query_id in miss_ids
        for gold_id in cast(list[object], partition_by_id[query_id]["gold_paper_ids"])
    ]
    pasa_by_id = database.lookup_arxiv_many(all_gold_ids)
    supplemental_root = (validation_root / "receipts").resolve()

    rows: list[dict[str, object]] = []
    for query_id in miss_ids:
        baseline_query = build_document_ranking_query(
            selected_package,
            query_id,
            baseline_paths[query_id],
        )
        _verify_baseline_snapshot(
            query_id,
            baseline_query.candidates,
            snapshots[query_id],
        )
        augmented_query = build_document_ranking_query(
            selected_package,
            query_id,
            baseline_paths[query_id] + supplemental_paths[query_id],
            additive_receipt_roots=(supplemental_root,),
        )
        raw_gold_ids = [
            str(value)
            for value in cast(list[object], partition_by_id[query_id]["gold_paper_ids"])
        ]
        pasa_gold = [
            pasa_by_id[gold_id.casefold()]
            for gold_id in raw_gold_ids
            if gold_id.casefold() in pasa_by_id
        ]
        batch = RecallActionBatch.model_validate(actions[query_id])
        if len(batch.actions) != 1:
            raise ValueError(f"frozen bridge action count changed: {query_id}")
        diagnostic = diagnostics[query_id]
        attribution = attribute_query_miss(
            query=augmented_query.query,
            action_text=batch.actions[0].payload.query_text,
            candidates=augmented_query.candidates,
            pasa_gold_papers=pasa_gold,
            expected_gold_count=len(raw_gold_ids),
            anchors=cast(list[str], diagnostic["anchors"]),
            expansion_terms=cast(list[str], diagnostic["expansion_terms"]),
        )
        rows.append(
            {
                "query_id": query_id,
                "signal": diagnostic["signal"],
                **attribution,
            }
        )

    category_counts = Counter(str(row["category"]) for row in rows)
    signal_category_counts: dict[str, dict[str, int]] = {}
    for signal in sorted({str(row["signal"]) for row in rows}):
        signal_category_counts[signal] = dict(
            sorted(
                Counter(
                    str(row["category"])
                    for row in rows
                    if row["signal"] == signal
                ).items()
            )
        )
    phrase_rows = [
        row for row in rows if cast(list[object], row["candidate_supported_gold_phrases"])
    ]
    analyzable = sum(int(row["pasa_gold_count"]) > 0 for row in rows)
    phrase_ratio = len(phrase_rows) / analyzable if analyzable else 0.0
    phrase_gate_passed = (
        len(phrase_rows) >= _PHRASE_QUERY_MINIMUM
        and phrase_ratio >= _PHRASE_QUERY_RATIO_MINIMUM
    )
    result: dict[str, object] = {
        "schema_version": "pasa-analysis-only-openalex-miss-attribution-v1",
        "query_count": len(rows),
        "pasa_analyzable_query_count": analyzable,
        "pasa_complete_gold_coverage_query_count": sum(
            bool(row["pasa_gold_coverage_complete"]) for row in rows
        ),
        "category_counts": dict(sorted(category_counts.items())),
        "category_counts_by_signal": signal_category_counts,
        "candidate_supported_phrase_query_count": len(phrase_rows),
        "candidate_supported_phrase_query_ratio": phrase_ratio,
        "phrase_expansion_gate": {
            "minimum_query_count": _PHRASE_QUERY_MINIMUM,
            "minimum_analyzable_query_ratio": _PHRASE_QUERY_RATIO_MINIMUM,
            "passed": phrase_gate_passed,
        },
        "interpretation_limits": {
            "absence_from_retrieved_candidates_proves_openalex_absence": False,
            "metadata_identity_gap_requires_exact_title_candidate": True,
            "pasa_used_for_action_generation": False,
        },
        "inputs": {
            "f5_candidate_ab_sha256": _sha256(ab_path.read_bytes()),
            "validation_manifest_sha256": _sha256(
                (validation_root / "manifest.json").read_bytes()
            ),
            "partition_sha256": _sha256(
                (validation_root / "partition.jsonl").read_bytes()
            ),
            "actions_sha256": _sha256((validation_root / "actions.json").read_bytes()),
            "diagnostics_sha256": _sha256(
                (validation_root / "proposal-diagnostics.jsonl").read_bytes()
            ),
            "production_manifest_sha256": _sha256(manifest_path.read_bytes()),
            "production_weights_sha256": _sha256(weights_path.read_bytes()),
            "pasa_index_sha256": database.index_sha256,
        },
        "safety": {
            "analysis_only": True,
            "network_request_count": 0,
            "llm_request_count": 0,
            "test_partition_touched": False,
            "production_lock_modified": False,
            "training_started": False,
        },
        "per_query": rows,
    }
    payload = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_immutable(output_path, payload)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument(
        "--production-selection",
        type=Path,
        default=DEFAULT_SELECTION,
    )
    parser.add_argument("--pasa-index", type=Path, default=DEFAULT_PASA_INDEX)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_query"},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
