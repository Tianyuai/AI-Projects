"""Build a Gold-blind PASA/OpenAlex alias map for one sealed receipt replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from paper_search.domain.models import Paper
from paper_search.evaluation.conservative_identity_aliases import (
    MAX_PUBLICATION_YEAR_DISTANCE,
    MIN_ABSTRACT_TOKEN_JACCARD,
    MIN_SHARED_ABSTRACT_TOKENS,
    MIN_TITLE_TOKEN_COUNT,
    build_conservative_pasa_identifier_aliases,
)
from paper_search.learning.large_scale_fusion_training import (
    build_document_ranking_query,
    index_training_receipts,
    load_training_package,
)
from paper_search.retrieval.pasa_paper_database import PasaPaperDatabase

try:
    from scripts.attribute_cross_vocabulary_misses_with_pasa import DEFAULT_PASA_INDEX
    from scripts.evaluate_cross_vocabulary_f5_topk import (
        DEFAULT_HANDOFF,
        DEFAULT_PARTITION,
        DEFAULT_SELECTION,
        DEFAULT_VALIDATION_ROOT,
        _baseline_snapshot_rows,
        _load_production_paths,
        _online_only_package,
        _resolve,
        _sha256,
        _supplemental_receipts,
        _verify_baseline_snapshot,
        _verify_frozen_package,
        _write_immutable,
        audit_candidate_pool_monotonicity,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from attribute_cross_vocabulary_misses_with_pasa import DEFAULT_PASA_INDEX
    from evaluate_cross_vocabulary_f5_topk import (
        DEFAULT_HANDOFF,
        DEFAULT_PARTITION,
        DEFAULT_SELECTION,
        DEFAULT_VALIDATION_ROOT,
        _baseline_snapshot_rows,
        _load_production_paths,
        _online_only_package,
        _resolve,
        _sha256,
        _supplemental_receipts,
        _verify_baseline_snapshot,
        _verify_frozen_package,
        _write_immutable,
        audit_candidate_pool_monotonicity,
    )


DEFAULT_MAP_NAME = "conservative-pasa-identity-alias-map-v1.json"
DEFAULT_EVIDENCE_NAME = "conservative-pasa-identity-alias-evidence-v1.json"


def run(args: argparse.Namespace) -> dict[str, object]:
    workspace_root = Path(args.workspace_root).resolve()
    validation_root = _resolve(workspace_root, args.validation_root)
    handoff_path = _resolve(workspace_root, args.handoff)
    partition_path = _resolve(workspace_root, args.partition)
    selection_path = _resolve(workspace_root, args.production_selection)
    pasa_index_path = _resolve(workspace_root, args.pasa_index)
    output_map = (
        _resolve(workspace_root, args.output_map)
        if args.output_map is not None
        else validation_root / DEFAULT_MAP_NAME
    )
    output_evidence = (
        _resolve(workspace_root, args.output_evidence)
        if args.output_evidence is not None
        else validation_root / DEFAULT_EVIDENCE_NAME
    )

    validation_manifest, partition_rows, actions = _verify_frozen_package(
        validation_root
    )
    query_ids = tuple(str(row["query_id"]) for row in partition_rows)
    if len(query_ids) != 128:
        raise ValueError("identity replay requires the sealed 128-query partition")
    manifest_path, weights_path, selection = _load_production_paths(selection_path)
    inputs = cast(dict[str, object], validation_manifest.get("inputs"))
    for label, path in (
        ("handoff_sha256", handoff_path),
        ("partition_sha256", partition_path),
        ("production_bundle_sha256", weights_path),
    ):
        if inputs.get(label) != _sha256(path.read_bytes()):
            raise ValueError(f"frozen validation input hash mismatch: {label}")

    package = _online_only_package(
        load_training_package(
            handoff_path=handoff_path,
            partition_path=partition_path,
            production_bundle_path=weights_path,
        )
    )
    selected_package = replace(
        package,
        query_ids=query_ids,
        rows_by_query_id={query_id: package.rows_by_query_id[query_id] for query_id in query_ids},
    )
    baseline_paths = index_training_receipts(selected_package)
    supplemental_paths = _supplemental_receipts(validation_root, query_ids, actions)
    baseline_snapshots = _baseline_snapshot_rows(validation_root)
    supplemental_root = (validation_root / "receipts").resolve()

    candidates: list[Paper] = []
    augmented_candidate_count = 0
    for query_id in query_ids:
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
        )
        audit_candidate_pool_monotonicity(
            baseline_query.candidates,
            augmented_query.candidates,
        )
        augmented_candidate_count += len(augmented_query.candidates)
        candidates.extend(candidate.paper for candidate in augmented_query.candidates)

    database = PasaPaperDatabase(pasa_index_path)
    references_by_title = database.lookup_normalized_titles(
        [candidate.title for candidate in candidates]
    )
    aliases, alias_evidence, decision_counts = (
        build_conservative_pasa_identifier_aliases(
            candidates,
            references_by_title,
        )
    )
    map_payload = (
        json.dumps(aliases, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_immutable(output_map, map_payload)

    result: dict[str, object] = {
        "schema_version": "conservative-pasa-openalex-identity-alias-evidence-v1",
        "query_count": len(query_ids),
        "augmented_candidate_occurrence_count": augmented_candidate_count,
        "matched_normalized_title_count": len(references_by_title),
        "accepted_identity_record_count": len(alias_evidence),
        "identifier_alias_count": len(aliases),
        "decision_counts": decision_counts,
        "policy": {
            "normalized_title_must_be_exact": True,
            "pasa_normalized_title_must_be_unique": True,
            "minimum_title_token_count": MIN_TITLE_TOKEN_COUNT,
            "minimum_abstract_token_jaccard": MIN_ABSTRACT_TOKEN_JACCARD,
            "minimum_shared_abstract_token_count": MIN_SHARED_ABSTRACT_TOKENS,
            "maximum_publication_year_distance": MAX_PUBLICATION_YEAR_DISTANCE,
            "title_only_alias_allowed": False,
        },
        "inputs": {
            "frozen_validation_manifest_sha256": _sha256(
                (validation_root / "manifest.json").read_bytes()
            ),
            "training_handoff_sha256": _sha256(handoff_path.read_bytes()),
            "training_partition_sha256": _sha256(partition_path.read_bytes()),
            "production_selection_sha256": _sha256(selection_path.read_bytes()),
            "production_manifest_sha256": _sha256(manifest_path.read_bytes()),
            "production_weights_sha256": _sha256(weights_path.read_bytes()),
            "pasa_index_sha256": database.index_sha256,
            "identifier_map_sha256": _sha256(map_payload),
            "production_default": selection["production_default"],
        },
        "safety": {
            "query_gold_associations_used_for_alias_derivation": False,
            "gold_identifier_membership_used_for_alias_derivation": False,
            "online_requests_made": 0,
            "llm_requests_made": 0,
            "training_started": False,
            "test_partition_touched": False,
            "production_lock_modified": False,
        },
        "evidence": alias_evidence,
    }
    evidence_payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_immutable(output_evidence, evidence_payload)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION)
    parser.add_argument("--production-selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--pasa-index", type=Path, default=DEFAULT_PASA_INDEX)
    parser.add_argument("--output-map", type=Path)
    parser.add_argument("--output-evidence", type=Path)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "query_count",
                    "augmented_candidate_occurrence_count",
                    "matched_normalized_title_count",
                    "accepted_identity_record_count",
                    "identifier_alias_count",
                    "decision_counts",
                    "safety",
                )
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
