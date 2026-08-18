"""Strict training-only receipt loading for supplemental document ranking."""

from __future__ import annotations

import json
from collections.abc import Sequence, Set
from pathlib import Path
from typing import Any

from paper_search.domain.models import Paper
from paper_search.learning.cpu_document_ranker import (
    DocumentRankingQuery,
    build_production_document_candidates,
)


_A_PRIME_LEXICAL_ACTIONS = (
    "ceiling-candidate-anchor",
    "ceiling-candidate-text-1",
    "ceiling-candidate-text-2",
    "ceiling-candidate-text-3",
)
_A_PRIME_BOOLEAN_ACTION = "ceiling-candidate-boolean-relaxed"
_SEMANTIC_ORIGINAL_ACTION = "semantic-backfill-original"


def _read_training_partition(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("role") != "training" or row.get("split") != "auto_train":
            raise ValueError("supplemental document ranking requires auto_train rows")
        query_id = str(row["query_id"])
        if query_id in rows:
            raise ValueError("supplemental partition query ids must be unique")
        rows[query_id] = row
    if not rows:
        raise ValueError("supplemental auto_train partition is empty")
    return rows


def _successful_receipts(
    roots: Sequence[Path],
) -> dict[str, dict[str, Any]]:
    if not roots:
        raise ValueError("supplemental receipt roots must not be empty")
    selected: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"supplemental receipt root is unavailable: {root}")
        for path in sorted(root.rglob("retrieval/attempt-01/*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("attempt_status") == "succeeded":
                query_id = str(payload["query_id"])
                if not isinstance(payload.get("results"), list):
                    raise ValueError(f"invalid retrieval receipt for {query_id}")
                selected[query_id] = payload
    return selected


def load_supplemental_document_ranking_queries(
    *,
    partition_path: Path,
    lexical_receipt_roots: Sequence[Path],
    semantic_receipt_roots: Sequence[Path],
    excluded_query_ids: Set[str],
    expected_query_count: int,
    maximum_lexical_action_count: int | None = None,
) -> list[DocumentRankingQuery]:
    """Pair lexical and semantic receipts without admitting dev/test rows."""

    if expected_query_count <= 0:
        raise ValueError("expected supplemental query count must be positive")
    partition = _read_training_partition(partition_path)
    lexical = _successful_receipts(lexical_receipt_roots)
    semantic = _successful_receipts(semantic_receipt_roots)
    selected_ids = sorted(
        (set(partition) & set(lexical) & set(semantic)) - set(excluded_query_ids)
    )
    if len(selected_ids) != expected_query_count:
        raise ValueError(
            "supplemental paired query count mismatch: "
            f"expected {expected_query_count}, got {len(selected_ids)}"
        )

    output: list[DocumentRankingQuery] = []
    for query_id in selected_ids:
        lexical_results = lexical[query_id]["results"]
        semantic_results = semantic[query_id]["results"]
        if not lexical_results or (
            maximum_lexical_action_count is not None
            and len(lexical_results) > maximum_lexical_action_count
        ):
            raise ValueError(f"unexpected lexical action count for {query_id}")
        semantic_original = [
            result
            for result in semantic_results
            if result.get("action_id") == _SEMANTIC_ORIGINAL_ACTION
        ]
        if len(semantic_original) != 1:
            raise ValueError(f"missing unique semantic-original for {query_id}")
        action_results = [
            (
                str(result["action_id"]),
                [Paper.model_validate(hit) for hit in result["hits"]],
            )
            for result in [*lexical_results, semantic_original[0]]
        ]
        row = partition[query_id]
        query = str(row["query"])
        output.append(
            DocumentRankingQuery(
                query_id=query_id,
                query=query,
                gold_paper_ids=list(row["gold_paper_ids"]),
                candidates=build_production_document_candidates(
                    query, action_results
                ),
            )
        )
    return output


def load_a_prime_folded_document_ranking_queries(
    *,
    freeze_path: Path,
    partition_path: Path,
    lexical_receipt_root: Path,
    semantic_receipt_root: Path,
) -> list[tuple[int, DocumentRankingQuery]]:
    """Load the frozen A-prime target domain with its preassigned folds."""

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("split") != "auto_train":
        raise ValueError("A-prime document ranking requires auto_train freeze")
    partition = _read_training_partition(partition_path)
    lexical = _successful_receipts([lexical_receipt_root])
    semantic = _successful_receipts([semantic_receipt_root])
    output: list[tuple[int, DocumentRankingQuery]] = []
    seen: set[str] = set()
    for selected in freeze.get("sample", []):
        query_id = str(selected["query_id"])
        if query_id in seen:
            raise ValueError("A-prime freeze query ids must be unique")
        seen.add(query_id)
        if query_id not in partition or query_id not in lexical or query_id not in semantic:
            raise ValueError(f"missing frozen A-prime input for {query_id}")
        fold = int(selected["fold"])
        if fold not in {1, 2, 3}:
            raise ValueError("A-prime freeze folds must be 1, 2, or 3")
        lexical_by_id = {
            str(result["action_id"]): result
            for result in lexical[query_id]["results"]
        }
        semantic_by_id = {
            str(result["action_id"]): result
            for result in semantic[query_id]["results"]
        }
        if _SEMANTIC_ORIGINAL_ACTION not in semantic_by_id:
            raise ValueError(f"missing semantic-original for {query_id}")
        action_results = [
            (
                action_id,
                [Paper.model_validate(hit) for hit in lexical_by_id[action_id]["hits"]],
            )
            for action_id in _A_PRIME_LEXICAL_ACTIONS
            if action_id in lexical_by_id
        ]
        action_results.append(
            (
                _SEMANTIC_ORIGINAL_ACTION,
                [
                    Paper.model_validate(hit)
                    for hit in semantic_by_id[_SEMANTIC_ORIGINAL_ACTION]["hits"]
                ],
            )
        )
        if _A_PRIME_BOOLEAN_ACTION in lexical_by_id:
            action_results.append(
                (
                    _A_PRIME_BOOLEAN_ACTION,
                    [
                        Paper.model_validate(hit)
                        for hit in lexical_by_id[_A_PRIME_BOOLEAN_ACTION]["hits"]
                    ],
                )
            )
        row = partition[query_id]
        query = str(row["query"])
        output.append(
            (
                fold,
                DocumentRankingQuery(
                    query_id=query_id,
                    query=query,
                    gold_paper_ids=list(row["gold_paper_ids"]),
                    candidates=build_production_document_candidates(
                        query, action_results
                    ),
                ),
            )
        )
    if not output:
        raise ValueError("A-prime freeze sample is empty")
    return output


def load_folded_document_ranking_evaluation_queries(
    *,
    manifest_path: Path,
    partition_path: Path,
    receipt_root: Path,
) -> list[tuple[int, DocumentRankingQuery]]:
    """Load frozen auto_dev receipts for evaluation only."""

    partition: dict[str, dict[str, Any]] = {}
    for line in partition_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("role") != "development" or row.get("split") != "auto_dev":
            raise ValueError("document ranking evaluation requires auto_dev rows")
        query_id = str(row["query_id"])
        if query_id in partition:
            raise ValueError("evaluation partition query ids must be unique")
        partition[query_id] = row
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipts = _successful_receipts([receipt_root])
    output: list[tuple[int, DocumentRankingQuery]] = []
    seen: set[str] = set()
    for selected in manifest.get("sample", []):
        query_id = str(selected["query_id"])
        if query_id in seen:
            raise ValueError("evaluation manifest query ids must be unique")
        seen.add(query_id)
        if query_id not in partition or query_id not in receipts:
            raise ValueError(f"missing frozen evaluation input for {query_id}")
        fold = int(selected["fold"])
        if fold not in {1, 2, 3}:
            raise ValueError("evaluation folds must be 1, 2, or 3")
        action_results = [
            (
                str(result["action_id"]),
                [Paper.model_validate(hit) for hit in result["hits"]],
            )
            for result in receipts[query_id]["results"]
        ]
        row = partition[query_id]
        query = str(row["query"])
        output.append(
            (
                fold,
                DocumentRankingQuery(
                    query_id=query_id,
                    query=query,
                    gold_paper_ids=list(row["gold_paper_ids"]),
                    candidates=build_production_document_candidates(
                        query, action_results
                    ),
                ),
            )
        )
    if not output:
        raise ValueError("evaluation manifest sample is empty")
    return output


__all__ = [
    "load_a_prime_folded_document_ranking_queries",
    "load_folded_document_ranking_evaluation_queries",
    "load_supplemental_document_ranking_queries",
]
