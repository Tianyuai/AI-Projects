from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from paper_search.evaluation.predictions import paper_evaluation_id
from paper_search.processing.deduplicate import deduplicate_papers
from audit_integrated_dev_failures import (
    _analysis_and_routes,
    _capture_manifests,
    _read_jsonl,
)


def _pool(routes: list[list[Any]]) -> tuple[list[str], int]:
    raw = [paper for route in routes for paper in route]
    deduplicated = deduplicate_papers(raw).papers
    identifiers = list(dict.fromkeys(paper_evaluation_id(paper) for paper in deduplicated))
    return identifiers, len(raw)


def _aggregate(rows: list[dict[str, Any]], pool_name: str) -> dict[str, Any]:
    recalls = [row[pool_name]["oracle_recall"] for row in rows]
    sizes = [row[pool_name]["candidate_count"] for row in rows]
    duplicate_rates = [row[pool_name]["duplicate_rate"] for row in rows]
    return {
        "query_count": len(rows),
        "gold_hit_query_count": sum(value > 0 for value in recalls),
        "oracle_macro_recall": mean(recalls),
        "mean_candidate_count": mean(sizes),
        "min_candidate_count": min(sizes),
        "max_candidate_count": max(sizes),
        "mean_duplicate_rate": mean(duplicate_rates),
    }


def _cohort_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pools = {
        name: _aggregate(rows, name) for name in ("lexical", "semantic", "hybrid")
    }
    best_single_recall = max(
        pools["lexical"]["oracle_macro_recall"],
        pools["semantic"]["oracle_macro_recall"],
    )
    return {
        **pools,
        "marginal": {
            "mean_cross_mode_jaccard": mean(
                row["cross_mode_jaccard"] for row in rows
            ),
            "mean_lexical_only_count": mean(
                row["lexical_only_count"] for row in rows
            ),
            "mean_semantic_only_count": mean(
                row["semantic_only_count"] for row in rows
            ),
            "mean_subquery_unique_counts": [
                mean(row["subquery_unique_counts"][index] for row in rows)
                for index in range(3)
            ],
            "hybrid_oracle_lift_over_best_single": (
                pools["hybrid"]["oracle_macro_recall"] - best_single_recall
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--failure-audit", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--captured-after", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    audit = json.loads(args.failure_audit.read_text(encoding="utf-8"))
    failure_ids = {
        item["query_id"]
        for item in audit["query_audits"]
        if item.get("candidate_failure_cause") == "search_actions_no_hit"
    }
    executions = result["executions"]
    partition = {row["query_id"]: row for row in _read_jsonl(args.partition)}
    manifests = _capture_manifests(
        args.capture_root,
        captured_after=datetime.fromisoformat(args.captured_after.replace("Z", "+00:00")),
        expected=len(executions),
    )

    comparisons: list[dict[str, Any]] = []
    for execution, manifest_path in zip(executions, manifests, strict=True):
        query_id = execution["query_id"]
        row = partition[query_id]
        _, routes = _analysis_and_routes(manifest_path)
        gold_ids = {str(value).casefold() for value in row["gold_paper_ids"]}
        route_groups = {
            "lexical": routes[0::2],
            "semantic": routes[1::2],
            "hybrid": routes,
        }
        pools: dict[str, dict[str, Any]] = {}
        pool_sets: dict[str, set[str]] = {}
        for name, selected_routes in route_groups.items():
            identifiers, raw_count = _pool(selected_routes)
            identifier_set = set(identifiers)
            pool_sets[name] = identifier_set
            hits = identifier_set.intersection(gold_ids)
            pools[name] = {
                "candidate_count": len(identifiers),
                "raw_result_count": raw_count,
                "duplicate_rate": 1.0 - len(identifiers) / raw_count,
                "gold_hit_count": len(hits),
                "oracle_recall": len(hits) / len(gold_ids),
            }
        overlap = pool_sets["lexical"].intersection(pool_sets["semantic"])
        union = pool_sets["hybrid"]
        subquery_sets = [
            set(_pool(routes[index : index + 2])[0]) for index in range(0, 6, 2)
        ]
        comparisons.append(
            {
                "query_id": query_id,
                "cohort": "six_action_no_hit" if query_id in failure_ids else "control",
                **pools,
                "cross_mode_overlap_count": len(overlap),
                "cross_mode_jaccard": len(overlap) / len(union),
                "lexical_only_count": len(pool_sets["lexical"] - pool_sets["semantic"]),
                "semantic_only_count": len(pool_sets["semantic"] - pool_sets["lexical"]),
                "subquery_unique_counts": [
                    len(
                        values
                        - set().union(
                            *(
                                other
                                for other_index, other in enumerate(subquery_sets)
                                if other_index != index
                            )
                        )
                    )
                    for index, values in enumerate(subquery_sets)
                ],
            }
        )

    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        cohorts[row["cohort"]].append(row)
    payload = {
        "schema_version": "failed-candidate-pool-comparison-v1",
        "sample_manifest_sha256": result["sample_manifest_sha256"],
        "test_partition_touched": False,
        "groups": ["lexical", "semantic", "hybrid"],
        "cohorts": {
            cohort: _cohort_summary(rows)
            for cohort, rows in sorted(cohorts.items())
        },
        "queries": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "queries"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
