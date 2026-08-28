from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "be",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "give",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "list",
    "me",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "provide",
    "reference",
    "references",
    "research",
    "show",
    "some",
    "studies",
    "study",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "used",
    "using",
    "was",
    "were",
    "what",
    "which",
    "who",
    "with",
    "work",
    "works",
    "you",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_RE.findall(text)
        if token.casefold() not in STOPWORDS and (len(token) > 2 or token.isdigit())
    }


def _gold_id(value: object) -> str:
    return "arxiv:" + re.sub(r"v\d+$", "", str(value).casefold())


def _alignment(question: str, titles: list[str]) -> dict[str, Any]:
    question_tokens = _tokens(question)
    title_tokens = [_tokens(title) for title in titles]
    recalls = [
        len(question_tokens & values) / len(values) if values else 0.0
        for values in title_tokens
    ]
    jaccards = [
        len(question_tokens & values) / len(question_tokens | values)
        if question_tokens | values
        else 0.0
        for values in title_tokens
    ]
    shared = question_tokens & set().union(*title_tokens)
    return {
        "question_tokens": sorted(question_tokens),
        "shared_gold_title_tokens": sorted(shared),
        "best_gold_title_token_recall": max(recalls, default=0.0),
        "mean_gold_title_token_recall": mean(recalls) if recalls else 0.0,
        "best_gold_title_jaccard": max(jaccards, default=0.0),
    }


def _cohort(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(rows),
        "mean_best_gold_title_token_recall": mean(
            row["best_gold_title_token_recall"] for row in rows
        ),
        "median_best_gold_title_token_recall": median(
            row["best_gold_title_token_recall"] for row in rows
        ),
        "mean_best_gold_title_jaccard": mean(
            row["best_gold_title_jaccard"] for row in rows
        ),
        "zero_title_overlap_query_count": sum(
            row["best_gold_title_token_recall"] == 0 for row in rows
        ),
    }


def _strata(
    rows: list[dict[str, Any]], field: str, common_misses: set[str]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        name: {
            "population_query_count": len(values),
            "retrievable_query_count": sum(
                bool(row["any_gold_available"]) for row in values
            ),
            "common_miss_query_count": sum(
                row["query_id"] in common_misses for row in values
            ),
            "common_miss_rate": sum(
                row["query_id"] in common_misses for row in values
            )
            / sum(bool(row["any_gold_available"]) for row in values),
        }
        for name, values in sorted(groups.items())
    }


def _hit_queries(rows: list[dict[str, Any]]) -> set[str]:
    return {row["query_id"] for row in rows if row["gold_hit_count"] > 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-audit", type=Path, required=True)
    parser.add_argument("--raw-dev", type=Path, required=True)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--exact-cache", type=Path, required=True)
    parser.add_argument("--base-labels", type=Path, required=True)
    parser.add_argument("--semantic-labels", type=Path, required=True)
    parser.add_argument("--graph-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failure_audit = json.loads(args.failure_audit.read_text(encoding="utf-8"))
    raw_dev = {row["qid"]: row for row in _read_jsonl(args.raw_dev)}
    raw_train = {row["qid"]: row for row in _read_jsonl(args.raw_train)}
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    frozen_rows = freeze["sample"]
    frozen_ids = {row["query_id"] for row in frozen_rows}
    availability = {
        row["gold_id"]: row["status"]
        for row in json.loads(args.exact_cache.read_text(encoding="utf-8"))["records"]
    }
    base_labels = _read_jsonl(args.base_labels)
    semantic_labels = _read_jsonl(args.semantic_labels)
    graph_labels = _read_jsonl(args.graph_labels)

    if {row["query_id"] for row in graph_labels} != frozen_ids:
        raise ValueError("graph labels do not match the frozen query set")

    base_hits = _hit_queries(base_labels)
    semantic_backfill_hits = _hit_queries(
        [
            row
            for row in semantic_labels
            if row["action"]["action_id"] == "semantic-backfill-original"
        ]
    )
    graph_hits = {
        row["query_id"] for row in graph_labels if row["graph_gold_hit_ids"]
    }
    any_family_hits = base_hits | semantic_backfill_hits | graph_hits
    common_misses = frozen_ids - any_family_hits

    large_rows: list[dict[str, Any]] = []
    all_available_ids: set[str] = set()
    any_available_ids: set[str] = set()
    for frozen in frozen_rows:
        query_id = frozen["query_id"]
        raw = raw_train[query_id]
        statuses = [
            availability.get(_gold_id(value), "missing")
            for value in raw["answer_arxiv_id"]
        ]
        if "missing" in statuses:
            raise ValueError(f"availability cache is incomplete for {query_id}")
        if statuses and all(status == "available" for status in statuses):
            all_available_ids.add(query_id)
        if "available" in statuses:
            any_available_ids.add(query_id)
        large_rows.append(
            {
                **frozen,
                **_alignment(raw["question"], raw["answer"]),
                "all_gold_available": statuses
                and all(status == "available" for status in statuses),
                "any_gold_available": "available" in statuses,
            }
        )

    eight_rows: list[dict[str, Any]] = []
    for audit in failure_audit["query_audits"]:
        if audit.get("candidate_failure_cause") != "search_actions_no_hit":
            continue
        raw = raw_dev[audit["query_id"]]
        alignment = _alignment(raw["question"], raw["answer"])
        subquery_tokens = set().union(*(_tokens(value) for value in audit["subqueries"]))
        question_tokens = set(alignment["question_tokens"])
        shared_tokens = set(alignment["shared_gold_title_tokens"])
        eight_rows.append(
            {
                "query_id": audit["query_id"],
                "analyzer_source": audit["analyzer_source"],
                "gold_count": len(raw["answer"]),
                "available_gold_count": sum(
                    value["status"] == "available"
                    for value in audit["gold_availability"].values()
                ),
                **{
                    key: value
                    for key, value in alignment.items()
                    if key not in {"question_tokens"}
                },
                "original_content_token_retention": (
                    len(question_tokens & subquery_tokens) / len(question_tokens)
                    if question_tokens
                    else 0.0
                ),
                "shared_gold_token_retention": (
                    len(shared_tokens & subquery_tokens) / len(shared_tokens)
                    if shared_tokens
                    else None
                ),
                "subquery_novel_token_ratio": (
                    len(subquery_tokens - question_tokens) / len(subquery_tokens)
                    if subquery_tokens
                    else 0.0
                ),
            }
        )

    eligible_common_misses = common_misses & any_available_ids
    missed_rows = [
        row for row in large_rows if row["query_id"] in eligible_common_misses
    ]
    hit_rows = [
        row
        for row in large_rows
        if row["query_id"] in any_family_hits & any_available_ids
    ]
    missed_alignment = _cohort(missed_rows)
    hit_alignment = _cohort(hit_rows)
    missed_zero_rate = (
        missed_alignment["zero_title_overlap_query_count"] / len(missed_rows)
    )
    hit_zero_rate = hit_alignment["zero_title_overlap_query_count"] / len(hit_rows)
    observed_folds = {str(row["fold"]) for row in missed_rows}
    observed_intents = {row["intent_family"] for row in missed_rows}
    observed_lengths = {row["length_bucket"] for row in missed_rows}
    observed_gold_buckets = {str(row["gold_count_bucket"]) for row in missed_rows}
    systemic_gate = (
        len(missed_rows) >= 20
        and len(observed_folds) >= 2
        and len(observed_intents) >= 2
        and len(observed_lengths) >= 2
        and len(observed_gold_buckets) >= 2
    )

    payload = {
        "schema_version": "candidate-generation-generality-audit-v1",
        "test_partition_touched": False,
        "policy_changed": False,
        "evidence_roles": {
            "discovery": "8 independent auto_dev six-action no-hit queries",
            "generality_validation": "frozen stratified 385-query auto_train audit",
        },
        "precommitted_generality_gate": {
            "minimum_independent_queries": 20,
            "minimum_folds": 2,
            "minimum_intent_families": 2,
            "minimum_length_buckets": 2,
            "minimum_gold_count_buckets": 2,
            "specific_change_requires_independent_intervention": True,
        },
        "discovery_eight": {
            "query_count": len(eight_rows),
            "queries": eight_rows,
            "aggregate": {
                "mean_original_content_token_retention": mean(
                    row["original_content_token_retention"] for row in eight_rows
                ),
                "mean_subquery_novel_token_ratio": mean(
                    row["subquery_novel_token_ratio"] for row in eight_rows
                ),
                "shared_gold_token_retention_observed_count": sum(
                    row["shared_gold_token_retention"] is not None
                    for row in eight_rows
                ),
            },
        },
        "frozen_385": {
            "query_count": len(frozen_ids),
            "all_gold_available_query_count": len(all_available_ids),
            "any_gold_available_query_count": len(any_available_ids),
            "base_hit_query_count": len(base_hits),
            "semantic_backfill_hit_query_count": len(semantic_backfill_hits),
            "graph_hit_query_count": len(graph_hits),
            "any_family_hit_query_count": len(any_family_hits),
            "common_miss_query_count": len(common_misses),
            "retrievable_common_miss_query_count": len(eligible_common_misses),
            "common_miss_all_gold_available_query_count": len(
                common_misses & all_available_ids
            ),
            "semantic_new_over_base_query_count": len(
                semantic_backfill_hits - base_hits
            ),
            "graph_new_over_base_semantic_query_count": len(
                graph_hits - (base_hits | semantic_backfill_hits)
            ),
            "alignment_comparison": {
                "retrievable_common_miss": missed_alignment,
                "any_family_hit": hit_alignment,
                "observed_association_not_causal": {
                    "mean_best_title_recall_gap_miss_minus_hit": (
                        missed_alignment["mean_best_gold_title_token_recall"]
                        - hit_alignment["mean_best_gold_title_token_recall"]
                    ),
                    "zero_title_overlap_rate_common_miss": missed_zero_rate,
                    "zero_title_overlap_rate_any_family_hit": hit_zero_rate,
                    "zero_title_overlap_risk_ratio": (
                        missed_zero_rate / hit_zero_rate
                    ),
                },
            },
            "common_miss_strata": {
                field: _strata(large_rows, field, eligible_common_misses)
                for field in (
                    "fold",
                    "intent_family",
                    "length_bucket",
                    "gold_count_bucket",
                )
            },
        },
        "decisions": {
            "systemic_candidate_generation_bottleneck": {
                "gate_passed": systemic_gate,
                "independent_query_count": len(missed_rows),
                "fold_count": len(observed_folds),
                "intent_family_count": len(observed_intents),
                "length_bucket_count": len(observed_lengths),
                "gold_count_bucket_count": len(observed_gold_buckets),
            },
            "add_entity_constraints": {
                "gate_passed": False,
                "reason": (
                    "No frozen independent intervention estimates the causal gain or "
                    "regression risk of entity constraints."
                ),
            },
            "modify_production_policy": {
                "gate_passed": False,
                "reason": "This audit performs attribution only; no intervention was run.",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "discovery_query_count": len(eight_rows),
                "retrievable_common_miss_query_count": len(eligible_common_misses),
                "systemic_gate_passed": systemic_gate,
                "entity_constraint_gate_passed": False,
                "test_partition_touched": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
