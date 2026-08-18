"""Deterministic blinded sampling for PaSa task-validity adjudication."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


_TOKEN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "for", "from", "in", "is", "of", "on",
        "paper", "papers", "study", "that", "the", "to", "what", "which",
        "with", "work", "works",
    }
)


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN.findall(value)
        if token.casefold() not in _STOPWORDS and len(token) > 2
    }


def _gold_id(value: object) -> str:
    return "arxiv:" + re.sub(r"v\d+$", "", str(value).casefold())


def build_task_validity_cases(
    *,
    raw_by_id: Mapping[str, Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    availability_by_gold_id: Mapping[str, str],
) -> list[dict[str, object]]:
    """Join frozen retrieval cohorts to raw PaSa query/Gold metadata."""

    cases: list[dict[str, object]] = []
    for evidence in evidence_rows:
        query_id = str(evidence["query_id"])
        if query_id not in raw_by_id:
            raise ValueError(f"raw PaSa row is missing for {query_id}")
        raw = raw_by_id[query_id]
        titles = [str(value) for value in raw["answer"]]
        gold_ids = [_gold_id(value) for value in raw["answer_arxiv_id"]]
        if len(titles) != len(gold_ids):
            raise ValueError(f"Gold title/identifier mismatch for {query_id}")
        query = str(raw["question"])
        query_tokens = _tokens(query)
        recalls = []
        for title in titles:
            title_tokens = _tokens(title)
            recalls.append(
                len(query_tokens & title_tokens) / len(title_tokens)
                if title_tokens
                else 0.0
            )
        cases.append(
            {
                "query_id": query_id,
                "role": str(evidence["role"]),
                "cohort": str(evidence["cohort"]),
                "fold": int(evidence["fold"]),
                "query": query,
                "gold_titles": titles,
                "gold_ids": gold_ids,
                "gold_availability": [
                    availability_by_gold_id.get(gold_id, "not_audited")
                    for gold_id in gold_ids
                ],
                "best_gold_title_token_recall": max(recalls, default=0.0),
            }
        )
    return cases


def _rank(seed: str, query_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{query_id}".encode("utf-8")).hexdigest()


def build_blind_review_packet(
    rows: Sequence[Mapping[str, Any]],
    *,
    targets: Mapping[str, Mapping[str, int]],
    seed: str,
) -> dict[str, object]:
    """Select exact role/cohort quotas while hiding retrieval outcome from review."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        query_id = str(row["query_id"])
        if query_id in seen_ids:
            raise ValueError(f"duplicate task-validity query: {query_id}")
        seen_ids.add(query_id)
        grouped[(str(row["role"]), str(row["cohort"]))].append(row)

    selected: list[Mapping[str, Any]] = []
    selection_counts: dict[str, dict[str, int]] = {}
    for role, cohort_targets in targets.items():
        selection_counts[role] = {}
        for cohort, count in cohort_targets.items():
            if type(count) is not int or count < 0:
                raise ValueError("task-validity target counts must be non-negative integers")
            candidates = sorted(
                grouped[(role, cohort)],
                key=lambda row: (_rank(seed, str(row["query_id"])), str(row["query_id"])),
            )
            if len(candidates) < count:
                raise ValueError(f"insufficient cases for {role}/{cohort}")
            selected.extend(candidates[:count])
            selection_counts[role][cohort] = count

    selected.sort(key=lambda row: (_rank(seed + ":shuffle", str(row["query_id"])), str(row["query_id"])))
    review_cases: list[dict[str, object]] = []
    private_key: list[dict[str, object]] = []
    for index, row in enumerate(selected, start=1):
        case_id = f"task-validity-{index:03d}"
        review_cases.append(
            {
                "case_id": case_id,
                "query": row["query"],
                "gold_titles": list(row["gold_titles"]),
                "gold_ids": list(row["gold_ids"]),
                "gold_availability": list(row["gold_availability"]),
                "best_gold_title_token_recall": row[
                    "best_gold_title_token_recall"
                ],
                "adjudication": {
                    "query_specificity": None,
                    "gold_relevance": None,
                    "gold_completeness": None,
                    "metadata_identity": None,
                    "document_evidence_needed": None,
                    "notes": None,
                },
            }
        )
        private_key.append(
            {
                "case_id": case_id,
                "query_id": row["query_id"],
                "role": row["role"],
                "cohort": row["cohort"],
                "fold": row["fold"],
            }
        )
    return {
        "schema_version": "pasa-task-validity-blind-review-v1",
        "seed": seed,
        "selection_counts": selection_counts,
        "review_cases": review_cases,
        "private_key": private_key,
        "test_partition_touched": False,
    }


def summarize_objective_task_validity_proxies(
    *,
    review_cases: Sequence[Mapping[str, Any]],
    private_key: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, int | float]]]:
    """Summarize label-risk proxies after a frozen blind packet is joined to its key."""

    cases = {str(row["case_id"]): row for row in review_cases}
    keys = {str(row["case_id"]): row for row in private_key}
    if len(cases) != len(review_cases) or len(keys) != len(private_key):
        raise ValueError("task-validity case ids must be unique")
    if set(cases) != set(keys):
        raise ValueError("review cases and private key do not match")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for case_id, key in keys.items():
        grouped[(str(key["role"]), str(key["cohort"]))].append(cases[case_id])

    output: dict[str, dict[str, dict[str, int | float]]] = {}
    for (role, cohort), rows in sorted(grouped.items()):
        overlaps = [float(row["best_gold_title_token_recall"]) for row in rows]
        gold_counts = [len(row["gold_titles"]) for row in rows]
        availability = [list(row["gold_availability"]) for row in rows]
        output.setdefault(role, {})[cohort] = {
            "query_count": len(rows),
            "zero_title_overlap_count": sum(value == 0 for value in overlaps),
            "zero_title_overlap_rate": sum(value == 0 for value in overlaps)
            / len(rows),
            "mean_best_gold_title_token_recall": sum(overlaps) / len(rows),
            "mean_gold_count": sum(gold_counts) / len(rows),
            "all_gold_available_count": sum(
                bool(statuses) and all(status == "available" for status in statuses)
                for statuses in availability
            ),
            "all_gold_available_rate": sum(
                bool(statuses) and all(status == "available" for status in statuses)
                for statuses in availability
            )
            / len(rows),
            "availability_not_audited_count": sum(
                "not_audited" in statuses for statuses in availability
            ),
        }
    return output


__all__ = [
    "build_blind_review_packet",
    "build_task_validity_cases",
    "summarize_objective_task_validity_proxies",
]
