"""Leakage-safe PASA candidate mixing and deterministic context freezing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from paper_search.domain.models import Paper
from paper_search.evaluation.predictions import (
    paper_evaluation_aliases,
    paper_matches_evaluation_ids,
)
from paper_search.learning.query_constraint_profile import QueryConstraintProfile
from paper_search.retrieval.pasa_paper_database import (
    mark_pasa_training_gold_injected,
)


_LOCAL_LABELS = frozenset({"method", "dataset", "task", "year", "negation"})


def _merge_local_task_row(
    row: Mapping[str, Any], *, profile: QueryConstraintProfile
) -> dict[str, Any]:
    output = dict(row)
    if output.get("tasks") or "task" not in profile.labels or not profile.tasks:
        return output
    confidence = max(0.8, float(profile.confidence))
    output["tasks"] = [
        {
            "normalized_value": value,
            "confidence": confidence,
            "evidence_span": value,
            "strength": "must",
        }
        for value in profile.tasks
    ]
    output["ambiguous_fields"] = []
    output["annotator_id"] = "unified-local-fusion-context-v3"
    output["task_label_status"] = "runtime_deterministic"
    return output


def build_mixed_pasa_candidates(
    *,
    lexical_papers: Sequence[Paper],
    gold_paper_ids: Sequence[str],
    gold_lookup: Mapping[str, Paper],
) -> tuple[list[Paper], dict[str, int | bool]]:
    """Mix lexical negatives and PASA Gold under one source action."""

    gold_ids = list(dict.fromkeys(gold_paper_ids))
    papers: list[Paper] = []
    aliases: set[str] = set()
    for paper in lexical_papers:
        paper_aliases = paper_evaluation_aliases(paper)
        if aliases.intersection(paper_aliases):
            continue
        papers.append(paper)
        aliases.update(paper_aliases)
    lexical_count = len(papers)
    lexical_gold_count = sum(
        paper_matches_evaluation_ids(paper, gold_ids) for paper in papers
    )
    direct_count = 0
    for gold_id in gold_ids:
        paper = gold_lookup.get(gold_id)
        if paper is None:
            continue
        paper_aliases = paper_evaluation_aliases(paper)
        if aliases.intersection(paper_aliases):
            continue
        papers.append(mark_pasa_training_gold_injected(paper))
        aliases.update(paper_aliases)
        direct_count += 1
    positive_count = sum(
        paper_matches_evaluation_ids(paper, gold_ids) for paper in papers
    )
    lexical_negative_count = lexical_count - lexical_gold_count
    if positive_count == 0 or lexical_negative_count == 0:
        raise ValueError(
            "mixed PASA candidates require both Gold and lexical negatives"
        )
    return papers, {
        "lexical_candidate_count": lexical_count,
        "lexical_gold_candidate_count": lexical_gold_count,
        "lexical_negative_candidate_count": lexical_negative_count,
        "direct_gold_candidate_count": direct_count,
        "positive_candidate_count": positive_count,
        "supplement_candidate_count": len(papers),
        "mixed_positive_negative": True,
    }


def merge_local_constraint_row(
    row: Mapping[str, Any],
    *,
    profile: QueryConstraintProfile,
) -> dict[str, Any]:
    """Freeze production-local structured constraints into one training row."""

    output = dict(row)
    labels = set(str(value) for value in output.get("labels", []))
    sources = dict(output.get("label_sources", {}))
    confidence = dict(output.get("label_confidence", {}))
    evidence = dict(output.get("evidence", {}))
    local_labels = set(profile.labels).intersection(_LOCAL_LABELS)
    entity_values = {
        "method": list(profile.methods),
        "dataset": list(profile.datasets),
        "task": list(profile.tasks),
    }
    entity_fields = {"method": "methods", "dataset": "datasets", "task": "tasks"}
    for label, values in entity_values.items():
        field = entity_fields[label]
        if sources.get(label) == "local_deterministic":
            output[field] = values
            if values and label in local_labels:
                labels.add(label)
                confidence[label] = max(0.8, float(profile.confidence))
                evidence[label] = values
            else:
                labels.discard(label)
                sources.pop(label, None)
                confidence.pop(label, None)
                evidence.pop(label, None)
            continue
        if label not in local_labels or not values:
            continue
        current = list(output.get(field, []))
        output[field] = list(dict.fromkeys([*current, *values]))
        was_confirmed = label in labels
        labels.add(label)
        if not was_confirmed:
            sources[label] = "local_deterministic"
            confidence[label] = max(0.8, float(profile.confidence))
            evidence[label] = values
    if sources.get("year") in {"rule", "local_deterministic"}:
        output["year_from"] = profile.year_from
        output["year_to"] = profile.year_to
        if "year" in local_labels:
            labels.add("year")
            confidence["year"] = 1.0
            evidence["year"] = [
                str(value)
                for value in (profile.year_from, profile.year_to)
                if value is not None
            ]
        else:
            labels.discard("year")
            sources.pop("year", None)
            confidence.pop("year", None)
            evidence.pop("year", None)
    elif "year" in local_labels and not (
        output.get("year_from") is not None or output.get("year_to") is not None
    ):
        output["year_from"] = profile.year_from
        output["year_to"] = profile.year_to
        labels.add("year")
        sources["year"] = "local_deterministic"
        confidence["year"] = 1.0
        evidence["year"] = [
            str(value)
            for value in (profile.year_from, profile.year_to)
            if value is not None
        ]
    if "negation" in local_labels and profile.exclusions:
        current_exclusions = list(output.get("exclusions", []))
        output["exclusions"] = list(
            dict.fromkeys([*current_exclusions, *profile.exclusions])
        )
        was_confirmed = "negation" in labels
        labels.add("negation")
        if not was_confirmed:
            sources["negation"] = "local_deterministic"
            confidence["negation"] = 1.0
            evidence["negation"] = list(profile.exclusions)
    output["labels"] = sorted(labels)
    output["label_sources"] = sources
    output["label_confidence"] = confidence
    output["evidence"] = evidence
    output["status"] = "accepted" if labels else "partial"
    return output


def build_unified_context_freeze_rows(
    *,
    strict_ready_query_ids: Sequence[str],
    partition_rows: Mapping[str, Mapping[str, Any]],
    task_rows_by_query: Mapping[str, Mapping[str, Any]],
    constraint_rows_by_query: Mapping[str, Mapping[str, Any]],
    local_profiles_by_query: Mapping[str, QueryConstraintProfile],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Freeze one deterministic context row for every strict-ready query."""

    strict_ids = list(dict.fromkeys(strict_ready_query_ids))
    if len(strict_ids) != len(strict_ready_query_ids):
        raise ValueError("strict-ready context scope contains duplicate query ids")
    required = set(strict_ids)
    sources = {
        "partition": set(partition_rows),
        "task labels": set(task_rows_by_query),
        "constraint labels": set(constraint_rows_by_query),
        "local profiles": set(local_profiles_by_query),
    }
    for label, available in sources.items():
        missing = required - available
        if missing:
            raise ValueError(f"strict-ready context is missing {len(missing)} {label}")
    frozen_tasks: list[dict[str, Any]] = []
    frozen_constraints: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    for query_id in sorted(strict_ids):
        partition = partition_rows[query_id]
        if partition.get("role") != "training" or partition.get("split") != "auto_train":
            raise ValueError("unified context freeze permits auto_train rows only")
        task = _merge_local_task_row(
            task_rows_by_query[query_id],
            profile=local_profiles_by_query[query_id],
        )
        constraint = merge_local_constraint_row(
            constraint_rows_by_query[query_id],
            profile=local_profiles_by_query[query_id],
        )
        frozen_tasks.append(task)
        frozen_constraints.append(constraint)
        label_counts.update(str(value) for value in constraint.get("labels", []))
    return frozen_tasks, frozen_constraints, {
        "query_count": len(strict_ids),
        "label_query_count": dict(sorted(label_counts.items())),
        "role": "training",
        "split": "auto_train",
        "test_partition_touched": False,
    }


def validate_priority_queue_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    strict_ready_query_ids: set[str],
    expected_count: int,
) -> list[dict[str, Any]]:
    """Validate the sealed dataset/task PASA augmentation scope."""

    validated = [dict(row) for row in rows]
    query_ids = [str(row.get("query_id", "")) for row in validated]
    if len(validated) != expected_count:
        raise ValueError(
            f"priority queue count mismatch: expected {expected_count}, got {len(validated)}"
        )
    if len(query_ids) != len(set(query_ids)) or any(not query_id for query_id in query_ids):
        raise ValueError("priority queue query ids must be non-empty and unique")
    if not set(query_ids).issubset(strict_ready_query_ids):
        raise ValueError("priority queue contains queries outside strict-ready scope")
    for row in validated:
        if row.get("recommended_action") != "pasa_mixed_lexical_gold_supplement":
            raise ValueError("priority queue contains an unsupported candidate action")
        signals = set(str(value) for value in row.get("eligible_signals", []))
        if not signals.intersection({"dataset", "task_provenance"}):
            raise ValueError(
                "priority queue requires dataset or task_provenance eligibility"
            )
    return sorted(validated, key=lambda row: str(row["query_id"]))


__all__ = [
    "build_mixed_pasa_candidates",
    "build_unified_context_freeze_rows",
    "merge_local_constraint_row",
    "validate_priority_queue_rows",
]
