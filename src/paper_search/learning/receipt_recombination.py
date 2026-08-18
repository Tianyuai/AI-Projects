"""Counterfactual recombination of frozen action-level retrieval receipts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from paper_search.learning.graph_method_labels import GraphMethodLabel
from paper_search.learning.provider_action_labels import ProviderActionLabel


SourceFamily = Literal["core_lexical", "boolean_phrase", "prf", "semantic"]


def _family(action_id: str) -> SourceFamily:
    if action_id == "semantic-backfill-original":
        return "semantic"
    if "boolean-relaxed" in action_id or "phrase-proximity" in action_id:
        return "boolean_phrase"
    if "candidate-prf-" in action_id:
        return "prf"
    return "core_lexical"


def _union(rows: Sequence[ProviderActionLabel]) -> set[str]:
    return set().union(*(set(row.gold_hit_ids) for row in rows)) if rows else set()


def _metrics(
    query_ids: Sequence[str],
    *,
    hits: Mapping[str, set[str]],
    action_counts: Mapping[str, int],
    gold_counts: Mapping[str, int],
) -> dict[str, int | float]:
    action_count = sum(action_counts[query_id] for query_id in query_ids)
    gold_hit_count = sum(len(hits[query_id]) for query_id in query_ids)
    return {
        "query_count": len(query_ids),
        "hit_query_count": sum(bool(hits[query_id]) for query_id in query_ids),
        "gold_hit_count": gold_hit_count,
        "action_count": action_count,
        "macro_recall": sum(
            len(hits[query_id]) / gold_counts[query_id] for query_id in query_ids
        )
        / len(query_ids),
        "gold_hits_per_action": (
            gold_hit_count / action_count if action_count else 0.0
        ),
    }


def _summary(
    query_ids: Sequence[str],
    *,
    fold_by_query: Mapping[str, int],
    hits: Mapping[str, set[str]],
    action_counts: Mapping[str, int],
    gold_counts: Mapping[str, int],
) -> tuple[dict[str, int | float], dict[str, dict[str, int | float]]]:
    overall = _metrics(
        query_ids,
        hits=hits,
        action_counts=action_counts,
        gold_counts=gold_counts,
    )
    folds = {
        str(fold): _metrics(
            [query_id for query_id in query_ids if fold_by_query[query_id] == fold],
            hits=hits,
            action_counts=action_counts,
            gold_counts=gold_counts,
        )
        for fold in (1, 2, 3)
    }
    return overall, folds


def _positive_fold_count(
    query_ids: Sequence[str],
    *,
    fold_by_query: Mapping[str, int],
    before: Mapping[str, set[str]],
    after: Mapping[str, set[str]],
    gold_counts: Mapping[str, int],
) -> int:
    return sum(
        sum(
            (len(after[query_id]) - len(before[query_id]))
            / gold_counts[query_id]
            for query_id in query_ids
            if fold_by_query[query_id] == fold
        )
        > 0
        for fold in (1, 2, 3)
    )


def analyze_receipt_recombination(
    *,
    frozen_rows: Sequence[Mapping[str, Any]],
    lexical_labels: Sequence[Mapping[str, Any]],
    semantic_labels: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Measure top-k lexical and candidate-source marginal recall by frozen fold."""

    fold_by_query = {
        str(row["query_id"]): int(row["fold"]) for row in frozen_rows
    }
    if not fold_by_query or set(fold_by_query.values()) != {1, 2, 3}:
        raise ValueError("frozen rows must contain unique queries across three folds")
    if len(fold_by_query) != len(frozen_rows):
        raise ValueError("frozen query IDs must be unique")
    expected = set(fold_by_query)

    lexical_grouped: dict[str, list[ProviderActionLabel]] = defaultdict(list)
    for raw in lexical_labels:
        row = ProviderActionLabel.model_validate(raw)
        if row.action.search_mode == "lexical":
            lexical_grouped[row.query_id].append(row)
    if set(lexical_grouped) != expected:
        raise ValueError("lexical labels do not match the frozen query set")

    semantic_grouped: dict[str, list[ProviderActionLabel]] = defaultdict(list)
    for raw in semantic_labels:
        row = ProviderActionLabel.model_validate(raw)
        if row.action.action_id == "semantic-backfill-original":
            semantic_grouped[row.query_id].append(row)
    if set(semantic_grouped) != expected or any(
        len(rows) != 1 for rows in semantic_grouped.values()
    ):
        raise ValueError("semantic labels do not match the frozen query set")

    query_ids = sorted(expected)
    gold_counts: dict[str, int] = {}
    for query_id in query_ids:
        counts = {
            row.gold_association_count
            for row in [*lexical_grouped[query_id], *semantic_grouped[query_id]]
        }
        if len(counts) != 1 or None in counts:
            raise ValueError(f"inconsistent Gold counts for query: {query_id}")
        gold_count = next(iter(counts))
        if gold_count is None:
            raise ValueError(f"missing Gold count for query: {query_id}")
        gold_counts[query_id] = gold_count

    semantic_hits = {
        query_id: _union(semantic_grouped[query_id]) for query_id in query_ids
    }
    maximum_k = max(len(lexical_grouped[query_id]) for query_id in query_ids)
    lexical_top_k: list[dict[str, object]] = []
    lexical_top_k_plus_semantic: list[dict[str, object]] = []
    previous_hits: dict[str, set[str]] = {
        query_id: set() for query_id in query_ids
    }
    for top_k in range(1, maximum_k + 1):
        topk_selected = {
            query_id: lexical_grouped[query_id][:top_k] for query_id in query_ids
        }
        hits = {
            query_id: _union(topk_selected[query_id]) for query_id in query_ids
        }
        action_counts = {
            query_id: len(topk_selected[query_id]) for query_id in query_ids
        }
        overall, folds = _summary(
            query_ids,
            fold_by_query=fold_by_query,
            hits=hits,
            action_counts=action_counts,
            gold_counts=gold_counts,
        )
        lexical_top_k.append(
            {
                "top_k": top_k,
                "overall": overall,
                "folds": folds,
                "marginal_new_gold_hit_count": sum(
                    len(hits[query_id] - previous_hits[query_id])
                    for query_id in query_ids
                ),
            }
        )
        combined = {
            query_id: hits[query_id] | semantic_hits[query_id]
            for query_id in query_ids
        }
        combined_counts = {
            query_id: action_counts[query_id] + 1 for query_id in query_ids
        }
        combined_overall, combined_folds = _summary(
            query_ids,
            fold_by_query=fold_by_query,
            hits=combined,
            action_counts=combined_counts,
            gold_counts=gold_counts,
        )
        lexical_top_k_plus_semantic.append(
            {
                "top_k": top_k,
                "overall": combined_overall,
                "folds": combined_folds,
                "semantic_new_gold_hit_count": sum(
                    len(combined[query_id] - hits[query_id])
                    for query_id in query_ids
                ),
            }
        )
        previous_hits = hits

    family_rows: dict[str, dict[SourceFamily, list[ProviderActionLabel]]] = {}
    for query_id in query_ids:
        grouped: dict[SourceFamily, list[ProviderActionLabel]] = {
            "core_lexical": [],
            "boolean_phrase": [],
            "prf": [],
            "semantic": semantic_grouped[query_id],
        }
        for row in lexical_grouped[query_id]:
            grouped[_family(row.action.action_id)].append(row)
        family_rows[query_id] = grouped

    core_hits = {
        query_id: _union(family_rows[query_id]["core_lexical"])
        for query_id in query_ids
    }
    core_plus_semantic = {
        query_id: core_hits[query_id] | semantic_hits[query_id]
        for query_id in query_ids
    }
    candidate_sources: dict[str, dict[str, int | float | dict[str, float]]] = {}
    for family in ("semantic", "boolean_phrase", "prf"):
        source_hits = {
            query_id: _union(family_rows[query_id][family])
            for query_id in query_ids
        }
        over_core = {
            query_id: core_hits[query_id] | source_hits[query_id]
            for query_id in query_ids
        }
        over_core_semantic = {
            query_id: core_plus_semantic[query_id] | source_hits[query_id]
            for query_id in query_ids
        }
        fold_deltas = {
            str(fold): sum(
                len(source_hits[query_id] - core_hits[query_id])
                / gold_counts[query_id]
                for query_id in query_ids
                if fold_by_query[query_id] == fold
            )
            / sum(fold_by_query[query_id] == fold for query_id in query_ids)
            for fold in (1, 2, 3)
        }
        candidate_sources[family] = {
            "eligible_query_count": sum(
                bool(family_rows[query_id][family]) for query_id in query_ids
            ),
            "action_count": sum(
                len(family_rows[query_id][family]) for query_id in query_ids
            ),
            "standalone_gold_hit_count": sum(
                len(source_hits[query_id]) for query_id in query_ids
            ),
            "new_gold_over_core_count": sum(
                len(over_core[query_id] - core_hits[query_id])
                for query_id in query_ids
            ),
            "new_gold_over_core_plus_semantic_count": sum(
                len(over_core_semantic[query_id] - core_plus_semantic[query_id])
                for query_id in query_ids
            ),
            "beneficial_query_count_over_core": sum(
                bool(source_hits[query_id] - core_hits[query_id])
                for query_id in query_ids
            ),
            "positive_fold_count_over_core": _positive_fold_count(
                query_ids,
                fold_by_query=fold_by_query,
                before=core_hits,
                after=over_core,
                gold_counts=gold_counts,
            ),
            "macro_recall_delta_over_core_by_fold": fold_deltas,
        }

    composition_specs = {
        "receipt_prefix5_semantic": ("prefix", 5, 0, False),
        "core5_semantic": ("core", 5, 0, False),
        "core4_semantic": ("core", 4, 0, False),
        "core4_semantic_boolean": ("core", 4, 1, False),
        "core3_semantic_boolean_phrase": ("core", 3, 2, False),
        "core4_semantic_prf": ("core", 4, 0, True),
        "core3_semantic_boolean_prf": ("core", 3, 1, True),
    }
    budget_six_compositions: dict[str, dict[str, object]] = {}
    for name, (backbone, backbone_limit, boolean_limit, include_prf) in (
        composition_specs.items()
    ):
        composition_selected: dict[str, list[ProviderActionLabel]] = {}
        for query_id in query_ids:
            if backbone == "prefix":
                rows = lexical_grouped[query_id][:backbone_limit]
            else:
                rows = family_rows[query_id]["core_lexical"][:backbone_limit]
            rows = [
                *rows,
                *semantic_grouped[query_id],
                *family_rows[query_id]["boolean_phrase"][:boolean_limit],
            ]
            if include_prf:
                rows.extend(family_rows[query_id]["prf"][:1])
            if len(rows) > 6:
                raise ValueError(f"composition exceeds six actions: {name}")
            composition_selected[query_id] = rows
        hits = {
            query_id: _union(composition_selected[query_id])
            for query_id in query_ids
        }
        action_counts = {
            query_id: len(composition_selected[query_id]) for query_id in query_ids
        }
        overall, folds = _summary(
            query_ids,
            fold_by_query=fold_by_query,
            hits=hits,
            action_counts=action_counts,
            gold_counts=gold_counts,
        )
        budget_six_compositions[name] = {
            "overall": overall,
            "folds": folds,
            "maximum_action_count": max(action_counts.values()),
            "unused_budget_query_count": sum(
                count < 6 for count in action_counts.values()
            ),
        }

    return {
        "schema_version": "receipt-recombination-audit-v1",
        "query_count": len(query_ids),
        "maximum_lexical_actions_per_query": maximum_k,
        "lexical_top_k": lexical_top_k,
        "lexical_top_k_plus_semantic": lexical_top_k_plus_semantic,
        "candidate_sources": candidate_sources,
        "budget_six_compositions": budget_six_compositions,
        "test_partition_touched": False,
    }


def analyze_structured_graph_marginals(
    *,
    frozen_rows: Sequence[Mapping[str, Any]],
    lexical_labels: Sequence[Mapping[str, Any]],
    semantic_labels: Sequence[Mapping[str, Any]],
    graph_labels: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Attribute structured and graph gains after rich lexical plus semantic."""

    fold_by_query = {
        str(row["query_id"]): int(row["fold"]) for row in frozen_rows
    }
    expected = set(fold_by_query)
    if not expected or len(expected) != len(frozen_rows):
        raise ValueError("frozen query IDs must be non-empty and unique")

    lexical_grouped: dict[str, list[ProviderActionLabel]] = defaultdict(list)
    for raw in lexical_labels:
        row = ProviderActionLabel.model_validate(raw)
        if row.action.search_mode == "lexical":
            lexical_grouped[row.query_id].append(row)
    semantic_grouped: dict[str, list[ProviderActionLabel]] = defaultdict(list)
    for raw in semantic_labels:
        row = ProviderActionLabel.model_validate(raw)
        if row.action.action_id == "semantic-backfill-original":
            semantic_grouped[row.query_id].append(row)
    graph_by_query = {
        row.query_id: row
        for row in (GraphMethodLabel.model_validate(raw) for raw in graph_labels)
    }
    if set(lexical_grouped) != expected:
        raise ValueError("lexical labels do not match the frozen query set")
    if set(semantic_grouped) != expected or any(
        len(rows) != 1 for rows in semantic_grouped.values()
    ):
        raise ValueError("semantic labels do not match the frozen query set")
    if set(graph_by_query) != expected:
        raise ValueError("graph labels do not match the frozen query set")

    query_ids = sorted(expected)
    gold_counts: dict[str, int] = {}
    for query_id in query_ids:
        counts = {
            row.gold_association_count
            for row in [*lexical_grouped[query_id], *semantic_grouped[query_id]]
        }
        counts.add(graph_by_query[query_id].gold_association_count)
        if len(counts) != 1 or None in counts:
            raise ValueError(f"inconsistent Gold counts for query: {query_id}")
        gold_count = next(iter(counts))
        if gold_count is None:
            raise ValueError(f"missing Gold count for query: {query_id}")
        gold_counts[query_id] = gold_count

    base = {
        query_id: _union(
            [*lexical_grouped[query_id], *semantic_grouped[query_id]]
        )
        for query_id in query_ids
    }

    def stage(
        *,
        before: Mapping[str, set[str]],
        additions: Mapping[str, set[str]],
        action_counts: Mapping[str, int],
    ) -> tuple[dict[str, int | float | dict[str, float]], dict[str, set[str]]]:
        after = {
            query_id: before[query_id] | additions[query_id]
            for query_id in query_ids
        }
        new_hits = {
            query_id: after[query_id] - before[query_id] for query_id in query_ids
        }
        action_count = sum(action_counts.values())
        new_hit_count = sum(len(hits) for hits in new_hits.values())
        fold_deltas = {
            str(fold): sum(
                len(new_hits[query_id]) / gold_counts[query_id]
                for query_id in query_ids
                if fold_by_query[query_id] == fold
            )
            / sum(fold_by_query[query_id] == fold for query_id in query_ids)
            for fold in (1, 2, 3)
        }
        return (
            {
                "beneficial_query_count": sum(bool(hits) for hits in new_hits.values()),
                "new_gold_hit_count": new_hit_count,
                "incremental_action_count": action_count,
                "marginal_gold_hits_per_action": (
                    new_hit_count / action_count if action_count else 0.0
                ),
                "macro_recall_delta": sum(fold_deltas.values()) / 3,
                "positive_fold_count": sum(value > 0 for value in fold_deltas.values()),
                "macro_recall_delta_by_fold": fold_deltas,
            },
            after,
        )

    structured_additions = {
        query_id: set(graph_by_query[query_id].pre_graph_gold_hit_ids)
        for query_id in query_ids
    }
    structured_action_counts = {
        query_id: graph_by_query[query_id].search_api_calls
        - graph_by_query[query_id].graph_action_count
        for query_id in query_ids
    }
    structured, after_structured = stage(
        before=base,
        additions=structured_additions,
        action_counts=structured_action_counts,
    )
    graph, _ = stage(
        before=after_structured,
        additions={
            query_id: set(graph_by_query[query_id].graph_gold_hit_ids)
            for query_id in query_ids
        },
        action_counts={
            query_id: graph_by_query[query_id].graph_action_count
            for query_id in query_ids
        },
    )
    return {
        "schema_version": "structured-graph-sequence-audit-v1",
        "query_count": len(query_ids),
        "structured": structured,
        "graph": graph,
        "test_partition_touched": False,
    }


__all__ = [
    "analyze_receipt_recombination",
    "analyze_structured_graph_marginals",
]
