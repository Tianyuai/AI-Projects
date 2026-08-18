"""Build method-level labels from saved structured graph canary receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from paper_search.evaluation.dataset import write_frozen_bytes
from paper_search.learning.contracts import PolicyActionCandidate
from paper_search.learning.data_isolation import DatasetRole
from paper_search.learning.graph_method_labels import (
    GraphMethodLabel,
    build_graph_method_label,
)
from paper_search.learning.method_route_labels import (
    MethodRouteLabel,
    semantic_method_labels,
)
from paper_search.learning.provider_action_dataset import (
    _load_partition,
    _restore_redacted_usage,
    _successful_or_failed_receipt,
)
from paper_search.learning.provider_action_labels import (
    ProviderActionObservation,
    build_provider_action_labels,
)
from paper_search.recall_experiments.contracts import (
    CitationExpandAction,
    RecallActionBatch,
    RetrievalActionResult,
    TextSearchAction,
    TitleSearchAction,
)


def _text(action: TextSearchAction | TitleSearchAction) -> str:
    if isinstance(action, TextSearchAction):
        return action.payload.query_text
    return action.payload.title_text


def build_structured_method_labels(
    *,
    dataset: str,
    split: str,
    role: DatasetRole,
    query_id: str,
    query: str,
    gold_paper_ids: list[str],
    actions: RecallActionBatch,
    results: list[RetrievalActionResult],
) -> tuple[MethodRouteLabel, GraphMethodLabel]:
    result_by_id = {result.action_id: result for result in results}
    if len(result_by_id) != len(results):
        raise ValueError(f"duplicate action result for {query_id}")
    if set(result_by_id) != {action.action_id for action in actions.actions}:
        raise ValueError(f"action/result mismatch for {query_id}")
    text_actions = [
        action
        for action in actions.actions
        if isinstance(action, (TextSearchAction, TitleSearchAction))
    ]
    graph_actions = [
        action
        for action in actions.actions
        if isinstance(action, CitationExpandAction)
    ]
    anchor_actions = [
        action
        for action in text_actions
        if action.strategy == "structured:anchor-original"
    ]
    if len(anchor_actions) != 1:
        raise ValueError(f"structured receipt requires one lexical anchor: {query_id}")
    observations: list[ProviderActionObservation] = []
    for action in text_actions:
        result = result_by_id[action.action_id]
        search_mode = (
            action.payload.search_mode
            if isinstance(action, TextSearchAction)
            else "lexical"
        )
        observations.append(
            ProviderActionObservation(
                provider="openalex",
                action=PolicyActionCandidate(
                    action_id=action.action_id,
                    action_type=action.action_type,
                    text=_text(action),
                    origin=(
                        "original_query"
                        if action.strategy == "structured:anchor-original"
                        else "deterministic_rule"
                    ),
                    provider_hint="openalex",
                    search_mode=search_mode,
                ),
                hits=result.hits,
                usage=result.usage,
                errors=result.errors,
                infrastructure_failure=result.infrastructure_failure,
            )
        )
    provider_labels = build_provider_action_labels(
        dataset=dataset,
        split=split,
        role=role,
        query_id=query_id,
        query=query,
        gold_paper_ids=gold_paper_ids,
        observations=observations,
    )
    semantic_labels = semantic_method_labels(provider_labels)
    if len(semantic_labels) != 1:
        raise ValueError(f"structured receipt requires one semantic action: {query_id}")
    semantic_action = next(
        action
        for action in text_actions
        if isinstance(action, TextSearchAction)
        and action.payload.search_mode == "semantic"
    )
    semantic_result = result_by_id[semantic_action.action_id]
    semantic = semantic_labels[0].model_copy(
        update={"search_api_calls": semantic_result.usage.search_api_calls}
    )

    anchor_result = result_by_id[anchor_actions[0].action_id]
    graph_results = [result_by_id[action.action_id] for action in graph_actions]
    graph = build_graph_method_label(
        dataset=dataset,
        split=split,
        role=role,
        query_id=query_id,
        query=query,
        gold_paper_ids=gold_paper_ids,
        anchor_hits=anchor_result.hits,
        pre_graph_hits=[
            paper
            for action in text_actions
            for paper in result_by_id[action.action_id].hits
        ],
        graph_hits=[paper for result in graph_results for paper in result.hits],
        seed_count=len(
            {action.payload.seed_canonical_id for action in graph_actions}
        ),
        graph_action_count=len(graph_actions),
        graph_infrastructure_failure=any(
            result.infrastructure_failure for result in graph_results
        ),
        search_api_calls=sum(
            result.usage.search_api_calls for result in graph_results
        ),
    )
    return semantic, graph


def load_structured_method_labels_from_canary_runs(
    *,
    partition_path: Path,
    provider_run_root: Path,
) -> tuple[list[MethodRouteLabel], list[GraphMethodLabel]]:
    partition = _load_partition(partition_path)
    semantic_labels: list[MethodRouteLabel] = []
    graph_labels: list[GraphMethodLabel] = []
    seen: set[str] = set()
    for report_path in sorted(provider_run_root.glob("*/canary-report.json")):
        batch = report_path.parent
        report = json.loads(report_path.read_text(encoding="utf-8"))
        actions_by_query = report.get("actions_by_query")
        if not isinstance(actions_by_query, dict):
            raise ValueError(f"invalid action map in {report_path}")
        for query_id, raw_actions in actions_by_query.items():
            if query_id in seen:
                raise ValueError(f"duplicate structured receipt: {query_id}")
            seen.add(query_id)
            row = partition.get(query_id)
            if row is None:
                raise ValueError(f"canary query is absent from partition: {query_id}")
            actions = RecallActionBatch.model_validate({"actions": raw_actions})
            receipt = _successful_or_failed_receipt(batch, query_id)
            raw_results = receipt.get("results")
            if not isinstance(raw_results, list):
                raise ValueError(f"retrieval receipt has no results: {query_id}")
            results = [
                RetrievalActionResult.model_validate(_restore_redacted_usage(item))
                for item in raw_results
            ]
            semantic, graph = build_structured_method_labels(
                dataset=cast(str, row["dataset"]),
                split=cast(str, row["split"]),
                role=cast(DatasetRole, row["role"]),
                query_id=query_id,
                query=cast(str, row["query"]),
                gold_paper_ids=cast(list[str], row["gold_paper_ids"]),
                actions=actions,
                results=results,
            )
            semantic_labels.append(semantic)
            graph_labels.append(graph)
    if not semantic_labels:
        raise ValueError("no structured method labels were loaded")
    return semantic_labels, graph_labels


def freeze_method_labels(rows: list[Any], path: Path) -> str:
    if not rows:
        raise ValueError("method labels are empty")
    ordered = sorted(rows, key=lambda row: row.query_id)
    content = "".join(
        json.dumps(
            row.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in ordered
    ).encode("utf-8")
    write_frozen_bytes(path, content)
    return "sha256:" + hashlib.sha256(content).hexdigest()


__all__ = [
    "build_structured_method_labels",
    "freeze_method_labels",
    "load_structured_method_labels_from_canary_runs",
]
