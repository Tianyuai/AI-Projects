"""Offline integrity audit for the fixed-budget OpenAlex action composition."""

from __future__ import annotations

import asyncio
import json
import unicodedata
from pathlib import Path
from typing import Any

from paper_search.domain.models import QuerySpec
from paper_search.learning.candidate_ceiling import (
    Core4SemanticBooleanQueryGenerator,
)
from paper_search.learning.candidates import query_content_terms
from paper_search.learning.structured_graph_candidates import (
    FixedBudgetOpenAlexQueryGenerator,
)
from paper_search.recall_experiments.contracts import RecallGenerationContext


ActionIdentity = tuple[str, str, str]


def _canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _identity(action: dict[str, Any]) -> ActionIdentity | None:
    action_type = action.get("action_type")
    payload = action.get("payload")
    if action_type not in {"text_search", "title_search"} or not isinstance(
        payload, dict
    ):
        return None
    if action_type == "text_search":
        text = payload.get("query_text")
        mode = payload.get("search_mode", "lexical")
    else:
        text = payload.get("title_text")
        mode = "lexical"
    if not isinstance(text, str) or mode not in {"lexical", "semantic"}:
        raise ValueError("saved search action has an invalid identity")
    return action_type, mode, _canonical_text(text)


def _load_actions(root: Path) -> dict[str, list[dict[str, Any]]]:
    actions_by_query: dict[str, list[dict[str, Any]]] = {}
    for report_path in sorted(root.rglob("canary-report.json")):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        raw_map = payload.get("actions_by_query")
        if not isinstance(raw_map, dict):
            raise ValueError(f"invalid action map: {report_path}")
        for query_id, raw_actions in raw_map.items():
            if isinstance(raw_actions, dict):
                raw_actions = [raw_actions]
            if not isinstance(query_id, str) or not isinstance(raw_actions, list):
                raise ValueError(f"invalid saved actions: {report_path}")
            actions = [
                action for action in raw_actions if isinstance(action, dict)
            ]
            if len(actions) != len(raw_actions):
                raise ValueError(f"invalid saved action item: {report_path}")
            previous = actions_by_query.get(query_id)
            if previous is not None and previous != actions:
                raise ValueError(f"conflicting retry actions for {query_id}")
            actions_by_query[query_id] = actions
    if not actions_by_query:
        raise ValueError(f"no saved action receipts under {root}")
    return actions_by_query


def _original_query(actions: list[dict[str, Any]], query_id: str) -> str:
    anchors = [
        action
        for action in actions
        if action.get("strategy") == "structured:anchor-original"
    ]
    if len(anchors) != 1:
        raise ValueError(f"structured receipt requires one original anchor: {query_id}")
    payload = anchors[0].get("payload")
    query = payload.get("query_text") if isinstance(payload, dict) else None
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"structured receipt has no original query: {query_id}")
    return query


def audit_fixed_budget_receipts(
    *,
    structured_root: Path,
    semantic_root: Path,
) -> dict[str, object]:
    """Rebuild scheme A and verify identities against frozen saved receipts."""

    structured = _load_actions(structured_root)
    semantic = _load_actions(semantic_root)
    if set(structured) != set(semantic):
        raise ValueError("structured and semantic receipts cover different query IDs")

    generator = FixedBudgetOpenAlexQueryGenerator(max_openalex_actions=6)
    maximum_action_count = 0
    minimum_action_count = 6
    duplicate_action_count = 0
    anchor_failure_count = 0
    graph_action_count = 0
    compatible_query_count = 0
    unused_budget_query_count = 0

    for query_id in sorted(structured):
        query = _original_query(structured[query_id], query_id)
        context = RecallGenerationContext(
            query_id=query_id,
            original_query=query,
            query_spec=QuerySpec(original_query=query, research_goal=query),
        )
        generation = asyncio.run(generator.generate(context))
        dumped = [
            action.model_dump(mode="json")
            for action in generation.action_batch.actions
        ]
        identities = [identity for action in dumped if (identity := _identity(action))]
        maximum_action_count = max(maximum_action_count, len(dumped))
        minimum_action_count = min(minimum_action_count, len(dumped))
        duplicate_action_count += len(identities) - len(set(identities))
        graph_action_count += sum(
            action.get("action_type") == "citation_expand" for action in dumped
        )
        original = _canonical_text(query)
        expected_lexical = ("text_search", "lexical", original)
        expected_semantic = ("text_search", "semantic", original)
        if identities.count(expected_lexical) != 1 or identities.count(
            expected_semantic
        ) != 1:
            anchor_failure_count += 1
        if len(dumped) < 6:
            unused_budget_query_count += 1

        saved_identities = {
            identity
            for action in [*structured[query_id], *semantic[query_id]]
            if (identity := _identity(action)) is not None
        }
        if set(identities).issubset(saved_identities):
            compatible_query_count += 1

    query_count = len(structured)
    passed = (
        maximum_action_count <= 6
        and duplicate_action_count == 0
        and anchor_failure_count == 0
        and graph_action_count == 0
        and compatible_query_count == query_count
    )
    return {
        "schema_version": "fixed-budget-openalex-offline-integrity-v1",
        "query_count": query_count,
        "minimum_action_count": minimum_action_count,
        "maximum_action_count": maximum_action_count,
        "unused_budget_query_count": unused_budget_query_count,
        "duplicate_action_count": duplicate_action_count,
        "anchor_failure_count": anchor_failure_count,
        "graph_action_count": graph_action_count,
        "receipt_compatible_query_count": compatible_query_count,
        "test_partition_touched": False,
        "passed": passed,
    }


def audit_core4_semantic_boolean_queries(
    rows: list[dict[str, Any]],
) -> dict[str, object]:
    """Generate A-prime actions offline and verify its frozen method contract."""

    if not rows:
        raise ValueError("A-prime integrity audit requires queries")
    generator = Core4SemanticBooleanQueryGenerator()
    maximum_action_count = 0
    minimum_action_count = 6
    unused_budget_query_count = 0
    duplicate_action_count = 0
    anchor_failure_count = 0
    semantic_original_failure_count = 0
    forbidden_action_count = 0
    composition_failure_count = 0
    seen_query_ids: set[str] = set()
    for row in rows:
        query_id = str(row["query_id"])
        query = str(row["query"])
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate A-prime audit query: {query_id}")
        seen_query_ids.add(query_id)
        context = RecallGenerationContext(
            query_id=query_id,
            original_query=query,
            query_spec=QuerySpec(original_query=query, research_goal=query),
        )
        generation = asyncio.run(generator.generate(context))
        dumped = [
            action.model_dump(mode="json")
            for action in generation.action_batch.actions
        ]
        identities = [identity for action in dumped if (identity := _identity(action))]
        maximum_action_count = max(maximum_action_count, len(dumped))
        minimum_action_count = min(minimum_action_count, len(dumped))
        unused_budget_query_count += len(dumped) < 6
        duplicate_action_count += len(identities) - len(set(identities))
        original = _canonical_text(query)
        lexical = ("text_search", "lexical", original)
        semantic = ("text_search", "semantic", original)
        anchor_failure_count += identities.count(lexical) != 1
        semantic_original_failure_count += identities.count(semantic) != 1
        core_count = sum(
            action.get("strategy") == "candidate-family:baseline"
            for action in dumped
        )
        semantic_count = sum(
            action.get("action_id") == "ceiling-candidate-semantic-original"
            for action in dumped
        )
        boolean_count = sum(
            action.get("action_id") == "ceiling-candidate-boolean-relaxed"
            for action in dumped
        )
        expected_boolean_count = int(len(query_content_terms(query)) >= 4)
        recognized_count = core_count + semantic_count + boolean_count
        composition_failure_count += not (
            1 <= core_count <= 4
            and semantic_count == 1
            and boolean_count == expected_boolean_count
            and recognized_count == len(dumped)
        )
        forbidden_action_count += sum(
            action.get("action_type") == "citation_expand"
            or "title-target" in str(action.get("action_id", ""))
            or "prf" in str(action.get("action_id", ""))
            for action in dumped
        )
    passed = (
        maximum_action_count <= 6
        and duplicate_action_count == 0
        and anchor_failure_count == 0
        and semantic_original_failure_count == 0
        and composition_failure_count == 0
        and forbidden_action_count == 0
    )
    return {
        "schema_version": "core4-semantic-boolean-offline-integrity-v1",
        "query_count": len(rows),
        "minimum_action_count": minimum_action_count,
        "maximum_action_count": maximum_action_count,
        "unused_budget_query_count": unused_budget_query_count,
        "duplicate_action_count": duplicate_action_count,
        "anchor_failure_count": anchor_failure_count,
        "semantic_original_failure_count": semantic_original_failure_count,
        "composition_failure_count": composition_failure_count,
        "forbidden_action_count": forbidden_action_count,
        "test_partition_touched": False,
        "passed": passed,
    }


__all__ = [
    "audit_core4_semantic_boolean_queries",
    "audit_fixed_budget_receipts",
]
