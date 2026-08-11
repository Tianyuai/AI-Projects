from __future__ import annotations

import asyncio
import json

import pytest

from paper_search.domain.models import QuerySpec
from paper_search.recall_experiments.contracts import RecallGenerationContext
from paper_search.recall_experiments.generation.fixed import FixedActionGenerator
from paper_search.recall_experiments.generation.manual import ManualActionGenerator


def _context(query_id: str) -> RecallGenerationContext:
    return RecallGenerationContext(
        query_id=query_id,
        original_query="graph retrieval",
        query_spec=QuerySpec(original_query="graph retrieval", research_goal="graph retrieval"),
    )


def _actions(query_text: str = "graph retrieval") -> dict[str, object]:
    return {
        "actions": [
            {
                "action_id": "search-1",
                "action_type": "text_search",
                "strategy": "fixed",
                "payload": {"query_text": query_text},
            }
        ]
    }


def test_fixed_generation_preserves_the_bound_action_bytes_and_validates_before_return() -> None:
    raw = json.dumps(_actions(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    generator = FixedActionGenerator(
        {"query-1": raw},
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )

    result = asyncio.run(generator.generate(_context("query-1")))

    assert result.query_id == "query-1"
    assert result.artifact_bytes == raw
    assert result.action_batch.actions[0].action_id == "search-1"


def test_fixed_generation_rejects_unknown_or_missing_query_ids() -> None:
    with pytest.raises(ValueError, match="coverage"):
        FixedActionGenerator(
            {"query-1": _actions()},
            expected_query_ids=["query-1", "query-2"],
            allowed_actions={"text_search"},
            max_actions=1,
        )

    generator = FixedActionGenerator(
        {"query-1": _actions()},
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )
    with pytest.raises(ValueError, match="unknown query"):
        asyncio.run(generator.generate(_context("query-2")))


def test_fixed_generation_freezes_nested_caller_actions_at_construction() -> None:
    caller_owned = {"query-1": _actions()}
    generator = FixedActionGenerator(
        caller_owned,
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )
    before = asyncio.run(generator.generate(_context("query-1")))

    actions = caller_owned["query-1"]["actions"]
    assert isinstance(actions, list)
    first = actions[0]
    assert isinstance(first, dict)
    payload = first["payload"]
    assert isinstance(payload, dict)
    payload["query_text"] = "mutated after construction"
    actions.append(
        {
            "action_id": "late-action",
            "action_type": "text_search",
            "strategy": "late",
            "payload": {"query_text": "late mutation"},
        }
    )

    after = asyncio.run(generator.generate(_context("query-1")))

    assert after.artifact_bytes == before.artifact_bytes
    assert after.action_batch == before.action_batch


def test_manual_generation_reads_user_prepared_json_without_an_llm(tmp_path) -> None:
    actions_path = tmp_path / "actions.json"
    actions_path.write_text(json.dumps({"query-1": _actions()}), encoding="utf-8")
    generator = ManualActionGenerator(
        actions_path,
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )

    result = asyncio.run(generator.generate(_context("query-1")))

    assert result.action_batch.actions[0].payload.query_text == "graph retrieval"
