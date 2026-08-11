from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from paper_search.domain.models import QuerySpec
from paper_search.domain.models import ErrorDetail
from paper_search.recall_experiments.contracts import GoldDocument, RecallGenerationContext
from paper_search.recall_experiments.generation.backends import LLMBackendResult
from paper_search.recall_experiments.generation.deepseek import (
    DeepSeekPromptGenerator,
    RecallGenerationFailure,
    RecallPromptArtifact,
    build_generation_payload,
    build_repair_payload,
    render_recall_prompt,
)
from paper_search.recall_experiments.generation.fixed import FixedActionGenerator
from paper_search.recall_experiments.generation.manual import ManualActionGenerator
from paper_search.recall_experiments.recipes import load_recall_recipe


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


@pytest.mark.parametrize("as_bytes", [True, False])
def test_fixed_generation_preserves_formatted_and_reordered_byte_or_string_sources(
    as_bytes: bool,
) -> None:
    source_text = """{
  "actions": [
    {
      "strategy": "fixed",
      "payload": {"query_text": "graph retrieval"},
      "action_type": "text_search",
      "action_id": "search-1"
    }
  ]
}"""
    source = source_text.encode("utf-8") if as_bytes else source_text
    generator = FixedActionGenerator(
        {"query-1": source},
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )

    result = asyncio.run(generator.generate(_context("query-1")))

    assert result.artifact_bytes == source_text.encode("utf-8")
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


class _RecordingLLMBackend:
    def __init__(self, results: list[LLMBackendResult]) -> None:
        self._results = iter(results)
        self.calls: list[tuple[str, object]] = []

    async def generate(self, request: object, call_kind: str) -> LLMBackendResult:
        self.calls.append((call_kind, request))
        return next(self._results)


def _prompt() -> RecallPromptArtifact:
    return RecallPromptArtifact(
        name="recall_scheme_b",
        version="recall-scheme-b-v1",
        model="deepseek-v4-flash",
        temperature=0,
        instructions=["Generate only permitted retrieval actions."],
    )


def _oracle_context(*, year_from: int | None = None) -> RecallGenerationContext:
    return RecallGenerationContext(
        query_id="query-1",
        original_query="graph retrieval",
        query_spec=QuerySpec(
            original_query="graph retrieval", research_goal="graph retrieval", year_from=year_from
        ),
        seed_queries=["graph retrieval survey"],
        gold_documents=[
            GoldDocument(
                title="OpenAlex and Semantic Scholar coverage study",
                abstract="This ordinary prose mentions OpenAlex and Semantic Scholar.",
                authors=["A. Researcher"],
                publication_year=2024,
                metadata_coverage={
                    "title": True,
                    "abstract": True,
                    "authors": True,
                    "publication_year": True,
                },
            )
        ],
    )


def _backend_result(data: dict[str, object]) -> LLMBackendResult:
    return LLMBackendResult(data=data)


def _error(code: str) -> LLMBackendResult:
    return LLMBackendResult(
        errors=[ErrorDetail(code=code, message="sealed fake failure", retryable=False, provider="fake")],
        infrastructure_failure=code != "invalid_json",
        repairable=code == "invalid_json",
    )


def _assert_no_identifier_material(value: object) -> None:
    forbidden = ("doi", "canonical_id", "openalex_id", "semantic_scholar_id", "request_id", "url")
    if isinstance(value, dict):
        assert not any(any(item in key.casefold() for item in forbidden) for key in value)
        for child in value.values():
            _assert_no_identifier_material(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_identifier_material(child)
    elif isinstance(value, str):
        assert "10.1234/" not in value
        assert "https://" not in value


def test_oracle_payload_contains_only_safe_generation_context_and_keeps_ordinary_prose() -> None:
    payload = build_generation_payload(
        _oracle_context(), visibility="oracle", allowed_actions={"text_search"}, max_actions=1
    )

    assert set(payload) == {"query", "seed_queries", "seed_candidates", "allowed_action_schema", "gold_documents"}
    gold = payload["gold_documents"]
    assert isinstance(gold, list)
    assert gold[0]["title"] == "OpenAlex and Semantic Scholar coverage study"
    assert "OpenAlex" in gold[0]["abstract"]
    _assert_no_identifier_material(payload)


def test_blind_payload_has_no_gold_documents_at_any_nesting_level() -> None:
    payload = build_generation_payload(
        _oracle_context(), visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    def _keys(value: object) -> list[str]:
        if isinstance(value, dict):
            return list(value) + [key for child in value.values() for key in _keys(child)]
        if isinstance(value, list):
            return [key for child in value for key in _keys(child)]
        return []

    assert "gold_documents" not in _keys(payload)


def test_historical_visibility_is_preserved_without_an_oracle_upgrade() -> None:
    payload = build_generation_payload(
        _oracle_context(), visibility="historical", historical_visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    assert "gold_documents" not in payload


def test_rendered_recall_prompt_is_deterministic_and_locks_deepseek_settings() -> None:
    message = render_recall_prompt(_prompt())

    assert "RecallActionBatch" in message
    assert "deepseek-v4-flash" in message
    assert "temperature 0" in message
    assert "text_search" in message


@pytest.mark.parametrize(
    ("invalid", "expected_code", "context"),
    [
        ({"actions": [{**_actions()["actions"][0]}, {**_actions()["actions"][0]}]}, "duplicate_action", _context("query-1")),
        ({"actions": [{**_actions()["actions"][0], "payload": {"query_text": "  "}}]}, "empty_action", _context("query-1")),
        ({"actions": [{**_actions()["actions"][0], "payload": {"query_text": "x" * 301}}]}, "action_too_long", _context("query-1")),
        ({"actions": [{**_actions()["actions"][0], "action_type": "title_search", "payload": {"title_text": "title"}}]}, "disallowed_action_type", _context("query-1")),
        ({"actions": [{**_actions()["actions"][0], "payload": {"query_text": "graph 2018"}}]}, "year_conflict", _oracle_context(year_from=2020)),
        ({"actions": [{"action_id": "search-1", "action_type": "text_search", "strategy": "fixed", "payload": {}}]}, "missing_required_field", _context("query-1")),
        ({"actions": [{"action_id": "cite-1", "action_type": "citation_expand", "strategy": "fixed", "payload": {"seed_canonical_id": "unknown", "direction": "references", "limit": 1}}]}, "unknown_seed_candidate", _context("query-1")),
        ({"actions": [_actions()["actions"][0], {"action_id": "search-2", "action_type": "text_search", "strategy": "fixed", "payload": {"query_text": "second"}}]}, "action_limit_exceeded", _context("query-1")),
    ],
)
def test_structured_validation_failure_triggers_one_repair(
    invalid: dict[str, object], expected_code: str, context: RecallGenerationContext
) -> None:
    backend = _RecordingLLMBackend([_backend_result(invalid), _backend_result(_actions())])
    generator = DeepSeekPromptGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search", "citation_expand"},
        max_actions=2 if expected_code == "duplicate_action" else 1,
    )

    result = asyncio.run(generator.generate(context))

    assert result.action_batch.actions[0].action_id == "search-1"
    assert [kind for kind, _ in backend.calls] == ["initial", "repair"]
    repair_request = backend.calls[1][1]
    repair_payload = getattr(repair_request, "payload")
    assert repair_payload["validation_errors"][0]["code"] == expected_code
    assert repair_payload["allowed_change_scope"] == ["actions"]


def test_analyzer_invalid_json_triggers_one_repair_with_previous_output() -> None:
    backend = _RecordingLLMBackend([_error("invalid_json"), _backend_result(_actions())])
    generator = DeepSeekPromptGenerator(
        backend=backend, prompt=_prompt(), visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    result = asyncio.run(generator.generate(_context("query-1")))

    assert result.action_batch.actions[0].action_id == "search-1"
    assert [kind for kind, _ in backend.calls] == ["initial", "repair"]
    repair_payload = getattr(backend.calls[1][1], "payload")
    assert repair_payload["validation_errors"] == [{"code": "invalid_json", "field_path": ""}]
    assert repair_payload["previous_output"] == {}


def test_second_invalid_output_is_a_generation_failure() -> None:
    backend = _RecordingLLMBackend(
        [_backend_result({"actions": [{}]}), _backend_result({"actions": [{}]})]
    )
    generator = DeepSeekPromptGenerator(
        backend=backend, prompt=_prompt(), visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    with pytest.raises(RecallGenerationFailure, match="generation_failure") as caught:
        asyncio.run(generator.generate(_context("query-1")))
    assert caught.value.code == "generation_failure"
    assert [kind for kind, _ in backend.calls] == ["initial", "repair"]


@pytest.mark.parametrize(
    "code", ["authentication_error", "rate_limited", "network_error", "snapshot_unavailable", "accounting_failure"]
)
def test_infrastructure_failures_never_consume_a_semantic_repair(code: str) -> None:
    backend = _RecordingLLMBackend([_error(code)])
    generator = DeepSeekPromptGenerator(
        backend=backend, prompt=_prompt(), visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    with pytest.raises(RecallGenerationFailure, match="infrastructure_failure") as caught:
        asyncio.run(generator.generate(_context("query-1")))
    assert caught.value.code == "infrastructure_failure"
    assert [kind for kind, _ in backend.calls] == ["initial"]


def test_repair_payload_is_limited_to_validation_diagnostics() -> None:
    from paper_search.recall_experiments.validation import ActionValidationFailure, ActionValidationIssue

    payload = build_repair_payload(
        ActionValidationFailure(
            [ActionValidationIssue(code="invalid_json", field_path="actions", message="bad")],
            previous_output={"actions": "bad"},
        )
    )

    assert payload == {
        "previous_output": {"actions": "bad"},
        "validation_errors": [{"code": "invalid_json", "field_path": "actions"}],
        "allowed_change_scope": ["actions"],
    }


@pytest.mark.parametrize("mode", ["oracle", "blind"])
def test_scheme_b_exploration_recipes_lock_the_deepseek_generation_recipe(mode: str) -> None:
    loaded = load_recall_recipe(Path(f"configs/recall_experiments/methods/scheme-b-{mode}.yaml"))

    assert loaded.recipe.method_id == f"scheme-b-{mode}"
    assert loaded.recipe.generator.type == "deepseek_prompt"
    assert loaded.recipe.generator.model == "deepseek-v4-flash"
    assert loaded.recipe.generator.temperature == 0
    assert loaded.recipe.generator.repair_attempts == 1
    assert loaded.recipe.generator.gold_visibility == mode
    assert loaded.prompt_bytes is not None
    assert loaded.prompt_sha256 is not None
