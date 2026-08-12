from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from paper_search.control.budget import HardBudgetController
from paper_search.domain.models import (
    ErrorDetail,
    Paper,
    ProviderResult,
    QuerySpec,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.recall_experiments.contracts import (
    GoldDocument,
    RecallGenerationContext,
    SeedCandidate,
)
from paper_search.recall_experiments.generation.backends import (
    BudgetedLLMBackend,
    LLMBackendResult,
)
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
from paper_search.recall_experiments.validation import (
    ActionValidationFailure,
    ActionValidationIssue,
)


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


def test_manual_generation_identity_hashes_the_exact_consumed_bytes(tmp_path) -> None:
    actions_path = tmp_path / "actions.json"
    original = json.dumps({"query-1": _actions()}).encode("utf-8")
    actions_path.write_bytes(original)
    generator = ManualActionGenerator(
        actions_path,
        expected_query_ids=["query-1"],
        allowed_actions={"text_search"},
        max_actions=1,
    )

    actions_path.write_text('{"query-1":{"actions":[]}}', encoding="utf-8")

    assert generator.source_sha256 == f"sha256:{hashlib.sha256(original).hexdigest()}"
    result = asyncio.run(generator.generate(_context("query-1")))
    assert result.action_batch.actions[0].payload.query_text == "graph retrieval"


class _RecordingLLMBackend:
    def __init__(self, results: list[LLMBackendResult]) -> None:
        self._results = iter(results)
        self.calls: list[tuple[str, object]] = []

    async def generate(self, request: object, call_kind: str) -> LLMBackendResult:
        self.calls.append((call_kind, request))
        result = next(self._results)
        return result.model_copy(
            update={"provenance": {"backend_call_id": f"call-{len(self.calls)}"}}
        )


def _prompt() -> RecallPromptArtifact:
    return RecallPromptArtifact.from_yaml_bytes(
        b"# source bytes are identity-sensitive\nname: recall_scheme_b\nversion: recall-scheme-b-v1\nmodel: deepseek-v4-flash\ntemperature: 0\ninstructions:\n  - Generate only permitted retrieval actions.\n"
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
    assert "supplied seed_canonical_id" in message


def test_rendered_prompt_declares_exact_action_payload_contracts() -> None:
    message = render_recall_prompt(_prompt())

    assert "action_id must be a unique non-empty JSON string" in message
    assert "actions must be a JSON array of action objects" in message
    assert (
        "action_type must be a JSON string exactly equal to one supplied "
        "allowed_action_types value" in message
    )
    assert "strategy must be a non-empty JSON string" in message
    assert (
        'text_search payload keys: exactly ["query_text"]; query_text must be a '
        "non-empty JSON string of at most 300 characters" in message
    )
    assert (
        'title_search payload keys: exactly ["title_text"]; title_text must be a '
        "non-empty JSON string of at most 300 characters" in message
    )
    assert (
        'citation_expand payload keys: exactly '
        '["seed_canonical_id", "direction", "limit"]' in message
    )
    assert "seed_canonical_id must copy a supplied seed verbatim" in message
    assert 'direction must be exactly "references", "citations", or "both"' in message
    assert "limit must be a positive JSON integer, never a boolean" in message
    assert "Do not add any unlisted keys to an action or payload" in message
    assert "Do not add limit to text_search or title_search payloads" in message


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
    assert repair_payload["validation_errors"][0]["message"]
    assert repair_payload["validation_errors"][0]["repair_instruction"]
    assert repair_payload["allowed_change_scope"] == ["actions"]
    assert repair_payload["repair_instruction"] == (
        "Correct every listed validation error; preserve only valid action content; "
        "return the complete corrected RecallActionBatch JSON object."
    )


def test_analyzer_invalid_json_triggers_one_repair_with_previous_output() -> None:
    backend = _RecordingLLMBackend([_error("invalid_json"), _backend_result(_actions())])
    generator = DeepSeekPromptGenerator(
        backend=backend, prompt=_prompt(), visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    result = asyncio.run(generator.generate(_context("query-1")))

    assert result.action_batch.actions[0].action_id == "search-1"
    assert [receipt.call_kind for receipt in result.call_receipts] == ["initial", "repair"]
    assert result.repair_count == 1
    assert len({receipt.provenance["backend_call_id"] for receipt in result.call_receipts}) == 2
    assert [kind for kind, _ in backend.calls] == ["initial", "repair"]
    repair_payload = getattr(backend.calls[1][1], "payload")
    assert repair_payload["validation_errors"] == [
        {
            "code": "invalid_json",
            "field_path": "",
            "message": "invalid JSON",
            "repair_instruction": "Return one valid JSON object with only the top-level actions key.",
        }
    ]
    assert repair_payload["previous_output"] == {}


def test_repair_feedback_explains_action_id_type_and_forbidden_payload_fields() -> None:
    invalid = {
        "actions": [
            {
                "action_id": 1,
                "action_type": "text_search",
                "strategy": "synthetic",
                "payload": {"query_text": "dataset distillation", "limit": 10},
            }
        ]
    }
    backend = _RecordingLLMBackend([_backend_result(invalid), _backend_result(_actions())])
    generator = DeepSeekPromptGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search"},
        max_actions=1,
    )

    asyncio.run(generator.generate(_context("query-1")))

    repair_payload = getattr(backend.calls[1][1], "payload")
    errors = repair_payload["validation_errors"]
    assert any(
        item["field_path"].endswith("action_id")
        and "valid string" in item["message"]
        and "non-empty JSON string" in item["repair_instruction"]
        for item in errors
    )
    assert any(
        item["field_path"].endswith("payload.limit")
        and "Extra inputs" in item["message"]
        and "Remove this unlisted field" in item["repair_instruction"]
        for item in errors
    )


@pytest.mark.parametrize(
    ("invalid_payload", "instruction_fragment"),
    [
        ({"query_text": ""}, "non-empty JSON string"),
        ({"query_text": "x" * 301}, "300 characters or fewer"),
    ],
)
def test_repair_feedback_gives_field_specific_text_correction(
    invalid_payload: dict[str, object], instruction_fragment: str
) -> None:
    invalid = {
        "actions": [
            {
                "action_id": "a-1",
                "action_type": "text_search",
                "strategy": "synthetic",
                "payload": invalid_payload,
            }
        ]
    }
    backend = _RecordingLLMBackend([_backend_result(invalid), _backend_result(_actions())])
    generator = DeepSeekPromptGenerator(
        backend=backend,
        prompt=_prompt(),
        visibility="blind",
        allowed_actions={"text_search"},
        max_actions=1,
    )

    asyncio.run(generator.generate(_context("query-1")))

    errors = getattr(backend.calls[1][1], "payload")["validation_errors"]
    assert instruction_fragment in errors[0]["repair_instruction"]


def test_initial_and_repair_requests_carry_the_same_rendered_prompt_and_exact_source_identity() -> None:
    source = b"# retained comment\nname: recall_scheme_b\nversion: recall-scheme-b-v1\nmodel: deepseek-v4-flash\ntemperature: 0\ninstructions:\n  - Generate only permitted retrieval actions.\n"
    prompt = RecallPromptArtifact.from_yaml_bytes(source)
    backend = _RecordingLLMBackend([_backend_result({"actions": [{}]}), _backend_result(_actions())])
    generator = DeepSeekPromptGenerator(
        backend=backend, prompt=prompt, visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    asyncio.run(generator.generate(_context("query-1")))

    initial_request = backend.calls[0][1]
    repair_request = backend.calls[1][1]
    expected_digest = "sha256:" + hashlib.sha256(source).hexdigest()
    assert getattr(initial_request, "prompt_instructions") == render_recall_prompt(prompt)
    assert getattr(repair_request, "prompt_instructions") == render_recall_prompt(prompt)
    assert getattr(initial_request, "prompt_bytes") == source
    assert getattr(repair_request, "prompt_bytes") == source
    assert getattr(initial_request, "prompt_artifact_sha256") == expected_digest
    assert getattr(repair_request, "prompt_artifact_sha256") == expected_digest


class _ReservationRecordingAnalyzer:
    def __init__(self) -> None:
        self.results = iter([{"actions": [{}]}, _actions()])
        self.calls: list[tuple[object, str | None, str | None]] = []

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: object,
        prompt_instructions: str | None = None,
        prompt_artifact_sha256: str | None = None,
    ) -> ProviderResult[dict[str, object]]:
        assert prompt_name == "recall_scheme_b"
        assert payload
        self.calls.append((reservation, prompt_instructions, prompt_artifact_sha256))
        return ProviderResult(
            data=next(self.results),
            usage=UsageActual(llm_calls=1, cost_cny=Decimal("0.01")),
            provenance={
                "provider": "sealed-fake",
                "endpoint": "/sealed",
                "model_id": "fake-model",
                "requested_at": datetime.now(UTC).isoformat(),
                "response_hash": "sha256:sealed",
            },
            cache_hit=True,
            latency_ms=0,
            errors=[],
        )


def test_initial_and_repair_use_distinct_budget_reservations_with_the_exact_prompt_identity() -> None:
    analyzer = _ReservationRecordingAnalyzer()
    controller = HardBudgetController(
        SearchBudget(
            max_search_api_calls=1,
            target_search_api_calls=1,
            max_llm_calls=2,
            target_llm_calls=2,
            max_total_tokens=10,
            max_cost_cny=Decimal("1"),
            max_elapsed_seconds=60,
            soft_deadline_seconds=10,
        )
    )
    backend = BudgetedLLMBackend(
        analyzer=analyzer,
        controller=controller,
        initial_estimate=UsageEstimate(llm_calls=1, cost_cny=Decimal("0.01")),
        repair_estimate=UsageEstimate(llm_calls=1, cost_cny=Decimal("0.01")),
    )
    prompt = _prompt()
    generator = DeepSeekPromptGenerator(
        backend=backend, prompt=prompt, visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    asyncio.run(generator.generate(_context("query-1")))

    assert len(analyzer.calls) == 2
    assert analyzer.calls[0][0].reservation_id != analyzer.calls[1][0].reservation_id
    assert [item[1] for item in analyzer.calls] == [render_recall_prompt(prompt)] * 2
    assert [item[2] for item in analyzer.calls] == [prompt.sha256] * 2
    assert [item["action"] for item in controller.export_state()["terminal_outcomes"]] == [
        "recall.generate.initial",
        "recall.generate.repair",
    ]


def test_prompt_digest_is_the_exact_yaml_bytes_digest_and_matches_the_recipe_loader() -> None:
    loaded = load_recall_recipe(Path("configs/recall_experiments/methods/scheme-b-blind.yaml"))
    assert loaded.prompt_bytes is not None
    prompt = RecallPromptArtifact.from_yaml_bytes(loaded.prompt_bytes)
    changed_whitespace = RecallPromptArtifact.from_yaml_bytes(loaded.prompt_bytes + b"# changed\n")

    assert prompt.sha256 == loaded.prompt_sha256
    assert changed_whitespace.sha256 != prompt.sha256


@pytest.mark.parametrize("forbidden", ["10.1234/seed", "https://example.invalid/paper", "W12345678", "S2:123"])
def test_seed_prose_identifier_patterns_are_rejected_after_identifier_fields_are_removed(forbidden: str) -> None:
    context = _context("query-1").model_copy(
        update={
            "seed_candidates": [
                SeedCandidate(
                    paper=Paper(
                        canonical_id="seed-1",
                        title=f"Ordinary {forbidden}",
                        abstract="OpenAlex remains ordinary prose.",
                        sources=["openalex"],
                    )
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="forbidden identifier pattern"):
        build_generation_payload(
            context, visibility="blind", allowed_actions={"text_search"}, max_actions=1
        )


def test_seed_prose_can_mention_openalex_without_an_identifier() -> None:
    context = _context("query-1").model_copy(
        update={
            "seed_candidates": [
                SeedCandidate(
                    paper=Paper(
                        canonical_id="seed-1",
                        title="OpenAlex coverage study",
                        abstract="Semantic Scholar is discussed as ordinary prose.",
                        sources=["openalex"],
                    )
                )
            ]
        }
    )

    payload = build_generation_payload(
        context, visibility="blind", allowed_actions={"text_search"}, max_actions=1
    )

    assert payload["seed_candidates"][0]["title"] == "OpenAlex coverage study"


def test_citation_seed_id_is_exposed_for_a_valid_generated_action() -> None:
    context = _context("query-1").model_copy(
        update={
            "seed_candidates": [
                SeedCandidate(
                    paper=Paper(
                        canonical_id="seed-1",
                        title="Frozen citation seed",
                        sources=["semantic_scholar"],
                    )
                )
            ]
        }
    )
    payload = build_generation_payload(
        context, visibility="blind", allowed_actions={"citation_expand"}, max_actions=1
    )

    assert payload["seed_candidates"][0]["seed_canonical_id"] == "seed-1"


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


def test_repair_payload_is_limited_to_local_validation_diagnostics() -> None:
    from paper_search.recall_experiments.validation import ActionValidationFailure, ActionValidationIssue

    payload = build_repair_payload(
        ActionValidationFailure(
            [ActionValidationIssue(code="invalid_json", field_path="actions", message="bad")],
            previous_output={"actions": "bad"},
        )
    )

    assert payload == {
        "previous_output": {"actions": "bad"},
        "validation_errors": [
            {
                "code": "invalid_json",
                "field_path": "actions",
                "message": "bad",
                "repair_instruction": "Replace actions with a JSON array of valid action objects.",
            }
        ],
        "allowed_change_scope": ["actions"],
        "repair_instruction": (
            "Correct every listed validation error; preserve only valid action content; "
            "return the complete corrected RecallActionBatch JSON object."
        ),
    }


def test_repair_payload_maps_multiple_citation_errors_to_their_own_fields() -> None:
    from paper_search.recall_experiments.validation import (
        ActionValidationFailure,
        ActionValidationIssue,
    )

    payload = build_repair_payload(
        ActionValidationFailure(
            [
                ActionValidationIssue(
                    code="invalid_json",
                    field_path="actions.0.0.citation_expand.payload.direction",
                    message="Input should be 'references', 'citations' or 'both'",
                ),
                ActionValidationIssue(
                    code="invalid_json",
                    field_path="actions.0.0.citation_expand.payload.limit",
                    message="Input should be a valid integer",
                ),
            ],
            previous_output={"actions": []},
        )
    )

    errors = payload["validation_errors"]
    assert errors[0]["field_path"].endswith("payload.direction")
    assert '"references", "citations", or "both"' in errors[0]["repair_instruction"]
    assert errors[1]["field_path"].endswith("payload.limit")
    assert "positive JSON integer" in errors[1]["repair_instruction"]


@pytest.mark.parametrize(
    ("issue", "expected_instruction"),
    [
        (
            ActionValidationIssue(
                code="disallowed_action_type",
                field_path="actions.0.action_type",
                message="action type is not allowed",
            ),
            "Delete the entire action containing this disallowed action_type.",
        ),
        (
            ActionValidationIssue(
                code="unknown_seed_candidate",
                field_path="actions.0.payload.seed_canonical_id",
                message="citation seed must be present in context seed_candidates",
            ),
            "Delete the entire citation_expand action containing this unknown seed.",
        ),
        (
            ActionValidationIssue(
                code="year_conflict",
                field_path="actions.0.payload.query_text",
                message="search text conflicts with query year constraints",
            ),
            "Remove every explicit year from this search text.",
        ),
    ],
)
def test_repair_instructions_are_executable_without_initial_request_context(
    issue: ActionValidationIssue, expected_instruction: str
) -> None:
    payload = build_repair_payload(
        ActionValidationFailure([issue], previous_output={"actions": []})
    )

    assert set(payload) == {
        "previous_output",
        "validation_errors",
        "allowed_change_scope",
        "repair_instruction",
    }
    assert payload["validation_errors"][0]["repair_instruction"] == expected_instruction


def test_title_informed_prompt_recipe_is_a_blind_four_action_text_search_method() -> None:
    loaded = load_recall_recipe(
        Path("configs/recall_experiments/methods/title-informed-blind-live.yaml")
    )

    assert loaded.recipe.method_id == "title-informed-blind"
    assert loaded.recipe.generator.type == "deepseek_prompt"
    assert loaded.recipe.generator.gold_visibility == "blind"
    assert loaded.recipe.generator.max_generated_actions == 4
    assert loaded.recipe.retrieval.allowed_actions == ["text_search"]
    assert loaded.recipe.retrieval.backend == "live_provider"
    assert loaded.recipe.retrieval.max_total_actions == 4
    assert loaded.prompt_bytes is not None
    prompt = RecallPromptArtifact.from_yaml_bytes(loaded.prompt_bytes)
    rendered = render_recall_prompt(prompt)
    assert "query.original_query is the fixed content insertion point" in rendered
    assert "anchor_full" in rendered
    assert "subject_task" in rendered
    assert "method_task" in rendered
    assert "dataset_task" in rendered
    assert "Omit a combination slot" in rendered
    assert "Do not use Gold" in rendered


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
