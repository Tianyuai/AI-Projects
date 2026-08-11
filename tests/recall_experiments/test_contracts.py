from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_search.domain.models import Paper, QuerySpec
from paper_search.recall_experiments import (
    ActionValidationFailure as ExportedActionValidationFailure,
    RecallActionBatch as ExportedRecallActionBatch,
    validate_action_batch as exported_validate_action_batch,
)
from paper_search.recall_experiments.contracts import (
    CitationExpandAction,
    GoldDocument,
    RecallActionBatch,
    RecallGenerationContext,
    SeedCandidate,
    TextSearchAction,
    TitleSearchAction,
    assert_no_forbidden_identifier_keys_or_patterns,
    generation_payload,
)
from paper_search.recall_experiments.validation import (
    ActionValidationFailure,
    validate_action_batch,
)


def _paper(canonical_id: str = "seed-1") -> Paper:
    return Paper(
        canonical_id=canonical_id,
        title="Provider normalized candidate",
        sources=["openalex"],
    )


def _context() -> RecallGenerationContext:
    return RecallGenerationContext(
        query_id="q-1",
        original_query="graph retrieval",
        query_spec=QuerySpec(
            original_query="graph retrieval",
            research_goal="find graph retrieval papers",
            year_from=2020,
            year_to=2024,
        ),
        seed_queries=["graph retrieval"],
        seed_candidates=[SeedCandidate(paper=_paper())],
        observable_state="low_yield",
        gold_documents=[
            GoldDocument(
                title="A gold document",
                abstract="OpenAlex and Semantic Scholar are named in prose.",
                authors=["Ada Author"],
                publication_year=2022,
                metadata_coverage={"abstract": True, "authors": True, "year": True},
            )
        ],
    )


def _valid_actions() -> list[dict[str, object]]:
    return [
        {
            "action_id": "a-text",
            "action_type": "text_search",
            "strategy": "facet expansion",
            "payload": {"query_text": "graph retrieval 2022"},
        },
        {
            "action_id": "a-title",
            "action_type": "title_search",
            "strategy": "known title",
            "payload": {"title_text": "Exact Candidate Title"},
        },
        {
            "action_id": "a-cite",
            "action_type": "citation_expand",
            "strategy": "citation neighborhood",
            "payload": {
                "seed_canonical_id": "seed-1",
                "direction": "both",
                "limit": 10,
            },
        },
    ]


def _validate(raw: object, *, max_actions: int = 3) -> RecallActionBatch:
    return validate_action_batch(
        raw,
        _context(),
        allowed_actions={"text_search", "title_search", "citation_expand"},
        max_actions=max_actions,
    )


def _failure_codes(raw: object, *, max_actions: int = 3) -> set[str]:
    with pytest.raises(ActionValidationFailure) as caught:
        _validate(raw, max_actions=max_actions)
    return {issue.code for issue in caught.value.issues}


def test_package_exports_recall_validation_api() -> None:
    assert ExportedActionValidationFailure is ActionValidationFailure
    assert ExportedRecallActionBatch is RecallActionBatch
    assert exported_validate_action_batch is validate_action_batch


def test_closed_models_reject_extra_fields_and_invalid_discriminated_payloads() -> None:
    with pytest.raises(ValidationError):
        GoldDocument(title="gold", metadata_coverage={}, canonical_id="forbidden")
    with pytest.raises(ValidationError):
        TextSearchAction.model_validate(
            {
                "action_id": "a-1",
                "action_type": "text_search",
                "strategy": "open strategy",
                "payload": {"title_text": "wrong payload"},
            }
        )
    with pytest.raises(ValidationError):
        CitationExpandAction.model_validate(
            {
                "action_id": "a-2",
                "action_type": "citation_expand",
                "strategy": "open strategy",
                "payload": {"seed_canonical_id": "seed-1", "direction": "sideways", "limit": 1},
            }
        )


def test_seed_candidates_require_provider_normalized_papers() -> None:
    with pytest.raises(ValidationError, match="provider-normalized"):
        SeedCandidate(paper=Paper(canonical_id="seed-1", title="Not provider normalized"))


def test_action_batch_preserves_input_order() -> None:
    batch = _validate({"actions": _valid_actions()})

    assert [action.action_id for action in batch.actions] == ["a-text", "a-title", "a-cite"]
    assert isinstance(batch.actions[0], TextSearchAction)
    assert isinstance(batch.actions[1], TitleSearchAction)
    assert isinstance(batch.actions[2], CitationExpandAction)


def test_validation_returns_typed_normalized_search_payloads() -> None:
    actions = _valid_actions()[:2]
    actions[0]["payload"] = {"query_text": "  graph   retrieval 2022  "}
    actions[1]["payload"] = {"title_text": "  Exact   Candidate Title  "}

    batch = _validate({"actions": actions})

    text_action, title_action = batch.actions
    assert isinstance(text_action, TextSearchAction)
    assert text_action.payload.query_text == "graph retrieval 2022"
    assert isinstance(title_action, TitleSearchAction)
    assert title_action.payload.title_text == "Exact Candidate Title"


def test_validation_rejects_unknown_batch_envelope_keys() -> None:
    with pytest.raises(ActionValidationFailure) as caught:
        _validate({"actions": _valid_actions()[:1], "unexpected": True})

    assert {issue.code for issue in caught.value.issues} == {"invalid_json"}
    assert caught.value.issues[0].field_path == "unexpected"


def test_validation_closes_phase_one_types_when_caller_allows_unknown_type() -> None:
    raw = {
        "actions": [
            {
                "action_id": "a-web",
                "action_type": "web_search",
                "strategy": "unsupported",
                "payload": {"query_text": "graph retrieval"},
            }
        ]
    }

    with pytest.raises(ActionValidationFailure) as caught:
        validate_action_batch(
            raw,
            _context(),
            allowed_actions={"text_search", "title_search", "citation_expand", "web_search"},
            max_actions=3,
        )

    assert {issue.code for issue in caught.value.issues} == {"disallowed_action_type"}


def test_validation_rejects_duplicate_action_ids() -> None:
    actions = _valid_actions()
    actions[1]["action_id"] = "a-text"

    assert _failure_codes({"actions": actions}) == {"duplicate_action"}


@pytest.mark.parametrize("duplicate_text", ["  graph   retrieval 2022  ", "graph retrieval \uff12\uff10\uff12\uff12"])
def test_validation_rejects_canonical_duplicate_search_text(duplicate_text: str) -> None:
    actions = _valid_actions()[:1]
    actions.append(
        {
            "action_id": "a-duplicate",
            "action_type": "text_search",
            "strategy": "alternative wording",
            "payload": {"query_text": duplicate_text},
        }
    )

    assert _failure_codes({"actions": actions}) == {"duplicate_action"}


@pytest.mark.parametrize(
    ("query_text", "expected_code"),
    [("   ", "empty_action"), ("x" * 301, "action_too_long")],
)
def test_validation_rejects_empty_and_too_long_search_text(
    query_text: str, expected_code: str
) -> None:
    action = _valid_actions()[0]
    action["payload"] = {"query_text": query_text}

    assert _failure_codes({"actions": [action]}) == {expected_code}


def test_validation_rejects_illegal_action_type() -> None:
    assert _failure_codes(
        {
            "actions": [
                {
                    "action_id": "a-illegal",
                    "action_type": "web_search",
                    "strategy": "unsupported",
                    "payload": {"query_text": "graph retrieval"},
                }
            ]
        }
    ) == {"disallowed_action_type"}


def test_validation_rejects_year_outside_query_spec() -> None:
    action = _valid_actions()[0]
    action["payload"] = {"query_text": "graph retrieval 2019"}

    assert _failure_codes({"actions": [action]}) == {"year_conflict"}


def test_validation_reports_missing_payload_fields() -> None:
    action = _valid_actions()[2]
    action["payload"] = {"seed_canonical_id": "seed-1", "direction": "both"}

    assert _failure_codes({"actions": [action]}) == {"missing_required_field"}


def test_validation_rejects_action_count_above_limit() -> None:
    assert _failure_codes({"actions": _valid_actions()}, max_actions=2) == {"action_limit_exceeded"}


def test_validation_requires_citation_seed_from_context_candidates() -> None:
    action = _valid_actions()[2]
    action["payload"] = {"seed_canonical_id": "not-a-seed", "direction": "references", "limit": 1}

    assert _failure_codes({"actions": [action]}) == {"unknown_seed_candidate"}


def test_oracle_payload_contains_only_identifier_free_gold_documents() -> None:
    oracle = _context()

    assert_no_forbidden_identifier_keys_or_patterns(oracle.model_dump(mode="json"))
    blind_payload = generation_payload(oracle, visibility="blind")

    assert "gold_documents" not in blind_payload
    assert generation_payload(oracle, visibility="oracle")["gold_documents"]


@pytest.mark.parametrize(
    "payload",
    [
        {"doi": "10.1000/example"},
        {"canonical_id": "seed-1"},
        {"openalex_id": "W1234567890"},
        {"url": "https://api.openalex.org/works/W1234567890"},
        {"provider_request_id": "request-123"},
    ],
)
def test_privacy_scanner_rejects_identifier_keys_and_patterns(payload: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_no_forbidden_identifier_keys_or_patterns(payload)


def test_privacy_scanner_allows_provider_names_in_ordinary_prose() -> None:
    assert_no_forbidden_identifier_keys_or_patterns(
        {"title": "OpenAlex coverage", "abstract": "Semantic Scholar is discussed."}
    )
