"""Mechanical validation for generated candidate-recall retrieval actions."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Collection, Mapping
from typing import Literal, NoReturn

from pydantic import ValidationError

from paper_search.domain.models import DomainModel
from paper_search.recall_experiments.contracts import (
    RecallActionBatch,
    RecallGenerationContext,
    RecallSearchAction,
    TextSearchAction,
    TextSearchPayload,
    TitleSearchAction,
    TitleSearchPayload,
)


ActionValidationCode = Literal[
    "invalid_json",
    "duplicate_action",
    "empty_action",
    "action_too_long",
    "disallowed_action_type",
    "year_conflict",
    "missing_required_field",
    "unknown_seed_candidate",
    "action_limit_exceeded",
]
_YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
_MAX_ACTION_TEXT_CHARS = 300
_PHASE_ONE_ACTION_TYPES = frozenset({"text_search", "title_search", "citation_expand"})


class ActionValidationIssue(DomainModel):
    code: ActionValidationCode
    field_path: str
    message: str


class ActionValidationFailure(ValueError):
    def __init__(
        self,
        issues: list[ActionValidationIssue],
        previous_output: object,
        allowed_change_scope: tuple[str, ...] = ("actions",),
    ) -> None:
        self.issues = tuple(issues)
        self.previous_output = previous_output
        self.allowed_change_scope = allowed_change_scope
        super().__init__("; ".join(issue.message for issue in issues))


def validate_action_batch(
    raw: object,
    context: RecallGenerationContext,
    allowed_actions: Collection[str],
    max_actions: int,
) -> RecallActionBatch:
    """Validate a generated action batch without consulting Gold or recall outcomes."""
    decoded = _decode_raw(raw)
    if not isinstance(decoded, Mapping):
        _raise("invalid_json", "", "action batch must be a JSON object", raw)
    unknown_keys = set(decoded).difference({"actions"})
    if unknown_keys:
        key = sorted(str(item) for item in unknown_keys)[0]
        _raise("invalid_json", key, "action batch contains an unknown field", raw)

    raw_actions = decoded.get("actions")
    if not isinstance(raw_actions, list):
        _raise("missing_required_field", "actions", "actions must be a list", raw)
    if len(raw_actions) > max_actions:
        _raise(
            "action_limit_exceeded",
            "actions",
            f"action batch exceeds the limit of {max_actions}",
            raw,
        )

    issues: list[ActionValidationIssue] = []
    actions: list[RecallSearchAction] = []
    seen_ids: set[str] = set()
    seen_search_text: set[tuple[str, str, str]] = set()
    seed_ids = {candidate.paper.canonical_id for candidate in context.seed_candidates}

    for index, candidate in enumerate(raw_actions):
        field_prefix = f"actions.{index}"
        if not isinstance(candidate, Mapping):
            issues.append(_issue("invalid_json", field_prefix, "action must be an object"))
            continue
        action_type = candidate.get("action_type")
        if (
            not isinstance(action_type, str)
            or action_type not in _PHASE_ONE_ACTION_TYPES
            or action_type not in allowed_actions
        ):
            issues.append(
                _issue(
                    "disallowed_action_type",
                    f"{field_prefix}.action_type",
                    "action type is not allowed",
                )
            )
            continue
        empty_issue = _empty_search_text_issue(candidate, field_prefix)
        if empty_issue is not None:
            issues.append(empty_issue)
            continue
        try:
            parsed = RecallActionBatch.model_validate({"actions": [candidate]}).actions[0]
        except ValidationError as error:
            issues.extend(_pydantic_issues(error, field_prefix))
            continue

        if parsed.action_id in seen_ids:
            issues.append(
                _issue(
                    "duplicate_action",
                    f"{field_prefix}.action_id",
                    "action_id must be unique within a batch",
                )
            )
        seen_ids.add(parsed.action_id)

        normalized = _normalize_action(
            parsed, field_prefix, context, seed_ids, seen_search_text, issues
        )
        if normalized is not None:
            actions.append(normalized)

    if issues:
        raise ActionValidationFailure(issues, raw)
    return RecallActionBatch(actions=actions)


def _decode_raw(raw: object) -> object:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        _raise("invalid_json", "", "action batch is not valid JSON", raw)


def _normalize_action(
    action: RecallSearchAction,
    field_prefix: str,
    context: RecallGenerationContext,
    seed_ids: set[str],
    seen_search_text: set[tuple[str, str, str]],
    issues: list[ActionValidationIssue],
) -> RecallSearchAction | None:
    if isinstance(action, (TextSearchAction, TitleSearchAction)):
        raw_text = (
            action.payload.query_text
            if isinstance(action, TextSearchAction)
            else action.payload.title_text
        )
        text = _canonical_text(raw_text)
        field_name = "query_text" if isinstance(action, TextSearchAction) else "title_text"
        field_path = f"{field_prefix}.payload.{field_name}"
        if not text:
            issues.append(_issue("empty_action", field_path, "search text must not be empty"))
            return None
        if len(text) > _MAX_ACTION_TEXT_CHARS:
            issues.append(
                _issue(
                    "action_too_long",
                    field_path,
                    "search text must be 300 characters or fewer",
                )
            )
            return None
        search_mode = (
            action.payload.search_mode
            if isinstance(action, TextSearchAction)
            else "lexical"
        )
        search_key = (action.action_type, search_mode, text.casefold())
        if search_key in seen_search_text:
            issues.append(_issue("duplicate_action", field_path, "search text is duplicated"))
            return None
        seen_search_text.add(search_key)
        if _has_year_conflict(text, context):
            issues.append(
                _issue("year_conflict", field_path, "search text conflicts with query year constraints")
            )
            return None
        if isinstance(action, TextSearchAction):
            return action.model_copy(
                update={
                    "payload": TextSearchPayload(
                        query_text=text,
                        search_mode=action.payload.search_mode,
                    )
                }
            )
        return action.model_copy(update={"payload": TitleSearchPayload(title_text=text)})

    if action.payload.seed_canonical_id not in seed_ids:
        issues.append(
            _issue(
                "unknown_seed_candidate",
                f"{field_prefix}.payload.seed_canonical_id",
                "citation seed must be present in context seed_candidates",
            )
        )
        return None
    return action


def _canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _empty_search_text_issue(
    candidate: Mapping[object, object], field_prefix: str
) -> ActionValidationIssue | None:
    action_type = candidate.get("action_type")
    if action_type == "text_search":
        field_name = "query_text"
    elif action_type == "title_search":
        field_name = "title_text"
    else:
        return None
    payload = candidate.get("payload")
    if isinstance(payload, Mapping) and isinstance(payload.get(field_name), str):
        if not _canonical_text(payload[field_name]):
            return _issue(
                "empty_action",
                f"{field_prefix}.payload.{field_name}",
                "search text must not be empty",
            )
    return None


def _has_year_conflict(text: str, context: RecallGenerationContext) -> bool:
    for match in _YEAR_PATTERN.finditer(text):
        year = int(match.group(1))
        if context.query_spec.year_from is not None and year < context.query_spec.year_from:
            return True
        if context.query_spec.year_to is not None and year > context.query_spec.year_to:
            return True
    return False


def _pydantic_issues(error: ValidationError, prefix: str) -> list[ActionValidationIssue]:
    issues: list[ActionValidationIssue] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"] if part != "actions")
        code: ActionValidationCode = (
            "missing_required_field" if detail["type"] == "missing" else "invalid_json"
        )
        issues.append(_issue(code, f"{prefix}.{location}".rstrip("."), detail["msg"]))
    return issues


def _issue(code: ActionValidationCode, field_path: str, message: str) -> ActionValidationIssue:
    return ActionValidationIssue(code=code, field_path=field_path, message=message)


def _raise(
    code: ActionValidationCode, field_path: str, message: str, raw: object
) -> NoReturn:
    raise ActionValidationFailure([_issue(code, field_path, message)], raw)
