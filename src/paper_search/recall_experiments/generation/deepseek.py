"""Safe, repair-once DeepSeek prompt generation for recall actions.

The generator deliberately depends on the narrow, budget-owning ``LLMBackend``
protocol.  It never constructs an LLM client, analyzer, snapshot reader, or
budget reservation itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from typing import Literal

import yaml
from pydantic import Field, model_validator

from paper_search.domain.models import DomainModel, ErrorDetail, NonEmptyStr
from paper_search.recall_experiments.contracts import (
    GoldVisibility,
    RecallGenerationContext,
    assert_no_forbidden_identifier_keys_or_patterns,
)
from paper_search.recall_experiments.generation.backends import (
    LLMBackend,
    LLMBackendResult,
    LLMGenerationRequest,
)
from paper_search.recall_experiments.generation.base import GenerationResult
from paper_search.recall_experiments.validation import (
    ActionValidationFailure,
    ActionValidationIssue,
    validate_action_batch,
)


_LOCKED_MODEL = "deepseek-v4-flash"
_LOCKED_TEMPERATURE = 0
_REPAIR_ATTEMPTS = 1


class RecallPromptArtifact(DomainModel):
    """The small, hashable prompt recipe used by a recall-action generator."""

    name: NonEmptyStr
    version: NonEmptyStr
    model: NonEmptyStr
    temperature: Literal[0]
    instructions: list[NonEmptyStr] = Field(min_length=1)
    source_bytes: bytes = Field(min_length=1, exclude=True, repr=False)

    @classmethod
    def from_yaml_bytes(cls, source_bytes: bytes) -> RecallPromptArtifact:
        """Parse YAML while preserving its exact bytes as the prompt identity."""
        try:
            decoded = yaml.safe_load(source_bytes)
        except yaml.YAMLError as error:
            raise ValueError("invalid recall prompt artifact") from error
        if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
            raise ValueError("invalid recall prompt artifact")
        return cls.model_validate({**decoded, "source_bytes": source_bytes})

    @model_validator(mode="after")
    def validate_locked_generation_settings(self) -> RecallPromptArtifact:
        if self.model != _LOCKED_MODEL:
            raise ValueError(f"recall prompt model must be {_LOCKED_MODEL}")
        if self.temperature != _LOCKED_TEMPERATURE:
            raise ValueError("recall prompt temperature must be 0")
        return self

    def canonical_bytes(self) -> bytes:
        """Stable parsed representation, useful for display but not snapshot identity."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.source_bytes).hexdigest()


class RecallGenerationFailure(RuntimeError):
    """A terminal outcome that callers can classify without provider details."""

    def __init__(self, code: Literal["generation_failure", "infrastructure_failure"], errors: list[ErrorDetail]) -> None:
        self.code = code
        self.errors = tuple(errors)
        super().__init__(code)


def render_recall_prompt(prompt: RecallPromptArtifact) -> str:
    """Render the deterministic system prompt; request payloads remain separate."""
    return "\n".join(
        [
            "Respond with one strict RecallActionBatch JSON object.",
            f"Prompt version: {prompt.version}.",
            f"Model: {prompt.model}; temperature {_LOCKED_TEMPERATURE}.",
            "Use only action types supplied by allowed_action_schema.",
            "Supported action contracts include text_search, title_search, and citation_expand.",
            "The top-level object must contain only actions.",
            "Each action requires action_id, action_type, strategy, and payload.",
            "Do not return Markdown, identifiers, URLs, prior hits, or evaluation results.",
            *(f"- {instruction}" for instruction in prompt.instructions),
        ]
    )


def build_generation_payload(
    context: RecallGenerationContext,
    visibility: GoldVisibility,
    *,
    allowed_actions: Collection[str],
    max_actions: int,
    historical_visibility: Literal["oracle", "blind"] | None = None,
) -> dict[str, object]:
    """Create the sole prompt-visible context, fail-closed for identifier leaks."""
    effective_visibility = _effective_visibility(visibility, historical_visibility)
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    allowed = sorted(set(allowed_actions))
    if not allowed:
        raise ValueError("at least one action type must be allowed")
    payload: dict[str, object] = {
        "query": {
            "original_query": context.original_query,
            "query_spec": context.query_spec.model_dump(mode="json"),
        },
        "seed_queries": list(context.seed_queries),
        "seed_candidates": [_safe_seed(candidate.paper.model_dump(mode="json")) for candidate in context.seed_candidates],
        "allowed_action_schema": {
            "allowed_action_types": allowed,
            "max_actions": max_actions,
            "top_level_keys": ["actions"],
        },
    }
    if effective_visibility == "oracle":
        payload["gold_documents"] = [
            {
                "title": document.title,
                "abstract": document.abstract,
                "authors": list(document.authors),
                "publication_year": document.publication_year,
                "metadata_coverage": dict(document.metadata_coverage),
            }
            for document in context.gold_documents
        ]
    assert_no_forbidden_identifier_keys_or_patterns(payload)
    return payload


def build_repair_payload(failure: ActionValidationFailure) -> dict[str, object]:
    """Keep repair context limited to the rejected output and validation facts."""
    return {
        "previous_output": failure.previous_output,
        "validation_errors": [
            {"code": issue.code, "field_path": issue.field_path} for issue in failure.issues
        ],
        "allowed_change_scope": list(failure.allowed_change_scope),
    }


class DeepSeekPromptGenerator:
    """Generate and validate recall actions, permitting precisely one repair."""

    def __init__(
        self,
        *,
        backend: LLMBackend,
        prompt: RecallPromptArtifact,
        visibility: GoldVisibility,
        allowed_actions: Collection[str],
        max_actions: int,
        historical_visibility: Literal["oracle", "blind"] | None = None,
    ) -> None:
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        if not allowed_actions:
            raise ValueError("at least one action type must be allowed")
        _effective_visibility(visibility, historical_visibility)
        self._backend = backend
        self._prompt = prompt
        self._visibility = visibility
        self._historical_visibility = historical_visibility
        self._allowed_actions = frozenset(allowed_actions)
        self._max_actions = max_actions

    @property
    def prompt_sha256(self) -> str:
        """Expose the bytes-bound prompt identity for composition/snapshot binding."""
        return self._prompt.sha256

    async def generate(self, context: RecallGenerationContext) -> GenerationResult:
        payload = build_generation_payload(
            context,
            self._visibility,
            historical_visibility=self._historical_visibility,
            allowed_actions=self._allowed_actions,
            max_actions=self._max_actions,
        )
        initial = await self._backend.generate(
            self._request(payload), "initial"
        )
        failure = _as_validation_failure(initial)
        if failure is None:
            try:
                return self._validated_result(initial, context)
            except ActionValidationFailure as error:
                failure = error
        repair = await self._backend.generate(
            self._request(build_repair_payload(failure)),
            "repair",
        )
        second_failure = _as_validation_failure(repair)
        if second_failure is not None:
            raise RecallGenerationFailure("generation_failure", list(repair.errors))
        try:
            return self._validated_result(repair, context)
        except ActionValidationFailure as error:
            raise RecallGenerationFailure("generation_failure", []) from error

    def _validated_result(
        self, backend_result: LLMBackendResult, context: RecallGenerationContext
    ) -> GenerationResult:
        output = backend_result.data
        action_batch = validate_action_batch(
            output,
            context,
            allowed_actions=self._allowed_actions,
            max_actions=self._max_actions,
        )
        return GenerationResult(
            query_id=context.query_id,
            action_batch=action_batch,
            artifact_bytes=json.dumps(
                output, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode("utf-8"),
            provenance=dict(backend_result.provenance),
        )

    def _request(self, payload: dict[str, object]) -> LLMGenerationRequest:
        return LLMGenerationRequest(
            prompt_name=self._prompt.name,
            payload=payload,
            prompt_instructions=render_recall_prompt(self._prompt),
            prompt_bytes=self._prompt.source_bytes,
            prompt_artifact_sha256=self._prompt.sha256,
        )


def _as_validation_failure(result: LLMBackendResult) -> ActionValidationFailure | None:
    if result.infrastructure_failure:
        raise RecallGenerationFailure("infrastructure_failure", list(result.errors))
    if result.errors:
        if not result.repairable:
            raise RecallGenerationFailure("infrastructure_failure", list(result.errors))
        return ActionValidationFailure(
            [ActionValidationIssue(code="invalid_json", field_path="", message="invalid JSON")],
            previous_output=result.data,
        )
    return None


def _effective_visibility(
    visibility: GoldVisibility, historical_visibility: Literal["oracle", "blind"] | None
) -> Literal["oracle", "blind"]:
    if visibility != "historical":
        if historical_visibility is not None:
            raise ValueError("historical visibility is only valid in historical mode")
        return visibility
    if historical_visibility is None:
        raise ValueError("historical generation requires recorded historical visibility")
    return historical_visibility


def _safe_seed(paper: Mapping[str, object]) -> dict[str, object]:
    """Retain useful paper prose while removing all provider/canonical identifiers."""
    safe_seed = {
        "seed_canonical_id": paper["canonical_id"],
        "title": paper["title"],
        "abstract": paper["abstract"],
        "authors": paper["authors"],
        "publication_year": paper["publication_year"],
    }
    return safe_seed


__all__ = [
    "DeepSeekPromptGenerator",
    "RecallGenerationFailure",
    "RecallPromptArtifact",
    "build_generation_payload",
    "build_repair_payload",
    "render_recall_prompt",
]
