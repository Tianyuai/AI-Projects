from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from paper_search.llm.prompt_artifacts import (
    PromptArtifact,
    load_prompt_artifact,
    render_prompt_system_message,
)


def test_load_prompt_artifact_validates_query_analyze_file() -> None:
    artifact = load_prompt_artifact(Path("configs/prompts/query_analyze.yaml").read_bytes())

    assert artifact == PromptArtifact(
        name="query_analyze",
        version="query-analyze-v1",
        temperature=0,
        response_model="QueryAnalysisResult",
        instructions=[
            "Preserve the original query and every explicit hard constraint.",
            "Return one QuerySpec and one SearchPlan as a JSON object.",
            "Generate three to five targeted subqueries.",
            "Do not infer facts that are not stated by the user.",
        ],
    )


def test_render_prompt_system_message_is_deterministic_for_supported_prompt() -> None:
    analyze_message = render_prompt_system_message(
        Path("configs/prompts/query_analyze.yaml").read_bytes()
    )
    assert analyze_message == "\n".join(
        [
            "Respond with a JSON object.",
            "The JSON object must match the QueryAnalysisResult contract.",
            "- Preserve the original query and every explicit hard constraint.",
            "- Return one QuerySpec and one SearchPlan as a JSON object.",
            "- Generate three to five targeted subqueries.",
            "- Do not infer facts that are not stated by the user.",
        ]
    )


def test_semantic_action_prompt_requests_meaningful_grounded_subqueries() -> None:
    artifact = load_prompt_artifact(
        Path("configs/prompts/query_analyze_semantic_actions_v2.yaml").read_bytes()
    )

    assert artifact.version == "query-analyze-semantic-actions-v2"
    instructions = " ".join(artifact.instructions).casefold()
    assert "retrieval hypothesis" in instructions
    assert "hard constraints" in instructions
    assert "soft conceptual terms" in instructions
    assert "frozen supervised lexical bridge" in instructions
    assert "at least two original concept anchors" in instructions
    assert "literature-side" in instructions
    assert "doi" in instructions
    assert "target_constraints" in instructions
    assert '"query_spec"' in instructions
    assert '"search_plan"' in instructions
    assert '"subqueries"' in instructions
    assert '"query_id"' in instructions
    assert '"text"' in instructions
    assert '"query_type"' in instructions
    assert '"provider_hint"' in instructions
    assert '"search_mode"' in instructions
    assert "never place literal double quotation marks inside string values" in instructions
    assert "use an empty ambiguities list" in instructions


def test_protected_action_prompt_requests_one_bounded_mixed_action_bank() -> None:
    artifact = load_prompt_artifact(
        Path("configs/prompts/query_analyze_protected_actions_v3.yaml").read_bytes()
    )

    assert artifact.version == "query-analyze-protected-actions-v3"
    instructions = " ".join(artifact.instructions).casefold()
    assert "three or four independent lexical" in instructions
    assert "at most one semantic challenger" in instructions
    assert "do not add the original query as a title-search fallback" in instructions
    assert "negation" in instructions
    assert "doi" in instructions
    assert '"query_spec"' in instructions
    assert '"search_plan"' in instructions


def test_load_prompt_artifact_rejects_malformed_yaml() -> None:
    with pytest.raises(ValueError, match="invalid prompt artifact"):
        load_prompt_artifact(b"name: [")


def test_load_prompt_artifact_rejects_wrong_field_types() -> None:
    with pytest.raises(ValidationError):
        load_prompt_artifact(
            b"""
name: query_analyze
version: query-analyze-v1
temperature: 0
response_model: QueryAnalysisResult
instructions: preserve the original query
"""
        )
