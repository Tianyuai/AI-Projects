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
