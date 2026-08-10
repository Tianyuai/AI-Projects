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


def test_render_prompt_system_message_is_deterministic_for_existing_prompt_files() -> None:
    analyze_message = render_prompt_system_message(
        Path("configs/prompts/query_analyze.yaml").read_bytes()
    )
    evolve_message = render_prompt_system_message(
        Path("configs/prompts/query_evolve.yaml").read_bytes()
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
    assert evolve_message == "\n".join(
        [
            "Respond with a JSON object.",
            "The JSON object must match the QueryEvolutionProposal contract.",
            "- Use only facts and facets present in the payload.",
            "- Do not infer gold papers, relevance labels, new venues, new years, or unrelated entities.",
            "- Return zero to two complementary OpenAlex queries as strict JSON.",
            "- Before returning, verify that each generated text differs after normalization from original_query, the text of every seed_subqueries item, and earlier generated subqueries.",
            "- Case-only, NFKC-equivalent, whitespace-only, or ?/*-only changes do not make a query novel.",
            "- Return only valid novel subqueries; one valid subquery is allowed and duplicate candidates must be omitted.",
            '- No-novel form: {"subqueries":[],"no_op_reason":"no_novel_query"}',
            '- Generated form: {"subqueries":[{"text":"string","source_facets":["exact payload facet"],"strategy":"synonym"}],"no_op_reason":null}',
            '- No-op form: {"subqueries":[],"no_op_reason":"insufficient_grounded_facets"}',
            "- Top-level keys must be exactly subqueries and no_op_reason.",
            "- Each subquery must contain exactly text, source_facets, and strategy.",
            "- Allowed strategy values are synonym, entity_alias, facet_combination, and task_decomposition.",
            "- Allowed no_op_reason values are insufficient_grounded_facets and no_novel_query.",
            "- Do not return payload or prompt_name wrappers, Markdown, or extra fields.",
            "- Always include no_op_reason; use null when subqueries contains items.",
            "- Copy every source_facets value exactly from the payload.",
        ]
    )


def test_query_evolve_artifact_requires_v2() -> None:
    prompt_bytes = Path("configs/prompts/query_evolve.yaml").read_bytes()
    artifact = load_prompt_artifact(prompt_bytes)

    assert artifact.version == "query-evolve-v2"
    with pytest.raises(
        ValidationError, match="query_evolve prompt version must be query-evolve-v2"
    ):
        load_prompt_artifact(
            prompt_bytes.replace(b"version: query-evolve-v2", b"version: query-evolve-v1")
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


def test_query_evolve_message_contains_complete_contract() -> None:
    message = render_prompt_system_message(
        Path("configs/prompts/query_evolve.yaml").read_bytes()
    )

    assert '"subqueries"' in message
    assert '"text"' in message
    assert '"source_facets"' in message
    assert '"strategy"' in message
    assert '"no_op_reason"' in message
    for value in (
        "synonym",
        "entity_alias",
        "facet_combination",
        "task_decomposition",
        "insufficient_grounded_facets",
        "no_novel_query",
    ):
        assert value in message
    assert "Do not return payload or prompt_name wrappers" in message
