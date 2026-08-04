from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest

from paper_search.domain.models import ErrorDetail, ProviderResult, UsageActual
from paper_search.query.parser import PlannerDependencyError, QueryParser
from paper_search.query.planner import QueryPlanner


def _provider_result(data: dict[str, object]) -> ProviderResult[dict]:
    return ProviderResult[dict](
        data=data,
        usage=UsageActual(llm_calls=1, input_tokens=10, output_tokens=10),
        provenance={
            "provider": "llm",
            "endpoint": "/chat/completions",
            "model_id": "fixture",
            "requested_at": datetime(2026, 7, 23, tzinfo=UTC).isoformat(),
            "response_hash": "sha256:fixture",
        },
        cache_hit=False,
        latency_ms=1,
        errors=[],
    )


def _valid_payload(query: str) -> dict[str, object]:
    return {
        "query_spec": {
            "original_query": query,
            "research_goal": "Find graph retrieval papers",
            "topics": ["graph retrieval"],
            "methods": [],
            "tasks": [],
            "datasets": [],
            "domains": [],
            "year_from": 2021,
            "year_to": 2024,
            "venues": ["NeurIPS"],
            "must_have": ["graph retrieval"],
            "should_have": [],
            "exclusions": ["survey"],
            "ambiguities": [],
        },
        "search_plan": {
            "subqueries": [
                {
                    "query_id": "model-1",
                    "text": "graph retrieval",
                    "query_type": "exact",
                    "target_constraints": ["graph retrieval"],
                    "priority": 1,
                    "provider_hint": "either",
                },
                {
                    "query_id": "model-2",
                    "text": "graph neural retrieval",
                    "query_type": "expanded",
                    "target_constraints": ["graph retrieval"],
                    "priority": 2,
                    "provider_hint": "openalex",
                },
                {
                    "query_id": "model-3",
                    "text": "NeurIPS graph retrieval",
                    "query_type": "decomposed",
                    "target_constraints": ["NeurIPS"],
                    "priority": 3,
                    "provider_hint": "semantic_scholar",
                },
            ],
            "inherited_hard_filters": {},
            "rationale": "fixture",
        },
    }


def test_valid_payload_uses_existing_domain_models() -> None:
    query = "graph retrieval without survey at NeurIPS from 2021 to 2024"
    parser = QueryParser(QueryPlanner())

    result = asyncio.run(parser.parse(query, _provider_result(_valid_payload(query))))

    assert result.query_spec.original_query == query
    assert result.query_spec.year_from == 2021
    assert result.search_plan.inherited_hard_filters["venues"] == ["NeurIPS"]
    assert len(result.search_plan.subqueries) == 3
    assert result.planner_status == "primary"


def _flexible_payload(query: str) -> dict[str, object]:
    return {
        "original_query": query,
        "query_spec": {
            "intent": "information_retrieval",
            "domain": "computer_vision",
            "core_concepts": [
                "motion trajectory prediction",
                "scene image conditioning",
            ],
            "constraints": {
                "content_type": "research papers",
                "output_target": "motion trajectory",
            },
            "excluded_topics": ["non-academic posts"],
        },
        "search_plan": {
            "strategy": "keyword_expansion_and_semantic_search",
            "subqueries": [
                "motion trajectory prediction conditioned on scene image",
                "visual context aware trajectory forecasting research papers",
                "scene image based motion prediction deep learning",
                "image-conditioned human motion trajectory prediction",
            ],
        },
    }


def test_flexible_model_payload_is_normalized_to_primary_analysis() -> None:
    query = "Which research papers propose motion trajectory conditioned on scene image?"

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(
            query,
            _provider_result(_flexible_payload(query)),
        )
    )

    assert result.planner_status == "primary"
    assert result.query_spec.original_query == query
    assert "motion trajectory prediction" in result.query_spec.topics
    assert result.query_spec.must_have
    assert result.query_spec.exclusions == ["non-academic posts"]
    assert len(result.search_plan.subqueries) == 4
    assert all(
        item.query_type in {"exact", "expanded", "decomposed"}
        for item in result.search_plan.subqueries
    )


def test_invalid_payload_is_repaired_once() -> None:
    query = "graph retrieval"
    calls = 0

    async def repair(_: str) -> ProviderResult[dict]:
        nonlocal calls
        calls += 1
        return _provider_result(_valid_payload(query))

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(
            query,
            _provider_result({"not": "valid"}),
            repair=repair,
        )
    )

    assert calls == 1
    assert result.query_spec.research_goal == "Find graph retrieval papers"
    assert result.planner_status == "repaired"


def test_failed_repair_uses_deterministic_rule_fallback() -> None:
    query = "graph retrieval without surveys at NeurIPS from 2021 to 2024"
    calls = 0

    async def repair(_: str) -> ProviderResult[dict]:
        nonlocal calls
        calls += 1
        return _provider_result({})

    parser = QueryParser(QueryPlanner())
    first = asyncio.run(parser.parse(query, _provider_result({}), repair=repair))
    second = asyncio.run(parser.parse(query, _provider_result({}), repair=repair))

    assert calls == 2
    assert first == second
    assert first.query_spec.year_from == 2021
    assert first.query_spec.year_to == 2024
    assert first.query_spec.venues == ["NeurIPS"]
    assert first.query_spec.exclusions == ["surveys"]
    assert 3 <= len(first.search_plan.subqueries) <= 5
    assert first.planner_status == "rules_fallback"


@pytest.mark.parametrize("code", ["timeout", "network_error", "authentication_error"])
def test_transport_or_authentication_failure_cannot_become_rules_fallback(
    code: str,
) -> None:
    failed = _provider_result({}).model_copy(
        update={
            "errors": [
                ErrorDetail(
                    code=code,
                    message="fixed safe message",
                    retryable=code != "authentication_error",
                    provider="llm",
                )
            ]
        }
    )
    repairs = 0

    async def repair(_: str) -> ProviderResult[dict]:
        nonlocal repairs
        repairs += 1
        return _provider_result(_valid_payload("graph retrieval"))

    with pytest.raises(PlannerDependencyError, match="planner dependency failure"):
        asyncio.run(
            QueryParser(QueryPlanner()).parse(
                "graph retrieval", failed, repair=repair
            )
        )
    assert repairs == 0


def test_provider_controlled_error_code_is_not_echoed() -> None:
    malicious_code = "sk-live-provider-secret"
    failed = _provider_result({}).model_copy(
        update={
            "errors": [
                ErrorDetail(
                    code=malicious_code,
                    message="fixed safe message",
                    retryable=False,
                    provider="llm",
                )
            ]
        }
    )

    with pytest.raises(PlannerDependencyError) as error:
        asyncio.run(QueryParser(QueryPlanner()).parse("graph retrieval", failed))

    assert str(error.value) == "planner dependency failure"
    assert malicious_code not in str(error.value)


def test_repair_callable_is_never_invoked_for_valid_payload() -> None:
    query = "graph retrieval"

    async def forbidden(_: str) -> ProviderResult[dict]:
        raise AssertionError("repair must not be called for valid JSON")

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(
            query,
            _provider_result(_valid_payload(query)),
            repair=forbidden,
        )
    )

    assert result.query_spec.original_query == query


@pytest.mark.parametrize(
    "query",
    [
        "面向医学影像的图神经网络检索",
        "retrieval for an unseen materials-science domain",
    ],
)
def test_rule_fallback_routes_bilingual_and_unseen_domains_without_fabrication(
    query: str,
) -> None:
    result = asyncio.run(QueryParser(QueryPlanner()).parse(query, _provider_result({})))

    assert result.query_spec.original_query == query
    assert result.query_spec.topics == [query]
    assert result.query_spec.venues == []
    assert result.query_spec.year_from is None
    assert len(result.search_plan.subqueries) == 3
    assert result.search_plan.subqueries[0].query_type == "exact"
    assert result.search_plan.subqueries[0].text == query


def test_rule_fallback_routes_simple_query_to_distinct_rewrites() -> None:
    result = asyncio.run(
        QueryParser(QueryPlanner()).parse("transformers", _provider_result({}))
    )

    texts = [subquery.text for subquery in result.search_plan.subqueries]
    assert texts[0] == "transformers"
    assert texts[1] == "transformers scholarly papers"
    assert texts[2] == "transformers methods"


Repair = Callable[[str], Awaitable[ProviderResult[dict]]]
assert Repair
