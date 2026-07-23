from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from paper_search.domain.models import ProviderResult, UsageActual
from paper_search.query.parser import QueryParser
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


Repair = Callable[[str], Awaitable[ProviderResult[dict]]]
assert Repair
