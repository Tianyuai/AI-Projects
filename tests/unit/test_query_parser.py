from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest

from paper_search.domain.models import ErrorDetail, ProviderResult, UsageActual
from paper_search.query.parser import (
    PlannerDependencyError,
    QueryParser,
    normalize_query_analysis,
    rule_fallback,
)
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


def test_semantic_action_aliases_and_plan_exclusions_are_normalized() -> None:
    query = (
        "Find molecular property prediction with graph attention networks "
        "without 3D conformers"
    )
    payload: dict[str, object] = {
        "query_spec": {
            "original_query": query,
            "research_goal": "Find the requested molecular prediction studies.",
            "subqueries": [
                {
                    "subquery": query,
                    "search_mode": "semantic",
                    "target_constraints": ["graph attention networks"],
                },
                {
                    "subquery": (
                        "graph attention networks message passing molecular "
                        "property prediction"
                    ),
                    "search_mode": "expanded",
                    "target_constraints": [
                        "graph attention networks",
                        "molecular property prediction",
                    ],
                    "provider_hint": "openalex",
                    "priority": 4,
                },
                {
                    "text": "molecular property prediction 2D graph",
                    "query_type": "decomposed",
                    "search_mode": "lexical",
                    "target_constraints": ["molecular property prediction"],
                },
            ],
        },
        "search_plan": {
            "exclusions": ["methods that require 3D conformers"],
            "rationale": "Use grounded terminology bridges.",
        },
    }

    normalized = normalize_query_analysis(payload, query)
    normalized_spec = normalized["query_spec"]
    normalized_plan = normalized["search_plan"]
    assert isinstance(normalized_spec, dict)
    assert isinstance(normalized_plan, dict)
    assert normalized_spec["exclusions"] == [
        "methods that require 3D conformers",
        "3D conformers",
    ]

    subqueries = normalized_plan["subqueries"]
    assert isinstance(subqueries, list)
    assert subqueries[0]["text"] == query
    assert subqueries[0]["query_type"] == "exact"
    assert subqueries[0]["search_mode"] == "semantic"
    assert subqueries[0]["target_constraints"] == ["graph attention networks"]
    assert subqueries[1]["query_type"] == "expanded"
    assert subqueries[1]["search_mode"] == "lexical"
    assert subqueries[1]["provider_hint"] == "openalex"
    assert subqueries[1]["priority"] == 4
    assert subqueries[2]["query_type"] == "decomposed"
    assert subqueries[2]["search_mode"] == "lexical"


def test_verbose_model_exclusion_cannot_bypass_strict_negation_abstention() -> None:
    query = (
        "Find empirical studies that predict molecular properties with graph "
        "attention networks but do not require 3D conformers."
    )
    expanded = (
        "graph attention networks molecular property prediction without 3D conformers"
    )
    alternative = (
        "graph attention networks molecular property prediction using 2D representations"
    )
    payload: dict[str, object] = {
        "query_spec": {
            "original_query": query,
            "research_goal": "Find the requested molecular property studies.",
            "subqueries": [
                {
                    "text": query,
                    "search_mode": "lexical",
                    "target_constraints": [
                        "graph attention networks",
                        "molecular properties",
                        "3D conformers",
                    ],
                },
                {
                    "text": expanded,
                    "search_mode": "expanded",
                    "target_constraints": [
                        "graph attention networks",
                        "3D conformers",
                    ],
                },
                {
                    "text": alternative,
                    "search_mode": "expanded",
                    "target_constraints": ["graph attention networks"],
                },
            ],
        },
        "search_plan": {
            "exclusions": ["methods that require 3D conformers"],
            "rationale": "Keep the negative condition as an exclusion.",
        },
    }

    result = asyncio.run(
        QueryParser(
            QueryPlanner(prompt_version="query-analyze-semantic-actions-v2")
        ).parse(query, _provider_result(payload))
    )

    assert result.planner_status == "primary"
    assert "3D conformers" in result.query_spec.exclusions
    assert expanded not in [item.text for item in result.search_plan.subqueries]
    assert alternative not in [item.text for item in result.search_plan.subqueries]


def test_strict_valid_payload_also_applies_negation_abstention() -> None:
    query = (
        "Find empirical studies that predict molecular properties with graph "
        "attention networks but do not require 3D conformers."
    )
    alternative = (
        "graph attention networks for molecular property prediction using "
        "2D representations"
    )
    payload = _valid_payload(query)
    spec = payload["query_spec"]
    plan = payload["search_plan"]
    assert isinstance(spec, dict)
    assert isinstance(plan, dict)
    spec.update(
        {
            "research_goal": "Find graph attention molecular prediction studies.",
            "topics": ["molecular property prediction"],
            "methods": ["graph attention networks"],
            "tasks": ["molecular property prediction"],
            "year_from": None,
            "year_to": None,
            "venues": [],
            "must_have": ["graph attention networks"],
            "exclusions": ["methods requiring 3D conformers"],
        }
    )
    plan["subqueries"] = [
        {
            "query_id": "model-1",
            "text": query,
            "query_type": "exact",
            "target_constraints": ["graph attention networks"],
            "priority": 1,
            "provider_hint": "either",
        },
        {
            "query_id": "model-2",
            "text": alternative,
            "query_type": "decomposed",
            "target_constraints": ["graph attention networks"],
            "priority": 2,
            "provider_hint": "either",
            "search_mode": "semantic",
        },
    ]

    result = asyncio.run(
        QueryParser(
            QueryPlanner(prompt_version="query-analyze-semantic-actions-v2")
        ).parse(query, _provider_result(payload))
    )

    assert result.planner_status == "primary"
    assert result.query_spec.exclusions == [
        "methods requiring 3D conformers",
        "3D conformers",
    ]
    assert alternative not in [item.text for item in result.search_plan.subqueries]


def test_live_method_bridge_shape_reaches_primary_semantic_gate() -> None:
    query = (
        "Which papers use weakly supervised optimal transport to adapt medical "
        "image segmentation models across hospitals?"
    )
    payload: dict[str, object] = {
        "query_spec": {
            "original_query": query,
            "research_goal": (
                "Identify papers that employ weakly supervised optimal transport "
                "for adapting medical image segmentation models across hospitals."
            ),
            "subqueries": [
                {
                    "subquery": query,
                    "search_mode": "semantic",
                    "target_constraints": [
                        "weakly supervised optimal transport",
                        "medical image segmentation",
                        "across hospitals",
                    ],
                },
                {
                    "subquery": (
                        "weakly supervised optimal transport domain adaptation "
                        "medical image segmentation"
                    ),
                    "search_mode": "expanded",
                    "target_constraints": [
                        "weakly supervised optimal transport",
                        "medical image segmentation",
                    ],
                },
                {
                    "subquery": (
                        "Wasserstein distance weakly supervised segmentation "
                        "cross-domain medical imaging"
                    ),
                    "search_mode": "expanded",
                    "target_constraints": ["weakly supervised", "segmentation"],
                },
            ],
        },
        "search_plan": {
            "exclusions": [],
            "rationale": "Preserve anchors while bridging terminology.",
        },
    }

    result = asyncio.run(
        QueryParser(
            QueryPlanner(prompt_version="query-analyze-semantic-actions-v2")
        ).parse(query, _provider_result(payload))
    )

    assert result.planner_status == "primary"
    actions = {item.text: item for item in result.search_plan.subqueries}
    bridge = (
        "weakly supervised optimal transport domain adaptation medical image "
        "segmentation"
    )
    assert bridge in actions
    assert actions[bridge].query_type == "expanded"
    assert actions[bridge].search_mode == "lexical"
    assert actions[bridge].target_constraints == [
        "weakly supervised optimal transport",
        "medical image segmentation",
    ]


def test_flexible_model_payload_with_two_subqueries_is_supplemented() -> None:
    payload = _flexible_payload("Which paper introduced the concept of dataset distillation?")
    payload["search_plan"]["subqueries"] = [
        "dataset distillation synthetic dataset",
        "distillation from large datasets efficient training",
    ]

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(
            "Which paper introduced the concept of dataset distillation?",
            _provider_result(payload),
        )
    )

    assert result.planner_status == "primary"
    assert len(result.search_plan.subqueries) >= 3
    assert any(
        item.text == "Which paper introduced the concept of dataset distillation?"
        for item in result.search_plan.subqueries
    )
    assert "research papers" in result.query_spec.must_have
    assert "research papers" in result.query_spec.topics


def test_steps_plan_and_constraint_descriptions_are_normalized_to_primary_analysis() -> None:
    query = (
        "What works introduce the feasibility of creating adversarial examples "
        "that can break LMMs?"
    )
    payload: dict[str, object] = {
        "query_spec": {
            "original_query": query,
            "constraints": [
                {
                    "type": "explicit",
                    "description": (
                        "Works must study adversarial examples against "
                        "large multimodal models."
                    ),
                }
            ],
            "subqueries": [
                "adversarial examples against large multimodal models",
                "breaking large multimodal models with adversarial examples",
            ],
        },
        "search_plan": {
            "steps": [
                {
                    "action": "search",
                    "query": "adversarial examples against large multimodal models",
                },
                {
                    "action": "search",
                    "query": (
                        "breaking large multimodal models with adversarial examples"
                    ),
                },
            ]
        },
    }

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(query, _provider_result(payload))
    )

    assert result.planner_status == "primary"
    assert any("adversarial examples" in item for item in result.query_spec.must_have)
    assert len(result.search_plan.subqueries) >= 2
    assert (
        result.search_plan.subqueries[0].text
        == "adversarial examples against large multimodal models"
    )


def test_spec_subqueries_fallback_is_normalized_to_primary_analysis() -> None:
    query = "Which papers describe a knowledge graph as a type of a heterogeneous graph?"
    payload: dict[str, object] = {
        "query_spec": {
            "original_query": query,
            "research_goal": "Find papers describing knowledge graphs as heterogeneous graphs",
            "subqueries": [
                "knowledge graph heterogeneous graph",
                "knowledge graph as heterogeneous graph papers",
            ],
        },
        "search_plan": {},
    }

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(query, _provider_result(payload))
    )

    assert result.planner_status == "primary"
    assert len(result.search_plan.subqueries) >= 2
    assert result.search_plan.subqueries[0].text == "knowledge graph heterogeneous graph"


def test_top_level_subqueries_without_query_spec_are_normalized_to_primary_analysis() -> None:
    query = (
        "What are the works that addressed the differences between individual "
        "annotators or the group-level attributes of annotators by adding "
        "individual layers?"
    )
    payload: dict[str, object] = {
        "query": query,
        "subqueries": [
            "works addressing individual annotator differences by adding individual layers",
            "works addressing group-level annotator attributes by adding individual layers",
        ],
        "search_plan": {
            "steps": [
                {
                    "action": "search",
                    "query": (
                        "works addressing individual annotator differences "
                        "by adding individual layers"
                    ),
                },
                {
                    "action": "search",
                    "query": (
                        "works addressing group-level annotator attributes "
                        "by adding individual layers"
                    ),
                },
            ]
        },
    }

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(query, _provider_result(payload))
    )

    assert result.planner_status == "primary"
    assert result.query_spec.research_goal == query
    assert len(result.search_plan.subqueries) >= 2
    assert (
        result.search_plan.subqueries[0].text
        == "works addressing individual annotator differences by adding individual layers"
    )


def test_wrapped_query_analysis_result_is_normalized_to_primary_analysis() -> None:
    query = "Which research papers propose motion trajectory conditioned on scene image?"
    payload = {"QueryAnalysisResult": _flexible_payload(query)}

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(
            query,
            _provider_result(payload),
        )
    )

    assert result.planner_status == "primary"
    assert result.query_spec.original_query == query
    assert "motion trajectory prediction" in result.query_spec.topics
    assert len(result.search_plan.subqueries) == 4


def test_split_query_spec_and_search_plan_is_normalized_to_primary_analysis() -> None:
    query = "Which research papers propose motion trajectory conditioned on scene image?"
    flexible = _flexible_payload(query)
    payload = {
        "QuerySpec": flexible["query_spec"],
        "SearchPlan": flexible["search_plan"],
    }

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(
            query,
            _provider_result(payload),
        )
    )

    assert result.planner_status == "primary"
    assert result.query_spec.original_query == query
    assert "motion trajectory prediction" in result.query_spec.topics
    assert len(result.search_plan.subqueries) == 4


def test_wrapped_pascal_case_analysis_is_normalized_to_primary_analysis() -> None:
    query = "Which research papers propose motion trajectory conditioned on scene image?"
    flexible = _flexible_payload(query)
    payload = {
        "QueryAnalysisResult": {
            "QuerySpec": flexible["query_spec"],
            "SearchPlan": flexible["search_plan"],
            "original_query": query,
        }
    }

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(
            query,
            _provider_result(payload),
        )
    )

    assert result.planner_status == "primary"
    assert result.query_spec.original_query == query
    assert "motion trajectory prediction" in result.query_spec.topics
    assert len(result.search_plan.subqueries) == 4


def test_wrapped_snake_case_analysis_is_normalized_to_primary_analysis() -> None:
    query = "Which research papers propose motion trajectory conditioned on scene image?"
    flexible = _flexible_payload(query)
    payload = {
        "query_analysis_result": {
            "original_query": query,
            "query_spec": flexible["query_spec"],
            "search_plan": flexible["search_plan"],
        }
    }

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(
            query,
            _provider_result(payload),
        )
    )

    assert result.planner_status == "primary"
    assert result.query_spec.original_query == query
    assert "motion trajectory prediction" in result.query_spec.topics
    assert len(result.search_plan.subqueries) == 4


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


def test_flexible_repair_payload_is_normalized_before_fallback() -> None:
    query = "graph retrieval"
    repaired_payload = {
        "query_spec": {
            "original_query": query,
            "search_terms": ["graph retrieval", "graph search"],
            "constraints": [],
            "filters": {},
            "additional_context": "research papers",
        },
        "search_plan": {
            "subqueries": [
                "graph retrieval",
                "graph search",
                "graph information retrieval",
            ]
        },
    }

    async def repair(_: str) -> ProviderResult[dict]:
        return _provider_result(repaired_payload)

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(
            query,
            _provider_result({}),
            repair=repair,
        )
    )

    assert result.planner_status == "repaired"
    assert len(result.search_plan.subqueries) == 3


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


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Which retrieval systems work without dense encoders?", "dense encoders"),
        ("Which retrieval systems work excluding dense encoders?", "dense encoders"),
    ],
)
def test_rule_fallback_normalizes_explicit_exclusion_boundaries(
    query: str, expected: str
) -> None:
    spec = rule_fallback(query)

    assert spec.exclusions == [expected]


@pytest.mark.parametrize(
    "query",
    [
        "What are the sources that mentioned the Algonauts 2023 challenge?",
        "Which papers performed best on the COCO 2017 dataset?",
        "Which studies found errors in the CoNLL 2003 benchmark?",
    ],
)
def test_rule_fallback_does_not_treat_entity_years_as_publication_filters(
    query: str,
) -> None:
    spec = rule_fallback(query)

    assert spec.year_from is None
    assert spec.year_to is None


@pytest.mark.parametrize(
    ("query", "year_from", "year_to"),
    [
        ("a comprehensive review of the field up to 2020", None, 2020),
        ("recent advances in reinforcement learning since 2021", 2021, None),
    ],
)
def test_rule_fallback_preserves_explicit_temporal_year_direction(
    query: str,
    year_from: int | None,
    year_to: int | None,
) -> None:
    spec = rule_fallback(query)

    assert spec.year_from == year_from
    assert spec.year_to == year_to


def test_valid_analysis_reconciles_explicit_runtime_constraints() -> None:
    query = (
        "Find empirical papers published since 2021 that apply "
        "retrieval-augmented generation to scientific question answering, "
        "excluding surveys and review articles."
    )
    payload = _valid_payload(query)
    spec = payload["query_spec"]
    assert isinstance(spec, dict)
    spec.update(
        {
            "research_goal": "Find empirical scientific question answering papers.",
            "topics": ["scientific question answering"],
            "methods": [],
            "tasks": [],
            "year_from": None,
            "year_to": None,
            "exclusions": [],
        }
    )

    result = asyncio.run(
        QueryParser(QueryPlanner()).parse(query, _provider_result(payload))
    )

    assert result.query_spec.year_from == 2021
    assert result.query_spec.year_to is None
    assert result.query_spec.methods == ["retrieval-augmented generation"]
    assert result.query_spec.tasks == ["scientific question answering"]
    exclusions = {value.casefold() for value in result.query_spec.exclusions}
    assert {"surveys", "review articles"}.issubset(exclusions)


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
