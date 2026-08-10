from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from paper_search.domain.models import (
    BudgetReservation,
    ErrorDetail,
    ProviderResult,
    QuerySpec,
    SearchPlan,
    UsageActual,
    UsageEstimate,
)
from paper_search.evolution.query_evolution import (
    QueryEvolutionGenerator,
    build_query_evolution_context,
    validate_query_evolution_proposal,
)


def _reservation() -> BudgetReservation:
    return BudgetReservation(
        reservation_id="reservation-1",
        action="query-evolve",
        reserved=UsageEstimate(llm_calls=1),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )


def _plan() -> SearchPlan:
    return SearchPlan(
        subqueries=[
            {
                "query_id": "sq-1",
                "text": "graph neural networks",
                "query_type": "expanded",
                "target_constraints": ["Node Classification", "GraphSAGE"],
                "priority": 1,
                "provider_hint": "openalex",
            },
            {
                "query_id": "sq-2",
                "text": "benchmarking gnn methods",
                "query_type": "decomposed",
                "target_constraints": ["ogbn-arxiv", "  GraphSAGE  "],
                "priority": 2,
                "provider_hint": "either",
            },
            {
                "query_id": "sq-3",
                "text": "message passing on citation graphs",
                "query_type": "exact",
                "target_constraints": [],
                "priority": 3,
                "provider_hint": "semantic_scholar",
            },
        ],
        inherited_hard_filters={"year_from": 2020, "year_to": 2024},
        rationale="bounded probe fixture",
    )


def _spec() -> QuerySpec:
    return QuerySpec(
        original_query="  GNNs for node classification  ",
        research_goal="Find Graph Neural Network papers for node classification",
        topics=["Graph Neural Networks", "graph neural networks"],
        methods=["GraphSAGE", "ＧＣＮ"],
        tasks=["Node Classification"],
        datasets=["ogbn-arxiv", "OGBN-ARXIV"],
        domains=["citation networks"],
        year_from=2020,
        year_to=2024,
        venues=["NeurIPS"],
        must_have=["inductive learning"],
        should_have=["scalability"],
    )


def _llm_provenance() -> dict[str, str]:
    return {
        "provider": "llm",
        "endpoint": "/chat/completions",
        "model_id": "fixture",
        "requested_at": datetime(2026, 8, 10, tzinfo=UTC).isoformat(),
        "response_hash": "sha256:" + "1" * 64,
        "snapshot_entry_id": "llm-1",
        "snapshot_cache_key": "sha256:" + "2" * 64,
        "snapshot_response_sha256": "sha256:" + "3" * 64,
        "snapshot_path": "snapshots/llm-1.json",
    }


def _context() -> Any:
    return build_query_evolution_context(
        _spec(),
        _plan(),
        candidate_count=17,
        top_titles=[
            "  GraphSAGE: Inductive Representation Learning on Large Graphs  ",
            "ＧＣＮ for Node Classification",
            "graphsage: inductive representation learning on large graphs",
            "Scalable GNNs for Node Classification",
            "One",
            "Two",
            "Three",
            "Four",
            "Five",
            "Six",
            "Seven",
            "Eight",
        ],
    )


def _collect_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(key)
            keys.update(_collect_keys(nested))
    elif isinstance(value, list):
        for item in value:
            keys.update(_collect_keys(item))
    return keys


class FakeAnalyzer:
    def __init__(
        self,
        data: dict[str, object],
        *,
        errors: list[str] | None = None,
    ) -> None:
        self.data = data
        self.errors = errors or []
        self.calls: list[tuple[str, dict[str, object], object]] = []

    async def generate_json(
        self,
        *,
        prompt_name: str,
        payload: dict[str, object],
        reservation: object,
    ) -> ProviderResult[dict[str, Any]]:
        self.calls.append((prompt_name, payload, reservation))
        return ProviderResult[dict[str, Any]](
            data=self.data,
            usage=UsageActual(llm_calls=1, input_tokens=11, output_tokens=7),
            provenance=_llm_provenance(),
            cache_hit=False,
            latency_ms=2,
            errors=[
                ErrorDetail(
                    code=code,
                    message="synthetic",
                    retryable=False,
                    provider="llm",
                )
                for code in self.errors
            ],
        )


def test_build_query_evolution_context_is_deterministic_and_scrubs_payload() -> None:
    context = _context()

    assert context.candidate_count == 17
    assert context.top_titles == [
        "GraphSAGE: Inductive Representation Learning on Large Graphs",
        "GCN for Node Classification",
        "Scalable GNNs for Node Classification",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Seven",
    ]
    assert context.facets == [
        "GNNs for node classification",
        "Find Graph Neural Network papers for node classification",
        "Graph Neural Networks",
        "GraphSAGE",
        "GCN",
        "Node Classification",
        "ogbn-arxiv",
        "citation networks",
        "NeurIPS",
        "inductive learning",
        "scalability",
    ]
    payload = context.model_dump(mode="json")
    keys = _collect_keys(payload)
    assert "inherited_hard_filters" not in keys
    assert "query_id" not in keys
    assert not {key for key in keys if "gold" in key.casefold()}
    assert not {key for key in keys if "label" in key.casefold()}


def test_build_query_evolution_context_supports_empty_constraint_specs() -> None:
    context = build_query_evolution_context(
        QuerySpec(original_query="rag", research_goal="find retrieval papers"),
        SearchPlan(
            subqueries=[
                {
                    "query_id": "q1",
                    "text": "retrieval augmented generation",
                    "query_type": "expanded",
                    "priority": 1,
                    "provider_hint": "either",
                },
                {
                    "query_id": "q2",
                    "text": "dense retrieval",
                    "query_type": "expanded",
                    "priority": 2,
                    "provider_hint": "either",
                },
                {
                    "query_id": "q3",
                    "text": "document ranking",
                    "query_type": "exact",
                    "priority": 3,
                    "provider_hint": "either",
                },
            ],
            inherited_hard_filters={"year_from": 2023},
            rationale="fixture",
        ),
        candidate_count=0,
        top_titles=[],
    )

    assert context.facets == ["rag", "find retrieval papers"]
    assert context.seed_subqueries[0].target_constraints == []
    assert context.top_titles == []


def test_validate_query_evolution_proposal_accepts_generated_and_no_op_forms() -> None:
    context = _context()

    generated = validate_query_evolution_proposal(
        {
            "subqueries": [
                {
                    "text": "graph neural networks for inductive node classification",
                    "source_facets": ["Graph Neural Networks", "inductive learning"],
                    "strategy": "facet_combination",
                }
            ],
            "no_op_reason": None,
        },
        context,
    )
    no_op = validate_query_evolution_proposal(
        {
            "subqueries": [],
            "no_op_reason": "no_novel_query",
        },
        context,
    )

    assert generated.no_op_reason is None
    assert no_op.no_op_reason == "no_novel_query"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (
            {
                "subqueries": [
                    {
                        "text": "one",
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "synonym",
                    },
                    {
                        "text": "two",
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "synonym",
                    },
                    {
                        "text": "three",
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "synonym",
                    },
                ],
                "no_op_reason": None,
            },
            "at most 2",
        ),
        (
            {
                "subqueries": [],
                "no_op_reason": None,
            },
            "no_op_reason must exist exactly for an empty proposal",
        ),
        (
            {
                "subqueries": [
                    {
                        "text": "query",
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "synonym",
                    }
                ],
                "no_op_reason": "no_novel_query",
            },
            "no_op_reason must exist exactly for an empty proposal",
        ),
        (
            {
                "subqueries": [],
                "no_op_reason": "no_novel_query",
                "extra": True,
            },
            "Extra inputs are not permitted",
        ),
        (
            {
                "subqueries": [],
            },
            "Field required",
        ),
    ],
)
def test_validate_query_evolution_proposal_enforces_strict_schema(
    raw: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_query_evolution_proposal(raw, _context())


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (
            {
                "subqueries": [
                    {
                        "text": "*",
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "synonym",
                    }
                ],
                "no_op_reason": None,
            },
            "must not be empty after normalization",
        ),
        (
            {
                "subqueries": [
                    {
                        "text": "foo?",
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "synonym",
                    },
                    {
                        "text": "foo",
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "entity_alias",
                    },
                ],
                "no_op_reason": None,
            },
            "duplicate subquery text",
        ),
        (
            {
                "subqueries": [
                    {
                        "text": "graph neural networks\u0007",
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "synonym",
                    }
                ],
                "no_op_reason": None,
            },
            "contains control characters",
        ),
        (
            {
                "subqueries": [
                    {
                        "text": "a" * 301,
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "synonym",
                    }
                ],
                "no_op_reason": None,
            },
            "must be 300 characters or fewer",
        ),
        (
            {
                "subqueries": [
                    {
                        "text": "graph neural networks in 2019",
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "synonym",
                    }
                ],
                "no_op_reason": None,
            },
            "conflicts with query year constraints",
        ),
        (
            {
                "subqueries": [
                    {
                        "text": "graph neural networks for inductive node classification",
                        "source_facets": ["made up facet"],
                        "strategy": "synonym",
                    }
                ],
                "no_op_reason": None,
            },
            "source_facets must come from context facets",
        ),
    ],
)
def test_validate_query_evolution_proposal_rejects_mechanical_violations(
    raw: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_query_evolution_proposal(raw, _context())


@pytest.mark.parametrize(
    "text",
    [
        "ＧＮＮｓ for node classification",
        "graph neural networks",
    ],
    ids=["original-query-after-nfkc", "seed-query"],
)
def test_validate_query_evolution_proposal_rejects_existing_queries(
    text: str,
) -> None:
    with pytest.raises(
        ValueError, match="duplicate subquery text after canonicalization"
    ):
        validate_query_evolution_proposal(
            {
                "subqueries": [
                    {
                        "text": text,
                        "source_facets": ["Graph Neural Networks"],
                        "strategy": "synonym",
                    }
                ],
                "no_op_reason": None,
            },
            _context(),
        )


def test_query_evolution_generator_returns_generated_result_and_snapshot_refs() -> None:
    analyzer = FakeAnalyzer(
        {
            "subqueries": [
                {
                    "text": "graph neural networks for inductive node classification",
                    "source_facets": ["Graph Neural Networks", "inductive learning"],
                    "strategy": "facet_combination",
                }
            ],
            "no_op_reason": None,
        }
    )
    context = _context()

    result = asyncio.run(
        QueryEvolutionGenerator(analyzer=analyzer).generate(context, _reservation())
    )

    assert result.status == "generated"
    assert result.proposal is not None
    assert result.proposal.subqueries[0].text == (
        "graph neural networks for inductive node classification"
    )
    assert len(result.snapshot_refs) == 1
    assert result.snapshot_refs[0].entry_id == "llm-1"
    assert result.diagnostics[0].dependency == "llm"
    prompt_name, payload, _ = analyzer.calls[0]
    assert prompt_name == "query_evolve"
    assert payload == context.model_dump(mode="json")


def test_query_evolution_generator_classifies_no_op_dependency_and_integrity_failures() -> None:
    context = _context()
    no_op = asyncio.run(
        QueryEvolutionGenerator(
            analyzer=FakeAnalyzer(
                {"subqueries": [], "no_op_reason": "insufficient_grounded_facets"}
            )
        ).generate(context, _reservation())
    )
    dependency_failure = asyncio.run(
        QueryEvolutionGenerator(
            analyzer=FakeAnalyzer(
                {"subqueries": [], "no_op_reason": "no_novel_query"},
                errors=["provider_error"],
            )
        ).generate(context, _reservation())
    )
    integrity_failure = asyncio.run(
        QueryEvolutionGenerator(
            analyzer=FakeAnalyzer(
                {
                    "subqueries": [
                        {
                            "text": "graph neural networks for node classification",
                            "source_facets": ["not in context"],
                            "strategy": "synonym",
                        }
                    ],
                    "no_op_reason": None,
                }
            )
        ).generate(context, _reservation())
    )

    assert no_op.status == "no_op"
    assert no_op.proposal is not None
    assert dependency_failure.status == "dependency_failure"
    assert dependency_failure.proposal is None
    assert integrity_failure.status == "integrity_failure"
    assert integrity_failure.proposal is None
    assert integrity_failure.snapshot_refs[0].entry_id == "llm-1"
