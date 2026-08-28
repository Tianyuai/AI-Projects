from __future__ import annotations

from paper_search.domain.models import QuerySpec, SearchPlan, SubQuery
from paper_search.learning.adaptive_openalex_recall import OpenAlexRecallDecision
from paper_search.query.low_confidence_supplement import (
    select_low_confidence_llm_action,
)


def _action(query_id: str, text: str, priority: int) -> SubQuery:
    return SubQuery(
        query_id=query_id,
        text=text,
        query_type="expanded",
        action_type="text_search",
        target_constraints=["adaptive rounding"],
        priority=priority,
        provider_hint="either",
        search_mode="lexical",
    )


def _decision(low_confidence: bool) -> OpenAlexRecallDecision:
    return OpenAlexRecallDecision(
        low_confidence=low_confidence,
        reason_codes=["low_query_alignment"] if low_confidence else [],
        candidate_count=100,
        action_count=5,
        cross_action_supported_candidate_count=20,
        required_facet_count=0,
        covered_facet_count=0,
        query_aligned_candidate_count=3,
        minimum_query_aligned_candidate_count=13,
    )


def test_selects_one_novel_finalized_llm_action_without_mutating_production() -> None:
    spec = QuerySpec(
        original_query="Find adaptive rounding per-neuron quantization papers",
        research_goal="Find adaptive rounding per-neuron quantization papers",
    )
    production = SearchPlan(
        subqueries=[
            _action("p-1", "adaptive rounding per-neuron quantization", 1),
            _action("p-2", "per-neuron quantization adaptive rounding", 2),
        ],
        inherited_hard_filters={},
        rationale="production",
    )
    candidate = SearchPlan(
        subqueries=[
            _action("c-1", "adaptive rounding per-neuron quantization", 1),
            _action(
                "c-2",
                "AdaRound greedy neuron-wise weight quantization",
                2,
            ),
        ],
        inherited_hard_filters={},
        rationale="candidate",
    )

    selected = select_low_confidence_llm_action(
        spec,
        production,
        candidate,
        _decision(True),
    )

    assert selected is not None
    assert selected.source_query_id == "c-2"
    assert selected.action.text == "AdaRound greedy neuron-wise weight quantization"
    assert [item.query_id for item in production.subqueries] == ["p-1", "p-2"]


def test_abstains_for_adequate_or_negation_query() -> None:
    production = SearchPlan(
        subqueries=[_action("p-1", "adaptive rounding quantization", 1)],
        inherited_hard_filters={},
        rationale="production",
    )
    candidate = SearchPlan(
        subqueries=[_action("c-1", "AdaRound neuron-wise quantization", 1)],
        inherited_hard_filters={},
        rationale="candidate",
    )
    adequate = QuerySpec(
        original_query="Find adaptive rounding quantization papers",
        research_goal="Find adaptive rounding quantization papers",
    )
    negation = adequate.model_copy(update={"exclusions": ["post-training"]})

    assert (
        select_low_confidence_llm_action(
            adequate,
            production,
            candidate,
            _decision(False),
        )
        is None
    )
    assert (
        select_low_confidence_llm_action(
            negation,
            production,
            candidate,
            _decision(True),
        )
        is None
    )


def test_openalex_low_confidence_prefers_independent_s2_action() -> None:
    spec = QuerySpec(
        original_query="Find adaptive rounding per-neuron quantization papers",
        research_goal="Find adaptive rounding per-neuron quantization papers",
    )
    production = SearchPlan(
        subqueries=[_action("p-1", "adaptive rounding quantization", 1)],
        inherited_hard_filters={},
        rationale="production",
    )
    broad_openalex = _action(
        "c-openalex",
        (
            "adaptive rounding quantization neural network weight compression "
            "calibration Hessian optimization"
        ),
        1,
    )
    s2 = _action(
        "c-s2",
        "AdaRound neuron-wise quantization",
        2,
    ).model_copy(update={"provider_hint": "semantic_scholar"})
    candidate = SearchPlan(
        subqueries=[broad_openalex, s2],
        inherited_hard_filters={},
        rationale="candidate",
    )

    selected = select_low_confidence_llm_action(
        spec,
        production,
        candidate,
        _decision(True),
    )

    assert selected is not None
    assert selected.source_query_id == "c-s2"


def test_revalidates_candidate_action_against_production_hard_constraints() -> None:
    spec = QuerySpec(
        original_query="Find AdaRound per-neuron quantization papers",
        research_goal="Find AdaRound per-neuron quantization papers",
        methods=["AdaRound"],
    )
    production = SearchPlan(
        subqueries=[_action("p-1", "AdaRound per-neuron quantization", 1)],
        inherited_hard_filters={},
        rationale="production",
    )
    candidate = SearchPlan(
        subqueries=[_action("c-1", "neuron-wise weight quantization", 1)],
        inherited_hard_filters={},
        rationale="candidate omitted the explicit method",
    )

    selected = select_low_confidence_llm_action(
        spec,
        production,
        candidate,
        _decision(True),
    )

    assert selected is None
