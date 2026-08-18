from __future__ import annotations

import pytest

from paper_search.learning.contracts import PolicyActionCandidate, QueryPolicyInput
from paper_search.learning.policy import BoundedQueryPolicy
from paper_search.learning.routing import RuleQueryRouter


class MappingScorer:
    model_id = "fake-action-ranker-v1"

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(
        self,
        request: QueryPolicyInput,
        candidates: list[PolicyActionCandidate],
    ) -> list[float]:
        del request
        return [self._scores[candidate.text] for candidate in candidates]


def test_router_separates_navigational_metadata_and_semantic_queries() -> None:
    router = RuleQueryRouter()

    navigational = router.route("find doi:10.1000/example")
    metadata = router.route("ACL 2024 papers that cite the transformers paper")
    semantic = router.route("methods for robust graph retrieval under distribution shift")

    assert navigational.query_kind == "navigational"
    assert metadata.query_kind == "metadata"
    assert metadata.query_spec.year_from == 2024
    assert metadata.query_spec.year_to == 2024
    assert metadata.query_spec.venues == ["ACL"]
    assert semantic.query_kind == "semantic"


def test_policy_ranks_bounded_candidates_and_always_retains_original_query() -> None:
    routed = RuleQueryRouter().route("graph retrieval papers")
    request = QueryPolicyInput(
        query_id="q1",
        original_query="graph retrieval papers",
        query_kind=routed.query_kind,
        query_spec=routed.query_spec,
        seed_actions=[
            PolicyActionCandidate(
                action_id="seed-1",
                action_type="text_search",
                text="neural graph retrieval",
                origin="seed_query",
                provider_hint="either",
            ),
            PolicyActionCandidate(
                action_id="seed-2",
                action_type="text_search",
                text="scientific document retrieval",
                origin="seed_query",
                provider_hint="openalex",
            ),
        ],
        allowed_action_types=["text_search"],
        max_actions=2,
    )
    policy = BoundedQueryPolicy(
        MappingScorer(
            {
                "graph retrieval papers": 0.1,
                "neural graph retrieval": 0.9,
                "scientific document retrieval": 0.8,
            }
        ),
        confidence_threshold=0.6,
    )

    output = policy.plan(request)

    assert len(output.ranked_actions) == 2
    assert output.ranked_actions[0].action.text == "neural graph retrieval"
    assert any(
        ranked.action.origin == "original_query"
        and ranked.action.text == "graph retrieval papers"
        for ranked in output.ranked_actions
    )
    assert output.confidence == 0.9
    assert output.fallback_required is False


def test_policy_requests_fallback_when_local_confidence_is_low() -> None:
    routed = RuleQueryRouter().route("graph retrieval papers")
    request = QueryPolicyInput(
        query_id="q1",
        original_query="graph retrieval papers",
        query_kind=routed.query_kind,
        query_spec=routed.query_spec,
        allowed_action_types=["text_search"],
        max_actions=3,
    )
    policy = BoundedQueryPolicy(
        MappingScorer({"graph retrieval papers": 0.49}),
        confidence_threshold=0.5,
    )

    output = policy.plan(request)

    assert output.fallback_required is True
    assert output.fallback_reason == "confidence_below_threshold"


def test_policy_keeps_lexical_and_semantic_actions_with_identical_text() -> None:
    query = "graph diffusion retrieval"
    routed = RuleQueryRouter().route(query)
    semantic = PolicyActionCandidate(
        action_id="semantic",
        action_type="text_search",
        text=query,
        origin="deterministic_rule",
        provider_hint="openalex",
        search_mode="semantic",
    )

    class ModeScorer:
        model_id = "mode-scorer"

        def score(self, request, candidates):
            del request
            return [
                0.9 if candidate.search_mode == "semantic" else 0.1
                for candidate in candidates
            ]

    output = BoundedQueryPolicy(
        ModeScorer(), confidence_threshold=0.0
    ).plan(
        QueryPolicyInput(
            query_id="q-mode",
            original_query=query,
            query_kind=routed.query_kind,
            query_spec=routed.query_spec,
            seed_actions=[semantic],
            allowed_action_types=["text_search"],
            max_actions=2,
        )
    )

    assert [item.action.search_mode for item in output.ranked_actions] == [
        "semantic",
        "lexical",
    ]


def test_policy_nfkc_normalizes_and_deduplicates_original_anchor() -> None:
    raw_query = "Find graph ℒ papers"
    canonical_query = "Find graph L papers"
    routed = RuleQueryRouter().route(raw_query)
    request = QueryPolicyInput(
        query_id="q-unicode",
        original_query=raw_query,
        query_kind=routed.query_kind,
        query_spec=routed.query_spec,
        seed_actions=[
            PolicyActionCandidate(
                action_id="canonical-anchor",
                action_type="text_search",
                text=canonical_query,
                origin="original_query",
                provider_hint="openalex",
            )
        ],
        allowed_action_types=["text_search"],
        max_actions=3,
    )

    output = BoundedQueryPolicy(
        MappingScorer({canonical_query: 1.0}),
        confidence_threshold=0.0,
    ).plan(request)

    assert len(output.ranked_actions) == 1
    assert output.ranked_actions[0].action.text == canonical_query


def test_policy_fallback_confidence_uses_non_anchor_actions_when_available() -> None:
    routed = RuleQueryRouter().route("graph retrieval papers")
    request = QueryPolicyInput(
        query_id="q1",
        original_query="graph retrieval papers",
        query_kind=routed.query_kind,
        query_spec=routed.query_spec,
        seed_actions=[
            PolicyActionCandidate(
                action_id="seed-1",
                action_type="text_search",
                text="weak expansion",
                origin="deterministic_rule",
                provider_hint="either",
            )
        ],
        allowed_action_types=["text_search"],
        max_actions=3,
    )
    policy = BoundedQueryPolicy(
        MappingScorer(
            {
                "graph retrieval papers": 0.99,
                "weak expansion": 0.2,
            }
        ),
        confidence_threshold=0.5,
    )

    output = policy.plan(request)

    assert output.confidence == 0.2
    assert output.fallback_required is True


def test_policy_rejects_ranker_score_count_mismatch() -> None:
    class BrokenScorer(MappingScorer):
        def score(
            self,
            request: QueryPolicyInput,
            candidates: list[PolicyActionCandidate],
        ) -> list[float]:
            del request, candidates
            return []

    routed = RuleQueryRouter().route("graph retrieval papers")
    request = QueryPolicyInput(
        query_id="q1",
        original_query="graph retrieval papers",
        query_kind=routed.query_kind,
        query_spec=routed.query_spec,
        allowed_action_types=["text_search"],
        max_actions=3,
    )

    with pytest.raises(ValueError, match="one score per candidate"):
        BoundedQueryPolicy(BrokenScorer({}), confidence_threshold=0.5).plan(request)
