from paper_search.domain.models import QuerySpec, SearchPlan, SubQuery
from paper_search.query.planner import QueryPlanner


def _spec() -> QuerySpec:
    return QuerySpec(
        original_query="graph retrieval without surveys at NeurIPS from 2021 to 2024",
        research_goal="Find graph retrieval methods",
        topics=["graph retrieval"],
        methods=["message passing"],
        datasets=["OpenGraph"],
        year_from=2021,
        year_to=2024,
        venues=["NeurIPS"],
        must_have=["graph retrieval"],
        exclusions=["surveys"],
    )


def test_finalize_is_deterministic_clipped_and_inherits_hard_constraints() -> None:
    source = SearchPlan(
        subqueries=[
            SubQuery(
                query_id="z",
                text="graph retrieval",
                query_type="exact",
                target_constraints=["graph retrieval"],
                priority=2,
                provider_hint="either",
            ),
            SubQuery(
                query_id="a",
                text="message passing OpenGraph",
                query_type="decomposed",
                target_constraints=["message passing", "OpenGraph"],
                priority=1,
                provider_hint="semantic_scholar",
            ),
            SubQuery(
                query_id="duplicate",
                text=" GRAPH   RETRIEVAL ",
                query_type="expanded",
                target_constraints=["graph retrieval"],
                priority=3,
                provider_hint="openalex",
            ),
            SubQuery(
                query_id="extra-1",
                text="graph neural search",
                query_type="expanded",
                target_constraints=["graph retrieval"],
                priority=4,
                provider_hint="either",
            ),
            SubQuery(
                query_id="extra-2",
                text="scientific graph search",
                query_type="expanded",
                target_constraints=["graph retrieval"],
                priority=5,
                provider_hint="either",
            ),
        ],
        inherited_hard_filters={"year_from": 1900},
        rationale="fixture",
    )
    planner = QueryPlanner()

    first = planner.finalize(_spec(), source, max_subqueries=4)
    second = planner.finalize(_spec(), source, max_subqueries=4)

    assert first == second
    assert 3 <= len(first.subqueries) <= 4
    assert [item.query_id for item in first.subqueries] == ["sq-1", "sq-2", "sq-3", "sq-4"]
    assert [item.text for item in first.subqueries][:2] == [
        "message passing OpenGraph",
        "graph retrieval",
    ]
    assert first.inherited_hard_filters == {
        "year_from": 2021,
        "year_to": 2024,
        "venues": ["NeurIPS"],
    }
    assert _spec().exclusions == ["surveys"]
    assert all(item.target_constraints for item in first.subqueries)


def test_finalize_builds_three_rule_queries_when_model_plan_is_missing() -> None:
    plan = QueryPlanner().finalize(_spec(), None, max_subqueries=5)

    assert len(plan.subqueries) >= 3
    assert plan.subqueries[0].text == _spec().original_query
    assert {item.query_type for item in plan.subqueries} >= {"exact", "decomposed"}
    assert all(item.provider_hint in {"openalex", "semantic_scholar", "either"} for item in plan.subqueries)


def test_finalize_rejects_invalid_max_subqueries() -> None:
    planner = QueryPlanner()

    try:
        planner.finalize(_spec(), None, max_subqueries=2)
    except ValueError as error:
        assert "at least 3" in str(error)
    else:
        raise AssertionError("max_subqueries below three must be rejected")


def test_finalize_clamps_caps_above_five() -> None:
    plan = QueryPlanner().finalize(_spec(), None, max_subqueries=99)

    assert len(plan.subqueries) == 3


def test_finalize_preserves_action_type_and_search_mode_identity() -> None:
    source = SearchPlan(
        subqueries=[
            SubQuery(
                query_id="title",
                text="Graph Retrieval",
                query_type="decomposed",
                action_type="title_search",
                priority=1,
                provider_hint="openalex",
            ),
            SubQuery(
                query_id="semantic",
                text="graph retrieval",
                query_type="exact",
                search_mode="semantic",
                priority=2,
                provider_hint="openalex",
            ),
            SubQuery(
                query_id="lexical",
                text="graph retrieval",
                query_type="expanded",
                priority=3,
                provider_hint="openalex",
            ),
        ],
        inherited_hard_filters={},
        rationale="test",
    )

    plan = QueryPlanner().finalize(_spec(), source)

    assert [
        (item.action_type, item.search_mode, item.text)
        for item in plan.subqueries[:3]
    ] == [
        ("title_search", "lexical", "Graph Retrieval"),
        ("text_search", "semantic", "graph retrieval"),
        ("text_search", "lexical", "graph retrieval"),
    ]


def _semantic_spec() -> QuerySpec:
    return QuerySpec(
        original_query="graph attention networks for molecular property prediction",
        research_goal="Find molecular property prediction papers using graph attention",
        topics=["molecular property prediction"],
        methods=["graph attention networks"],
        must_have=["graph attention networks", "molecular property prediction"],
    )


def _negation_semantic_spec() -> QuerySpec:
    return _semantic_spec().model_copy(
        update={
            "original_query": (
                "graph attention networks for molecular property prediction "
                "without 3D conformers"
            ),
            "research_goal": (
                "Find molecular property prediction papers using graph attention "
                "without requiring 3D conformers"
            ),
            "exclusions": ["3D conformers"],
        }
    )


def _semantic_plan(*subqueries: SubQuery) -> SearchPlan:
    return SearchPlan(
        subqueries=list(subqueries),
        inherited_hard_filters={},
        rationale="Distinct literature-side retrieval hypotheses",
    )


def test_semantic_action_v2_preserves_original_and_accepts_grounded_bridge() -> None:
    source = _semantic_plan(
        SubQuery(
            query_id="bridge",
            text=(
                "graph attention networks attention based message passing "
                "molecular property prediction"
            ),
            query_type="expanded",
            target_constraints=[
                "graph attention networks",
                "molecular property prediction",
            ],
            priority=1,
            provider_hint="openalex",
            search_mode="semantic",
        ),
        SubQuery(
            query_id="decomposed",
            text="molecular property prediction 2D molecular graph neural network",
            query_type="decomposed",
            target_constraints=["molecular property prediction"],
            priority=2,
            provider_hint="either",
        ),
        SubQuery(
            query_id="hallucinated",
            text="quantum turbulence spectral reconstruction",
            query_type="expanded",
            target_constraints=["quantum turbulence"],
            priority=3,
            provider_hint="openalex",
        ),
    )

    plan = QueryPlanner(
        prompt_version="query-analyze-semantic-actions-v2"
    ).finalize(_semantic_spec(), source)

    texts = [item.text for item in plan.subqueries]
    assert texts[0] == _semantic_spec().original_query
    assert source.subqueries[0].text in texts
    assert source.subqueries[1].text not in texts
    assert source.subqueries[2].text not in texts


def test_semantic_action_v2_replaces_only_lowest_value_fallback_at_budget() -> None:
    accepted = [
        SubQuery(
            query_id=f"llm-{index}",
            text=(
                "graph attention networks molecular property prediction "
                f"{suffix}"
            ),
            query_type="expanded",
            target_constraints=[
                "graph attention networks",
                "molecular property prediction",
            ],
            priority=index,
            provider_hint="openalex",
            search_mode="semantic",
        )
        for index, suffix in enumerate(
            ("message passing", "neural inference", "graph benchmark"),
            start=1,
        )
    ]

    plan = QueryPlanner(
        prompt_version="query-analyze-semantic-actions-v2"
    ).finalize(_semantic_spec(), _semantic_plan(*accepted), max_subqueries=5)

    identities = [
        (item.action_type, item.search_mode, item.text)
        for item in plan.subqueries
    ]
    assert len(identities) == 5
    assert identities[0] == (
        "text_search",
        "lexical",
        _semantic_spec().original_query,
    )
    assert all(item.text in [candidate.text for candidate in plan.subqueries] for item in accepted)
    assert (
        "text_search",
        "semantic",
        _semantic_spec().original_query,
    ) in identities
    assert (
        "title_search",
        "lexical",
        _semantic_spec().original_query,
    ) not in identities


def test_semantic_action_v2_keeps_fallbacks_when_replacement_budget_is_not_full() -> None:
    accepted = [
        SubQuery(
            query_id=f"llm-{index}",
            text=(
                "graph attention networks molecular property prediction "
                f"{suffix}"
            ),
            query_type="expanded",
            target_constraints=[
                "graph attention networks",
                "molecular property prediction",
            ],
            priority=index,
            provider_hint="openalex",
            search_mode="semantic",
        )
        for index, suffix in enumerate(
            ("message passing", "neural inference"),
            start=1,
        )
    ]

    plan = QueryPlanner(
        prompt_version="query-analyze-semantic-actions-v2"
    ).finalize(_semantic_spec(), _semantic_plan(*accepted), max_subqueries=5)

    identities = {
        (item.action_type, item.search_mode, item.text)
        for item in plan.subqueries
    }
    assert len(plan.subqueries) == 5
    assert (
        "text_search",
        "semantic",
        _semantic_spec().original_query,
    ) in identities
    assert (
        "title_search",
        "lexical",
        _semantic_spec().original_query,
    ) in identities


def test_semantic_action_v2_accepts_evidence_backed_soft_concept_rewrite() -> None:
    query = (
        "What research learns compact representations of long scientific documents "
        "so that conceptually related papers can be retrieved even when they use "
        "different terminology?"
    )
    spec = QuerySpec(
        original_query=query,
        research_goal=query,
        topics=["compact representations", "long scientific documents"],
        methods=["representation learning", "semantic retrieval"],
        tasks=["document retrieval"],
    )
    supported = SubQuery(
        query_id="supported-soft-rewrite",
        text=(
            "compact vector representations of long documents for concept-based "
            "retrieval in scientific literature"
        ),
        query_type="decomposed",
        target_constraints=[
            "compact vector representations",
            "long documents",
            "concept-based retrieval",
            "scientific literature",
        ],
        priority=1,
        provider_hint="either",
        search_mode="semantic",
    )
    unsupported = SubQuery(
        query_id="unsupported-soft-rewrite",
        text=(
            "dense representations of full-text scientific articles for cross-lingual "
            "terminology-agnostic paper retrieval"
        ),
        query_type="expanded",
        target_constraints=[
            "dense representations",
            "full-text scientific articles",
            "paper retrieval",
        ],
        priority=2,
        provider_hint="either",
        search_mode="semantic",
    )

    plan = QueryPlanner(
        prompt_version="query-analyze-semantic-actions-v2",
        soft_concept_evidence=lambda _: ("retrieval",),
    ).finalize(spec, _semantic_plan(supported, unsupported))

    texts = [item.text for item in plan.subqueries]
    assert supported.text in texts
    assert unsupported.text not in texts


def test_semantic_action_v2_requires_evidence_for_extra_soft_concept() -> None:
    query = "compact representations for long scientific document retrieval"
    spec = QuerySpec(original_query=query, research_goal=query)
    candidate = SubQuery(
        query_id="learned-soft-concept",
        text=(
            "compact vector embeddings for long documents semantic retrieval "
            "literature"
        ),
        query_type="expanded",
        target_constraints=["compact vector embeddings", "long documents"],
        priority=1,
        provider_hint="openalex",
        search_mode="semantic",
    )

    without_evidence = QueryPlanner(
        prompt_version="query-analyze-semantic-actions-v2"
    ).finalize(spec, _semantic_plan(candidate))
    with_evidence = QueryPlanner(
        prompt_version="query-analyze-semantic-actions-v2",
        soft_concept_evidence=lambda _: ("embeddings",),
    ).finalize(spec, _semantic_plan(candidate))

    assert candidate.text not in [item.text for item in without_evidence.subqueries]
    assert candidate.text in [item.text for item in with_evidence.subqueries]


def test_semantic_action_v2_keeps_explicit_method_hard() -> None:
    relaxed_method = SubQuery(
        query_id="drops-explicit-method",
        text="molecular property prediction with generic neural message passing",
        query_type="decomposed",
        target_constraints=["molecular property prediction"],
        priority=1,
        provider_hint="either",
        search_mode="semantic",
    )

    plan = QueryPlanner(
        prompt_version="query-analyze-semantic-actions-v2",
        soft_concept_evidence=lambda _: ("message passing",),
    ).finalize(_semantic_spec(), _semantic_plan(relaxed_method))

    assert relaxed_method.text not in [item.text for item in plan.subqueries]


def test_semantic_action_v2_keeps_named_entity_hard() -> None:
    query = "find clinical retrieval papers using BioBERT"
    spec = QuerySpec(
        original_query=query,
        research_goal=query,
        must_have=["BioBERT"],
    )
    drops_entity = SubQuery(
        query_id="drops-explicit-entity",
        text="find clinical retrieval papers using biomedical language models",
        query_type="expanded",
        target_constraints=["clinical retrieval"],
        priority=1,
        provider_hint="either",
        search_mode="semantic",
    )

    plan = QueryPlanner(
        prompt_version="query-analyze-semantic-actions-v2",
        soft_concept_evidence=lambda _: ("biomedical", "language models"),
    ).finalize(spec, _semantic_plan(drops_entity))

    assert drops_entity.text not in [item.text for item in plan.subqueries]


def test_semantic_action_v2_rejects_generated_identifiers() -> None:
    source = _semantic_plan(
        SubQuery(
            query_id="doi",
            text="graph attention networks molecular prediction 10.1234/fake-paper",
            query_type="expanded",
            target_constraints=["graph attention networks"],
            priority=1,
            provider_hint="openalex",
        ),
        SubQuery(
            query_id="openalex-id",
            text="graph attention networks OpenAlex W123456789",
            query_type="expanded",
            target_constraints=["graph attention networks"],
            priority=2,
            provider_hint="openalex",
        ),
    )

    plan = QueryPlanner(
        prompt_version="query-analyze-semantic-actions-v2"
    ).finalize(_semantic_spec(), source)

    texts = [item.text for item in plan.subqueries]
    assert all("10.1234" not in text for text in texts)
    assert all("W123456789" not in text for text in texts)


def test_semantic_action_v2_strictly_abstains_for_negation_queries() -> None:
    source = _semantic_plan(
        SubQuery(
            query_id="negative-wording",
            text=(
                "graph attention networks molecular property prediction "
                "without 3D conformers"
            ),
            query_type="expanded",
            target_constraints=["graph attention networks"],
            priority=1,
            provider_hint="either",
        ),
        SubQuery(
            query_id="implicit-alternative",
            text=(
                "graph attention networks molecular property prediction "
                "using 2D representations"
            ),
            query_type="decomposed",
            target_constraints=["graph attention networks"],
            priority=2,
            provider_hint="either",
            search_mode="semantic",
        ),
    )

    plan = QueryPlanner(
        prompt_version="query-analyze-semantic-actions-v2"
    ).finalize(_negation_semantic_spec(), source)

    texts = [item.text for item in plan.subqueries]
    assert all(item.text not in texts for item in source.subqueries)


def test_semantic_action_v2_rejects_unbounded_novel_vocabulary() -> None:
    verbose = SubQuery(
        query_id="verbose",
        text=(
            "graph attention networks quantum turbulence spectral reconstruction "
            "topological hydrodynamics astrophysical simulation stochastic geometry"
        ),
        query_type="expanded",
        target_constraints=["graph attention networks"],
        priority=1,
        provider_hint="openalex",
    )

    plan = QueryPlanner(
        prompt_version="query-analyze-semantic-actions-v2"
    ).finalize(_semantic_spec(), _semantic_plan(verbose))

    assert verbose.text not in [item.text for item in plan.subqueries]


def test_protected_action_v3_uses_one_plan_and_preserves_independent_lexical_actions() -> None:
    lexical = [
        SubQuery(
            query_id=f"lexical-{index}",
            text=text,
            query_type="expanded",
            target_constraints=[
                "graph attention networks",
                "molecular property prediction",
            ],
            priority=index,
            provider_hint="openalex",
            search_mode="lexical",
        )
        for index, text in enumerate(
            (
                "graph attention networks molecular property prediction message passing",
                "graph attention networks molecular property prediction chemical graphs",
                "graph attention networks molecular property prediction molecular benchmarks",
            ),
            start=1,
        )
    ]
    challenger = SubQuery(
        query_id="semantic-challenger",
        text=(
            "graph attention networks molecular property prediction "
            "attention based graph inference"
        ),
        query_type="expanded",
        target_constraints=[
            "graph attention networks",
            "molecular property prediction",
        ],
        priority=4,
        provider_hint="either",
        search_mode="semantic",
    )

    plan = QueryPlanner(
        prompt_version="query-analyze-protected-actions-v3",
        soft_concept_evidence=lambda _: (
            "message passing",
            "chemical graphs",
            "molecular benchmarks",
            "attention based graph inference",
        ),
    ).finalize(_semantic_spec(), _semantic_plan(*lexical, challenger))

    texts = [item.text for item in plan.subqueries]
    assert len(plan.subqueries) == 5
    assert all(item.text in texts for item in lexical)
    assert challenger.text in texts
    assert not any(
        item.action_type == "title_search"
        and item.text == _semantic_spec().original_query
        for item in plan.subqueries
    )


def test_protected_action_v3_rejects_redundant_semantic_challenger() -> None:
    lexical = SubQuery(
        query_id="lexical",
        text="graph attention networks molecular property prediction message passing",
        query_type="expanded",
        target_constraints=[
            "graph attention networks",
            "molecular property prediction",
        ],
        priority=1,
        provider_hint="openalex",
        search_mode="lexical",
    )
    redundant = lexical.model_copy(
        update={
            "query_id": "semantic-redundant",
            "priority": 2,
            "search_mode": "semantic",
        }
    )

    plan = QueryPlanner(
        prompt_version="query-analyze-protected-actions-v3",
        soft_concept_evidence=lambda _: ("message passing",),
    ).finalize(_semantic_spec(), _semantic_plan(lexical, redundant))

    identities = {(item.search_mode, item.text) for item in plan.subqueries}
    assert ("lexical", lexical.text) in identities
    assert ("semantic", redundant.text) not in identities


def test_protected_action_v3_strictly_abstains_from_semantic_negation() -> None:
    lexical = SubQuery(
        query_id="lexical-safe",
        text="graph attention networks molecular property prediction message passing",
        query_type="expanded",
        target_constraints=[
            "graph attention networks",
            "molecular property prediction",
        ],
        priority=1,
        provider_hint="openalex",
        search_mode="lexical",
    )
    semantic = lexical.model_copy(
        update={
            "query_id": "semantic-unsafe-for-negation",
            "priority": 2,
            "search_mode": "semantic",
            "text": (
                "graph attention networks molecular property prediction "
                "without 3D conformers"
            ),
        }
    )

    plan = QueryPlanner(
        prompt_version="query-analyze-protected-actions-v3",
        soft_concept_evidence=lambda _: ("message passing",),
    ).finalize(_negation_semantic_spec(), _semantic_plan(lexical, semantic))

    assert lexical.text in [item.text for item in plan.subqueries]
    assert all(item.search_mode != "semantic" for item in plan.subqueries)
