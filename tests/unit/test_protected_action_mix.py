from __future__ import annotations

from paper_search.domain.models import QuerySpec, SearchPlan, SubQuery
from paper_search.query.protected_action_mix import select_protected_action_mix


def _action(
    query_id: str,
    text: str,
    priority: int,
    *,
    search_mode: str = "lexical",
    action_type: str = "text_search",
) -> SubQuery:
    return SubQuery(
        query_id=query_id,
        text=text,
        query_type="expanded",
        action_type=action_type,
        target_constraints=["target"],
        priority=priority,
        provider_hint="openalex",
        search_mode=search_mode,
    )


def _plan(*actions: SubQuery) -> SearchPlan:
    return SearchPlan(
        subqueries=list(actions),
        inherited_hard_filters={},
        rationale="fixture",
    )


def test_mix_protects_independent_production_phrases_from_redundant_rewrite() -> None:
    query = "research about improving model speed using dynamic or static token pruning"
    production = _plan(
        _action("p1", "dynamic token pruning for model speedup", 1),
        _action("p2", "static token pruning for inference acceleration", 2),
        _action("p3", "token pruning techniques for transformer efficiency", 3),
        _action("p4", "dynamic vs static token pruning performance comparison", 4),
        _action("p5", "token pruning for reducing computational cost in NLP models", 5),
    )
    semantic = _plan(
        _action("exact", query, 1),
        _action(
            "rewrite",
            "static token pruning efficiency model acceleration",
            2,
            search_mode="semantic",
        ),
        _action("title", query, 3, action_type="title_search"),
    )

    selection = select_protected_action_mix(
        QuerySpec(original_query=query, research_goal=query),
        production,
        semantic,
    )

    assert [item.source_query_id for item in selection.actions] == [
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
    ]
    assert selection.replaced_production_ids == ()
    assert selection.added_semantic_ids == ()


def test_mix_replaces_one_redundant_fallback_with_novel_semantic_phrase() -> None:
    query = "Which paper introduced the Omni3D dataset?"
    production = _plan(
        _action("p1", "Omni3D dataset paper", 1),
        _action("p2", "paper introducing Omni3D dataset", 2),
        _action("p3", "Omni3D dataset landscape enhancement", 3),
        _action("p4", "Omni3D dataset paper title", 4),
        _action("p5", "Omni3D dataset publication", 5),
    )
    semantic = _plan(
        _action("exact", query, 1),
        _action(
            "semantic-novel",
            "introducing Omni3D dataset for 3D perception",
            2,
            search_mode="semantic",
        ),
        _action("title", query, 3, action_type="title_search"),
    )

    selection = select_protected_action_mix(
        QuerySpec(
            original_query=query,
            research_goal=query,
            datasets=["Omni3D"],
        ),
        production,
        semantic,
    )

    assert len(selection.actions) == 5
    assert len(selection.replaced_production_ids) == 1
    assert selection.added_semantic_ids == ("semantic-novel",)
    assert "semantic-novel" in [item.source_query_id for item in selection.actions]
    assert sum(item.origin == "production" for item in selection.actions) == 4


def test_mix_fills_unused_slots_without_evicting_production_actions() -> None:
    query = "Which research introduced the InterHuman dataset?"
    production = _plan(
        _action("p1", "InterHuman dataset introduction research paper", 1),
        _action("p2", "InterHuman dataset origin publication", 2),
        _action("p3", "Who introduced InterHuman dataset?", 3),
    )
    semantic = _plan(
        _action("exact", query, 1),
        _action("s1", "InterHuman dataset paper", 2),
        _action("s2", "InterHuman dataset benchmark paper", 3),
        _action("title", query, 4, action_type="title_search"),
    )

    selection = select_protected_action_mix(
        QuerySpec(
            original_query=query,
            research_goal=query,
            datasets=["InterHuman"],
        ),
        production,
        semantic,
    )

    assert len(selection.actions) == 5
    assert selection.replaced_production_ids == ()
    assert {"p1", "p2", "p3"}.issubset(
        {item.source_query_id for item in selection.actions}
    )
    assert selection.added_semantic_ids == ("s1", "s2")


def test_mix_strictly_abstains_from_semantic_replacement_for_negation() -> None:
    query = "graph retrieval without citation expansion"
    production = _plan(
        *[
            _action(f"p{index}", f"graph retrieval lexical expression {index}", index)
            for index in range(1, 6)
        ]
    )
    semantic = _plan(
        _action(
            "semantic",
            "graph document discovery using dense vector search",
            1,
            search_mode="semantic",
        )
    )

    selection = select_protected_action_mix(
        QuerySpec(
            original_query=query,
            research_goal=query,
            exclusions=["citation expansion"],
        ),
        production,
        semantic,
    )

    assert all(item.origin == "production" for item in selection.actions)
    assert selection.added_semantic_ids == ()
    assert selection.replaced_production_ids == ()
