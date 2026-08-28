from __future__ import annotations

from paper_search.domain.models import QuerySpec, SearchPlan, SubQuery
from paper_search.learning.lexical_bridge import (
    LexicalBridgeExample,
    SupervisedLexicalBridge,
)
from paper_search.learning.lexical_bridge_deployment import LoadedLexicalBridge
from paper_search.learning.production_query_expansion import (
    SupervisedLexicalBridgePlanEnricher,
)
from paper_search.query.parser import ClassifiedQueryAnalysis


def _enricher() -> SupervisedLexicalBridgePlanEnricher:
    bridge = SupervisedLexicalBridge.fit(
        [
            LexicalBridgeExample(
                query="multimodal representation learning",
                gold_titles=("Cross modal retrieval with aligned embeddings",),
            ),
            LexicalBridgeExample(
                query="multi modal representation alignment",
                gold_titles=("Cross modal retrieval using joint embeddings",),
            ),
            LexicalBridgeExample(
                query="protein structure prediction",
                gold_titles=("Protein folding with geometric networks",),
            ),
        ],
        representation="word_char",
        learning_objective="neighbor_idf",
    )
    loaded = LoadedLexicalBridge(
        bridge=bridge,
        source_sha256="sha256:" + "a" * 64,
        max_expansion_terms=6,
        neighbors=12,
        min_neighbor_support=2,
        manifest={"training_query_count": 3},
    )
    return SupervisedLexicalBridgePlanEnricher(loaded, max_total_subqueries=6)


def _analysis() -> ClassifiedQueryAnalysis:
    query = "multimodality representations"
    return ClassifiedQueryAnalysis(
        query_spec=QuerySpec(original_query=query, research_goal=query),
        search_plan=SearchPlan(
            subqueries=[
                SubQuery(
                    query_id=f"sq-{index}",
                    text=f"{query} base {index}",
                    query_type="expanded",
                    priority=index,
                    provider_hint="openalex",
                )
                for index in range(1, 6)
            ],
            inherited_hard_filters={},
            rationale="fixture plan",
        ),
        planner_status="primary",
    )


def test_supervised_bridge_appends_one_hash_traced_openalex_action() -> None:
    enriched, receipt = _enricher().enrich(_analysis())

    assert len(enriched.search_plan.subqueries) == 6
    action = enriched.search_plan.subqueries[-1]
    assert action.query_id == "sq-supervised-lexical-bridge"
    assert action.provider_hint == "openalex"
    assert action.priority == 6
    assert "retrieval" in action.text
    assert receipt["status"] == "appended"
    assert receipt["model_sha256"] == "sha256:" + "a" * 64
    assert receipt["training_query_count"] == 3
    assert "retrieval" in receipt["expansion_terms"]
    assert receipt["configured_action_budget"] == 6
    assert receipt["action_count_before"] == 5
    assert receipt["action_count_after"] == 6
    assert receipt["budget_policy"] == "llm-replaces-rule-fallback-before-local-bridge"


def test_supervised_bridge_exposes_frozen_soft_concept_evidence() -> None:
    enricher = _enricher()

    terms = enricher.soft_concept_terms("multimodality representations")

    assert "retrieval" in terms


def test_supervised_bridge_abstains_when_six_action_bound_is_full() -> None:
    analysis = _analysis()
    analysis = analysis.model_copy(
        update={
            "search_plan": analysis.search_plan.model_copy(
                update={
                    "subqueries": [
                        *analysis.search_plan.subqueries,
                        SubQuery(
                            query_id="sq-6",
                            text="already full",
                            query_type="expanded",
                            priority=6,
                            provider_hint="openalex",
                        ),
                    ]
                }
            )
        }
    )

    enriched, receipt = _enricher().enrich(analysis)

    assert enriched == analysis
    assert receipt["status"] == "action_budget_exhausted"
    assert receipt["configured_action_budget"] == 6
    assert receipt["action_count_before"] == 6
    assert receipt["action_count_after"] == 6
