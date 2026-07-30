import importlib
import importlib.util
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError


REQUIRED_MODELS = {
    "BudgetReservation",
    "CandidateEvidence",
    "CitationEdge",
    "CitationExpansion",
    "ErrorDetail",
    "Paper",
    "ProviderPaperId",
    "ProviderResult",
    "QueryAnalysisResult",
    "QuerySpec",
    "RankedPaper",
    "ResolvedCitationEdge",
    "SearchBudget",
    "SearchPlan",
    "StructuredSearchResponse",
    "SubQuery",
    "UsageActual",
    "UsageEstimate",
}


def models_module() -> object:
    module_name = "paper_search.domain.models"
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, f"{module_name} must be implemented"
    return importlib.import_module(module_name)


def make_query_analysis(models: object) -> object:
    spec = models.QuerySpec(original_query="RAG evaluation", research_goal="Find evaluations")
    plan = models.SearchPlan(
        subqueries=[
            models.SubQuery(
                query_id="sq-1",
                text="retrieval augmented generation evaluation",
                query_type="exact",
                target_constraints=["evaluation"],
                priority=1,
                provider_hint="either",
            )
        ],
        inherited_hard_filters={},
        rationale="Direct query",
    )
    return models.QueryAnalysisResult(query_spec=spec, search_plan=plan)


def make_paper(models: object) -> object:
    return models.Paper(
        canonical_id="doi:10.1/example",
        title="Retrieval Augmented Generation",
    )


def make_evidence(models: object) -> object:
    return models.CandidateEvidence(
        paper_id="doi:10.1/example",
        lexical_score=0.8,
        embedding_score=0.7,
        constraint_coverage=1.0,
        source_agreement=0.5,
        authority_score=0.4,
        recency_score=0.9,
        final_score=0.8,
        scoring_version="v1",
        relevance_level="high",
    )


def test_all_prd_models_are_exposed_and_rebuild() -> None:
    models = models_module()

    assert REQUIRED_MODELS <= set(dir(models))
    for name in REQUIRED_MODELS:
        assert getattr(models, name).model_rebuild() is not False


@pytest.mark.parametrize(
    ("year_from", "year_to"),
    [(1899, 2020), (2020, datetime.now(UTC).year + 2), (2021, 2020)],
)
def test_query_spec_rejects_invalid_year_ranges(year_from: int, year_to: int) -> None:
    models = models_module()

    with pytest.raises(ValidationError):
        models.QuerySpec(
            original_query="valid query",
            research_goal="valid goal",
            year_from=year_from,
            year_to=year_to,
        )


def test_query_spec_defaults_optional_collections_to_empty_lists() -> None:
    models = models_module()

    query = models.QuerySpec(original_query="  RAG  ", research_goal="  Find papers  ")

    assert query.original_query == "RAG"
    assert query.research_goal == "Find papers"
    assert query.topics == []
    assert query.ambiguities == []


def test_paper_rejects_empty_title_and_unknown_fields() -> None:
    models = models_module()

    with pytest.raises(ValidationError):
        models.Paper(canonical_id="paper:1", title="   ")
    with pytest.raises(ValidationError):
        models.Paper(canonical_id="paper:1", title="Valid", unexpected=True)


def test_citation_edge_requires_provider_consistency() -> None:
    models = models_module()
    openalex_id = models.ProviderPaperId(provider="openalex", value="W1")
    semantic_id = models.ProviderPaperId(provider="semantic_scholar", value="S1")

    with pytest.raises(ValidationError):
        models.CitationEdge(
            provider="openalex",
            citing_provider_id=openalex_id,
            cited_provider_id=semantic_id,
        )


def test_usage_rejects_negative_values_and_preserves_unknown_cost() -> None:
    models = models_module()

    usage = models.UsageEstimate(search_api_calls=1)

    assert usage.cost_cny is None
    assert models.UsageActual(cost_cny=0.1).cost_cny == Decimal("0.1")
    with pytest.raises(ValidationError):
        models.UsageEstimate(input_tokens=-1)
    with pytest.raises(ValidationError):
        models.UsageActual(cost_cny=-0.01)
    with pytest.raises(ValidationError):
        models.UsageActual(cost_cny=Decimal("0.0000001"))


def test_search_budget_uses_prd_defaults_but_requires_token_and_cost_limits() -> None:
    models = models_module()
    budget = models.SearchBudget(max_total_tokens=24_000, max_cost_cny=0.30)

    assert budget.max_search_api_calls == 12
    assert budget.target_search_api_calls == 8
    assert budget.max_llm_calls == 5
    assert budget.max_elapsed_seconds == 90
    assert budget.soft_deadline_seconds == 80
    with pytest.raises(ValidationError):
        models.SearchBudget(max_cost_cny=0.30)
    with pytest.raises(ValidationError):
        models.SearchBudget(max_total_tokens=-1, max_cost_cny=0.30)


def test_search_budget_accepts_zero_quotas_but_rejects_coercion_and_infinity() -> None:
    models = models_module()
    budget = models.SearchBudget(
        max_search_api_calls=0,
        target_search_api_calls=0,
        max_llm_calls=0,
        target_llm_calls=0,
        max_rerank_candidates=0,
        max_citation_seeds=0,
        target_citation_seeds=0,
        max_total_tokens=0,
        max_cost_cny=0.0,
    )

    assert budget.max_llm_calls == 0
    with pytest.raises(ValidationError):
        models.SearchBudget(max_total_tokens=True, max_cost_cny=0.30)
    with pytest.raises(ValidationError):
        models.SearchBudget(max_total_tokens=100, max_cost_cny=float("inf"))


def test_provider_result_validates_generic_data_and_provenance() -> None:
    models = models_module()
    result_type = models.ProviderResult[list[models.Paper]]
    provenance = {
        "provider": "openalex",
        "endpoint": "/works",
        "model_id": "none",
        "requested_at": "2026-07-15T00:00:00Z",
        "response_hash": "sha256:abc",
    }

    result = result_type(
        data=[make_paper(models)],
        usage=models.UsageActual(search_api_calls=1),
        provenance=provenance,
        cache_hit=False,
        latency_ms=10,
        errors=[],
    )

    assert result.data[0].canonical_id == "doi:10.1/example"
    with pytest.raises(ValidationError):
        result_type(
            data=[make_paper(models)],
            usage=models.UsageActual(),
            provenance={"provider": "openalex"},
            cache_hit=True,
            latency_ms=0,
            errors=[],
        )
    with pytest.raises(ValidationError):
        result_type(
            data=[make_paper(models)],
            usage=models.UsageActual(),
            provenance={**provenance, "endpoint": "   "},
            cache_hit=True,
            latency_ms=0,
            errors=[],
        )


def test_structured_response_resolves_forward_references() -> None:
    models = models_module()
    ranked = models.RankedPaper(paper=make_paper(models), evidence=make_evidence(models))

    response = models.StructuredSearchResponse(
        query_id="q-1",
        query_analysis=make_query_analysis(models),
        selected_paper_ids=["doi:10.1/example"],
        high_relevance=[ranked],
        partial_relevance=[],
        citation_edges=[],
        search_trace=[],
        usage=models.UsageActual(),
        stop_reason="completed",
        is_partial=False,
        warnings=[],
        config_hash="sha256:" + "a" * 64,
        git_sha="abc1234",
    )

    assert response.high_relevance[0].paper.title == "Retrieval Augmented Generation"
