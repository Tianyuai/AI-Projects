from __future__ import annotations

import asyncio
import ast
import importlib
from pathlib import Path

import httpx

from paper_search.domain.models import Paper
from paper_search.evaluation.runner import PipelineResult
from paper_search.processing import (
    AcceptedPaper,
    DeduplicationResult,
    FilterResult,
    MergeDecision,
    RejectedPaper,
)
from paper_search.ranking import LexicalScore
from paper_search.ui.app import MAX_FORM_BODY_BYTES, MAX_FORM_FIELDS, create_app


def _paper(identifier: str, title: str) -> Paper:
    return Paper(
        canonical_id=identifier,
        title=title,
        authors=["Ada Lovelace", "Grace Hopper"],
        publication_year=2024,
        venue="Journal of Safe Search",
        sources=["openalex", "semantic_scholar"],
    )


def _pipeline_result(*, ranked_title: str = "Auditable Search") -> PipelineResult:
    ranked_paper = _paper("doi:10.1000/ranked", ranked_title)
    rejected_paper = _paper("doi:10.1000/rejected", "Rejected Search")
    return PipelineResult(
        deduplication=DeduplicationResult(
            papers=[ranked_paper, rejected_paper],
            decisions=[
                MergeDecision(
                    representative_id=ranked_paper.canonical_id,
                    member_ids=[ranked_paper.canonical_id, "openalex:W123"],
                    match_rule="external_id",
                    match_value="doi:10.1000/ranked",
                )
            ],
        ),
        filtering=FilterResult(
            accepted=[
                AcceptedPaper(
                    paper=ranked_paper,
                    uncertainty_reasons=["unknown_retraction_status"],
                    score_multiplier=0.9,
                )
            ],
            rejected=[
                RejectedPaper(
                    paper=rejected_paper,
                    reason_code="retracted",
                    reason="Paper is marked as retracted.",
                )
            ],
        ),
        ranked=[
            LexicalScore(
                paper=ranked_paper,
                bm25_score=1.25,
                normalized_bm25=1.0,
                keyword_coverage=0.5,
                uncertainty_multiplier=0.9,
                final_score=0.765,
            )
        ],
    )


class FakeSearchService:
    def __init__(self, result: PipelineResult) -> None:
        self.result = result
        self.queries: list[str] = []

    async def __call__(self, query: str) -> PipelineResult:
        self.queries.append(query)
        return self.result


async def _request_home() -> httpx.Response:
    service = FakeSearchService(_pipeline_result())
    application = create_app(service)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        return await client.get("/")


def test_home_has_one_named_query_input() -> None:
    response = asyncio.run(_request_home())

    assert response.status_code == 200
    assert response.text.count('name="query"') == 1
    assert '<form method="post" action="/search">' in response.text


async def _request_result(
    service: FakeSearchService,
) -> httpx.Response:
    application = create_app(service)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        return await client.post(
            "/search",
            content=b"query=%20auditable+retrieval%20",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )


def test_result_renders_ranked_papers_and_audit_details() -> None:
    service = FakeSearchService(_pipeline_result())

    response = asyncio.run(_request_result(service))

    assert response.status_code == 200
    assert service.queries == ["auditable retrieval"]
    for expected in (
        "Auditable Search",
        "Ada Lovelace",
        "Grace Hopper",
        "2024",
        "Journal of Safe Search",
        "openalex",
        "semantic_scholar",
        "0.765",
        "unknown_retraction_status",
        "external_id",
        "retracted",
    ):
        assert expected in response.text
    assert response.text.index("Auditable Search") < response.text.index("Merge decisions")
    assert response.text.count("<details>") == 4


def test_result_renders_collapsible_score_breakdown() -> None:
    response = asyncio.run(_request_result(FakeSearchService(_pipeline_result())))

    for expected in (
        "Scoring breakdown",
        "Raw BM25: 1.25",
        "Normalized BM25: 1.0",
        "Keyword coverage: 0.5",
        "Uncertainty multiplier: 0.9",
        "Final score: 0.765",
    ):
        assert expected in response.text
    scoring_summary = response.text.index("<summary>Scoring breakdown</summary>")
    assert response.text.rfind("<details>", 0, scoring_summary) != -1
    assert response.text.find("</details>", scoring_summary) != -1


def test_result_escapes_external_values() -> None:
    service = FakeSearchService(
        _pipeline_result(ranked_title='<script>alert("credential")</script>')
    )

    response = asyncio.run(_request_result(service))

    assert "<script>" not in response.text
    assert "&lt;script&gt;alert(&quot;credential&quot;)&lt;/script&gt;" in response.text


class FailingSearchService:
    async def __call__(self, query: str) -> PipelineResult:
        raise RuntimeError("sentinel-secret-credential")


async def _request_failure(application: object) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        return await client.post(
            "/search",
            content=b"query=safe",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )


def test_service_failure_returns_constant_safe_error() -> None:
    response = asyncio.run(_request_failure(create_app(FailingSearchService())))

    assert response.status_code == 503
    assert response.text == "search is temporarily unavailable"
    assert "sentinel-secret-credential" not in response.text


def test_default_app_uses_unavailable_service() -> None:
    ui_module = importlib.import_module("paper_search.ui.app")

    response = asyncio.run(_request_failure(ui_module.app))

    assert response.status_code == 503
    assert response.text == "search is temporarily unavailable"


async def _request_invalid_query(content: bytes, service: FakeSearchService) -> httpx.Response:
    application = create_app(service)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        return await client.post(
            "/search",
            content=content,
            headers={"content-type": "application/x-www-form-urlencoded"},
        )


def test_empty_query_is_rejected_without_calling_service() -> None:
    service = FakeSearchService(_pipeline_result())

    response = asyncio.run(_request_invalid_query(b"query=+", service))

    assert response.status_code == 400
    assert service.queries == []


def test_repeated_query_is_rejected_without_calling_service() -> None:
    service = FakeSearchService(_pipeline_result())

    response = asyncio.run(
        _request_invalid_query(b"query=first&query=second", service)
    )

    assert response.status_code == 400
    assert service.queries == []


def test_oversized_form_body_returns_constant_413_without_calling_service() -> None:
    service = FakeSearchService(_pipeline_result())
    content = b"query=" + (b"x" * MAX_FORM_BODY_BYTES)

    response = asyncio.run(_request_invalid_query(content, service))

    assert response.status_code == 413
    assert response.text == "request body is too large"
    assert service.queries == []


def test_excess_form_fields_returns_constant_400_without_calling_service() -> None:
    service = FakeSearchService(_pipeline_result())
    extra_fields = b"&".join(
        f"field{index}=x".encode() for index in range(MAX_FORM_FIELDS)
    )
    content = b"query=safe&" + extra_fields

    response = asyncio.run(_request_invalid_query(content, service))

    assert response.status_code == 400
    assert response.text == "query must be provided exactly once"
    assert service.queries == []


def test_ui_has_no_candidate_processing_algorithms_or_rank_bm25_import() -> None:
    ui_module = importlib.import_module("paper_search.ui.app")
    source = Path(ui_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "rank_bm25" not in source
    assert not any(
        name.startswith(("dedup", "filter", "rank")) for name in function_names
    )
