from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from paper_search.evaluation.official_adapter import AstaPaperFindingQuery
from scripts.run_evaluator_batch import run_batch_queries


def test_batch_client_preserves_input_order_and_uses_minimal_search_payload() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "query_id": payload["query_id"],
                "selected_paper_ids": [f"arxiv:{payload['query_id']}"],
            },
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        ) as client:
            return await run_batch_queries(
                [
                    AstaPaperFindingQuery(query_id="q1", query="first"),
                    AstaPaperFindingQuery(query_id="q2", query="second"),
                ],
                client=client,
                mode="live",
            )

    predictions = asyncio.run(run())

    assert [row.query_id for row in predictions] == ["q1", "q2"]
    assert [row.selected_paper_ids for row in predictions] == [
        ["arxiv:q1"],
        ["arxiv:q2"],
    ]
    assert payloads == [
        {
            "query_id": "q1",
            "query": "first",
            "budget_profile": "balanced",
            "include_trace": False,
            "mode": "live",
        },
        {
            "query_id": "q2",
            "query": "second",
            "budget_profile": "balanced",
            "include_trace": False,
            "mode": "live",
        },
    ]


def test_batch_client_fails_closed_on_query_id_mismatch() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"query_id": "wrong", "selected_paper_ids": []}
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        ) as client:
            return await run_batch_queries(
                [AstaPaperFindingQuery(query_id="q1", query="first")],
                client=client,
                mode="replay",
            )

    with pytest.raises(RuntimeError, match="query_id mismatch"):
        asyncio.run(run())


def test_batch_client_reports_safe_server_error_code_and_detail() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "code": "snapshot_unavailable",
                "detail": "Required replay data is unavailable",
                "retryable": True,
            },
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        ) as client:
            return await run_batch_queries(
                [AstaPaperFindingQuery(query_id="q1", query="first")],
                client=client,
                mode="replay",
            )

    with pytest.raises(
        RuntimeError,
        match="snapshot_unavailable: Required replay data is unavailable",
    ):
        asyncio.run(run())
