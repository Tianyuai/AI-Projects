from __future__ import annotations

import asyncio
import json

import httpx

from paper_search.evaluation.official_adapter import AstaPaperFindingQuery
from scripts.run_delivery_rehearsal import run_delivery_rehearsal


def test_rehearsal_calls_live_and_replay_and_returns_a_passing_report() -> None:
    seen_modes: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_modes.append(payload["mode"])
        return httpx.Response(
            200,
            json={
                "query_id": payload["query_id"],
                "selected_paper_ids": ["openalex:W1"],
            },
        )

    async def scenario():
        transport = httpx.MockTransport(handler)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://live") as live,
            httpx.AsyncClient(transport=transport, base_url="http://replay") as replay,
        ):
            return await run_delivery_rehearsal(
                [AstaPaperFindingQuery(query_id="q1", query="graph retrieval")],
                live_client=live,
                replay_client=replay,
            )

    live_predictions, replay_predictions, report = asyncio.run(scenario())

    assert seen_modes == ["live", "replay"]
    assert live_predictions == replay_predictions
    assert report["passed"] is True
