from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from paper_search.domain.models import BudgetReservation, UsageEstimate
from paper_search.retrieval.openalex import OpenAlexProvider
from paper_search.storage.cache import SQLiteResponseCache, validate_snapshot_manifest


LIVE_QUERIES = (
    "retrieval augmented generation evaluation",
    "dense scholarly search benchmark",
    "academic paper recommendation evidence",
)


def reservation(index: int) -> BudgetReservation:
    return BudgetReservation(
        reservation_id=f"live-{index}",
        action="openalex-live-smoke",
        reserved=UsageEstimate(search_api_calls=3),
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )


async def run_live_queries(api_key: str, output: Path) -> None:
    cache = SQLiteResponseCache(output / "openalex-cache.sqlite3")
    summaries: list[dict[str, object]] = []
    ordered_keys: list[str] = []
    async with httpx.AsyncClient(
        base_url="https://api.openalex.org",
        timeout=httpx.Timeout(30.0),
    ) as client:
        provider = OpenAlexProvider(client=client, cache=cache, api_key=api_key)
        for index, query in enumerate(LIVE_QUERIES, start=1):
            result = await provider.search(query, {}, 3, reservation(index))
            assert result.data
            ordered_keys.extend(json.loads(result.provenance["cache_keys"]))
            summaries.append(
                {
                    "error_codes": [error.code for error in result.errors],
                    "latency_ms": result.latency_ms,
                    "paper_count": len(result.data),
                    "response_hash": result.provenance["response_hash"],
                }
            )
    manifest = cache.export_snapshot(list(dict.fromkeys(ordered_keys)), output)
    validate_snapshot_manifest(manifest)
    (output / "provider.json").write_text(
        json.dumps({"queries": summaries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.online
def test_three_live_queries_produce_safe_snapshot(tmp_path: Path) -> None:
    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        pytest.skip("OPENALEX_API_KEY is not set in the process environment")

    asyncio.run(run_live_queries(api_key, tmp_path))

    serialized = (tmp_path / "provider.json").read_text(encoding="utf-8")
    assert api_key not in serialized
    assert len(json.loads(serialized)["queries"]) == 3
    validate_snapshot_manifest(tmp_path / "snapshot_manifest.json")
