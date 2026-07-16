from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from paper_search.config import RuntimeConfig, load_budget
from paper_search.domain.models import (
    BudgetReservation,
    Paper,
    ProviderResult,
    UsageActual,
)
from paper_search.evaluation.dataset import EvaluationQuery, read_jsonl
from paper_search.evaluation.runner import run_evaluation
from paper_search.storage import SQLiteResponseCache
from paper_search.storage.cache import validate_snapshot_manifest


FIXTURES = Path(__file__).parents[1] / "fixtures" / "week1"
CONFIGS = Path(__file__).parents[2] / "configs"


class FixedProvider:
    def __init__(self, results: dict[str, list[Paper]]) -> None:
        self._results = results

    async def search(
        self,
        query: str,
        filters: dict[str, object],
        limit: int,
        reservation: BudgetReservation,
    ) -> ProviderResult[list[Paper]]:
        del filters, reservation
        query_index = list(self._results).index(query) + 1
        return ProviderResult(
            data=self._results[query][:limit],
            usage=UsageActual(search_api_calls=1, elapsed_ms=query_index),
            provenance={
                "provider": "openalex",
                "endpoint": "/works",
                "model_id": "openalex-api",
                "requested_at": "2026-07-17T00:00:00+00:00",
                "response_hash": f"sha256:{'0' * 64}",
                "cache_keys": json.dumps([f"fixture-page-{query_index}"]),
            },
            cache_hit=True,
            latency_ms=query_index,
            errors=[],
        )


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        budget_profile="low",
        budget=load_budget(CONFIGS / "budget_low.yaml"),
        llm_base_url="https://llm.invalid/v1",
        llm_model_primary="test-primary",
        llm_model_fallback="test-fallback",
    )


def test_fixed_week1_fixture_runs_full_pipeline_and_snapshot(tmp_path: Path) -> None:
    gold = read_jsonl(FIXTURES / "gold.jsonl", EvaluationQuery)
    fixture_bytes = (FIXTURES / "openalex_results.json").read_bytes()
    payload = json.loads(fixture_bytes)
    results = {
        query: [Paper.model_validate(record) for record in records]
        for query, records in payload.items()
    }
    cache = SQLiteResponseCache(tmp_path / ".cache" / "openalex.sqlite3")
    now = datetime(2026, 7, 17, tzinfo=UTC)
    for index, query in enumerate(results, start=1):
        raw_response = json.dumps(
            {"query": query, "results": payload[query]},
            sort_keys=True,
        ).encode("utf-8")
        cache.put_response(
            key=f"fixture-page-{index}",
            provider="openalex",
            endpoint="/works",
            cache_version="v1",
            params={"search": query},
            raw_response=raw_response,
            requested_at=now,
            ttl=timedelta(days=7),
            safe_headers={},
        )
    output = tmp_path / "run"

    result = asyncio.run(
        run_evaluation(
            gold,
            provider=FixedProvider(results),
            cache=cache,
            config=_runtime_config(),
            output=output,
        )
    )

    assert result.evaluation.summary.query_count == 2
    assert result.evaluation.summary.macro_f1 > 0
    assert result.query_runs[0].pipeline.deduplication.decisions
    assert result.query_runs[0].pipeline.filtering.rejected[0].reason_code == "retracted"
    assert any(
        accepted.paper.publication_year is None
        for accepted in result.query_runs[1].pipeline.filtering.accepted
    )
    validate_snapshot_manifest(output / "snapshot_manifest.json")
