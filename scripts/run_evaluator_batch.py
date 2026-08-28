"""Run evaluator query-only JSONL through one already-started VivaAI server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import httpx

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = str(_REPOSITORY_ROOT / "src")
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

from paper_search.evaluation.dataset import read_jsonl, write_jsonl_atomic  # noqa: E402
from paper_search.evaluation.official_adapter import (  # noqa: E402
    AstaPaperFindingQuery,
    InternalPredictionRecord,
)
from paper_search.evaluation.submission_contract import (  # noqa: E402
    validate_submission_records,
)


async def run_batch_queries(
    queries: Sequence[AstaPaperFindingQuery],
    *,
    client: httpx.AsyncClient,
    mode: Literal["replay", "live"],
) -> list[InternalPredictionRecord]:
    """Run queries sequentially so provider rate limits remain predictable."""

    predictions: list[InternalPredictionRecord] = []
    for raw_query in queries:
        query = AstaPaperFindingQuery.model_validate(raw_query)
        try:
            response = await client.post(
                "/v1/search",
                json={
                    "query_id": query.query_id,
                    "query": query.query,
                    "budget_profile": "balanced",
                    "include_trace": False,
                    "mode": mode,
                },
            )
            if response.is_error:
                try:
                    error_payload = response.json()
                except json.JSONDecodeError:
                    error_payload = None
                if isinstance(error_payload, dict):
                    code = error_payload.get("code")
                    detail = error_payload.get("detail")
                    if isinstance(code, str) and isinstance(detail, str):
                        raise RuntimeError(
                            f"search failed for query {query.query_id}: "
                            f"{code[:80]}: {detail[:300]}"
                        )
            response.raise_for_status()
            payload = response.json()
        except RuntimeError:
            raise
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            raise RuntimeError(f"search failed for query {query.query_id}") from error
        if not isinstance(payload, dict) or payload.get("query_id") != query.query_id:
            raise RuntimeError(f"search response query_id mismatch for {query.query_id}")
        try:
            prediction = InternalPredictionRecord(
                query_id=query.query_id,
                selected_paper_ids=payload["selected_paper_ids"],
            )
        except (KeyError, ValueError, TypeError) as error:
            raise RuntimeError(
                f"invalid search response for query {query.query_id}"
            ) from error
        predictions.append(prediction)
    validate_submission_records(queries, predictions)
    return predictions


async def _main_async(args: argparse.Namespace) -> int:
    queries = read_jsonl(args.queries, AstaPaperFindingQuery)
    timeout = httpx.Timeout(args.timeout_seconds)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        readiness = await client.get("/health/ready")
        if readiness.status_code != 200:
            raise RuntimeError("VivaAI server is not ready")
        predictions = await run_batch_queries(
            queries,
            client=client,
            mode=args.mode,
        )
    write_jsonl_atomic(args.output, predictions)
    summary = validate_submission_records(queries, predictions)
    print(
        json.dumps(
            {**summary, "mode": args.mode, "output": str(args.output)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("replay", "live"), default="live")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
