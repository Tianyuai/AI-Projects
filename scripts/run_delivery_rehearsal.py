"""Run the same evaluator queries through live and replay and compare outputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.evaluation.dataset import read_jsonl, write_jsonl_atomic  # noqa: E402
from paper_search.evaluation.delivery_rehearsal import (  # noqa: E402
    compare_delivery_predictions,
)
from paper_search.evaluation.official_adapter import (  # noqa: E402
    AstaPaperFindingQuery,
    InternalPredictionRecord,
)
from scripts.run_evaluator_batch import run_batch_queries  # noqa: E402


async def run_delivery_rehearsal(
    queries: Sequence[AstaPaperFindingQuery],
    *,
    live_client: httpx.AsyncClient,
    replay_client: httpx.AsyncClient,
) -> tuple[
    list[InternalPredictionRecord],
    list[InternalPredictionRecord],
    dict[str, object],
]:
    live = await run_batch_queries(queries, client=live_client, mode="live")
    replay = await run_batch_queries(queries, client=replay_client, mode="replay")
    return live, replay, compare_delivery_predictions(live, replay)


async def _main_async(args: argparse.Namespace) -> int:
    queries = read_jsonl(args.queries, AstaPaperFindingQuery)
    timeout = httpx.Timeout(args.timeout_seconds)
    async with (
        httpx.AsyncClient(base_url=args.live_base_url, timeout=timeout) as live_client,
        httpx.AsyncClient(base_url=args.replay_base_url, timeout=timeout) as replay_client,
    ):
        for label, client in (("live", live_client), ("replay", replay_client)):
            readiness = await client.get("/health/ready")
            if readiness.status_code != 200:
                raise RuntimeError(f"{label} VivaAI server is not ready")
        live, replay, report = await run_delivery_rehearsal(
            queries, live_client=live_client, replay_client=replay_client
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dir / "live-predictions.jsonl", live)
    write_jsonl_atomic(args.output_dir / "replay-predictions.jsonl", replay)
    report_path = args.output_dir / "live-replay-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "report": str(report_path)}, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--replay-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
