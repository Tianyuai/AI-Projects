"""One-command in-process evaluator runner for public or hidden query JSONL."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import httpx

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.api.app import create_app  # noqa: E402
from paper_search.application.composition import CompositionRoot  # noqa: E402
from paper_search.application.contracts import SearchRequest  # noqa: E402
from paper_search.evaluation.dataset import read_jsonl, write_jsonl_atomic  # noqa: E402
from paper_search.evaluation.official_adapter import (  # noqa: E402
    AstaPaperFindingQuery,
    InternalPredictionRecord,
)
from paper_search.evaluation.submission_contract import (  # noqa: E402
    validate_submission_records,
)
from scripts.run_evaluator_batch import run_batch_queries  # noqa: E402


async def _run(args: argparse.Namespace) -> dict[str, object]:
    queries = read_jsonl(args.queries, AstaPaperFindingQuery)
    if args.mode == "live":
        input_lock_bytes = args.lock.read_bytes()
        verified_replays = 0
        bundle = CompositionRoot.compose(
            lock_path=args.lock,
            mode="live",
            artifact_root=args.artifact_root,
            output_root=args.capture_output_root,
            snapshot_manifest_path=None,
            network_authorized=True,
            environ=os.environ,
        )
        try:
            predictions: list[InternalPredictionRecord] = []
            for index, query in enumerate(queries, start=1):
                run_id = f"evaluator-{index:06d}-{uuid4().hex[:12]}"
                session = bundle.artifact_factory.start_capture(
                    run_id=run_id,
                    input_lock_bytes=input_lock_bytes,
                    expected_config_hash=bundle.config_hash,
                )
                try:
                    execution = await bundle.service.execute(
                        SearchRequest(
                            query_id=query.query_id,
                            query=query.query,
                            budget_profile="balanced",
                            include_trace=False,
                            mode="live",
                        ),
                        run_id=run_id,
                    )
                    session.record_execution(execution)
                    outcome = execution.outcome
                    if outcome.kind != "success":
                        session.fail(outcome.error.code)
                        raise RuntimeError(
                            f"search failed for query {query.query_id}: "
                            f"{outcome.error.code}: {outcome.error.detail}"
                        )
                    session.seal()
                    published = session.publish()
                    if args.verify_replay:
                        replay_bundle = CompositionRoot.compose(
                            lock_path=published / "replay.lock.yaml",
                            mode="replay",
                            artifact_root=args.artifact_root,
                            output_root=args.capture_output_root / "replay-checks",
                            snapshot_manifest_path=(
                                published / "snapshot-manifest.json"
                            ),
                            network_authorized=False,
                            environ={},
                        )
                        try:
                            replay_execution = await replay_bundle.service.execute(
                                SearchRequest(
                                    query_id=query.query_id,
                                    query=query.query,
                                    budget_profile="balanced",
                                    include_trace=False,
                                    mode="replay",
                                ),
                                run_id=f"replay-check-{index:06d}",
                            )
                        finally:
                            await replay_bundle.aclose()
                        replay_outcome = replay_execution.outcome
                        if replay_outcome.kind != "success":
                            raise RuntimeError(
                                f"replay verification failed for query {query.query_id}: "
                                f"{replay_outcome.error.code}: "
                                f"{replay_outcome.error.detail}"
                            )
                        if (
                            replay_outcome.response.selected_paper_ids
                            != outcome.response.selected_paper_ids
                        ):
                            raise RuntimeError(
                                f"live/replay ranked IDs differ for query {query.query_id}"
                            )
                        verified_replays += 1
                except BaseException:
                    work_dir = getattr(session, "work_dir", None)
                    if work_dir is not None and work_dir.exists():
                        try:
                            session.fail("internal_error")
                        except (OSError, RuntimeError, ValueError):
                            pass
                    raise
                predictions.append(
                    InternalPredictionRecord(
                        query_id=query.query_id,
                        selected_paper_ids=outcome.response.selected_paper_ids,
                    )
                )
        finally:
            await bundle.aclose()
        write_jsonl_atomic(args.output, predictions)
        return {
            **validate_submission_records(queries, predictions),
            "mode": args.mode,
            "output": str(args.output),
            "test_partition_touched": False,
            "live_replay_verified_queries": verified_replays,
        }

    server = CompositionRoot.compose_server(
        replay_lock_path=args.lock,
        snapshot_manifest_path=args.snapshot_manifest,
        artifact_root=args.artifact_root,
        capture_output_root=args.capture_output_root,
        live_authorized=False,
        environ=os.environ,
    )
    try:
        transport = httpx.ASGITransport(app=create_app(server.service_router))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://vivaai.local",
            timeout=httpx.Timeout(args.timeout_seconds),
        ) as client:
            readiness = await client.get("/health/ready")
            if readiness.status_code != 200:
                raise RuntimeError("VivaAI evaluator service is not ready")
            predictions = await run_batch_queries(queries, client=client, mode="replay")
    finally:
        await server.aclose()
    write_jsonl_atomic(args.output, predictions)
    return {
        **validate_submission_records(queries, predictions),
        "mode": args.mode,
        "output": str(args.output),
        "test_partition_touched": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--capture-output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("replay", "live"), default="live")
    parser.add_argument(
        "--verify-replay",
        action="store_true",
        help="after each live query, replay its newly sealed capture and require exact ranked IDs",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.mode == "replay" and args.snapshot_manifest is None:
        parser.error("--snapshot-manifest is required in replay mode")
    if args.mode == "replay" and args.verify_replay:
        parser.error("--verify-replay is only valid in live mode")
    summary = asyncio.run(_run(args))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
