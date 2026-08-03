"""Stable command-line root for offline-first paper-search workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from paper_search.application.composition import ApplicationBundle, CompositionRoot
from paper_search.application.contracts import (
    SearchErrorCode,
    SearchFailure,
    SearchRequest,
)
from paper_search.evaluation.runner import (
    EvaluationRunRequest,
    EvaluationRunResult,
    run_evaluation,
)
from paper_search.evaluation.validator import (
    compare_replay_command,
    verify_run_command,
)


_SMOKE_QUERY = "resource-aware scholarly paper search"
_SMOKE_QUERY_ID = "smoke-query-1"


class _SmokeFailure(RuntimeError):
    def __init__(self, code: SearchErrorCode) -> None:
        super().__init__(code)
        self.code = code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paper-search",
        description="Offline-first scholarly paper search workflows.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke",
        help="run one replay-default end-to-end smoke query",
    )
    smoke.add_argument("--lock", type=Path, required=True)
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--mode", choices=("replay", "live"), default="replay")
    smoke.add_argument("--snapshot-manifest", type=Path)
    smoke.add_argument("--allow-network", action="store_true")
    evaluate = commands.add_parser("evaluate", help="run one formal evaluation")
    evaluate.add_argument("--lock", type=Path, required=True)
    evaluate.add_argument("--split", choices=("dev", "validation"), required=True)
    evaluate.add_argument("--mode", choices=("replay", "live"), default="replay")
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--snapshot-manifest", type=Path)
    evaluate.add_argument("--allow-network", action="store_true")
    verify = commands.add_parser("verify-run", help="verify a formal run")
    verify.add_argument("run_directory", type=Path)
    compare = commands.add_parser("compare-replay", help="compare capture and replay")
    compare.add_argument("capture_run", type=Path)
    compare.add_argument("replay_run", type=Path)
    serve = commands.add_parser("serve", help="serve the canonical replay API")
    serve.add_argument("--lock", type=Path, required=True)
    serve.add_argument("--mode", choices=("replay",), required=True)
    serve.add_argument("--snapshot-manifest", type=Path, required=True)
    serve.add_argument("--capture-output-root", type=Path, required=True)
    serve.add_argument("--allow-live", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


async def _close_bundle(bundle: ApplicationBundle | None) -> None:
    if bundle is not None:
        await bundle.aclose()


async def _run_smoke(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    if args.mode == "replay" and args.snapshot_manifest is None:
        raise _SmokeFailure("snapshot_unavailable")
    if args.mode == "live" and not args.allow_network:
        raise _SmokeFailure("live_not_authorized")
    if args.mode == "live" and args.snapshot_manifest is not None:
        raise _SmokeFailure("invalid_request")

    lock_path = Path(args.lock)
    try:
        input_lock_bytes = lock_path.read_bytes()
    except OSError as error:
        raise _SmokeFailure("config_mismatch") from error

    bundle: ApplicationBundle | None = None
    session = None
    try:
        bundle = CompositionRoot.compose(
            lock_path=lock_path,
            mode=args.mode,
            artifact_root=Path.cwd(),
            output_root=Path(args.output_root),
            snapshot_manifest_path=args.snapshot_manifest,
            network_authorized=bool(args.allow_network),
            lock_bytes=input_lock_bytes,
        )
        run_id = f"smoke-{uuid4()}"
        session = bundle.artifact_factory.start_capture(
            run_id=run_id,
            input_lock_bytes=input_lock_bytes,
        )
        budget_profile = next(iter(bundle.service._budgets))  # noqa: SLF001
        execution = await bundle.service.execute(
            SearchRequest(
                query_id=_SMOKE_QUERY_ID,
                query=_SMOKE_QUERY,
                budget_profile=budget_profile,
                mode=args.mode,
            ),
            run_id=run_id,
        )
        session.record_execution(execution)
        if isinstance(execution.outcome, SearchFailure):
            failed_path = session.fail(execution.outcome.error.code)
            return 2, {
                "run_id": run_id,
                "status": "failed",
                "path": str(failed_path),
                "error_code": execution.outcome.error.code,
            }
        if args.mode == "live":
            session.seal()
        published = session.publish()
        return 0, {
            "run_id": run_id,
            "status": "complete",
            "path": str(published),
            "business_result_sha256": execution.business_result_sha256,
        }
    except _SmokeFailure:
        raise
    except ValueError as error:
        if session is not None and session.work_dir.exists():
            session.fail("config_mismatch")
        raise _SmokeFailure("config_mismatch") from error
    except Exception as error:
        if session is not None and session.work_dir.exists():
            session.fail("internal_error")
        raise _SmokeFailure("internal_error") from error
    finally:
        await _close_bundle(bundle)


def _run_serve(args: argparse.Namespace) -> int:
    """Run the only HTTP server boundary after validating local-only binding."""
    if not 1 <= args.port <= 65_535:
        raise ValueError("invalid server binding")

    from paper_search.api.app import create_app

    bundle = CompositionRoot.compose_server(
        replay_lock_path=Path(args.lock),
        snapshot_manifest_path=Path(args.snapshot_manifest),
        artifact_root=Path.cwd(),
        capture_output_root=Path(args.capture_output_root),
        live_authorized=bool(args.allow_live),
    )

    @asynccontextmanager
    async def lifespan(_: object) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await bundle.aclose()

    try:
        import uvicorn

        config = uvicorn.Config(
            create_app(bundle.service_router, lifespan=lifespan),
            host=args.host,
            port=args.port,
            access_log=False,
            log_config=None,
            log_level="critical",
        )
        server = uvicorn.Server(config)
        interrupted = False
        previous_handler = signal.getsignal(signal.SIGINT)

        def request_shutdown(_: int, __: object) -> None:
            nonlocal interrupted
            interrupted = True
            server.should_exit = True

        signal.signal(signal.SIGINT, request_shutdown)
        break_signal = getattr(signal, "SIGBREAK", None)
        previous_break_handler = None
        if break_signal is not None:
            previous_break_handler = signal.getsignal(break_signal)
            signal.signal(break_signal, request_shutdown)
        setattr(server, "install_signal_handlers", lambda: None)
        try:
            try:
                server.run()
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise RuntimeError("server did not start") from error
                raise
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            if break_signal is not None and previous_break_handler is not None:
                signal.signal(break_signal, previous_break_handler)
        if interrupted:
            return 130
        if not server.started:
            raise RuntimeError("server did not start")
        return 0
    finally:
        awaitable = bundle.aclose()
        try:
            asyncio.run(awaitable)
        except RuntimeError:
            awaitable.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify-run":
        return verify_run_command(args.run_directory)
    if args.command == "compare-replay":
        return compare_replay_command(args.capture_run, args.replay_run)
    if args.command == "serve":
        try:
            return _run_serve(args)
        except (KeyboardInterrupt, asyncio.CancelledError):
            return 130
        except (OSError, RuntimeError, TypeError, ValueError):
            print("serve failed: startup error", file=sys.stderr)
            return 2
    if args.command == "evaluate":
        try:
            result = asyncio.run(
                run_evaluation(
                    EvaluationRunRequest(
                        split=args.split,
                        mode=args.mode,
                        lock_path=args.lock,
                        output_root=args.output_root,
                        snapshot_manifest_path=args.snapshot_manifest,
                        network_authorized=bool(args.allow_network),
                    )
                )
            )
            if not isinstance(result, EvaluationRunResult):
                raise TypeError("formal runner returned a legacy result")
        except (KeyboardInterrupt, asyncio.CancelledError):
            return 130
        except (OSError, RuntimeError, TypeError, ValueError):
            print("evaluation failed: invalid input", file=sys.stderr)
            return 2
        except Exception:  # noqa: BLE001
            print("evaluation failed: internal error", file=sys.stderr)
            return 6
        print(
            json.dumps(
                {
                    "gate_result": result.gate_result,
                    "path": str(result.run_path),
                    "run_id": result.run_id,
                    "status": result.status,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 5 if result.gate_result == "failed" else 0
    try:
        exit_code, summary = asyncio.run(_run_smoke(args))
    except _SmokeFailure as error:
        print(f"smoke failed: {error.code}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
