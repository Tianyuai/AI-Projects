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
from typing import TYPE_CHECKING
from uuid import uuid4

from paper_search.application.composition import ApplicationBundle, CompositionRoot
from paper_search.application.contracts import (
    SearchErrorCode,
    SearchFailure,
    SearchRequest,
)
from paper_search.config import RuntimeConfig, load_runtime_config
from paper_search.evaluation.runner import (
    EvaluationRunRequest,
    EvaluationRunResult,
    run_evaluation,
)
from paper_search.evaluation.validator import (
    compare_replay_command,
    verify_run_command,
)

if TYPE_CHECKING:
    from paper_search.recall_experiments.composition import RecallRuntimeFactory


_SMOKE_QUERY = "resource-aware scholarly paper search"
_SMOKE_QUERY_ID = "smoke-query-1"
_PROJECT_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _resolve_project_config(path: Path) -> Path:
    if path.exists():
        return path
    fallback = _PROJECT_CONFIG_ROOT / path.name
    return fallback if fallback.exists() else path


def _load_optional_runtime_config(path: Path | None) -> RuntimeConfig | None:
    if path is None:
        return None
    return load_runtime_config(_resolve_project_config(path))


def _resolve_ablation_config(config_path: Path | None) -> Path:
    if config_path is not None:
        sibling = config_path.parent / "ablations.yaml"
        if sibling.exists():
            return sibling
    return _PROJECT_CONFIG_ROOT / "ablations.yaml"


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
    smoke.add_argument("--config", type=Path)
    evaluate = commands.add_parser("evaluate", help="run one formal evaluation")
    evaluate.add_argument("--lock", type=Path, required=True)
    evaluate.add_argument("--split", choices=("dev", "validation"), required=True)
    evaluate.add_argument("--mode", choices=("replay", "live"), default="replay")
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--snapshot-manifest", type=Path)
    evaluate.add_argument("--allow-network", action="store_true")
    evaluate.add_argument("--config", type=Path)
    verify = commands.add_parser("verify-run", help="verify a formal run")
    verify.add_argument("run_directory", type=Path)
    compare = commands.add_parser("compare-replay", help="compare capture and replay")
    compare.add_argument("capture_run", type=Path)
    compare.add_argument("replay_run", type=Path)
    ranking = commands.add_parser(
        "ranking-metrics",
        help="compute standalone ranked-retrieval metrics (MRR/NDCG)",
    )
    ranking.add_argument("--gold", type=Path, required=True)
    ranking.add_argument("--pred", type=Path, required=True)
    ranking.add_argument("--out", type=Path, required=True)
    ranking.add_argument("--id-map", type=Path)
    serve = commands.add_parser("serve", help="serve the canonical replay API")
    serve.add_argument("--lock", type=Path, required=True)
    serve.add_argument("--mode", choices=("replay",), required=True)
    serve.add_argument("--snapshot-manifest", type=Path, required=True)
    serve.add_argument("--capture-output-root", type=Path, required=True)
    serve.add_argument("--allow-live", action="store_true")
    serve.add_argument("--config", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    recall = commands.add_parser("recall", help="run isolated candidate-recall workflows")
    recall_commands = recall.add_subparsers(dest="recall_command", required=True)
    prepare = recall_commands.add_parser(
        "prepare-context", help="verify frozen inputs and write safe generation contexts"
    )
    prepare.add_argument("--recipe", type=Path, required=True)
    prepare.add_argument("--sample", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    validate = recall_commands.add_parser(
        "validate-actions", help="validate pasted action batches without retrieval"
    )
    validate.add_argument("--recipe", type=Path, required=True)
    validate.add_argument("--contexts", type=Path, required=True)
    validate.add_argument("--actions", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    run_recall = recall_commands.add_parser("run", help="run one candidate-recall recipe")
    run_recall.add_argument("--recipe", type=Path, required=True)
    run_recall.add_argument("--sample", type=Path, required=True)
    run_recall.add_argument("--actions", type=Path)
    run_recall.add_argument("--snapshot-manifest", type=Path)
    run_recall.add_argument("--allow-live", action="store_true")
    run_recall.add_argument("--out", type=Path, required=True)
    compare_recall = recall_commands.add_parser("compare", help="compare explicit recall artifacts")
    compare_recall.add_argument("--current", type=Path, required=True)
    compare_recall.add_argument("--historical", type=Path)
    compare_recall.add_argument("--out", type=Path, required=True)
    inventory = recall_commands.add_parser(
        "inventory-history", help="verify frozen historical recall evidence"
    )
    inventory.add_argument(
        "--config-root", type=Path, default=_PROJECT_CONFIG_ROOT / "recall_experiments" / "historical"
    )
    inventory.add_argument("--out", type=Path, required=True)
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
        config_path = getattr(args, "config", None)
        runtime_config = _load_optional_runtime_config(config_path)
        bundle = CompositionRoot.compose(
            lock_path=lock_path,
            mode=args.mode,
            artifact_root=Path.cwd(),
            output_root=Path(args.output_root),
            snapshot_manifest_path=args.snapshot_manifest,
            network_authorized=bool(args.allow_network),
            lock_bytes=input_lock_bytes,
            runtime_config=runtime_config,
            ablation_config=_resolve_ablation_config(config_path),
        )
        run_id = f"smoke-{uuid4()}"
        session = bundle.artifact_factory.start_capture(
            run_id=run_id,
            input_lock_bytes=input_lock_bytes,
            expected_config_hash=bundle.config_hash,
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
        runtime_config=_load_optional_runtime_config(getattr(args, "config", None)),
        ablation_config=_resolve_ablation_config(getattr(args, "config", None)),
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
        previous_term_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, request_shutdown)
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
            signal.signal(signal.SIGTERM, previous_term_handler)
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


def main(
    argv: Sequence[str] | None = None,
    *,
    recall_runtime_factory: RecallRuntimeFactory | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "recall":
        return _run_recall_command(args, recall_runtime_factory=recall_runtime_factory)
    if args.command == "verify-run":
        return verify_run_command(args.run_directory)
    if args.command == "compare-replay":
        return compare_replay_command(args.capture_run, args.replay_run)
    if args.command == "ranking-metrics":
        from paper_search.evaluation.ranking_metrics import run_cli

        return run_cli(args.gold, args.pred, args.out, args.id_map)
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
            runtime_config = _load_optional_runtime_config(args.config)
            result = asyncio.run(
                run_evaluation(
                    EvaluationRunRequest(
                        split=args.split,
                        mode=args.mode,
                        lock_path=args.lock,
                        output_root=args.output_root,
                        snapshot_manifest_path=args.snapshot_manifest,
                        network_authorized=bool(args.allow_network),
                        runtime_config=runtime_config,
                        ablation_config_path=_resolve_ablation_config(args.config),
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


def _run_recall_command(
    args: argparse.Namespace,
    *,
    recall_runtime_factory: RecallRuntimeFactory | None,
) -> int:
    """Dispatch recall composition without opening runtime state for offline paths."""
    from paper_search.recall_experiments.composition import (
        RecallTerminalError,
        compare_recall_artifacts,
        prepare_recall_run,
        run_recall_experiment,
        validate_pasted_actions,
        write_prepared_contexts,
    )

    output = Path(getattr(args, "out", Path(".")))
    summary: dict[str, object]
    try:
        if args.recall_command == "prepare-context":
            prepared = prepare_recall_run(args.recipe, args.sample, workspace_root=Path.cwd())
            path = write_prepared_contexts(prepared, output)
        elif args.recall_command == "validate-actions":
            path = validate_pasted_actions(
                recipe_path=args.recipe,
                contexts_path=args.contexts,
                actions_path=args.actions,
                output_path=output,
            )
        elif args.recall_command == "run":
            path = asyncio.run(
                run_recall_experiment(
                    recipe_path=args.recipe,
                    sample_path=args.sample,
                    output_path=output,
                    workspace_root=Path.cwd(),
                    actions_path=args.actions,
                    allow_live=bool(args.allow_live),
                    snapshot_manifest_path=args.snapshot_manifest,
                    live_runtime_factory=recall_runtime_factory,
                )
            )
        elif args.recall_command == "compare":
            comparison = compare_recall_artifacts(
                current_run=args.current,
                historical_run=args.historical,
                output_path=output,
            )
            path = output
            summary = {"path": str(path), "status": "complete", **comparison}
        elif args.recall_command == "inventory-history":
            from paper_search.recall_experiments.inventory import build_inventory

            report = build_inventory(args.config_root, workspace_root=Path.cwd())
            output.mkdir(parents=True, exist_ok=False)
            (output / "source-inventory.json").write_text(
                json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
            )
            path = output
        else:
            raise RecallTerminalError("config_mismatch")
    except RecallTerminalError as error:
        print(
            json.dumps(
                {"error_code": error.code, "path": str(output), "status": "failed"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    except (OSError, RuntimeError, TypeError, ValueError):
        print(
            json.dumps(
                {"error_code": "config_mismatch", "path": str(output), "status": "failed"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    if args.recall_command != "compare":
        summary = {"path": str(path), "status": "complete"}
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
