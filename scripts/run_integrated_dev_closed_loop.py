from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

from paper_search.application.composition import CompositionRoot
from paper_search.application.contracts import SearchRequest, SearchSuccess
from paper_search.evaluation.dev_closed_loop import (
    aggregate_development_closed_loop,
    score_development_query,
)
from paper_search.evaluation.predictions import (
    paper_evaluation_id,
    prediction_from_response,
)
from paper_search.learning.deployment import (
    build_cpu_pairwise_action_analyzer_decorator,
)
from paper_search.learning.gold_retrievability_audit import FrozenAuditManifest


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_development_rows(
    path: Path,
    *,
    limit: int,
    sample_manifest: Path | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != "pasa":
                raise ValueError("closed-loop canary requires PaSa rows")
            if row.get("split") != "auto_dev" or row.get("role") != "development":
                raise ValueError("closed-loop canary may only use isolated auto_dev rows")
            rows.append(row)
            if sample_manifest is None and len(rows) == limit:
                break
    if sample_manifest is not None:
        manifest = FrozenAuditManifest.model_validate_json(
            sample_manifest.read_text(encoding="utf-8")
        )
        if manifest.split != "auto_dev" or manifest.role != "development":
            raise ValueError("closed-loop sample manifest must be isolated auto_dev")
        if manifest.source_sha256 != _sha256(path):
            raise ValueError("closed-loop sample manifest source hash mismatch")
        if manifest.sample_query_count != limit:
            raise ValueError("limit must equal the frozen sample size")
        by_query_id = {str(row["query_id"]): row for row in rows}
        selected_ids = [item.query_id for item in manifest.sample]
        if any(query_id not in by_query_id for query_id in selected_ids):
            raise ValueError("frozen sample query is absent from development partition")
        rows = [by_query_id[query_id] for query_id in selected_ids]
    if len(rows) != limit:
        raise ValueError("development partition does not contain the requested rows")
    return rows


def _load_environment(path: Path) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in dotenv_values(path).items()
        if value is not None
    }


def _capture_root(workspace: Path) -> Path:
    """Keep hashed snapshot paths below Windows' conservative path boundary."""

    return workspace / "runs" / "e2e-captures"


async def _run(args: argparse.Namespace) -> dict[str, object]:
    rows = _load_development_rows(
        args.partition,
        limit=args.limit,
        sample_manifest=args.sample_manifest,
    )
    environment = _load_environment(args.env_file)
    decorator = build_cpu_pairwise_action_analyzer_decorator(
        model_path=args.model,
        manifest_path=args.model_manifest,
        max_actions=5,
    )
    capture_root = _capture_root(Path.cwd())
    bundle = CompositionRoot.compose(
        lock_path=args.lock,
        mode="live",
        artifact_root=Path.cwd(),
        output_root=capture_root,
        network_authorized=True,
        environ=environment,
        analyzer_decorator=decorator,
    )
    scores = []
    executions: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    try:
        for row in rows:
            query_id = str(row["query_id"])
            execution = await bundle.service.execute(
                SearchRequest(
                    query_id=query_id,
                    query=str(row["query"]),
                    mode="live",
                ),
                run_id=f"integrated-dev-{query_id}-{uuid4().hex[:8]}",
            )
            if not isinstance(execution.outcome, SearchSuccess):
                failures.append(
                    {
                        "query_id": query_id,
                        "error_code": execution.outcome.error.code,
                        "stop_reason": execution.outcome.stop_reason,
                        "usage": execution.outcome.usage.model_dump(mode="json"),
                        "diagnostics": [
                            {
                                "dependency": item.dependency,
                                "endpoint": item.endpoint,
                                "model_id": item.model_id,
                                "error_codes": [error.code for error in item.errors],
                                "usage": item.usage.model_dump(mode="json"),
                                "snapshot_ref_count": len(item.snapshot_refs),
                            }
                            for item in execution.diagnostics
                        ],
                    }
                )
                continue
            response = execution.outcome.response
            candidate_ids = list(
                dict.fromkeys(
                    paper_evaluation_id(paper)
                    for paper in execution.pre_truncation_candidates
                )
            )
            final_ids = prediction_from_response(response).selected_paper_ids
            raw_gold_ids = row["gold_paper_ids"]
            if not isinstance(raw_gold_ids, list):
                raise ValueError("development gold_paper_ids must be a list")
            score = score_development_query(
                query_id=query_id,
                gold_paper_ids=[str(value) for value in raw_gold_ids],
                candidate_paper_ids=candidate_ids,
                final_paper_ids=final_ids,
            )
            scores.append(score)
            retrieval_trace = [
                item for item in response.search_trace if item.get("step") == "retrieve"
            ]
            executions.append(
                {
                    **score.model_dump(mode="json"),
                    "pre_truncation_candidate_count": len(candidate_ids),
                    "final_output_count": len(final_ids),
                    "planner_status": response.planner_status,
                    "planner_fallback": response.planner_fallback,
                    "openalex_lexical_calls": sum(
                        item.get("provider") == "openalex"
                        and item.get("search_mode") == "lexical"
                        for item in retrieval_trace
                    ),
                    "openalex_semantic_calls": sum(
                        item.get("provider") == "openalex"
                        and item.get("search_mode") == "semantic"
                        for item in retrieval_trace
                    ),
                    "semantic_scholar_calls": sum(
                        item.get("provider") == "semantic_scholar"
                        for item in retrieval_trace
                    ),
                    "graph_expansion_calls": sum(
                        item.get("step") == "citation" for item in response.search_trace
                    ),
                }
            )
    finally:
        await bundle.aclose()

    summary = aggregate_development_closed_loop(scores)
    return {
        "schema_version": "integrated-dev-closed-loop-v2",
        "dataset": "pasa",
        "split": "auto_dev",
        "test_partition_touched": False,
        "requested_query_count": len(rows),
        "completed_query_count": len(scores),
        "failure_count": len(failures),
        "candidate_oracle_macro_recall": summary.candidate_oracle_macro_recall,
        "final_macro_recall": summary.final_macro_recall,
        "oracle_final_macro_gap": summary.oracle_final_macro_gap,
        "model_sha256": _sha256(args.model),
        "model_manifest_sha256": _sha256(args.model_manifest),
        "sample_manifest_sha256": (
            _sha256(args.sample_manifest) if args.sample_manifest is not None else None
        ),
        "retrieval_policy": "fixed-hybrid-openalex-v1",
        "graph_enabled": False,
        "executions": executions,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("runs/candidate.lock.yaml"))
    parser.add_argument(
        "--partition",
        type=Path,
        default=Path("data/training_private/freeze-v1/partitions/pasa_auto_dev.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "data/training_private/models/cpu-pairwise-action-ranker-openalex-v1.f64"
        ),
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("data/training/cpu-pairwise-action-ranker-openalex-v1.json"),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(r"D:\AI Projects\Projects\.env"),
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    payload = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if not payload["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
