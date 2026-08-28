from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from paper_search.domain.models import (
    BudgetReservation,
    ProviderResult,
    QueryAnalysisResult,
    UsageActual,
)
from paper_search.evaluation.dev_closed_loop import (
    aggregate_development_closed_loop,
    score_development_query,
)
from paper_search.evaluation.predictions import paper_evaluation_id
from paper_search.learning.deployment import build_cpu_pairwise_action_analyzer_decorator
from paper_search.processing.deduplicate import deduplicate_papers
from paper_search.processing.filter import apply_hard_filters
from paper_search.query.parser import normalize_query_analysis
from paper_search.ranking.fusion import fuse_provider_results
from audit_integrated_dev_failures import (
    _analysis_and_routes,
    _capture_manifests,
    _read_jsonl,
)


def _provider_result(route_index: int, papers: list[Any]) -> ProviderResult[list[Any]]:
    return ProviderResult[list[Any]](
        data=papers,
        usage=UsageActual(search_api_calls=1),
        provenance={
            "provider": "openalex",
            "endpoint": "/works",
            "model_id": "openalex-works-v1",
            "requested_at": datetime(2026, 8, 16, tzinfo=UTC).isoformat(),
            "response_hash": "sha256:" + f"{route_index:064x}",
        },
        cache_hit=True,
        latency_ms=0,
        errors=[],
    )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    baseline = json.loads(args.result.read_text(encoding="utf-8"))
    executions = baseline["executions"]
    partition = {row["query_id"]: row for row in _read_jsonl(args.partition)}
    manifests = _capture_manifests(
        args.capture_root,
        captured_after=datetime.fromisoformat(args.captured_after.replace("Z", "+00:00")),
        expected=len(executions),
    )
    decorator = build_cpu_pairwise_action_analyzer_decorator(
        model_path=args.model,
        manifest_path=args.model_manifest,
        max_actions=5,
    )

    async def unexpected_fallback(query: str, reservation: Any) -> Any:
        del query, reservation
        raise ValueError("local analysis unexpectedly requested fallback")

    local_analyzer = decorator(unexpected_fallback)
    scores = []
    replayed: list[dict[str, Any]] = []
    for execution, manifest_path in zip(executions, manifests, strict=True):
        row = partition[execution["query_id"]]
        content, routes = _analysis_and_routes(manifest_path)
        if content is None:
            local_result = await local_analyzer(
                str(row["query"]), cast(BudgetReservation, None)
            )
            content = local_result.data
        analysis = QueryAnalysisResult.model_validate(
            normalize_query_analysis(content, str(row["query"]))
        )
        merged = deduplicate_papers([paper for route in routes for paper in route])
        accepted_ids = {
            item.paper.canonical_id
            for item in apply_hard_filters(merged.papers, analysis.query_spec).accepted
        }
        fused = fuse_provider_results(
            {
                f"openalex:route-{index}": _provider_result(index, papers)
                for index, papers in enumerate(routes, start=1)
            },
            method="rrf",
        )
        candidates = [item.paper for item in fused if item.paper.canonical_id in accepted_ids]
        candidate_ids = list(dict.fromkeys(paper_evaluation_id(paper) for paper in candidates))
        final_ids = list(
            dict.fromkeys(paper_evaluation_id(paper) for paper in candidates[:50])
        )
        gold_ids = [str(value) for value in row["gold_paper_ids"]]
        score = score_development_query(
            query_id=str(row["query_id"]),
            gold_paper_ids=gold_ids,
            candidate_paper_ids=candidate_ids,
            final_paper_ids=final_ids,
        )
        scores.append(score)
        gold_set = {value.casefold() for value in gold_ids}
        corrected_positions = {
            paper_id: index
            for index, paper_id in enumerate(candidate_ids, start=1)
            if paper_id in gold_set
        }
        replayed.append(
            {
                **score.model_dump(mode="json"),
                "baseline_candidate_oracle_recall": execution[
                    "candidate_oracle_recall"
                ],
                "baseline_final_recall": execution["final_recall"],
                "corrected_gold_positions": corrected_positions,
            }
        )
    summary = aggregate_development_closed_loop(scores)
    return {
        "schema_version": "integrated-dev-fixed-replay-v1",
        "source_result": args.result.as_posix(),
        "sample_manifest_sha256": baseline["sample_manifest_sha256"],
        "test_partition_touched": False,
        "query_count": len(replayed),
        "baseline_candidate_oracle_macro_recall": baseline[
            "candidate_oracle_macro_recall"
        ],
        "corrected_candidate_oracle_macro_recall": summary.candidate_oracle_macro_recall,
        "baseline_final_macro_recall": baseline["final_macro_recall"],
        "corrected_final_macro_recall": summary.final_macro_recall,
        "corrected_oracle_final_macro_gap": summary.oracle_final_macro_gap,
        "executions": replayed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--captured-after", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "executions"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
