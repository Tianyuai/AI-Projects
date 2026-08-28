from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import httpx
from dotenv import dotenv_values

from paper_search.evaluation.predictions import paper_evaluation_id
from paper_search.learning.deployment import build_cpu_pairwise_action_analyzer_decorator
from paper_search.learning.gold_retrievability_audit import probe_gold_identifier
from paper_search.processing.deduplicate import deduplicate_papers
from paper_search.processing.filter import apply_hard_filters
from paper_search.processing.normalize import normalize_openalex_work
from paper_search.query.parser import normalize_query_analysis
from paper_search.domain.models import BudgetReservation, QueryAnalysisResult


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _capture_manifests(root: Path, *, captured_after: datetime, expected: int) -> list[Path]:
    selected: list[tuple[datetime, Path]] = []
    for path in root.rglob("snapshot-manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sealed_at = datetime.fromisoformat(str(payload["sealed_at"]).replace("Z", "+00:00"))
        if sealed_at >= captured_after:
            selected.append((sealed_at, path))
    selected.sort()
    if len(selected) != expected:
        raise ValueError(f"expected {expected} captures after boundary, found {len(selected)}")
    return [path for _, path in selected]


def _analysis_and_routes(
    manifest_path: Path,
) -> tuple[dict[str, Any] | None, list[list[Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    llm_entries = [
        item for item in manifest["entries"] if item["request"]["dependency"] == "llm"
    ]
    if len(llm_entries) > 1:
        raise ValueError("audit found multiple captured analyzer responses")
    content = None
    if llm_entries:
        llm_payload = json.loads((base / llm_entries[0]["response_path"]).read_bytes())
        content = json.loads(llm_payload["choices"][0]["message"]["content"])

    openalex_entries = sorted(
        (
            item
            for item in manifest["entries"]
            if item["request"]["dependency"] == "openalex"
        ),
        key=lambda item: item["captured_at"],
    )
    if len(openalex_entries) != 6:
        raise ValueError("audit requires the frozen six-action OpenAlex policy")
    routes: list[list[Any]] = []
    for entry in openalex_entries:
        payload = json.loads((base / entry["response_path"]).read_bytes())
        routes.append([normalize_openalex_work(item) for item in payload["results"]])
    return content, routes


def _route_gold_hits(routes: list[list[Any]], gold_ids: set[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, papers in enumerate(routes):
        hits = [
            {"gold_id": paper_evaluation_id(paper), "rank": rank}
            for rank, paper in enumerate(papers, start=1)
            if paper_evaluation_id(paper) in gold_ids
        ]
        output.append(
            {
                "route_index": index + 1,
                "search_mode": "lexical" if index % 2 == 0 else "semantic",
                "subquery_index": index // 2 + 1,
                "gold_hits": hits,
            }
        )
    return output


async def _availability(gold_ids: list[str], api_key: str | None) -> dict[str, dict[str, Any]]:
    semaphore = asyncio.Semaphore(4)
    async with httpx.AsyncClient(timeout=20.0) as client:
        async def probe(gold_id: str) -> tuple[str, dict[str, Any]]:
            async with semaphore:
                record = await probe_gold_identifier(gold_id, client=client, api_key=api_key)
                return gold_id, record.model_dump(mode="json")

        return dict(await asyncio.gather(*(probe(gold_id) for gold_id in gold_ids)))


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    result = json.loads(args.result.read_text(encoding="utf-8"))
    executions = result["executions"]
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

    query_audits: list[dict[str, Any]] = []
    probe_ids: set[str] = set()
    for execution, manifest_path in zip(executions, manifests, strict=True):
        query_id = execution["query_id"]
        row = partition[query_id]
        gold_ids = {str(value).casefold() for value in row["gold_paper_ids"]}
        content, routes = _analysis_and_routes(manifest_path)
        analyzer_source = "llm_fallback"
        if content is None:
            local_result = await local_analyzer(
                str(row["query"]), cast(BudgetReservation, None)
            )
            content = local_result.data
            analyzer_source = "cpu_pairwise"
        normalized = normalize_query_analysis(content, str(row["query"]))
        analysis = QueryAnalysisResult.model_validate(normalized)

        combined = []
        seen: set[str] = set()
        for papers in routes:
            for paper in papers:
                if paper.canonical_id not in seen:
                    seen.add(paper.canonical_id)
                    combined.append(paper)
        deduplicated = deduplicate_papers(combined)
        filtered = apply_hard_filters(deduplicated.papers, analysis.query_spec)
        candidates = [item.paper for item in filtered.accepted]
        candidate_positions: dict[str, int] = {}
        evaluation_candidate_ids: list[str] = []
        seen_evaluation_ids: set[str] = set()
        for rank, paper in enumerate(candidates, start=1):
            evaluation_id = paper_evaluation_id(paper)
            if evaluation_id not in seen_evaluation_ids:
                seen_evaluation_ids.add(evaluation_id)
                evaluation_candidate_ids.append(evaluation_id)
            if evaluation_id in gold_ids:
                candidate_positions.setdefault(evaluation_id, rank)
        if len(evaluation_candidate_ids) != execution["pre_truncation_candidate_count"]:
            raise ValueError(
                "candidate reconstruction mismatch for "
                f"{query_id}: raw={sum(len(route) for route in routes)}, "
                f"provider_unique={len(combined)}, deduplicated={len(deduplicated.papers)}, "
                f"accepted={len(candidates)}, evaluation_unique={len(evaluation_candidate_ids)}, "
                f"expected={execution['pre_truncation_candidate_count']}"
            )
        if len(candidate_positions) != execution["candidate_hit_count"]:
            raise ValueError(f"Gold reconstruction mismatch for {query_id}")

        if execution["candidate_hit_count"] == 0:
            failure_stage = "candidate_generation"
        elif execution["oracle_final_gap"] > 0:
            failure_stage = "ranking_truncation"
        else:
            failure_stage = "retained_or_partial"
        if failure_stage != "retained_or_partial":
            probe_ids.update(gold_ids)
        route_hits = _route_gold_hits(routes, gold_ids)
        raw_gold_ids = {
            paper_evaluation_id(paper)
            for route in routes
            for paper in route
            if paper_evaluation_id(paper) in gold_ids
        }
        dedup_identity_losses: list[dict[str, Any]] = []
        for gold_id in sorted(raw_gold_ids.difference(candidate_positions)):
            member_ids = {
                paper.canonical_id
                for route in routes
                for paper in route
                if paper_evaluation_id(paper) == gold_id
            }
            for decision in deduplicated.decisions:
                if member_ids.intersection(decision.member_ids):
                    representative = next(
                        paper
                        for paper in deduplicated.papers
                        if paper.canonical_id == decision.representative_id
                    )
                    dedup_identity_losses.append(
                        {
                            "gold_id": gold_id,
                            "match_rule": decision.match_rule,
                            "representative_id": decision.representative_id,
                            "representative_evaluation_id": paper_evaluation_id(
                                representative
                            ),
                            "member_ids": decision.member_ids,
                        }
                    )
                    break
        candidate_failure_cause = None
        if failure_stage == "candidate_generation":
            candidate_failure_cause = (
                "dedup_identifier_loss"
                if dedup_identity_losses
                else "search_actions_no_hit"
            )
        query_audits.append(
            {
                "query_id": query_id,
                "query": row["query"],
                "gold_ids": sorted(gold_ids),
                "failure_stage": failure_stage,
                "candidate_failure_cause": candidate_failure_cause,
                "analyzer_source": analyzer_source,
                "subqueries": [item.text for item in analysis.search_plan.subqueries[:3]],
                "candidate_count": len(evaluation_candidate_ids),
                "candidate_gold_positions": candidate_positions,
                "route_gold_hits": route_hits,
                "dedup_identity_losses": dedup_identity_losses,
                "raw_route_result_counts": [len(papers) for papers in routes],
            }
        )

    environment = dotenv_values(args.env_file) if args.env_file else {}
    api_key_value = environment.get("OPENALEX_API_KEY")
    availability: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        for item in prior.get("query_audits", []):
            availability.update(item.get("gold_availability", {}))
    missing_probe_ids = sorted(probe_ids.difference(availability))
    if missing_probe_ids:
        availability.update(
            await _availability(
                missing_probe_ids,
                str(api_key_value) if api_key_value else None,
            )
        )
    for item in query_audits:
        item["gold_availability"] = {
            gold_id: availability[gold_id]
            for gold_id in item["gold_ids"]
            if gold_id in availability
        }

    candidate_failures = [item for item in query_audits if item["failure_stage"] == "candidate_generation"]
    ranking_failures = [item for item in query_audits if item["failure_stage"] == "ranking_truncation"]
    candidate_cause_counts = Counter(
        item["candidate_failure_cause"] for item in candidate_failures
    )
    status_counts = Counter(
        record["status"]
        for item in candidate_failures
        for record in item["gold_availability"].values()
    )
    ranking_positions = [
        position
        for item in ranking_failures
        for position in item["candidate_gold_positions"].values()
        if position > 50
    ]
    return {
        "schema_version": "integrated-dev-failure-audit-v1",
        "source_result": args.result.as_posix(),
        "sample_manifest_sha256": result["sample_manifest_sha256"],
        "test_partition_touched": False,
        "query_count": len(query_audits),
        "candidate_generation_failure_query_count": len(candidate_failures),
        "ranking_truncation_failure_query_count": len(ranking_failures),
        "candidate_failure_gold_availability": dict(sorted(status_counts.items())),
        "candidate_failure_cause_counts": dict(sorted(candidate_cause_counts.items())),
        "ranking_lost_gold_positions": sorted(ranking_positions),
        "query_audits": query_audits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--captured-after", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "query_audits"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
