from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values

from paper_search.control.budget import HardBudgetController
from paper_search.control.pricing import ActualCostPricer, parse_pricing_policy_bytes
from paper_search.domain.models import (
    BudgetReservation,
    QueryAnalysisResult,
    SearchBudget,
    UsageActual,
    UsageEstimate,
)
from paper_search.llm.client import OpenAICompatibleLLMClient
from paper_search.llm.prompt_artifacts import render_prompt_system_message
from paper_search.llm.snapshot_adapters import (
    LiveCaptureLLMAnalyzer,
    ReplayLLMAnalyzer,
)
from paper_search.query.parser import QueryParser, normalize_query_analysis
from paper_search.query.planner import QueryPlanner
from paper_search.query.semantic_actions import SEMANTIC_ACTION_PROMPT_VERSION
from paper_search.storage.dependency_snapshot import (
    DependencyCaptureStore,
    DependencySnapshotReader,
)


_MODEL = "deepseek-v4-flash"
_BASE_URL = "https://api.deepseek.com/v1"
_QUERIES = (
    (
        "negation-relation",
        "Find empirical studies that predict molecular properties with graph "
        "attention networks but do not require 3D conformers.",
    ),
    (
        "method-bridge",
        "Which papers use weakly supervised optimal transport to adapt medical "
        "image segmentation models across hospitals?",
    ),
    (
        "unconstrained-cross-vocabulary",
        "What research learns compact representations of long scientific "
        "documents so that conceptually related papers can be retrieved even "
        "when they use different terminology?",
    ),
)


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_analysis(data: dict[str, Any], query: str) -> QueryAnalysisResult:
    try:
        return QueryAnalysisResult.model_validate(data)
    except ValueError:
        return QueryAnalysisResult.model_validate(
            normalize_query_analysis(data, query)
        )


def _normalized_identity(text: str, search_mode: str) -> tuple[str, str]:
    return search_mode, " ".join(text.split()).casefold()


def _validation_budget() -> SearchBudget:
    return SearchBudget(
        max_search_api_calls=1,
        target_search_api_calls=0,
        max_llm_calls=3,
        target_llm_calls=3,
        max_iterations=1,
        max_subqueries=5,
        max_rerank_candidates=0,
        max_output_papers=1,
        max_citation_seeds=0,
        target_citation_seeds=0,
        max_elapsed_seconds=240,
        soft_deadline_seconds=230,
        max_total_tokens=12_000,
        max_cost_cny=1.0,
    )


def _result_summary(
    *,
    case_id: str,
    query: str,
    raw_result: Any,
    parsed: Any,
) -> dict[str, object]:
    try:
        model_analysis = _canonical_analysis(dict(raw_result.data), query)
    except ValueError:
        model_analysis = None
    final_identities = {
        _normalized_identity(item.text, item.search_mode)
        for item in parsed.search_plan.subqueries
    }
    raw_actions = [] if model_analysis is None else model_analysis.search_plan.subqueries
    accepted = [
        item.model_dump(mode="json")
        for item in raw_actions
        if _normalized_identity(item.text, item.search_mode) in final_identities
    ]
    rejected = [
        item.model_dump(mode="json")
        for item in raw_actions
        if _normalized_identity(item.text, item.search_mode) not in final_identities
    ]
    pricing_receipt: object | None = None
    raw_pricing = raw_result.provenance.get("pricing_receipt")
    if raw_pricing:
        pricing_receipt = json.loads(raw_pricing)
    return {
        "case_id": case_id,
        "query": query,
        "planner_status": parsed.planner_status,
        "model_output": raw_result.data,
        "research_goal": parsed.query_spec.research_goal,
        "rationale": parsed.search_plan.rationale,
        "accepted_model_actions": accepted,
        "rejected_model_actions": rejected,
        "final_actions": [
            item.model_dump(mode="json") for item in parsed.search_plan.subqueries
        ],
        "usage": raw_result.usage.model_dump(mode="json"),
        "pricing_receipt": pricing_receipt,
        "errors": [item.model_dump(mode="json") for item in raw_result.errors],
        "provenance": dict(raw_result.provenance),
    }


async def _run(args: argparse.Namespace) -> None:
    output_root = args.output.resolve()
    if output_root.exists():
        raise ValueError("output directory already exists")
    output_root.mkdir(parents=True)

    environment = dotenv_values(args.env_file)
    api_key = environment.get("LLM_API_KEY")
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("LLM_API_KEY is unavailable")

    prompt_bytes = args.prompt.read_bytes()
    prompt_sha256 = _sha256(prompt_bytes)
    prompt_instructions = render_prompt_system_message(prompt_bytes)
    pricer = ActualCostPricer(
        parse_pricing_policy_bytes(args.pricing_policy.read_bytes())
    )
    estimate_actual = pricer.value_actual(
        dependency="llm",
        model_or_adapter=_MODEL,
        usage=UsageActual(
            llm_calls=1,
            input_tokens=2_500,
            uncached_input_tokens=2_500,
            output_tokens=1_500,
        ),
    )
    estimate = UsageEstimate(
        llm_calls=1,
        input_tokens=2_500,
        uncached_input_tokens=2_500,
        output_tokens=1_500,
        cost_cny=estimate_actual.cost_cny,
        elapsed_ms=60_000,
    )
    controller = HardBudgetController(_validation_budget(), formal_live=True)
    capture = DependencyCaptureStore(output_root / "capture")
    parser = QueryParser(
        QueryPlanner(prompt_version=SEMANTIC_ACTION_PROMPT_VERSION)
    )
    live_results: list[tuple[str, str, Any]] = []
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=60, write=20, pool=5)
    ) as http_client:
        llm_client = OpenAICompatibleLLMClient(
            client=http_client,
            base_url=_BASE_URL,
            model=_MODEL,
            api_key=api_key,
            prompt_version=SEMANTIC_ACTION_PROMPT_VERSION,
        )
        analyzer = LiveCaptureLLMAnalyzer(
            client=llm_client,
            capture_store=capture,
            pricer=pricer,
            controller=controller,
            prompt_artifact_sha256=prompt_sha256,
            prompt_instructions=prompt_instructions,
        )
        case_summaries: list[dict[str, object]] = []
        for index, (case_id, query) in enumerate(_QUERIES, start=1):
            reservation = controller.reserve(
                f"semantic-action-validation:{index}",
                estimate,
            )
            result = await analyzer.generate_json(
                prompt_name="query_analyze",
                payload={"query": query},
                reservation=reservation,
            )
            parsed = await parser.parse(query, result)
            case_summaries.append(
                _result_summary(
                    case_id=case_id,
                    query=query,
                    raw_result=result,
                    parsed=parsed,
                )
            )
            live_results.append((case_id, query, result))

    manifest = capture.seal()
    reader = DependencySnapshotReader(
        capture.manifest_path,
        snapshot_manifest_sha256=capture.manifest_sha256,
        snapshot_set_id=manifest.snapshot_set_id,
    )
    replay = ReplayLLMAnalyzer(
        reader=reader,
        model_id=_MODEL,
        prompt_artifact_sha256=prompt_sha256,
        prompt_version=SEMANTIC_ACTION_PROMPT_VERSION,
    )
    replay_matches: list[bool] = []
    for index, (_case_id, query, live_result) in enumerate(live_results, start=1):
        replay_result = await replay.generate_json(
            prompt_name="query_analyze",
            payload={"query": query},
            reservation=BudgetReservation(
                reservation_id=f"replay-{index}",
                action=f"semantic-action-replay:{index}",
                reserved=estimate,
                expires_at=datetime(2030, 1, 1, tzinfo=UTC),
            ),
        )
        replay_matches.append(
            replay_result.data == live_result.data
            and replay_result.errors == live_result.errors
        )

    summary = {
        "schema_version": "semantic-action-llm-live-validation-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model": _MODEL,
        "prompt_version": SEMANTIC_ACTION_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "network_scope": ["llm"],
        "query_source": "synthetic_non_gold_non_final_test",
        "snapshot_set_id": manifest.snapshot_set_id,
        "snapshot_manifest_sha256": capture.manifest_sha256,
        "replay_matches": replay_matches,
        "all_replay_matches": all(replay_matches),
        "committed_usage": controller.committed_usage.model_dump(mode="json"),
        "cases": case_summaries,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    secret = api_key.encode("utf-8")
    leaked_paths = [
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file() and secret in path.read_bytes()
    ]
    if leaked_paths:
        raise RuntimeError("credential bytes detected in validation artifacts")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "case_count": len(case_summaries),
                "all_replay_matches": all(replay_matches),
                "credential_leak_detected": False,
                "snapshot_set_id": manifest.snapshot_set_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("configs/prompts/query_analyze_semantic_actions_v2.yaml"),
    )
    parser.add_argument(
        "--pricing-policy",
        type=Path,
        default=Path("data/annotation_work/pricing_v1.yaml"),
    )
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
