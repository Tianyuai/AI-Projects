from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_search.domain.models import (
    BudgetReservation,
    ErrorDetail,
    ProviderResult,
    QueryAnalysisResult,
    UsageActual,
    UsageEstimate,
)
from paper_search.llm.snapshot_adapters import ReplayLLMAnalyzer
from paper_search.query.parser import QueryParser, normalize_query_analysis
from paper_search.query.planner import QueryPlanner
from paper_search.query.semantic_actions import SEMANTIC_ACTION_PROMPT_VERSION
from paper_search.storage.dependency_snapshot import DependencySnapshotReader


_MALFORMED_CONTENT_CODES = {"invalid_response", "empty_response", "invalid_json"}


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _identity(action: object) -> tuple[str, str, str]:
    action_type = str(getattr(action, "action_type"))
    search_mode = str(getattr(action, "search_mode"))
    text = " ".join(str(getattr(action, "text")).split()).casefold()
    return action_type, search_mode, text


def _provider_result(case: dict[str, Any]) -> ProviderResult[dict[str, Any]]:
    raw_errors = case.get("errors", [])
    errors = (
        [ErrorDetail.model_validate(item) for item in raw_errors]
        if isinstance(raw_errors, list)
        else []
    )
    raw_usage = case.get("usage", {})
    usage = UsageActual.model_validate(raw_usage if isinstance(raw_usage, dict) else {})
    raw_provenance = case.get("provenance", {})
    provenance = dict(raw_provenance) if isinstance(raw_provenance, dict) else {}
    model_output = case.get("model_output")
    if not isinstance(model_output, dict):
        raise ValueError("case model_output must be a JSON object")
    return ProviderResult[dict[str, Any]](
        data=model_output,
        usage=usage,
        provenance=provenance,
        cache_hit=True,
        latency_ms=0,
        errors=errors,
    )


def _canonical_model_analysis(
    model_output: dict[str, Any], query: str
) -> QueryAnalysisResult:
    try:
        return QueryAnalysisResult.model_validate(model_output)
    except ValueError:
        return QueryAnalysisResult.model_validate(
            normalize_query_analysis(model_output, query)
        )


async def build_offline_reparse(
    source_bytes: bytes,
    *,
    source_path: str,
    snapshot_manifest: Path,
) -> dict[str, object]:
    source = json.loads(source_bytes)
    if not isinstance(source, dict):
        raise ValueError("source summary must be a JSON object")
    raw_cases = source.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("source summary must contain cases")

    prompt_sha256 = source.get("prompt_sha256")
    prompt_version = source.get("prompt_version")
    model = source.get("model")
    snapshot_manifest_sha256 = source.get("snapshot_manifest_sha256")
    snapshot_set_id = source.get("snapshot_set_id")
    if not all(
        isinstance(value, str) and value
        for value in (
            prompt_sha256,
            prompt_version,
            model,
            snapshot_manifest_sha256,
            snapshot_set_id,
        )
    ):
        raise ValueError("source summary replay identity is incomplete")
    reader = DependencySnapshotReader(
        snapshot_manifest,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        snapshot_set_id=snapshot_set_id,
    )
    replay = ReplayLLMAnalyzer(
        reader=reader,
        model_id=model,
        prompt_artifact_sha256=prompt_sha256,
        prompt_version=prompt_version,
    )
    parser = QueryParser(
        QueryPlanner(prompt_version=SEMANTIC_ACTION_PROMPT_VERSION)
    )
    cases: list[dict[str, object]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("source case must be a JSON object")
        query = raw_case.get("query")
        case_id = raw_case.get("case_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("source case query is invalid")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("source case id is invalid")

        summary_result = _provider_result(raw_case)
        provider_result = await replay.generate_json(
            prompt_name="query_analyze",
            payload={"query": query},
            reservation=BudgetReservation(
                reservation_id=f"offline-reparse:{case_id}",
                action=f"offline-reparse:{case_id}",
                reserved=UsageEstimate(),
                expires_at=datetime(2030, 1, 1, tzinfo=UTC),
            ),
        )
        blocking_errors = [
            error.code
            for error in provider_result.errors
            if error.code not in _MALFORMED_CONTENT_CODES
        ]
        if blocking_errors:
            raise RuntimeError("sealed response replay failed")
        if provider_result.data != summary_result.data:
            raise RuntimeError("sealed response differs from source summary")
        parsed = await parser.parse(query, provider_result)
        try:
            model_analysis = _canonical_model_analysis(provider_result.data, query)
        except ValueError:
            model_analysis = None
        final_identities = {
            _identity(action) for action in parsed.search_plan.subqueries
        }
        raw_model_actions = (
            [] if model_analysis is None else model_analysis.search_plan.subqueries
        )
        accepted = [
            action
            for action in raw_model_actions
            if _identity(action) in final_identities
        ]
        rejected = [
            action
            for action in raw_model_actions
            if _identity(action) not in final_identities
        ]
        query_identity = " ".join(query.split()).casefold()
        accepted_novel = [
            action
            for action in accepted
            if " ".join(action.text.split()).casefold() != query_identity
        ]
        cases.append(
            {
                "case_id": case_id,
                "query": query,
                "planner_status_before": raw_case.get("planner_status"),
                "planner_status_after": parsed.planner_status,
                "snapshot_replay_matches_summary": True,
                "snapshot_response_sha256": provider_result.provenance.get(
                    "snapshot_response_sha256"
                ),
                "replay_error_codes": [
                    error.code for error in provider_result.errors
                ],
                "query_spec_exclusions_after": list(parsed.query_spec.exclusions),
                "rationale_after": parsed.search_plan.rationale,
                "accepted_model_actions_after": [
                    action.model_dump(mode="json") for action in accepted
                ],
                "accepted_novel_model_actions_after": [
                    action.model_dump(mode="json") for action in accepted_novel
                ],
                "rejected_model_actions_after": [
                    action.model_dump(mode="json") for action in rejected
                ],
                "final_actions_after": [
                    action.model_dump(mode="json")
                    for action in parsed.search_plan.subqueries
                ],
            }
        )

    return {
        "schema_version": "semantic-action-offline-reparse-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "network_calls": 0,
        "replayed_from_snapshot": True,
        "source_summary_path": source_path,
        "source_summary_sha256": _sha256(source_bytes),
        "source_prompt_version": source.get("prompt_version"),
        "source_prompt_sha256": source.get("prompt_sha256"),
        "source_snapshot_set_id": source.get("snapshot_set_id"),
        "source_snapshot_manifest_sha256": source.get(
            "snapshot_manifest_sha256"
        ),
        "snapshot_manifest_path": str(snapshot_manifest.resolve()),
        "all_snapshot_replays_match_summary": all(
            case["snapshot_replay_matches_summary"] for case in cases
        ),
        "case_count": len(cases),
        "primary_count_after": sum(
            case["planner_status_after"] == "primary" for case in cases
        ),
        "accepted_novel_action_count_after": sum(
            len(case["accepted_novel_model_actions_after"]) for case in cases
        ),
        "cases": cases,
    }


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--source", type=Path, required=True)
    argument_parser.add_argument("--snapshot-manifest", type=Path)
    argument_parser.add_argument("--output", type=Path, required=True)
    args = argument_parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists")
    source_bytes = source.read_bytes()
    snapshot_manifest = (
        args.snapshot_manifest.resolve()
        if args.snapshot_manifest is not None
        else source.parent / "capture" / "snapshot-manifest.json"
    )
    result = asyncio.run(
        build_offline_reparse(
            source_bytes,
            source_path=str(source),
            snapshot_manifest=snapshot_manifest,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "network_calls": result["network_calls"],
                "replayed_from_snapshot": result["replayed_from_snapshot"],
                "all_snapshot_replays_match_summary": result[
                    "all_snapshot_replays_match_summary"
                ],
                "case_count": result["case_count"],
                "primary_count_after": result["primary_count_after"],
                "accepted_novel_action_count_after": result[
                    "accepted_novel_action_count_after"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
