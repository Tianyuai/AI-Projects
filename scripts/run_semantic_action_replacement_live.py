"""Run a disjoint paired full-live gate for six-slot LLM action replacement.

The network boundary receives query text only. Gold identifiers are loaded from
the frozen training partition only after both live conditions have completed.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)


SEMANTIC_PROMPT_VERSION = "query-analyze-semantic-actions-v2"
SEMANTIC_PROMPT_PATH = "configs/prompts/query_analyze_semantic_actions_v2.yaml"
_APPROVAL_REF = "user-authorized-semantic-action-six-budget-canary-2026-08-28"
_ENTITY = re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9-]*|[A-Z]+\d+[A-Za-z0-9-]*)\b")
_QUERY_ID = re.compile(r"AutoScholarQuery_train_\d+")
_DEFAULT_STRATUM_QUOTAS = {
    "unconstrained": 6,
    "method": 6,
    "dataset": 5,
    "negation": 3,
    "entity": 4,
}


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def parse_stratum_quotas(value: str | None) -> dict[str, int]:
    """Parse an explicit 24-query allocation without silently filling a stratum."""

    if value is None:
        return dict(_DEFAULT_STRATUM_QUOTAS)
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("stratum quotas must be valid JSON") from error
    if (
        not isinstance(raw, dict)
        or set(raw).difference(_DEFAULT_STRATUM_QUOTAS)
        or any(type(item) is not int or item < 0 for item in raw.values())
    ):
        raise ValueError("stratum quotas are invalid")
    quotas = {str(key): int(item) for key, item in raw.items()}
    if sum(quotas.values()) != 24:
        raise ValueError("stratum quotas must total 24")
    return quotas


def _normalized(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _changed_paths(before: object, after: object, prefix: str = "") -> set[str]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        changed: set[str] = set()
        for key in set(before).union(after):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                changed.add(path)
            else:
                changed.update(_changed_paths(before[key], after[key], path))
        return changed
    return set() if before == after else {prefix}


def build_candidate_lock_bytes(
    production_lock_bytes: bytes,
    *,
    prompt_bytes: bytes,
    prompt_path: str = SEMANTIC_PROMPT_PATH,
    prompt_version: str = SEMANTIC_PROMPT_VERSION,
    approval_ref: str = _APPROVAL_REF,
) -> bytes:
    """Bind the semantic prompt while proving every other production field stable."""

    source = yaml.safe_load(production_lock_bytes)
    if not isinstance(source, dict):
        raise ValueError("production lock is invalid")
    candidate = copy.deepcopy(source)
    baseline = candidate.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("production baseline lock is invalid")
    planner = baseline.get("planner")
    if not isinstance(planner, dict):
        raise ValueError("production planner lock is invalid")
    baseline["prompt_version"] = prompt_version
    planner["prompt_config"] = {
        "path": prompt_path,
        "sha256": _sha256(prompt_bytes),
    }
    candidate["approval_ref"] = approval_ref
    allowed = {
        "approval_ref",
        "baseline.prompt_version",
        "baseline.planner.prompt_config.path",
        "baseline.planner.prompt_config.sha256",
    }
    changed = _changed_paths(source, candidate)
    if changed != allowed:
        raise ValueError(f"candidate lock changed unexpected fields: {sorted(changed)}")
    return yaml.safe_dump(
        candidate,
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")


def _labels(row: Mapping[str, object]) -> list[str]:
    raw = row.get("labels", [])
    return [str(value) for value in raw] if isinstance(raw, list) else []


def _stratum(
    query: str,
    labels: Sequence[str],
    *,
    supported: set[str],
) -> str:
    normalized = set(labels)
    active = [value for value in ("method", "dataset", "year", "negation") if value in normalized]
    if len(active) >= 2 and "multi_constraint" in supported:
        return "multi_constraint"
    for value in ("negation", "method", "dataset", "year"):
        if value in normalized and value in supported:
            return value
    if "entity" in supported and _ENTITY.search(query):
        return "entity"
    return "unconstrained"


def select_disjoint_cases(
    partition_rows: Sequence[Mapping[str, object]],
    *,
    context_by_id: Mapping[str, Mapping[str, object]],
    priority_by_id: Mapping[str, Mapping[str, object]],
    excluded_query_ids: set[str],
    quotas: Mapping[str, int],
    seed: str,
) -> list[dict[str, object]]:
    """Select deterministic recall-gap cases without retaining any Gold field."""

    if not quotas or any(type(value) is not int or value < 0 for value in quotas.values()):
        raise ValueError("stratum quotas are invalid")
    pools: dict[str, list[dict[str, object]]] = {name: [] for name in quotas}
    for raw in partition_rows:
        query_id = str(raw.get("query_id", ""))
        query = " ".join(str(raw.get("query", "")).split())
        if (
            not query_id
            or not query
            or query_id in excluded_query_ids
            or raw.get("role") != "training"
            or raw.get("split") != "auto_train"
        ):
            continue
        context = context_by_id.get(query_id)
        priority = priority_by_id.get(query_id)
        if (
            context is None
            or priority is None
            or context.get("role") != "training"
            or context.get("split") != "auto_train"
            or priority.get("base_gold_hit_count") != 0
        ):
            continue
        stratum = _stratum(query, _labels(context), supported=set(quotas))
        if stratum not in pools:
            continue
        payload = {"query": query}
        item: dict[str, object] = {
            "case_id": query_id,
            "query": query,
            "query_sha256": _sha256(query.encode("utf-8")),
            "stratum": stratum,
            "context_signals": _labels(context),
            "difficulty_evidence": "known-recall-gap",
            "network_payload": payload,
            "network_payload_sha256": _sha256(_canonical_bytes(payload)),
            "prior_validation_overlap": False,
        }
        pools[stratum].append(item)
    shortages = {
        name: {"required": target, "available": len(pools[name])}
        for name, target in quotas.items()
        if len(pools[name]) < target
    }
    if shortages:
        availability = {name: len(values) for name, values in pools.items()}
        raise ValueError(
            f"insufficient disjoint strata: {shortages}; availability={availability}"
        )
    selected: list[dict[str, object]] = []
    for name, target in quotas.items():
        pool = sorted(
            pools[name],
            key=lambda item: hashlib.sha256(
                f"{seed}|{item['case_id']}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(pool[:target])
    ids = [str(item["case_id"]) for item in selected]
    if len(ids) != len(set(ids)) or set(ids).intersection(excluded_query_ids):
        raise ValueError("selected cases are not disjoint")
    return selected


def candidate_action_budget_report(
    *,
    query: str,
    subqueries: Sequence[Mapping[str, object]],
    trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Audit that LLM novelty occupies planner slots rather than a seventh slot."""

    original = _normalized(query)
    bridge = [
        item
        for item in subqueries
        if item.get("query_id") == "sq-supervised-lexical-bridge"
    ]
    novel = [
        item
        for item in subqueries
        if _normalized(item.get("text")) != original
        and item.get("query_id") != "sq-supervised-lexical-bridge"
    ]
    receipt = next(
        (
            dict(item)
            for item in trace
            if item.get("step") == "supervised_query_expansion"
        ),
        {},
    )
    configured = receipt.get("configured_action_budget")
    original_title_present = any(
        item.get("action_type") == "title_search"
        and item.get("search_mode", "lexical") == "lexical"
        and _normalized(item.get("text")) == original
        for item in subqueries
    )
    return {
        "configured_action_budget": configured,
        "final_action_count": len(subqueries),
        "within_six_action_budget": (
            configured == 6
            and len(subqueries) <= 6
            and receipt.get("budget_policy")
            == "llm-replaces-rule-fallback-before-local-bridge"
        ),
        "novel_llm_action_count": len(novel),
        "bridge_inside_budget": len(bridge) <= 1 and len(subqueries) <= 6,
        "lowest_value_title_fallback_present": original_title_present,
        "replacement_observed": (
            bool(novel)
            and receipt.get("action_count_before") == 5
            and not original_title_present
        ),
        "action_count_before_bridge": receipt.get("action_count_before"),
        "action_count_after_bridge": receipt.get("action_count_after"),
    }


def promotion_decision(metrics: Mapping[str, object]) -> dict[str, object]:
    """Apply the strict canary gate without treating the sample as an official score."""

    values = {key: int(value) for key, value in metrics.items()}
    count = values.get("query_count", 0)
    failed: list[str] = []
    for key, label in (
        ("live_replay_exact_query_count", "live_replay_incomplete"),
        ("candidate_action_budget_pass_query_count", "action_budget_failure"),
        ("candidate_f5_query_count", "f5_not_active"),
    ):
        if values.get(key, -1) != count:
            failed.append(label)
    if values.get("candidate_gold_pool_regressed_query_count", 0) != 0:
        failed.append("gold_pool_query_regression")
    pairs = (
        ("gold_pool", "baseline_gold_pool_hit_query_count", "candidate_gold_pool_hit_query_count"),
        ("top5", "baseline_top5_hit_query_count", "candidate_top5_hit_query_count"),
        ("top10", "baseline_top10_hit_query_count", "candidate_top10_hit_query_count"),
        ("top20", "baseline_top20_hit_query_count", "candidate_top20_hit_query_count"),
    )
    strict_gain = False
    for label, baseline, candidate in pairs:
        before = values.get(baseline, 0)
        after = values.get(candidate, 0)
        if after < before:
            failed.append(f"{label}_regression")
        strict_gain = strict_gain or after > before
    if values.get("stratum_regression_count", 0) != 0:
        failed.append("stratum_regression")
    if values.get("candidate_llm_novel_query_count", 0) == 0:
        failed.append("no_llm_actions_admitted")
    if values.get("candidate_replacement_observed_query_count", 0) == 0:
        failed.append("replacement_not_observed")
    if not strict_gain:
        failed.append("no_strict_gain")
    return {
        "passed": not failed,
        "decision": "eligible_for_promotion_review" if not failed else "keep_current_production",
        "failed_gates": failed,
        "metrics": values,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {path}")
        rows.append(value)
    return rows


def _ids_from_json(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {"query_id", "case_id"} and isinstance(nested, str):
                if _QUERY_ID.fullmatch(nested):
                    yield nested
            elif key == "query_ids" and isinstance(nested, list):
                for item in nested:
                    if isinstance(item, str) and _QUERY_ID.fullmatch(item):
                        yield item
            yield from _ids_from_json(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _ids_from_json(nested)


def collect_excluded_query_ids(
    roots: Sequence[Path],
    *,
    ignored_roots: Sequence[Path] = (),
) -> set[str]:
    """Collect prior validation identities without traversing training partitions."""

    selected_names = {
        "sample-manifest.json",
        "query-selection.json",
        "openalex-selection.json",
        "selection.json",
    }
    ignored = tuple(path.resolve() for path in ignored_roots)
    ids: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            resolved = path.resolve()
            if any(resolved == item or item in resolved.parents for item in ignored):
                continue
            if not path.is_file():
                continue
            match = _QUERY_ID.fullmatch(path.stem)
            if match is not None:
                ids.add(match.group(0))
            if path.suffix == ".jsonl" and "partition" in path.name:
                try:
                    ids.update(
                        str(row["query_id"])
                        for row in _load_jsonl(path)
                        if isinstance(row.get("query_id"), str)
                        and _QUERY_ID.fullmatch(str(row["query_id"]))
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    continue
            elif path.name in selected_names:
                try:
                    ids.update(_ids_from_json(json.loads(path.read_bytes())))
                except (OSError, json.JSONDecodeError):
                    continue
    return ids


def _source_binding(path: Path, workspace: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(workspace.resolve())),
        "sha256": _sha256(path.read_bytes()),
    }


def prepare_sample(
    *,
    workspace: Path,
    partition_path: Path,
    context_path: Path,
    priority_path: Path,
    output_path: Path,
    excluded_roots: Sequence[Path],
    quotas: Mapping[str, int],
    seed: str,
) -> dict[str, object]:
    partition_rows = _load_jsonl(partition_path)
    context_rows = _load_jsonl(context_path)
    priority_rows = _load_jsonl(priority_path)
    context = {str(row["query_id"]): row for row in context_rows}
    priority = {str(row["query_id"]): row for row in priority_rows}
    if len(context) != len(context_rows) or len(priority) != len(priority_rows):
        raise ValueError("sample source query identities are not unique")
    excluded = collect_excluded_query_ids(
        excluded_roots,
        ignored_roots=(output_path.parent,),
    )
    cases = select_disjoint_cases(
        partition_rows,
        context_by_id=context,
        priority_by_id=priority,
        excluded_query_ids=excluded,
        quotas=quotas,
        seed=seed,
    )
    manifest: dict[str, object] = {
        "schema_version": "semantic-action-replacement-full-live-sample-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "selection_seed": seed,
        "query_count": len(cases),
        "query_source": "strict-ready-training-auto-train-recall-gaps",
        "prior_validation_query_id_count": len(excluded),
        "prior_validation_inventory_sha256": _sha256(
            _canonical_bytes(sorted(excluded))
        ),
        "stratum_quotas": dict(quotas),
        "stratum_counts": dict(Counter(str(item["stratum"]) for item in cases)),
        "partition_checks": {
            "all_training_auto_train": True,
            "unique_query_ids": len(cases)
            == len({str(item["case_id"]) for item in cases}),
            "prior_overlap_count": sum(
                int(str(item["case_id"]) in excluded) for item in cases
            ),
            "final_test_touched": False,
        },
        "network_payload_fields": ["query"],
        "gold_identifiers_in_sample_manifest": False,
        "sources": {
            "partition": _source_binding(partition_path, workspace),
            "constraint_context": _source_binding(context_path, workspace),
            "recall_priority_queue": _source_binding(priority_path, workspace),
        },
        "cases": cases,
    }
    content = _canonical_bytes(manifest)
    if output_path.exists():
        existing = json.loads(output_path.read_bytes())
        stable = copy.deepcopy(manifest)
        if isinstance(existing, dict) and isinstance(existing.get("created_at"), str):
            stable["created_at"] = existing["created_at"]
        content = _canonical_bytes(stable)
    _write_immutable(output_path, content)
    return json.loads(content)


def _safe_environment(env_file: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    values = dotenv_values(env_file)
    environment = {
        str(key): value
        for key, value in values.items()
        if isinstance(key, str) and isinstance(value, str) and value
    }
    if not environment.get("LLM_API_KEY"):
        raise ValueError("LLM API credential is unavailable")
    if not any(key == "OPENALEX_API_KEY" or key.startswith("OPENALEX_API_KEY_") for key in environment):
        raise ValueError("OpenAlex API credential is unavailable")
    return environment


def _execution_payload(execution: object) -> dict[str, object]:
    dumped = getattr(execution, "model_dump")(mode="json")
    if not isinstance(dumped, dict):
        raise ValueError("execution serialization failed")
    return dumped


def _search_semantics(execution: object) -> list[tuple[object, ...]]:
    outcome = execution.outcome
    if outcome.kind != "success":
        return []
    return [
        (
            item.get("step"),
            item.get("provider"),
            item.get("query_text"),
            item.get("action_type"),
            item.get("search_mode"),
            item.get("subquery_id"),
        )
        for item in outcome.response.search_trace
        if item.get("step") in {"retrieve", "retrieve_supplement", "skip_optional_provider"}
    ]


def _paper_ids(papers: Sequence[object]) -> list[str]:
    return [str(getattr(item, "canonical_id")) for item in papers]


def _replay_exact(live: object, replay: object) -> tuple[bool, dict[str, bool]]:
    if live.outcome.kind != "success" or replay.outcome.kind != "success":
        return False, {"both_success": False}
    left = live.outcome.response
    right = replay.outcome.response
    checks = {
        "both_success": True,
        "query_analysis": left.query_analysis == right.query_analysis,
        "selected_ids_and_order": left.selected_paper_ids == right.selected_paper_ids,
        "pre_truncation_ids_and_order": _paper_ids(live.pre_truncation_candidates)
        == _paper_ids(replay.pre_truncation_candidates),
        "post_filter_ids_and_order": live.post_filter_paper_ids
        == replay.post_filter_paper_ids,
        "retrieved_ids_and_order": live.retrieved_paper_ids
        == replay.retrieved_paper_ids,
        "search_action_semantics": _search_semantics(live) == _search_semantics(replay),
        "business_result": live.business_result_sha256 == replay.business_result_sha256,
    }
    return all(checks.values()), checks


def _next_run_id(root: Path, base: str) -> str:
    for attempt in range(1, 100):
        candidate = base if attempt == 1 else f"{base}-r{attempt}"
        if not (root / candidate).exists() and not (root / f"{candidate}.failed").exists():
            return candidate
    raise ValueError("no resumable run id is available")


async def _run_condition(
    *,
    name: str,
    lock_path: Path,
    cases: Sequence[Mapping[str, object]],
    workspace: Path,
    output_root: Path,
    environment: Mapping[str, str],
) -> list[dict[str, object]]:
    from paper_search.application.composition import CompositionRoot
    from paper_search.application.contracts import SearchRequest

    condition_root = output_root / name
    captures_root = condition_root / "captures"
    results_root = condition_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    input_lock_bytes = lock_path.read_bytes()
    bundle = CompositionRoot.compose(
        lock_path=lock_path,
        mode="live",
        artifact_root=workspace,
        output_root=captures_root,
        network_authorized=True,
        environ=environment,
    )
    records: list[dict[str, object]] = []
    try:
        for index, case in enumerate(cases):
            case_id = str(case["case_id"])
            record_path = results_root / f"{index:02d}.json"
            if record_path.exists():
                record = json.loads(record_path.read_bytes())
                if record.get("case_id") != case_id or record.get("condition") != name:
                    raise ValueError("resumable live record identity changed")
                records.append(record)
                print(f"{name} {index + 1}/{len(cases)} resumed", flush=True)
                continue
            run_base = f"{name[0]}-{index:02d}-{hashlib.sha256(case_id.encode()).hexdigest()[:10]}"
            run_id = _next_run_id(captures_root, run_base)
            session = bundle.artifact_factory.start_capture(
                run_id=run_id,
                input_lock_bytes=input_lock_bytes,
                expected_config_hash=bundle.config_hash,
            )
            try:
                execution = await bundle.service.execute(
                    SearchRequest(
                        query_id=case_id,
                        query=str(case["query"]),
                        budget_profile="balanced",
                        include_trace=True,
                        mode="live",
                    ),
                    run_id=run_id,
                )
                session.record_execution(execution)
                if execution.outcome.kind != "success":
                    session.fail(execution.outcome.error.code)
                    raise RuntimeError(
                        f"{name} live failed for {case_id}: {execution.outcome.error.code}"
                    )
                _manifest, _replay_lock = session.seal()
                published = session.publish()
            except BaseException:
                if session.work_dir.exists():
                    try:
                        session.fail("internal_error")
                    except (OSError, RuntimeError, ValueError):
                        pass
                raise
            replay_bundle = CompositionRoot.compose(
                lock_path=published / "replay.lock.yaml",
                mode="replay",
                artifact_root=workspace,
                output_root=condition_root / "replay-checks",
                snapshot_manifest_path=published / "snapshot-manifest.json",
                network_authorized=False,
                environ={},
            )
            try:
                replay = await replay_bundle.service.execute(
                    SearchRequest(
                        query_id=case_id,
                        query=str(case["query"]),
                        budget_profile="balanced",
                        include_trace=True,
                        mode="replay",
                    ),
                    run_id=f"rp-{name[0]}-{index:02d}",
                )
            finally:
                await replay_bundle.aclose()
            exact, checks = _replay_exact(execution, replay)
            response = execution.outcome.response
            budget_report = (
                candidate_action_budget_report(
                    query=str(case["query"]),
                    subqueries=[
                        item.model_dump(mode="json")
                        for item in response.query_analysis.search_plan.subqueries
                    ],
                    trace=response.search_trace,
                )
                if name == "candidate"
                else None
            )
            record = {
                "schema_version": "semantic-action-replacement-full-live-case-v1",
                "condition": name,
                "case_id": case_id,
                "stratum": str(case["stratum"]),
                "capture_path": str(published.resolve()),
                "live_replay_exact": exact,
                "live_replay_checks": checks,
                "candidate_action_budget": budget_report,
                "live_execution": _execution_payload(execution),
                "gold_loaded_before_or_during_network": False,
                "final_test_touched": False,
            }
            _write_immutable(record_path, _canonical_bytes(record))
            records.append(record)
            trace = response.search_trace
            openalex = sum(
                int(item.get("provider") == "openalex")
                for item in trace
                if item.get("step") in {"retrieve", "retrieve_supplement"}
            )
            semantic = sum(
                int(item.get("provider") == "semantic_scholar")
                for item in trace
                if item.get("step") == "retrieve"
            )
            print(
                f"{name} {index + 1}/{len(cases)} live+replay exact={exact} "
                f"openalex={openalex} s2={semantic}",
                flush=True,
            )
    finally:
        await bundle.aclose()
    return records


def _load_execution(record: Mapping[str, object]) -> object:
    from paper_search.application.contracts import SearchExecutionResult

    return SearchExecutionResult.model_validate(record["live_execution"])


def _ranked_papers(execution: object) -> list[object]:
    if execution.outcome.kind != "success":
        return []
    return [item.paper for item in execution.outcome.response.fused_papers]


def _first_gold_rank(
    papers: Sequence[object],
    gold_ids: Sequence[str],
    *,
    identifier_map: object,
) -> int | None:
    from paper_search.evaluation.predictions import paper_matches_evaluation_ids

    return next(
        (
            index
            for index, paper in enumerate(papers, start=1)
            if paper_matches_evaluation_ids(
                paper,
                gold_ids,
                identifier_map=identifier_map,
            )
        ),
        None,
    )


def _document_rank_role(execution: object) -> str | None:
    if execution.outcome.kind != "success":
        return None
    trace = execution.outcome.response.search_trace
    return next(
        (
            str(item.get("deployment_role"))
            for item in trace
            if item.get("step") == "document_rank"
            and isinstance(item.get("deployment_role"), str)
        ),
        None,
    )


def _usage_totals(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    from decimal import Decimal

    search_calls = 0
    llm_calls = 0
    input_tokens = 0
    cached_input_tokens = 0
    uncached_input_tokens = 0
    output_tokens = 0
    cost = Decimal("0")
    for record in records:
        execution = _load_execution(record)
        usage = (
            execution.outcome.response.usage
            if execution.outcome.kind == "success"
            else execution.outcome.usage
        )
        search_calls += usage.search_api_calls
        llm_calls += usage.llm_calls
        input_tokens += usage.input_tokens
        cached_input_tokens += usage.cached_input_tokens
        uncached_input_tokens += usage.uncached_input_tokens
        output_tokens += usage.output_tokens
        if usage.cost_cny is not None:
            cost += usage.cost_cny
    return {
        "search_api_calls": search_calls,
        "llm_calls": llm_calls,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "cost_cny": str(cost),
    }


def condition_comparison_label(
    baseline_records: Sequence[Mapping[str, object]],
    candidate_records: Sequence[Mapping[str, object]],
) -> str:
    """Derive the comparison identity from sealed executions, never a legacy label."""

    def only_version(records: Sequence[Mapping[str, object]]) -> str:
        versions: set[str] = set()
        for record in records:
            execution = record.get("live_execution")
            if not isinstance(execution, Mapping):
                raise ValueError("live execution is unavailable")
            outcome = execution.get("outcome")
            if not isinstance(outcome, Mapping):
                raise ValueError("live outcome is unavailable")
            response = outcome.get("response")
            if not isinstance(response, Mapping):
                raise ValueError("successful live response is unavailable")
            version = response.get("prompt_version")
            if not isinstance(version, str) or not version:
                raise ValueError("live prompt version is unavailable")
            versions.add(version)
        if len(versions) != 1:
            raise ValueError("condition mixes prompt versions")
        return next(iter(versions))

    return f"{only_version(baseline_records)}-versus-{only_version(candidate_records)}"


def evaluate_conditions(
    *,
    workspace: Path,
    production_lock_path: Path,
    partition_path: Path,
    cases: Sequence[Mapping[str, object]],
    baseline_records: Sequence[Mapping[str, object]],
    candidate_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Load Gold only after network completion and calculate paired recall/rank gates."""

    from paper_search.application.composition import load_locked_identifier_map
    from paper_search.application.locks import load_verified_input_lock

    if len(cases) != len(baseline_records) or len(cases) != len(candidate_records):
        raise ValueError("paired condition records are incomplete")
    partition = {str(row["query_id"]): row for row in _load_jsonl(partition_path)}
    verified = load_verified_input_lock(
        production_lock_path,
        artifact_root=workspace,
    )
    identifier_map, alias_count = load_locked_identifier_map(
        verified.lock,
        verified.artifact_bytes,
    )
    if identifier_map is None:
        raise ValueError("production identifier aliases are unavailable")
    case_reports: list[dict[str, object]] = []
    for case, baseline_record, candidate_record in zip(
        cases,
        baseline_records,
        candidate_records,
        strict=True,
    ):
        case_id = str(case["case_id"])
        if (
            baseline_record.get("case_id") != case_id
            or candidate_record.get("case_id") != case_id
        ):
            raise ValueError("paired condition order changed")
        source = partition.get(case_id)
        if source is None:
            raise ValueError(f"selected query is absent from frozen partition: {case_id}")
        gold = source.get("gold_paper_ids")
        if not isinstance(gold, list) or not all(isinstance(item, str) for item in gold):
            raise ValueError(f"selected query lacks frozen Gold identifiers: {case_id}")
        baseline = _load_execution(baseline_record)
        candidate = _load_execution(candidate_record)
        baseline_pool = list(baseline.pre_truncation_candidates)
        candidate_pool = list(candidate.pre_truncation_candidates)
        baseline_ranked = _ranked_papers(baseline)
        candidate_ranked = _ranked_papers(candidate)
        baseline_pool_rank = _first_gold_rank(
            baseline_pool,
            gold,
            identifier_map=identifier_map,
        )
        candidate_pool_rank = _first_gold_rank(
            candidate_pool,
            gold,
            identifier_map=identifier_map,
        )
        baseline_rank = _first_gold_rank(
            baseline_ranked,
            gold,
            identifier_map=identifier_map,
        )
        candidate_rank = _first_gold_rank(
            candidate_ranked,
            gold,
            identifier_map=identifier_map,
        )
        budget = candidate_record.get("candidate_action_budget")
        if not isinstance(budget, Mapping):
            raise ValueError("candidate action-budget receipt is unavailable")
        case_reports.append(
            {
                "case_id": case_id,
                "stratum": str(case["stratum"]),
                "baseline_gold_in_pool": baseline_pool_rank is not None,
                "candidate_gold_in_pool": candidate_pool_rank is not None,
                "gold_pool_improved": baseline_pool_rank is None
                and candidate_pool_rank is not None,
                "gold_pool_regressed": baseline_pool_rank is not None
                and candidate_pool_rank is None,
                "baseline_first_gold_rank": baseline_rank,
                "candidate_first_gold_rank": candidate_rank,
                "top5": {
                    "baseline": baseline_rank is not None and baseline_rank <= 5,
                    "candidate": candidate_rank is not None and candidate_rank <= 5,
                },
                "top10": {
                    "baseline": baseline_rank is not None and baseline_rank <= 10,
                    "candidate": candidate_rank is not None and candidate_rank <= 10,
                },
                "top20": {
                    "baseline": baseline_rank is not None and baseline_rank <= 20,
                    "candidate": candidate_rank is not None and candidate_rank <= 20,
                },
                "baseline_pool_count": len(baseline_pool),
                "candidate_pool_count": len(candidate_pool),
                "baseline_f5_role": _document_rank_role(baseline),
                "candidate_f5_role": _document_rank_role(candidate),
                "live_replay_exact": baseline_record.get("live_replay_exact") is True
                and candidate_record.get("live_replay_exact") is True,
                "candidate_action_budget": dict(budget),
            }
        )
    metrics: dict[str, object] = {
        "query_count": len(case_reports),
        "live_replay_exact_query_count": sum(
            int(item["live_replay_exact"] is True) for item in case_reports
        ),
        "candidate_action_budget_pass_query_count": sum(
            int(
                isinstance(item["candidate_action_budget"], Mapping)
                and item["candidate_action_budget"].get("within_six_action_budget") is True
                and item["candidate_action_budget"].get("bridge_inside_budget") is True
            )
            for item in case_reports
        ),
        "candidate_f5_query_count": sum(
            int(item["candidate_f5_role"] == "F5-gated-fusion")
            for item in case_reports
        ),
        "candidate_llm_novel_query_count": sum(
            int(
                int(item["candidate_action_budget"].get("novel_llm_action_count", 0))
                > 0
            )
            for item in case_reports
        ),
        "candidate_replacement_observed_query_count": sum(
            int(item["candidate_action_budget"].get("replacement_observed") is True)
            for item in case_reports
        ),
        "baseline_gold_pool_hit_query_count": sum(
            int(item["baseline_gold_in_pool"] is True) for item in case_reports
        ),
        "candidate_gold_pool_hit_query_count": sum(
            int(item["candidate_gold_in_pool"] is True) for item in case_reports
        ),
        "candidate_gold_pool_improved_query_count": sum(
            int(item["gold_pool_improved"] is True) for item in case_reports
        ),
        "candidate_gold_pool_regressed_query_count": sum(
            int(item["gold_pool_regressed"] is True) for item in case_reports
        ),
    }
    for cutoff in (5, 10, 20):
        key = f"top{cutoff}"
        metrics[f"baseline_{key}_hit_query_count"] = sum(
            int(item[key]["baseline"] is True) for item in case_reports
        )
        metrics[f"candidate_{key}_hit_query_count"] = sum(
            int(item[key]["candidate"] is True) for item in case_reports
        )
    strata: dict[str, dict[str, int]] = {}
    for name in sorted({str(item["stratum"]) for item in case_reports}):
        group = [item for item in case_reports if item["stratum"] == name]
        values = {
            "query_count": len(group),
            "baseline_gold_pool": sum(int(item["baseline_gold_in_pool"]) for item in group),
            "candidate_gold_pool": sum(int(item["candidate_gold_in_pool"]) for item in group),
        }
        for cutoff in (5, 10, 20):
            key = f"top{cutoff}"
            values[f"baseline_{key}"] = sum(
                int(item[key]["baseline"]) for item in group
            )
            values[f"candidate_{key}"] = sum(
                int(item[key]["candidate"]) for item in group
            )
        strata[name] = values
    stratum_regressions = []
    for name, values in strata.items():
        for metric in ("gold_pool", "top5", "top10", "top20"):
            if values[f"candidate_{metric}"] < values[f"baseline_{metric}"]:
                stratum_regressions.append(f"{name}:{metric}")
    metrics["stratum_regression_count"] = len(stratum_regressions)
    decision = promotion_decision(metrics)
    return {
        "schema_version": "semantic-action-replacement-full-live-gate-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison": condition_comparison_label(
            baseline_records,
            candidate_records,
        ),
        "sample_role": "disjoint-training-auto-train-canary-not-official-test",
        "gold_loaded_after_all_live_conditions": True,
        "gold_or_final_test_sent": False,
        "final_test_touched": False,
        "production_lock_modified": False,
        "identifier_alias_count": alias_count,
        "metrics": metrics,
        "strata": strata,
        "stratum_regressions": stratum_regressions,
        "decision": decision,
        "network_usage": {
            "baseline": _usage_totals(baseline_records),
            "candidate": _usage_totals(candidate_records),
        },
        "cases": case_reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the six-slot semantic-action full-live paired gate."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--production-lock",
        type=Path,
        default=Path("deliverables/evaluator/live-evaluator.lock.yaml"),
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path(SEMANTIC_PROMPT_PATH),
    )
    parser.add_argument(
        "--prompt-version",
        default=SEMANTIC_PROMPT_VERSION,
    )
    parser.add_argument(
        "--approval-ref",
        default=_APPROVAL_REF,
    )
    parser.add_argument(
        "--selection-seed",
        default="semantic-action-six-slot-full-live-24-v1",
    )
    parser.add_argument(
        "--stratum-quotas-json",
        default=None,
        help="Optional JSON object totaling 24; zero preserves exhausted strata as absent.",
    )
    parser.add_argument(
        "--partition",
        type=Path,
        default=Path(
            "data/training_private/partitions/"
            "pasa_auto_train_strict_ready_21429_20260828_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=Path(
            "data/training_private/training_runs/"
            "openalex-pasa-high-recall-v3-no-leakage-context-21429-v2/"
            "constraint-labels.merged.jsonl"
        ),
    )
    parser.add_argument(
        "--priority",
        type=Path,
        default=Path(
            "data/training_private/evaluations/"
            "openalex-pasa-dataset-task-priority-queue-6242-v1.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/llm-openalex-validations/"
            "semantic-action-six-slot-full-live-24-20260828"
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(r"D:\AI Projects\Projects\.env"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> dict[str, object]:
    workspace = args.workspace.resolve()
    production_lock = (workspace / args.production_lock).resolve()
    prompt = (workspace / args.prompt).resolve()
    partition = (workspace / args.partition).resolve()
    context = (workspace / args.context).resolve()
    priority = (workspace / args.priority).resolve()
    output_root = (workspace / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_before = _sha256(production_lock.read_bytes())
    candidate_lock = output_root / "candidate-prompt.lock.yaml"
    _write_immutable(
        candidate_lock,
        build_candidate_lock_bytes(
            production_lock.read_bytes(),
            prompt_bytes=prompt.read_bytes(),
            prompt_path=str(prompt.relative_to(workspace)).replace("\\", "/"),
            prompt_version=args.prompt_version,
            approval_ref=args.approval_ref,
        ),
    )
    excluded_roots = (
        workspace / "data/training_private/recall_policy",
        workspace / "data/training_private/online_recall",
        workspace / "artifacts/llm-openalex-validations",
        workspace / "artifacts/oa-v2-20260828",
    )
    sample = prepare_sample(
        workspace=workspace,
        partition_path=partition,
        context_path=context,
        priority_path=priority,
        output_path=output_root / "sample-manifest.json",
        excluded_roots=excluded_roots,
        quotas=parse_stratum_quotas(args.stratum_quotas_json),
        seed=args.selection_seed,
    )
    cases = sample.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError("frozen full-live sample is invalid")
    preflight = {
        "schema_version": "semantic-action-replacement-full-live-preflight-v1",
        "ready": True,
        "created_at": datetime.now(UTC).isoformat(),
        "query_count": len(cases),
        "conditions": ["baseline", "candidate"],
        "maximum_planner_actions_per_condition": 6,
        "existing_bounded_cross_vocabulary_supplement_max": 1,
        "maximum_openalex_logical_calls": len(cases) * 2 * 7,
        "maximum_semantic_scholar_logical_calls": len(cases) * 2 * 2,
        "maximum_llm_logical_calls": len(cases) * 2,
        "network_payload_fields": ["query"],
        "gold_loaded": False,
        "gold_or_final_test_sent": False,
        "final_test_touched": False,
        "production_lock_sha256": lock_before,
        "candidate_lock_sha256": _sha256(candidate_lock.read_bytes()),
        "production_lock_modified": _sha256(production_lock.read_bytes()) != lock_before,
    }
    preflight_path = output_root / "preflight.json"
    if preflight_path.exists():
        existing = json.loads(preflight_path.read_bytes())
        if isinstance(existing, dict) and isinstance(existing.get("created_at"), str):
            preflight["created_at"] = existing["created_at"]
    _write_immutable(preflight_path, _canonical_bytes(preflight))
    if args.prepare_only:
        return preflight
    environment = _safe_environment(args.env_file.resolve())
    baseline_records = await _run_condition(
        name="baseline",
        lock_path=production_lock,
        cases=cases,
        workspace=workspace,
        output_root=output_root,
        environment=environment,
    )
    candidate_records = await _run_condition(
        name="candidate",
        lock_path=candidate_lock,
        cases=cases,
        workspace=workspace,
        output_root=output_root,
        environment=environment,
    )
    report = evaluate_conditions(
        workspace=workspace,
        production_lock_path=production_lock,
        partition_path=partition,
        cases=cases,
        baseline_records=baseline_records,
        candidate_records=candidate_records,
    )
    if _sha256(production_lock.read_bytes()) != lock_before:
        raise ValueError("production lock changed during validation")
    report["production_lock_file_sha256"] = lock_before
    report_path = output_root / "promotion-gate.json"
    if report_path.exists():
        existing = json.loads(report_path.read_bytes())
        if isinstance(existing, dict) and isinstance(existing.get("created_at"), str):
            report["created_at"] = existing["created_at"]
    _write_immutable(report_path, _canonical_bytes(report))
    return report


def main() -> None:
    args = _parser().parse_args()
    report = asyncio.run(_run(args))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
