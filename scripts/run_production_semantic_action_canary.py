"""Gold-blind helpers for the current production-module semantic action canary.

The live validation itself is intentionally composed from the production parser,
planner, provider snapshot adapters, orchestrator, identifier aliases, and F5
ranker.  These helpers keep the paired comparison payload small and make it
impossible to carry evaluator-only Gold material into a provider request.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from paper_search.recall_experiments.contracts import (
    assert_no_forbidden_identifier_keys_or_patterns,
)


def _normalized_text(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def normalized_action_identity(
    case_id: object,
    action: Mapping[str, object],
) -> tuple[str, str, str]:
    """Return the immutable query/action identity used for no-repeat checks."""

    return (
        str(case_id),
        str(action.get("search_mode", "lexical")),
        _normalized_text(action.get("text")),
    )


def build_openalex_request_spec(
    case: Mapping[str, object],
    *,
    limit: int = 50,
) -> dict[str, object]:
    """Return the only Gold-blind payload allowed for the online canary."""

    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("OpenAlex canary limit must be between one and fifty")
    action = case.get("selected_action")
    if not isinstance(action, Mapping):
        actions = case.get("accepted_novel_model_actions")
        action = (
            actions[0]
            if isinstance(actions, Sequence)
            and actions
            and isinstance(actions[0], Mapping)
            else None
        )
    if not isinstance(action, Mapping):
        raise ValueError("selected OpenAlex action is unavailable")
    if action.get("action_type", "text_search") != "text_search":
        raise ValueError("OpenAlex canary only accepts text search actions")
    query_text = " ".join(str(action.get("text", "")).split())
    if not query_text:
        raise ValueError("selected OpenAlex action text is unavailable")
    search_mode = str(action.get("search_mode", "lexical"))
    if search_mode not in {"lexical", "semantic"}:
        raise ValueError("OpenAlex search mode must be lexical or semantic")
    filters: dict[str, object] = {}
    if search_mode == "semantic":
        filters["_search_mode"] = "semantic"
    return {
        "query_text": query_text,
        "search_mode": search_mode,
        "filters": filters,
        "limit": limit,
    }


def online_source_ranks(candidate: Mapping[str, object]) -> dict[str, int]:
    """Keep frozen online ranks while excluding every PASA-local source."""

    raw = candidate.get("source_ranks")
    if not isinstance(raw, Mapping):
        raise ValueError("candidate source ranks are unavailable")
    ranks: dict[str, int] = {}
    for source, rank in raw.items():
        if not isinstance(source, str) or not source:
            raise ValueError("candidate source id is invalid")
        if "pasa" in source.casefold():
            continue
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("candidate source rank is invalid")
        ranks[source] = rank
    return ranks


def _exact_action(case: Mapping[str, object]) -> dict[str, object]:
    query = str(case.get("query", ""))
    model_output = case.get("model_output")
    if not isinstance(model_output, Mapping):
        raise ValueError("case model_output is unavailable")
    search_plan = model_output.get("search_plan")
    if not isinstance(search_plan, Mapping):
        raise ValueError("case search plan is unavailable")
    subqueries = search_plan.get("subqueries")
    if not isinstance(subqueries, Sequence):
        raise ValueError("case subqueries are unavailable")
    query_identity = _normalized_text(query)
    for item in subqueries:
        if (
            isinstance(item, Mapping)
            and _normalized_text(item.get("text")) == query_identity
            and item.get("action_type", "text_search") == "text_search"
        ):
            dumped = dict(item)
            dumped.setdefault("action_type", "text_search")
            dumped.setdefault("search_mode", "lexical")
            return dumped
    template = next(
        (dict(item) for item in subqueries if isinstance(item, Mapping)),
        {},
    )
    return {
        **template,
        "query_id": "canary-anchor",
        "text": query,
        "query_type": "exact",
        "action_type": "text_search",
        "target_constraints": list(template.get("target_constraints", [])),
        "priority": 1,
        "provider_hint": "either",
        "search_mode": "lexical",
    }


def build_paired_model_outputs(
    case: Mapping[str, object],
    selected_action: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build common-baseline and one-action-augmented LLM outputs."""

    model_output = case.get("model_output")
    if not isinstance(model_output, Mapping):
        raise ValueError("case model_output is unavailable")
    exact = _exact_action(case)
    selected = dict(selected_action)
    selected.setdefault("action_type", "text_search")
    selected.setdefault("search_mode", "lexical")
    if selected["action_type"] != "text_search":
        raise ValueError("paired OpenAlex action must be a text search")
    if _normalized_text(selected.get("text")) == _normalized_text(case.get("query")):
        raise ValueError("paired OpenAlex action must be novel")

    baseline = copy.deepcopy(dict(model_output))
    augmented = copy.deepcopy(dict(model_output))
    baseline_plan = baseline.get("search_plan")
    augmented_plan = augmented.get("search_plan")
    if not isinstance(baseline_plan, dict) or not isinstance(augmented_plan, dict):
        raise ValueError("case search plan is unavailable")
    baseline_plan["subqueries"] = [exact]
    augmented_plan["subqueries"] = [exact, selected]
    return baseline, augmented


def _eligible_action(
    case: Mapping[str, object],
    action: object,
    completed: set[tuple[str, str, str]],
) -> dict[str, object] | None:
    if not isinstance(action, Mapping):
        return None
    provider_hint = action.get("provider_hint", "either")
    if provider_hint not in {"either", "openalex"}:
        return None
    if action.get("action_type", "text_search") != "text_search":
        return None
    if _normalized_text(action.get("text")) == _normalized_text(case.get("query")):
        return None
    dumped = dict(action)
    dumped.setdefault("action_type", "text_search")
    dumped.setdefault("search_mode", "lexical")
    if normalized_action_identity(case.get("case_id"), dumped) in completed:
        return None
    return dumped


def select_openalex_cases(
    cases: Iterable[Mapping[str, object]],
    completed_action_identities: set[tuple[str, str, str]],
    *,
    limit: int,
) -> list[dict[str, object]]:
    """Choose at most one new, OpenAlex-compatible action per eligible query."""

    if type(limit) is not int or limit < 1:
        raise ValueError("OpenAlex case limit must be positive")
    selected: list[dict[str, object]] = []
    for case in cases:
        if case.get("openalex_pair_eligible") is not True:
            continue
        raw_actions = case.get("accepted_novel_model_actions", [])
        if not isinstance(raw_actions, Sequence):
            continue
        action = next(
            (
                eligible
                for item in raw_actions
                if (
                    eligible := _eligible_action(
                        case,
                        item,
                        completed_action_identities,
                    )
                )
                is not None
            ),
            None,
        )
        if action is None:
            continue
        item = {
            "case_id": str(case.get("case_id", "")),
            "query": str(case.get("query", "")),
            "stratum": str(case.get("stratum", "unclassified")),
            "selected_action": action,
            "model_output": copy.deepcopy(case.get("model_output")),
        }
        assert_no_forbidden_identifier_keys_or_patterns(item)
        selected.append(item)
        if len(selected) == limit:
            break
    return selected


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_immutable(path: Path, value: object) -> None:
    content = _canonical_bytes(value)
    if path.exists():
        existing_content = path.read_bytes()
        if existing_content != content and isinstance(value, Mapping):
            try:
                existing = json.loads(existing_content)
            except json.JSONDecodeError:
                existing = None
            if (
                isinstance(existing, dict)
                and "created_at" in existing
                and "created_at" in value
            ):
                stable = dict(value)
                stable["created_at"] = existing["created_at"]
                content = _canonical_bytes(stable)
        if existing_content != content:
            raise ValueError(f"immutable artifact already differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _load_selection(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    raw = json.loads(path.read_bytes())
    if not isinstance(raw, dict) or raw.get("schema_version") != (
        "production-semantic-action-openalex-selection-v1"
    ):
        raise ValueError("semantic action selection is invalid")
    selected = raw.get("selected")
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) > 12
        or not all(isinstance(item, dict) for item in selected)
    ):
        raise ValueError("semantic action selection must contain one to twelve cases")
    case_ids = [str(item.get("case_id", "")) for item in selected]
    if any(not value for value in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("semantic action selection has invalid case identities")
    return raw, [dict(item) for item in selected]


def _load_production_components(workspace: Path, lock_path: Path) -> dict[str, object]:
    from paper_search.application.composition import (
        _locked_document_ranker,
        _locked_query_plan_enricher,
        load_locked_identifier_map,
    )
    from paper_search.application.locks import load_verified_input_lock, lock_sha256
    from paper_search.control.pricing import (
        ActualCostPricer,
        parse_pricing_policy_bytes,
    )

    verified = load_verified_input_lock(lock_path, artifact_root=workspace)
    ranker = _locked_document_ranker(
        verified.lock,
        verified.artifact_bytes,
        verified.ranker_artifact_failures,
    )
    if ranker is None:
        raise ValueError("verified production document ranker is unavailable")
    bridge = _locked_query_plan_enricher(verified.lock, verified.artifact_bytes)
    identifier_map, alias_count = load_locked_identifier_map(
        verified.lock,
        verified.artifact_bytes,
    )
    lock = verified.lock
    pricer = ActualCostPricer(
        parse_pricing_policy_bytes(verified.artifact_bytes[lock.pricing_policy.path])
    )
    return {
        "verified": verified,
        "lock": lock,
        "lock_sha256": lock_sha256(lock),
        "ranker": ranker,
        "bridge": bridge,
        "identifier_map": identifier_map,
        "identifier_alias_count": alias_count,
        "pricer": pricer,
    }


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    workspace = args.workspace.resolve()
    lock_path = args.lock.resolve()
    selection_raw, selected = _load_selection(args.selection.resolve())
    components = _load_production_components(workspace, lock_path)
    lock = components["lock"]
    ranker = components["ranker"]
    requests = [
        {
            "case_id": str(case["case_id"]),
            "stratum": str(case.get("stratum", "unclassified")),
            "request": build_openalex_request_spec(case),
        }
        for case in selected
    ]
    assert_no_forbidden_identifier_keys_or_patterns(requests)
    capture_probe = (
        args.capture_root.resolve()
        / "c"
        / "00"
        / "s"
        / "responses"
        / "openalex"
        / ("a" * 64 + ".bin")
    )
    maximum_path_length = len(str(capture_probe))
    if maximum_path_length >= 240:
        raise ValueError("OpenAlex capture path preflight exceeds the safe Windows bound")
    retrieval = lock.baseline.retrieval
    role = getattr(ranker, "deployment_role", None)
    if role != "F5-gated-fusion":
        raise ValueError("production F5 is not the active verified ranker")
    if components["bridge"] is None:
        raise ValueError("production supervised lexical bridge is unavailable")
    if components["identifier_map"] is None:
        raise ValueError("production identifier aliases are unavailable")
    lock_before = _sha256(lock_path.read_bytes())
    report: dict[str, object] = {
        "schema_version": "production-semantic-action-openalex-preflight-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "ready": True,
        "selection_path": str(args.selection.resolve()),
        "selection_sha256": _sha256(args.selection.read_bytes()),
        "selection_schema_version": selection_raw["schema_version"],
        "selected_query_count": len(selected),
        "request_specs": requests,
        "network_payload_fields": ["query_text", "search_mode", "filters", "limit"],
        "online_execution_scope": "selected_llm_action_only",
        "historical_complete_actions_replayed_online": 0,
        "gold_data_loaded": False,
        "gold_or_final_test_sent": False,
        "llm_calls_planned": 0,
        "openalex_logical_calls_planned": len(selected),
        "openalex_raw_attempt_hard_maximum": len(selected) * 3,
        "semantic_scholar_calls_planned": 0,
        "production_modules": {
            "document_ranker_role": role,
            "supervised_lexical_bridge_loaded": True,
            "identifier_alias_count": components["identifier_alias_count"],
            "candidate_caps": {
                "raw": retrieval.max_raw_candidates,
                "deduplicated": retrieval.max_deduplicated_candidates,
                "output": retrieval.max_output_papers,
            },
            "fair_primary_action_merge": "round-robin-raw-cap-plus-rrf",
        },
        "production_lock_path": str(lock_path),
        "production_lock_file_sha256": lock_before,
        "production_lock_model_sha256": components["lock_sha256"],
        "production_lock_modified": _sha256(lock_path.read_bytes()) != lock_before,
        "capture_root": str(args.capture_root.resolve()),
        "capture_path_length_upper_probe": maximum_path_length,
        "network_calls_made": 0,
        "final_test_touched": False,
    }
    _write_immutable(args.output_root.resolve() / "openalex-preflight-v2.json", report)
    return report


def _search_budget() -> object:
    from paper_search.domain.models import SearchBudget

    return SearchBudget(
        max_search_api_calls=3,
        target_search_api_calls=1,
        max_llm_calls=1,
        target_llm_calls=0,
        max_iterations=1,
        max_subqueries=1,
        max_rerank_candidates=0,
        max_output_papers=50,
        max_citation_seeds=0,
        target_citation_seeds=0,
        max_elapsed_seconds=240,
        soft_deadline_seconds=230,
        max_total_tokens=1,
        max_cost_cny=0.10,
    )


def _openalex_keys(env_path: Path) -> tuple[list[str], str | None]:
    from dotenv import dotenv_values

    environment = dotenv_values(env_path)
    keys: list[str] = []
    for index in range(1, 100):
        name = "OPENALEX_API_KEY" if index == 1 else f"OPENALEX_API_KEY_{index}"
        value = environment.get(name)
        if not isinstance(value, str) or not value:
            if index == 1:
                continue
            break
        if value not in keys:
            keys.append(value)
    if not keys:
        raise ValueError("OpenAlex API credential is unavailable")
    mailto = environment.get("OPENALEX_MAILTO")
    return keys, mailto if isinstance(mailto, str) and mailto else None


def _manifest_reader(capture_root: Path) -> object:
    from paper_search.storage.dependency_snapshot import DependencySnapshotReader

    manifest_path = capture_root / "snapshot-manifest.json"
    content = manifest_path.read_bytes()
    raw = json.loads(content)
    return DependencySnapshotReader(
        manifest_path,
        snapshot_manifest_sha256=_sha256(content),
        snapshot_set_id=str(raw["snapshot_set_id"]),
    )


async def _replay_one(capture_root: Path, request: Mapping[str, object]) -> object:
    from paper_search.domain.models import BudgetReservation, UsageEstimate
    from paper_search.retrieval.snapshot_adapters import ReplaySearchProvider

    provider = ReplaySearchProvider(
        dependency="openalex",
        reader=_manifest_reader(capture_root),
    )
    return await provider.search(
        str(request["query_text"]),
        dict(request["filters"]),
        int(request["limit"]),
        BudgetReservation(
            reservation_id="offline-replay",
            action="openalex.semantic-action-canary.replay",
            reserved=UsageEstimate(search_api_calls=3, cost_cny=Decimal("0")),
            expires_at=datetime(2030, 1, 1, tzinfo=UTC),
        ),
    )


def _safe_result_record(
    *,
    case: Mapping[str, object],
    request: Mapping[str, object],
    result: object,
    replay: object,
    capture_root: Path,
) -> dict[str, object]:
    live_ids = [item.canonical_id for item in result.data]
    replay_ids = [item.canonical_id for item in replay.data]
    live_errors = [item.code for item in result.errors]
    replay_errors = [item.code for item in replay.errors]
    manifest_bytes = (capture_root / "snapshot-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    return {
        "schema_version": "production-semantic-action-openalex-case-v1",
        "case_id": str(case["case_id"]),
        "stratum": str(case.get("stratum", "unclassified")),
        "request": dict(request),
        "result_count": len(live_ids),
        "paper_ids": live_ids,
        "error_codes": live_errors,
        "usage": result.usage.model_dump(mode="json"),
        "response_hash": result.provenance.get("response_hash", "unavailable"),
        "snapshot_manifest_path": str(capture_root / "snapshot-manifest.json"),
        "snapshot_manifest_sha256": _sha256(manifest_bytes),
        "snapshot_set_id": manifest["snapshot_set_id"],
        "live_replay_paper_ids_and_order_match": live_ids == replay_ids,
        "live_replay_errors_match": live_errors == replay_errors,
        "gold_data_loaded": False,
        "gold_or_final_test_sent": False,
        "llm_calls_made": 0,
        "semantic_scholar_calls_made": 0,
    }


async def _collect_online(
    args: argparse.Namespace,
    selected: Sequence[Mapping[str, object]],
    components: Mapping[str, object],
) -> list[dict[str, object]]:
    import httpx

    from paper_search.control.budget import HardBudgetController
    from paper_search.domain.models import UsageActual, UsageEstimate
    from paper_search.retrieval.snapshot_adapters import LiveCaptureSearchProvider
    from paper_search.storage.dependency_snapshot import DependencyCaptureStore

    keys, mailto = _openalex_keys(args.env_file.resolve())
    pricer = components["pricer"]
    valued = pricer.value_actual(
        dependency="openalex",
        model_or_adapter="openalex-works-v1",
        usage=UsageActual(search_api_calls=3),
    )
    estimate = UsageEstimate(
        search_api_calls=3,
        cost_cny=valued.cost_cny,
        elapsed_ms=180_000,
    )
    records: list[dict[str, object]] = []
    timeout = httpx.Timeout(connect=10, read=45, write=20, pool=5)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index, case in enumerate(selected):
            request = build_openalex_request_spec(case)
            case_root = args.capture_root.resolve() / "c" / f"{index:02d}"
            capture_root = case_root / "s"
            record_path = case_root / "result.json"
            if (capture_root / "snapshot-manifest.json").exists():
                replay = await _replay_one(capture_root, request)
                if not record_path.exists():
                    raise ValueError("sealed OpenAlex capture lacks its case receipt")
                record = json.loads(record_path.read_bytes())
                if record.get("request") != request:
                    raise ValueError("sealed OpenAlex capture request identity changed")
                if record.get("paper_ids") != [item.canonical_id for item in replay.data]:
                    raise ValueError("sealed OpenAlex capture no longer replays exactly")
                records.append(record)
                print(f"openalex canary {index + 1}/{len(selected)} replayed", flush=True)
                continue
            if capture_root.exists() and any(capture_root.iterdir()):
                raise ValueError("unsealed OpenAlex capture requires manual audit")
            capture = DependencyCaptureStore(capture_root)
            controller = HardBudgetController(
                _search_budget(),
                reservation_ttl_seconds=240,
                formal_live=True,
            )
            provider = LiveCaptureSearchProvider(
                dependency="openalex",
                client=client,
                capture_store=capture,
                pricer=pricer,
                controller=controller,
                api_key=keys[0],
                additional_api_keys=keys[1:],
                mailto=mailto,
                minimum_request_interval_seconds=0.15,
            )
            reservation = controller.reserve(
                f"openalex.semantic-action-canary:{index:02d}",
                estimate,
            )
            result = await provider.search(
                str(request["query_text"]),
                dict(request["filters"]),
                int(request["limit"]),
                reservation,
            )
            capture.seal()
            replay = await _replay_one(capture_root, request)
            record = _safe_result_record(
                case=case,
                request=request,
                result=result,
                replay=replay,
                capture_root=capture_root,
            )
            _write_immutable(record_path, record)
            records.append(record)
            print(f"openalex canary {index + 1}/{len(selected)} captured", flush=True)
    return records


def _load_baseline_rows(
    *,
    selected: Sequence[Mapping[str, object]],
    handoff_path: Path,
    shard_root: Path,
) -> dict[str, dict[str, object]]:
    handoff = json.loads(handoff_path.read_bytes())
    ordered = handoff.get("cumulative_unique_ready_query_ids")
    if not isinstance(ordered, list) or len(ordered) != len(set(ordered)):
        raise ValueError("frozen strict-ready query order is invalid")
    positions = {str(query_id): index for index, query_id in enumerate(ordered)}
    manifest_bytes = (shard_root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("schema_version") not in {
            "large-scale-fusion-query-shards-v1",
            "large-scale-fusion-query-shards-v2",
        }
        or manifest.get("test_partition_touched") is not False
    ):
        raise ValueError("frozen query shard manifest safety checks failed")
    shard_meta = {
        int(item["batch_index"]): item for item in manifest["completed_shards"]
    }
    requested_by_shard: dict[int, set[str]] = {}
    for case in selected:
        case_id = str(case["case_id"])
        if case_id not in positions:
            raise ValueError(f"selected query is absent from strict-ready order: {case_id}")
        requested_by_shard.setdefault(positions[case_id] // 64, set()).add(case_id)
    rows: dict[str, dict[str, object]] = {}
    for shard_index, case_ids in sorted(requested_by_shard.items()):
        path = shard_root / f"shard-{shard_index:05d}.jsonl.gz"
        content = path.read_bytes()
        if _sha256(content) != shard_meta[shard_index]["sha256"]:
            raise ValueError(f"frozen candidate shard hash mismatch: {shard_index}")
        for line in gzip.decompress(content).splitlines():
            row = json.loads(line)
            case_id = str(row.get("query_id", ""))
            if case_id in case_ids:
                rows[case_id] = row
    missing = {str(case["case_id"]) for case in selected}.difference(rows)
    if missing:
        raise ValueError("frozen candidate rows are incomplete")
    return rows


def _source_papers(
    row: Mapping[str, object],
    *,
    added: Sequence[object] = (),
    added_source: str | None = None,
) -> dict[str, list[tuple[int, object]]]:
    from paper_search.learning.cpu_document_ranker import DocumentCandidateEvidence

    sources: dict[str, list[tuple[int, object]]] = {}
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("frozen candidate row is invalid")
    for raw in candidates:
        if not isinstance(raw, Mapping):
            raise ValueError("frozen candidate is invalid")
        item = DocumentCandidateEvidence.model_validate(raw)
        for source, rank in online_source_ranks(raw).items():
            sources.setdefault(source, []).append((rank, item.paper))
    for values in sources.values():
        values.sort(key=lambda value: (value[0], value[1].canonical_id))
    if added_source is not None:
        sources[added_source] = [
            (rank, paper) for rank, paper in enumerate(added, start=1)
        ]
    return sources


def _round_robin_cap_sources(
    sources: Mapping[str, Sequence[tuple[int, object]]],
    limit: int,
) -> dict[str, list[tuple[int, object]]]:
    if sum(len(values) for values in sources.values()) <= limit:
        return {source: list(values) for source, values in sources.items()}
    retained = {source: [] for source in sources}
    depth = max((len(values) for values in sources.values()), default=0)
    count = 0
    for index in range(depth):
        for source, values in sources.items():
            if index >= len(values):
                continue
            retained[source].append(values[index])
            count += 1
            if count == limit:
                return retained
    return retained


def _fuse_ranked_sources(
    sources: Mapping[str, Sequence[tuple[int, object]]],
    *,
    identifier_map: object,
) -> list[object]:
    from paper_search.domain.models import FusedPaper
    from paper_search.processing.deduplicate import deduplicate_papers

    flattened = [paper for values in sources.values() for _rank, paper in values]
    if not flattened:
        return []
    merged = deduplicate_papers(flattened, id_map=identifier_map)
    member_to_rep = {paper.canonical_id: paper.canonical_id for paper in flattened}
    for decision in merged.decisions:
        for member in decision.member_ids:
            member_to_rep[member] = decision.representative_id
    ranks: dict[str, dict[str, int]] = {
        paper.canonical_id: {} for paper in merged.papers
    }
    for source, values in sources.items():
        for rank, paper in values:
            representative = member_to_rep[paper.canonical_id]
            target = ranks.setdefault(representative, {})
            target[source] = min(target.get(source, rank), rank)
    fused = [
        FusedPaper(
            paper=paper,
            score=sum(1.0 / (60 + rank) for rank in ranks[paper.canonical_id].values()),
            source_ranks=ranks[paper.canonical_id],
        )
        for paper in merged.papers
    ]
    fused.sort(key=lambda item: (-item.score, item.paper.canonical_id))
    return fused


def _rank_condition(
    *,
    query: str,
    query_spec: object,
    sources: Mapping[str, Sequence[tuple[int, object]]],
    ranker: object,
    identifier_map: object,
    raw_cap: int,
    deduplicated_cap: int,
    output_cap: int,
) -> dict[str, object]:
    from paper_search.processing.deduplicate import deduplicate_papers
    from paper_search.processing.filter import apply_hard_filters

    capped = _round_robin_cap_sources(sources, raw_cap)
    fused = _fuse_ranked_sources(capped, identifier_map=identifier_map)
    deduplicated_before = len(fused)
    fused = fused[:deduplicated_cap]
    merged = deduplicate_papers(
        [item.paper for item in fused],
        id_map=identifier_map,
    )
    filtered = apply_hard_filters(merged.papers, query_spec)
    accepted = {item.paper.canonical_id: item for item in filtered.accepted}
    selected = [
        item.model_copy(
            update={
                "score": item.score
                * accepted[item.paper.canonical_id].score_multiplier
            }
        )
        for item in fused
        if item.paper.canonical_id in accepted
    ]
    selected.sort(key=lambda item: (-item.score, item.paper.canonical_id))
    contextual = getattr(ranker, "rank_with_context", None)
    ranked = (
        contextual(query, selected, query_spec=query_spec)
        if callable(contextual)
        else ranker.rank(query, selected)
    )
    if {item.paper.canonical_id for item in ranked} != {
        item.paper.canonical_id for item in selected
    }:
        raise ValueError("production ranker changed candidate membership")
    return {
        "ranked": ranked[:output_cap],
        "pool": ranked,
        "raw_before": sum(len(values) for values in sources.values()),
        "raw_after": sum(len(values) for values in capped.values()),
        "deduplicated_before": deduplicated_before,
        "post_filter_count": len(selected),
    }


def _gold_hits(
    papers: Sequence[object],
    gold_ids: Sequence[str],
    *,
    identifier_map: object,
) -> list[int]:
    from paper_search.evaluation.predictions import paper_matches_evaluation_ids

    return [
        index
        for index, item in enumerate(papers, start=1)
        if paper_matches_evaluation_ids(
            item.paper if hasattr(item, "paper") else item,
            gold_ids,
            identifier_map=identifier_map,
        )
    ]


async def _evaluate(
    args: argparse.Namespace,
    selected: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    components: Mapping[str, object],
) -> dict[str, object]:
    from paper_search.domain.models import QuerySpec

    rows = _load_baseline_rows(
        selected=selected,
        handoff_path=args.handoff.resolve(),
        shard_root=args.shard_root.resolve(),
    )
    replay_results = {
        str(case["case_id"]): await _replay_one(
            args.capture_root.resolve() / "c" / f"{index:02d}" / "s",
            build_openalex_request_spec(case),
        )
        for index, case in enumerate(selected)
    }
    lock = components["lock"]
    retrieval = lock.baseline.retrieval
    ranker = components["ranker"]
    identifier_map = components["identifier_map"]
    case_reports: list[dict[str, object]] = []
    for case, record in zip(selected, records, strict=True):
        case_id = str(case["case_id"])
        row = rows[case_id]
        gold_ids = row.get("gold_paper_ids")
        if not isinstance(gold_ids, list) or not all(
            isinstance(item, str) and item for item in gold_ids
        ):
            raise ValueError("frozen query lacks valid Gold identifiers")
        model_output = case.get("model_output")
        if not isinstance(model_output, Mapping):
            raise ValueError("sealed LLM model output is unavailable")
        query_spec = QuerySpec.model_validate(model_output.get("query_spec"))
        query = str(case["query"])
        baseline_sources = _source_papers(row)
        selected_result = replay_results[case_id]
        augmented_sources = _source_papers(
            row,
            added=selected_result.data,
            added_source=f"openalex:llm-semantic-actions-v2:{case_id}",
        )
        baseline = _rank_condition(
            query=query,
            query_spec=query_spec,
            sources=baseline_sources,
            ranker=ranker,
            identifier_map=identifier_map,
            raw_cap=retrieval.max_raw_candidates,
            deduplicated_cap=retrieval.max_deduplicated_candidates,
            output_cap=retrieval.max_output_papers,
        )
        augmented = _rank_condition(
            query=query,
            query_spec=query_spec,
            sources=augmented_sources,
            ranker=ranker,
            identifier_map=identifier_map,
            raw_cap=retrieval.max_raw_candidates,
            deduplicated_cap=retrieval.max_deduplicated_candidates,
            output_cap=retrieval.max_output_papers,
        )
        baseline_pool_hits = _gold_hits(
            baseline["pool"], gold_ids, identifier_map=identifier_map
        )
        augmented_pool_hits = _gold_hits(
            augmented["pool"], gold_ids, identifier_map=identifier_map
        )
        baseline_rank_hits = _gold_hits(
            baseline["ranked"], gold_ids, identifier_map=identifier_map
        )
        augmented_rank_hits = _gold_hits(
            augmented["ranked"], gold_ids, identifier_map=identifier_map
        )
        action_hits = _gold_hits(
            selected_result.data, gold_ids, identifier_map=identifier_map
        )
        baseline_ids = {item.paper.canonical_id for item in baseline["pool"]}
        augmented_ids = {item.paper.canonical_id for item in augmented["pool"]}
        top = {}
        for cutoff in (5, 10, 20):
            top[str(cutoff)] = {
                "baseline": any(rank <= cutoff for rank in baseline_rank_hits),
                "augmented": any(rank <= cutoff for rank in augmented_rank_hits),
            }
        case_reports.append(
            {
                "case_id": case_id,
                "stratum": str(case.get("stratum", "unclassified")),
                "selected_action_result_count": int(record["result_count"]),
                "selected_action_gold_hit": bool(action_hits),
                "baseline_openalex_only_source_count": len(baseline_sources),
                "baseline_pool_count": len(baseline["pool"]),
                "augmented_pool_count": len(augmented["pool"]),
                "new_pool_member_count": len(augmented_ids.difference(baseline_ids)),
                "baseline_member_displaced_count": len(
                    baseline_ids.difference(augmented_ids)
                ),
                "baseline_gold_in_pool": bool(baseline_pool_hits),
                "augmented_gold_in_pool": bool(augmented_pool_hits),
                "gold_pool_improved": bool(augmented_pool_hits)
                and not bool(baseline_pool_hits),
                "gold_pool_regressed": bool(baseline_pool_hits)
                and not bool(augmented_pool_hits),
                "baseline_first_gold_rank": min(baseline_rank_hits, default=None),
                "augmented_first_gold_rank": min(augmented_rank_hits, default=None),
                "top_k_gold_hit": top,
                "raw_candidate_cap": {
                    "baseline_before": baseline["raw_before"],
                    "baseline_after": baseline["raw_after"],
                    "augmented_before": augmented["raw_before"],
                    "augmented_after": augmented["raw_after"],
                },
            }
        )
    aggregates: dict[str, object] = {
        "query_count": len(case_reports),
        "selected_action_nonempty_query_count": sum(
            int(item["selected_action_result_count"] > 0) for item in case_reports
        ),
        "selected_action_gold_hit_query_count": sum(
            int(item["selected_action_gold_hit"]) for item in case_reports
        ),
        "baseline_gold_in_pool_query_count": sum(
            int(item["baseline_gold_in_pool"]) for item in case_reports
        ),
        "augmented_gold_in_pool_query_count": sum(
            int(item["augmented_gold_in_pool"]) for item in case_reports
        ),
        "gold_pool_improved_query_count": sum(
            int(item["gold_pool_improved"]) for item in case_reports
        ),
        "gold_pool_regressed_query_count": sum(
            int(item["gold_pool_regressed"]) for item in case_reports
        ),
        "new_pool_member_count": sum(
            int(item["new_pool_member_count"]) for item in case_reports
        ),
        "baseline_member_displaced_count": sum(
            int(item["baseline_member_displaced_count"]) for item in case_reports
        ),
    }
    for cutoff in (5, 10, 20):
        key = str(cutoff)
        aggregates[f"baseline_top{cutoff}_gold_hit_query_count"] = sum(
            int(item["top_k_gold_hit"][key]["baseline"]) for item in case_reports
        )
        aggregates[f"augmented_top{cutoff}_gold_hit_query_count"] = sum(
            int(item["top_k_gold_hit"][key]["augmented"]) for item in case_reports
        )
    report = {
        "schema_version": "production-semantic-action-openalex-paired-evaluation-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison": (
            "frozen-openalex-only-pool versus same-pool-plus-one-new-llm-action"
        ),
        "candidate_merge": "current-round-robin-raw-cap-plus-rrf",
        "ranker_role": getattr(ranker, "deployment_role", None),
        "identifier_alias_count": components["identifier_alias_count"],
        "candidate_caps": {
            "raw": retrieval.max_raw_candidates,
            "deduplicated": retrieval.max_deduplicated_candidates,
            "output": retrieval.max_output_papers,
        },
        "pasa_only_candidates_included": False,
        "gold_loaded_after_all_online_calls": True,
        "gold_or_final_test_sent": False,
        "llm_calls_during_openalex_stage": 0,
        "semantic_scholar_calls": 0,
        "production_lock_modified": False,
        "final_test_touched": False,
        "aggregates": aggregates,
        "cases": case_reports,
    }
    _write_immutable(
        args.output_root.resolve() / "openalex-paired-evaluation.json",
        report,
    )
    return report


async def _run(args: argparse.Namespace) -> None:
    preflight = _preflight(args)
    if preflight.get("ready") is not True:
        raise ValueError("OpenAlex canary preflight did not pass")
    _selection_raw, selected = _load_selection(args.selection.resolve())
    components = _load_production_components(
        args.workspace.resolve(), args.lock.resolve()
    )
    lock_before = _sha256(args.lock.read_bytes())
    records = await _collect_online(args, selected, components)
    if any(
        record.get("live_replay_paper_ids_and_order_match") is not True
        or record.get("live_replay_errors_match") is not True
        for record in records
    ):
        raise ValueError("OpenAlex live/replay consistency gate failed")
    report = await _evaluate(args, selected, records, components)
    if _sha256(args.lock.read_bytes()) != lock_before:
        raise ValueError("production lock changed during canary")
    collection = {
        "schema_version": "production-semantic-action-openalex-collection-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(records),
        "logical_openalex_call_count": len(records),
        "raw_openalex_attempt_count": sum(
            int(record["usage"]["search_api_calls"]) for record in records
        ),
        "nonempty_query_count": sum(int(record["result_count"] > 0) for record in records),
        "error_query_count": sum(int(bool(record["error_codes"])) for record in records),
        "live_replay_consistent_query_count": sum(
            int(record["live_replay_paper_ids_and_order_match"])
            for record in records
        ),
        "llm_calls_made": 0,
        "semantic_scholar_calls_made": 0,
        "gold_or_final_test_sent": False,
        "production_lock_modified": False,
        "evaluation_aggregates": report["aggregates"],
        "case_receipts": [
            str(args.capture_root.resolve() / "c" / f"{index:02d}" / "result.json")
            for index in range(len(records))
        ],
    }
    _write_immutable(
        args.output_root.resolve() / "openalex-collection-summary.json",
        collection,
    )
    print(json.dumps(collection, ensure_ascii=False, indent=2), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current-module LLM semantic-action OpenAlex production canary."
        )
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--selection",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("deliverables/evaluator/live-evaluator.lock.yaml"),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(r"D:\AI Projects\Projects\.env"),
    )
    parser.add_argument(
        "--handoff",
        type=Path,
        default=Path(
            "data/training_private/training_runs/"
            "openalex-pasa-high-recall-v2-fast64-expanded-21429-v2/"
            "ranking-training-handoff-expanded.json"
        ),
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=Path(
            "data/training_private/training_runs/"
            "fusion-21429-openalex-pasa-high-recall-v2-fast64-context-v4-"
            "20260825-v2/query-shards"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the exact online payload and current production modules without network.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.preflight_only:
        report = _preflight(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    asyncio.run(_run(args))


__all__ = [
    "build_openalex_request_spec",
    "build_paired_model_outputs",
    "online_source_ranks",
    "normalized_action_identity",
    "select_openalex_cases",
]


if __name__ == "__main__":
    main()
