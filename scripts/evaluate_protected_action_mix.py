"""Recombine frozen paired receipts under a Gold-blind protected action policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)

from paper_search.application.composition import (  # noqa: E402
    _locked_document_ranker,
    load_locked_identifier_map,
)
from paper_search.application.contracts import SearchExecutionResult  # noqa: E402
from paper_search.application.locks import load_verified_input_lock  # noqa: E402
from paper_search.domain.models import (  # noqa: E402
    FusedPaper,
    Paper,
    ProviderResult,
    QuerySpec,
    SearchPlan,
    SubQuery,
    UsageActual,
)
from paper_search.evaluation.predictions import (  # noqa: E402
    paper_matches_evaluation_ids,
)
from paper_search.processing.deduplicate import deduplicate_papers  # noqa: E402
from paper_search.processing.filter import apply_hard_filters  # noqa: E402
from paper_search.query.protected_action_mix import (  # noqa: E402
    ProtectedActionMix,
    select_protected_action_mix,
)
from paper_search.ranking.fusion import fuse_provider_results  # noqa: E402
from paper_search.retrieval.openalex import decode_openalex_page  # noqa: E402
from paper_search.retrieval.semantic_scholar import (  # noqa: E402
    decode_semantic_scholar_search,
)


_DEFAULT_VALIDATION = Path(
    "artifacts/llm-openalex-validations/"
    "semantic-action-six-slot-full-live-24-20260828"
)


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


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return cast(dict[str, object], value)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required at {path}:{line_number}")
        rows.append(cast(dict[str, object], value))
    return rows


def _execution(record: Mapping[str, object]) -> SearchExecutionResult:
    return SearchExecutionResult.model_validate(record["live_execution"])


def paper_identity_record(paper: Paper) -> dict[str, object]:
    """Keep only fields required for alias-aware offline relevance matching."""

    return {
        "canonical_id": paper.canonical_id,
        "title": paper.title,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "openalex_id": paper.openalex_id,
        "semantic_scholar_id": paper.semantic_scholar_id,
    }


def paper_identity_from_record(record: Mapping[str, object]) -> Paper:
    """Restore a minimal Paper without dropping DOI/arXiv/provider aliases."""

    return Paper.model_validate(dict(record))


def _success_response(record: Mapping[str, object]) -> object:
    execution = _execution(record)
    if execution.outcome.kind != "success":
        raise ValueError(f"frozen execution was not successful: {record.get('case_id')}")
    return execution.outcome.response


def round_robin_cap_action_results(
    results: Mapping[str, ProviderResult[list[Paper]]],
    limit: int | None,
) -> dict[str, ProviderResult[list[Paper]]]:
    """Match the production candidate cap while retaining equal action depths."""

    if limit is None or sum(len(item.data) for item in results.values()) <= limit:
        return dict(results)
    source_ids = list(results)
    retained: dict[str, list[Paper]] = {source_id: [] for source_id in source_ids}
    retained_count = 0
    maximum_depth = max(
        (len(results[source_id].data) for source_id in source_ids), default=0
    )
    for rank in range(maximum_depth):
        for source_id in source_ids:
            data = results[source_id].data
            if rank >= len(data):
                continue
            retained[source_id].append(data[rank])
            retained_count += 1
            if retained_count == limit:
                return {
                    item_id: results[item_id].model_copy(
                        update={"data": retained[item_id]}
                    )
                    for item_id in source_ids
                }
    return dict(results)


def offline_confirmation_decision(metrics: Mapping[str, int]) -> dict[str, object]:
    """Decide only whether another disjoint live confirmation is justified."""

    query_count = metrics.get("query_count", 0)
    failed: list[str] = []
    if metrics.get("within_action_budget_query_count") != query_count:
        failed.append("action_budget")
    if metrics.get("f5_query_count") != query_count:
        failed.append("production_f5")
    if metrics.get("hybrid_gold_pool_regressed_query_count", 0) != 0:
        failed.append("gold_pool_query_regression")
    for metric in ("gold_pool", "top5", "top10", "top20"):
        if metrics.get(f"hybrid_{metric}_hit_query_count", 0) < metrics.get(
            f"baseline_{metric}_hit_query_count", 0
        ):
            failed.append(f"{metric}_regression")
    if metrics.get("stratum_regression_count", 0) != 0:
        failed.append("stratum_regression")
    strict_gain = any(
        metrics.get(f"hybrid_{metric}_hit_query_count", 0)
        > metrics.get(f"baseline_{metric}_hit_query_count", 0)
        for metric in ("gold_pool", "top5", "top10", "top20")
    )
    if not strict_gain:
        failed.append("no_strict_gain")
    passed = not failed
    return {
        "passed": passed,
        "decision": (
            "eligible_for_new_disjoint_live_confirmation"
            if passed
            else "keep_current_production"
        ),
        "failed_gates": failed,
        "metrics": dict(metrics),
    }


@dataclass(frozen=True)
class _ActionObservation:
    provider: str
    subquery_id: str
    search_mode: str
    step: str
    result: ProviderResult[list[Paper]]


def _decode_observations(
    record: Mapping[str, object],
    *,
    validation_root: Path,
) -> list[_ActionObservation]:
    response = _success_response(record)
    capture_value = record.get("capture_path")
    if not isinstance(capture_value, str):
        raise ValueError("capture_path is unavailable")
    capture = Path(capture_value).resolve(strict=True)
    root = validation_root.resolve(strict=True)
    if not capture.is_relative_to(root):
        raise ValueError("capture_path escaped the frozen validation root")
    manifest = _load_json(capture / "snapshot-manifest.json")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("snapshot manifest entries are unavailable")
    by_identity: dict[tuple[str, str], Mapping[str, object]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            continue
        request = raw_entry.get("request")
        dependency = request.get("dependency") if isinstance(request, Mapping) else None
        response_hash = raw_entry.get("response_sha256")
        if isinstance(dependency, str) and isinstance(response_hash, str):
            by_identity[(dependency, response_hash)] = raw_entry

    observations: list[_ActionObservation] = []
    for item in response.search_trace:
        if item.get("step") not in {"retrieve", "retrieve_supplement"}:
            continue
        provider = item.get("provider")
        response_hash = item.get("response_hash")
        subquery_id = item.get("subquery_id")
        search_mode = item.get("search_mode")
        if not all(
            isinstance(value, str)
            for value in (provider, response_hash, subquery_id, search_mode)
        ):
            raise ValueError("retrieval trace is incomplete")
        entry = by_identity.get((str(provider), str(response_hash)))
        if entry is None:
            raise ValueError(
                f"snapshot response is unavailable for {provider}:{subquery_id}"
            )
        response_path = entry.get("response_path")
        if not isinstance(response_path, str):
            raise ValueError("snapshot response path is unavailable")
        raw_path = (capture / response_path).resolve(strict=True)
        if not raw_path.is_relative_to(capture):
            raise ValueError("snapshot response escaped its capture")
        raw = raw_path.read_bytes()
        if _sha256(raw) != response_hash:
            raise ValueError("snapshot response hash changed")
        if provider == "openalex":
            decoded = decode_openalex_page(raw, limit=50)
            papers = decoded.papers
            errors = decoded.errors
        elif provider == "semantic_scholar":
            decoded_s2 = decode_semantic_scholar_search(raw, limit=50)
            papers = decoded_s2.papers
            errors = decoded_s2.errors
        else:
            continue
        requested_at = entry.get("captured_at")
        observations.append(
            _ActionObservation(
                provider=provider,
                subquery_id=subquery_id,
                search_mode=search_mode,
                step=str(item["step"]),
                result=ProviderResult(
                    data=papers,
                    usage=UsageActual(),
                    provenance={
                        "provider": provider,
                        "endpoint": "/works"
                        if provider == "openalex"
                        else "/paper/search",
                        "model_id": "frozen-receipt-replay-v1",
                        "requested_at": str(requested_at or "unavailable"),
                        "response_hash": response_hash,
                    },
                    cache_hit=True,
                    latency_ms=0,
                    errors=errors,
                ),
            )
        )
    return observations


def _plan(record: Mapping[str, object]) -> SearchPlan:
    return _success_response(record).query_analysis.search_plan


def _query_spec(record: Mapping[str, object]) -> QuerySpec:
    return _success_response(record).query_analysis.query_spec


def _bridge(plan: SearchPlan) -> SubQuery | None:
    return next(
        (
            item
            for item in plan.subqueries
            if "supervised-lexical-bridge" in item.query_id.casefold()
        ),
        None,
    )


def _copy_action_results(
    observations: Sequence[_ActionObservation],
    *,
    source_query_id: str,
    target_query_id: str,
    supplement: bool = False,
) -> dict[str, ProviderResult[list[Paper]]]:
    expected_step = "retrieve_supplement" if supplement else "retrieve"
    copied: dict[str, ProviderResult[list[Paper]]] = {}
    for item in observations:
        if item.step != expected_step or item.subquery_id != source_query_id:
            continue
        source_id = f"{item.provider}:{target_query_id}:{item.search_mode}"
        copied[source_id] = item.result
    return copied


def _fair_merge(
    baseline: Sequence[FusedPaper],
    supplemental: Sequence[FusedPaper],
    *,
    identifier_map: object,
) -> list[FusedPaper]:
    baseline_papers = [item.paper for item in baseline]
    merged = list(baseline)
    for candidate in supplemental:
        if len(
            deduplicate_papers(
                [*baseline_papers, candidate.paper],
                id_map=identifier_map,
            ).papers
        ) == len(baseline_papers):
            continue
        merged.append(candidate)
    return sorted(merged, key=lambda item: item.score, reverse=True)


def _rank_hybrid(
    *,
    spec: QuerySpec,
    mix: ProtectedActionMix,
    baseline_observations: Sequence[_ActionObservation],
    semantic_observations: Sequence[_ActionObservation],
    baseline_plan: SearchPlan,
    semantic_plan: SearchPlan,
    document_ranker: object,
    identifier_map: object,
) -> tuple[list[Paper], dict[str, object]]:
    primary: dict[str, ProviderResult[list[Paper]]] = {}
    for index, selected in enumerate(mix.actions, start=1):
        observations = (
            baseline_observations
            if selected.origin == "production"
            else semantic_observations
        )
        copied = _copy_action_results(
            observations,
            source_query_id=selected.source_query_id,
            target_query_id=f"sq-{index}",
        )
        if not any(source.startswith("openalex:") for source in copied):
            raise ValueError(
                f"selected action lacks frozen OpenAlex evidence: {selected.source_query_id}"
            )
        primary.update(copied)

    bridge = _bridge(semantic_plan) or _bridge(baseline_plan)
    bridge_origin = semantic_observations if _bridge(semantic_plan) else baseline_observations
    if bridge is not None and len(mix.actions) < 6:
        primary.update(
            _copy_action_results(
                bridge_origin,
                source_query_id=bridge.query_id,
                target_query_id="sq-supervised-lexical-bridge",
            )
        )

    supplemental: dict[str, ProviderResult[list[Paper]]] = {}
    supplement_ids: list[str] = []
    for item in semantic_observations:
        if item.step != "retrieve_supplement" or item.subquery_id in supplement_ids:
            continue
        supplement_ids.append(item.subquery_id)
    for source_query_id in supplement_ids[:1]:
        supplemental.update(
            _copy_action_results(
                semantic_observations,
                source_query_id=source_query_id,
                target_query_id=source_query_id,
                supplement=True,
            )
        )

    primary = round_robin_cap_action_results(primary, 300)
    supplemental = round_robin_cap_action_results(supplemental, 50)
    combined = round_robin_cap_action_results({**primary, **supplemental}, 350)
    primary = {key: value for key, value in combined.items() if key not in supplemental}
    supplemental = {
        key: value for key, value in combined.items() if key in supplemental
    }
    primary_fused = fuse_provider_results(
        primary,
        method="rrf",
        id_map=identifier_map,
    )
    if supplemental:
        supplemental_fused = fuse_provider_results(
            supplemental,
            method="rrf",
            id_map=identifier_map,
        )
        fused = _fair_merge(
            primary_fused,
            supplemental_fused,
            identifier_map=identifier_map,
        )
    else:
        fused = primary_fused
    fused = fused[:200]
    merged = deduplicate_papers(
        [item.paper for item in fused],
        id_map=identifier_map,
    )
    accepted = apply_hard_filters(merged.papers, spec)
    accepted_by_id = {item.paper.canonical_id: item for item in accepted.accepted}
    selected_fused = [
        item.model_copy(
            update={
                "score": item.score
                * accepted_by_id[item.paper.canonical_id].score_multiplier
            }
        )
        for item in fused
        if item.paper.canonical_id in accepted_by_id
    ]
    selected_fused.sort(key=lambda item: (-item.score, item.paper.canonical_id))
    contextual_rank = getattr(document_ranker, "rank_with_context", None)
    if not callable(contextual_rank):
        raise ValueError("production document ranker lacks unified context")
    ranked = contextual_rank(
        spec.original_query,
        selected_fused,
        query_spec=spec,
    )
    deployment_role = getattr(document_ranker, "deployment_role", None)
    return [item.paper for item in ranked], {
        "planner_action_count": len(mix.actions),
        "bridge_action_count": int(bridge is not None),
        "total_primary_action_count": len(mix.actions) + int(bridge is not None),
        "supplement_action_count": len(supplement_ids[:1]),
        "primary_source_count": len(primary),
        "supplement_source_count": len(supplemental),
        "pool_count": len(ranked),
        "deployment_role": deployment_role,
    }


def _first_gold_rank(
    papers: Sequence[Paper],
    gold_ids: Sequence[str],
    *,
    identifier_map: object,
) -> int | None:
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


def _ranked_papers(record: Mapping[str, object]) -> list[Paper]:
    return [item.paper for item in _success_response(record).fused_papers]


def _selection_receipt(
    *,
    case_id: str,
    stratum: str,
    mix: ProtectedActionMix,
    execution_receipt: Mapping[str, object],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "stratum": stratum,
        "selection_policy": "protected-production-lexical-single-slot-v1",
        "selected_actions": [
            {
                "origin": item.origin,
                "source_query_id": item.source_query_id,
                "text": item.action.text,
                "search_mode": item.action.search_mode,
                "action_type": item.action.action_type,
            }
            for item in mix.actions
        ],
        "protected_production_ids": list(mix.protected_production_ids),
        "replaced_production_ids": list(mix.replaced_production_ids),
        "added_semantic_ids": list(mix.added_semantic_ids),
        "execution": dict(execution_receipt),
        "gold_loaded": False,
    }


def _metric_counts(cases: Sequence[Mapping[str, object]]) -> dict[str, int]:
    metrics = {
        "query_count": len(cases),
        "within_action_budget_query_count": sum(
            int(
                isinstance(item.get("execution"), Mapping)
                and int(item["execution"].get("total_primary_action_count", 99)) <= 6
            )
            for item in cases
        ),
        "f5_query_count": sum(
            int(
                isinstance(item.get("execution"), Mapping)
                and item["execution"].get("deployment_role") == "F5-gated-fusion"
            )
            for item in cases
        ),
        "hybrid_gold_pool_regressed_query_count": sum(
            int(item.get("hybrid_gold_pool_regressed") is True) for item in cases
        ),
    }
    for condition in ("baseline", "candidate", "hybrid"):
        metrics[f"{condition}_gold_pool_hit_query_count"] = sum(
            int(item.get(f"{condition}_gold_in_pool") is True) for item in cases
        )
        for cutoff in (5, 10, 20):
            metrics[f"{condition}_top{cutoff}_hit_query_count"] = sum(
                int(
                    isinstance(item.get(f"{condition}_first_gold_rank"), int)
                    and cast(int, item[f"{condition}_first_gold_rank"]) <= cutoff
                )
                for item in cases
            )
    metrics["hybrid_gold_pool_improved_query_count"] = sum(
        int(item.get("hybrid_gold_pool_improved") is True) for item in cases
    )
    return metrics


def _strata(cases: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], list[str]]:
    strata: dict[str, object] = {}
    regressions: list[str] = []
    for stratum in sorted({str(item["stratum"]) for item in cases}):
        rows = [item for item in cases if item["stratum"] == stratum]
        values: dict[str, int] = {"query_count": len(rows)}
        for condition in ("baseline", "candidate", "hybrid"):
            values[f"{condition}_gold_pool"] = sum(
                int(item[f"{condition}_gold_in_pool"] is True) for item in rows
            )
            for cutoff in (5, 10, 20):
                values[f"{condition}_top{cutoff}"] = sum(
                    int(
                        isinstance(item[f"{condition}_first_gold_rank"], int)
                        and cast(int, item[f"{condition}_first_gold_rank"]) <= cutoff
                    )
                    for item in rows
                )
        for metric in ("gold_pool", "top5", "top10", "top20"):
            if values[f"hybrid_{metric}"] < values[f"baseline_{metric}"]:
                regressions.append(f"{stratum}:{metric}")
        strata[stratum] = values
    return strata, regressions


def evaluate(
    *,
    workspace: Path,
    validation_root: Path,
    production_lock_path: Path,
    output_root: Path,
) -> dict[str, object]:
    sample = _load_json(validation_root / "sample-manifest.json")
    raw_cases = sample.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("sample cases are unavailable")
    cases = [cast(dict[str, object], item) for item in raw_cases if isinstance(item, dict)]
    baseline_paths = sorted((validation_root / "baseline" / "results").glob("*.json"))
    semantic_paths = sorted((validation_root / "candidate" / "results").glob("*.json"))
    baseline_records = [_load_json(path) for path in baseline_paths]
    semantic_records = [_load_json(path) for path in semantic_paths]
    if not (len(cases) == len(baseline_records) == len(semantic_records)):
        raise ValueError("paired frozen receipts are incomplete")

    verified = load_verified_input_lock(
        production_lock_path,
        artifact_root=workspace,
    )
    document_ranker = _locked_document_ranker(
        verified.lock,
        verified.artifact_bytes,
        verified.ranker_artifact_failures,
    )
    if document_ranker is None:
        raise ValueError("production document ranker is unavailable")
    identifier_map, alias_count = load_locked_identifier_map(
        verified.lock,
        verified.artifact_bytes,
    )
    if identifier_map is None:
        raise ValueError("production identifier aliases are unavailable")

    hybrid_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for case, baseline, semantic in zip(
        cases, baseline_records, semantic_records, strict=True
    ):
        case_id = str(case["case_id"])
        if baseline.get("case_id") != case_id or semantic.get("case_id") != case_id:
            raise ValueError("paired case order changed")
        spec = _query_spec(semantic)
        production_plan = _plan(baseline)
        semantic_plan = _plan(semantic)
        mix = select_protected_action_mix(spec, production_plan, semantic_plan)
        baseline_observations = _decode_observations(
            baseline,
            validation_root=validation_root,
        )
        semantic_observations = _decode_observations(
            semantic,
            validation_root=validation_root,
        )
        ranked, execution_receipt = _rank_hybrid(
            spec=spec,
            mix=mix,
            baseline_observations=baseline_observations,
            semantic_observations=semantic_observations,
            baseline_plan=production_plan,
            semantic_plan=semantic_plan,
            document_ranker=document_ranker,
            identifier_map=identifier_map,
        )
        selection_rows.append(
            _selection_receipt(
                case_id=case_id,
                stratum=str(case["stratum"]),
                mix=mix,
                execution_receipt=execution_receipt,
            )
        )
        hybrid_rows.append(
            {
                "case_id": case_id,
                "stratum": str(case["stratum"]),
                "papers": [paper_identity_record(paper) for paper in ranked],
                "execution": execution_receipt,
            }
        )

    selection_artifact = {
        "schema_version": "protected-semantic-action-selection-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "gold_loaded": False,
        "network_calls": 0,
        "query_count": len(selection_rows),
        "rows": selection_rows,
    }
    selection_bytes = _canonical_bytes(selection_artifact)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selection-receipt.json").write_bytes(selection_bytes)
    (output_root / "hybrid-ranked-pools.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in hybrid_rows
        ),
        encoding="utf-8",
    )

    sources = sample.get("sources")
    partition_source = sources.get("partition") if isinstance(sources, Mapping) else None
    partition_value = (
        partition_source.get("path") if isinstance(partition_source, Mapping) else None
    )
    if not isinstance(partition_value, str):
        raise ValueError("frozen partition path is unavailable")
    partition_path = (workspace / partition_value).resolve(strict=True)
    if not partition_path.is_relative_to(workspace.resolve(strict=True)):
        raise ValueError("partition path escaped the workspace")
    partition = {str(row["query_id"]): row for row in _load_jsonl(partition_path)}

    case_reports: list[dict[str, object]] = []
    for case, baseline, semantic, hybrid, selection in zip(
        cases,
        baseline_records,
        semantic_records,
        hybrid_rows,
        selection_rows,
        strict=True,
    ):
        case_id = str(case["case_id"])
        source = partition.get(case_id)
        if source is None:
            raise ValueError(f"query is absent from frozen partition: {case_id}")
        gold = source.get("gold_paper_ids")
        if not isinstance(gold, list) or not all(isinstance(item, str) for item in gold):
            raise ValueError(f"query lacks frozen Gold identifiers: {case_id}")
        baseline_execution = _execution(baseline)
        semantic_execution = _execution(semantic)
        baseline_pool = list(baseline_execution.pre_truncation_candidates)
        semantic_pool = list(semantic_execution.pre_truncation_candidates)
        raw_hybrid_papers = hybrid.get("papers")
        if not isinstance(raw_hybrid_papers, list) or not all(
            isinstance(item, Mapping) for item in raw_hybrid_papers
        ):
            raise ValueError("hybrid ranked paper identities are unavailable")
        hybrid_ranked = [
            paper_identity_from_record(cast(Mapping[str, object], item))
            for item in raw_hybrid_papers
        ]

        baseline_pool_rank = _first_gold_rank(
            baseline_pool,
            gold,
            identifier_map=identifier_map,
        )
        semantic_pool_rank = _first_gold_rank(
            semantic_pool,
            gold,
            identifier_map=identifier_map,
        )
        baseline_rank = _first_gold_rank(
            _ranked_papers(baseline),
            gold,
            identifier_map=identifier_map,
        )
        semantic_rank = _first_gold_rank(
            _ranked_papers(semantic),
            gold,
            identifier_map=identifier_map,
        )
        hybrid_rank = _first_gold_rank(
            hybrid_ranked,
            gold,
            identifier_map=identifier_map,
        )
        case_reports.append(
            {
                "case_id": case_id,
                "stratum": str(case["stratum"]),
                "baseline_gold_in_pool": baseline_pool_rank is not None,
                "candidate_gold_in_pool": semantic_pool_rank is not None,
                "hybrid_gold_in_pool": hybrid_rank is not None,
                "hybrid_gold_pool_improved": baseline_pool_rank is None
                and hybrid_rank is not None,
                "hybrid_gold_pool_regressed": baseline_pool_rank is not None
                and hybrid_rank is None,
                "baseline_first_gold_rank": baseline_rank,
                "candidate_first_gold_rank": semantic_rank,
                "hybrid_first_gold_rank": hybrid_rank,
                "baseline_pool_count": len(baseline_pool),
                "candidate_pool_count": len(semantic_pool),
                "hybrid_pool_count": len(hybrid_ranked),
                "protected_production_count": len(
                    cast(list[object], selection["protected_production_ids"])
                ),
                "replaced_production_count": len(
                    cast(list[object], selection["replaced_production_ids"])
                ),
                "added_semantic_count": len(
                    cast(list[object], selection["added_semantic_ids"])
                ),
                "execution": selection["execution"],
            }
        )

    metrics = _metric_counts(case_reports)
    strata, stratum_regressions = _strata(case_reports)
    metrics["stratum_regression_count"] = len(stratum_regressions)
    decision = offline_confirmation_decision(metrics)
    report = {
        "schema_version": "protected-semantic-action-offline-gate-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "comparison": "current-production-v1-versus-protected-semantic-mix-v1",
        "sample_role": "reused-disjoint-live-canary-offline-recombination",
        "network_calls": 0,
        "gold_loaded_after_hybrid_construction": True,
        "selection_receipt_sha256": _sha256(selection_bytes),
        "identifier_alias_count": alias_count,
        "production_lock_file_sha256": _sha256(production_lock_path.read_bytes()),
        "production_lock_modified": False,
        "metrics": metrics,
        "strata": strata,
        "stratum_regressions": stratum_regressions,
        "decision": decision,
        "selection_summary": {
            "query_count_with_protected_production_actions": sum(
                int(item["protected_production_count"] > 0) for item in case_reports
            ),
            "query_count_with_replacement": sum(
                int(item["replaced_production_count"] > 0) for item in case_reports
            ),
            "query_count_with_semantic_addition": sum(
                int(item["added_semantic_count"] > 0) for item in case_reports
            ),
        },
        "cases": case_reports,
    }
    (output_root / "offline-gate.json").write_bytes(_canonical_bytes(report))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline protected-action recombination over paired live receipts."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--validation-root", type=Path, default=_DEFAULT_VALIDATION)
    parser.add_argument(
        "--production-lock",
        type=Path,
        default=Path("deliverables/evaluator/live-evaluator.lock.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_VALIDATION / "protected-action-mix-offline-v1",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    workspace = args.workspace.resolve(strict=True)
    validation_root = (workspace / args.validation_root).resolve(strict=True)
    production_lock = (workspace / args.production_lock).resolve(strict=True)
    output_root = (workspace / args.output_root).resolve()
    for path in (validation_root, production_lock):
        if not path.is_relative_to(workspace):
            raise ValueError("input escaped the workspace")
    if not output_root.is_relative_to(workspace):
        raise ValueError("output escaped the workspace")
    report = evaluate(
        workspace=workspace,
        validation_root=validation_root,
        production_lock_path=production_lock,
        output_root=output_root,
    )
    print(json.dumps(report["decision"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "evaluate",
    "offline_confirmation_decision",
    "paper_identity_from_record",
    "paper_identity_record",
    "round_robin_cap_action_results",
]
