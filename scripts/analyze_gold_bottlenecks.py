"""Aggregate-only gold availability and bottleneck attribution diagnostics."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from paper_search.evaluation.dataset import (
    EvaluationQuery,
    IdentifierMap,
    normalize_paper_id,
    read_jsonl,
    sha256_file,
)


AvailabilityStatus = Literal[
    "available",
    "exact_not_found",
    "unknown_transient",
    "invalid_identifier",
    "integrity_failure",
]
PipelineStage = Literal[
    "selected_top50",
    "ranked_outside_top50",
    "filtered_out",
    "not_retrieved",
]

AVAILABILITY_STATUSES: tuple[AvailabilityStatus, ...] = (
    "available",
    "exact_not_found",
    "unknown_transient",
    "invalid_identifier",
    "integrity_failure",
)
PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    "selected_top50",
    "ranked_outside_top50",
    "filtered_out",
    "not_retrieved",
)
RECOMMENDATION_DIRECTIONS = {
    "new_data_source_probe",
    "retrieval_query_evolution_probe",
    "hard_filter_diagnosis",
    "selector_rerank_offline",
}
REASON_CODES = {
    "exact_not_found_dominant",
    "available_not_retrieved_dominant",
    "available_filtered_out_dominant",
    "available_ranked_out_dominant",
    "unknown_transient_present",
    "invalid_identifier_present",
    "integrity_failure_present",
    "largest_bucket_tie",
    "no_recoverable_loss",
}
FORBIDDEN_OUTPUT_KEYS = frozenset(
    {"query_id", "paper_id", "title", "request_id", "response"}
)
_EXPECTED_DENOMINATORS = (60, 143, 139, 134, 128, 6)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class GoldIndex:
    query_count: int
    raw_gold_identifier_count: int
    normalized_query_work_count: int
    unique_work_count: int
    terminal_identifier_counts: Mapping[str, int]
    query_to_works: Mapping[str, frozenset[str]]
    work_to_identifier_kind: Mapping[str, str]


@dataclass(frozen=True)
class OfflineContext:
    gold_index: GoldIndex
    pipeline_stage_by_association: Mapping[tuple[str, str], PipelineStage]
    unique_stage_counts: Mapping[str, int]
    input_hashes: Mapping[str, str]
    source_run_id: str
    source_git_sha: str


@dataclass(frozen=True)
class ProbeCounters:
    http_attempts: int
    http_status_counts: Mapping[str, int]
    timeout_count: int


@dataclass(frozen=True)
class ProbeBatch:
    status_by_work: Mapping[str, AvailabilityStatus]
    counters: ProbeCounters


@dataclass(frozen=True)
class DiagnosticUsage:
    unique_requests_planned: int
    http_attempts: int
    retries: int
    http_200: int
    http_404: int
    http_429: int
    http_5xx: int
    timeouts: int
    ledger_checkpoint_before_sha256: str
    ledger_checkpoint_after_sha256: str


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid JSONL: {path}") from error
    objects: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL line: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL: {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record is not an object: {path}:{line_number}")
        objects.append(value)
    return objects


def _sha256_hex(path: Path) -> str:
    value = sha256_file(path)
    if not value.startswith("sha256:"):
        raise ValueError(f"unexpected hash format: {path}")
    result = value.removeprefix("sha256:")
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"invalid SHA-256: {path}")
    return result


def _resolve_ids(values: object, identifier_map: IdentifierMap, field: str) -> set[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{field} is invalid")
    resolved: set[str] = set()
    for value in values:
        normalized = normalize_paper_id(value)
        resolved.add(identifier_map.resolve(normalized))
    return resolved


def build_gold_index(gold_path: Path, identifier_map: IdentifierMap) -> GoldIndex:
    records = read_jsonl(gold_path, EvaluationQuery)
    query_to_works: dict[str, frozenset[str]] = {}
    raw_count = 0
    for record in records:
        raw_count += len(record.relevant_paper_ids)
        resolved = frozenset(
            identifier_map.resolve(normalize_paper_id(value))
            for value in record.relevant_paper_ids
        )
        query_to_works[record.query_id] = resolved

    unique_works = set().union(*query_to_works.values()) if query_to_works else set()
    work_to_identifier_kind: dict[str, str] = {}
    for work in unique_works:
        kind, separator, _ = work.partition(":")
        if not separator or kind not in {"doi", "openalex"}:
            work_to_identifier_kind[work] = "invalid_identifier"
        else:
            work_to_identifier_kind[work] = kind
    terminal_identifier_counts = {
        kind: sum(value == kind for value in work_to_identifier_kind.values())
        for kind in ("doi", "openalex")
    }
    return GoldIndex(
        query_count=len(records),
        raw_gold_identifier_count=raw_count,
        normalized_query_work_count=sum(len(values) for values in query_to_works.values()),
        unique_work_count=len(unique_works),
        terminal_identifier_counts=terminal_identifier_counts,
        query_to_works=query_to_works,
        work_to_identifier_kind=work_to_identifier_kind,
    )


def _validate_run(run: Path) -> tuple[dict[str, object], dict[str, object]]:
    run_record = _read_json_object(run / "run.json")
    gates = _read_json_object(run / "gates.json")
    if run_record.get("run_id") != run.name:
        raise ValueError("run ID does not match run directory")
    if run_record.get("status") != "complete":
        raise ValueError("run status is not complete")
    if run_record.get("gate_result") != "passed":
        raise ValueError("Gate result is not passed")
    source_git_sha = run_record.get("source_git_sha")
    if not isinstance(source_git_sha, str) or not _GIT_SHA_RE.fullmatch(source_git_sha):
        raise ValueError("source Git SHA is invalid")
    if (
        gates.get("formal_valid") is not True
        or gates.get("quality_passed") is not True
        or gates.get("gate_result") != "passed"
    ):
        raise ValueError("Gate validity is not passed")
    checks = gates.get("checks")
    if not isinstance(checks, list):
        raise ValueError("Gate checks are invalid")
    provenance_checks = [
        item
        for item in checks
        if isinstance(item, dict) and item.get("rule_id") == "provenance-failures"
    ]
    if len(provenance_checks) != 1:
        raise ValueError("provenance Gate check is missing")
    provenance = provenance_checks[0]
    measure = provenance.get("measure")
    numerator = measure.get("numerator") if isinstance(measure, dict) else None
    if provenance.get("passed") is not True or str(numerator) != "0":
        raise ValueError("provenance-failures Gate check is not zero")
    return run_record, gates


def _stage_for_association(
    work: str,
    selected: set[str],
    post_filter: set[str],
    retrieved: set[str],
) -> PipelineStage:
    if work in selected:
        return "selected_top50"
    if work in post_filter:
        return "ranked_outside_top50"
    if work in retrieved:
        return "filtered_out"
    return "not_retrieved"


def load_offline_context(run: Path, gold_path: Path, id_map_path: Path) -> OfflineContext:
    run_record, _ = _validate_run(run)
    identifier_map = IdentifierMap.from_path(id_map_path)
    gold_index = build_gold_index(gold_path, identifier_map)
    executions = _read_jsonl_objects(run / "executions.jsonl")
    businesses = _read_jsonl_objects(run / "business-results.jsonl")
    execution_by_query = {item.get("query_id"): item for item in executions}
    business_by_query = {item.get("query_id"): item for item in businesses}
    gold_queries = set(gold_index.query_to_works)
    if (
        len(executions) != len(gold_queries)
        or len(businesses) != len(gold_queries)
        or set(execution_by_query) != gold_queries
        or set(business_by_query) != gold_queries
        or any(not isinstance(query_id, str) for query_id in execution_by_query)
        or any(not isinstance(query_id, str) for query_id in business_by_query)
    ):
        raise ValueError("gold, execution, and business query sets differ")

    stages: dict[tuple[str, str], PipelineStage] = {}
    unique_stage_works: dict[PipelineStage, set[str]] = {
        stage: set() for stage in PIPELINE_STAGES
    }
    for query_id in sorted(gold_queries):
        execution = execution_by_query[query_id]
        business = business_by_query[query_id]
        if not isinstance(execution, dict) or not isinstance(business, dict):
            raise ValueError("query records are invalid")
        selected = _resolve_ids(business.get("selected_paper_ids"), identifier_map, "selected IDs")
        post_filter = _resolve_ids(
            execution.get("post_filter_paper_ids"), identifier_map, "post-filter IDs"
        )
        retrieved = _resolve_ids(
            execution.get("retrieved_paper_ids"), identifier_map, "retrieved IDs"
        )
        if not selected <= post_filter <= retrieved:
            raise ValueError("selected ⊆ post-filter ⊆ retrieved invariant failed")
        for work in gold_index.query_to_works[query_id]:
            stage = _stage_for_association(work, selected, post_filter, retrieved)
            stages[(query_id, work)] = stage
            unique_stage_works[stage].add(work)

    input_hashes = {
        "gold_sha256": _sha256_hex(gold_path),
        "identifier_map_sha256": _sha256_hex(id_map_path),
        "executions_sha256": _sha256_hex(run / "executions.jsonl"),
        "business_results_sha256": _sha256_hex(run / "business-results.jsonl"),
        "gates_sha256": _sha256_hex(run / "gates.json"),
        "run_sha256": _sha256_hex(run / "run.json"),
    }
    context = OfflineContext(
        gold_index=gold_index,
        pipeline_stage_by_association=stages,
        unique_stage_counts={stage: len(unique_stage_works[stage]) for stage in PIPELINE_STAGES},
        input_hashes=input_hashes,
        source_run_id=str(run_record["run_id"]),
        source_git_sha=str(run_record["source_git_sha"]),
    )
    if (
        context.gold_index.query_count,
        context.gold_index.raw_gold_identifier_count,
        context.gold_index.normalized_query_work_count,
        context.gold_index.unique_work_count,
        context.gold_index.terminal_identifier_counts.get("doi", 0),
        context.gold_index.terminal_identifier_counts.get("openalex", 0),
    ) != _EXPECTED_DENOMINATORS:
        raise ValueError("gold denominator counts do not match the fixed diagnostic input")
    return context


def _direction_and_reasons(
    context: OfflineContext,
    probe: ProbeBatch,
) -> tuple[bool, str | None, list[str]]:
    counts = {status: 0 for status in AVAILABILITY_STATUSES}
    for work in context.gold_index.work_to_identifier_kind:
        status = probe.status_by_work.get(work)
        if status is None:
            raise ValueError("probe status is incomplete")
        counts[status] += 1
    reasons = [
        f"{status}_present"
        for status in ("unknown_transient", "invalid_identifier", "integrity_failure")
        if counts[status]
    ]
    if reasons:
        return False, None, sorted(reasons)

    buckets = {
        "exact_not_found_dominant": 0,
        "available_not_retrieved_dominant": 0,
        "available_filtered_out_dominant": 0,
        "available_ranked_out_dominant": 0,
    }
    for association, stage in context.pipeline_stage_by_association.items():
        status = probe.status_by_work[association[1]]
        if status == "exact_not_found" and stage != "selected_top50":
            buckets["exact_not_found_dominant"] += 1
        elif status == "available" and stage == "not_retrieved":
            buckets["available_not_retrieved_dominant"] += 1
        elif status == "available" and stage == "filtered_out":
            buckets["available_filtered_out_dominant"] += 1
        elif status == "available" and stage == "ranked_outside_top50":
            buckets["available_ranked_out_dominant"] += 1
    largest = max(buckets.values(), default=0)
    if largest == 0:
        return True, None, ["no_recoverable_loss"]
    winners = [name for name, value in buckets.items() if value == largest]
    if len(winners) != 1:
        return True, None, ["largest_bucket_tie"]
    directions = {
        "exact_not_found_dominant": "new_data_source_probe",
        "available_not_retrieved_dominant": "retrieval_query_evolution_probe",
        "available_filtered_out_dominant": "hard_filter_diagnosis",
        "available_ranked_out_dominant": "selector_rerank_offline",
    }
    return True, directions[winners[0]], [winners[0]]


def assemble_report(
    context: OfflineContext,
    probe: ProbeBatch,
    usage: DiagnosticUsage,
) -> dict[str, object]:
    expected_works = set(context.gold_index.work_to_identifier_kind)
    if set(probe.status_by_work) != expected_works:
        raise ValueError("probe status keys do not match unique gold works")
    availability = {status: 0 for status in AVAILABILITY_STATUSES}
    for status in probe.status_by_work.values():
        if status not in availability:
            raise ValueError("unknown availability status")
        availability[status] += 1

    pipeline_stages = {stage: 0 for stage in PIPELINE_STAGES}
    cross_tab = {
        status: {stage: 0 for stage in PIPELINE_STAGES}
        for status in AVAILABILITY_STATUSES
    }
    query_coverage = {
        status: {stage: set() for stage in PIPELINE_STAGES}
        for status in AVAILABILITY_STATUSES
    }
    for (query_id, work), stage in context.pipeline_stage_by_association.items():
        status = probe.status_by_work[work]
        pipeline_stages[stage] += 1
        cross_tab[status][stage] += 1
        query_coverage[status][stage].add(query_id)
    complete, direction, reasons = _direction_and_reasons(context, probe)
    payload: dict[str, object] = {
        "schema_version": "gold-bottleneck-attribution-v1",
        "source_run_id": context.source_run_id,
        "source_git_sha": context.source_git_sha,
        "input_hashes": dict(context.input_hashes),
        "counts": {
            "query_count": context.gold_index.query_count,
            "raw_gold_identifier_count": context.gold_index.raw_gold_identifier_count,
            "normalized_query_work_count": context.gold_index.normalized_query_work_count,
            "unique_work_count": context.gold_index.unique_work_count,
            "doi_work_count": context.gold_index.terminal_identifier_counts.get("doi", 0),
            "openalex_work_count": context.gold_index.terminal_identifier_counts.get(
                "openalex", 0
            ),
        },
        "availability": availability,
        "pipeline_stages": pipeline_stages,
        "cross_tab": cross_tab,
        "query_coverage": {
            status: {
                stage: len(query_ids)
                for stage, query_ids in stages.items()
            }
            for status, stages in query_coverage.items()
        },
        "usage": {
            "unique_requests_planned": usage.unique_requests_planned,
            "http_attempts": usage.http_attempts,
            "retries": usage.retries,
            "http_200": usage.http_200,
            "http_404": usage.http_404,
            "http_429": usage.http_429,
            "http_5xx": usage.http_5xx,
            "timeouts": usage.timeouts,
            "ledger_checkpoint_before_sha256": usage.ledger_checkpoint_before_sha256,
            "ledger_checkpoint_after_sha256": usage.ledger_checkpoint_after_sha256,
        },
        "diagnostic_complete": complete,
        "recommended_direction": direction,
        "reason_codes": reasons,
    }
    assert_safe_report(payload)
    return payload


def _validate_json_value(value: object, *, key: str | None = None) -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if child_key in FORBIDDEN_OUTPUT_KEYS:
                raise ValueError(f"forbidden key: {child_key}")
            if not isinstance(child_key, str):
                raise ValueError("report object key is not a string")
            _validate_json_value(child_value, key=child_key)
        return
    if isinstance(value, list):
        for child in value:
            _validate_json_value(child, key=key)
        return
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("report contains non-finite number")
        return
    if not isinstance(value, str):
        raise ValueError("report contains unsupported value")
    if key == "source_run_id" and _RUN_ID_RE.fullmatch(value):
        return
    if key == "source_git_sha" and _GIT_SHA_RE.fullmatch(value):
        return
    if key is not None and key.endswith("_sha256") and _SHA256_RE.fullmatch(value):
        return
    if value in {
        "gold-bottleneck-attribution-v1",
        *AVAILABILITY_STATUSES,
        *PIPELINE_STAGES,
        *RECOMMENDATION_DIRECTIONS,
        *REASON_CODES,
    }:
        return
    raise ValueError("forbidden string value in report")


def assert_safe_report(payload: Mapping[str, object]) -> None:
    forbidden_top_level = FORBIDDEN_OUTPUT_KEYS.intersection(payload)
    if forbidden_top_level:
        name = sorted(forbidden_top_level)[0]
        raise ValueError(f"forbidden key: {name}")
    expected_keys = {
        "schema_version",
        "source_run_id",
        "source_git_sha",
        "input_hashes",
        "counts",
        "availability",
        "pipeline_stages",
        "cross_tab",
        "query_coverage",
        "usage",
        "diagnostic_complete",
        "recommended_direction",
        "reason_codes",
    }
    if set(payload) != expected_keys:
        raise ValueError("report has extra keys or missing keys")
    _validate_json_value(dict(payload))
    if payload["schema_version"] != "gold-bottleneck-attribution-v1":
        raise ValueError("invalid report schema version")
    for field in ("availability", "pipeline_stages"):
        value = payload[field]
        expected = AVAILABILITY_STATUSES if field == "availability" else PIPELINE_STAGES
        if not isinstance(value, dict) or tuple(value) != expected:
            raise ValueError(f"invalid {field} keys")
    for field in ("cross_tab", "query_coverage"):
        value = payload[field]
        if not isinstance(value, dict) or tuple(value) != AVAILABILITY_STATUSES:
            raise ValueError(f"invalid {field} status keys")
        if any(
            not isinstance(row, dict) or tuple(row) != PIPELINE_STAGES
            for row in value.values()
        ):
            raise ValueError(f"invalid {field} stage keys")
    direction = payload["recommended_direction"]
    if direction is not None and direction not in RECOMMENDATION_DIRECTIONS:
        raise ValueError("invalid recommended direction")
    reasons = payload["reason_codes"]
    if not isinstance(reasons, list) or any(item not in REASON_CODES for item in reasons):
        raise ValueError("invalid reason codes")
    if reasons != sorted(set(reasons)):
        raise ValueError("reason codes are not unique and sorted")


def write_atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
