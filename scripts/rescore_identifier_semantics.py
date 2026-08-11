"""Strict offline adapters for the four sealed identifier-rescore sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import ValidationError, model_validator

import scripts.probe_query_evolution as probe
from paper_search.domain.models import DomainModel, Paper, Sha256
from paper_search.evaluation.business_results import (
    BusinessResultRecord,
    business_result_sha256,
)
from paper_search.evaluation.execution_adapter import EvaluationExecutionRecord
from paper_search.evaluation.query_evolution_probe import (
    FrozenProbeInputs,
    ProbeProjection,
    PublicProbeReport,
    merge_probe_results,
    offline_provider_result,
    reconstruct_frozen_baseline,
)
from paper_search.evaluation.semantic_rescore import (
    SourceLabel,
    SourceProjection,
)
from paper_search.evaluation.validator import validate_run_directory
from paper_search.storage.dependency_snapshot import (
    DependencySnapshotManifestV2,
    DependencySnapshotReader,
)


ROOT = probe.ROOT
_FORMAL_FILES = (
    "run.json",
    "gates.json",
    "predictions.jsonl",
    "executions.jsonl",
    "business-results.jsonl",
)
_PROBE_SOURCE_HASH_KEYS = frozenset(
    {
        "business_results_sha256",
        "executions_sha256",
        "run_sha256",
        "snapshot_manifest_sha256",
    }
)
ModelT = TypeVar("ModelT", bound=DomainModel)


class _ProbeResult(DomainModel):
    schema_version: Literal["query-evolution-probe-result-v1"]
    capture_business_sha256: Sha256
    replay_business_sha256: Sha256
    capture_replay_match: Literal["matched"]
    public_report: PublicProbeReport
    snapshot_manifest_sha256: Sha256
    snapshot_set_id: Sha256
    ledger_checkpoint_sha256: Sha256

    @model_validator(mode="after")
    def validate_business_hashes(self) -> _ProbeResult:
        if self.capture_business_sha256 != self.replay_business_sha256:
            raise ValueError("probe result business hashes must match")
        return self


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path.name}")
    return value


def _jsonl_objects(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name} contains a non-object record")
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL artifact: {path.name}") from error
    return tuple(records)


def _jsonl_models(path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    records: list[ModelT] = []
    try:
        with path.open("rb") as handle:
            for line in handle:
                records.append(model.model_validate_json(line))
    except (OSError, ValidationError) as error:
        raise ValueError(f"invalid {path.name}") from error
    return tuple(records)


def _require_query_ids(
    records: Sequence[BusinessResultRecord | EvaluationExecutionRecord],
    expected_query_ids: tuple[str, ...],
    *,
    label: str,
) -> None:
    query_ids = tuple(record.query_id for record in records)
    if len(query_ids) != len(set(query_ids)) or set(query_ids) != set(
        expected_query_ids
    ):
        raise ValueError(f"{label} query IDs must exactly match expected query IDs")
    if query_ids != expected_query_ids:
        raise ValueError(f"{label} query order must exactly match expected query order")


def _record_maps(
    run_dir: Path,
    expected_query_ids: tuple[str, ...],
    *,
    label: str,
) -> tuple[dict[str, BusinessResultRecord], dict[str, EvaluationExecutionRecord]]:
    business_records = _jsonl_models(
        run_dir / "business-results.jsonl", BusinessResultRecord
    )
    execution_records = _jsonl_models(
        run_dir / "executions.jsonl", EvaluationExecutionRecord
    )
    _require_query_ids(business_records, expected_query_ids, label=f"{label} business")
    _require_query_ids(execution_records, expected_query_ids, label=f"{label} execution")
    business = {record.query_id: record for record in business_records}
    executions = {record.query_id: record for record in execution_records}
    if set(business) != set(executions):
        raise ValueError(f"{label} business and execution query IDs must match")
    for query_id in expected_query_ids:
        if executions[query_id].business_result_sha256 != business_result_sha256(
            business[query_id]
        ):
            raise ValueError(f"{label} business result hash mismatch")
    return business, executions


def _require_raw_subsets(
    *,
    query_id: str,
    retrieved: Sequence[str],
    post_filter: Sequence[str],
    selected: Sequence[str],
) -> None:
    if not set(selected) <= set(post_filter):
        raise ValueError(f"selected IDs must be a subset of post-filter IDs for {query_id}")
    if not set(post_filter) <= set(retrieved):
        raise ValueError(f"post-filter IDs must be a subset of retrieved IDs for {query_id}")


def _projection_from_records(
    *,
    label: SourceLabel,
    kind: Literal["formal_run", "legacy_hash_bound_run"],
    verification_status: Literal["formal_validated", "legacy_hash_bound"],
    binding_hashes: dict[str, str],
    expected_query_ids: tuple[str, ...],
    business: Mapping[str, BusinessResultRecord],
    executions: Mapping[str, EvaluationExecutionRecord],
) -> SourceProjection:
    for query_id in expected_query_ids:
        _require_raw_subsets(
            query_id=query_id,
            retrieved=executions[query_id].retrieved_paper_ids,
            post_filter=executions[query_id].post_filter_paper_ids,
            selected=business[query_id].selected_paper_ids,
        )
    return SourceProjection(
        label=label,
        kind=kind,
        verification_status=verification_status,
        capture_replay_status="not_applicable",
        binding_hashes=binding_hashes,
        query_ids=expected_query_ids,
        retrieved_paper_ids={
            query_id: tuple(executions[query_id].retrieved_paper_ids)
            for query_id in expected_query_ids
        },
        post_filter_paper_ids={
            query_id: tuple(executions[query_id].post_filter_paper_ids)
            for query_id in expected_query_ids
        },
        selected_paper_ids={
            query_id: tuple(business[query_id].selected_paper_ids)
            for query_id in expected_query_ids
        },
    )


def load_formal_source(
    label: SourceLabel,
    run_dir: Path,
    expected_query_ids: tuple[str, ...],
) -> SourceProjection:
    """Validate and project one immutable formal run."""
    validation = validate_run_directory(run_dir)
    if not validation.valid:
        codes = ",".join(issue.code for issue in validation.issues)
        raise ValueError(f"formal run validation failed: {codes}")
    business, executions = _record_maps(
        run_dir, expected_query_ids, label="formal"
    )
    return _projection_from_records(
        label=label,
        kind="formal_run",
        verification_status="formal_validated",
        binding_hashes={
            name.replace("-", "_").replace(".jsonl", "_sha256").replace(
                ".json", "_sha256"
            ): _sha256_file(run_dir / name)
            for name in _FORMAL_FILES
        },
        expected_query_ids=expected_query_ids,
        business=business,
        executions=executions,
    )


def load_legacy_source(
    run_dir: Path,
    evidence_path: Path,
    expected_query_ids: tuple[str, ...],
) -> SourceProjection:
    """Hash-bind and project the immutable pre-formal legacy run."""
    evidence = _json_object(evidence_path)
    if (
        evidence.get("schema_version") != "title-retention-offline-v1"
        or evidence.get("run_id") != run_dir.name
    ):
        raise ValueError("legacy title-retention evidence identity mismatch")
    input_hashes = evidence.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("legacy title-retention evidence hashes are invalid")
    business_hash = _sha256_file(run_dir / "business-results.jsonl")
    execution_hash = _sha256_file(run_dir / "executions.jsonl")
    if business_hash != input_hashes.get("business_results_sha256"):
        raise ValueError("legacy business results hash mismatch")
    if execution_hash != input_hashes.get("executions_sha256"):
        raise ValueError("legacy executions hash mismatch")
    business, executions = _record_maps(
        run_dir, expected_query_ids, label="legacy"
    )
    return _projection_from_records(
        label="legacy_title_2026_08_05",
        kind="legacy_hash_bound_run",
        verification_status="legacy_hash_bound",
        binding_hashes={
            "business_results_sha256": business_hash,
            "executions_sha256": execution_hash,
            "title_retention_evidence_sha256": _sha256_file(evidence_path),
        },
        expected_query_ids=expected_query_ids,
        business=business,
        executions=executions,
    )


def _load_probe_result(path: Path) -> _ProbeResult:
    payload = _json_object(path)
    if payload.get("capture_replay_match") != "matched":
        raise ValueError("probe result capture_replay_match must be matched")
    capture_hash = payload.get("capture_business_sha256")
    replay_hash = payload.get("replay_business_sha256")
    if isinstance(capture_hash, str) and isinstance(replay_hash, str):
        if capture_hash != replay_hash:
            raise ValueError("probe result business hashes must match")
    try:
        return _ProbeResult.model_validate(payload)
    except ValidationError as error:
        raise ValueError("invalid probe result") from error


def _require_ordered_probe_subset(
    lock: probe.ProbeLock, expected_query_ids: tuple[str, ...]
) -> None:
    if len(lock.query_ids) != len(set(lock.query_ids)):
        raise ValueError("probe lock contains duplicate query IDs")
    unknown = set(lock.query_ids).difference(expected_query_ids)
    if unknown:
        raise ValueError(f"probe lock contains unknown query IDs: {sorted(unknown)}")
    locked = set(lock.query_ids)
    if tuple(query_id for query_id in expected_query_ids if query_id in locked) != lock.query_ids:
        raise ValueError("probe lock query IDs must be an ordered subset of expected query IDs")


def _probe_additions(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[object]]:
    additions: dict[str, list[object]] = {}
    for row in rows:
        query_id = row["query_id"]
        if not isinstance(query_id, str):
            raise ValueError("probe outcome query_id is invalid")
        terminal = row.get("terminal")
        if terminal not in {"generated", "no_op"}:
            raise ValueError("probe outcome contains an error terminal")
        searches = row.get("searches")
        if not isinstance(searches, list):
            raise ValueError("probe outcome searches are invalid")
        converted: list[object] = []
        for search in searches:
            if not isinstance(search, dict):
                raise ValueError("probe outcome search is invalid")
            errors = search.get("errors")
            if not isinstance(errors, list):
                raise ValueError("probe outcome search errors are invalid")
            if errors:
                raise ValueError("probe outcome contains search errors")
            data = search.get("data")
            if not isinstance(data, list):
                raise ValueError("probe outcome search data are invalid")
            try:
                papers = [Paper.model_validate(value) for value in data]
            except ValidationError as error:
                raise ValueError("probe outcome contains an invalid paper") from error
            converted.append(offline_provider_result(papers))
        additions[query_id] = converted
    return additions


def _stable_union(*groups: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            if value not in seen:
                seen.add(value)
                values.append(value)
    return tuple(values)


def _probe_projection(
    *,
    baseline_inputs: FrozenProbeInputs,
    baseline_executions: Mapping[str, EvaluationExecutionRecord],
    additions: Mapping[str, Sequence[object]],
    expected_query_ids: tuple[str, ...],
) -> ProbeProjection:
    baseline = reconstruct_frozen_baseline(baseline_inputs, None)
    if baseline.query_ids != expected_query_ids:
        raise ValueError("probe baseline query order must match expected query order")
    projection = merge_probe_results(baseline, additions)  # type: ignore[arg-type]
    if tuple(projection.by_query) != expected_query_ids:
        raise ValueError("probe projection query order must match expected query order")
    for query_id in expected_query_ids:
        row = projection.by_query[query_id]
        post_filter = _stable_union(
            baseline_executions[query_id].post_filter_paper_ids,
            row.post_filter_ids,
        )
        _require_raw_subsets(
            query_id=query_id,
            retrieved=row.retrieved_ids,
            post_filter=post_filter,
            selected=row.top50_ids,
        )
    return projection


def load_probe_source(
    run_dir: Path,
    expected_query_ids: tuple[str, ...],
) -> SourceProjection:
    """Verify and project the sealed Query Evolution capture/replay source."""
    lock_path = run_dir / "probe.lock.json"
    lock = probe.load_probe_lock(lock_path)
    expected_directory = (ROOT / lock.expected_run_directory).resolve()
    if run_dir.resolve() != expected_directory:
        raise ValueError("probe lock expected directory does not match source directory")
    if set(lock.source_hashes) != _PROBE_SOURCE_HASH_KEYS:
        raise ValueError("probe lock source hash keys are invalid")
    _require_ordered_probe_subset(lock, expected_query_ids)
    source_run = probe.verify_probe_source_bindings(lock)

    result_path = run_dir / "result.json"
    outcomes_path = run_dir / "outcomes.jsonl"
    manifest_path = run_dir / "snapshots/snapshot-manifest.json"
    result = _load_probe_result(result_path)
    reader = DependencySnapshotReader(
        manifest_path,
        snapshot_manifest_sha256=result.snapshot_manifest_sha256,
        snapshot_set_id=result.snapshot_set_id,
    )
    try:
        manifest = DependencySnapshotManifestV2.model_validate_json(
            manifest_path.read_bytes()
        )
    except (OSError, ValidationError) as error:
        raise ValueError("invalid probe snapshot manifest") from error
    for entry in manifest.entries:
        reader.read(entry.request)

    outcome_rows = _jsonl_objects(outcomes_path)
    raw_outcome_ids = tuple(row.get("query_id") for row in outcome_rows)
    if not all(isinstance(query_id, str) for query_id in raw_outcome_ids):
        raise ValueError("probe outcome query ID must be a string")
    outcome_ids = tuple(
        query_id for query_id in raw_outcome_ids if isinstance(query_id, str)
    )
    if len(outcome_ids) != len(set(outcome_ids)):
        raise ValueError("probe outcomes contain duplicate query IDs")
    unknown = set(outcome_ids).difference(lock.query_ids)
    if unknown:
        raise ValueError(f"probe outcomes contain unknown query IDs: {sorted(unknown)}")
    if outcome_ids != lock.query_ids:
        raise ValueError("probe outcome order must exactly match the lock")
    additions = _probe_additions(outcome_rows)
    hash_payload = {
        query_id: {
            key: value
            for key, value in row.items()
            if key != "query_id"
        }
        for query_id, row in zip(lock.query_ids, outcome_rows, strict=True)
    }
    if probe.probe_outcome_hash(lock, hash_payload) != result.capture_business_sha256:
        raise ValueError("probe outcome hash mismatch")

    baseline_inputs, _ = probe.frozen_probe_inputs(lock)
    _, baseline_executions = _record_maps(
        source_run, expected_query_ids, label="probe baseline"
    )
    projection = _probe_projection(
        baseline_inputs=baseline_inputs,
        baseline_executions=baseline_executions,
        additions=additions,
        expected_query_ids=expected_query_ids,
    )
    post_filter_paper_ids = {
        query_id: _stable_union(
            baseline_executions[query_id].post_filter_paper_ids,
            projection.by_query[query_id].post_filter_ids,
        )
        for query_id in expected_query_ids
    }
    return SourceProjection(
        label="query_evolution_prompt_v2",
        kind="sealed_probe",
        verification_status="probe_verified",
        capture_replay_status="matched",
        binding_hashes={
            "probe_lock_sha256": _sha256_file(lock_path),
            "probe_result_sha256": _sha256_file(result_path),
            "probe_outcomes_sha256": _sha256_file(outcomes_path),
            "probe_snapshot_manifest_sha256": _sha256_file(manifest_path),
            **{
                f"source_{key}": value
                for key, value in sorted(lock.source_hashes.items())
            },
        },
        query_ids=expected_query_ids,
        retrieved_paper_ids={
            query_id: tuple(projection.by_query[query_id].retrieved_ids)
            for query_id in expected_query_ids
        },
        post_filter_paper_ids=post_filter_paper_ids,
        selected_paper_ids={
            query_id: tuple(projection.by_query[query_id].top50_ids)
            for query_id in expected_query_ids
        },
    )


def load_fixed_sources(
    expected_query_ids: tuple[str, ...], root: Path = ROOT
) -> tuple[SourceProjection, ...]:
    """Load exactly the four fixed offline sources in scorer order."""
    return (
        load_formal_source(
            "formal_baseline_2026_08_10",
            root / "runs/dev-20260810T104256Z-d9e89476d484",
            expected_query_ids,
        ),
        load_formal_source(
            "formal_baseline_2026_08_09",
            root / "runs/dev-20260809T061903Z-9bd861e90299",
            expected_query_ids,
        ),
        load_legacy_source(
            root / "runs/dev-20260805T035209Z-7af4b103f6cc",
            root / "docs/evidence/title-retention-offline-2026-08-09.json",
            expected_query_ids,
        ),
        load_probe_source(
            root
            / "runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810",
            expected_query_ids,
        ),
    )
