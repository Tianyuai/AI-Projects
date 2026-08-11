"""Strict offline adapters for the four sealed identifier-rescore sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import ValidationError, model_validator

import scripts.probe_query_evolution as probe
from paper_search.control.pricing import parse_quality_gate_policy_bytes
from paper_search.domain.models import DomainModel, Paper, ProviderResult, Sha256
from paper_search.evaluation.business_results import (
    BusinessResultRecord,
    business_result_sha256,
)
from paper_search.evaluation.dataset import EvaluationQuery, read_jsonl
from paper_search.evaluation.identifier_semantics import (
    assert_public_json_safe,
    assert_public_markdown_safe,
    load_verified_identifier_generation,
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
    GenerationHashes,
    SemanticRescoreReport,
    SourceLabel,
    SourceProjection,
    build_rescore_report,
)
from paper_search.evaluation.validator import validate_run_directory
from paper_search.storage.dependency_snapshot import (
    DependencySnapshotManifestV2,
    DependencySnapshotReader,
)


ROOT = probe.ROOT
PUBLIC_AUDIT = ROOT / "docs/evidence/identifier-map-semantic-audit-2026-08-11.json"
GOLD = ROOT / "data/dev/gold.jsonl"
IDENTITY_EVIDENCE = ROOT / "data/annotation_work/identifier_semantics/identity-evidence.json"
SNAPSHOT_MANIFEST = ROOT / "data/annotation_work/identifier_semantics/snapshots/snapshot-manifest.json"
PRIVATE_AUDIT = ROOT / "data/annotation_work/identifier_semantics/relation-audit.v2.json"
VERIFIED_MAP = ROOT / "data/annotation_work/identifier_semantics/dev-identifier-map.semantic-v2.json"
QUALITY_POLICY = ROOT / "configs/quality_gates_v1.yaml"
OUT_JSON = ROOT / "docs/evidence/identifier-map-semantic-rescore-2026-08-11.json"
OUT_MARKDOWN = ROOT / "docs/identifier-map-semantic-rescore-2026-08-11.md"
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


@dataclass(frozen=True)
class VerifiedProbeMaterials:
    baseline_inputs: FrozenProbeInputs
    baseline_executions: dict[str, EvaluationExecutionRecord]
    additions: dict[str, tuple[ProviderResult[list[Paper]], ...]]
    binding_hashes: dict[str, str]


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
) -> dict[str, tuple[ProviderResult[list[Paper]], ...]]:
    additions: dict[str, tuple[ProviderResult[list[Paper]], ...]] = {}
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
        converted: list[ProviderResult[list[Paper]]] = []
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
        additions[query_id] = tuple(converted)
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
    materials = load_verified_probe_materials(run_dir, expected_query_ids)
    projection = _probe_projection(
        baseline_inputs=materials.baseline_inputs,
        baseline_executions=materials.baseline_executions,
        additions=materials.additions,
        expected_query_ids=expected_query_ids,
    )
    post_filter_paper_ids = {
        query_id: _stable_union(
            materials.baseline_executions[query_id].post_filter_paper_ids,
            projection.by_query[query_id].post_filter_ids,
        )
        for query_id in expected_query_ids
    }
    return SourceProjection(
        label="query_evolution_prompt_v2",
        kind="sealed_probe",
        verification_status="probe_verified",
        capture_replay_status="matched",
        binding_hashes=materials.binding_hashes,
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


def load_verified_probe_materials(
    run_dir: Path,
    expected_query_ids: tuple[str, ...],
) -> VerifiedProbeMaterials:
    """Verify and load reusable sealed Query Evolution probe materials."""
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

    return VerifiedProbeMaterials(
        baseline_inputs=baseline_inputs,
        baseline_executions=baseline_executions,
        additions=additions,
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


def canonical_report_bytes(report: SemanticRescoreReport) -> bytes:
    """Return the one canonical public JSON representation."""
    return (
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def render_markdown(report: SemanticRescoreReport) -> str:
    """Render the aggregate report without recalculating any value."""
    lines = [
        "# Verified-Identifier Offline Rescore v2",
        "",
        f"Status: {report.status}",
        f"Gold associations: {report.total_gold_associations}",
        "",
        "| Source | TP | Macro F1 | Macro recall | Micro recall | MRR | NDCG | Direct | not_retrieved | filtered_out | ranked_outside_top50 | selected_top50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report.runs:
        stages = run.pipeline_stages
        lines.append(
            f"| {run.label} | {run.true_positive_count} | {run.macro_f1} | "
            f"{run.macro_recall} | {run.micro_recall} | {run.macro_mrr} | "
            f"{run.macro_ndcg} | {run.direct_same_arxiv_hit_count} | "
            f"{stages.not_retrieved} | {stages.filtered_out} | "
            f"{stages.ranked_outside_top50} | {stages.selected_top50} |"
        )
    decision = report.decision
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- primary_loss_stage: {decision.primary_loss_stage}",
            f"- next_direction: {decision.next_direction}",
            f"- reason_codes: {', '.join(decision.reason_codes) or 'none'}",
            "",
        ]
    )
    return "\n".join(lines)


def build_fixed_report() -> SemanticRescoreReport:
    """Build one report from only the fixed verified generation and sources."""
    generation = load_verified_identifier_generation(
        audit_path=PUBLIC_AUDIT,
        gold_path=GOLD,
        evidence_path=IDENTITY_EVIDENCE,
        snapshot_manifest_path=SNAPSHOT_MANIFEST,
        private_audit_path=PRIVATE_AUDIT,
        map_path=VERIFIED_MAP,
    )
    gold = read_jsonl(GOLD, EvaluationQuery)
    expected_query_ids = tuple(query.query_id for query in gold)
    sources = load_fixed_sources(expected_query_ids)
    policy_bytes = QUALITY_POLICY.read_bytes()
    policy = parse_quality_gate_policy_bytes(policy_bytes)
    report = build_rescore_report(
        gold=gold,
        identifier_map=generation.identifier_map,
        sources=sources,
        policy=policy,
        generation_hashes=GenerationHashes(
            public_audit_sha256=_sha256_file(PUBLIC_AUDIT),
            gold_sha256=_sha256_file(GOLD),
            identity_evidence_sha256=_sha256_file(IDENTITY_EVIDENCE),
            snapshot_manifest_sha256=_sha256_file(SNAPSHOT_MANIFEST),
            private_audit_sha256=_sha256_file(PRIVATE_AUDIT),
            candidate_map_sha256=_sha256_file(VERIFIED_MAP),
        ),
        quality_policy_sha256=f"sha256:{hashlib.sha256(policy_bytes).hexdigest()}",
    )
    if report.runs[0].direct_same_arxiv_hit_count != 12:
        raise ValueError("designated direct hit count must equal 12")
    return report


def _write_no_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            raise ValueError("publication target exists") from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def publish_report(
    report: SemanticRescoreReport, *, json_path: Path, markdown_path: Path
) -> None:
    if json_path.exists() or markdown_path.exists():
        raise ValueError("publication target exists")
    json_content = canonical_report_bytes(report)
    markdown = render_markdown(report)
    assert_public_json_safe(json_content)
    assert_public_markdown_safe(markdown)
    _write_no_replace(json_path, json_content)
    _write_no_replace(markdown_path, markdown.encode("utf-8"))


def render_markdown_from_json(json_path: Path, markdown_path: Path) -> None:
    if markdown_path.exists():
        raise ValueError("publication target exists")
    try:
        content = json_path.read_bytes()
        report = SemanticRescoreReport.model_validate_json(content)
    except (OSError, ValidationError):
        raise ValueError("formal JSON is invalid") from None
    if canonical_report_bytes(report) != content:
        raise ValueError("formal JSON is not canonical")
    assert_public_json_safe(content)
    markdown = render_markdown(report)
    assert_public_markdown_safe(markdown)
    _write_no_replace(markdown_path, markdown.encode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verified identifier offline rescore")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    commands.add_parser("render-markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            publish_report(build_fixed_report(), json_path=OUT_JSON, markdown_path=OUT_MARKDOWN)
        else:
            render_markdown_from_json(OUT_JSON, OUT_MARKDOWN)
    except (OSError, ValueError):
        print("identifier rescore failed", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
