"""Fixed zero-network sealed Query Evolution recomposition diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

import scripts.probe_query_evolution as probe
from paper_search.evaluation.dataset import EvaluationQuery, read_jsonl
from paper_search.evaluation.identifier_semantics import (
    assert_public_json_safe,
    assert_public_markdown_safe,
    load_verified_identifier_generation,
)
from paper_search.evaluation.query_recomposition import (
    RecompositionInput,
    SealedQueryRecompositionReport,
    build_report as build_recomposition_report,
    project_all,
)
from paper_search.evaluation.semantic_rescore import SemanticRescoreReport
from scripts.rescore_identifier_semantics import (
    IDENTITY_EVIDENCE,
    PRIVATE_AUDIT,
    PUBLIC_AUDIT,
    SNAPSHOT_MANIFEST,
    VERIFIED_MAP,
    VerifiedProbeMaterials,
    _sha256_file,
    load_verified_probe_materials,
)


ROOT = probe.ROOT
GOLD = ROOT / "data/dev/gold.jsonl"
PROBE_RUN_DIR = (
    ROOT / "runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810"
)
EXPECTED_PROBE_SOURCE_RUN_ID = "dev-20260809T061903Z-9bd861e90299"
EXTERNAL_RESCORE_JSON = (
    ROOT / "docs/evidence/identifier-map-semantic-rescore-2026-08-11.json"
)
OUT_JSON = ROOT / "docs/evidence/sealed-query-recomposition-offline-2026-08-11.json"
OUT_MARKDOWN = ROOT / "docs/sealed-query-recomposition-offline-2026-08-11.md"
_APPEND_GATE = {
    "not_retrieved": 101,
    "filtered_out": 0,
    "ranked_outside_top50": 23,
    "selected_top50": 19,
}


@dataclass(frozen=True)
class ExternalBenchmark:
    current_formal_selected: int
    legacy_title_selected: int
    generation_hashes: dict[str, str]


def _generation_hashes() -> dict[str, str]:
    return {
        "public_audit_sha256": _sha256_file(PUBLIC_AUDIT),
        "gold_sha256": _sha256_file(GOLD),
        "identity_evidence_sha256": _sha256_file(IDENTITY_EVIDENCE),
        "snapshot_manifest_sha256": _sha256_file(SNAPSHOT_MANIFEST),
        "private_audit_sha256": _sha256_file(PRIVATE_AUDIT),
        "candidate_map_sha256": _sha256_file(VERIFIED_MAP),
    }


def _canonical_rescore_bytes(report: SemanticRescoreReport) -> bytes:
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


def load_external_rescore_benchmark(
    path: Path,
    generation_hashes: Mapping[str, str],
) -> ExternalBenchmark:
    """Load the canonical passed rescore benchmark and bind generation hashes."""
    try:
        content = path.read_bytes()
        report = SemanticRescoreReport.model_validate_json(content)
    except (OSError, ValidationError) as error:
        raise ValueError("external rescore benchmark is invalid") from error
    if _canonical_rescore_bytes(report) != content:
        raise ValueError("external rescore benchmark is not canonical")
    if report.status != "passed":
        raise ValueError("external rescore benchmark is not passed")
    actual_hashes = report.generation_hashes.model_dump(mode="json")
    if actual_hashes != dict(generation_hashes):
        raise ValueError("external rescore generation hashes mismatch")
    current = next(
        run
        for run in report.runs
        if run.label == "formal_baseline_2026_08_10"
    )
    legacy = next(run for run in report.runs if run.label == "legacy_title_2026_08_05")
    return ExternalBenchmark(
        current_formal_selected=current.pipeline_stages.selected_top50,
        legacy_title_selected=legacy.pipeline_stages.selected_top50,
        generation_hashes=actual_hashes,
    )


def recomposition_inputs_from_materials(
    materials: VerifiedProbeMaterials,
    expected_query_ids: tuple[str, ...],
) -> tuple[RecompositionInput, ...]:
    """Project verified probe materials into pure recomposition inputs."""
    by_query = {record.query_id: record for record in materials.baseline_inputs.queries}
    if tuple(by_query) != expected_query_ids:
        raise ValueError("probe material query order must match Gold order")
    if materials.baseline_inputs.source_run_id != EXPECTED_PROBE_SOURCE_RUN_ID:
        raise ValueError("probe source run binding mismatch")
    records: list[RecompositionInput] = []
    for query_id in expected_query_ids:
        baseline = by_query[query_id]
        execution = materials.baseline_executions[query_id]
        records.append(
            RecompositionInput(
                query_id=query_id,
                query_spec=baseline.query_spec,
                baseline_slots=tuple(
                    tuple(result.data) for result in baseline.baseline_results
                ),
                addition_slots=tuple(
                    tuple(result.data)
                    for result in materials.additions.get(query_id, ())
                ),
                retrieved_paper_ids=tuple(execution.retrieved_paper_ids),
                post_filter_paper_ids=tuple(execution.post_filter_paper_ids),
            )
        )
    return tuple(records)


def _integrity_failure_like(
    report: SealedQueryRecompositionReport,
) -> SealedQueryRecompositionReport:
    total = report.rows[0].total_gold_associations
    payload = report.model_dump(mode="json")
    payload["conclusion"] = "integrity_failure"
    payload["rows"] = [
        {
            **row,
            "true_positive_count": 0,
            "not_retrieved": total,
            "filtered_out": 0,
            "ranked_outside_top50": 0,
            "selected_top50": 0,
            "macro_f1": 0.0,
            "macro_recall": 0.0,
            "micro_recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "retains_append_selected_gold": False,
            "retrieved_streams_unchanged": False,
            "post_filter_streams_unchanged": False,
            "usable_signal": False,
        }
        for row in payload["rows"]
    ]
    return SealedQueryRecompositionReport.model_validate(payload)


def enforce_append_gate(
    report: SealedQueryRecompositionReport,
) -> SealedQueryRecompositionReport:
    """Require append_v2 to reproduce the canonical stage counts before interpretation."""
    append = report.rows[0]
    actual = {
        "not_retrieved": append.not_retrieved,
        "filtered_out": append.filtered_out,
        "ranked_outside_top50": append.ranked_outside_top50,
        "selected_top50": append.selected_top50,
    }
    if append.method != "append_v2" or actual != _APPEND_GATE:
        return _integrity_failure_like(report)
    return report


def build_fixed_report() -> SealedQueryRecompositionReport:
    """Build the fixed aggregate recomposition report from sealed inputs only."""
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
    materials = load_verified_probe_materials(PROBE_RUN_DIR, expected_query_ids)
    inputs = recomposition_inputs_from_materials(materials, expected_query_ids)
    projections = project_all(inputs)
    benchmark = load_external_rescore_benchmark(
        EXTERNAL_RESCORE_JSON,
        _generation_hashes(),
    )
    report = build_recomposition_report(
        gold=gold,
        identifier_map=generation.identifier_map,
        projections=projections,
        input_hashes={
            **materials.binding_hashes,
            **benchmark.generation_hashes,
            "external_rescore_sha256": _sha256_file(EXTERNAL_RESCORE_JSON),
        },
        current_formal_selected=benchmark.current_formal_selected,
        legacy_title_selected=benchmark.legacy_title_selected,
    )
    return enforce_append_gate(report)


def canonical_report_bytes(report: SealedQueryRecompositionReport) -> bytes:
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


def render_markdown(report: SealedQueryRecompositionReport) -> str:
    """Render the aggregate report without recalculating any value."""
    lines = [
        "# Sealed Query Recomposition Offline Diagnostic",
        "",
        f"Conclusion: {report.conclusion}",
        f"Current formal selected gold: {report.current_formal_selected}",
        f"Legacy title selected gold: {report.legacy_title_selected}",
        "",
        "| Method | TP | Macro F1 | Macro recall | Micro recall | MRR | NDCG | not_retrieved | filtered_out | ranked_outside_top50 | selected_top50 | usable_signal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.method} | {row.true_positive_count} | {row.macro_f1} | "
            f"{row.macro_recall} | {row.micro_recall} | {row.mrr} | "
            f"{row.ndcg} | {row.not_retrieved} | {row.filtered_out} | "
            f"{row.ranked_outside_top50} | {row.selected_top50} | "
            f"{str(row.usable_signal).lower()} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_no_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="xb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
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
    report: SealedQueryRecompositionReport,
    *,
    json_path: Path,
    markdown_path: Path,
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
        report = SealedQueryRecompositionReport.model_validate_json(content)
    except (OSError, ValidationError):
        raise ValueError("formal JSON is invalid") from None
    if canonical_report_bytes(report) != content:
        raise ValueError("formal JSON is not canonical")
    assert_public_json_safe(content)
    markdown = render_markdown(report)
    assert_public_markdown_safe(markdown)
    _write_no_replace(markdown_path, markdown.encode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed sealed query recomposition diagnostic"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    commands.add_parser("render-markdown")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            publish_report(
                build_fixed_report(),
                json_path=OUT_JSON,
                markdown_path=OUT_MARKDOWN,
            )
        else:
            render_markdown_from_json(OUT_JSON, OUT_MARKDOWN)
    except (OSError, ValueError):
        print("sealed query recomposition failed", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
