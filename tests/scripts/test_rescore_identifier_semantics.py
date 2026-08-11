from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import dotenv
import scripts.probe_query_evolution as probe
import scripts.rescore_identifier_semantics as rescore
import pytest

from paper_search.domain.models import ProviderResult, UsageActual
from paper_search.evaluation.execution_adapter import EvaluationExecutionRecord
from paper_search.evaluation.identifier_semantics import (
    assert_public_json_safe,
    assert_public_markdown_safe,
)
from paper_search.evaluation.query_evolution_probe import FrozenProbeInputs
from paper_search.evaluation.semantic_rescore import SemanticRescoreReport


ROOT = Path(__file__).resolve().parents[2]


def test_probe_script_public_helpers_are_thin_wrappers(monkeypatch) -> None:
    marker = object()
    lock = object()
    outcomes = object()

    monkeypatch.setattr(probe, "_verify_probe_source_bindings", lambda value: (marker, value))
    monkeypatch.setattr(probe, "_frozen_inputs", lambda value: (marker, value))
    monkeypatch.setattr(
        probe,
        "_capture_replay_hash",
        lambda lock_value, outcome_value: (marker, lock_value, outcome_value),
    )

    assert probe.verify_probe_source_bindings(lock) == (marker, lock)
    assert probe.frozen_probe_inputs(lock) == (marker, lock)
    assert probe.probe_outcome_hash(lock, outcomes) == (marker, lock, outcomes)


HASH = "sha256:" + "a" * 64


def _quality_checks() -> list[dict[str, object]]:
    return [
        {
            "rule_id": "hard-filter-recall-loss",
            "measure": "hard_filter_absolute_recall_loss",
            "numerator": "0",
            "denominator": "12",
            "value": "0",
            "operator": "lte",
            "threshold": "0.02",
            "passed": True,
        },
        {
            "rule_id": "macro-recall-positive",
            "measure": "macro_identifier_map_recall",
            "numerator": "12",
            "denominator": "12",
            "value": "1",
            "operator": "gt",
            "threshold": 0,
            "passed": True,
        },
        {
            "rule_id": "micro-recall-positive",
            "measure": "micro_identifier_map_recall",
            "numerator": "12",
            "denominator": "12",
            "value": "1",
            "operator": "gt",
            "threshold": 0,
            "passed": True,
        },
    ]


def _report() -> SemanticRescoreReport:
    metadata = (
        ("formal_baseline_2026_08_10", "formal_run", "formal_validated", "not_applicable"),
        ("formal_baseline_2026_08_09", "formal_run", "formal_validated", "not_applicable"),
        ("legacy_title_2026_08_05", "legacy_hash_bound_run", "legacy_hash_bound", "not_applicable"),
        ("query_evolution_prompt_v2", "sealed_probe", "probe_verified", "matched"),
    )
    return SemanticRescoreReport.model_validate(
        {
            "generation_hashes": {
                "public_audit_sha256": HASH,
                "gold_sha256": HASH,
                "identity_evidence_sha256": HASH,
                "snapshot_manifest_sha256": HASH,
                "private_audit_sha256": HASH,
                "candidate_map_sha256": HASH,
            },
            "quality_policy_sha256": HASH,
            "total_gold_associations": 12,
            "runs": [
                {
                    "label": label,
                    "kind": kind,
                    "verification_status": verification,
                    "capture_replay_status": replay,
                    "binding_hashes": {"source_sha256": HASH},
                    "true_positive_count": 12,
                    "macro_f1": 1.0,
                    "macro_recall": 1.0,
                    "micro_recall": 1.0,
                    "macro_mrr": 1.0,
                    "macro_ndcg": 1.0,
                    "direct_same_arxiv_hit_count": 12 if index == 0 else 0,
                    "pipeline_stages": {
                        "total_gold_associations": 12,
                        "not_retrieved": 0,
                        "filtered_out": 0,
                        "ranked_outside_top50": 0,
                        "selected_top50": 12,
                    },
                    "metric_quality_checks": _quality_checks()
                    if kind == "formal_run"
                    else [],
                }
                for index, (label, kind, verification, replay) in enumerate(metadata)
            ],
            "decision": {
                "designated_source": "formal_baseline_2026_08_10",
                "primary_loss_stage": None,
                "next_direction": None,
                "reason_codes": ["largest_loss_tie"],
            },
        }
    )


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _execution(
    query_id: str,
    *,
    retrieved: list[str] | None = None,
    post_filter: list[str] | None = None,
) -> EvaluationExecutionRecord:
    return EvaluationExecutionRecord(
        query_id=query_id,
        run_id="dev-20260809T061903Z-9bd861e90299",
        outcome_kind="success",
        business_result_sha256=HASH,
        usage=UsageActual(),
        diagnostics=[],
        retrieved_paper_ids=retrieved or ["openalex:W1"],
        post_filter_paper_ids=post_filter or ["openalex:W1"],
        is_partial=False,
        planner_status="primary",
        planner_fallback=False,
        stop_reason="complete",
    )


def test_verified_probe_material_loader_reuses_full_probe_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "runs" / "_diag_query_evolution_query-evolution-prompt-v2-full-20260810"
    (run_dir / "snapshots").mkdir(parents=True)
    for relative in (
        "probe.lock.json",
        "result.json",
        "snapshots/snapshot-manifest.json",
    ):
        (run_dir / relative).write_bytes(relative.encode())
    outcomes = (
        b'{"query_id":"q-1","terminal":"generated","searches":'
        b'[{"errors":[],"data":[{"canonical_id":"openalex:W1","title":"One"}]}]}\n'
    )
    (run_dir / "outcomes.jsonl").write_bytes(outcomes)
    baseline_inputs = FrozenProbeInputs(
        queries=[],
        source_run_id="dev-20260809T061903Z-9bd861e90299",
        source_hashes={"business_results_sha256": HASH},
        expected_query_count=1,
        expected_total_selected=1,
    )
    baseline_execution = _execution("q-1")
    events: list[object] = []
    lock = SimpleNamespace(
        expected_run_directory="runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810",
        source_hashes={
            "business_results_sha256": HASH,
            "executions_sha256": HASH,
            "run_sha256": HASH,
            "snapshot_manifest_sha256": HASH,
        },
        query_ids=("q-1",),
    )
    result = SimpleNamespace(
        capture_business_sha256=HASH,
        replay_business_sha256=HASH,
        capture_replay_match="matched",
        snapshot_manifest_sha256=HASH,
        snapshot_set_id=HASH,
        ledger_checkpoint_sha256=HASH,
    )

    class Reader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            events.append(("reader", args, kwargs))

        def read(self, request: object) -> bytes:
            events.append(("snapshot-read", request))
            return b"snapshot"

    monkeypatch.setattr(rescore, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "load_probe_lock", lambda path: events.append("lock") or lock)
    monkeypatch.setattr(
        probe,
        "verify_probe_source_bindings",
        lambda value: events.append("source-bindings") or (tmp_path / "source"),
    )
    monkeypatch.setattr(rescore, "_load_probe_result", lambda path: events.append("result") or result)
    monkeypatch.setattr(rescore, "DependencySnapshotReader", Reader)
    monkeypatch.setattr(
        rescore,
        "DependencySnapshotManifestV2",
        SimpleNamespace(
            model_validate_json=lambda content: SimpleNamespace(
                entries=(SimpleNamespace(request="first"), SimpleNamespace(request="second"))
            )
        ),
    )
    monkeypatch.setattr(
        probe,
        "probe_outcome_hash",
        lambda lock_value, payload: events.append(("outcome-hash", payload)) or HASH,
    )
    monkeypatch.setattr(
        probe,
        "frozen_probe_inputs",
        lambda value: events.append("frozen-inputs") or (baseline_inputs, {}),
    )
    monkeypatch.setattr(
        rescore,
        "_record_maps",
        lambda source, expected, label: events.append(("records", source, expected, label))
        or ({}, {"q-1": baseline_execution}),
    )

    materials = rescore.load_verified_probe_materials(run_dir, ("q-1",))

    assert isinstance(materials, rescore.VerifiedProbeMaterials)
    assert materials.baseline_inputs is baseline_inputs
    assert materials.baseline_executions == {"q-1": baseline_execution}
    assert isinstance(materials.additions["q-1"][0], ProviderResult)
    assert [paper.canonical_id for paper in materials.additions["q-1"][0].data] == [
        "openalex:W1"
    ]
    assert materials.binding_hashes == {
        "probe_lock_sha256": _sha256(b"probe.lock.json"),
        "probe_result_sha256": _sha256(b"result.json"),
        "probe_outcomes_sha256": _sha256(outcomes),
        "probe_snapshot_manifest_sha256": _sha256(b"snapshots/snapshot-manifest.json"),
        "source_business_results_sha256": HASH,
        "source_executions_sha256": HASH,
        "source_run_sha256": HASH,
        "source_snapshot_manifest_sha256": HASH,
    }
    assert events == [
        "lock",
        "source-bindings",
        "result",
        ("reader", (run_dir / "snapshots/snapshot-manifest.json",), {"snapshot_manifest_sha256": HASH, "snapshot_set_id": HASH}),
        ("snapshot-read", "first"),
        ("snapshot-read", "second"),
        ("outcome-hash", {"q-1": {"terminal": "generated", "searches": [{"errors": [], "data": [{"canonical_id": "openalex:W1", "title": "One"}]}]}}),
        "frozen-inputs",
        ("records", tmp_path / "source", ("q-1",), "probe baseline"),
    ]


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("source", "expected directory"),
        ("hash", "outcome hash mismatch"),
        ("replay", "capture_replay_match must be matched"),
        ("snapshot", "snapshot entry hash mismatch"),
    ],
)
def test_verified_probe_material_loader_rejects_concrete_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    message: str,
) -> None:
    run_dir = tmp_path / "runs" / "probe"
    (run_dir / "snapshots").mkdir(parents=True)
    (run_dir / "probe.lock.json").write_bytes(b"lock")
    (run_dir / "result.json").write_bytes(b"result")
    (run_dir / "outcomes.jsonl").write_bytes(
        b'{"query_id":"q-1","terminal":"generated","searches":[]}\n'
    )
    (run_dir / "snapshots" / "snapshot-manifest.json").write_bytes(b"manifest")
    lock = SimpleNamespace(
        expected_run_directory="runs/other" if tamper == "source" else "runs/probe",
        source_hashes={
            "business_results_sha256": HASH,
            "executions_sha256": HASH,
            "run_sha256": HASH,
            "snapshot_manifest_sha256": HASH,
        },
        query_ids=("q-1",),
    )
    result = SimpleNamespace(
        capture_business_sha256=HASH,
        replay_business_sha256=HASH,
        capture_replay_match="matched",
        snapshot_manifest_sha256=HASH,
        snapshot_set_id=HASH,
        ledger_checkpoint_sha256=HASH,
    )

    class Reader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def read(self, request: object) -> bytes:
            if tamper == "snapshot":
                raise ValueError("snapshot entry hash mismatch")
            return b"snapshot"

    monkeypatch.setattr(rescore, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "load_probe_lock", lambda path: lock)
    monkeypatch.setattr(
        probe, "verify_probe_source_bindings", lambda value: tmp_path / "source"
    )
    if tamper == "replay":
        monkeypatch.setattr(
            rescore,
            "_load_probe_result",
            lambda path: (_ for _ in ()).throw(
                ValueError("probe result capture_replay_match must be matched")
            ),
        )
    else:
        monkeypatch.setattr(rescore, "_load_probe_result", lambda path: result)
    monkeypatch.setattr(rescore, "DependencySnapshotReader", Reader)
    monkeypatch.setattr(
        rescore,
        "DependencySnapshotManifestV2",
        SimpleNamespace(
            model_validate_json=lambda content: SimpleNamespace(
                entries=(SimpleNamespace(request="entry"),)
            )
        ),
    )
    monkeypatch.setattr(
        probe,
        "probe_outcome_hash",
        lambda lock_value, payload: (
            "sha256:" + "b" * 64 if tamper == "hash" else HASH
        ),
    )

    with pytest.raises(ValueError, match=message):
        rescore.load_verified_probe_materials(run_dir, ("q-1",))


def test_load_probe_source_is_thin_projection_over_verified_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_inputs = FrozenProbeInputs(
        queries=[],
        source_run_id="dev-20260809T061903Z-9bd861e90299",
        source_hashes={},
    )
    baseline_execution = _execution(
        "q-1",
        retrieved=["openalex:W1", "openalex:W2", "openalex:W3"],
        post_filter=["openalex:W1", "openalex:W2"],
    )
    materials = rescore.VerifiedProbeMaterials(
        baseline_inputs=baseline_inputs,
        baseline_executions={"q-1": baseline_execution},
        additions={},
        binding_hashes={"probe_lock_sha256": HASH},
    )
    projection = SimpleNamespace(
        by_query={
            "q-1": SimpleNamespace(
                retrieved_ids=("openalex:W1", "openalex:W2", "openalex:W3"),
                post_filter_ids=("openalex:W2", "openalex:W3"),
                top50_ids=("openalex:W3",),
            )
        }
    )
    calls: list[object] = []
    monkeypatch.setattr(
        rescore,
        "load_verified_probe_materials",
        lambda run_dir, expected_query_ids: calls.append((run_dir, expected_query_ids))
        or materials,
    )
    monkeypatch.setattr(
        rescore,
        "_probe_projection",
        lambda **kwargs: calls.append(kwargs) or projection,
    )

    source = rescore.load_probe_source(tmp_path / "probe", ("q-1",))

    assert source.label == "query_evolution_prompt_v2"
    assert source.kind == "sealed_probe"
    assert source.capture_replay_status == "matched"
    assert source.binding_hashes == {"probe_lock_sha256": HASH}
    assert source.retrieved_paper_ids == {
        "q-1": ("openalex:W1", "openalex:W2", "openalex:W3")
    }
    assert source.post_filter_paper_ids == {
        "q-1": ("openalex:W1", "openalex:W2", "openalex:W3")
    }
    assert source.selected_paper_ids == {"q-1": ("openalex:W3",)}
    assert calls == [
        (tmp_path / "probe", ("q-1",)),
        {
            "baseline_inputs": baseline_inputs,
            "baseline_executions": {"q-1": baseline_execution},
            "additions": {},
            "expected_query_ids": ("q-1",),
        },
    ]


def test_rescore_module_help_lists_only_fixed_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.rescore_identifier_semantics", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "run" in completed.stdout
    assert "render-markdown" in completed.stdout


def test_cli_has_no_path_network_env_or_ledger_options() -> None:
    parser = rescore.build_parser()

    assert parser.parse_args(["run"]).command == "run"
    assert parser.parse_args(["render-markdown"]).command == "render-markdown"
    for option in (
        "--json-path",
        "--markdown-path",
        "--source",
        "--network",
        "--env-file",
        "--ledger",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(["run", option, "value"])


def test_generation_failure_stops_before_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rescore,
        "load_verified_identifier_generation",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("identifier semantic audit is not passed")
        ),
    )
    monkeypatch.setattr(
        rescore,
        "load_fixed_sources",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sources must not be read")
        ),
    )

    with pytest.raises(ValueError, match="identifier semantic audit"):
        rescore.build_fixed_report()


def test_build_fixed_report_uses_verified_generation_and_fixed_raw_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contents = {
        "PUBLIC_AUDIT": b"public-audit\n",
        "GOLD": (
            b'{"query_id":"q-1","query":"one","relevant_paper_ids":[]}\n'
            b'{"query_id":"q-2","query":"two","relevant_paper_ids":[]}\n'
        ),
        "IDENTITY_EVIDENCE": b"identity-evidence\n",
        "SNAPSHOT_MANIFEST": b"snapshot-manifest\n",
        "PRIVATE_AUDIT": b"private-audit\n",
        "VERIFIED_MAP": b"verified-map\n",
        "QUALITY_POLICY": b"quality-policy\n",
    }
    paths: dict[str, Path] = {}
    for name, content in contents.items():
        path = tmp_path / name.casefold()
        path.write_bytes(content)
        paths[name] = path
        monkeypatch.setattr(rescore, name, path)

    calls: list[object] = []
    identifier_map = object()
    policy = object()
    sources = object()
    report = _report()

    def load_generation(**kwargs: Path) -> SimpleNamespace:
        calls.append(("generation", kwargs))
        return SimpleNamespace(identifier_map=identifier_map)

    def load_sources(expected_query_ids: tuple[str, ...]) -> object:
        calls.append(("sources", expected_query_ids))
        return sources

    def parse_policy(content: bytes) -> object:
        calls.append(("policy", content))
        return policy

    def build_report(**kwargs: object) -> SemanticRescoreReport:
        calls.append(("report", kwargs))
        return report

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("external entry point must not be used")

    monkeypatch.setattr(rescore, "load_verified_identifier_generation", load_generation)
    monkeypatch.setattr(rescore, "load_fixed_sources", load_sources)
    monkeypatch.setattr(rescore, "parse_quality_gate_policy_bytes", parse_policy)
    monkeypatch.setattr(rescore, "build_rescore_report", build_report)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(dotenv, "dotenv_values", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)

    assert rescore.build_fixed_report() is report
    assert [call[0] for call in calls] == ["generation", "sources", "policy", "report"]
    assert calls[0][1] == {
        "audit_path": paths["PUBLIC_AUDIT"],
        "gold_path": paths["GOLD"],
        "evidence_path": paths["IDENTITY_EVIDENCE"],
        "snapshot_manifest_path": paths["SNAPSHOT_MANIFEST"],
        "private_audit_path": paths["PRIVATE_AUDIT"],
        "map_path": paths["VERIFIED_MAP"],
    }
    assert calls[1] == ("sources", ("q-1", "q-2"))
    assert calls[2] == ("policy", contents["QUALITY_POLICY"])
    report_kwargs = calls[3][1]
    assert report_kwargs["identifier_map"] is identifier_map
    assert report_kwargs["sources"] is sources
    assert report_kwargs["policy"] is policy
    assert tuple(query.query_id for query in report_kwargs["gold"]) == ("q-1", "q-2")
    assert report_kwargs["generation_hashes"].model_dump() == {
        "public_audit_sha256": _sha256(contents["PUBLIC_AUDIT"]),
        "gold_sha256": _sha256(contents["GOLD"]),
        "identity_evidence_sha256": _sha256(contents["IDENTITY_EVIDENCE"]),
        "snapshot_manifest_sha256": _sha256(contents["SNAPSHOT_MANIFEST"]),
        "private_audit_sha256": _sha256(contents["PRIVATE_AUDIT"]),
        "candidate_map_sha256": _sha256(contents["VERIFIED_MAP"]),
    }
    assert report_kwargs["quality_policy_sha256"] == _sha256(
        contents["QUALITY_POLICY"]
    )


def test_build_fixed_report_requires_exact_designated_direct_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    for name in (
        "PUBLIC_AUDIT",
        "GOLD",
        "IDENTITY_EVIDENCE",
        "SNAPSHOT_MANIFEST",
        "PRIVATE_AUDIT",
        "VERIFIED_MAP",
        "QUALITY_POLICY",
    ):
        path = tmp_path / name.casefold()
        path.write_bytes(
            b'{"query_id":"q-1","query":"one","relevant_paper_ids":[]}\n'
            if name == "GOLD"
            else b"content\n"
        )
        monkeypatch.setattr(rescore, name, path)
        paths.append(path)
    payload = _report().model_dump(mode="python")
    payload["runs"][0]["direct_same_arxiv_hit_count"] = 11
    report = SemanticRescoreReport.model_validate(payload)
    monkeypatch.setattr(
        rescore,
        "load_verified_identifier_generation",
        lambda **kwargs: SimpleNamespace(identifier_map=object()),
    )
    monkeypatch.setattr(rescore, "load_fixed_sources", lambda query_ids: object())
    monkeypatch.setattr(rescore, "parse_quality_gate_policy_bytes", lambda content: object())
    monkeypatch.setattr(rescore, "build_rescore_report", lambda **kwargs: report)

    with pytest.raises(ValueError, match="designated direct hit count"):
        rescore.build_fixed_report()


def test_report_rendering_is_canonical_finite_deterministic_and_public() -> None:
    report = _report()

    serialized = rescore.canonical_report_bytes(report)
    markdown = rescore.render_markdown(report)

    assert serialized == (
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert serialized.endswith(b"\n") and not serialized.endswith(b"\n\n")
    assert rescore.render_markdown(report) == markdown
    assert markdown.endswith("\n")
    assert "formal_baseline_2026_08_10" in markdown
    assert "selected_top50" in markdown
    assert "largest_loss_tie" in markdown
    assert_public_json_safe(serialized)
    assert_public_markdown_safe(markdown)


def test_publish_scans_both_artifacts_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    original_json_scanner = rescore.assert_public_json_safe
    original_markdown_scanner = rescore.assert_public_markdown_safe
    original_write = rescore._write_no_replace

    def scan_json(content: bytes) -> None:
        original_json_scanner(content)
        events.append("scan-json")

    def scan_markdown(content: str) -> None:
        original_markdown_scanner(content)
        events.append("scan-markdown")

    def write(path: Path, content: bytes) -> None:
        events.append(f"write-{path.suffix}")
        original_write(path, content)

    monkeypatch.setattr(rescore, "assert_public_json_safe", scan_json)
    monkeypatch.setattr(rescore, "assert_public_markdown_safe", scan_markdown)
    monkeypatch.setattr(rescore, "_write_no_replace", write)
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    rescore.publish_report(_report(), json_path=json_path, markdown_path=markdown_path)

    assert events == ["scan-json", "scan-markdown", "write-.json", "write-.md"]
    assert json_path.read_bytes() == rescore.canonical_report_bytes(_report())
    assert markdown_path.read_text(encoding="utf-8") == rescore.render_markdown(_report())


@pytest.mark.parametrize("existing_suffix", [".json", ".md"])
def test_publish_never_overwrites_either_target(
    tmp_path: Path, existing_suffix: str
) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    existing = json_path if existing_suffix == ".json" else markdown_path
    existing.write_bytes(b"existing\n")

    with pytest.raises(ValueError, match="publication target exists"):
        rescore.publish_report(_report(), json_path=json_path, markdown_path=markdown_path)

    assert existing.read_bytes() == b"existing\n"
    other = markdown_path if existing is json_path else json_path
    assert not other.exists()


def test_render_markdown_recovers_only_from_canonical_json_without_rescoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    canonical = rescore.canonical_report_bytes(_report())
    json_path.write_bytes(canonical)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("recovery must not reload sources or rescore")

    monkeypatch.setattr(rescore, "build_fixed_report", forbidden)
    monkeypatch.setattr(rescore, "build_rescore_report", forbidden)

    rescore.render_markdown_from_json(json_path, markdown_path)

    assert json_path.read_bytes() == canonical
    assert markdown_path.read_text(encoding="utf-8") == rescore.render_markdown(_report())


@pytest.mark.parametrize(
    "invalid_json",
    [
        b'{"schema_version":"identifier-semantic-rescore-v2"}\n',
        b'{ "schema_version": "identifier-semantic-rescore-v2" }\n',
        b'{"schema_version":"identifier-semantic-rescore-v2"}\n\n',
    ],
)
def test_render_markdown_rejects_invalid_or_noncanonical_json(
    tmp_path: Path, invalid_json: bytes
) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    json_path.write_bytes(invalid_json)

    with pytest.raises(ValueError):
        rescore.render_markdown_from_json(json_path, markdown_path)

    assert json_path.read_bytes() == invalid_json
    assert not markdown_path.exists()


def test_render_markdown_never_overwrites_existing_markdown(tmp_path: Path) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    json_path.write_bytes(rescore.canonical_report_bytes(_report()))
    markdown_path.write_bytes(b"existing\n")

    with pytest.raises(ValueError, match="publication target exists"):
        rescore.render_markdown_from_json(json_path, markdown_path)

    assert markdown_path.read_bytes() == b"existing\n"


def test_main_contains_only_expected_failures(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rescore,
        "build_fixed_report",
        lambda: (_ for _ in ()).throw(ValueError("private details")),
    )

    assert rescore.main(["run"]) == 3
    assert capsys.readouterr().err == "identifier rescore failed\n"

    monkeypatch.setattr(
        rescore,
        "build_fixed_report",
        lambda: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    with pytest.raises(RuntimeError, match="unexpected"):
        rescore.main(["run"])


def test_fixed_source_orchestrator_uses_only_the_four_sealed_sources(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        rescore,
        "load_formal_source",
        lambda *args: calls.append(("formal", *args)) or args[0],
    )
    monkeypatch.setattr(
        rescore,
        "load_legacy_source",
        lambda *args: calls.append(("legacy", *args)) or "legacy_title_2026_08_05",
    )
    monkeypatch.setattr(
        rescore,
        "load_probe_source",
        lambda *args: calls.append(("probe", *args)) or "query_evolution_prompt_v2",
    )
    expected_query_ids = ("q-1", "q-2")

    sources = rescore.load_fixed_sources(expected_query_ids, root=ROOT)

    assert sources == (
        "formal_baseline_2026_08_10",
        "formal_baseline_2026_08_09",
        "legacy_title_2026_08_05",
        "query_evolution_prompt_v2",
    )
    assert calls == [
        (
            "formal",
            "formal_baseline_2026_08_10",
            ROOT / "runs/dev-20260810T104256Z-d9e89476d484",
            expected_query_ids,
        ),
        (
            "formal",
            "formal_baseline_2026_08_09",
            ROOT / "runs/dev-20260809T061903Z-9bd861e90299",
            expected_query_ids,
        ),
        (
            "legacy",
            ROOT / "runs/dev-20260805T035209Z-7af4b103f6cc",
            ROOT / "docs/evidence/title-retention-offline-2026-08-09.json",
            expected_query_ids,
        ),
        (
            "probe",
            ROOT / "runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810",
            expected_query_ids,
        ),
    ]
