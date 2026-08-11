from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.analyze_sealed_query_recomposition as analyze
import scripts.rescore_identifier_semantics as rescore
from paper_search.domain.models import Paper, ProviderResult, QuerySpec, UsageActual
from paper_search.evaluation.dataset import EvaluationQuery, IdentifierMap
from paper_search.evaluation.execution_adapter import EvaluationExecutionRecord
from paper_search.evaluation.query_evolution_probe import FrozenProbeInputs, FrozenQueryRecord
from paper_search.evaluation.query_recomposition import (
    RecompositionInput,
    SealedQueryRecompositionReport,
)
from paper_search.evaluation.semantic_rescore import SemanticRescoreReport


ROOT = Path(__file__).resolve().parents[2]
HASH = "sha256:" + "1" * 64


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _paper(identifier: str) -> Paper:
    return Paper(canonical_id=identifier, title=f"Title {identifier}")


def _provider_result(*papers: Paper) -> ProviderResult[list[Paper]]:
    return ProviderResult[list[Paper]](
        data=list(papers),
        usage=UsageActual(),
        provenance={
            "provider": "openalex",
            "endpoint": "sealed-offline-recomposition",
            "model_id": "sealed-offline-recomposition",
            "requested_at": "2026-08-11T00:00:00+00:00",
            "response_hash": HASH,
        },
        cache_hit=True,
        latency_ms=0,
        errors=[],
    )


def _execution(query_id: str) -> EvaluationExecutionRecord:
    return EvaluationExecutionRecord(
        query_id=query_id,
        run_id="dev-20260809T061903Z-9bd861e90299",
        outcome_kind="success",
        business_result_sha256=HASH,
        usage=UsageActual(),
        diagnostics=[],
        retrieved_paper_ids=["openalex:W1"],
        post_filter_paper_ids=["openalex:W1"],
        is_partial=False,
        planner_status="primary",
        planner_fallback=False,
        stop_reason="complete",
    )


def _recomposition_report(
    *,
    conclusion: str = "legacy_benchmark_met",
    append_counts: tuple[int, int, int, int] = (101, 0, 23, 19),
) -> SealedQueryRecompositionReport:
    rows = []
    for method, selected in (
        ("append_v2", append_counts[3]),
        ("round_robin_slots", 29),
        ("rrf_slots_k60", 30),
    ):
        counts = append_counts if method == "append_v2" else (100, 0, 143 - 100 - selected, selected)
        rows.append(
            {
                "method": method,
                "true_positive_count": selected,
                "total_gold_associations": sum(counts),
                "not_retrieved": counts[0],
                "filtered_out": counts[1],
                "ranked_outside_top50": counts[2],
                "selected_top50": counts[3],
                "macro_f1": 0.5,
                "macro_recall": 0.5,
                "micro_recall": 0.5,
                "mrr": 0.5,
                "ndcg": 0.5,
                "retains_append_selected_gold": True,
                "retrieved_streams_unchanged": True,
                "post_filter_streams_unchanged": True,
                "usable_signal": method != "append_v2",
            }
        )
    return SealedQueryRecompositionReport.model_validate(
        {
            "input_hashes": {"gold_sha256": HASH},
            "current_formal_selected": 17,
            "legacy_title_selected": 30,
            "rows": rows,
            "conclusion": conclusion,
        }
    )


def _rescore_report(generation_hashes: dict[str, str] | None = None) -> SemanticRescoreReport:
    hashes = generation_hashes or {
        "public_audit_sha256": HASH,
        "gold_sha256": HASH,
        "identity_evidence_sha256": HASH,
        "snapshot_manifest_sha256": HASH,
        "private_audit_sha256": HASH,
        "candidate_map_sha256": HASH,
    }
    return SemanticRescoreReport.model_validate(
        {
            "generation_hashes": hashes,
            "quality_policy_sha256": HASH,
            "total_gold_associations": 143,
            "runs": [
                {
                    "label": label,
                    "kind": kind,
                    "verification_status": verification,
                    "capture_replay_status": replay,
                    "binding_hashes": {"source_sha256": HASH},
                    "true_positive_count": selected,
                    "macro_f1": 0.5,
                    "macro_recall": 0.5,
                    "micro_recall": 0.5,
                    "macro_mrr": 0.5,
                    "macro_ndcg": 0.5,
                    "direct_same_arxiv_hit_count": 0,
                    "pipeline_stages": {
                        "total_gold_associations": 143,
                        "not_retrieved": 100,
                        "filtered_out": 0,
                        "ranked_outside_top50": 143 - 100 - selected,
                        "selected_top50": selected,
                    },
                    "metric_quality_checks": checks,
                }
                for label, kind, verification, replay, selected, checks in (
                    (
                        "formal_baseline_2026_08_10",
                        "formal_run",
                        "formal_validated",
                        "not_applicable",
                        17,
                        [
                            {
                                "rule_id": "hard-filter-recall-loss",
                                "measure": "hard_filter_absolute_recall_loss",
                                "numerator": "0",
                                "denominator": "143",
                                "value": "0",
                                "operator": "lte",
                                "threshold": "0.02",
                                "passed": True,
                            },
                            {
                                "rule_id": "macro-recall-positive",
                                "measure": "macro_identifier_map_recall",
                                "numerator": "1",
                                "denominator": "1",
                                "value": "1",
                                "operator": "gt",
                                "threshold": 0,
                                "passed": True,
                            },
                            {
                                "rule_id": "micro-recall-positive",
                                "measure": "micro_identifier_map_recall",
                                "numerator": "1",
                                "denominator": "1",
                                "value": "1",
                                "operator": "gt",
                                "threshold": 0,
                                "passed": True,
                            },
                        ],
                    ),
                    (
                        "formal_baseline_2026_08_09",
                        "formal_run",
                        "formal_validated",
                        "not_applicable",
                        16,
                        [
                            {
                                "rule_id": "hard-filter-recall-loss",
                                "measure": "hard_filter_absolute_recall_loss",
                                "numerator": "0",
                                "denominator": "143",
                                "value": "0",
                                "operator": "lte",
                                "threshold": "0.02",
                                "passed": True,
                            },
                            {
                                "rule_id": "macro-recall-positive",
                                "measure": "macro_identifier_map_recall",
                                "numerator": "1",
                                "denominator": "1",
                                "value": "1",
                                "operator": "gt",
                                "threshold": 0,
                                "passed": True,
                            },
                            {
                                "rule_id": "micro-recall-positive",
                                "measure": "micro_identifier_map_recall",
                                "numerator": "1",
                                "denominator": "1",
                                "value": "1",
                                "operator": "gt",
                                "threshold": 0,
                                "passed": True,
                            },
                        ],
                    ),
                    (
                        "legacy_title_2026_08_05",
                        "legacy_hash_bound_run",
                        "legacy_hash_bound",
                        "not_applicable",
                        30,
                        [],
                    ),
                    (
                        "query_evolution_prompt_v2",
                        "sealed_probe",
                        "probe_verified",
                        "matched",
                        19,
                        [],
                    ),
                )
            ],
            "decision": {
                "designated_source": "formal_baseline_2026_08_10",
                "primary_loss_stage": "not_retrieved",
                "next_direction": "retrieval_query",
                "reason_codes": [],
            },
        }
    )


def test_module_help_lists_only_fixed_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.analyze_sealed_query_recomposition", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "run" in completed.stdout
    assert "render-markdown" in completed.stdout


def test_cli_has_no_experimental_options() -> None:
    parser = analyze.build_parser()

    assert parser.parse_args(["run"]).command == "run"
    assert parser.parse_args(["render-markdown"]).command == "render-markdown"
    for option in (
        "--json-path",
        "--markdown-path",
        "--source",
        "--network",
        "--env-file",
        "--ledger",
        "--method",
        "--weight",
        "--threshold",
        "--rrf-k",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(["run", option, "value"])


def test_fixed_paths_and_prompt_v2_source_binding_are_canonical() -> None:
    assert analyze.PROBE_RUN_DIR == ROOT / "runs/_diag_query_evolution_query-evolution-prompt-v2-full-20260810"
    assert analyze.EXPECTED_PROBE_SOURCE_RUN_ID == "dev-20260809T061903Z-9bd861e90299"
    assert analyze.EXTERNAL_RESCORE_JSON == ROOT / "docs/evidence/identifier-map-semantic-rescore-2026-08-11.json"
    assert analyze.OUT_JSON == ROOT / "docs/evidence/sealed-query-recomposition-offline-2026-08-11.json"
    assert analyze.OUT_MARKDOWN == ROOT / "docs/sealed-query-recomposition-offline-2026-08-11.md"


def test_generation_failure_stops_before_probe_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analyze,
        "load_verified_identifier_generation",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("identifier generation failed")),
    )
    monkeypatch.setattr(
        analyze,
        "load_verified_probe_materials",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("probe must not be read")),
    )

    with pytest.raises(ValueError, match="identifier generation failed"):
        analyze.build_fixed_report()


def test_external_rescore_must_be_canonical_passed_and_generation_bound(
    tmp_path: Path,
) -> None:
    hashes = {
        "public_audit_sha256": _sha256(b"public"),
        "gold_sha256": _sha256(b"gold"),
        "identity_evidence_sha256": _sha256(b"identity"),
        "snapshot_manifest_sha256": _sha256(b"snapshots"),
        "private_audit_sha256": _sha256(b"private"),
        "candidate_map_sha256": _sha256(b"map"),
    }
    path = tmp_path / "rescore.json"
    path.write_bytes(rescore.canonical_report_bytes(_rescore_report(hashes)))

    benchmark = analyze.load_external_rescore_benchmark(path, hashes)

    assert benchmark.current_formal_selected == 17
    assert benchmark.legacy_title_selected == 30
    assert benchmark.generation_hashes == hashes

    path.write_text(json.dumps(_rescore_report(hashes).model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        analyze.load_external_rescore_benchmark(path, hashes)

    mismatched = dict(hashes)
    mismatched["gold_sha256"] = HASH
    path.write_bytes(rescore.canonical_report_bytes(_rescore_report(hashes)))
    with pytest.raises(ValueError, match="generation hashes"):
        analyze.load_external_rescore_benchmark(path, mismatched)


def test_composition_occurs_before_identifier_map_enters_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    identifier_map = IdentifierMap.from_bytes(b"{}")
    generation = SimpleNamespace(identifier_map=identifier_map)
    gold = [EvaluationQuery(query_id="q-1", query="query", relevant_paper_ids=["openalex:W1"])]
    materials = SimpleNamespace(binding_hashes={"probe_lock_sha256": HASH})
    projections = {"append_v2": {"q-1": object()}}
    benchmark = analyze.ExternalBenchmark(
        current_formal_selected=17,
        legacy_title_selected=30,
        generation_hashes={"gold_sha256": HASH},
    )
    report = _recomposition_report()

    for name in (
        "PUBLIC_AUDIT",
        "GOLD",
        "IDENTITY_EVIDENCE",
        "SNAPSHOT_MANIFEST",
        "PRIVATE_AUDIT",
        "VERIFIED_MAP",
    ):
        path = tmp_path / name.casefold()
        path.write_bytes(b'{"query_id":"q-1","query":"query","relevant_paper_ids":["openalex:W1"]}\n')
        monkeypatch.setattr(analyze, name, path)

    def load_generation(**kwargs: Path) -> object:
        calls.append("generation")
        return generation

    def read_gold(path: Path, model: type[EvaluationQuery]) -> list[EvaluationQuery]:
        calls.append("gold-order")
        return gold

    def load_materials(run_dir: Path, expected_query_ids: tuple[str, ...]) -> object:
        calls.append("materials")
        assert expected_query_ids == ("q-1",)
        return materials

    def compose_inputs(
        material_value: object, expected_query_ids: tuple[str, ...]
    ) -> tuple[RecompositionInput, ...]:
        calls.append("inputs")
        assert material_value is materials
        assert expected_query_ids == ("q-1",)
        return ()

    def project(inputs: tuple[RecompositionInput, ...]) -> object:
        calls.append("compose")
        assert inputs == ()
        return projections

    def load_benchmark(path: Path, generation_hashes: dict[str, str]) -> object:
        calls.append("external")
        assert generation_hashes["gold_sha256"] == _sha256((tmp_path / "gold").read_bytes())
        return benchmark

    def build_report(**kwargs: object) -> SealedQueryRecompositionReport:
        calls.append("score")
        assert kwargs["gold"] is gold
        assert kwargs["identifier_map"] is identifier_map
        assert kwargs["projections"] is projections
        assert kwargs["current_formal_selected"] == 17
        assert kwargs["legacy_title_selected"] == 30
        return report

    monkeypatch.setattr(analyze, "load_verified_identifier_generation", load_generation)
    monkeypatch.setattr(analyze, "read_jsonl", read_gold)
    monkeypatch.setattr(analyze, "load_verified_probe_materials", load_materials)
    monkeypatch.setattr(analyze, "recomposition_inputs_from_materials", compose_inputs)
    monkeypatch.setattr(analyze, "project_all", project)
    monkeypatch.setattr(analyze, "load_external_rescore_benchmark", load_benchmark)
    monkeypatch.setattr(analyze, "build_recomposition_report", build_report)

    assert analyze.build_fixed_report() is report
    assert calls == [
        "generation",
        "gold-order",
        "materials",
        "inputs",
        "compose",
        "external",
        "score",
    ]


def test_recomposition_inputs_use_verified_materials_without_gold_or_identifier_map() -> None:
    baseline = _paper("openalex:W1")
    addition = _paper("openalex:W2")
    query = FrozenQueryRecord(
        query_id="q-1",
        query_spec=QuerySpec(original_query="query", research_goal="goal"),
        search_plan=None,
        baseline_results=[_provider_result(baseline)],
        retrieved_paper_ids=["openalex:W1"],
        source_index=0,
    )
    materials = rescore.VerifiedProbeMaterials(
        baseline_inputs=FrozenProbeInputs(
            queries=[query],
            source_run_id="dev-20260809T061903Z-9bd861e90299",
            source_hashes={},
        ),
        baseline_executions={"q-1": _execution("q-1")},
        additions={"q-1": (_provider_result(addition),)},
        binding_hashes={"probe_lock_sha256": HASH},
    )

    inputs = analyze.recomposition_inputs_from_materials(materials, ("q-1",))

    assert inputs == (
        RecompositionInput(
            query_id="q-1",
            query_spec=query.query_spec,
            baseline_slots=((baseline,),),
            addition_slots=((addition,),),
            retrieved_paper_ids=("openalex:W1",),
            post_filter_paper_ids=("openalex:W1",),
        ),
    )


def test_exact_append_gate_publishes_integrity_failure_before_other_interpretation() -> None:
    report = _recomposition_report(
        conclusion="legacy_benchmark_met",
        append_counts=(100, 0, 24, 19),
    )

    gated = analyze.enforce_append_gate(report)

    assert gated.conclusion == "integrity_failure"
    assert [row.method for row in gated.rows] == [
        "append_v2",
        "round_robin_slots",
        "rrf_slots_k60",
    ]
    assert all(row.not_retrieved == 143 for row in gated.rows)
    assert analyze.enforce_append_gate(_recomposition_report()).conclusion == "legacy_benchmark_met"


@pytest.mark.parametrize("failure", ["source", "hash", "replay", "snapshot", "privacy-json", "privacy-markdown"])
def test_source_and_privacy_failures_publish_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    if failure in {"source", "hash", "replay", "snapshot"}:
        monkeypatch.setattr(
            analyze,
            "build_fixed_report",
            lambda: (_ for _ in ()).throw(ValueError(f"{failure} failed")),
        )
    else:
        monkeypatch.setattr(analyze, "build_fixed_report", _recomposition_report)
        if failure == "privacy-json":
            monkeypatch.setattr(
                analyze,
                "assert_public_json_safe",
                lambda content: (_ for _ in ()).throw(ValueError("private JSON")),
            )
        else:
            monkeypatch.setattr(
                analyze,
                "assert_public_markdown_safe",
                lambda content: (_ for _ in ()).throw(ValueError("private Markdown")),
            )
    monkeypatch.setattr(analyze, "OUT_JSON", json_path)
    monkeypatch.setattr(analyze, "OUT_MARKDOWN", markdown_path)

    assert analyze.main(["run"]) == 3
    assert not json_path.exists()
    assert not markdown_path.exists()


def test_report_rendering_is_canonical_and_public() -> None:
    report = _recomposition_report()

    serialized = analyze.canonical_report_bytes(report)
    markdown = analyze.render_markdown(report)

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
    assert analyze.render_markdown(report) == markdown
    assert "append_v2" in markdown
    assert "legacy_benchmark_met" in markdown


def test_publish_scans_both_artifacts_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    original_json_scanner = analyze.assert_public_json_safe
    original_markdown_scanner = analyze.assert_public_markdown_safe
    original_write = analyze._write_no_replace

    def scan_json(content: bytes) -> None:
        original_json_scanner(content)
        events.append("scan-json")

    def scan_markdown(content: str) -> None:
        original_markdown_scanner(content)
        events.append("scan-markdown")

    def write(path: Path, content: bytes) -> None:
        events.append(f"write-{path.suffix}")
        original_write(path, content)

    monkeypatch.setattr(analyze, "assert_public_json_safe", scan_json)
    monkeypatch.setattr(analyze, "assert_public_markdown_safe", scan_markdown)
    monkeypatch.setattr(analyze, "_write_no_replace", write)

    analyze.publish_report(
        _recomposition_report(),
        json_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
    )

    assert events == ["scan-json", "scan-markdown", "write-.json", "write-.md"]


@pytest.mark.parametrize("existing_suffix", [".json", ".md"])
def test_either_existing_publication_target_prevents_all_writes(
    tmp_path: Path, existing_suffix: str
) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    existing = json_path if existing_suffix == ".json" else markdown_path
    existing.write_bytes(b"existing\n")

    with pytest.raises(ValueError, match="publication target exists"):
        analyze.publish_report(_recomposition_report(), json_path=json_path, markdown_path=markdown_path)

    assert existing.read_bytes() == b"existing\n"
    other = markdown_path if existing is json_path else json_path
    assert not other.exists()


def test_render_markdown_recovers_only_from_canonical_json_without_recomposing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    canonical = analyze.canonical_report_bytes(_recomposition_report())
    json_path.write_bytes(canonical)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("recovery must not recompose or rescore")

    monkeypatch.setattr(analyze, "build_fixed_report", forbidden)
    monkeypatch.setattr(analyze, "build_recomposition_report", forbidden)
    monkeypatch.setattr(analyze, "project_all", forbidden)

    analyze.render_markdown_from_json(json_path, markdown_path)

    assert json_path.read_bytes() == canonical
    assert markdown_path.read_text(encoding="utf-8") == analyze.render_markdown(_recomposition_report())


def test_render_markdown_rejects_noncanonical_json_without_overwriting(tmp_path: Path) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    json_path.write_text(json.dumps(_recomposition_report().model_dump(mode="json")), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical"):
        analyze.render_markdown_from_json(json_path, markdown_path)

    assert not markdown_path.exists()


def test_main_safe_exception_boundary(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        analyze,
        "build_fixed_report",
        lambda: (_ for _ in ()).throw(ValueError("private details")),
    )

    assert analyze.main(["run"]) == 3
    assert capsys.readouterr().err == "sealed query recomposition failed\n"

    monkeypatch.setattr(
        analyze,
        "build_fixed_report",
        lambda: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    with pytest.raises(RuntimeError, match="unexpected"):
        analyze.main(["run"])
