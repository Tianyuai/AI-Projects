import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

import paper_search.evaluation.runner as runner_module
import paper_search.evaluation.validator as validator_module
from paper_search.evaluation.business_results import (
    BusinessResultRecord,
    business_result_sha256,
)
from paper_search.evaluation.validator import (
    RunValidationResult,
    ValidationIssue,
    validate_run_directory,
)


FIXTURE_ROOT = Path("tests/fixtures/formal_run")


def test_missing_run_directory_returns_stable_safe_issue(tmp_path: Path) -> None:
    path = tmp_path / "private-run-name"

    result = validate_run_directory(path)

    assert result == RunValidationResult(
        valid=False,
        run_id=None,
        issues=[
            ValidationIssue(
                code="run_directory_unavailable",
                artifact="run.json",
                detail="Required formal run evidence is unavailable",
            )
        ],
    )
    assert "private-run-name" not in result.model_dump_json()


def test_synthetic_capture_and_replay_are_valid() -> None:
    assert validate_run_directory(FIXTURE_ROOT / "capture").valid
    assert validate_run_directory(FIXTURE_ROOT / "replay").valid


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("extra", "file_set_invalid"),
        ("status", "run_not_complete"),
        ("business", "business_hash_invalid"),
        ("secret", "sanitization_invalid"),
    ],
)
def test_validator_rejects_major_artifact_categories(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    run = tmp_path / "capture"
    shutil.copytree(FIXTURE_ROOT / "capture", run)
    if mutation == "extra":
        (run / "extra.json").write_text("{}\n", encoding="utf-8")
    elif mutation == "status":
        payload = json.loads((run / "run.json").read_bytes())
        payload["status"] = "failed"
        (run / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "business":
        lines = (run / "business-results.jsonl").read_text(encoding="utf-8").splitlines()
        payload = json.loads(lines[0])
        payload["warnings"] = ["changed"]
        (run / "business-results.jsonl").write_text(
            json.dumps(payload) + "\n" + "\n".join(lines[1:]) + "\n",
            encoding="utf-8",
        )
    else:
        payload = json.loads((run / "run.json").read_bytes())
        payload["experiment_name"] = "authorization"
        (run / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    result = validate_run_directory(run)

    assert not result.valid
    assert expected_code in {issue.code for issue in result.issues}


def test_capture_rejects_unbound_nested_snapshot_file(tmp_path: Path) -> None:
    run = tmp_path / "capture"
    shutil.copytree(FIXTURE_ROOT / "capture", run)
    (run / "snapshots" / "unbound.bin").write_bytes(b"private")

    result = validate_run_directory(run)

    assert not result.valid
    assert "snapshot_tree_invalid" in {issue.code for issue in result.issues}


def test_validator_recomputes_metrics_from_frozen_evidence(tmp_path: Path) -> None:
    run = tmp_path / "capture"
    shutil.copytree(FIXTURE_ROOT / "capture", run)
    payload = json.loads((run / "metrics.json").read_bytes())
    payload["summary"]["macro_recall"] = 0.25
    (run / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    result = validate_run_directory(run)

    assert not result.valid
    assert "metrics_invalid" in {issue.code for issue in result.issues}


def test_replay_rejects_absolute_capture_run_id(tmp_path: Path) -> None:
    fixture = tmp_path / "formal_run"
    shutil.copytree(FIXTURE_ROOT, fixture)
    replay_lock_path = fixture / "replay" / "replay.lock.yaml"
    payload = yaml.safe_load(replay_lock_path.read_bytes())
    payload["source_capture_run_id"] = str((fixture / "capture").resolve())
    replay_lock_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    result = validate_run_directory(fixture / "replay")

    assert not result.valid
    assert "artifact_invalid" in {issue.code for issue in result.issues}


def test_validator_rejects_forged_cost_and_open_reservations(tmp_path: Path) -> None:
    run = tmp_path / "capture"
    shutil.copytree(FIXTURE_ROOT / "capture", run)
    payload = json.loads((run / "usage.json").read_bytes())
    payload["actual"]["cost_cny"] = "999"
    payload["reserved"]["search_api_calls"] = 1
    payload["reserved"]["cost_cny"] = "1"
    payload["project_actual_cny"] = "0"
    payload["within_caps"] = True
    (run / "usage.json").write_text(json.dumps(payload), encoding="utf-8")

    result = validate_run_directory(run)

    assert not result.valid
    assert "ledger_invalid" in {issue.code for issue in result.issues}


def test_capture_replay_lock_must_inherit_live_config(tmp_path: Path) -> None:
    run = tmp_path / "capture"
    shutil.copytree(FIXTURE_ROOT / "capture", run)
    replay_lock_path = run / "replay.lock.yaml"
    payload = yaml.safe_load(replay_lock_path.read_bytes())
    payload["budget_config"]["path"] = "configs/unrelated_budget.yaml"
    replay_lock_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    result = validate_run_directory(run)

    assert not result.valid
    assert "replay_binding_invalid" in {issue.code for issue in result.issues}


def test_capture_replay_lock_must_inherit_project_ledger_anchor(tmp_path: Path) -> None:
    run = tmp_path / "capture"
    shutil.copytree(FIXTURE_ROOT / "capture", run)
    replay_lock_path = run / "replay.lock.yaml"
    payload = yaml.safe_load(replay_lock_path.read_bytes())
    payload["project_ledger"] = {
        "receipt_count": 1,
        "receipts_sha256": "sha256:" + "b" * 64,
    }
    replay_lock_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    result = validate_run_directory(run)

    assert not result.valid
    assert "replay_binding_invalid" in {issue.code for issue in result.issues}


def test_replay_uses_fresh_ledger_while_retaining_capture_anchor() -> None:
    payload = yaml.safe_load(
        (FIXTURE_ROOT / "replay" / "config.lock.yaml").read_bytes()
    )
    payload["project_ledger"] = {
        "receipt_count": 7,
        "receipts_sha256": "sha256:" + "b" * 64,
    }
    lock = runner_module.ReplayLock.model_validate(payload)

    assert validator_module._project_ledger_anchor(lock) == (
        0,
        validator_module._receipts_sha256([]),
    )


def test_validator_rejects_private_platform_path(tmp_path: Path) -> None:
    run = tmp_path / "capture"
    shutil.copytree(FIXTURE_ROOT / "capture", run)
    payload = json.loads((run / "run.json").read_bytes())
    payload["experiment_name"] = "C:\\Users\\alice\\secret\\trace.log"
    (run / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    result = validate_run_directory(run)

    assert not result.valid
    assert "sanitization_invalid" in {issue.code for issue in result.issues}


def test_failure_diagnostics_digest_is_recomputed(tmp_path: Path) -> None:
    run = tmp_path / "capture"
    shutil.copytree(FIXTURE_ROOT / "capture", run)
    business_lines = (run / "business-results.jsonl").read_text(encoding="utf-8").splitlines()
    business_payload = json.loads(business_lines[0])
    business_payload.update(
        {
            "hard_failure_code": "dependency_failure",
            "stop_reason": "dependency_failure",
        }
    )
    business_record = BusinessResultRecord.model_validate(business_payload)
    business_lines[0] = business_record.model_dump_json()
    (run / "business-results.jsonl").write_text(
        "\n".join(business_lines) + "\n", encoding="utf-8"
    )
    execution_lines = (run / "executions.jsonl").read_text(encoding="utf-8").splitlines()
    execution = json.loads(execution_lines[0])
    execution.update(
        {
            "outcome_kind": "failure",
            "stop_reason": "dependency_failure",
            "business_result_sha256": business_result_sha256(business_record),
        }
    )
    execution_lines[0] = json.dumps(execution)
    (run / "executions.jsonl").write_text(
        "\n".join(execution_lines) + "\n", encoding="utf-8"
    )
    failure = {
        "schema_version": "evaluation-failure-v1",
        "query_id": "q1",
        "run_id": "capture",
        "error_code": "dependency_failure",
        "retryable": True,
        "stop_reason": "dependency_failure",
        "usage": execution["usage"],
        "dependency_error_codes": [],
        "diagnostics": [],
        "diagnostics_sha256": "sha256:" + "a" * 64,
    }
    (run / "failures.jsonl").write_text(json.dumps(failure) + "\n", encoding="utf-8")
    manifest = json.loads((run / "run.json").read_bytes())
    manifest["failure_count"] = 1
    (run / "run.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run_directory(run)

    assert not result.valid
    assert "diagnostic_hash_invalid" in {issue.code for issue in result.issues}


def test_standalone_replay_validates_complete_source_capture(tmp_path: Path) -> None:
    fixture = tmp_path / "formal_run"
    shutil.copytree(FIXTURE_ROOT, fixture)
    source_manifest_path = fixture / "capture" / "run.json"
    source_manifest = json.loads(source_manifest_path.read_bytes())
    source_manifest["status"] = "failed"
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    result = validate_run_directory(fixture / "replay")

    assert not result.valid
    assert "source_capture_invalid" in {issue.code for issue in result.issues}


def test_frozen_evidence_requires_v2_schema_and_identifier_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality_bytes = Path("configs/quality_gates_v1.yaml").read_bytes()
    partition_bytes = b'{"query_id":"q1","query":"one","metadata":{"split":"dev"}}\n'
    identifier_bytes = b"{}\n"
    manifest_bytes = json.dumps(
        {
            "schema_version": "not-v2",
            "partitions": [
                {
                    "name": "dev",
                    "path": "dev.jsonl",
                    "query_count": 1,
                    "sha256": runner_module._sha256_bytes(partition_bytes),
                }
            ],
            "identifier_map": {
                "path": "identifier-map.json",
                "sha256": "sha256:" + "f" * 64,
            },
        }
    ).encode()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "manifest.json").write_bytes(manifest_bytes)
    (inputs / "dev.jsonl").write_bytes(partition_bytes)
    (inputs / "identifier-map.json").write_bytes(identifier_bytes)
    (tmp_path / "quality.yaml").write_bytes(quality_bytes)
    lock_payload = yaml.safe_load(
        Path("tests/fixtures/application/candidate.lock.yaml").read_bytes()
    )
    lock_payload["frozen_data"] = {
        "manifest": {
            "path": "inputs/manifest.json",
            "sha256": runner_module._sha256_bytes(manifest_bytes),
        },
        "identifier_map": {
            "path": "inputs/identifier-map.json",
            "sha256": runner_module._sha256_bytes(identifier_bytes),
        },
        "split": "dev",
        "query_count": 1,
        "partition_sha256": runner_module._sha256_bytes(partition_bytes),
    }
    lock_payload["quality_gates"] = {
        "path": "quality.yaml",
        "sha256": runner_module._sha256_bytes(quality_bytes),
    }
    lock = runner_module.CandidateLock.model_validate(lock_payload)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError):
        validator_module._frozen_evidence(lock)


def test_execution_snapshot_refs_must_bind_exact_manifest_entries(
    tmp_path: Path,
) -> None:
    run = tmp_path / "capture"
    shutil.copytree(FIXTURE_ROOT / "capture", run)
    lines = (run / "executions.jsonl").read_text(encoding="utf-8").splitlines()
    execution = json.loads(lines[0])
    execution["diagnostics"] = [
        {
            "dependency": "openalex",
            "endpoint": "dependency",
            "model_id": None,
            "usage": {
                "search_api_calls": 0,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_cny": "0",
                "elapsed_ms": 0,
            },
            "latency_ms": 0,
            "cache_hit": True,
            "snapshot_refs": [
                {
                    "entry_id": "forged-entry",
                    "dependency": "openalex",
                    "cache_key": "sha256:" + "a" * 64,
                    "response_sha256": "sha256:" + "b" * 64,
                    "captured_at": "2026-08-02T00:00:00Z",
                    "snapshot_path": "responses/openalex/forged.bin",
                }
            ],
            "errors": [],
        }
    ]
    lines[0] = json.dumps(execution)
    (run / "executions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_run_directory(run)

    assert not result.valid
    assert "snapshot_ref_invalid" in {issue.code for issue in result.issues}


def test_validator_rejects_foreign_project_receipts(tmp_path: Path) -> None:
    run = tmp_path / "capture"
    shutil.copytree(FIXTURE_ROOT / "capture", run)
    payload = json.loads((run / "usage.json").read_bytes())
    foreign = dict(payload["receipts"][0])
    foreign.update(
        {
            "reservation_id": "foreign-reservation",
            "run_id": "foreign-run",
            "query_id": "foreign-query",
        }
    )
    foreign["actual"] = dict(foreign["actual"])
    foreign["actual"]["cost_cny"] = "100"
    foreign["estimate"] = dict(foreign["estimate"])
    foreign["estimate"]["cost_cny"] = "100"
    payload["receipts"].append(foreign)
    payload["project_actual_cny"] = str(
        sum(Decimal(item["actual"]["cost_cny"]) for item in payload["receipts"])
    )
    payload["project_receipt_count"] = len(payload["receipts"])
    canonical = json.dumps(
        payload["receipts"],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    payload["project_receipts_sha256"] = (
        f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    )
    (run / "usage.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = json.loads((run / "run.json").read_bytes())
    manifest["project_receipt_count"] = payload["project_receipt_count"]
    manifest["project_receipts_sha256"] = payload["project_receipts_sha256"]
    (run / "run.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_run_directory(run)

    assert not result.valid
    assert "ledger_invalid" in {issue.code for issue in result.issues}


def test_canonical_formal_pair_publishes_every_gate_check() -> None:
    policy = yaml.safe_load(Path("configs/quality_gates_v1.yaml").read_bytes())
    expected = [
        rule["rule_id"]
        for rule in policy["rules"]
        if "dev" in rule["applies_to"]
    ]
    payload = json.loads((FIXTURE_ROOT / "capture" / "gates.json").read_bytes())

    assert [check["rule_id"] for check in payload["checks"] if check["applies"]] == expected


def test_canonical_fixture_has_nonempty_gold_and_filter_evidence() -> None:
    gold = [
        json.loads(line)
        for line in (FIXTURE_ROOT / "inputs" / "dev.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    executions = [
        json.loads(line)
        for line in (FIXTURE_ROOT / "capture" / "executions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert all(item["relevant_paper_ids"] for item in gold)
    assert all(item["retrieved_paper_ids"] for item in executions)
    assert all(item["post_filter_paper_ids"] for item in executions)


def test_canonical_gate_evidence_has_no_missing_reporting_values() -> None:
    gate = json.loads((FIXTURE_ROOT / "capture" / "gates.json").read_bytes())
    applicable_reporting = [
        check
        for check in gate["checks"]
        if check["applies"] and check["classification"] == "reporting_only"
    ]
    checks_by_id = {check["rule_id"]: check for check in gate["checks"]}

    assert len(applicable_reporting) == 30
    assert all(check["measure"]["value"] is not None for check in applicable_reporting)
    assert checks_by_id["retrieval-response-rate"]["measure"]["value"] == "1"
    assert checks_by_id["hard-filter-recall-loss"]["measure"]["value"] == "0"


def test_validator_rejects_symlink_run_root(tmp_path: Path) -> None:
    link = tmp_path / "capture-link"
    try:
        link.symlink_to(FIXTURE_ROOT.resolve() / "capture", target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    result = validate_run_directory(link)

    assert not result.valid
    assert "run_directory_unavailable" in {issue.code for issue in result.issues}


def test_canonical_formal_pair_has_nonempty_dependency_provenance() -> None:
    snapshot = json.loads(
        (FIXTURE_ROOT / "capture" / "snapshot-manifest.json").read_bytes()
    )
    executions = [
        json.loads(line)
        for line in (FIXTURE_ROOT / "capture" / "executions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    entry_ids = {entry["entry_id"] for entry in snapshot["entries"]}
    ref_ids = {
        ref["entry_id"]
        for execution in executions
        for diagnostic in execution["diagnostics"]
        for ref in diagnostic["snapshot_refs"]
    }

    assert {entry["request"]["dependency"] for entry in snapshot["entries"]} >= {
        "llm",
        "openalex",
    }
    assert ref_ids == entry_ids
