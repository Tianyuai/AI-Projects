import json
import shutil
from pathlib import Path

import pytest
import yaml

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
