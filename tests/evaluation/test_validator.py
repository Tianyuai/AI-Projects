import json
import shutil
from pathlib import Path

import pytest

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
