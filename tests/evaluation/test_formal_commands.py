import json
from pathlib import Path

import pytest

import paper_search.cli as cli_module
from paper_search.evaluation.runner import EvaluationRunResult
from paper_search.evaluation.validator import (
    compare_replay_command,
    verify_run_command,
)


def test_formal_commands_use_stable_invalid_and_mismatch_exit_codes(
    tmp_path: Path,
) -> None:
    assert verify_run_command(tmp_path / "missing") == 3
    assert compare_replay_command(tmp_path / "capture", tmp_path / "replay") == 3


def test_root_parser_exposes_formal_commands() -> None:
    parser = cli_module.build_parser()

    assert parser.parse_args(["verify-run", "run"]).command == "verify-run"
    assert parser.parse_args(["compare-replay", "capture", "replay"]).command == "compare-replay"
    evaluate = parser.parse_args(
        [
            "evaluate",
            "--lock",
            "replay.lock.yaml",
            "--split",
            "dev",
            "--mode",
            "replay",
            "--output-root",
            "runs",
            "--snapshot-manifest",
            "snapshot-manifest.json",
        ]
    )
    assert evaluate.command == "evaluate"


def test_evaluate_complete_gate_failure_returns_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run(request: object) -> EvaluationRunResult:
        del request
        return EvaluationRunResult(
            run_id="formal-1",
            run_path=tmp_path / "runs" / "formal-1",
            status="complete",
            gate_result="failed",
        )

    monkeypatch.setattr(cli_module, "run_evaluation", fake_run)

    exit_code = cli_module.main(
        [
            "evaluate",
            "--lock",
            "replay.lock.yaml",
            "--split",
            "dev",
            "--output-root",
            str(tmp_path / "runs"),
            "--snapshot-manifest",
            "snapshot-manifest.json",
        ]
    )

    assert exit_code == 5
    output = capsys.readouterr()
    assert output.err == ""
    assert set(json.loads(output.out)) == {
        "gate_result",
        "path",
        "run_id",
        "status",
    }
